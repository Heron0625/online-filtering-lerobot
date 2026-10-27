from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from lerobot.policies.common.vla_utils import pad_vector, resize_with_pad
from lerobot.policies.pi05.online_filtering import DINOv3PatchEncoder
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.xvla.modeling_xvla import XVLAModel, XVLAPolicy
from lerobot.policies.xvla.soft_transformer import timestep_embedding
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_online_filtering_xvla import OnlineFilteringXVLAConfig


class XVLAFastTokenResampler(nn.Module):
    """Compress current DINO patches into 32 spatial tokens per real camera plus proprio."""

    def __init__(
        self, feature_dim: int, output_dim: int, state_dim: int, num_visual_tokens: int, num_heads: int
    ) -> None:
        super().__init__()
        if num_visual_tokens % 2:
            raise ValueError("two-camera spatial resampling requires an even token count")
        self.output_dim = output_dim
        self.tokens_per_camera = num_visual_tokens // 2
        self.feature_proj = nn.Linear(feature_dim, output_dim)
        self.position_proj = nn.Linear(2, output_dim)
        self.camera_embedding = nn.Parameter(torch.zeros(2, output_dim))
        self.modality_embedding = nn.Parameter(torch.zeros(2, output_dim))
        self.queries = nn.Parameter(torch.randn(2, self.tokens_per_camera, output_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(output_dim, num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(output_dim)
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, patches: Tensor, state: Tensor) -> Tensor:
        if patches.ndim != 4 or patches.shape[1] != 2:
            raise ValueError("DINO patches must have shape (batch, 2, patches, feature_dim)")
        batch_size, cameras, patch_count, _ = patches.shape
        side = int(patch_count**0.5)
        if side * side != patch_count:
            raise ValueError("DINO patch tokens must form a square grid")
        values = self.feature_proj(patches)
        axis = torch.linspace(-1, 1, side, device=values.device, dtype=values.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        coords = torch.stack((xx, yy), dim=-1).reshape(patch_count, 2)
        values = values + self.position_proj(coords)[None, None]
        values = values + self.camera_embedding[None, :, None].to(values.dtype)
        queries = self.queries[None].expand(batch_size, -1, -1, -1).to(values.dtype)
        queries = queries + self.camera_embedding[None, :, None].to(values.dtype)
        sampled, _ = self.cross_attention(
            queries.reshape(batch_size * cameras, self.tokens_per_camera, self.output_dim),
            values.reshape(batch_size * cameras, patch_count, self.output_dim),
            values.reshape(batch_size * cameras, patch_count, self.output_dim),
        )
        visual = (
            self.output_norm(
                queries.reshape(batch_size, -1, self.output_dim)
                + sampled.reshape(batch_size, -1, self.output_dim)
            )
            + self.modality_embedding[0]
        )
        proprio = self.state_proj(state) + self.modality_embedding[1]
        return torch.cat((visual, proprio[:, None]), dim=1)


class XVLAFastAdapter(nn.Module):
    """Zero-init residual cross-attention with only the causal availability mask."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.action_norm = nn.LayerNorm(hidden_dim)
        self.fast_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, action_hidden: Tensor, condition: Tensor, delays: Tensor) -> Tensor:
        fast = self.fast_norm(condition)
        correction, _ = self.cross_attention(self.action_norm(action_hidden), fast, fast)
        positions = torch.arange(action_hidden.shape[1], device=action_hidden.device)[None]
        available = positions >= delays[:, None]
        return action_hidden + self.output_proj(correction) * available[..., None]


class OnlineFilteringXVLAModel(XVLAModel):
    def __init__(self, config, florence_config, proprio_dim: int, fast_feature_dim: int) -> None:
        super().__init__(config, florence_config, proprio_dim)
        self.fast_resampler = XVLAFastTokenResampler(
            fast_feature_dim, config.hidden_size, proprio_dim, config.fast_num_tokens, config.fast_num_heads
        )
        self.fast_adapters = nn.ModuleDict(
            {
                str(layer): XVLAFastAdapter(config.hidden_size, config.fast_num_heads)
                for layer in config.fast_adapter_layers
            }
        )
        self.condition_roles = nn.Parameter(torch.randn(2, 1, config.hidden_size) * 0.02)
        self.delay_pair_proj = nn.Sequential(
            nn.Linear(3, config.hidden_size), nn.SiLU(), nn.Linear(config.hidden_size, config.hidden_size)
        )
        dtype = self._get_target_dtype()
        self.fast_resampler.to(dtype=dtype)
        self.fast_adapters.to(dtype=dtype)
        self.condition_roles.data = self.condition_roles.data.to(dtype)
        self.delay_pair_proj.to(dtype=dtype)

    def make_online_condition(self, old: Tensor, new: Tensor, old_delay: Tensor, new_delay: Tensor) -> Tensor:
        if old.shape != new.shape:
            raise ValueError("old/new fast tokens must have identical shapes")
        scale = float(max(1, self.config.max_inference_delay))
        features = torch.stack(
            (old_delay.float() / scale, new_delay.float() / scale, (new_delay - old_delay).float() / scale),
            dim=-1,
        )
        dtype = next(self.delay_pair_proj.parameters()).dtype
        delay = self.delay_pair_proj(features.to(device=old.device, dtype=dtype))[:, None]
        return torch.cat(
            (old + self.condition_roles[0] + delay, new + self.condition_roles[1] + delay), dim=1
        )

    def encode_slow_condition(
        self, input_ids: Tensor, images: Tensor, image_mask: Tensor
    ) -> dict[str, Tensor | int]:
        encoded = self.forward_vlm(input_ids, images.to(self._get_target_dtype()), image_mask)
        aux_count = encoded["aux_visual_inputs"].shape[1]
        primary_count = aux_count // max(images.shape[1] - 1, 1)
        return {**encoded, "primary_visual_tokens": primary_count}

    def predict_endpoint(
        self,
        *,
        slow_condition: dict[str, Tensor | int],
        domain_id: Tensor,
        base_proprio: Tensor,
        old_fast_tokens: Tensor,
        new_fast_tokens: Tensor,
        old_delay: Tensor,
        new_delay: Tensor,
        particle: Tensor,
        flow_time: Tensor,
        drop_slow_visual: Tensor | None = None,
    ) -> Tensor:
        dtype = self._get_target_dtype()
        tr = self.transformer
        batch_size, horizon = particle.shape[:2]
        proprio, particle = self.action_space.preprocess(base_proprio.to(dtype), particle.to(dtype))
        times = timestep_embedding(flow_time.to(dtype), tr.dim_time)[:, None].expand(-1, horizon, -1)
        states = proprio[:, None].expand(-1, horizon, -1)
        action_hidden = tr.action_encoder(torch.cat((particle, states, times), dim=-1), domain_id)
        vlm = slow_condition["vlm_features"]
        aux = slow_condition["aux_visual_inputs"]
        if tr.use_hetero_proj:
            vlm, aux = tr.vlm_proj(vlm, domain_id), tr.aux_visual_proj(aux, domain_id)
        else:
            vlm, aux = tr.vlm_proj(vlm), tr.aux_visual_proj(aux)
        x = torch.cat((action_hidden, vlm, aux), dim=1)
        if x.shape[1] > tr.pos_emb.shape[1]:
            raise ValueError("X-VLA slow sequence exceeds max_len_seq")
        x = x + tr.pos_emb[:, : x.shape[1]]
        padding = None
        if drop_slow_visual is not None and drop_slow_visual.any():
            padding = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
            rows = torch.nonzero(drop_slow_visual, as_tuple=False).flatten()
            primary = int(slow_condition["primary_visual_tokens"])
            padding[rows, horizon : horizon + primary] = True
            aux_start = horizon + vlm.shape[1]
            padding[rows, aux_start : aux_start + aux.shape[1]] = True
        if tr.len_soft_prompts:
            prompts = tr.soft_prompt_hub(domain_id).view(batch_size, tr.len_soft_prompts, tr.hidden_size)
            x = torch.cat((x, prompts), dim=1)
            if padding is not None:
                padding = torch.cat(
                    (padding, torch.zeros(batch_size, prompts.shape[1], dtype=torch.bool, device=x.device)),
                    dim=1,
                )
        condition = self.make_online_condition(
            old_fast_tokens.to(dtype), new_fast_tokens.to(dtype), old_delay, new_delay
        )
        for index, block in enumerate(tr.blocks):
            x = block(x, key_padding_mask=padding)
            if str(index) in self.fast_adapters:
                action_hidden = self.fast_adapters[str(index)](x[:, :horizon], condition, new_delay)
                x = torch.cat((action_hidden, x[:, horizon:]), dim=1)
        return tr.action_decoder(tr.norm(x[:, :horizon]), domain_id)


@dataclass
class XVLAObservationPacket:
    tokens: Tensor
    domain_id: Tensor
    capture_step: int
    generation: int
    executed_prefix: Tensor


class OnlineFilteringXVLAPolicy(XVLAPolicy):
    config_class = OnlineFilteringXVLAConfig
    name = "online_filtering_xvla"

    def __init__(self, config: OnlineFilteringXVLAConfig, **kwargs) -> None:
        require_package("transformers", extra="xvla")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.fast_image_features = tuple(key for key in config.image_features if "empty_camera" not in key)
        if len(self.fast_image_features) != 2:
            raise ValueError("LIBERO online X-VLA requires exactly two real fast cameras")
        self.fast_encoder = DINOv3PatchEncoder.from_pretrained(
            config.dinov3_model_id, freeze=config.freeze_dinov3
        )
        feature_dim = int(self.fast_encoder.backbone.config.hidden_size)
        proprio_dim = config.max_state_dim if config.use_proprio else 0
        self.model = OnlineFilteringXVLAModel(config, config.get_florence_config(), proprio_dim, feature_dim)
        self._queues = {ACTION: deque(maxlen=config.n_action_steps)}
        object.__setattr__(self, "_online_flow_teacher", None)
        self._online_generation = 0
        self._pending_online_packet: XVLAObservationPacket | None = None
        self._training_step = 0

    def initialize_online_flow_teacher(self) -> None:
        if self.config.training_stage != "stage_b" or self._online_flow_teacher is not None:
            return
        teacher = copy.deepcopy(self)
        object.__setattr__(teacher, "_online_flow_teacher", None)
        teacher.requires_grad_(False).eval()
        object.__setattr__(self, "_online_flow_teacher", teacher)

    @torch.no_grad()
    def update_online_flow_teacher(self) -> None:
        teacher = self._online_flow_teacher
        if teacher is None:
            return
        decay = self.config.online_flow_ema_decay
        for target, source in zip(teacher.parameters(), self.parameters(), strict=True):
            target.lerp_(source.detach(), 1 - decay)
        for target, source in zip(teacher.buffers(), self.buffers(), strict=True):
            target.copy_(source.detach())

    def online_flow_teacher_state_dict(self) -> dict[str, Tensor] | None:
        """Save EMA values only for parameters which the student actually trains."""
        teacher = self._online_flow_teacher
        if teacher is None:
            return None
        trainable = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {
            name: value.detach().cpu() for name, value in teacher.state_dict().items() if name in trainable
        }

    def load_online_flow_teacher_state_dict(self, state: dict[str, Tensor]) -> None:
        teacher = self._online_flow_teacher
        if teacher is None:
            raise RuntimeError("online-flow teacher must be initialized before loading its state")
        incompatible = teacher.load_state_dict(state, strict=False)
        unexpected = [name for name in incompatible.unexpected_keys if name in state]
        if unexpected:
            raise ValueError(f"unexpected online-flow teacher keys: {unexpected}")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_dinov3:
            self.fast_encoder.eval()
        if self._online_flow_teacher is not None:
            self._online_flow_teacher.eval()
        return self

    def reset(self) -> None:
        super().reset()
        self._online_generation = getattr(self, "_online_generation", 0) + 1
        self._pending_online_packet = None

    def set_training_step(self, step: int) -> None:
        self._training_step = int(step)

    def _slow_dropout_probability(self) -> float:
        start, steps = (
            self.config.slow_visual_dropout_probability,
            self.config.slow_visual_dropout_anneal_steps,
        )
        if steps <= 0:
            return start
        progress = min(
            max((self._training_step - self.config.slow_visual_dropout_anneal_start_step) / steps, 0.0), 1.0
        )
        return start + progress * (self.config.slow_visual_dropout_final_probability - start)

    def _sample_slow_dropout(self, batch_size: int, device: torch.device) -> Tensor | None:
        probability = self._slow_dropout_probability()
        if not self.training or probability <= 0:
            return None
        return torch.rand(batch_size, device=device) < probability

    def _prepare_images_at(self, batch: dict[str, Tensor], indices: Tensor) -> tuple[Tensor, Tensor]:
        rows, images, masks = torch.arange(indices.shape[0], device=indices.device), [], []
        for key in self.config.image_features:
            if key not in batch:
                continue
            source = batch[key]
            image = source[rows, indices] if source.ndim == 5 else source
            if self.config.resize_imgs_with_padding is not None:
                image = resize_with_pad(image, *self.config.resize_imgs_with_padding, pad_value=0.0)
            images.append(image)
            masks.append(torch.ones(image.shape[0], dtype=torch.bool, device=image.device))
        if not images:
            raise ValueError("all configured slow image features are missing")
        stacked, mask = torch.stack(images, dim=1), torch.stack(masks, dim=1)
        total = max(self.config.num_image_views or stacked.shape[1], stacked.shape[1])
        if total > stacked.shape[1]:
            stacked = torch.cat(
                (stacked, stacked.new_zeros(stacked.shape[0], total - stacked.shape[1], *stacked.shape[2:])),
                dim=1,
            )
            mask = torch.cat((mask, mask.new_zeros(mask.shape[0], total - mask.shape[1])), dim=1)
        return stacked, mask

    def _prepare_state_at(self, batch: dict[str, Tensor], indices: Tensor) -> Tensor:
        state = batch[OBS_STATE]
        if state.ndim > 2:
            rows = torch.arange(indices.shape[0], device=indices.device)
            state = state[rows, indices]
        return pad_vector(state, self.model.dim_proprio, truncate=True)

    def _fast_pixels_at(self, batch: dict[str, Tensor], indices: Tensor) -> Tensor:
        rows, cameras = torch.arange(indices.shape[0], device=indices.device), []
        for key in self.fast_image_features:
            source = batch[key]
            image = source[rows, indices] if source.ndim == 5 else source
            was_uint8 = image.dtype == torch.uint8
            image = image.float()
            if image.shape[1] != 3:
                image = image.permute(0, 3, 1, 2)
            if was_uint8:
                image = image / 255.0
                mean = image.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
                std = image.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
                image = (image - mean) / std
            cameras.append(
                functional.interpolate(image, size=(self.config.fast_image_size,) * 2, mode="bilinear")
            )
        return torch.stack(cameras, dim=1)

    def _make_fast_tokens(self, batch: dict[str, Tensor], delays: Tensor) -> Tensor:
        patches = self.fast_encoder(
            self._fast_pixels_at(batch, delays).to(next(self.fast_encoder.parameters()).device)
        )
        dtype = self.model.fast_resampler.feature_proj.weight.dtype
        device = self.model.fast_resampler.feature_proj.weight.device
        state = self._prepare_state_at(batch, delays).to(device=device, dtype=dtype)
        return self.model.fast_resampler(patches.to(device=device, dtype=dtype), state)

    def _observation_validity(self, batch: dict[str, Tensor]) -> Tensor:
        first = batch[self.fast_image_features[0]]
        window = first.shape[1] if first.ndim == 5 else 1
        valid = torch.ones(first.shape[0], window, dtype=torch.bool, device=first.device)
        for key in (*self.fast_image_features, OBS_STATE):
            pad = batch.get(f"{key}_is_pad")
            if pad is not None and pad.ndim == 2:
                valid &= ~pad.bool()
        return valid

    def _sample_total_delays(self, valid: Tensor) -> Tensor:
        total = torch.zeros(valid.shape[0], dtype=torch.long, device=valid.device)
        draws = torch.rand(valid.shape[0], device=valid.device)
        for row in range(valid.shape[0]):
            candidates = torch.nonzero(valid[row, 1:], as_tuple=False).flatten() + 1
            if not candidates.numel():
                continue
            if draws[row] < self.config.main_delay_probability:
                selected = candidates[candidates <= min(10, self.config.max_inference_delay)]
            elif draws[row] < self.config.main_delay_probability + self.config.medium_delay_probability:
                selected = candidates[
                    (candidates >= 11) & (candidates <= min(20, self.config.max_inference_delay))
                ]
            else:
                selected = candidates[candidates >= 21]
            selected = selected if selected.numel() else candidates
            total[row] = selected[torch.randint(selected.numel(), (), device=valid.device)]
        return total

    def _sample_schedule(self, valid: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        total = self._sample_total_delays(valid)
        query = torch.randint(self.config.num_denoising_steps, (valid.shape[0],), device=valid.device)
        query = torch.where(total > 0, query, torch.zeros_like(query))
        new = query * total // (self.config.num_denoising_steps - 1)
        old = (query - 1).clamp_min(0) * total // (self.config.num_denoising_steps - 1)
        return old, new, 1 - query.float() / self.config.num_denoising_steps, total

    def _slow_inputs(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor | int], Tensor, Tensor]:
        batch_size = batch[OBS_LANGUAGE_TOKENS].shape[0]
        zero = torch.zeros(batch_size, dtype=torch.long, device=batch[OBS_LANGUAGE_TOKENS].device)
        images, mask = self._prepare_images_at(batch, zero)
        return (
            self.model.encode_slow_condition(batch[OBS_LANGUAGE_TOKENS], images, mask),
            self._prepare_state_at(batch, zero),
            self._get_domain_id(batch, batch_size, images.device),
        )

    def _masked_action_loss(
        self, prediction: Tensor, targets: Tensor, delays: Tensor, batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        valid = torch.arange(targets.shape[1], device=targets.device)[None] >= delays[:, None]
        if "action_is_pad" in batch:
            valid &= ~batch["action_is_pad"].to(device=valid.device, dtype=torch.bool)
        terms: dict[str, list[Tensor]] = {}
        for row in range(targets.shape[0]):
            if not valid[row].any():
                continue
            for name, value in self.model.action_space.compute_loss(
                prediction[row : row + 1, valid[row]], targets[row : row + 1, valid[row]]
            ).items():
                terms.setdefault(name, []).append(value)
        if not terms:
            raise ValueError("batch contains no valid suffix actions")
        losses = {name: torch.stack(values).mean() for name, values in terms.items()}
        return sum(losses.values()), losses

    def _predict(
        self,
        batch: dict[str, Tensor],
        particle: Tensor,
        time: Tensor,
        old: Tensor,
        new: Tensor,
        *,
        slow=None,
        drop=True,
    ) -> Tensor:
        slow, base_state, domain = self._slow_inputs(batch) if slow is None else slow
        return self.model.predict_endpoint(
            slow_condition=slow,
            domain_id=domain,
            base_proprio=base_state,
            old_fast_tokens=self._make_fast_tokens(batch, old),
            new_fast_tokens=self._make_fast_tokens(batch, new),
            old_delay=old,
            new_delay=new,
            particle=particle,
            flow_time=time,
            drop_slow_visual=self._sample_slow_dropout(particle.shape[0], particle.device) if drop else None,
        )

    def _stage_a_forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        targets = self._prepare_action_targets(batch).to(self.model._get_target_dtype())
        old, new, time, total = self._sample_schedule(self._observation_validity(batch))
        noise = torch.randn_like(targets)
        prediction = self._predict(
            batch, noise * time[:, None, None] + targets * (1 - time[:, None, None]), time, old, new
        )
        loss, losses = self._masked_action_loss(prediction, targets, new, batch)
        metrics = {name: value.detach().item() for name, value in losses.items()}
        metrics.update(
            loss=loss.detach().item(),
            mean_old_delay=old.float().mean().item(),
            mean_new_delay=new.float().mean().item(),
            mean_total_delay=total.float().mean().item(),
            switch_fraction=(new > old).float().mean().item(),
            slow_visual_dropout_probability=self._slow_dropout_probability(),
        )
        return loss, metrics

    def _stage_b_forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        teacher = self._online_flow_teacher
        if teacher is None:
            raise RuntimeError("EMA teacher was not initialized by the trainer")
        targets = self._prepare_action_targets(batch).to(self.model._get_target_dtype())
        total = self._sample_total_delays(self._observation_validity(batch))
        rollout = torch.randint(
            1, self.config.online_flow_rollout_steps + 1, (targets.shape[0],), device=targets.device
        )
        noise = torch.randn_like(targets)
        particle = noise
        teacher_slow = teacher._slow_inputs(batch)
        with torch.no_grad():
            for step in range(int(rollout.max().item())):
                active = step < rollout
                old = max(step - 1, 0) * total // (self.config.num_denoising_steps - 1)
                new = step * total // (self.config.num_denoising_steps - 1)
                time = torch.full(
                    (targets.shape[0],),
                    1 - step / self.config.num_denoising_steps,
                    device=targets.device,
                    dtype=targets.dtype,
                )
                endpoint = teacher._predict(batch, particle, time, old, new, slow=teacher_slow, drop=False)
                next_time = 1 - (step + 1) / self.config.num_denoising_steps
                proposal = noise * next_time + endpoint * (1 - next_time)
                positions = torch.arange(targets.shape[1], device=targets.device)[None]
                proposal = torch.where((positions < new[:, None])[..., None], targets, proposal)
                particle = torch.where(active[:, None, None], proposal, particle)
        old = (rollout - 1).clamp_min(0) * total // (self.config.num_denoising_steps - 1)
        new = rollout * total // (self.config.num_denoising_steps - 1)
        time = 1 - rollout.to(targets.dtype) / self.config.num_denoising_steps
        prediction = self._predict(batch, particle.detach(), time, old, new)
        loss, losses = self._masked_action_loss(prediction, targets, new, batch)
        metrics = {name: value.detach().item() for name, value in losses.items()}
        metrics.update(
            online_rollout_loss=loss.detach().item(),
            loss=loss.detach().item(),
            mean_rollout_steps=rollout.float().mean().item(),
            mean_old_delay=old.float().mean().item(),
            mean_new_delay=new.float().mean().item(),
            switch_fraction=(new > old).float().mean().item(),
            slow_visual_dropout_probability=self._slow_dropout_probability(),
        )
        return loss, metrics

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        return (
            self._stage_b_forward(batch)
            if self.config.training_stage == "stage_b"
            else self._stage_a_forward(batch)
        )

    def begin_online_generation(self) -> int:
        self._online_generation += 1
        self._pending_online_packet = None
        return self._online_generation

    @torch.no_grad()
    def _encode_runtime_fast(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        batch_size = batch[OBS_LANGUAGE_TOKENS].shape[0]
        zero = torch.zeros(batch_size, dtype=torch.long, device=batch[OBS_LANGUAGE_TOKENS].device)
        return self._make_fast_tokens(batch, zero), self._get_domain_id(batch, batch_size, zero.device)

    @torch.no_grad()
    def encode_fast_observation_packet(
        self,
        batch: dict[str, Tensor],
        *,
        capture_step: int,
        generation: int,
        executed_prefix: Tensor,
        **kwargs,
    ) -> XVLAObservationPacket:
        if kwargs.keys() - {"initial_batch", "capture_timestamp", "camera_valid_mask"}:
            raise TypeError(f"unsupported packet arguments: {sorted(kwargs)}")
        tokens, domain = self._encode_runtime_fast(batch)
        return XVLAObservationPacket(
            tokens.detach(), domain.detach(), capture_step, generation, executed_prefix.detach()
        )

    def push_fast_observation(self, packet: XVLAObservationPacket) -> None:
        if packet.generation == self._online_generation and (
            self._pending_online_packet is None
            or packet.capture_step >= self._pending_online_packet.capture_step
        ):
            self._pending_online_packet = packet

    def _take_online_packet(self, after_delay: int, generation: int) -> XVLAObservationPacket | None:
        packet = self._pending_online_packet
        if packet is None or packet.generation != generation or packet.capture_step <= after_delay:
            return None
        self._pending_online_packet = None
        return packet

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        callback = kwargs.pop("observation_callback", None)
        if callback is None:
            return super().predict_action_chunk(batch, **kwargs)
        generation = kwargs.pop("online_generation", None) or self.begin_online_generation()
        assimilation = kwargs.pop("assimilation_callback", None)
        kwargs.pop("observation_age_override", None)
        steps = int(kwargs.pop("num_steps", self.config.num_denoising_steps))
        noise = kwargs.pop("noise", None)
        if kwargs:
            raise TypeError(f"unsupported online inference arguments: {sorted(kwargs)}")
        slow = self._slow_inputs(batch)
        current_fast, domain = self._encode_runtime_fast(batch)
        _, base_state, _ = slow
        dtype, device = self.model._get_target_dtype(), base_state.device
        noise = (
            noise
            if noise is not None
            else torch.randn(1, self.config.chunk_size, self.model.dim_action, device=device, dtype=dtype)
        )
        endpoint, current_delay, committed = torch.zeros_like(noise), 0, None
        for query in range(steps):
            time = torch.full((1,), 1 - query / steps, device=device, dtype=dtype)
            particle = noise * time[:, None, None] + endpoint * (1 - time[:, None, None])
            callback(query, steps, generation)
            packet = self._take_online_packet(current_delay, generation)
            old_fast, old_delay = current_fast, current_delay
            if packet is not None:
                current_fast, domain, current_delay = packet.tokens, packet.domain_id, packet.capture_step
                committed = torch.zeros_like(particle)
                committed[:, :current_delay, : packet.executed_prefix.shape[-1]] = packet.executed_prefix[
                    :, :current_delay
                ]
                if assimilation is not None:
                    assimilation(query, current_delay, self.config.training_stage == "stage_b")
            positions = torch.arange(self.config.chunk_size, device=device)[None]
            if committed is not None:
                particle = torch.where((positions < current_delay)[..., None], committed, particle)
            slow_condition, base_state, _ = slow
            endpoint = self.model.predict_endpoint(
                slow_condition=slow_condition,
                domain_id=domain,
                base_proprio=base_state,
                old_fast_tokens=old_fast,
                new_fast_tokens=current_fast,
                old_delay=torch.tensor([old_delay], device=device),
                new_delay=torch.tensor([current_delay], device=device),
                particle=particle,
                flow_time=time,
            )
            if committed is not None:
                endpoint = torch.where((positions < current_delay)[..., None], committed, endpoint)
        return self.model.action_space.postprocess(endpoint)
