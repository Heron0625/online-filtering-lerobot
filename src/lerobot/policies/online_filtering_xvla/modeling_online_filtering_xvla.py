from __future__ import annotations

import copy
from collections import deque

import torch
from torch import Tensor, nn

from lerobot.policies.common.vla_utils import pad_vector, resize_with_pad
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.xvla.modeling_xvla import XVLAModel, XVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_online_filtering_xvla import OnlineFilteringXVLAConfig


class DelayConditionGate(nn.Module):
    """Feature-wise old/new mixer conditioned on old delay, new delay and flow time."""

    def __init__(self, width: int) -> None:
        super().__init__()
        hidden = min(256, width)
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, old_delay: Tensor, new_delay: Tensor, flow_time: Tensor) -> Tensor:
        values = torch.stack((old_delay, new_delay, flow_time), dim=-1)
        return torch.sigmoid(self.net(values))


class OnlineFilteringXVLAModel(XVLAModel):
    def __init__(self, config, florence_config, proprio_dim: int) -> None:
        super().__init__(config, florence_config, proprio_dim)
        width = florence_config.vision_config.projection_dim
        self.condition_gate = DelayConditionGate(width).to(dtype=self._get_target_dtype())
        self.proprio_gate = DelayConditionGate(proprio_dim).to(dtype=self._get_target_dtype())

    def encode_two_conditions(
        self,
        input_ids: Tensor,
        old_images: Tensor,
        old_image_mask: Tensor,
        new_images: Tensor,
        new_image_mask: Tensor,
        old_delay: Tensor,
        new_delay: Tensor,
        flow_time: Tensor,
    ) -> dict[str, Tensor]:
        old = self.forward_vlm(input_ids, old_images, old_image_mask)
        new = self.forward_vlm(input_ids, new_images, new_image_mask)
        scale = float(max(1, self.config.max_inference_delay))
        gate = self.condition_gate(old_delay / scale, new_delay / scale, flow_time)
        return {
            key: old[key] + gate[:, None, :] * (new[key] - old[key])
            for key in ("vlm_features", "aux_visual_inputs")
        }

    def predict_endpoint(
        self,
        *,
        input_ids: Tensor,
        old_images: Tensor,
        old_image_mask: Tensor,
        new_images: Tensor,
        new_image_mask: Tensor,
        domain_id: Tensor,
        old_proprio: Tensor,
        new_proprio: Tensor,
        old_delay: Tensor,
        new_delay: Tensor,
        particle: Tensor,
        flow_time: Tensor,
    ) -> Tensor:
        dtype = self._get_target_dtype()
        enc = self.encode_two_conditions(
            input_ids,
            old_images.to(dtype),
            old_image_mask,
            new_images.to(dtype),
            new_image_mask,
            old_delay.to(dtype),
            new_delay.to(dtype),
            flow_time.to(dtype),
        )
        scale = float(max(1, self.config.max_inference_delay))
        proprio_gate = self.proprio_gate(old_delay.to(dtype) / scale, new_delay.to(dtype) / scale, flow_time.to(dtype))
        proprio = old_proprio.to(dtype) + proprio_gate * (new_proprio.to(dtype) - old_proprio.to(dtype))
        proprio_m, particle_m = self.action_space.preprocess(proprio, particle.to(dtype))
        return self.transformer(
            domain_id=domain_id,
            action_with_noise=particle_m,
            t=flow_time.to(dtype),
            proprio=proprio_m,
            **enc,
        )


class OnlineFilteringXVLAPolicy(XVLAPolicy):
    config_class = OnlineFilteringXVLAConfig
    name = "online_filtering_xvla"

    def __init__(self, config: OnlineFilteringXVLAConfig, **kwargs) -> None:
        require_package("transformers", extra="xvla")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        florence_config = config.get_florence_config()
        proprio_dim = config.max_state_dim if config.use_proprio else 0
        self.model = OnlineFilteringXVLAModel(config, florence_config, proprio_dim)
        self._queues = {ACTION: deque(maxlen=config.n_action_steps)}
        object.__setattr__(self, "_online_flow_teacher", None)

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
        for teacher_parameter, student_parameter in zip(teacher.parameters(), self.parameters(), strict=True):
            teacher_parameter.lerp_(student_parameter.detach(), 1 - decay)
        for teacher_buffer, student_buffer in zip(teacher.buffers(), self.buffers(), strict=True):
            teacher_buffer.copy_(student_buffer.detach())

    def train(self, mode: bool = True):
        super().train(mode)
        if self._online_flow_teacher is not None:
            self._online_flow_teacher.eval()
        return self

    def _prepare_images_at(self, batch: dict[str, Tensor], indices: Tensor) -> tuple[Tensor, Tensor]:
        images, masks = [], []
        rows = torch.arange(indices.shape[0], device=indices.device)
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
            raise ValueError("all configured image features are missing")
        stacked = torch.stack(images, dim=1)
        mask = torch.stack(masks, dim=1)
        total = max(self.config.num_image_views or stacked.shape[1], stacked.shape[1])
        if total > stacked.shape[1]:
            stacked = torch.cat((stacked, stacked.new_zeros(stacked.shape[0], total - stacked.shape[1], *stacked.shape[2:])), dim=1)
            mask = torch.cat((mask, mask.new_zeros(mask.shape[0], total - mask.shape[1])), dim=1)
        return stacked, mask

    def _prepare_state_at(self, batch: dict[str, Tensor], indices: Tensor) -> Tensor:
        if not self.config.use_proprio or OBS_STATE not in batch:
            return torch.zeros(indices.shape[0], 0, device=indices.device)
        state = batch[OBS_STATE]
        if state.ndim > 2:
            rows = torch.arange(indices.shape[0], device=indices.device)
            state = state[rows, indices]
        return pad_vector(state, self.model.dim_proprio, truncate=True)

    def _sample_delays(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        new = torch.randint(self.config.max_inference_delay + 1, (batch_size,), device=device)
        old = torch.floor(torch.rand(batch_size, device=device) * (new + 1)).long()
        return old, new

    def _condition_inputs(self, batch: dict[str, Tensor], old_delay: Tensor, new_delay: Tensor) -> dict[str, Tensor]:
        old_images, old_mask = self._prepare_images_at(batch, old_delay)
        new_images, new_mask = self._prepare_images_at(batch, new_delay)
        batch_size = old_delay.shape[0]
        return {
            "input_ids": batch[OBS_LANGUAGE_TOKENS],
            "old_images": old_images,
            "old_image_mask": old_mask,
            "new_images": new_images,
            "new_image_mask": new_mask,
            "domain_id": self._get_domain_id(batch, batch_size, old_images.device),
            "old_proprio": self._prepare_state_at(batch, old_delay),
            "new_proprio": self._prepare_state_at(batch, new_delay),
            "old_delay": old_delay,
            "new_delay": new_delay,
        }

    def _stage_a_forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        targets = self._prepare_action_targets(batch)
        batch_size = targets.shape[0]
        old_delay, new_delay = self._sample_delays(batch_size, targets.device)
        dtype = self.model._get_target_dtype()
        flow_time = torch.rand(batch_size, device=targets.device, dtype=dtype).clamp_max(1 - 1e-5)
        particle = torch.randn_like(targets, dtype=dtype) * flow_time[:, None, None] + targets.to(dtype) * (1 - flow_time[:, None, None])
        prediction = self.model.predict_endpoint(
            **self._condition_inputs(batch, old_delay, new_delay), particle=particle, flow_time=flow_time
        )
        losses = self.model.action_space.compute_loss(prediction, targets.to(dtype))
        loss = sum(losses.values())
        metrics = {key: value.detach().item() for key, value in losses.items()}
        metrics.update(loss=loss.detach().item(), mean_old_delay=old_delay.float().mean().item(), mean_new_delay=new_delay.float().mean().item())
        return loss, metrics

    def _stage_b_forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        teacher = self._online_flow_teacher
        if teacher is None:
            raise RuntimeError("EMA teacher was not initialized by the trainer")
        targets = self._prepare_action_targets(batch)
        batch_size, horizon = targets.shape[:2]
        rollout_steps = torch.randint(1, self.config.online_flow_rollout_steps + 1, (batch_size,), device=targets.device)
        total_delay = torch.randint(1, self.config.max_inference_delay + 1, (batch_size,), device=targets.device)
        noise = torch.randn_like(targets, dtype=self.model._get_target_dtype())
        endpoint = torch.zeros_like(noise)
        particle = noise
        with torch.no_grad():
            for step in range(int(rollout_steps.max().item())):
                active = step < rollout_steps
                old_delay = step * total_delay // (self.config.num_denoising_steps - 1)
                new_delay = (step + 1) * total_delay // (self.config.num_denoising_steps - 1)
                flow_time = torch.full((batch_size,), 1 - step / self.config.num_denoising_steps, device=targets.device, dtype=particle.dtype)
                predicted_endpoint = teacher.model.predict_endpoint(**teacher._condition_inputs(batch, old_delay, new_delay), particle=particle, flow_time=flow_time)
                endpoint = torch.where(active[:, None, None], predicted_endpoint, endpoint)
                next_time = (1 - (step + 1) / self.config.num_denoising_steps)
                proposal = noise * next_time + endpoint * (1 - next_time)
                positions = torch.arange(horizon, device=targets.device)[None]
                proposal = torch.where((positions < new_delay[:, None])[..., None], targets.to(proposal.dtype), proposal)
                particle = torch.where(active[:, None, None], proposal, particle)
        step = rollout_steps
        old_delay = (step - 1) * total_delay // (self.config.num_denoising_steps - 1)
        new_delay = step * total_delay // (self.config.num_denoising_steps - 1)
        flow_time = 1 - step.to(particle.dtype) / self.config.num_denoising_steps
        prediction = self.model.predict_endpoint(**self._condition_inputs(batch, old_delay, new_delay), particle=particle.detach(), flow_time=flow_time)
        suffix = torch.arange(horizon, device=targets.device)[None] >= new_delay[:, None]
        # Reuse XVLA's native action-space objective. Slicing each row is deliberate:
        # it excludes the variable-length committed prefix without changing the
        # position/rotation/gripper weighting between Stage A and Stage B.
        loss_terms: dict[str, list[Tensor]] = {}
        for row in range(batch_size):
            row_prediction = prediction[row : row + 1, suffix[row]]
            row_target = targets[row : row + 1, suffix[row]].to(prediction.dtype)
            for name, value in self.model.action_space.compute_loss(row_prediction, row_target).items():
                loss_terms.setdefault(name, []).append(value)
        losses = {name: torch.stack(values).mean() for name, values in loss_terms.items()}
        loss = sum(losses.values())
        switch = new_delay > old_delay
        metrics = {name: value.detach().item() for name, value in losses.items()}
        metrics.update(
            {
                "online_rollout_loss": loss.detach().item(),
                "mean_rollout_steps": rollout_steps.float().mean().item(),
                "mean_old_delay": old_delay.float().mean().item(),
                "mean_new_delay": new_delay.float().mean().item(),
                "switch_fraction": switch.float().mean().item(),
            }
        )
        return loss, metrics

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.training_stage == "stage_b":
            return self._stage_b_forward(batch)
        return self._stage_a_forward(batch)

    @torch.no_grad()
    def _get_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        input_ids = batch[OBS_LANGUAGE_TOKENS]
        batch_size = input_ids.shape[0]
        device = input_ids.device
        delay = torch.zeros(batch_size, dtype=torch.long, device=device)
        condition = self._condition_inputs(batch, delay, delay)
        noise = torch.randn(batch_size, self.config.chunk_size, self.model.dim_action, device=device, dtype=self.model._get_target_dtype())
        endpoint = torch.zeros_like(noise)
        for step in range(self.config.num_denoising_steps, 0, -1):
            flow_time = torch.full((batch_size,), step / self.config.num_denoising_steps, device=device, dtype=noise.dtype)
            particle = noise * flow_time[:, None, None] + endpoint * (1 - flow_time[:, None, None])
            endpoint = self.model.predict_endpoint(**condition, particle=particle, flow_time=flow_time)
        return self.model.action_space.postprocess(endpoint)

