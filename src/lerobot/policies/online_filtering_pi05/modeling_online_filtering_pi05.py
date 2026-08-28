#!/usr/bin/env python

from __future__ import annotations

import time as wall_time

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.common.vla_utils import pad_vector
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, get_gemma_config
from lerobot.policies.pi05.online_filtering import (
    ConditionTokenResampler,
    DINOv3PatchEncoder,
    FastObservationAdapter,
    FastObservationBuffer,
    FastObservationPacket,
    FastTokenResampler,
    PosteriorResidualUpdater,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from .configuration_online_filtering_pi05 import OnlineFilteringPI05Config


def sample_causal_delays(valid_observations: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """Uniformly sample one causally available observation index per example."""
    if valid_observations.ndim != 2 or valid_observations.shape[1] < 1:
        raise ValueError("valid_observations must have shape (batch, observation_window)")
    valid = valid_observations.bool()
    if not torch.all(valid[:, 0]):
        raise ValueError("the initial observation at delay 0 must be valid for every sample")
    probabilities = valid.to(torch.float32)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(1)


def select_temporal_steps(values: Tensor, indices: Tensor) -> Tensor:
    """Select a different temporal index for every batch row."""
    if values.ndim < 2 or indices.shape != (values.shape[0],):
        raise ValueError("values must start with (batch, time) and indices must have shape (batch,)")
    if torch.any(indices < 0) or torch.any(indices >= values.shape[1]):
        raise ValueError("temporal indices are outside the observation window")
    batch_indices = torch.arange(values.shape[0], device=values.device)
    return values[batch_indices, indices]


def sample_filtration_pairs(
    valid_observations: Tensor,
    *,
    min_gap: int = 1,
    max_gap: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample valid old/new physical conditions with variable observation gaps."""
    if valid_observations.ndim != 2 or valid_observations.shape[1] < 2:
        raise ValueError("Stage-B requires an observation window with at least two frames")
    window = valid_observations.shape[1]
    max_gap = window - 1 if max_gap is None else min(max_gap, window - 1)
    if not 1 <= min_gap <= max_gap:
        raise ValueError("filtration gaps must satisfy 1 <= min_gap <= max_gap")
    old_indices = []
    new_indices = []
    valid = valid_observations.bool()
    for row in range(valid.shape[0]):
        valid_gaps = []
        valid_starts_by_gap = {}
        for gap in range(min_gap, max_gap + 1):
            starts = torch.nonzero(valid[row, :-gap] & valid[row, gap:], as_tuple=False).flatten()
            if starts.numel():
                valid_gaps.append(gap)
                valid_starts_by_gap[gap] = starts
        if not valid_gaps:
            raise ValueError("every Stage-B sample must contain at least one valid filtration pair")
        gap_choice = torch.randint(len(valid_gaps), (), generator=generator).item()
        selected_gap = valid_gaps[gap_choice]
        starts = valid_starts_by_gap[selected_gap]
        start_choice = torch.randint(starts.numel(), (), generator=generator).item()
        old = starts[start_choice]
        old_indices.append(old)
        new_indices.append(old + selected_gap)
    return torch.stack(old_indices), torch.stack(new_indices)


def sample_adjacent_filtrations(
    valid_observations: Tensor, generator: torch.Generator | None = None
) -> Tensor:
    """Compatibility wrapper returning the old index of an adjacent pair."""
    old, _ = sample_filtration_pairs(valid_observations, generator=generator, max_gap=1)
    return old


def masked_particle_mse(prediction: Tensor, target: Tensor, mask: Tensor, action_dim: int) -> Tensor:
    """Mean squared error over selected action positions and real action dimensions."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share shape (batch, horizon, action_dim)")
    if mask.shape != prediction.shape[:2]:
        raise ValueError("mask must match particle batch and horizon dimensions")
    selected = mask.bool().unsqueeze(-1).expand(-1, -1, action_dim)
    if not torch.any(selected):
        raise ValueError("particle loss mask selects no elements")
    errors = (prediction[..., :action_dim] - target[..., :action_dim]).square()
    return errors[selected].mean()


class OnlineFilteringPI05Policy(PI05Policy):
    """PI0.5 Stage-A policy conditioned on observations arriving during chunk overlap."""

    config_class = OnlineFilteringPI05Config
    name = "online_filtering_pi05"

    def __init__(
        self,
        config: OnlineFilteringPI05Config,
        *,
        fast_encoder: DINOv3PatchEncoder | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self.config = config
        self.fast_encoder = fast_encoder or DINOv3PatchEncoder.from_pretrained(
            config.dinov3_model_id, freeze=config.freeze_dinov3
        )
        feature_dim = int(self.fast_encoder.backbone.config.hidden_size)
        expert_dim = get_gemma_config(config.action_expert_variant).width
        self.model.fast_resampler = FastTokenResampler(
            feature_dim=feature_dim,
            output_dim=expert_dim,
            num_fast_tokens=config.fast_num_tokens,
            num_heads=config.fast_num_heads,
            num_cameras=len(config.image_features),
            state_dim=config.max_state_dim,
        )
        self.model.fast_adapters = nn.ModuleDict(
            {
                str(layer): FastObservationAdapter(expert_dim, num_heads=config.fast_num_heads)
                for layer in config.fast_adapter_layers
            }
        )
        self.posterior_updater = PosteriorResidualUpdater(
            action_dim=config.max_action_dim,
            hidden_dim=config.posterior_hidden_dim,
            num_heads=config.posterior_num_heads,
            num_layers=config.posterior_num_layers,
            max_horizon=config.chunk_size,
        )
        self.posterior_fast_proj = nn.Linear(expert_dim, config.posterior_hidden_dim)
        slow_dim = get_gemma_config(config.paligemma_variant).width
        self.posterior_slow_resampler = ConditionTokenResampler(
            input_dim=slow_dim,
            output_dim=config.posterior_hidden_dim,
            num_tokens=config.posterior_slow_tokens,
            num_heads=config.posterior_num_heads,
        )
        expert_dtype = (
            self.model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
        )
        self.fast_encoder_device = config.fast_encoder_device or config.device
        self.fast_encoder.to(self.fast_encoder_device)
        self.model.fast_resampler.to(device=config.device, dtype=expert_dtype)
        self.model.fast_adapters.to(device=config.device, dtype=expert_dtype)
        self.posterior_updater.to(config.device)
        self.posterior_fast_proj.to(config.device)
        self.posterior_slow_resampler.to(config.device)
        self.fast_observation_buffer = FastObservationBuffer(
            capacity=config.fast_buffer_capacity,
            max_age_s=config.fast_observation_max_age_s,
        )
        self._online_generation = 0
        if config.training_stage == "stage_b":
            self._freeze_stage_a_teacher()

    def _freeze_stage_a_teacher(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.posterior_updater.requires_grad_(True)
        self.posterior_fast_proj.requires_grad_(True)
        self.posterior_slow_resampler.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.training_stage == "stage_b":
            self.model.eval()
            self.fast_encoder.eval()
            self.posterior_updater.train(mode)
            self.posterior_fast_proj.train(mode)
            self.posterior_slow_resampler.train(mode)
        return self

    def _prepare_pretrained_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        state_dict = super()._prepare_pretrained_state_dict(state_dict)
        current = self.state_dict()
        for key, value in current.items():
            if key.startswith(
                (
                    "fast_encoder.",
                    "model.fast_resampler.",
                    "model.fast_adapters.",
                    "posterior_updater.",
                    "posterior_fast_proj.",
                    "posterior_slow_resampler.",
                )
            ):
                state_dict.setdefault(key, value)
        return state_dict

    def _observation_validity(self, batch: dict[str, Tensor]) -> Tensor:
        first_image_key = next(iter(self.config.image_features))
        first_image = batch[first_image_key]
        if first_image.ndim != 5:
            raise ValueError(
                "Stage-A training requires temporal images shaped (batch, time, channels, height, width)"
            )
        batch_size, window = first_image.shape[:2]
        expected_window = self.config.max_inference_delay + 1
        if window != expected_window:
            raise ValueError(f"expected observation window {expected_window}, got {window}")
        valid = torch.ones(batch_size, window, dtype=torch.bool, device=first_image.device)
        for key in (*self.config.image_features, OBS_STATE):
            pad_key = f"{key}_is_pad"
            if pad_key in batch:
                valid &= ~batch[pad_key].bool()
        return valid

    def _initial_slow_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        initial = dict(batch)
        for key in (*self.config.image_features, OBS_STATE):
            value = batch.get(key)
            if value is not None and ((key == OBS_STATE and value.ndim == 3) or value.ndim == 5):
                initial[key] = value[:, 0]
            pad_key = f"{key}_is_pad"
            if pad_key in batch and batch[pad_key].ndim == 2:
                initial[pad_key] = batch[pad_key][:, 0]
        return initial

    def _fast_pixels(self, batch: dict[str, Tensor], delays: Tensor) -> tuple[Tensor, Tensor]:
        initial_cameras = []
        current_cameras = []
        mean = torch.tensor(self.config.fast_image_mean, device=delays.device).view(1, 3, 1, 1)
        std = torch.tensor(self.config.fast_image_std, device=delays.device).view(1, 3, 1, 1)
        for key in self.config.image_features:
            raw_images = batch[key]
            images = raw_images.to(device=delays.device, dtype=torch.float32)
            if raw_images.dtype == torch.uint8:
                images = images / 255.0
            initial = images[:, 0]
            current = select_temporal_steps(images, delays)
            if initial.shape[1] != 3:
                initial = initial.permute(0, 3, 1, 2)
                current = current.permute(0, 3, 1, 2)
            initial = F.interpolate(
                initial, size=(self.config.fast_image_size, self.config.fast_image_size), mode="bilinear"
            )
            current = F.interpolate(
                current, size=(self.config.fast_image_size, self.config.fast_image_size), mode="bilinear"
            )
            initial_cameras.append((initial - mean) / std)
            current_cameras.append((current - mean) / std)
        return torch.stack(initial_cameras, dim=1), torch.stack(current_cameras, dim=1)

    def _make_fast_tokens(self, batch: dict[str, Tensor], delays: Tensor) -> Tensor:
        initial_pixels, current_pixels = self._fast_pixels(batch, delays)
        initial_features = self.fast_encoder(initial_pixels.to(self.fast_encoder_device))
        current_features = self.fast_encoder(current_pixels.to(self.fast_encoder_device))
        resampler_dtype = self.model.fast_resampler.feature_proj.weight.dtype
        resampler_device = self.model.fast_resampler.feature_proj.weight.device
        initial_features = initial_features.to(device=resampler_device, dtype=resampler_dtype)
        current_features = current_features.to(device=resampler_device, dtype=resampler_dtype)
        states = select_temporal_steps(batch[OBS_STATE], delays)
        states = pad_vector(states, self.config.max_state_dim).to(
            device=initial_features.device, dtype=resampler_dtype
        )
        return self.model.fast_resampler(
            initial_features,
            current_features,
            delays=delays,
            states=states,
        )

    def _make_single_fast_tokens(self, batch: dict[str, Tensor], delay: int) -> Tensor:
        """Encode one non-temporal multi-camera observation and absolute proprio state."""
        cameras = []
        first = batch[next(iter(self.config.image_features))]
        batch_size = first.shape[0]
        device = first.device
        mean = torch.tensor(self.config.fast_image_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(self.config.fast_image_std, device=device).view(1, 3, 1, 1)
        for key in self.config.image_features:
            image = batch[key].to(device=device, dtype=torch.float32)
            if batch[key].dtype == torch.uint8:
                image = image / 255.0
            if image.shape[1] != 3:
                image = image.permute(0, 3, 1, 2)
            image = F.interpolate(
                image,
                size=(self.config.fast_image_size, self.config.fast_image_size),
                mode="bilinear",
            )
            cameras.append((image - mean) / std)
        pixels = torch.stack(cameras, dim=1)
        features = self.fast_encoder(pixels.to(self.fast_encoder_device))
        dtype = self.model.fast_resampler.feature_proj.weight.dtype
        resampler_device = self.model.fast_resampler.feature_proj.weight.device
        features = features.to(device=resampler_device, dtype=dtype)
        states = pad_vector(batch[OBS_STATE], self.config.max_state_dim).to(
            device=resampler_device, dtype=dtype
        )
        delays = torch.full((batch_size,), delay, dtype=torch.long, device=resampler_device)
        return self.model.fast_resampler(features, features, delays=delays, states=states)

    @torch.no_grad()
    def encode_fast_observation_packet(
        self,
        batch: dict[str, Tensor],
        *,
        capture_step: int,
        capture_timestamp: float,
        generation: int,
        camera_valid_mask: Tensor | None = None,
    ) -> FastObservationPacket:
        """Encode and timestamp a complete multi-camera/proprio observation packet."""
        camera_count = len(self.config.image_features)
        if camera_valid_mask is None:
            camera_valid_mask = torch.ones(camera_count, dtype=torch.bool)
        if camera_valid_mask.shape != (camera_count,):
            raise ValueError(f"camera_valid_mask must have shape ({camera_count},)")
        tokens = self._make_single_fast_tokens(batch, capture_step)
        return FastObservationPacket(
            tokens=tokens,
            capture_step=capture_step,
            capture_timestamp=float(capture_timestamp),
            ready_timestamp=wall_time.monotonic(),
            proprio=batch[OBS_STATE].detach(),
            camera_valid_mask=camera_valid_mask.detach().cpu(),
            generation=generation,
        )

    def push_fast_observation(self, packet: FastObservationPacket) -> bool:
        return self.fast_observation_buffer.push(packet)

    def _forward_stage_a(self, batch: dict[str, Tensor], reduction: str) -> tuple[Tensor, dict]:
        valid_observations = self._observation_validity(batch)
        delays = sample_causal_delays(valid_observations)
        initial_batch = self._initial_slow_batch(batch)
        images, img_masks = self._preprocess_images(initial_batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)
        fast_tokens = self._make_fast_tokens(batch, delays)

        losses = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise,
            time,
            fast_adapters=self.model.fast_adapters,
            fast_tokens=fast_tokens,
            fast_delays=delays,
        )
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_per_dim = losses.mean(dim=(0, 1))
        loss_dict = {
            "loss_per_dim": loss_per_dim.detach().cpu().numpy().tolist(),
            "mean_delay": delays.float().mean().item(),
        }
        if reduction == "none":
            loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = loss.mean().item()
            return loss, loss_dict
        loss = losses.mean()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    @torch.no_grad()
    def _integrate_teacher_particle(
        self,
        *,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
        actions: Tensor,
        base_noise: Tensor,
        target_time: Tensor,
        delays: Tensor,
        fast_tokens: Tensor,
    ) -> Tensor:
        """Integrate shared base noise from PI0.5 time 1 to ``target_time``."""
        particle = base_noise.clone()
        step_size = (target_time - 1.0) / self.config.teacher_integration_steps
        for step in range(self.config.teacher_integration_steps):
            current_time = 1.0 + step * step_size
            velocity = self.model.predict_velocity(
                images,
                img_masks,
                tokens,
                masks,
                particle,
                current_time,
                fast_adapters=self.model.fast_adapters,
                fast_tokens=fast_tokens,
                fast_delays=delays,
            )
            particle = particle + step_size[:, None, None] * velocity
        return particle

    def _forward_stage_b(self, batch: dict[str, Tensor], reduction: str) -> tuple[Tensor, dict]:
        if reduction != "mean":
            raise ValueError("Stage-B currently supports reduction='mean' only")
        valid_observations = self._observation_validity(batch)
        old_delays, new_delays = sample_filtration_pairs(
            valid_observations,
            min_gap=self.config.stage_b_min_gap,
            max_gap=self.config.stage_b_max_gap,
        )
        initial_batch = self._initial_slow_batch(batch)
        images, img_masks = self._preprocess_images(initial_batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        base_noise = self.model.sample_noise(actions.shape, actions.device)
        target_time = self.model.sample_time(actions.shape[0], actions.device)

        with torch.no_grad():
            prefix_embs, prefix_valid, _ = self.model.embed_prefix(
                images, img_masks, tokens, masks
            )
            old_fast_tokens = self._make_fast_tokens(batch, old_delays)
            new_fast_tokens = self._make_fast_tokens(batch, new_delays)
            old_particle = self._integrate_teacher_particle(
                images=images,
                img_masks=img_masks,
                tokens=tokens,
                masks=masks,
                actions=actions,
                base_noise=base_noise,
                target_time=target_time,
                delays=old_delays,
                fast_tokens=old_fast_tokens,
            )
            new_particle = self._integrate_teacher_particle(
                images=images,
                img_masks=img_masks,
                tokens=tokens,
                masks=masks,
                actions=actions,
                base_noise=base_noise,
                target_time=target_time,
                delays=new_delays,
                fast_tokens=new_fast_tokens,
            )

        drop_observation = torch.rand(actions.shape[0], device=actions.device) < (
            self.config.posterior_identity_dropout
        )
        target_particle = torch.where(drop_observation[:, None, None], old_particle, new_particle)
        student_fast_tokens = torch.where(
            drop_observation[:, None, None], old_fast_tokens, new_fast_tokens
        )
        slow_tokens = self.posterior_slow_resampler(prefix_embs.detach().float(), prefix_valid)
        fast_condition_tokens = self.posterior_fast_proj(student_fast_tokens.float())
        condition_tokens = torch.cat([slow_tokens, fast_condition_tokens], dim=1)
        prediction = self.posterior_updater(
            old_particle.float(),
            old_delays=old_delays,
            new_delays=new_delays,
            flow_times=target_time.float(),
            condition_tokens=condition_tokens,
        )

        action_dim = self.config.output_features[ACTION].shape[0]
        endpoint_loss = (prediction[..., :action_dim] - target_particle[..., :action_dim].float()).square().mean()
        loss = self.config.posterior_endpoint_weight * endpoint_loss
        return loss, {
            "loss": loss.item(),
            "posterior_endpoint_loss": endpoint_loss.item(),
            "mean_old_delay": old_delays.float().mean().item(),
            "mean_new_delay": new_delays.float().mean().item(),
            "mean_gap": (new_delays - old_delays).float().mean().item(),
            "identity_fraction": drop_observation.float().mean().item(),
        }

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        if self.config.training_stage == "stage_a":
            return self._forward_stage_a(batch, reduction)
        return self._forward_stage_b(batch, reduction)

    def begin_online_generation(self) -> int:
        """Start an online generation and return its stale-work rejection token."""
        self._online_generation += 1
        self.fast_observation_buffer.clear()
        return self._online_generation

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Generate a chunk while assimilating the latest completed fast packets."""
        self.eval()
        if any(batch[key].ndim != 4 for key in self.config.image_features):
            raise ValueError("online inference expects one non-temporal image per camera")
        if batch[OBS_STATE].ndim != 2 or batch[OBS_STATE].shape[0] != 1:
            raise ValueError("online inference currently requires batch size one")

        generation = kwargs.pop("online_generation", None)
        if generation is None:
            generation = self.begin_online_generation()
        elif generation != self._online_generation:
            raise ValueError("online_generation is stale or was not started by this policy")

        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        prefix_embs, prefix_pad_masks, past_key_values = self.model.prepare_prefix_cache(
            images, img_masks, tokens, masks
        )
        slow_condition_tokens = self.posterior_slow_resampler(
            prefix_embs.detach().float(), prefix_pad_masks
        )
        num_steps = int(kwargs.pop("num_steps", self.config.num_inference_steps))
        noise = kwargs.pop("noise", None)
        if kwargs:
            raise TypeError(f"unsupported online inference arguments: {sorted(kwargs)}")
        if noise is None:
            noise = self.model.sample_noise(
                (1, self.config.chunk_size, self.config.max_action_dim), tokens.device
            )

        initial_timestamp = wall_time.monotonic()
        current_fast_tokens = self._make_single_fast_tokens(batch, delay=0)
        current_version = 0
        current_capture_timestamp = initial_timestamp
        particle = noise
        step_size = -1.0 / num_steps

        for step in range(num_steps):
            flow_time = torch.full(
                (1,), 1.0 - step / num_steps, device=particle.device, dtype=torch.float32
            )
            query_timestamp = wall_time.monotonic()
            packet = self.fast_observation_buffer.latest_ready_after(
                current_version,
                generation=generation,
                query_timestamp=query_timestamp,
            )
            if packet is not None:
                old_delay = torch.tensor([current_version], device=particle.device)
                new_delay = torch.tensor([packet.capture_step], device=particle.device)
                age = torch.tensor(
                    [query_timestamp - packet.capture_timestamp],
                    device=particle.device,
                    dtype=torch.float32,
                )
                fast_tokens = packet.tokens.to(device=particle.device)
                fast_condition_tokens = self.posterior_fast_proj(fast_tokens.float())
                condition_tokens = torch.cat(
                    [slow_condition_tokens, fast_condition_tokens], dim=1
                )
                particle = self.posterior_updater(
                    particle.float(),
                    old_delays=old_delay,
                    new_delays=new_delay,
                    flow_times=flow_time,
                    observation_ages=age,
                    condition_tokens=condition_tokens,
                ).to(noise.dtype)
                current_fast_tokens = fast_tokens
                current_version = packet.capture_step
                current_capture_timestamp = packet.capture_timestamp

            age = torch.tensor(
                [query_timestamp - current_capture_timestamp],
                device=particle.device,
                dtype=torch.float32,
            )
            delay = torch.tensor([current_version], device=particle.device)
            velocity = self.model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=particle,
                timestep=flow_time,
                fast_adapters=self.model.fast_adapters,
                fast_tokens=current_fast_tokens,
                fast_delays=delay,
                fast_observation_ages=age,
            )
            particle = particle + step_size * velocity

        action_dim = self.config.output_features[ACTION].shape[0]
        return particle[:, :, :action_dim]
