#!/usr/bin/env python

from __future__ import annotations

import copy
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
    hard_commit_action_prefix,
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


def sample_online_flow_transitions(
    valid_observations: Tensor,
    num_inference_steps: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample one transition from the continuous-online LIBERO control/flow clock.

    A per-sample total delay is uniform over its valid positive observation indices. Query
    k observes floor(k * total_delay / (N - 1)) control steps, exactly matching the
    continuous_online_v1 evaluator. Repeated indices naturally produce identity transitions.
    """
    valid = valid_observations.bool()
    if valid.ndim != 2 or valid.shape[1] < 1 or not torch.all(valid[:, 0]):
        raise ValueError("valid_observations must have shape (batch, window) and valid delay zero")
    if num_inference_steps < 2:
        raise ValueError("online flow requires at least two inference steps")

    device = valid.device
    batch_size = valid.shape[0]
    total_delays = torch.zeros(batch_size, dtype=torch.long, device=device)
    query_steps = torch.randint(
        num_inference_steps, (batch_size,), device=device, generator=generator
    )
    for row in range(batch_size):
        positive = torch.nonzero(valid[row, 1:], as_tuple=False).flatten() + 1
        if positive.numel() == 0:
            query_steps[row] = 0
            continue
        choice = torch.randint(positive.numel(), (), device=device, generator=generator)
        total_delays[row] = positive[choice]

    new_delays = query_steps * total_delays // (num_inference_steps - 1)
    previous_steps = (query_steps - 1).clamp_min(0)
    old_delays = previous_steps * total_delays // (num_inference_steps - 1)
    flow_times = 1.0 - query_steps.to(torch.float32) / num_inference_steps
    rows = torch.arange(batch_size, device=device)
    if not torch.all(valid[rows, old_delays] & valid[rows, new_delays]):
        raise ValueError("online delay clock selected a padded observation")
    return old_delays, new_delays, flow_times, total_delays


def sample_stage_a_delays_and_branches(
    valid_observations: Tensor,
    clean_probability: float,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample clean/switch examples and valid delays.

    Switch examples always use a positive delay. Clean examples split evenly between delay zero
    and a positive delay when the latter is available. Rows without a future observation are
    necessarily clean at delay zero.
    """
    if not 0 <= clean_probability <= 1:
        raise ValueError("clean_probability must be in [0, 1]")
    valid = valid_observations.bool()
    if valid.ndim != 2 or valid.shape[1] < 1 or not torch.all(valid[:, 0]):
        raise ValueError("valid_observations must have shape (batch, window) and a valid delay zero")

    delays = torch.zeros(valid.shape[0], dtype=torch.long, device=valid.device)
    switch = torch.zeros(valid.shape[0], dtype=torch.bool, device=valid.device)
    draws = torch.rand((valid.shape[0], 2), device=valid.device, generator=generator)
    for row in range(valid.shape[0]):
        positive = torch.nonzero(valid[row, 1:], as_tuple=False).flatten() + 1
        if positive.numel() == 0:
            continue
        switch[row] = draws[row, 0] >= clean_probability
        use_positive_delay = switch[row] or draws[row, 1] >= 0.5
        if use_positive_delay:
            choice = torch.randint(positive.numel(), (), device=valid.device, generator=generator)
            delays[row] = positive[choice]
    return delays, switch


def make_structured_action_corruption(
    actions: Tensor,
    delays: Tensor,
    action_dim: int,
    target_rms: Tensor,
    corruption_types: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Create low-frequency synthetic old plans and return their realized suffix RMS.

    The three rules are persistent bias, low-frequency drift, and temporal lag. They operate on
    the six continuous delta-EE dimensions; the discrete gripper command is preserved.
    Corruption is normalized using only action positions at or after ``delay``.
    """
    if actions.ndim != 3 or delays.shape != actions.shape[:1] or target_rms.shape != actions.shape[:1]:
        raise ValueError("actions, delays, and target_rms have incompatible shapes")
    if not 0 < action_dim <= actions.shape[-1]:
        raise ValueError("action_dim must select at least one real action dimension")
    batch_size, horizon, _ = actions.shape
    device = actions.device
    dtype = actions.dtype
    if corruption_types is None:
        corruption_types = torch.randint(3, (batch_size,), device=device, generator=generator)
    if corruption_types.shape != (batch_size,) or torch.any((corruption_types < 0) | (corruption_types > 2)):
        raise ValueError("corruption_types must have shape (batch,) with values in [0, 2]")

    clean = actions[..., :action_dim]
    delta = torch.zeros_like(clean)
    steps = torch.arange(horizon, device=device, dtype=dtype)
    arm_dim = max(action_dim - 1, 1)
    for row in range(batch_size):
        kind = int(corruption_types[row].item())
        delay = int(delays[row].item())
        if kind == 0:  # Persistent bias with a smooth suffix decay.
            direction = torch.randn(arm_dim, device=device, dtype=dtype, generator=generator)
            decay = torch.exp(-torch.clamp(steps - delay, min=0) / max(horizon / 4, 1))
            delta[row, :, :arm_dim] = decay[:, None] * direction[None, :]
        elif kind == 1:  # Low-frequency constant/linear/sinusoidal drift.
            phase = torch.rand((), device=device, generator=generator).to(dtype) * (2 * torch.pi)
            basis = torch.stack(
                [
                    torch.ones_like(steps),
                    steps / max(horizon - 1, 1),
                    torch.sin(torch.pi * steps / horizon + phase),
                ]
            )
            coefficients = torch.randn(3, arm_dim, device=device, dtype=dtype, generator=generator)
            delta[row, :, :arm_dim] = basis.T @ coefficients
        else:  # A one- or two-step temporal lag without circular wraparound.
            lag = int(torch.randint(1, 3, (), device=device, generator=generator).item())
            indices = torch.clamp(torch.arange(horizon, device=device) - lag, min=0)
            delta[row, :, :arm_dim] = clean[row, indices, :arm_dim] - clean[row, :, :arm_dim]

    suffix_mask = torch.arange(horizon, device=device)[None, :] >= delays[:, None]
    selected = suffix_mask.unsqueeze(-1).expand(-1, -1, action_dim)
    squared_sum = (delta.square() * selected).sum(dim=(1, 2))
    counts = selected.sum(dim=(1, 2)).clamp_min(1)
    raw_rms = torch.sqrt(squared_sum / counts).clamp_min(1e-6)
    delta = delta * (target_rms / raw_rms)[:, None, None]
    corrupted = actions.clone()
    corrupted[..., :action_dim] = clean + delta
    realized_rms = torch.sqrt((delta.square() * selected).sum(dim=(1, 2)) / counts)
    return corrupted, realized_rms


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


def sample_midflow_times(
    batch_size: int,
    num_inference_steps: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample runtime-aligned switch steps and their strictly interior flow times."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_inference_steps < 2:
        raise ValueError("mid-flow switching requires at least two inference steps")
    switch_steps = torch.randint(
        1,
        num_inference_steps,
        (batch_size,),
        device=device,
        generator=generator,
    )
    flow_times = 1.0 - switch_steps.float() / num_inference_steps
    return switch_steps, flow_times


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


def masked_residual_objective(
    prediction: Tensor,
    old_particle: Tensor,
    target_particle: Tensor,
    mask: Tensor,
    action_dim: int,
    relative_floor: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Scale-balanced residual loss, cosine loss, and per-row magnitude ratio.

    Each row is normalized by its detached teacher residual energy.  The floor
    prevents identity or near-identity pairs from receiving unbounded weight.
    """
    if relative_floor <= 0:
        raise ValueError("relative_floor must be positive")
    selected = mask.bool().unsqueeze(-1).expand(-1, -1, action_dim)
    if not torch.all(selected.flatten(1).any(dim=1)):
        raise ValueError("every row must select at least one residual element")
    teacher = target_particle[..., :action_dim] - old_particle[..., :action_dim]
    student = prediction[..., :action_dim] - old_particle[..., :action_dim]
    squared_error = (student - teacher).square()
    teacher_energy = teacher.square()
    relative_rows = []
    cosine_rows = []
    magnitude_rows = []
    for row in range(prediction.shape[0]):
        row_mask = selected[row]
        teacher_row = teacher[row][row_mask]
        student_row = student[row][row_mask]
        error_mse = squared_error[row][row_mask].mean()
        target_mse = teacher_energy[row][row_mask].mean()
        relative_rows.append(error_mse / target_mse.detach().clamp_min(relative_floor))
        target_norm = teacher_row.norm()
        student_norm = student_row.norm()
        cosine = torch.nn.functional.cosine_similarity(
            teacher_row.unsqueeze(0), student_row.unsqueeze(0), eps=1e-4
        ).squeeze(0)
        # Near-identity rows carry no reliable direction target.
        cosine_rows.append(torch.where(target_mse > relative_floor, 1 - cosine, cosine * 0))
        magnitude_rows.append(student_norm / target_norm.clamp_min(1e-12))
    return (
        torch.stack(relative_rows).mean(),
        torch.stack(cosine_rows).mean(),
        torch.stack(magnitude_rows).mean(),
    )


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
            spatial_tokens=config.fast_spatial_tokens,
        )
        self.model.fast_adapters = nn.ModuleDict(
            {
                str(layer): FastObservationAdapter(expert_dim, num_heads=config.fast_num_heads)
                for layer in config.fast_adapter_layers
            }
        )
        self.online_condition_roles: nn.Parameter | None = None
        self.online_delay_pair_proj: nn.Module | None = None
        if config.training_stage == "online_flow":
            self.online_condition_roles = nn.Parameter(torch.randn(2, 1, expert_dim) * 0.02)
            self.online_delay_pair_proj = nn.Sequential(
                nn.Linear(3, expert_dim),
                nn.SiLU(),
                nn.Linear(expert_dim, expert_dim),
            )
        self.posterior_updater = PosteriorResidualUpdater(
            action_dim=config.max_action_dim,
            hidden_dim=config.posterior_hidden_dim,
            num_heads=config.posterior_num_heads,
            num_layers=config.posterior_num_layers,
            max_horizon=config.chunk_size,
            use_learned_gate=config.posterior_use_learned_gate,
        )
        self.posterior_fast_proj = nn.Linear(expert_dim, config.posterior_hidden_dim)
        # The role embeddings are only consumed by the Stage-B updater.  Do not
        # register them in Stage A: older Stage-A checkpoints (and their Adam
        # states) correctly contain no such parameter, so registering it would
        # make optimizer-state resume fail with a parameter-group length
        # mismatch.
        self.posterior_condition_roles: nn.Parameter | None = None
        if config.training_stage == "stage_b":
            self.posterior_condition_roles = nn.Parameter(
                torch.randn(3, 1, config.posterior_hidden_dim) * 0.02
            )
        slow_dim = get_gemma_config(config.paligemma_variant).width
        self.posterior_slow_resampler = ConditionTokenResampler(
            input_dim=slow_dim,
            output_dim=config.posterior_hidden_dim,
            num_tokens=config.posterior_slow_tokens,
            num_heads=config.posterior_num_heads,
        )
        expert_dtype = self.model.paligemma_with_expert.gemma_expert.model.layers[
            0
        ].self_attn.q_proj.weight.dtype
        self.fast_encoder_device = config.fast_encoder_device or config.device
        self.fast_encoder.to(self.fast_encoder_device)
        self.model.fast_resampler.to(device=config.device, dtype=expert_dtype)
        self.model.fast_adapters.to(device=config.device, dtype=expert_dtype)
        if self.online_condition_roles is not None:
            self.online_condition_roles.data = self.online_condition_roles.data.to(
                device=config.device, dtype=expert_dtype
            )
        if self.online_delay_pair_proj is not None:
            self.online_delay_pair_proj.to(device=config.device, dtype=expert_dtype)
        self.posterior_updater.to(config.device)
        self.posterior_fast_proj.to(config.device)
        if self.posterior_condition_roles is not None:
            self.posterior_condition_roles.data = self.posterior_condition_roles.data.to(config.device)
        self.posterior_slow_resampler.to(config.device)
        self.fast_observation_buffer = FastObservationBuffer(
            capacity=config.fast_buffer_capacity,
            max_age_s=config.fast_observation_max_age_s,
        )
        self._online_generation = 0
        self._training_step = 0
        # Kept outside nn.Module registration: the rollout teacher must not enter the optimizer
        # or the student checkpoint. It is reconstructed from the loaded student on every run.
        object.__setattr__(self, "_online_flow_teacher", None)
        if config.training_stage == "stage_b":
            self._freeze_stage_a_teacher()

    def initialize_online_flow_teacher(self) -> None:
        """Create the per-rank EMA rollout teacher after student weights have been loaded."""
        if self.config.training_stage != "online_flow" or self.config.online_flow_phase != "ema_rollout":
            return
        if self._online_flow_teacher is not None:
            return
        # FastObservationBuffer owns a thread lock and is inference-only. Exclude it from the
        # copy rather than teaching pickle semantics to a runtime synchronization primitive.
        student_buffer = self.fast_observation_buffer
        self.fast_observation_buffer = None
        try:
            teacher = copy.deepcopy(self)
        finally:
            self.fast_observation_buffer = student_buffer
        teacher.fast_observation_buffer = None
        object.__setattr__(teacher, "_online_flow_teacher", None)
        teacher.requires_grad_(False)
        teacher.eval()
        object.__setattr__(self, "_online_flow_teacher", teacher)

    @torch.no_grad()
    def update_online_flow_teacher(self) -> None:
        teacher = self._online_flow_teacher
        if teacher is None:
            return
        decay = self.config.online_flow_ema_decay
        for teacher_parameter, student_parameter in zip(
            teacher.parameters(), self.parameters(), strict=True
        ):
            teacher_parameter.lerp_(student_parameter.detach(), 1 - decay)
        for teacher_buffer, student_buffer in zip(teacher.buffers(), self.buffers(), strict=True):
            teacher_buffer.copy_(student_buffer.detach())

    def _freeze_stage_a_teacher(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.posterior_updater.requires_grad_(True)
        if not self.config.posterior_use_learned_gate:
            self.posterior_updater.residual_gate.requires_grad_(False)
        self.posterior_fast_proj.requires_grad_(True)
        if self.posterior_condition_roles is not None:
            self.posterior_condition_roles.requires_grad_(True)
        self.posterior_slow_resampler.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._online_flow_teacher is not None:
            self._online_flow_teacher.eval()
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
        # Older online-flow checkpoints used a third explicit ``new-old`` role.
        # The two-condition model can still initialize from them by retaining
        # only the old/new role embeddings.
        role_key = "online_condition_roles"
        if (
            role_key in state_dict
            and role_key in current
            and state_dict[role_key].shape != current[role_key].shape
            and state_dict[role_key].shape[0] >= current[role_key].shape[0]
        ):
            state_dict[role_key] = state_dict[role_key][: current[role_key].shape[0]]
        for key, value in current.items():
            if key.startswith(
                (
                    "fast_encoder.",
                    "model.fast_resampler.",
                    "model.fast_adapters.",
                    "online_condition_roles",
                    "online_delay_pair_proj.",
                    "posterior_updater.",
                    "posterior_fast_proj.",
                    "posterior_condition_roles",
                    "posterior_slow_resampler.",
                )
            ):
                state_dict.setdefault(key, value)
        return state_dict

    def _make_online_condition_tokens(
        self,
        old_fast_tokens: Tensor,
        new_fast_tokens: Tensor,
        old_delays: Tensor,
        new_delays: Tensor,
    ) -> Tensor:
        """Combine explicitly tagged old and new tokens for the online vector field."""
        if self.online_condition_roles is None or self.online_delay_pair_proj is None:
            raise RuntimeError("two-condition tokens require training_stage='online_flow'")
        if old_fast_tokens.shape != new_fast_tokens.shape:
            raise ValueError("old and new fast tokens must have identical shapes")
        if old_delays.shape != new_delays.shape or old_delays.shape != old_fast_tokens.shape[:1]:
            raise ValueError("old/new delays must match the fast-token batch")
        scale = float(max(self.config.max_inference_delay, 1))
        delay_features = torch.stack(
            [
                old_delays.to(torch.float32) / scale,
                new_delays.to(torch.float32) / scale,
                (new_delays - old_delays).to(torch.float32) / scale,
            ],
            dim=-1,
        )
        projection_dtype = next(self.online_delay_pair_proj.parameters()).dtype
        delay_embedding = self.online_delay_pair_proj(
            delay_features.to(device=old_fast_tokens.device, dtype=projection_dtype)
        ).unsqueeze(1)
        roles = self.online_condition_roles
        old = old_fast_tokens + roles[0] + delay_embedding
        new = new_fast_tokens + roles[1] + delay_embedding
        return torch.cat([old, new], dim=1)

    def _make_posterior_condition_tokens(
        self,
        slow_tokens: Tensor,
        old_fast_tokens: Tensor,
        new_fast_tokens: Tensor,
    ) -> Tensor:
        """Expose old, new, and innovation fast conditions with explicit semantic roles."""
        old_tokens = self.posterior_fast_proj(old_fast_tokens.float())
        new_tokens = self.posterior_fast_proj(new_fast_tokens.float())
        innovation_tokens = new_tokens - old_tokens
        roles = self.posterior_condition_roles
        return torch.cat(
            [
                slow_tokens,
                old_tokens + roles[0],
                new_tokens + roles[1],
                innovation_tokens + roles[2],
            ],
            dim=1,
        )

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

    def _make_paired_fast_tokens(
        self,
        initial_batch: dict[str, Tensor],
        current_batch: dict[str, Tensor],
        delay: int,
    ) -> Tensor:
        """Encode a deterministic initial/current observation pair for delayed evaluation."""
        if not 0 <= delay <= self.config.max_inference_delay:
            raise ValueError(f"delay must be in [0, {self.config.max_inference_delay}], got {delay}")
        initial_cameras = []
        current_cameras = []
        first = initial_batch[next(iter(self.config.image_features))]
        batch_size = first.shape[0]
        device = first.device
        mean = torch.tensor(self.config.fast_image_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(self.config.fast_image_std, device=device).view(1, 3, 1, 1)
        for key in self.config.image_features:
            initial = initial_batch[key].to(device=device, dtype=torch.float32)
            current = current_batch[key].to(device=device, dtype=torch.float32)
            if initial_batch[key].dtype == torch.uint8:
                initial = initial / 255.0
            if current_batch[key].dtype == torch.uint8:
                current = current / 255.0
            if initial.shape[1] != 3:
                initial = initial.permute(0, 3, 1, 2)
                current = current.permute(0, 3, 1, 2)
            initial_cameras.append(
                (F.interpolate(initial, size=(self.config.fast_image_size,) * 2, mode="bilinear") - mean)
                / std
            )
            current_cameras.append(
                (F.interpolate(current, size=(self.config.fast_image_size,) * 2, mode="bilinear") - mean)
                / std
            )
        initial_pixels = torch.stack(initial_cameras, dim=1)
        current_pixels = torch.stack(current_cameras, dim=1)
        initial_features = self.fast_encoder(initial_pixels.to(self.fast_encoder_device))
        current_features = self.fast_encoder(current_pixels.to(self.fast_encoder_device))
        dtype = self.model.fast_resampler.feature_proj.weight.dtype
        resampler_device = self.model.fast_resampler.feature_proj.weight.device
        states = pad_vector(current_batch[OBS_STATE], self.config.max_state_dim).to(
            device=resampler_device, dtype=dtype
        )
        delays = torch.full((batch_size,), delay, dtype=torch.long, device=resampler_device)
        return self.model.fast_resampler(
            initial_features.to(device=resampler_device, dtype=dtype),
            current_features.to(device=resampler_device, dtype=dtype),
            delays=delays,
            states=states,
        )

    @torch.no_grad()
    def encode_fast_observation_packet(
        self,
        batch: dict[str, Tensor],
        *,
        capture_step: int,
        capture_timestamp: float,
        generation: int,
        camera_valid_mask: Tensor | None = None,
        executed_prefix: Tensor | None = None,
        initial_batch: dict[str, Tensor] | None = None,
    ) -> FastObservationPacket:
        """Encode and timestamp a complete multi-camera/proprio observation packet."""
        camera_count = len(self.config.image_features)
        if camera_valid_mask is None:
            camera_valid_mask = torch.ones(camera_count, dtype=torch.bool)
        if camera_valid_mask.shape != (camera_count,):
            raise ValueError(f"camera_valid_mask must have shape ({camera_count},)")
        tokens = (
            self._make_paired_fast_tokens(initial_batch, batch, capture_step)
            if initial_batch is not None
            else self._make_single_fast_tokens(batch, capture_step)
        )
        if capture_step > 0 and (
            executed_prefix is None
            or executed_prefix.ndim != 3
            or executed_prefix.shape[:2] != (1, capture_step)
        ):
            raise ValueError(
                "online fast observation requires executed_prefix with shape (1, capture_step, action_dim)"
            )
        return FastObservationPacket(
            tokens=tokens,
            capture_step=capture_step,
            capture_timestamp=float(capture_timestamp),
            ready_timestamp=wall_time.monotonic(),
            proprio=batch[OBS_STATE].detach(),
            camera_valid_mask=camera_valid_mask.detach().cpu(),
            generation=generation,
            executed_prefix=executed_prefix.detach() if executed_prefix is not None else None,
        )

    def push_fast_observation(self, packet: FastObservationPacket) -> bool:
        return self.fast_observation_buffer.push(packet)

    def set_training_step(self, step: int) -> None:
        """Update schedule state supplied by the outer training loop."""
        self._training_step = int(step)

    def _slow_visual_dropout_probability(self) -> float:
        initial = self.config.slow_visual_dropout_probability
        anneal_steps = self.config.slow_visual_dropout_anneal_steps
        if anneal_steps <= 0:
            return initial
        progress = (self._training_step - self.config.slow_visual_dropout_anneal_start_step) / anneal_steps
        progress = min(max(progress, 0.0), 1.0)
        final = self.config.slow_visual_dropout_final_probability
        return initial + progress * (final - initial)

    def _sample_slow_visual_masks(self, images: list[Tensor], batch_size: int) -> list[Tensor] | None:
        """Sample slow-vision token masks shared by all online-flow training paths."""
        probability = self._slow_visual_dropout_probability()
        if not self.training or probability <= 0:
            return None
        device = images[0].device
        selected = (
            torch.rand(batch_size, device=device) < probability
        )
        side = (
            self.model.paligemma_with_expert.paligemma.config.vision_config.image_size
            // self.model.paligemma_with_expert.paligemma.config.vision_config.patch_size
        )
        count = side * side
        masks = []
        for _image in images:
            fraction = torch.empty(batch_size, device=device).uniform_(
                self.config.slow_visual_dropout_min, self.config.slow_visual_dropout_max
            )
            ranks = torch.rand(batch_size, count, device=device).argsort(1).argsort(1)
            keep = ranks >= (fraction * count).long()[:, None]
            masks.append(keep | ~selected[:, None])
        return masks

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

        # A mask is sampled once per sample/camera before slow VLM fusion.
        # It is reused during checkpoint recomputation through the supplied tensors.
        slow_visual_masks = self._sample_slow_visual_masks(images, actions.shape[0])

        original_action_dim = self.config.output_features[ACTION].shape[0]
        positions = torch.arange(actions.shape[1], device=actions.device)[None, :]
        suffix_mask = positions >= delays[:, None]
        model_time = torch.where(suffix_mask, time[:, None], torch.zeros_like(time[:, None]))
        model_particle = model_time.unsqueeze(-1) * noise + (1 - model_time.unsqueeze(-1)) * actions
        target_velocity = noise - actions

        predicted_velocity = self.model.predict_velocity(
            images,
            img_masks,
            tokens,
            masks,
            model_particle,
            model_time,
            states=None,
            state_masks=None,
            fast_adapters=self.model.fast_adapters,
            fast_tokens=fast_tokens,
            fast_delays=delays,
            slow_visual_masks=slow_visual_masks,
        )
        losses = (predicted_velocity - target_velocity).square()[..., :original_action_dim]

        # delay is measured in action/control steps. Zero-based action index ``delay`` is the
        # first action that can still be executed after the new observation becomes available.
        if "action_is_pad" in batch:
            suffix_mask &= ~batch["action_is_pad"].to(device=actions.device, dtype=torch.bool)
        mask = suffix_mask.unsqueeze(-1)
        per_sample_count = suffix_mask.sum(dim=1).clamp_min(1) * original_action_dim
        loss_per_sample = (losses * mask).sum(dim=(1, 2)) / per_sample_count
        loss_per_dim = (losses * mask).sum(dim=(0, 1)) / suffix_mask.sum().clamp_min(1)

        loss_dict = {
            "loss_per_dim": loss_per_dim.detach().cpu().numpy().tolist(),
            "mean_delay": delays.float().mean().item(),
            "clean_flow_loss": loss_per_sample.mean().item(),
        }
        if reduction == "none":
            loss = loss_per_sample
            loss_dict["loss"] = loss.mean().item()
            return loss, loss_dict
        loss = loss_per_sample.mean()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def _forward_online_flow_canonical(
        self, batch: dict[str, Tensor], reduction: str
    ) -> tuple[Tensor, dict]:
        """Train the unified old/new-conditioned vector field on the deployment clock."""
        valid_observations = self._observation_validity(batch)
        old_delays, new_delays, time, total_delays = sample_online_flow_transitions(
            valid_observations, self.config.num_inference_steps
        )
        initial_batch = self._initial_slow_batch(batch)
        images, img_masks = self._preprocess_images(initial_batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        noise = self.model.sample_noise(actions.shape, actions.device)
        old_fast_tokens = self._make_fast_tokens(batch, old_delays)
        new_fast_tokens = self._make_fast_tokens(batch, new_delays)
        online_tokens = self._make_online_condition_tokens(
            old_fast_tokens, new_fast_tokens, old_delays, new_delays
        )
        slow_visual_masks = self._sample_slow_visual_masks(images, actions.shape[0])

        action_dim = self.config.output_features[ACTION].shape[0]
        positions = torch.arange(actions.shape[1], device=actions.device)[None, :]
        suffix_mask = positions >= new_delays[:, None]
        model_time = torch.where(suffix_mask, time[:, None], torch.zeros_like(time[:, None]))
        particle = model_time.unsqueeze(-1) * noise + (1 - model_time.unsqueeze(-1)) * actions
        target_velocity = noise - actions
        predicted_velocity = self.model.predict_velocity(
            images,
            img_masks,
            tokens,
            masks,
            particle,
            model_time,
            states=None,
            state_masks=None,
            fast_adapters=self.model.fast_adapters,
            fast_tokens=online_tokens,
            fast_delays=new_delays,
            slow_visual_masks=slow_visual_masks,
        )
        losses = (predicted_velocity - target_velocity).square()[..., :action_dim]
        if "action_is_pad" in batch:
            suffix_mask &= ~batch["action_is_pad"].to(device=actions.device, dtype=torch.bool)
        mask = suffix_mask.unsqueeze(-1)
        counts = suffix_mask.sum(dim=1).clamp_min(1) * action_dim
        loss_per_sample = (losses * mask).sum(dim=(1, 2)) / counts
        loss_per_dim = (losses * mask).sum(dim=(0, 1)) / suffix_mask.sum().clamp_min(1)
        metrics = {
            "loss_per_dim": loss_per_dim.detach().cpu().numpy().tolist(),
            "canonical_flow_loss": loss_per_sample.mean().item(),
            "mean_old_delay": old_delays.float().mean().item(),
            "mean_new_delay": new_delays.float().mean().item(),
            "mean_total_delay": total_delays.float().mean().item(),
            "identity_fraction": (old_delays == new_delays).float().mean().item(),
            "mean_flow_time": time.mean().item(),
        }
        if reduction == "none":
            metrics["loss"] = loss_per_sample.mean().item()
            return loss_per_sample, metrics
        loss = loss_per_sample.mean()
        metrics["loss"] = loss.item()
        return loss, metrics

    def _sample_rollout_schedule(self, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample one recovery point from the same linear clock used at inference."""
        num_steps = self.config.num_inference_steps
        schedules = torch.zeros(valid.shape[0], num_steps, dtype=torch.long, device=valid.device)
        recovery_steps = torch.ones(valid.shape[0], dtype=torch.long, device=valid.device)
        schedule_modes = torch.zeros(valid.shape[0], dtype=torch.long, device=valid.device)
        max_step = min(self.config.online_flow_rollout_steps, num_steps - 1)
        for row in range(valid.shape[0]):
            positive = torch.nonzero(valid[row, 1:], as_tuple=False).flatten() + 1
            if not positive.numel():
                continue
            total = int(positive[torch.randint(positive.numel(), (), device=valid.device)].item())
            mode = 0
            schedule = torch.arange(num_steps, device=valid.device) * total // (num_steps - 1)
            schedules[row] = schedule
            schedule_modes[row] = mode
            recovery_steps[row] = torch.randint(1, max_step + 1, (), device=valid.device)
        return schedules, recovery_steps, schedule_modes

    def _forward_online_flow_ema_rollout(
        self, batch: dict[str, Tensor], reduction: str
    ) -> tuple[Tensor, dict]:
        """Recover from particles visited by a causal, hard-committed EMA rollout."""
        if reduction != "mean":
            raise ValueError("EMA-rollout phase currently supports reduction='mean' only")
        teacher = self._online_flow_teacher
        if teacher is None:
            raise RuntimeError("EMA rollout teacher was not initialized by the trainer")

        valid = self._observation_validity(batch)
        actions = self.prepare_action(batch)
        delay_schedules, rollout_lengths, schedule_modes = self._sample_rollout_schedule(valid)
        batch_size = actions.shape[0]
        particle = self.model.sample_noise(actions.shape, actions.device)
        initial_batch = self._initial_slow_batch(batch)

        with torch.no_grad():
            teacher_images, teacher_img_masks = teacher._preprocess_images(initial_batch)
            tokens = batch[OBS_LANGUAGE_TOKENS]
            masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
            teacher_fast_tokens: dict[int, Tensor] = {}
            for step in range(int(rollout_lengths.max().item())):
                active = step < rollout_lengths
                old_step = max(step - 1, 0)
                old_delays = delay_schedules[:, old_step]
                new_delays = delay_schedules[:, step]
                if old_step not in teacher_fast_tokens:
                    teacher_fast_tokens[old_step] = teacher._make_fast_tokens(batch, old_delays)
                if step not in teacher_fast_tokens:
                    teacher_fast_tokens[step] = teacher._make_fast_tokens(batch, new_delays)
                old_fast = teacher_fast_tokens[old_step]
                new_fast = teacher_fast_tokens[step]
                condition = teacher._make_online_condition_tokens(
                    old_fast, new_fast, old_delays, new_delays
                )
                current_time = torch.full(
                    (batch_size,),
                    1.0 - step / self.config.num_inference_steps,
                    device=actions.device,
                    dtype=torch.float32,
                )
                positions = torch.arange(actions.shape[1], device=actions.device)[None, :]
                model_time = torch.where(
                    positions < new_delays[:, None],
                    torch.zeros_like(current_time[:, None]),
                    current_time[:, None],
                )
                velocity = teacher.model.predict_velocity(
                    teacher_images,
                    teacher_img_masks,
                    tokens,
                    masks,
                    particle,
                    model_time,
                    fast_adapters=teacher.model.fast_adapters,
                    fast_tokens=condition,
                    fast_delays=new_delays,
                )
                proposal = particle - velocity / self.config.num_inference_steps
                proposal = hard_commit_action_prefix(proposal, actions, new_delays)
                particle = torch.where(active[:, None, None], proposal, particle)

        recovery_step = rollout_lengths
        old_steps = (recovery_step - 1).clamp_min(0)
        rows = torch.arange(batch_size, device=actions.device)
        old_delays = delay_schedules[rows, old_steps]
        new_delays = delay_schedules[rows, recovery_step]
        particle = hard_commit_action_prefix(particle, actions, new_delays).detach()
        time = 1.0 - recovery_step.to(torch.float32) / self.config.num_inference_steps
        target_velocity = (particle - actions) / time[:, None, None].clamp_min(1e-6)

        images, img_masks = self._preprocess_images(initial_batch)
        slow_visual_masks = self._sample_slow_visual_masks(images, batch_size)
        old_fast = self._make_fast_tokens(batch, old_delays)
        new_fast = self._make_fast_tokens(batch, new_delays)
        condition = self._make_online_condition_tokens(old_fast, new_fast, old_delays, new_delays)
        positions = torch.arange(actions.shape[1], device=actions.device)[None, :]
        suffix_mask = positions >= new_delays[:, None]
        model_time = torch.where(suffix_mask, time[:, None], torch.zeros_like(time[:, None]))
        prediction = self.model.predict_velocity(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            particle,
            model_time,
            fast_adapters=self.model.fast_adapters,
            fast_tokens=condition,
            fast_delays=new_delays,
            slow_visual_masks=slow_visual_masks,
        )
        action_dim = self.config.output_features[ACTION].shape[0]
        if "action_is_pad" in batch:
            suffix_mask &= ~batch["action_is_pad"].to(device=actions.device, dtype=torch.bool)
        squared = (prediction - target_velocity).square()[..., :action_dim]
        counts = suffix_mask.sum(dim=1).clamp_min(1) * action_dim
        rollout_loss = (squared * suffix_mask.unsqueeze(-1)).sum(dim=(1, 2)) / counts
        rollout_loss = rollout_loss.mean()
        loss = self.config.online_flow_rollout_weight * rollout_loss
        switch_rows = new_delays > old_delays
        per_sample_loss = (squared * suffix_mask.unsqueeze(-1)).sum(dim=(1, 2)) / counts
        metrics = {
            "online_rollout_loss": rollout_loss.detach().item(),
            "mean_rollout_steps": rollout_lengths.float().mean().item(),
            "rollout_mean_old_delay": old_delays.float().mean().item(),
            "rollout_mean_new_delay": new_delays.float().mean().item(),
            "rollout_switch_fraction": switch_rows.float().mean().item(),
            "switch_rollout_loss": per_sample_loss[switch_rows].mean().detach().item()
            if switch_rows.any()
            else 0.0,
            "identity_rollout_loss": per_sample_loss[~switch_rows].mean().detach().item()
            if (~switch_rows).any()
            else 0.0,
            "linear_schedule_fraction": (schedule_modes == 0).float().mean().item(),
            "jitter_schedule_fraction": (schedule_modes == 1).float().mean().item(),
            "burst_schedule_fraction": (schedule_modes == 2).float().mean().item(),
            "stall_schedule_fraction": (schedule_modes == 3).float().mean().item(),
            "slow_visual_dropout_probability": self._slow_visual_dropout_probability(),
            "loss": loss.detach().item(),
        }
        return loss, metrics

    @torch.no_grad()
    def _integrate_paired_teacher_particles(
        self,
        *,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
        base_noise: Tensor,
        executed_actions: Tensor,
        switch_steps: Tensor,
        old_delays: Tensor,
        new_delays: Tensor,
        old_fast_tokens: Tensor,
        new_fast_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the frozen Stage-A expert under old/new conditions from shared noise to one grid time."""
        batch_size = base_noise.shape[0]
        particle = torch.cat([base_noise, base_noise], dim=0)
        paired_actions = torch.cat([executed_actions, executed_actions], dim=0)
        paired_steps = torch.cat([switch_steps, switch_steps], dim=0)
        paired_delays = torch.cat([old_delays, new_delays], dim=0)
        particle = hard_commit_action_prefix(particle, paired_actions, paired_delays)
        paired_fast_tokens = torch.cat([old_fast_tokens, new_fast_tokens], dim=0)
        paired_images = [torch.cat([image, image], dim=0) for image in images]
        paired_img_masks = [torch.cat([mask, mask], dim=0) for mask in img_masks]
        paired_tokens = torch.cat([tokens, tokens], dim=0)
        paired_masks = torch.cat([masks, masks], dim=0)
        step_size = -1.0 / self.config.num_inference_steps
        for step in range(int(switch_steps.max().item())):
            current_time = torch.full(
                (2 * batch_size,),
                1.0 - step / self.config.num_inference_steps,
                device=particle.device,
                dtype=torch.float32,
            )
            positions = torch.arange(particle.shape[1], device=particle.device)[None, :]
            model_time = torch.where(
                positions < paired_delays[:, None],
                torch.zeros_like(current_time[:, None]),
                current_time[:, None],
            )
            velocity = self.model.predict_velocity(
                paired_images,
                paired_img_masks,
                paired_tokens,
                paired_masks,
                particle,
                model_time,
                fast_adapters=self.model.fast_adapters,
                fast_tokens=paired_fast_tokens,
                fast_delays=paired_delays,
            )
            active = step < paired_steps
            particle = torch.where(active[:, None, None], particle + step_size * velocity, particle)
            particle = hard_commit_action_prefix(particle, paired_actions, paired_delays)
        return particle[:batch_size], particle[batch_size:]

    @torch.no_grad()
    def _make_stage_b_pair(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Materialize one paired teacher target for training or fixed-pair diagnostics."""
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
        switch_steps, target_time = sample_midflow_times(
            actions.shape[0], self.config.num_inference_steps, actions.device
        )

        identity_rows = torch.rand(actions.shape[0], device=actions.device) < (
            self.config.posterior_identity_dropout
        )
        old_delays = torch.where(identity_rows, new_delays, old_delays)

        prefix_embs, prefix_valid, _ = self.model.embed_prefix(images, img_masks, tokens, masks)
        old_fast_tokens = self._make_fast_tokens(batch, old_delays)
        new_fast_tokens = self._make_fast_tokens(batch, new_delays)
        old_particle, target_particle = self._integrate_paired_teacher_particles(
            images=images,
            img_masks=img_masks,
            tokens=tokens,
            masks=masks,
            base_noise=base_noise,
            executed_actions=actions,
            switch_steps=switch_steps,
            old_delays=old_delays,
            new_delays=new_delays,
            old_fast_tokens=old_fast_tokens,
            new_fast_tokens=new_fast_tokens,
        )
        suffix_mask = torch.arange(actions.shape[1], device=actions.device)[None, :] >= new_delays[:, None]
        if "action_is_pad" in batch:
            suffix_mask &= ~batch["action_is_pad"].to(device=actions.device, dtype=torch.bool)
        return {
            "prefix_embs": prefix_embs.detach(),
            "prefix_valid": prefix_valid,
            "old_fast_tokens": old_fast_tokens.detach(),
            "new_fast_tokens": new_fast_tokens.detach(),
            "old_particle": old_particle.detach(),
            "target_particle": target_particle.detach(),
            "executed_actions": actions.detach(),
            "old_delays": old_delays,
            "new_delays": new_delays,
            "target_time": target_time,
            "switch_steps": switch_steps,
            "identity_rows": identity_rows,
            "suffix_mask": suffix_mask,
        }

    def _stage_b_loss_from_pair(self, pair: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Train the updater from a materialized pair without rerunning the frozen teacher."""
        prefix_embs = pair["prefix_embs"]
        prefix_valid = pair["prefix_valid"]
        old_fast_tokens = pair["old_fast_tokens"]
        new_fast_tokens = pair["new_fast_tokens"]
        old_particle = pair["old_particle"]
        target_particle = pair["target_particle"]
        actions = pair["executed_actions"]
        old_delays = pair["old_delays"]
        new_delays = pair["new_delays"]
        target_time = pair["target_time"]
        suffix_mask = pair["suffix_mask"]
        action_dim = self.config.output_features[ACTION].shape[0]
        slow_tokens = self.posterior_slow_resampler(prefix_embs.detach().float(), prefix_valid)
        condition_tokens = self._make_posterior_condition_tokens(
            slow_tokens, old_fast_tokens.detach(), new_fast_tokens.detach()
        )
        prediction, residual_gate = self.posterior_updater(
            old_particle.float(),
            old_delays=old_delays,
            new_delays=new_delays,
            flow_times=target_time.float(),
            condition_tokens=condition_tokens,
            executed_actions=actions.float(),
            return_gate=True,
        )

        teacher_delta_mse = masked_particle_mse(
            target_particle.float(), old_particle.float(), suffix_mask, action_dim
        )
        endpoint_loss = masked_particle_mse(prediction, target_particle.float(), suffix_mask, action_dim)
        relative_loss, cosine_loss, delta_magnitude_ratio = masked_residual_objective(
            prediction,
            old_particle.float(),
            target_particle.float(),
            suffix_mask,
            action_dim,
            self.config.posterior_relative_floor,
        )
        student_delta_mse = masked_particle_mse(prediction, old_particle.float(), suffix_mask, action_dim)
        teacher_delta = target_particle.float() - old_particle.float()
        student_delta = prediction - old_particle.float()
        selected = suffix_mask.unsqueeze(-1).expand(-1, -1, action_dim)
        teacher_selected = teacher_delta[..., :action_dim][selected]
        student_selected = student_delta[..., :action_dim][selected]
        delta_cosine = torch.dot(teacher_selected, student_selected) / (
            teacher_selected.norm() * student_selected.norm()
        ).clamp_min(1e-12)
        bridge_gain = 1 - endpoint_loss / teacher_delta_mse.clamp_min(1e-12)
        gate_selected = residual_gate[suffix_mask]
        loss = (
            self.config.posterior_endpoint_weight * endpoint_loss
            + self.config.posterior_relative_weight * relative_loss
            + self.config.posterior_cosine_weight * cosine_loss
        )
        return loss, {
            "loss": loss.item(),
            "posterior_endpoint_loss": endpoint_loss.item(),
            "posterior_relative_loss": relative_loss.item(),
            "posterior_cosine_loss": cosine_loss.item(),
            "teacher_delta_mse": teacher_delta_mse.item(),
            "student_delta_mse": student_delta_mse.detach().item(),
            "delta_cosine_similarity": delta_cosine.detach().item(),
            "bridge_gain": bridge_gain.detach().item(),
            "posterior_gate_mean": gate_selected.detach().mean().item(),
            "delta_magnitude_ratio": delta_magnitude_ratio.detach().item(),
            "mean_old_delay": old_delays.float().mean().item(),
            "mean_new_delay": new_delays.float().mean().item(),
            "mean_gap": (new_delays - old_delays).float().mean().item(),
            "identity_fraction": pair["identity_rows"].float().mean().item(),
            "mean_switch_step": pair["switch_steps"].float().mean().item(),
            "mean_switch_flow_time": target_time.mean().item(),
        }

    def _forward_stage_b(self, batch: dict[str, Tensor], reduction: str) -> tuple[Tensor, dict]:
        if reduction != "mean":
            raise ValueError("Stage-B currently supports reduction='mean' only")
        return self._stage_b_loss_from_pair(self._make_stage_b_pair(batch))

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        if self.config.training_stage == "stage_a":
            return self._forward_stage_a(batch, reduction)
        if self.config.training_stage == "online_flow":
            if self.config.online_flow_phase == "ema_rollout":
                return self._forward_online_flow_ema_rollout(batch, reduction)
            return self._forward_online_flow_canonical(batch, reduction)
        return self._forward_stage_b(batch, reduction)

    def begin_online_generation(self) -> int:
        """Start an online generation and return its stale-work rejection token."""
        self._online_generation += 1
        self.fast_observation_buffer.clear()
        return self._online_generation

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Generate a chunk while assimilating the latest completed fast packets.

        A simulator may supply ``observation_callback(step, num_steps, generation)``
        to advance its control clock and push causal packets before every query.
        ``assimilation_callback(step, capture_step, updater_used)`` reports consumed
        packets. Stage A switches conditions directly; Stage B reanchors first.
        """
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
        use_updater = self.config.training_stage == "stage_b"
        use_online_flow = self.config.training_stage == "online_flow"
        slow_condition_tokens = (
            self.posterior_slow_resampler(prefix_embs.detach().float(), prefix_pad_masks)
            if use_updater
            else None
        )
        num_steps = int(kwargs.pop("num_steps", self.config.num_inference_steps))
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        noise = kwargs.pop("noise", None)
        observation_callback = kwargs.pop("observation_callback", None)
        if observation_callback is not None and num_steps < 2:
            raise ValueError("continuous observations require at least two denoising steps")
        assimilation_callback = kwargs.pop("assimilation_callback", None)
        observation_age_override = kwargs.pop("observation_age_override", None)
        if observation_age_override is not None and observation_age_override < 0:
            raise ValueError("observation_age_override must be non-negative")
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
        committed_actions = None
        step_size = -1.0 / num_steps

        for step in range(num_steps):
            flow_time = torch.full((1,), 1.0 - step / num_steps, device=particle.device, dtype=torch.float32)
            previous_fast_tokens = current_fast_tokens
            previous_version = current_version
            if observation_callback is not None:
                observation_callback(step, num_steps, generation)
            query_timestamp = wall_time.monotonic()
            packet = self.fast_observation_buffer.latest_ready_after(
                current_version,
                generation=generation,
                query_timestamp=query_timestamp,
            )
            if packet is not None:
                if packet.executed_prefix is None:
                    raise ValueError("online observation packet is missing executed_prefix")
                if packet.capture_step >= particle.shape[1]:
                    raise ValueError("observation capture_step exceeds action chunk horizon")
                committed_actions = torch.zeros_like(particle)
                prefix = packet.executed_prefix.to(committed_actions)
                if prefix.shape[-1] > committed_actions.shape[-1]:
                    raise ValueError("executed_prefix action dimension exceeds model action dimension")
                committed_actions[:, : packet.capture_step, : prefix.shape[-1]] = prefix
                old_delay = torch.tensor([current_version], device=particle.device)
                new_delay = torch.tensor([packet.capture_step], device=particle.device)
                age = torch.tensor(
                    [
                        observation_age_override
                        if observation_age_override is not None
                        else query_timestamp - packet.capture_timestamp
                    ],
                    device=particle.device,
                    dtype=torch.float32,
                )
                fast_tokens = packet.tokens.to(device=particle.device)
                if use_updater:
                    condition_tokens = self._make_posterior_condition_tokens(
                        slow_condition_tokens, current_fast_tokens, fast_tokens
                    )
                    particle = self.posterior_updater(
                        particle.float(),
                        old_delays=old_delay,
                        new_delays=new_delay,
                        flow_times=flow_time,
                        observation_ages=age,
                        condition_tokens=condition_tokens,
                        executed_actions=committed_actions,
                    ).to(noise.dtype)
                else:
                    particle = hard_commit_action_prefix(particle, committed_actions, new_delay)
                if assimilation_callback is not None:
                    assimilation_callback(step, packet.capture_step, use_updater)
                current_fast_tokens = fast_tokens
                current_version = packet.capture_step
                current_capture_timestamp = packet.capture_timestamp

            age = torch.tensor(
                [
                    observation_age_override
                    if observation_age_override is not None
                    else query_timestamp - current_capture_timestamp
                ],
                device=particle.device,
                dtype=torch.float32,
            )
            delay = torch.tensor([current_version], device=particle.device)
            if use_online_flow:
                old_delay = torch.tensor([previous_version], device=particle.device)
                fast_tokens_for_step = self._make_online_condition_tokens(
                    previous_fast_tokens,
                    current_fast_tokens,
                    old_delay,
                    delay,
                )
            else:
                fast_tokens_for_step = current_fast_tokens
            velocity = self.model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=particle,
                timestep=flow_time,
                prefix_embs=prefix_embs,
                fast_adapters=self.model.fast_adapters,
                fast_tokens=fast_tokens_for_step,
                fast_delays=delay,
                fast_observation_ages=age,
                committed_delays=delay if committed_actions is not None else None,
            )
            particle = particle + step_size * velocity
            if committed_actions is not None:
                particle = hard_commit_action_prefix(particle, committed_actions, delay)

        action_dim = self.config.output_features[ACTION].shape[0]
        return particle[:, :, :action_dim]

    @torch.no_grad()
    def predict_action_chunk_with_delay(
        self,
        initial_batch: dict[str, Tensor],
        current_batch: dict[str, Tensor],
        delay: int,
        *,
        noise: Tensor | None = None,
        num_steps: int | None = None,
        switch_step: int | None = None,
        executed_prefix: Tensor | None = None,
    ) -> Tensor:
        """Generate a chunk for deterministic simulated control-step delay.

        ``executed_prefix`` contains policy-normalized actions issued during the delay.

        Stage A directly generates with the observation at ``t+delay``. Stage B starts under the
        old condition, reanchors the current particle at a strictly interior flow time when the
        new observation arrives, and completes the remaining denoising under the new condition.
        """
        self.eval()
        if initial_batch[OBS_STATE].shape[0] != 1 or current_batch[OBS_STATE].shape[0] != 1:
            raise ValueError("delayed online inference currently requires batch size one")
        if delay > 0 and executed_prefix is None:
            raise ValueError("delayed inference requires the executed action prefix for hard commitment")
        images, img_masks = self._preprocess_images(initial_batch)
        tokens = initial_batch[OBS_LANGUAGE_TOKENS]
        masks = initial_batch[OBS_LANGUAGE_ATTENTION_MASK]
        prefix_embs, prefix_pad_masks, past_key_values = self.model.prepare_prefix_cache(
            images, img_masks, tokens, masks
        )
        new_fast_tokens = self._make_paired_fast_tokens(initial_batch, current_batch, delay)
        use_posterior_bridge = self.config.training_stage == "stage_b" and delay > 0
        fast_tokens = (
            self._make_single_fast_tokens(initial_batch, delay=0) if use_posterior_bridge else new_fast_tokens
        )
        if noise is None:
            noise = self.model.sample_noise(
                (1, self.config.chunk_size, self.config.max_action_dim), tokens.device
            )
        particle = noise
        hard_prefix_mask = None
        if executed_prefix is not None:
            if executed_prefix.ndim != 3 or executed_prefix.shape[:2] != (1, delay):
                raise ValueError(f"executed_prefix must have shape (1, {delay}, action_dim)")
            if executed_prefix.shape[-1] > particle.shape[-1]:
                raise ValueError("executed_prefix action dimension exceeds the model dimension")
            hard_prefix = torch.zeros_like(particle)
            hard_prefix[:, :delay, : executed_prefix.shape[-1]] = executed_prefix.to(hard_prefix)
            hard_prefix_mask = torch.arange(particle.shape[1], device=particle.device)[None, :, None] < delay
            if not use_posterior_bridge:
                particle = torch.where(hard_prefix_mask, hard_prefix, particle)
        steps = self.config.num_inference_steps if num_steps is None else num_steps
        if steps < 2:
            raise ValueError("mid-flow delayed inference requires at least two denoising steps")
        if use_posterior_bridge:
            switch_step = steps // 2 if switch_step is None else switch_step
            if not 1 <= switch_step < steps:
                raise ValueError(f"switch_step must satisfy 1 <= switch_step < {steps}")
        step_size = -1.0 / steps
        delay_tensor = torch.tensor([0 if use_posterior_bridge else delay], device=particle.device)
        for step in range(steps):
            flow_time = torch.full((1,), 1.0 - step / steps, device=particle.device, dtype=torch.float32)
            if use_posterior_bridge and step == switch_step:
                slow_condition_tokens = self.posterior_slow_resampler(
                    prefix_embs.detach().float(), prefix_pad_masks
                )
                condition_tokens = self._make_posterior_condition_tokens(
                    slow_condition_tokens, fast_tokens, new_fast_tokens
                )
                particle = self.posterior_updater(
                    particle.float(),
                    old_delays=torch.zeros(1, dtype=torch.long, device=particle.device),
                    new_delays=torch.tensor([delay], dtype=torch.long, device=particle.device),
                    flow_times=flow_time,
                    condition_tokens=condition_tokens,
                    executed_actions=hard_prefix if hard_prefix_mask is not None else None,
                ).to(noise.dtype)
                fast_tokens = new_fast_tokens
                delay_tensor.fill_(delay)
            velocity = self.model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=particle,
                timestep=flow_time,
                prefix_embs=prefix_embs,
                fast_adapters=self.model.fast_adapters,
                fast_tokens=fast_tokens,
                fast_delays=delay_tensor,
                committed_delays=delay_tensor
                if hard_prefix_mask is not None and (not use_posterior_bridge or step >= switch_step)
                else None,
            )
            particle = particle + step_size * velocity
            if hard_prefix_mask is not None and (not use_posterior_bridge or step >= switch_step):
                particle = torch.where(hard_prefix_mask, hard_prefix, particle)
        action_dim = self.config.output_features[ACTION].shape[0]
        return particle[:, :, :action_dim]
