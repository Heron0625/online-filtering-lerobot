#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("online_filtering_pi05")
@dataclass
class OnlineFilteringPI05Config(PI05Config):
    """Observation-filtered PI0.5 configuration."""

    max_inference_delay: int = 10
    prompt_state_temporal_index: int = 0

    dinov3_model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    freeze_dinov3: bool = True
    fast_encoder_device: str | None = None
    fast_image_size: int = 224
    fast_num_tokens: int = 16
    fast_spatial_tokens: bool = False  # Legacy checkpoints retain their original resampler.
    slow_visual_dropout_probability: float = 0.0
    slow_visual_dropout_final_probability: float = 0.0
    slow_visual_dropout_anneal_start_step: int = 0
    slow_visual_dropout_anneal_steps: int = 0
    slow_visual_dropout_min: float = 0.5
    slow_visual_dropout_max: float = 0.75
    fast_num_heads: int = 8
    fast_proprio_num_tokens: int = 1
    fast_proprio_hidden_dim: int = 256
    fast_adapter_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17)
    fast_image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    fast_image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    training_stage: str = "stage_a"
    online_flow_phase: str = "canonical"
    online_flow_ema_decay: float = 0.999
    online_flow_rollout_steps: int = 4
    online_flow_switch_probability: float = 0.8
    online_flow_canonical_weight: float = 1.0
    online_flow_rollout_weight: float = 1.0
    # Retained for loading older checkpoints; current Stage A trains clean flow only.
    stage_a_clean_probability: float = 1.0
    stage_a_corruption_mild_probability: float = 0.7
    stage_a_corruption_medium_probability: float = 0.25
    stage_a_corruption_mild_rms: float = 0.05
    stage_a_corruption_medium_rms: float = 0.12
    stage_a_corruption_strong_rms: float = 0.2
    stage_a_corruption_clip: float = 5.0
    teacher_integration_steps: int = 10
    stage_b_min_gap: int = 1
    stage_b_max_gap: int | None = None
    posterior_hidden_dim: int = 256
    posterior_num_heads: int = 8
    posterior_num_layers: int = 2
    posterior_slow_tokens: int = 4
    posterior_endpoint_weight: float = 1.0
    posterior_relative_weight: float = 1.0
    posterior_cosine_weight: float = 0.1
    posterior_relative_floor: float = 1e-6
    posterior_use_learned_gate: bool = False
    posterior_identity_dropout: float = 0.0
    # Kept for backward-compatible checkpoint loading. Mid-flow Stage B must leave this at zero.
    stage_b_endpoint_probability: float = 0.0

    fast_buffer_capacity: int = 4
    fast_observation_max_age_s: float = 0.5
    merge_horizon: int = 4

    # Stage A keeps the slow VLM frozen while training the action expert and new branch.
    train_expert_only: bool = True

    def __post_init__(self) -> None:
        # The parent RTC sampler is not the source of Stage-A delay masks.
        self.rtc_training_max_delay = 0
        super().__post_init__()
        if not 0 <= self.max_inference_delay < self.chunk_size:
            raise ValueError(
                "max_inference_delay must satisfy "
                f"0 <= delay < chunk_size ({self.chunk_size}), got {self.max_inference_delay}"
            )
        if self.use_visual_memory or self.use_proprioceptive_memory:
            raise ValueError("online_filtering_pi05 does not combine Stage-A windows with PI0.5 MEM")
        if self.fast_image_size < 1 or self.fast_num_tokens < 1:
            raise ValueError("fast_image_size and fast_num_tokens must be positive")
        if (
            self.fast_spatial_tokens
            and self.image_features
            and self.fast_num_tokens % len(self.image_features)
        ):
            raise ValueError("fast_num_tokens must be divisible by the camera count")
        if not 0 <= self.slow_visual_dropout_probability <= 1:
            raise ValueError("slow_visual_dropout_probability must be in [0, 1]")
        if not 0 <= self.slow_visual_dropout_final_probability <= 1:
            raise ValueError("slow_visual_dropout_final_probability must be in [0, 1]")
        if self.slow_visual_dropout_anneal_start_step < 0 or self.slow_visual_dropout_anneal_steps < 0:
            raise ValueError("slow visual dropout anneal steps must be non-negative")
        if not 0 <= self.slow_visual_dropout_min <= self.slow_visual_dropout_max <= 1:
            raise ValueError("slow visual dropout range must satisfy 0 <= min <= max <= 1")
        if self.fast_proprio_num_tokens != 1:
            raise ValueError("the first online-filtering version requires one proprio token")
        if self.fast_proprio_hidden_dim < 1:
            raise ValueError("fast_proprio_hidden_dim must be positive")
        if not self.fast_adapter_layers:
            raise ValueError("at least one fast adapter layer is required")
        if len(set(self.fast_adapter_layers)) != len(self.fast_adapter_layers):
            raise ValueError("fast_adapter_layers must not contain duplicates")
        if min(self.fast_adapter_layers) < 0 or max(self.fast_adapter_layers) >= 18:
            raise ValueError("fast_adapter_layers must be valid action-expert layer indices in [0, 18)")
        if self.training_stage not in {"stage_a", "stage_b", "online_flow"}:
            raise ValueError("training_stage must be 'stage_a', 'stage_b', or 'online_flow'")
        if self.online_flow_phase not in {"canonical", "ema_rollout"}:
            raise ValueError("online_flow_phase must be 'canonical' or 'ema_rollout'")
        if not 0 < self.online_flow_ema_decay < 1:
            raise ValueError("online_flow_ema_decay must be in (0, 1)")
        if not 1 <= self.online_flow_rollout_steps <= self.num_inference_steps:
            raise ValueError("online_flow_rollout_steps must be within the inference grid")
        if not 0 <= self.online_flow_switch_probability <= 1:
            raise ValueError("online_flow_switch_probability must be in [0, 1]")
        if self.online_flow_canonical_weight < 0 or self.online_flow_rollout_weight < 0:
            raise ValueError("online-flow loss weights must be non-negative")
        if not 0 <= self.stage_a_clean_probability <= 1:
            raise ValueError("stage_a_clean_probability must be in [0, 1]")
        if not 0 <= self.stage_a_corruption_mild_probability <= 1:
            raise ValueError("stage_a_corruption_mild_probability must be in [0, 1]")
        if not 0 <= self.stage_a_corruption_medium_probability <= 1:
            raise ValueError("stage_a_corruption_medium_probability must be in [0, 1]")
        if self.stage_a_corruption_mild_probability + self.stage_a_corruption_medium_probability > 1:
            raise ValueError("Stage-A mild and medium corruption probabilities must sum to at most 1")
        if not (
            0
            < self.stage_a_corruption_mild_rms
            <= self.stage_a_corruption_medium_rms
            <= self.stage_a_corruption_strong_rms
        ):
            raise ValueError("Stage-A corruption RMS levels must be positive and nondecreasing")
        if self.stage_a_corruption_clip <= 0:
            raise ValueError("stage_a_corruption_clip must be positive")
        if self.teacher_integration_steps < 1:
            raise ValueError("teacher_integration_steps must be positive")
        if self.stage_b_min_gap < 1:
            raise ValueError("stage_b_min_gap must be positive")
        if self.stage_b_max_gap is None:
            self.stage_b_max_gap = self.max_inference_delay
        if not self.stage_b_min_gap <= self.stage_b_max_gap <= self.max_inference_delay:
            raise ValueError("Stage-B gaps must satisfy 1 <= min_gap <= max_gap <= max_inference_delay")
        if self.posterior_hidden_dim % self.posterior_num_heads != 0:
            raise ValueError("posterior_hidden_dim must be divisible by posterior_num_heads")
        if self.posterior_num_layers < 1:
            raise ValueError("posterior_num_layers must be positive")
        if self.posterior_slow_tokens < 1:
            raise ValueError("posterior_slow_tokens must be positive")
        if min(
            self.posterior_endpoint_weight,
            self.posterior_relative_weight,
            self.posterior_cosine_weight,
        ) < 0:
            raise ValueError("posterior loss weights must be non-negative")
        if self.posterior_relative_floor <= 0:
            raise ValueError("posterior_relative_floor must be positive")
        if not 0 <= self.posterior_identity_dropout <= 1:
            raise ValueError("posterior_identity_dropout must be in [0, 1]")
        if not 0 <= self.stage_b_endpoint_probability <= 1:
            raise ValueError("stage_b_endpoint_probability must be in [0, 1]")
        if self.fast_buffer_capacity < 1:
            raise ValueError("fast_buffer_capacity must be positive")
        if self.fast_observation_max_age_s <= 0:
            raise ValueError("fast_observation_max_age_s must be positive")
        if self.merge_horizon < 0:
            raise ValueError("merge_horizon must be non-negative")

    @property
    def image_observation_delta_indices(self) -> list[int]:
        return list(range(self.max_inference_delay + 1))

    @property
    def state_observation_delta_indices(self) -> list[int]:
        return list(range(self.max_inference_delay + 1))

    @property
    def drop_n_last_frames(self) -> int:
        """Exclude episode tails that cannot form a valid Stage-B filtration pair."""
        return self.stage_b_min_gap if self.training_stage == "stage_b" else 0
