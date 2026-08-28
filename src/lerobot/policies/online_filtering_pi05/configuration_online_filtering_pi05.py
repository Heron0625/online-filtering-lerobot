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
    fast_num_heads: int = 8
    fast_proprio_num_tokens: int = 1
    fast_proprio_hidden_dim: int = 256
    fast_adapter_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17)
    fast_image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    fast_image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    training_stage: str = "stage_a"
    teacher_integration_steps: int = 10
    stage_b_min_gap: int = 1
    stage_b_max_gap: int | None = None
    posterior_hidden_dim: int = 256
    posterior_num_heads: int = 8
    posterior_num_layers: int = 2
    posterior_slow_tokens: int = 4
    posterior_endpoint_weight: float = 1.0
    posterior_identity_dropout: float = 0.1

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
        if self.training_stage not in {"stage_a", "stage_b"}:
            raise ValueError("training_stage must be 'stage_a' or 'stage_b'")
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
        if self.posterior_endpoint_weight < 0:
            raise ValueError("posterior loss weights must be non-negative")
        if not 0 <= self.posterior_identity_dropout <= 1:
            raise ValueError("posterior_identity_dropout must be in [0, 1]")
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
