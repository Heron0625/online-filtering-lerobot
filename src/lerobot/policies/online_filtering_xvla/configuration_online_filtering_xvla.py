from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.xvla.configuration_xvla import XVLAConfig


@PreTrainedConfig.register_subclass("online_filtering_xvla")
@dataclass
class OnlineFilteringXVLAConfig(XVLAConfig):
    """X-VLA with a cached slow condition and an independent DINO fast stream."""

    training_stage: str = "stage_a"
    max_inference_delay: int = 25
    online_flow_ema_decay: float = 0.999
    online_flow_rollout_steps: int = 9
    dinov3_model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    freeze_dinov3: bool = True
    fast_image_size: int = 224
    fast_num_tokens: int = 64
    fast_num_heads: int = 8
    fast_adapter_layers: tuple[int, ...] = (3, 7, 11, 15, 19, 23)
    slow_visual_dropout_probability: float = 0.0
    slow_visual_dropout_final_probability: float = 0.25
    slow_visual_dropout_anneal_start_step: int = 500
    slow_visual_dropout_anneal_steps: int = 2000
    main_delay_probability: float = 0.5
    medium_delay_probability: float = 0.25

    # Full fine-tuning is the default. These inherited switches remain available
    # for memory ablations and do not change the two-condition architecture.
    freeze_vision_encoder: bool = False
    freeze_language_encoder: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.training_stage not in {"stage_a", "stage_b"}:
            raise ValueError("training_stage must be 'stage_a' or 'stage_b'")
        if not 0 <= self.max_inference_delay < self.chunk_size:
            raise ValueError("max_inference_delay must satisfy 0 <= delay < chunk_size")
        if self.training_stage == "stage_b" and self.max_inference_delay == 0:
            raise ValueError("stage_b requires max_inference_delay > 0")
        if not 0 < self.online_flow_ema_decay < 1:
            raise ValueError("online_flow_ema_decay must be in (0, 1)")
        if not 1 <= self.online_flow_rollout_steps < self.num_denoising_steps:
            raise ValueError("online_flow_rollout_steps must be in [1, num_denoising_steps)")
        if self.fast_num_tokens < 2 or self.fast_num_tokens % 2:
            raise ValueError("fast_num_tokens must be a positive even number for two LIBERO cameras")
        if self.hidden_size % self.fast_num_heads:
            raise ValueError("hidden_size must be divisible by fast_num_heads")
        if not self.fast_adapter_layers or min(self.fast_adapter_layers) < 0:
            raise ValueError("fast_adapter_layers must contain valid transformer layers")
        if max(self.fast_adapter_layers) >= self.depth:
            raise ValueError("fast_adapter_layers exceed the X-VLA transformer depth")
        if not 0 <= self.slow_visual_dropout_probability <= 1:
            raise ValueError("slow_visual_dropout_probability must be in [0, 1]")
        if not 0 <= self.slow_visual_dropout_final_probability <= 1:
            raise ValueError("slow_visual_dropout_final_probability must be in [0, 1]")
        if self.slow_visual_dropout_anneal_start_step < 0 or self.slow_visual_dropout_anneal_steps < 0:
            raise ValueError("slow visual dropout schedule must be non-negative")
        if not 0 <= self.main_delay_probability <= 1:
            raise ValueError("main_delay_probability must be in [0, 1]")
        if not 0 <= self.medium_delay_probability <= 1 - self.main_delay_probability:
            raise ValueError("delay mixture probabilities are invalid")

    @property
    def image_observation_delta_indices(self) -> list[int]:
        return list(range(self.max_inference_delay + 1))

    @property
    def state_observation_delta_indices(self) -> list[int]:
        return list(range(self.max_inference_delay + 1))
