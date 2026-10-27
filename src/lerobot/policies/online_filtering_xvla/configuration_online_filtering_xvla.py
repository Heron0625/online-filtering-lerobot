from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.xvla.configuration_xvla import XVLAConfig


@PreTrainedConfig.register_subclass("online_filtering_xvla")
@dataclass
class OnlineFilteringXVLAConfig(XVLAConfig):
    """X-VLA with a shared old/new observation encoder and online-flow training."""

    training_stage: str = "stage_a"
    max_inference_delay: int = 10
    online_flow_ema_decay: float = 0.999
    online_flow_rollout_steps: int = 9

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

    @property
    def image_observation_delta_indices(self) -> list[int]:
        return list(range(self.max_inference_delay + 1))

    @property
    def state_observation_delta_indices(self) -> list[int]:
        return list(range(self.max_inference_delay + 1))

