import pytest
import torch

from lerobot.policies.online_filtering_xvla.configuration_online_filtering_xvla import (
    OnlineFilteringXVLAConfig,
)
from lerobot.policies.online_filtering_xvla.modeling_online_filtering_xvla import DelayConditionGate


def test_delay_condition_gate_starts_as_symmetric_average() -> None:
    gate = DelayConditionGate(7)
    value = gate(torch.tensor([0.0, 0.2]), torch.tensor([0.4, 1.0]), torch.tensor([1.0, 0.5]))
    torch.testing.assert_close(value, torch.full((2, 7), 0.5))


def test_online_filtering_xvla_temporal_window() -> None:
    config = OnlineFilteringXVLAConfig(chunk_size=30, n_action_steps=30, max_inference_delay=10)
    assert config.image_observation_delta_indices == list(range(11))
    assert config.state_observation_delta_indices == list(range(11))
    assert not config.freeze_vision_encoder
    assert not config.freeze_language_encoder


def test_stage_b_requires_positive_delay() -> None:
    with pytest.raises(ValueError, match="requires max_inference_delay"):
        OnlineFilteringXVLAConfig(training_stage="stage_b", max_inference_delay=0)

