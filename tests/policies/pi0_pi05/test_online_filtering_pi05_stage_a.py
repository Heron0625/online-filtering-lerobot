#!/usr/bin/env python

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies.online_filtering_pi05.configuration_online_filtering_pi05 import (
    OnlineFilteringPI05Config,
)
from lerobot.policies.online_filtering_pi05.modeling_online_filtering_pi05 import (
    OnlineFilteringPI05Policy,
    masked_particle_mse,
    sample_adjacent_filtrations,
    sample_causal_delays,
    select_temporal_steps,
)
from lerobot.policies.online_filtering_pi05.processor_online_filtering_pi05 import (
    reconcile_online_filtering_pi05_pretrained_processors,
)
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import PolicyProcessorPipeline


def _config(**kwargs) -> OnlineFilteringPI05Config:
    return OnlineFilteringPI05Config(
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        **kwargs,
    )


def test_policy_registration_resolves_config_and_model() -> None:
    assert PreTrainedConfig.get_choice_class("online_filtering_pi05") is OnlineFilteringPI05Config
    assert get_policy_class("online_filtering_pi05") is OnlineFilteringPI05Policy


def test_stage_a_delta_indices_are_future_closed_loop_window() -> None:
    config = _config(max_inference_delay=4)
    assert config.image_observation_delta_indices == [0, 1, 2, 3, 4]
    assert config.state_observation_delta_indices == [0, 1, 2, 3, 4]
    assert config.prompt_state_temporal_index == 0


def test_sample_causal_delays_never_selects_padding() -> None:
    valid = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
    generator = torch.Generator().manual_seed(7)
    for _ in range(20):
        delays = sample_causal_delays(valid, generator=generator)
        assert torch.all(valid[torch.arange(3), delays])
        assert delays[1].item() == 0


def test_select_temporal_steps_is_per_example() -> None:
    values = torch.arange(3 * 4 * 2).reshape(3, 4, 2)
    indices = torch.tensor([0, 2, 3])
    selected = select_temporal_steps(values, indices)
    assert torch.equal(selected, torch.stack([values[0, 0], values[1, 2], values[2, 3]]))


def test_config_rejects_mem_and_invalid_delay() -> None:
    with pytest.raises(ValueError, match="max_inference_delay"):
        _config(max_inference_delay=50)
    with pytest.raises(ValueError, match="does not combine"):
        _config(use_visual_memory=True)


def test_stage_b_samples_only_valid_adjacent_filtrations() -> None:
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 1]], dtype=torch.bool)
    generator = torch.Generator().manual_seed(11)
    for _ in range(20):
        delays = sample_adjacent_filtrations(valid, generator=generator)
        rows = torch.arange(valid.shape[0])
        assert torch.all(valid[rows, delays])
        assert torch.all(valid[rows, delays + 1])
        assert delays[1].item() == 0
        assert delays[2].item() == 2


def test_stage_b_particle_loss_masks_prefix_and_padded_dimensions() -> None:
    prediction = torch.zeros(1, 4, 5)
    target = torch.zeros_like(prediction)
    prediction[:, :2, :3] = 10
    prediction[:, 2:, :3] = 2
    prediction[:, 2:, 3:] = 100
    loss = masked_particle_mse(prediction, target, torch.tensor([[0, 0, 1, 1]]), action_dim=3)
    assert loss.item() == 4


def test_stage_b_config_validation() -> None:
    assert _config(training_stage="stage_b").training_stage == "stage_b"
    with pytest.raises(ValueError, match="training_stage"):
        _config(training_stage="joint")
    with pytest.raises(ValueError, match="divisible"):
        _config(posterior_hidden_dim=30, posterior_num_heads=8)


def test_pretrained_processor_is_reconciled_to_initial_state() -> None:
    state_step = Pi05PrepareStateTokenizerProcessorStep(temporal_index=-1)
    preprocessor = PolicyProcessorPipeline(steps=[state_step])
    postprocessor = PolicyProcessorPipeline(steps=[])
    reconciled, _ = reconcile_online_filtering_pi05_pretrained_processors(
        _config(), preprocessor, postprocessor
    )
    assert reconciled.steps[0].temporal_index == 0
