#!/usr/bin/env python

from types import SimpleNamespace

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies.online_filtering_pi05.configuration_online_filtering_pi05 import (
    OnlineFilteringPI05Config,
)
from lerobot.policies.online_filtering_pi05.modeling_online_filtering_pi05 import (
    OnlineFilteringPI05Policy,
    make_structured_action_corruption,
    masked_particle_mse,
    masked_residual_objective,
    sample_adjacent_filtrations,
    sample_causal_delays,
    sample_midflow_times,
    sample_online_flow_transitions,
    sample_stage_a_delays_and_branches,
    select_temporal_steps,
)
from lerobot.policies.online_filtering_pi05.processor_online_filtering_pi05 import (
    reconcile_online_filtering_pi05_pretrained_processors,
)
from lerobot.policies.pi05.modeling_pi05 import _mask_committed_queries_from_free_actions
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


def test_full_slow_visual_dropout_masks_every_patch() -> None:
    fake = SimpleNamespace(
        training=True,
        _slow_visual_dropout_probability=lambda: 1.0,
        config=SimpleNamespace(
            slow_visual_dropout_probability=1.0,
            slow_visual_dropout_min=1.0,
            slow_visual_dropout_max=1.0,
        ),
        model=SimpleNamespace(
            paligemma_with_expert=SimpleNamespace(
                paligemma=SimpleNamespace(
                    config=SimpleNamespace(
                        vision_config=SimpleNamespace(image_size=32, patch_size=16)
                    )
                )
            )
        ),
    )
    images = [torch.zeros(3, 3, 32, 32), torch.zeros(3, 3, 32, 32)]
    masks = OnlineFilteringPI05Policy._sample_slow_visual_masks(fake, images, batch_size=3)
    assert masks is not None
    assert len(masks) == 2
    assert all(mask.shape == (3, 4) for mask in masks)
    assert not any(mask.any() for mask in masks)


def test_slow_visual_dropout_probability_anneals_to_inference_distribution() -> None:
    fake = SimpleNamespace(
        _training_step=0,
        config=SimpleNamespace(
            slow_visual_dropout_probability=0.25,
            slow_visual_dropout_final_probability=0.0,
            slow_visual_dropout_anneal_start_step=6_000,
            slow_visual_dropout_anneal_steps=18_000,
        ),
    )
    probability = OnlineFilteringPI05Policy._slow_visual_dropout_probability
    assert probability(fake) == pytest.approx(0.25)
    fake._training_step = 15_000
    assert probability(fake) == pytest.approx(0.125)
    fake._training_step = 24_000
    assert probability(fake) == pytest.approx(0.0)


def test_online_condition_contains_only_old_and_new_blocks() -> None:
    hidden = 8
    fake = SimpleNamespace(
        config=SimpleNamespace(max_inference_delay=10),
        online_condition_roles=torch.nn.Parameter(torch.zeros(2, 1, hidden)),
        online_delay_pair_proj=torch.nn.Linear(3, hidden, bias=False),
    )
    torch.nn.init.zeros_(fake.online_delay_pair_proj.weight)
    old = torch.randn(2, 5, hidden)
    new = torch.randn(2, 5, hidden)
    condition = OnlineFilteringPI05Policy._make_online_condition_tokens(
        fake, old, new, torch.tensor([1, 2]), torch.tensor([2, 4])
    )
    assert condition.shape == (2, 10, hidden)
    assert torch.equal(condition[:, :5], old)
    assert torch.equal(condition[:, 5:], new)


def test_stage_b_relative_residual_objective_balances_target_scales() -> None:
    old = torch.zeros(2, 2, 1)
    target = torch.tensor([[[1.0], [1.0]], [[0.01], [0.01]]])
    prediction = target * 0.5
    mask = torch.ones(2, 2, dtype=torch.bool)
    relative, cosine, magnitude_ratio = masked_residual_objective(
        prediction, old, target, mask, action_dim=1, relative_floor=1e-8
    )
    assert relative.item() == pytest.approx(0.25)
    assert cosine.item() == pytest.approx(0.0, abs=1e-6)
    assert magnitude_ratio.item() == pytest.approx(0.5)


def test_stage_b_relative_residual_objective_ignores_zero_direction() -> None:
    particle = torch.zeros(1, 2, 1)
    relative, cosine, _ = masked_residual_objective(
        particle, particle, particle, torch.ones(1, 2, dtype=torch.bool), 1, 1e-6
    )
    assert relative.item() == 0
    assert cosine.item() == 0


def test_sample_causal_delays_never_selects_padding() -> None:
    valid = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
    generator = torch.Generator().manual_seed(7)
    for _ in range(20):
        delays = sample_causal_delays(valid, generator=generator)
        assert torch.all(valid[torch.arange(3), delays])
        assert delays[1].item() == 0


def test_online_flow_transitions_match_continuous_evaluation_clock() -> None:
    valid = torch.ones(128, 11, dtype=torch.bool)
    old, new, times, total = sample_online_flow_transitions(
        valid, 10, generator=torch.Generator().manual_seed(31)
    )
    query_steps = torch.round((1 - times) * 10).long()
    assert torch.equal(new, query_steps * total // 9)
    assert torch.equal(old, (query_steps - 1).clamp_min(0) * total // 9)
    assert torch.all((total >= 1) & (total <= 10))
    assert torch.all(old <= new)
    assert torch.all((times >= 0.1) & (times <= 1.0))


def test_online_flow_transition_uses_only_valid_observations() -> None:
    valid = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
    old, new, _, total = sample_online_flow_transitions(
        valid, 10, generator=torch.Generator().manual_seed(37)
    )
    rows = torch.arange(2)
    assert torch.all(valid[rows, old])
    assert torch.all(valid[rows, new])
    assert total[0] in (1, 2)
    assert total[1] == 0


def test_stage_a_clean_flow_uses_one_delay_for_prefix_and_suffix() -> None:
    from lerobot.policies.pi05.online_filtering import build_stage_a_flow_inputs

    actions = torch.arange(2 * 6 * 7, dtype=torch.float32).reshape(2, 6, 7)
    noise = torch.full_like(actions, -1)
    delays = torch.tensor([2, 4])
    inputs = build_stage_a_flow_inputs(actions, noise, torch.tensor([0.5, 0.25]), delays)
    for row, delay in enumerate(delays.tolist()):
        assert torch.equal(inputs.noisy_actions[row, :delay], actions[row, :delay])
        assert torch.all(inputs.model_time[row, :delay] == 0)
        assert torch.all(inputs.timeline.free_mask[row, :delay] == 0)
        assert torch.all(inputs.timeline.free_mask[row, delay:])


def test_delay_masks_future_action_keys_from_committed_queries() -> None:
    attention = torch.ones(2, 6, 6, dtype=torch.bool)
    masked = _mask_committed_queries_from_free_actions(attention, torch.tensor([2, 4]))
    assert not masked[0, :2, 2:].any()
    assert not masked[1, :4, 4:].any()
    assert masked[0, 2:].all()
    assert masked[1, 4:].all()


def test_stage_b_teacher_particles_commit_each_delay_before_integrating() -> None:
    calls = []

    def predict_velocity(*args, **kwargs):
        calls.append((args[4].clone(), args[5].clone(), kwargs["fast_delays"].clone()))
        return torch.ones_like(args[4])

    fake = SimpleNamespace(
        config=SimpleNamespace(num_inference_steps=10),
        model=SimpleNamespace(predict_velocity=predict_velocity, fast_adapters=None),
    )
    actions = torch.arange(6 * 7, dtype=torch.float32).reshape(1, 6, 7)
    old, new = OnlineFilteringPI05Policy._integrate_paired_teacher_particles(
        fake,
        images=[torch.zeros(1, 1)],
        img_masks=[torch.ones(1, 1, dtype=torch.bool)],
        tokens=torch.zeros(1, 1),
        masks=torch.ones(1, 1, dtype=torch.bool),
        base_noise=torch.zeros_like(actions),
        executed_actions=actions,
        switch_steps=torch.tensor([2]),
        old_delays=torch.tensor([2]),
        new_delays=torch.tensor([4]),
        old_fast_tokens=torch.zeros(1, 1, 1),
        new_fast_tokens=torch.zeros(1, 1, 1),
    )
    assert torch.equal(old[:, :2], actions[:, :2])
    assert torch.equal(new[:, :4], actions[:, :4])
    assert torch.all(calls[0][1][0, :2] == 0)
    assert torch.all(calls[0][1][1, :4] == 0)
    assert calls[0][2].tolist() == [2, 4]


def test_stage_a_switch_examples_always_have_positive_valid_delay() -> None:
    valid = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 0, 1]], dtype=torch.bool)
    generator = torch.Generator().manual_seed(17)
    for _ in range(20):
        delays, switch = sample_stage_a_delays_and_branches(valid, 0.0, generator)
        rows = torch.arange(valid.shape[0])
        assert torch.all(valid[rows, delays])
        assert not switch[1]
        assert torch.all(delays[switch] > 0)


@pytest.mark.parametrize("corruption_type", range(3))
def test_structured_corruption_has_requested_suffix_rms(corruption_type: int) -> None:
    actions = torch.linspace(-1, 1, 2 * 8 * 7).reshape(2, 8, 7)
    delays = torch.tensor([2, 4])
    target_rms = torch.tensor([0.1, 0.2])
    corrupted, realized_rms = make_structured_action_corruption(
        actions,
        delays,
        action_dim=7,
        target_rms=target_rms,
        corruption_types=torch.full((2,), corruption_type),
        generator=torch.Generator().manual_seed(23),
    )
    assert corrupted.shape == actions.shape
    assert torch.allclose(realized_rms, target_rms, atol=1e-5)
    assert torch.equal(corrupted[..., -1], actions[..., -1])


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


def test_stage_b_switch_times_match_inference_grid_and_exclude_endpoints() -> None:
    steps, times = sample_midflow_times(
        256, 10, "cpu", generator=torch.Generator().manual_seed(29)
    )
    assert torch.all((steps >= 1) & (steps < 10))
    assert torch.all((times > 0) & (times < 1))
    assert torch.allclose(times, 1.0 - steps.float() / 10)


def test_stage_b_config_validation() -> None:
    stage_a = _config(training_stage="stage_a", stage_b_min_gap=2)
    stage_b = _config(training_stage="stage_b", stage_b_min_gap=2, stage_b_max_gap=4)
    assert stage_a.drop_n_last_frames == 0
    assert stage_b.drop_n_last_frames == 2
    online = _config(training_stage="online_flow")
    assert online.online_flow_phase == "canonical"
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
