"""Regression for newly executed prefixes when the online teacher delay grows."""

from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.online_filtering_pi05.modeling_online_filtering_pi05 import OnlineFilteringPI05Policy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


@pytest.mark.parametrize("lengths", [[4], [1, 4]])
def test_rollout_commits_new_prefix_before_each_teacher_prediction(lengths):
    batch_size = len(lengths)
    recovery = torch.tensor(lengths)
    actions = torch.arange(1, batch_size * 100 + 1, dtype=torch.float32).reshape(batch_size, 50, 2)
    schedule = torch.tensor([[0, 3, 6, 10, 13, 16, 20, 23, 26, 30]]).expand(batch_size, -1)
    calls = []

    def predict(*args, **kwargs):
        particle = args[4]
        delays = kwargs["fast_delays"]
        for row, delay in enumerate(delays.tolist()):
            assert torch.equal(particle[row, :delay], actions[row, :delay]), (
                "teacher/student must see all newly executed actions before predicting",
                row,
                delay,
            )
            if torch.is_grad_enabled():
                # Stopped rows must retain their actual rollout suffix, without later GT commits.
                assert torch.allclose(
                    particle[row, delay:], torch.full_like(particle[row, delay:], -lengths[row] / 10)
                )
        calls.append(delays.tolist())
        return torch.ones_like(particle, requires_grad=torch.is_grad_enabled())

    pixels = [torch.zeros(batch_size, 3, 2, 2)]
    masks = [torch.ones(batch_size, dtype=torch.bool)]
    def fast(batch, delays):
        return torch.zeros(batch_size, 1, 2)

    def condition(old, new, old_delay, new_delay):
        return torch.cat([old, new], 1)
    teacher = SimpleNamespace(
        model=SimpleNamespace(predict_velocity=predict, fast_adapters=None),
        _preprocess_images=lambda batch: (pixels, masks),
        _make_fast_tokens=fast,
        _make_online_condition_tokens=condition,
    )
    policy = SimpleNamespace(
        config=SimpleNamespace(
            num_inference_steps=10,
            online_flow_rollout_weight=1.0,
            output_features={ACTION: SimpleNamespace(shape=(2,))},
        ),
        _online_flow_teacher=teacher,
        _observation_validity=lambda batch: torch.ones(batch_size, 31, dtype=torch.bool),
        prepare_action=lambda batch: actions,
        _sample_rollout_schedule=lambda valid: (
            schedule,
            recovery,
            torch.zeros(batch_size, dtype=torch.long),
        ),
        _initial_slow_batch=lambda batch: batch,
        model=SimpleNamespace(
            sample_noise=lambda shape, device: torch.zeros(shape, device=device),
            predict_velocity=predict,
            fast_adapters=None,
        ),
        _preprocess_images=lambda batch: (pixels, masks),
        _make_fast_tokens=fast,
        _make_online_condition_tokens=condition,
        _sample_slow_visual_masks=lambda images, batch_size: None,
        _slow_visual_dropout_probability=lambda: 0.0,
    )
    batch = {
        OBS_LANGUAGE_TOKENS: torch.zeros(batch_size, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, 2, dtype=torch.bool),
    }
    loss, metrics = OnlineFilteringPI05Policy._forward_online_flow_ema_rollout(policy, batch, "mean")
    assert len(calls) == 5
    assert calls[:4] == [[d] * batch_size for d in [0, 3, 6, 10]]
    assert calls[-1] == [int(schedule[row, length]) for row, length in enumerate(lengths)]
    assert torch.isfinite(loss) and metrics["online_rollout_loss"] == metrics["loss"]
    loss.backward()
