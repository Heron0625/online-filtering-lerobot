"""Exercise the actual denoising loop with multiple causally arriving packets."""

import time
from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.online_filtering_pi05.modeling_online_filtering_pi05 import OnlineFilteringPI05Policy
from lerobot.policies.pi05.online_filtering import (
    FastObservationBuffer,
    FastObservationPacket,
    hard_commit_action_prefix,
)
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


@pytest.mark.parametrize("initial_age", [0.0, 0.25])
@pytest.mark.parametrize("stage", ["stage_a", "stage_b", "online_flow"])
@pytest.mark.parametrize("delivery", ["buffered", "direct"])
def test_multiple_packets_assimilated_before_later_denoising_steps(stage, delivery, initial_age):
    updates, denoise_calls, assimilations, observation_ages = [], [], [], []

    def updater(x, **kwargs):
        updates.append(
            (kwargs["old_delays"].item(), kwargs["new_delays"].item(), kwargs["flow_times"].item())
        )
        return hard_commit_action_prefix(x + 3, kwargs["executed_actions"], kwargs["new_delays"])

    def denoise(**kwargs):
        denoise_calls.append(kwargs["fast_delays"].item())
        observation_ages.append(kwargs["fast_observation_ages"].item())
        return torch.zeros_like(kwargs["x_t"])

    model = SimpleNamespace(
        prepare_prefix_cache=lambda *args: (torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool), None),
        denoise_step=denoise,
        fast_adapters=None,
    )
    fake = SimpleNamespace(
        config=SimpleNamespace(
            training_stage=stage,
            image_features={"image": None},
            chunk_size=8,
            max_action_dim=2,
            num_inference_steps=5,
            output_features={"action": SimpleNamespace(shape=(2,))},
        ),
        model=model,
        eval=lambda: None,
        _online_generation=1,
        _preprocess_images=lambda b: ([b["image"]], [torch.ones(1, dtype=torch.bool)]),
        _make_single_fast_tokens=lambda b, delay: torch.zeros(1, 1, 4),
        posterior_slow_resampler=lambda *args: torch.zeros(1, 1, 4),
        _make_posterior_condition_tokens=lambda *args: torch.zeros(1, 3, 4),
        posterior_updater=updater,
        _make_online_condition_tokens=lambda old, new, *args: torch.cat([old, new], dim=1),
        fast_observation_buffer=FastObservationBuffer(),
    )
    batch = {
        "image": torch.zeros(1, 3, 2, 2),
        OBS_STATE: torch.zeros(1, 8),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.bool),
    }

    def arrive(step, num_steps, generation):
        if step not in [1, 3]:
            return
        timestamp = time.monotonic()
        packet = FastObservationPacket(
            tokens=torch.ones(1, 1, 4) * step,
            capture_step=step,
            capture_timestamp=timestamp,
            ready_timestamp=timestamp,
            proprio=batch[OBS_STATE],
            camera_valid_mask=torch.ones(1, dtype=torch.bool),
            generation=generation,
            executed_prefix=torch.full((1, step, 2), 10.0),
        )
        if delivery == "direct":
            return packet
        fake.fast_observation_buffer.push(packet)
        return None

    result = OnlineFilteringPI05Policy.predict_action_chunk(
        fake,
        batch,
        online_generation=1,
        initial_capture_timestamp=time.monotonic() - initial_age if initial_age else None,
        noise=torch.zeros(1, 8, 2),
        observation_callback=arrive,
        assimilation_callback=lambda *args: assimilations.append(args),
    )
    assert observation_ages[0] >= initial_age
    assert denoise_calls == [0, 1, 1, 3, 3]
    assert assimilations == [(1, 1, stage == "stage_b"), (3, 3, stage == "stage_b")]
    assert torch.equal(result[:, :3], torch.full((1, 3, 2), 10.0))
    if stage == "stage_b":
        assert [(a, b) for a, b, _ in updates] == [(0, 1), (1, 3)]
        assert torch.all(result[:, 3:] == 6)
    else:
        assert updates == []
        assert torch.all(result[:, 3:] == 0)


@pytest.mark.parametrize("delivery", ["buffered", "direct"])
def test_online_pairs_retain_previous_denoising_condition_with_front_loaded_delay(delivery):
    """A [step=0,capture=9] assimilation still supplies both old=0 and new=9."""
    pairs = []

    def condition(old_tokens, new_tokens, old_delay, new_delay):
        pair = (old_delay.item(), new_delay.item())
        assert old_tokens.item() == pair[0]
        assert new_tokens.item() == pair[1]
        pairs.append(pair)
        return torch.cat([old_tokens, new_tokens], dim=1)

    fake = SimpleNamespace(
        config=SimpleNamespace(
            training_stage="online_flow",
            image_features={"image": None},
            chunk_size=50,
            max_action_dim=2,
            num_inference_steps=4,
            output_features={"action": SimpleNamespace(shape=(2,))},
        ),
        model=SimpleNamespace(
            prepare_prefix_cache=lambda *args: (
                torch.zeros(1, 2, 4),
                torch.ones(1, 2, dtype=torch.bool),
                None,
            ),
            denoise_step=lambda **kwargs: torch.zeros_like(kwargs["x_t"]),
            fast_adapters=None,
        ),
        eval=lambda: None,
        _online_generation=1,
        _preprocess_images=lambda batch: ([batch["image"]], [torch.ones(1, dtype=torch.bool)]),
        _make_single_fast_tokens=lambda batch, delay: torch.full((1, 1, 1), float(delay)),
        _make_online_condition_tokens=condition,
        fast_observation_buffer=FastObservationBuffer(),
    )
    batch = {
        "image": torch.zeros(1, 3, 2, 2),
        OBS_STATE: torch.zeros(1, 14),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.bool),
    }
    assimilations = []

    def arrive(step, num_steps, generation):
        capture_step = {0: 9, 1: 12, 3: 15}.get(step)
        if capture_step is None:
            return None
        timestamp = time.monotonic()
        packet = FastObservationPacket(
            tokens=torch.full((1, 1, 1), float(capture_step)),
            capture_step=capture_step,
            capture_timestamp=timestamp,
            ready_timestamp=timestamp,
            proprio=batch[OBS_STATE],
            camera_valid_mask=torch.ones(1, dtype=torch.bool),
            generation=generation,
            executed_prefix=torch.full((1, capture_step, 2), 7.0),
        )
        if delivery == "direct":
            return packet
        fake.fast_observation_buffer.push(packet)
        return None

    result = OnlineFilteringPI05Policy.predict_action_chunk(
        fake,
        batch,
        online_generation=1,
        noise=torch.zeros(1, 50, 2),
        observation_callback=arrive,
        assimilation_callback=lambda *args: assimilations.append(args),
    )
    assert pairs == [(0, 9), (9, 12), (12, 12), (12, 15)]
    assert assimilations == [(0, 9, False), (1, 12, False), (3, 15, False)]
    assert torch.equal(result[:, :15], torch.full((1, 15, 2), 7.0))


@pytest.mark.parametrize("total_delay", [1, 9, 23, 30])
def test_training_rollout_clock_starts_at_zero_and_advances_linearly(monkeypatch, total_delay):
    import lerobot.policies.online_filtering_pi05.modeling_online_filtering_pi05 as modeling

    monkeypatch.setattr(
        modeling,
        "sample_online_total_delays",
        lambda valid, **kwargs: torch.full((valid.shape[0],), total_delay, dtype=torch.long),
    )
    fake = SimpleNamespace(
        config=SimpleNamespace(
            num_inference_steps=10,
            online_flow_rollout_steps=4,
            main_delay_probability=0.5,
            medium_delay_probability=0.25,
            delay_sampling_ranges=None,
            delay_sampling_probabilities=None,
        )
    )
    schedule, recovery_steps, modes = OnlineFilteringPI05Policy._sample_rollout_schedule(
        fake,
        torch.ones(4, 31, dtype=torch.bool),
    )
    expected = torch.arange(10) * total_delay // 9
    assert torch.equal(schedule, expected.expand(4, -1))
    assert schedule[:, 0].eq(0).all()
    assert torch.diff(schedule, dim=1).max().item() <= (total_delay + 8) // 9
    assert ((recovery_steps >= 1) & (recovery_steps <= 4)).all()
    assert modes.eq(0).all()  # Current code does not actually sample jitter/burst/stall clocks.
