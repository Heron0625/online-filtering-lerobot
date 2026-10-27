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


@pytest.mark.parametrize("stage", ["stage_a", "stage_b"])
def test_multiple_packets_assimilated_before_later_denoising_steps(stage):
    updates, denoise_calls, assimilations = [], [], []

    def updater(x, **kwargs):
        updates.append(
            (kwargs["old_delays"].item(), kwargs["new_delays"].item(), kwargs["flow_times"].item())
        )
        return hard_commit_action_prefix(x + 3, kwargs["executed_actions"], kwargs["new_delays"])

    def denoise(**kwargs):
        denoise_calls.append(kwargs["fast_delays"].item())
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
        fake.fast_observation_buffer.push(
            FastObservationPacket(
                tokens=torch.ones(1, 1, 4) * step,
                capture_step=step,
                capture_timestamp=timestamp,
                ready_timestamp=timestamp,
                proprio=batch[OBS_STATE],
                camera_valid_mask=torch.ones(1, dtype=torch.bool),
                generation=generation,
                executed_prefix=torch.full((1, step, 2), 10.0),
            )
        )

    result = OnlineFilteringPI05Policy.predict_action_chunk(
        fake,
        batch,
        online_generation=1,
        noise=torch.zeros(1, 8, 2),
        observation_callback=arrive,
        assimilation_callback=lambda *args: assimilations.append(args),
    )
    assert denoise_calls == [0, 1, 1, 3, 3]
    assert assimilations == [(1, 1, stage == "stage_b"), (3, 3, stage == "stage_b")]
    assert torch.equal(result[:, :3], torch.full((1, 3, 2), 10.0))
    if stage == "stage_b":
        assert [(a, b) for a, b, _ in updates] == [(0, 1), (1, 3)]
        assert torch.all(result[:, 3:] == 6)
    else:
        assert updates == []
        assert torch.all(result[:, 3:] == 0)
