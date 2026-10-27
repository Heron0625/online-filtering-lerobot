from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.online_filtering_xvla.configuration_online_filtering_xvla import (
    OnlineFilteringXVLAConfig,
)
from lerobot.policies.online_filtering_xvla.modeling_online_filtering_xvla import (
    OnlineFilteringXVLAPolicy,
    XVLAFastAdapter,
    XVLAObservationPacket,
)
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS


def test_fast_adapter_starts_as_exact_identity_and_keeps_prefix() -> None:
    adapter = XVLAFastAdapter(8, 2)
    action = torch.randn(2, 5, 8)
    condition = torch.randn(2, 6, 8)
    output = adapter(action, condition, torch.tensor([0, 3]))
    torch.testing.assert_close(output, action)
    (output.square().mean()).backward()
    assert adapter.output_proj.weight.grad is not None


def test_online_filtering_xvla_temporal_window() -> None:
    config = OnlineFilteringXVLAConfig(chunk_size=30, n_action_steps=30, max_inference_delay=25)
    assert config.image_observation_delta_indices == list(range(26))
    assert config.state_observation_delta_indices == list(range(26))
    assert not config.freeze_vision_encoder
    assert not config.freeze_language_encoder


def test_stage_b_requires_positive_delay() -> None:
    with pytest.raises(ValueError, match="requires max_inference_delay"):
        OnlineFilteringXVLAConfig(training_stage="stage_b", max_inference_delay=0)


def test_online_xvla_assimilates_packet_and_commits_prefix() -> None:
    calls = []

    class FakeModel:
        dim_action = 2
        action_space = SimpleNamespace(postprocess=lambda value: value)

        @staticmethod
        def _get_target_dtype():
            return torch.float32

        @staticmethod
        def predict_endpoint(**kwargs):
            calls.append((kwargs["old_delay"].item(), kwargs["new_delay"].item()))
            return torch.ones_like(kwargs["particle"])

    policy = SimpleNamespace(
        config=SimpleNamespace(chunk_size=4, num_denoising_steps=3, training_stage="stage_b"),
        model=FakeModel(),
        _online_generation=0,
        _pending_online_packet=None,
    )
    policy.begin_online_generation = lambda: OnlineFilteringXVLAPolicy.begin_online_generation(policy)
    policy._take_online_packet = lambda delay, generation: OnlineFilteringXVLAPolicy._take_online_packet(
        policy, delay, generation
    )
    condition = {
        "vlm_features": torch.zeros(1, 1, 2),
        "aux_visual_inputs": torch.zeros(1, 1, 2),
        "primary_visual_tokens": 1,
    }
    policy._slow_inputs = lambda batch: (
        condition,
        torch.zeros(1, 2),
        torch.zeros(1, dtype=torch.long),
    )
    policy._encode_runtime_fast = lambda batch: (
        torch.zeros(1, 1, 2),
        torch.zeros(1, dtype=torch.long),
    )
    batch = {OBS_LANGUAGE_TOKENS: torch.zeros(1, 2, dtype=torch.long)}

    def callback(step, num_steps, generation):
        if step == 1:
            packet = XVLAObservationPacket(
                tokens=torch.ones(1, 1, 2),
                domain_id=torch.zeros(1, dtype=torch.long),
                capture_step=1,
                generation=generation,
                executed_prefix=torch.zeros(1, 1, 2),
            )
            OnlineFilteringXVLAPolicy.push_fast_observation(policy, packet)

    output = OnlineFilteringXVLAPolicy.predict_action_chunk(
        policy,
        batch,
        observation_callback=callback,
        noise=torch.zeros(1, 4, 2),
    )
    assert output.shape == (1, 4, 2)
    assert calls == [(0, 0), (0, 1), (1, 1)]
    torch.testing.assert_close(output[:, 0], torch.zeros(1, 2))
