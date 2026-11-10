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
    # State is also paired with every relative action to construct OSC goals.
    assert config.state_observation_delta_indices == list(range(30))
    assert not config.freeze_vision_encoder
    assert not config.freeze_language_encoder


def test_stage_b_requires_positive_delay() -> None:
    with pytest.raises(ValueError, match="requires max_inference_delay"):
        OnlineFilteringXVLAConfig(training_stage="stage_b", max_inference_delay=0)


@pytest.mark.parametrize("with_callback", [True, False])
def test_online_xvla_assimilates_packet_and_commits_prefix(with_callback) -> None:
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
        observation_callback=callback if with_callback else None,
        noise=torch.zeros(1, 4, 2),
    )
    assert output.shape == (1, 4, 2)
    assert calls == ([(0, 0), (0, 1), (1, 1)] if with_callback else [(0, 0)] * 3)
    torch.testing.assert_close(output[:, 0], torch.zeros(1, 2) if with_callback else torch.ones(1, 2))


def test_stage_a_commits_executed_prefix_before_prediction() -> None:
    targets = torch.arange(2 * 5 * 20).float().reshape(2, 5, 20) / 100
    new = torch.tensor([2, 4])
    captured = []

    def predict(batch, particle, time, old_delay, new_delay):
        captured.append(particle)
        return torch.zeros_like(particle)

    policy = SimpleNamespace(
        model=SimpleNamespace(_get_target_dtype=lambda: torch.float32),
        _prepare_action_targets=lambda batch: targets,
        _observation_validity=lambda batch: torch.ones(2, 5, dtype=torch.bool),
        _sample_schedule=lambda valid: (
            torch.tensor([1, 3]),
            new,
            torch.tensor([0.7, 0.3]),
            torch.tensor([3, 4]),
        ),
        _predict=predict,
        _masked_action_loss=lambda prediction, target, delay, batch: (
            torch.tensor(1.0),
            {"position_loss": torch.tensor(1.0)},
        ),
        _slow_dropout_probability=lambda: 0.0,
    )
    OnlineFilteringXVLAPolicy._stage_a_forward(policy, {})
    for row, delay in enumerate(new):
        torch.testing.assert_close(captured[0][row, :delay], targets[row, :delay])
        assert not torch.allclose(captured[0][row, delay:], targets[row, delay:])


def test_stage_b_commits_prefix_for_teacher_and_student() -> None:
    targets = torch.arange(2 * 5 * 20).float().reshape(2, 5, 20) / 100
    calls = []

    def predict(batch, particle, time, old, new, **kwargs):
        for row, delay in enumerate(new):
            torch.testing.assert_close(particle[row, :delay], targets[row, :delay])
        calls.append(new.clone())
        return torch.zeros_like(particle)

    teacher = SimpleNamespace(_slow_inputs=lambda batch: None, _predict=predict)
    policy = SimpleNamespace(
        config=SimpleNamespace(online_flow_rollout_steps=1, num_denoising_steps=2),
        _online_flow_teacher=teacher,
        model=SimpleNamespace(_get_target_dtype=lambda: torch.float32),
        _prepare_action_targets=lambda batch: targets,
        _observation_validity=lambda batch: torch.ones(2, 5, dtype=torch.bool),
        _sample_total_delays=lambda valid: torch.tensor([2, 4]),
        _predict=predict,
        _masked_action_loss=lambda prediction, target, delay, batch: (
            torch.tensor(1.0),
            {"position_loss": torch.tensor(1.0)},
        ),
        _slow_dropout_probability=lambda: 0.25,
    )
    OnlineFilteringXVLAPolicy._stage_b_forward(policy, {})
    assert len(calls) == 2
    torch.testing.assert_close(calls[-1], torch.tensor([2, 4]))


def test_slow_visual_dropout_blocks_image_information_in_language_tokens() -> None:
    from lerobot.policies.xvla.modeling_xvla import XVLAModel

    def encode(*, attention_mask, inputs_embeds):
        visible = inputs_embeds * attention_mask[..., None]
        return (inputs_embeds + visible.sum(dim=1, keepdim=True),)

    vlm = SimpleNamespace(
        get_image_features=lambda pixels: SimpleNamespace(
            pooler_output=pixels.mean(dim=(-1, -2))[:, None].expand(-1, 2, -1)
        ),
        get_input_embeddings=lambda: lambda ids: ids[..., None].float().expand(-1, -1, 3),
        language_model=SimpleNamespace(encoder=encode),
    )
    model = SimpleNamespace(vlm=vlm)
    images = torch.ones(2, 3, 3, 4, 4)
    mask = torch.ones(2, 3, dtype=torch.bool)
    ids = torch.tensor([[1, 2], [1, 2]])
    drop = torch.tensor([True, False])
    a = XVLAModel.forward_vlm(model, ids, images, mask, visual_drop_mask=drop)
    b = XVLAModel.forward_vlm(model, ids, images * 9, mask, visual_drop_mask=drop)
    torch.testing.assert_close(a["vlm_features"][0, 2:], b["vlm_features"][0, 2:])
    assert not torch.allclose(a["vlm_features"][1, 2:], b["vlm_features"][1, 2:])
    assert torch.count_nonzero(a["aux_visual_inputs"][0]) == 0


def test_training_visual_dropout_is_applied_before_slow_encoding() -> None:
    drop = torch.tensor([True, False])
    captured = []

    def slow_inputs(batch, *, visual_drop_mask):
        captured.append(visual_drop_mask)
        return {}, torch.zeros(2, 1), torch.zeros(2, dtype=torch.long)

    policy = SimpleNamespace(
        _sample_slow_dropout=lambda size, device: drop,
        _slow_inputs=slow_inputs,
        _make_fast_tokens=lambda batch, delays: torch.zeros(2, 3, 4),
        model=SimpleNamespace(predict_endpoint=lambda **kwargs: kwargs["particle"]),
    )
    particle = torch.zeros(2, 5, 20)
    OnlineFilteringXVLAPolicy._predict(policy, {}, particle, torch.ones(2), torch.zeros(2), torch.zeros(2))
    torch.testing.assert_close(captured[0], drop)


@pytest.mark.parametrize("delay", [5, 15, 20, 25])
def test_direct_simulator_packet_matches_latest_buffer_with_shared_noise(delay) -> None:
    def run(direct):
        seen = []

        def endpoint(**kw):
            seen.append((int(kw["old_delay"][0]), int(kw["new_delay"][0]), kw["particle"].clone()))
            return kw["particle"] * 0.2 + kw["new_fast_tokens"].mean(dim=1)[:, None] * 0.3

        policy = SimpleNamespace(
            config=SimpleNamespace(chunk_size=50, num_denoising_steps=10, training_stage="stage_b"),
            model=SimpleNamespace(
                dim_action=2,
                _get_target_dtype=lambda: torch.float32,
                predict_endpoint=endpoint,
                action_space=SimpleNamespace(postprocess=lambda x: x),
            ),
            _online_generation=0,
            _pending_online_packet=None,
            _slow_inputs=lambda batch: ({}, torch.zeros(1, 2), torch.zeros(1, dtype=torch.long)),
            _encode_runtime_fast=lambda batch: (torch.zeros(1, 1, 2), torch.zeros(1, dtype=torch.long)),
        )
        policy.begin_online_generation = lambda: OnlineFilteringXVLAPolicy.begin_online_generation(policy)
        policy._take_online_packet = lambda d, g: OnlineFilteringXVLAPolicy._take_online_packet(policy, d, g)
        elapsed = 0

        def packet(step, generation):
            return XVLAObservationPacket(
                tokens=torch.full((1, 1, 2), step / 25),
                domain_id=torch.zeros(1, dtype=torch.long),
                capture_step=step,
                generation=generation,
                executed_prefix=torch.arange(step * 2).reshape(1, step, 2).float() / 100,
            )

        def callback(query, count, generation):
            nonlocal elapsed
            target = query * delay // (count - 1)
            if target == elapsed:
                return None
            if direct:
                elapsed = target
                return packet(target, generation)
            while elapsed < target:
                elapsed += 1
                OnlineFilteringXVLAPolicy.push_fast_observation(policy, packet(elapsed, generation))
            return None

        output = OnlineFilteringXVLAPolicy.predict_action_chunk(
            policy,
            {},
            observation_callback=callback,
            noise=torch.linspace(-1, 1, 100).reshape(1, 50, 2),
        )
        return output, seen

    direct_output, direct_seen = run(True)
    buffered_output, buffered_seen = run(False)
    torch.testing.assert_close(direct_output, buffered_output, rtol=0, atol=0)
    for direct, buffered in zip(direct_seen, buffered_seen, strict=True):
        assert direct[:2] == buffered[:2]
        torch.testing.assert_close(direct[2], buffered[2], rtol=0, atol=0)
