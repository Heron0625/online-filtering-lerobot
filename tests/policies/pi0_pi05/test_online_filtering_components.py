#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

from threading import Event

import pytest
import torch
from torch import nn

from lerobot.policies.pi05.gaussian_reanchoring import (
    AffineGaussianFlow,
    linear_gaussian_update,
    naive_condition_switch,
    shared_base_reanchor,
)
from lerobot.policies.pi05.online_filtering import (
    AsynchronousChunkController,
    DINOv3PatchEncoder,
    FastObservationAdapter,
    FastObservationBuffer,
    FastObservationPacket,
    FastTokenResampler,
    LatestOnlyAsyncProcessor,
    PosteriorResidualUpdater,
    TimestampedFastTokenCache,
    build_overlap_timeline,
    build_stage_a_flow_inputs,
    masked_suffix_mean,
    shared_base_teacher_pair,
)


class _MockDINOv3(nn.Module):
    def __init__(self, hidden_dim=12, patches=6, register_tokens=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patches = patches
        self.config = type("Config", (), {"num_register_tokens": register_tokens})()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, *, pixel_values):
        batch = pixel_values.shape[0]
        token_count = 1 + self.config.num_register_tokens + self.patches
        values = torch.arange(token_count * self.hidden_dim, device=pixel_values.device).float()
        hidden = values.reshape(1, token_count, self.hidden_dim).expand(batch, -1, -1) * self.scale
        return type("Output", (), {"last_hidden_state": hidden})()


def _empirical_covariance(samples: torch.Tensor) -> torch.Tensor:
    centered = samples - samples.mean(dim=0)
    return centered.mT @ centered / (samples.shape[0] - 1)


def test_overlap_timeline_ten_step_delay_executes_eleventh_action():
    timeline = build_overlap_timeline(torch.tensor([10]), horizon=50)

    assert timeline.elapsed_mask[0, :10].all()
    assert not timeline.elapsed_mask[0, 10:].any()
    assert not timeline.free_mask[0, :10].any()
    assert timeline.free_mask[0, 10:].all()
    assert timeline.next_action_index.item() == 10


@pytest.mark.parametrize("delay", [0, 1, 10, 49])
def test_overlap_masks_are_complementary(delay):
    timeline = build_overlap_timeline(torch.tensor([delay]), horizon=50)
    assert torch.equal(timeline.elapsed_mask ^ timeline.free_mask, torch.ones(1, 50, dtype=torch.bool))
    assert not (timeline.elapsed_mask & timeline.free_mask).any()


@pytest.mark.parametrize("delay", [-1, 50])
def test_overlap_timeline_rejects_invalid_delay(delay):
    with pytest.raises(ValueError):
        build_overlap_timeline(torch.tensor([delay]), horizon=50)


def test_valid_mask_removes_padded_suffix_positions():
    valid = torch.zeros(1, 6, dtype=torch.bool)
    valid[:, :4] = True
    timeline = build_overlap_timeline(torch.tensor([2]), horizon=6, valid_mask=valid)
    assert torch.equal(timeline.free_mask, torch.tensor([[False, False, True, True, False, False]]))


def test_masked_suffix_mean_matches_manual_reduction():
    losses = torch.tensor([[[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]]])
    free = torch.tensor([[False, True, True]])
    assert masked_suffix_mean(losses, free) == torch.tensor(8.0)


def test_stage_a_inputs_keep_elapsed_prefix_clean_and_noise_suffix():
    actions = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    noise = actions + 10
    result = build_stage_a_flow_inputs(actions, noise, torch.tensor([0.25]), torch.tensor([2]))
    assert torch.equal(result.noisy_actions[:, :2], actions[:, :2])
    assert torch.allclose(result.noisy_actions[:, 2:], 0.25 * noise[:, 2:] + 0.75 * actions[:, 2:])
    assert torch.equal(result.model_time, torch.tensor([[0.0, 0.0, 0.25, 0.25]]))


def test_shared_base_teacher_pair_passes_identical_base_and_time():
    calls = []

    def old_map(base, time):
        calls.append((base.data_ptr(), time))
        return base + 1

    def new_map(base, time):
        calls.append((base.data_ptr(), time))
        return base - 2

    base = torch.randn(2, 3)
    old, new = shared_base_teacher_pair(base, 0.4, old_map, new_map)
    assert torch.equal(old, base + 1)
    assert torch.equal(new, base - 2)
    assert calls == [(base.data_ptr(), 0.4), (base.data_ptr(), 0.4)]


def test_linear_gaussian_update_matches_scalar_reference():
    mean = torch.tensor([0.0], dtype=torch.float64)
    covariance = torch.tensor([[2.0]], dtype=torch.float64)
    observation = torch.tensor([1.0], dtype=torch.float64)
    matrix = torch.tensor([[1.0]], dtype=torch.float64)
    noise = torch.tensor([[3.0]], dtype=torch.float64)

    posterior_mean, posterior_covariance = linear_gaussian_update(
        mean, covariance, observation, matrix, noise
    )

    assert torch.allclose(posterior_covariance, torch.tensor([[1.2]], dtype=torch.float64))
    assert torch.allclose(posterior_mean, torch.tensor([0.4], dtype=torch.float64))


def test_affine_gaussian_flow_inverse_recovers_shared_base():
    flow = AffineGaussianFlow(
        mean=torch.tensor([1.0, -0.5], dtype=torch.float64),
        covariance=torch.tensor([[1.5, 0.2], [0.2, 0.8]], dtype=torch.float64),
    )
    base = torch.randn(64, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    value = flow.map(base, 0.63)
    recovered = flow.inverse(value, 0.63)
    assert torch.allclose(recovered, base, atol=1e-10, rtol=1e-10)


def test_shared_base_reanchoring_matches_new_gaussian_moments():
    generator = torch.Generator().manual_seed(7)
    base = torch.randn(10_000, 2, dtype=torch.float64, generator=generator)
    new_flow = AffineGaussianFlow(
        mean=torch.tensor([1.25, -0.75], dtype=torch.float64),
        covariance=torch.tensor([[1.4, 0.35], [0.35, 0.7]], dtype=torch.float64),
    )

    samples = shared_base_reanchor(base, new_flow, 1.0)

    assert torch.allclose(samples.mean(0), new_flow.mean, atol=4e-2, rtol=0)
    assert torch.allclose(_empirical_covariance(samples), new_flow.covariance, atol=6e-2, rtol=0)


def test_naive_condition_switch_is_not_new_posterior_but_reanchoring_is():
    generator = torch.Generator().manual_seed(11)
    base = torch.randn(5_000, 2, dtype=torch.float64, generator=generator)
    old_flow = AffineGaussianFlow(
        mean=torch.tensor([-1.5, 0.5], dtype=torch.float64),
        covariance=torch.tensor([[0.6, 0.1], [0.1, 1.2]], dtype=torch.float64),
    )
    new_flow = AffineGaussianFlow(
        mean=torch.tensor([1.5, -1.0], dtype=torch.float64),
        covariance=torch.tensor([[1.3, -0.25], [-0.25, 0.5]], dtype=torch.float64),
    )

    naive = naive_condition_switch(base, old_flow, new_flow, 0.55, steps_before=32, steps_after=32)
    reanchored = shared_base_reanchor(base, new_flow, 1.0)

    naive_mean_error = torch.linalg.vector_norm(naive.mean(0) - new_flow.mean)
    reanchored_mean_error = torch.linalg.vector_norm(reanchored.mean(0) - new_flow.mean)
    naive_cov_error = torch.linalg.matrix_norm(_empirical_covariance(naive) - new_flow.covariance)
    reanchored_cov_error = torch.linalg.matrix_norm(_empirical_covariance(reanchored) - new_flow.covariance)

    assert naive_mean_error > 0.1
    assert naive_cov_error > 0.05
    assert reanchored_mean_error < 0.05
    assert reanchored_cov_error < 0.08


def test_shared_base_reanchoring_remains_correct_after_sequential_observations():
    dtype = torch.float64
    mean = torch.tensor([0.0, 0.0], dtype=dtype)
    covariance = torch.tensor([[1.5, 0.2], [0.2, 1.0]], dtype=dtype)
    observations = [
        (torch.tensor([0.8], dtype=dtype), torch.tensor([[1.0, 0.0]], dtype=dtype)),
        (torch.tensor([-0.4], dtype=dtype), torch.tensor([[0.0, 1.0]], dtype=dtype)),
        (torch.tensor([0.1], dtype=dtype), torch.tensor([[1.0, -1.0]], dtype=dtype)),
    ]
    noise = torch.tensor([[0.25]], dtype=dtype)
    base = torch.randn(12_000, 2, dtype=dtype, generator=torch.Generator().manual_seed(19))

    for observation, matrix in observations:
        mean, covariance = linear_gaussian_update(mean, covariance, observation, matrix, noise)
        samples = shared_base_reanchor(base, AffineGaussianFlow(mean, covariance), 1.0)
        assert torch.allclose(samples.mean(0), mean, atol=3e-2, rtol=0)
        assert torch.allclose(_empirical_covariance(samples), covariance, atol=4e-2, rtol=0)


def test_affine_flow_euler_integration_matches_closed_form():
    flow = AffineGaussianFlow(
        mean=torch.tensor([0.7, -0.2], dtype=torch.float64),
        covariance=torch.tensor([[1.7, 0.3], [0.3, 0.9]], dtype=torch.float64),
    )
    base = torch.randn(128, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    exact = flow.map(base, 1.0)
    # For this affine interpolation, recovering the base variable inside the exact
    # velocity field makes explicit Euler exact up to floating-point roundoff.
    coarse_error = (flow.integrate(base, 1.0, steps=2) - exact).abs().max()
    fine_error = (flow.integrate(base, 1.0, steps=16) - exact).abs().max()
    assert coarse_error < 1e-12
    assert fine_error < 1e-12


def test_fast_resampler_shapes_and_zero_visual_residual():
    module = FastTokenResampler(
        feature_dim=12,
        output_dim=32,
        num_fast_tokens=8,
        num_heads=4,
        num_cameras=2,
        state_dim=5,
    )
    initial = torch.randn(2, 2, 6, 12)
    current = initial.clone()
    output = module(initial, current, delays=torch.tensor([0, 3]), states=torch.randn(2, 5))
    assert output.shape == (2, 8, 32)
    concatenated = torch.cat([current, initial, current - initial], dim=-1)
    assert torch.count_nonzero(concatenated[..., -12:]) == 0


def test_dinov3_patch_encoder_strips_cls_and_register_tokens():
    backbone = _MockDINOv3(hidden_dim=12, patches=6, register_tokens=4)
    encoder = DINOv3PatchEncoder(backbone, freeze=True)
    patches = encoder(torch.randn(2, 2, 3, 16, 16))
    assert patches.shape == (2, 2, 6, 12)
    assert not backbone.scale.requires_grad
    encoder.train()
    assert not backbone.training


def test_fast_resampler_honors_patch_mask():
    torch.manual_seed(0)
    module = FastTokenResampler(4, 16, num_fast_tokens=4, num_heads=4, num_cameras=2)
    initial = torch.randn(1, 2, 3, 4)
    current = torch.randn(1, 2, 3, 4)
    mask = torch.tensor([[[True, True, False], [True, False, False]]])
    baseline = module(initial, current, delays=torch.tensor([1]), patch_mask=mask)

    modified_initial = initial.clone()
    modified_current = current.clone()
    modified_initial[~mask] = 10_000
    modified_current[~mask] = -10_000
    modified = module(modified_initial, modified_current, delays=torch.tensor([1]), patch_mask=mask)
    assert torch.allclose(baseline, modified, atol=1e-5, rtol=1e-5)


def test_timestamped_fast_token_cache_reuses_only_matching_frame():
    cache = TimestampedFastTokenCache()
    tokens = torch.randn(1, 4, 8)
    assert cache.get(10) is None
    cache.update(10, tokens)
    assert cache.get(10) is tokens
    assert cache.get(11) is None
    cache.clear()
    assert cache.get(10) is None


def _packet(step: int, *, ready: float, capture: float, generation: int = 1, valid=True):
    return FastObservationPacket(
        tokens=torch.full((1, 2, 3), float(step)),
        capture_step=step,
        capture_timestamp=capture,
        ready_timestamp=ready,
        proprio=torch.zeros(1, 7),
        camera_valid_mask=torch.tensor([valid, valid]),
        generation=generation,
    )


def test_fast_observation_buffer_is_bounded_latest_ready_causal_and_generation_scoped():
    buffer = FastObservationBuffer(capacity=2, max_age_s=0.5)
    assert buffer.push(_packet(1, ready=1.1, capture=1.0))
    assert buffer.push(_packet(2, ready=1.3, capture=1.2))
    assert buffer.push(_packet(3, ready=1.5, capture=1.4))
    assert len(buffer) == 2
    assert buffer.latest_ready_after(0, generation=1, query_timestamp=1.4).capture_step == 2
    assert buffer.latest_ready_after(2, generation=1, query_timestamp=1.4) is None
    assert buffer.latest_ready_after(0, generation=2, query_timestamp=1.6) is None
    assert buffer.latest_ready_after(0, generation=1, query_timestamp=2.0) is None
    assert not buffer.push(_packet(4, ready=2.0, capture=2.0, valid=False))


def test_latest_only_processor_keeps_newest_pending_request():
    release = Event()
    completed = Event()
    processed = []
    ready = []

    def process(step):
        processed.append(step)
        if step == 1:
            assert release.wait(timeout=2)
        return _packet(step, ready=float(step), capture=float(step))

    def on_ready(packet):
        ready.append(packet.capture_step)
        if packet.capture_step == 3:
            completed.set()

    processor = LatestOnlyAsyncProcessor(process, on_ready)
    processor.submit(1)
    processor.submit(2)
    processor.submit(3)
    release.set()
    assert completed.wait(timeout=2)
    processor.close()
    assert processed == [1, 3]
    assert ready == [1, 3]


def test_latest_only_processor_reports_error_and_continues_with_pending():
    release = Event()
    completed = Event()
    errors = []

    def process(step):
        if step == 1:
            assert release.wait(timeout=2)
            raise ValueError("bad frame")
        return _packet(step, ready=float(step), capture=float(step))

    processor = LatestOnlyAsyncProcessor(
        process,
        lambda packet: completed.set(),
        on_error=errors.append,
    )
    processor.submit(1)
    processor.submit(2)
    release.set()
    assert completed.wait(timeout=2)
    processor.close()
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)


def test_async_controller_measures_delay_and_linearly_merges_future_actions():
    controller = AsynchronousChunkController()
    old_chunk = torch.arange(50, dtype=torch.float32).unsqueeze(-1)
    new_chunk = (100 + torch.arange(50, dtype=torch.float32)).unsqueeze(-1)
    controller.reset(old_chunk)
    generation = controller.begin_inference()

    executed_while_inference_runs = torch.stack([controller.pop_action() for _ in range(10)])
    delay = controller.complete_inference(generation, new_chunk)
    first_new_action = controller.pop_action()

    assert torch.equal(executed_while_inference_runs.squeeze(-1), torch.arange(10, dtype=torch.float32))
    assert delay == 10
    assert first_new_action.item() == 30
    assert controller.queued_actions == 39


def test_async_controller_rejects_stale_result_and_reset_invalidates_pending_work():
    controller = AsynchronousChunkController()
    controller.reset(torch.zeros(5, 2))
    stale_generation = controller.begin_inference()
    current_generation = controller.begin_inference()
    assert controller.complete_inference(stale_generation, torch.ones(5, 2)) is None
    assert controller.complete_inference(current_generation, torch.ones(5, 2)) == 0

    pending = controller.begin_inference()
    controller.reset(torch.full((3, 2), 2.0))
    assert controller.complete_inference(pending, torch.ones(5, 2)) is None
    assert torch.equal(controller.pop_action(), torch.full((2,), 2.0))


def test_zero_initialized_fast_adapter_is_identity():
    adapter = FastObservationAdapter(hidden_dim=32, num_heads=4)
    actions = torch.randn(2, 10, 32)
    fast = torch.randn(2, 8, 32)
    output = adapter(actions, fast, delays=torch.tensor([3, 5]), flow_times=torch.tensor([0.2, 0.8]))
    assert torch.equal(output, actions)


def test_fast_adapter_can_update_the_complete_particle():
    torch.manual_seed(0)
    adapter = FastObservationAdapter(hidden_dim=16, num_heads=4)
    torch.nn.init.normal_(adapter.output_proj.weight)
    actions = torch.randn(1, 8, 16)
    output = adapter(actions, torch.randn(1, 4, 16), delays=torch.tensor([3]), flow_times=torch.tensor([0.5]))
    assert not torch.equal(output[:, :3], actions[:, :3])
    assert not torch.equal(output[:, 3:], actions[:, 3:])


def test_zero_initialized_posterior_updater_is_identity():
    updater = PosteriorResidualUpdater(action_dim=7, hidden_dim=32, num_heads=4, num_layers=1)
    particle = torch.randn(2, 12, 7)
    old_delays = torch.tensor([2, 5])
    output = updater(
        particle,
        old_delays=old_delays,
        new_delays=old_delays + 2,
        flow_times=torch.tensor([0.3, 0.7]),
    )
    assert torch.equal(output, particle)


def test_posterior_updater_can_update_the_complete_particle():
    torch.manual_seed(0)
    updater = PosteriorResidualUpdater(action_dim=3, hidden_dim=16, num_heads=4, num_layers=1)
    torch.nn.init.normal_(updater.output.weight)
    particle = torch.randn(1, 8, 3)
    output = updater(
        particle,
        old_delays=torch.tensor([2]),
        new_delays=torch.tensor([5]),
        flow_times=torch.tensor([0.4]),
    )
    assert not torch.equal(output, particle)
    assert not torch.equal(output[:, :2], particle[:, :2])


def test_posterior_updater_learns_synthetic_full_particle_transport():
    torch.manual_seed(4)
    updater = PosteriorResidualUpdater(action_dim=2, hidden_dim=16, num_heads=4, num_layers=1)
    optimizer = torch.optim.Adam(updater.parameters(), lr=3e-3)
    old = torch.randn(8, 6, 2)
    delays = torch.full((8,), 1, dtype=torch.long)
    target = old + 0.5

    def suffix_loss():
        prediction = updater(
            old,
            old_delays=delays,
            new_delays=delays + 2,
            flow_times=torch.full((8,), 0.4),
        )
        return (prediction - target).square().mean()

    initial = suffix_loss().item()
    for _ in range(40):
        optimizer.zero_grad()
        loss = suffix_loss()
        loss.backward()
        optimizer.step()
    assert suffix_loss().item() < 0.25 * initial
