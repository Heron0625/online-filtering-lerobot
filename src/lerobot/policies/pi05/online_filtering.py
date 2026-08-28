#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core components for observation-filtered PI0.5 action generation.

The components in this module deliberately do not load PI0.5 or DINOv3 weights. This
keeps timeline, masking, fast-token fusion, and posterior-update contracts testable on
CPU with synthetic features.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class OverlapTimeline:
    """Masks and indices for a batch of asynchronous chunk delays."""

    elapsed_mask: Tensor
    free_mask: Tensor
    next_action_index: Tensor


@dataclass(frozen=True)
class OnlineFilteringFlowInputs:
    """Stage-A noised action tensor, per-position time, and overlap masks."""

    noisy_actions: Tensor
    model_time: Tensor
    timeline: OverlapTimeline


@dataclass
class TimestampedFastTokenCache:
    """Minimal cache contract for observation-derived fast tokens."""

    timestamp: int | float | None = None
    tokens: Tensor | None = None

    def get(self, timestamp: int | float) -> Tensor | None:
        if self.timestamp != timestamp:
            return None
        return self.tokens

    def update(self, timestamp: int | float, tokens: Tensor) -> None:
        self.timestamp = timestamp
        self.tokens = tokens

    def clear(self) -> None:
        self.timestamp = None
        self.tokens = None


@dataclass(frozen=True)
class FastObservationPacket:
    """A causally timestamped, fully valid multi-camera fast condition."""

    tokens: Tensor
    capture_step: int
    capture_timestamp: float
    ready_timestamp: float
    proprio: Tensor
    camera_valid_mask: Tensor
    generation: int


class FastObservationBuffer:
    """Thread-safe bounded latest-ready buffer for asynchronous fast conditions."""

    def __init__(self, capacity: int = 4, max_age_s: float = 0.5) -> None:
        if capacity < 1 or max_age_s <= 0:
            raise ValueError("capacity and max_age_s must be positive")
        self.capacity = capacity
        self.max_age_s = max_age_s
        self._packets: deque[FastObservationPacket] = deque(maxlen=capacity)
        self._lock = Lock()

    def push(self, packet: FastObservationPacket) -> bool:
        """Store a packet only when every configured camera is valid."""
        if not bool(packet.camera_valid_mask.bool().all()):
            return False
        with self._lock:
            self._packets.append(packet)
        return True

    def latest_ready_after(
        self,
        version: int,
        *,
        generation: int,
        query_timestamp: float,
    ) -> FastObservationPacket | None:
        with self._lock:
            candidates = [
                packet
                for packet in self._packets
                if packet.generation == generation
                and packet.capture_step > version
                and packet.ready_timestamp <= query_timestamp
                and query_timestamp - packet.capture_timestamp <= self.max_age_s
                and bool(packet.camera_valid_mask.bool().all())
            ]
        return max(candidates, key=lambda packet: packet.capture_step, default=None)

    def clear(self) -> None:
        with self._lock:
            self._packets.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._packets)


class LatestOnlyAsyncProcessor[RequestT]:
    """Run one job and retain only the newest pending request while busy."""

    def __init__(
        self,
        process: Callable[[RequestT], FastObservationPacket],
        on_ready: Callable[[FastObservationPacket], object],
        on_error: Callable[[BaseException], object] | None = None,
    ) -> None:
        self._process = process
        self._on_ready = on_ready
        self._on_error = on_error
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="FastObservation")
        self._lock = Lock()
        self._running: Future[FastObservationPacket] | None = None
        self._pending: RequestT | None = None
        self._closed = False

    def submit(self, request: RequestT) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("async processor is closed")
            if self._running is not None and not self._running.done():
                self._pending = request
                return
            self._launch_locked(request)

    def _launch_locked(self, request: RequestT) -> None:
        self._running = self._executor.submit(self._process, request)
        self._running.add_done_callback(self._complete)

    def _complete(self, future: Future[FastObservationPacket]) -> None:
        try:
            self._on_ready(future.result())
        except BaseException as error:
            if self._on_error is not None:
                self._on_error(error)
        finally:
            with self._lock:
                self._running = None
                pending = self._pending
                self._pending = None
                if pending is not None and not self._closed:
                    self._launch_locked(pending)

    def discard_pending(self) -> None:
        with self._lock:
            self._pending = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._running is not None and not self._running.done()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._pending = None
        self._executor.shutdown(wait=True)


@dataclass(frozen=True)
class FastObservationRequest:
    """Raw observation metadata handed from the control loop to the encoder worker."""

    batch: dict[str, Tensor]
    capture_step: int
    capture_timestamp: float
    generation: int
    camera_valid_mask: Tensor


class OnlineFilteringRuntime:
    """Small local runtime joining chunk timing, fast encoding, and packet buffering.

    The policy object is intentionally duck-typed to keep this scheduling layer free of
    model imports. It must expose ``encode_fast_observation_packet`` and
    ``push_fast_observation``.
    """

    def __init__(self, policy: Any, *, merge_horizon: int = 4) -> None:
        self.policy = policy
        self.controller = AsynchronousChunkController(merge_horizon=merge_horizon)
        self._generation: int | None = None
        self._controller_generation: int | None = None
        self._errors: deque[BaseException] = deque(maxlen=1)
        self.fast_processor = LatestOnlyAsyncProcessor(
            self._encode,
            policy.push_fast_observation,
            on_error=self._errors.append,
        )

    def reset(self, initial_chunk: Tensor | None = None) -> None:
        """Reset action state and invalidate queued or in-flight fast observations."""
        self.fast_processor.discard_pending()
        self.controller.reset(initial_chunk)
        self.policy.begin_online_generation()
        self._generation = None
        self._controller_generation = None

    def pop_action(self) -> Tensor:
        return self.controller.pop_action()

    def _encode(self, request: FastObservationRequest) -> FastObservationPacket:
        return self.policy.encode_fast_observation_packet(
            request.batch,
            capture_step=request.capture_step,
            capture_timestamp=request.capture_timestamp,
            generation=request.generation,
            camera_valid_mask=request.camera_valid_mask,
        )

    def begin_chunk(self) -> int:
        """Start one generation in both the controller and policy namespaces."""
        controller_generation = self.controller.begin_inference()
        policy_generation = self.policy.begin_online_generation()
        # Only the policy token is embedded into packets and consumed by the model.
        self._generation = policy_generation
        self._controller_generation = controller_generation
        return policy_generation

    def submit_observation(
        self,
        batch: dict[str, Tensor],
        *,
        capture_step: int,
        capture_timestamp: float,
        camera_valid_mask: Tensor,
    ) -> None:
        if self._generation is None:
            raise RuntimeError("begin_chunk must be called before submitting observations")
        if not bool(camera_valid_mask.bool().all()):
            return
        self.fast_processor.submit(
            FastObservationRequest(
                batch=batch,
                capture_step=capture_step,
                capture_timestamp=capture_timestamp,
                generation=self._generation,
                camera_valid_mask=camera_valid_mask,
            )
        )

    def complete_chunk(self, chunk: Tensor) -> int | None:
        if self._generation is None:
            raise RuntimeError("no chunk generation is active")
        if self._controller_generation is None:
            raise RuntimeError("controller generation is missing")
        delay = self.controller.complete_inference(self._controller_generation, chunk)
        self._generation = None
        self._controller_generation = None
        self.fast_processor.discard_pending()
        return delay

    def raise_if_failed(self) -> None:
        if self._errors:
            raise RuntimeError("fast observation encoding failed") from self._errors.popleft()

    def close(self) -> None:
        self.fast_processor.close()


class AsynchronousChunkController:
    """Thread-safe action-queue contract for overlap-aligned chunk replacement."""

    def __init__(self, merge_horizon: int = 4) -> None:
        if merge_horizon < 0:
            raise ValueError("merge_horizon must be non-negative")
        self._lock = Lock()
        self.merge_horizon = merge_horizon
        self._queue: deque[Tensor] = deque()
        self._generation = 0
        self._pending_generation: int | None = None
        self._elapsed_during_inference = 0

    def reset(self, initial_chunk: Tensor | None = None) -> None:
        """Invalidate pending inference and optionally initialize the action queue."""
        with self._lock:
            self._generation += 1
            self._pending_generation = None
            self._elapsed_during_inference = 0
            self._queue.clear()
            if initial_chunk is not None:
                if initial_chunk.ndim != 2:
                    raise ValueError("initial_chunk must have shape (horizon, action_dim)")
                self._queue.extend(initial_chunk.unbind(0))

    def begin_inference(self) -> int:
        """Return a generation token used to reject stale completion callbacks."""
        with self._lock:
            self._generation += 1
            self._pending_generation = self._generation
            self._elapsed_during_inference = 0
            return self._generation

    def pop_action(self) -> Tensor:
        """Pop a queued action and advance delay when inference is pending."""
        with self._lock:
            if not self._queue:
                raise RuntimeError("action queue is empty")
            action = self._queue.popleft()
            if self._pending_generation is not None:
                self._elapsed_during_inference += 1
            return action

    def complete_inference(self, generation: int, new_chunk: Tensor) -> int | None:
        """Atomically replace the queue with the non-elapsed suffix of ``new_chunk``."""
        if new_chunk.ndim != 2:
            raise ValueError("new_chunk must have shape (horizon, action_dim)")
        with self._lock:
            if generation != self._pending_generation:
                return None
            delay = self._elapsed_during_inference
            if delay >= new_chunk.shape[0]:
                raise RuntimeError(
                    f"inference delay {delay} exhausted new chunk horizon {new_chunk.shape[0]}"
                )
            new_suffix = new_chunk[delay:]
            old_future = list(self._queue)
            merge_steps = min(self.merge_horizon, len(old_future), new_suffix.shape[0])
            merged: list[Tensor] = []
            for index in range(merge_steps):
                weight = (index + 1) / (merge_steps + 1)
                merged.append((1 - weight) * old_future[index] + weight * new_suffix[index])
            self._queue.clear()
            self._queue.extend(merged)
            self._queue.extend(new_suffix[merge_steps:].unbind(0))
            self._pending_generation = None
            self._elapsed_during_inference = 0
            return delay

    @property
    def queued_actions(self) -> int:
        """Return the number of actions currently available to the control loop."""
        with self._lock:
            return len(self._queue)


class DINOv3PatchEncoder(nn.Module):
    """Extract spatial patch tokens from a DINOv3-compatible backbone.

    Image normalization/resizing belongs to the policy processor. This module accepts
    ready-to-encode ``pixel_values`` and strips CLS/register tokens according to the
    backbone configuration.
    """

    def __init__(self, backbone: nn.Module, *, freeze: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze = freeze
        if freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @classmethod
    def from_pretrained(cls, model_id: str, *, freeze: bool = True) -> DINOv3PatchEncoder:
        """Load a DINOv3 backbone through Transformers without importing it at module load."""
        from transformers import DINOv3ViTModel

        return cls(DINOv3ViTModel.from_pretrained(model_id), freeze=freeze)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Encode ``(B, cameras, C, H, W)`` images into spatial patch tokens."""
        if pixel_values.ndim != 5:
            raise ValueError("pixel_values must have shape (batch, cameras, channels, height, width)")
        batch_size, cameras = pixel_values.shape[:2]
        flat_pixels = pixel_values.flatten(0, 1)
        if self.freeze:
            with torch.no_grad():
                output = self.backbone(pixel_values=flat_pixels)
        else:
            output = self.backbone(pixel_values=flat_pixels)
        hidden = output.last_hidden_state
        num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
        prefix_tokens = 1 + num_register_tokens
        if hidden.shape[1] <= prefix_tokens:
            raise ValueError("DINOv3 output does not contain any spatial patch tokens")
        patches = hidden[:, prefix_tokens:]
        return patches.unflatten(0, (batch_size, cameras))


def build_overlap_timeline(delays: Tensor, horizon: int, valid_mask: Tensor | None = None) -> OverlapTimeline:
    """Build 0-based elapsed/free masks for asynchronous action chunks.

    A delay ``d`` means action positions ``[0, d)`` elapsed while inference ran,
    and position ``d`` is the next action that may be executed.
    """
    if delays.ndim != 1:
        raise ValueError(f"delays must have shape (batch,), got {tuple(delays.shape)}")
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if delays.dtype == torch.bool or delays.is_floating_point():
        raise TypeError("delays must use an integer dtype")
    if torch.any(delays < 0) or torch.any(delays >= horizon):
        raise ValueError(f"all delays must satisfy 0 <= delay < {horizon}")

    positions = torch.arange(horizon, device=delays.device).unsqueeze(0)
    elapsed = positions < delays.unsqueeze(1)
    free = ~elapsed
    if valid_mask is not None:
        if valid_mask.shape != free.shape:
            raise ValueError(f"valid_mask must have shape {tuple(free.shape)}, got {tuple(valid_mask.shape)}")
        free = free & valid_mask.bool()
    return OverlapTimeline(elapsed_mask=elapsed, free_mask=free, next_action_index=delays)


def build_stage_a_flow_inputs(
    actions: Tensor,
    noise: Tensor,
    time: Tensor,
    delays: Tensor,
    valid_mask: Tensor | None = None,
) -> OnlineFilteringFlowInputs:
    """Keep elapsed actions clean and noise only the valid free suffix.

    This follows PI0.5's convention ``x_t = t * noise + (1 - t) * action``.
    """
    if actions.shape != noise.shape or actions.ndim != 3:
        raise ValueError("actions and noise must share shape (batch, horizon, action_dim)")
    batch_size, horizon = actions.shape[:2]
    if time.shape != (batch_size,):
        raise ValueError(f"time must have shape ({batch_size},), got {tuple(time.shape)}")
    timeline = build_overlap_timeline(delays, horizon, valid_mask)
    model_time = time[:, None].expand(batch_size, horizon)
    model_time = torch.where(timeline.free_mask, model_time, torch.zeros_like(model_time))
    noisy_actions = model_time.unsqueeze(-1) * noise + (1 - model_time.unsqueeze(-1)) * actions
    return OnlineFilteringFlowInputs(noisy_actions=noisy_actions, model_time=model_time, timeline=timeline)


def shared_base_teacher_pair(
    base: Tensor,
    time: Tensor | float,
    old_flow_map,
    new_flow_map,
) -> tuple[Tensor, Tensor]:
    """Generate Stage-C teacher particles with identical base noise and flow time."""
    return old_flow_map(base, time), new_flow_map(base, time)


def masked_suffix_mean(losses: Tensor, free_mask: Tensor) -> Tensor:
    """Reduce per-element losses over valid free action positions only."""
    if losses.ndim < 2:
        raise ValueError("losses must have batch and horizon dimensions")
    if free_mask.shape != losses.shape[:2]:
        raise ValueError(
            f"free_mask must match the first two loss dimensions {tuple(losses.shape[:2])}, "
            f"got {tuple(free_mask.shape)}"
        )
    expanded = free_mask
    while expanded.ndim < losses.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(losses)
    denominator = expanded.sum()
    if denominator == 0:
        raise ValueError("free_mask selects no valid loss elements")
    return (losses * expanded).sum() / denominator


class FastTokenResampler(nn.Module):
    """Compress multi-camera initial/current DINO patch features into fast tokens."""

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        num_fast_tokens: int = 16,
        num_heads: int = 8,
        num_cameras: int = 2,
        state_dim: int = 0,
    ) -> None:
        super().__init__()
        if output_dim % num_heads != 0:
            raise ValueError("output_dim must be divisible by num_heads")
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_fast_tokens = num_fast_tokens
        self.num_cameras = num_cameras

        self.feature_proj = nn.Linear(3 * feature_dim, output_dim)
        self.camera_embedding = nn.Parameter(torch.zeros(num_cameras, output_dim))
        self.queries = nn.Parameter(torch.randn(num_fast_tokens, output_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(output_dim, num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(output_dim)
        self.state_proj = (
            nn.Sequential(
                nn.Linear(state_dim, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
            if state_dim > 0
            else None
        )
        self.delay_proj = nn.Sequential(nn.Linear(1, output_dim), nn.SiLU(), nn.Linear(output_dim, output_dim))

    def forward(
        self,
        initial_features: Tensor,
        current_features: Tensor,
        *,
        delays: Tensor,
        states: Tensor | None = None,
        patch_mask: Tensor | None = None,
    ) -> Tensor:
        """Return fast tokens of shape ``(B, num_fast_tokens, output_dim)``.

        Args:
            initial_features: ``(B, cameras, patches, feature_dim)``.
            current_features: Same shape as ``initial_features``.
            delays: Integer or floating tensor with shape ``(B,)``.
            states: Optional latest proprioception with shape ``(B, state_dim)``.
            patch_mask: Optional valid-token mask ``(B, cameras, patches)``.
        """
        if initial_features.shape != current_features.shape:
            raise ValueError("initial_features and current_features must have identical shapes")
        if initial_features.ndim != 4:
            raise ValueError("DINO features must have shape (batch, cameras, patches, feature_dim)")
        batch_size, cameras, patches, feature_dim = initial_features.shape
        if cameras != self.num_cameras or feature_dim != self.feature_dim:
            raise ValueError(
                f"expected cameras={self.num_cameras}, feature_dim={self.feature_dim}; "
                f"got cameras={cameras}, feature_dim={feature_dim}"
            )
        if delays.shape != (batch_size,):
            raise ValueError(f"delays must have shape ({batch_size},), got {tuple(delays.shape)}")

        residual_features = torch.cat(
            [current_features, initial_features, current_features - initial_features], dim=-1
        )
        tokens = self.feature_proj(residual_features)
        tokens = tokens + self.camera_embedding[None, :, None, :].to(tokens.dtype)
        tokens = tokens.reshape(batch_size, cameras * patches, self.output_dim)

        key_padding_mask = None
        if patch_mask is not None:
            if patch_mask.shape != (batch_size, cameras, patches):
                raise ValueError(
                    f"patch_mask must have shape {(batch_size, cameras, patches)}, got {tuple(patch_mask.shape)}"
                )
            key_padding_mask = ~patch_mask.reshape(batch_size, cameras * patches).bool()

        queries = self.queries[None].expand(batch_size, -1, -1).to(tokens.dtype)
        delay_condition = self.delay_proj(delays.to(tokens.dtype).unsqueeze(-1))
        queries = queries + delay_condition.unsqueeze(1)
        if self.state_proj is not None:
            if states is None:
                raise ValueError("states are required when state_dim > 0")
            state_token = self.state_proj(states).unsqueeze(1)
            tokens = torch.cat([tokens, state_token], dim=1)
            if key_padding_mask is not None:
                state_valid = torch.zeros((batch_size, 1), dtype=torch.bool, device=tokens.device)
                key_padding_mask = torch.cat([key_padding_mask, state_valid], dim=1)

        output, _ = self.cross_attention(queries, tokens, tokens, key_padding_mask=key_padding_mask)
        return self.output_norm(queries + output)


class FastObservationAdapter(nn.Module):
    """Zero-initialized gated cross-attention for action-expert hidden states."""

    def __init__(self, hidden_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.action_norm = nn.LayerNorm(hidden_dim)
        self.fast_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_mlp = nn.Sequential(nn.Linear(3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        action_hidden: Tensor,
        fast_tokens: Tensor,
        *,
        delays: Tensor,
        flow_times: Tensor,
        observation_ages: Tensor | None = None,
    ) -> Tensor:
        batch_size, horizon, _ = action_hidden.shape
        if fast_tokens.shape[0] != batch_size:
            raise ValueError("action_hidden and fast_tokens batch sizes must match")
        if delays.shape != (batch_size,) or flow_times.shape != (batch_size,):
            raise ValueError("delays and flow_times must have shape (batch,)")
        if observation_ages is None:
            observation_ages = torch.zeros_like(flow_times)

        correction, _ = self.cross_attention(
            self.action_norm(action_hidden), self.fast_norm(fast_tokens), self.fast_norm(fast_tokens)
        )
        positions = torch.arange(horizon, device=action_hidden.device).unsqueeze(0)
        relative_position = (positions - delays.unsqueeze(1)).clamp(min=0).to(action_hidden.dtype)
        relative_position = relative_position / (horizon - delays).clamp(min=1).unsqueeze(1).to(action_hidden.dtype)
        gate_inputs = torch.stack(
            [
                flow_times.to(action_hidden.dtype).unsqueeze(1).expand(-1, horizon),
                relative_position,
                observation_ages.to(action_hidden.dtype).unsqueeze(1).expand(-1, horizon),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_mlp(gate_inputs))
        correction = self.output_proj(correction) * gate
        return action_hidden + correction


class ConditionTokenResampler(nn.Module):
    """Compress cached slow-prefix embeddings into a few bridge condition tokens."""

    def __init__(self, input_dim: int, output_dim: int, num_tokens: int = 4, num_heads: int = 8) -> None:
        super().__init__()
        if output_dim % num_heads != 0:
            raise ValueError("output_dim must be divisible by num_heads")
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.queries = nn.Parameter(torch.randn(num_tokens, output_dim) * 0.02)
        self.attention = nn.MultiheadAttention(output_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, condition: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if condition.ndim != 3:
            raise ValueError("condition must have shape (batch, tokens, dim)")
        values = self.input_proj(condition)
        queries = self.queries.unsqueeze(0).expand(condition.shape[0], -1, -1).to(values.dtype)
        key_padding_mask = None if valid_mask is None else ~valid_mask.bool()
        output, _ = self.attention(queries, values, values, key_padding_mask=key_padding_mask)
        return self.norm(queries + output)


class PosteriorResidualUpdater(nn.Module):
    """One-step student that predicts a masked posterior-particle residual."""

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        max_horizon: int = 50,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.max_horizon = max_horizon
        self.action_in = nn.Linear(action_dim, hidden_dim)
        self.condition_in = nn.Linear(5, hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(max_horizon, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.context_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        old_particle: Tensor,
        *,
        old_delays: Tensor,
        new_delays: Tensor,
        flow_times: Tensor,
        observation_ages: Tensor | None = None,
        condition_tokens: Tensor | None = None,
    ) -> Tensor:
        batch_size, horizon, action_dim = old_particle.shape
        if horizon > self.max_horizon or action_dim != self.action_dim:
            raise ValueError("old_particle shape is incompatible with updater configuration")
        if old_delays.shape != (batch_size,) or new_delays.shape != (batch_size,):
            raise ValueError("old_delays and new_delays must have shape (batch,)")
        if observation_ages is None:
            observation_ages = torch.zeros_like(flow_times)
        gap = new_delays - old_delays
        condition = torch.stack(
            [
                old_delays.to(old_particle.dtype),
                new_delays.to(old_particle.dtype),
                gap.to(old_particle.dtype),
                flow_times.to(old_particle.dtype),
                observation_ages.to(old_particle.dtype),
            ],
            dim=-1,
        )
        hidden = self.action_in(old_particle) + self.position_embedding[None, :horizon]
        hidden = hidden + self.condition_in(condition).unsqueeze(1)
        hidden = self.transformer(hidden)
        if condition_tokens is not None:
            if condition_tokens.ndim != 3 or condition_tokens.shape[0] != batch_size:
                raise ValueError("condition_tokens must have shape (batch, tokens, hidden_dim)")
            context, _ = self.context_attention(hidden, condition_tokens, condition_tokens)
            hidden = self.context_norm(hidden + context)
        residual = self.output(hidden)
        return old_particle + residual
