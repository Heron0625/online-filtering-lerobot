"""Strict native checkpoint loading and isolated preprocessing for fast observations."""

from __future__ import annotations

import copy
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


def load_runner(cfg):
    # Read-only reuse of the tested loader; no imports of ROS or control code.
    legacy = Path(cfg["online"]["legacy_infer_dir"])
    sys.path.insert(0, str(legacy))
    from policy_worker import PolicyRunner

    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    runner = PolicyRunner(cfg)
    if runner.policy.config.type != "online_filtering_pi05":
        raise ValueError("This server requires online_filtering_pi05")
    if runner.policy.config.training_stage != "online_flow":
        raise ValueError("This runtime is validated for online_flow checkpoints")
    if runner.policy.config.use_relative_actions:
        raise ValueError("Only absolute state14/action16 checkpoints are supported")
    normalizers = [step for step in runner.pre.steps if isinstance(step, NormalizerProcessorStep)]
    if len(normalizers) != 1:
        raise ValueError("Expected exactly one checkpoint normalizer")
    runner.fast_normalizer = copy.deepcopy(normalizers[0]).to(device="cpu")
    runner.fast_stream = runner.torch.cuda.Stream(device=runner.policy.config.device)
    runner.torch.cuda.synchronize()
    return runner


def raw_batch(runner, data):
    torch = runner.torch
    state = np.asarray(data["state"], dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("Expected finite state14")
    batch = {"observation.state": torch.from_numpy(state.copy()), "task": runner.cfg["model"]["task"]}
    for name, camera in runner.cfg["cameras"].items():
        rgb = data[name]
        if rgb.dtype != np.uint8 or rgb.shape != (camera["height"], camera["width"], 3):
            raise ValueError(f"Invalid RGB observation: {name}")
        batch[f"observation.images.{name}"] = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
    return batch


def fast_packet(runner, data, generation):
    torch = runner.torch
    step = int(data["capture_step"])
    config = runner.policy.config
    if not 1 <= step <= min(config.max_inference_delay, config.chunk_size - 1):
        raise ValueError("Fast capture_step outside checkpoint delay range")
    prefix = np.asarray(data["executed_prefix"], dtype=np.float32)
    if prefix.shape != (step, 16) or not np.isfinite(prefix).all():
        raise ValueError("executed_prefix must contain actual finite action16 commands")
    batch = raw_batch(runner, data)
    batch.pop("task")
    # Independent CPU normalizer: never run the stateful slow tokenizer concurrently.
    batch = {key: value.unsqueeze(0) for key, value in batch.items()}
    batch = runner.fast_normalizer._normalize_observation(batch, inverse=False)
    prefix = runner.fast_normalizer._normalize_action(torch.from_numpy(prefix).unsqueeze(0), inverse=False)
    with torch.inference_mode(), torch.cuda.stream(runner.fast_stream):
        batch = {key: value.to(config.device) for key, value in batch.items()}
        packet = runner.policy.encode_fast_observation_packet(
            batch,
            capture_step=step,
            capture_timestamp=float(data["capture_server_s"]),
            generation=generation,
            executed_prefix=prefix.to(config.device),
        )
        # Publish only after GPU completion. Synchronizes this stream, not the expert stream.
        event = torch.cuda.Event()
        event.record(runner.fast_stream)
        event.synchronize()
    return replace(packet, ready_timestamp=time.monotonic())
