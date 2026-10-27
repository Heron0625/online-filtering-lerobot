#!/usr/bin/env python
"""Deterministic LIBERO evaluation with simulated inference delay."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from lerobot.configs import PreTrainedConfig
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.lerobot_types import TransitionKey
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.online_filtering_pi05.modeling_online_filtering_pi05 import (
    OnlineFilteringPI05Policy,
)
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed


def normalize_executed_command(normalizer, sent: torch.Tensor) -> torch.Tensor:
    """Convert the final environment command, not the pre-postprocessing plan, to model units."""
    return normalizer({TransitionKey.ACTION: sent})[TransitionKey.ACTION]


def parse_args() -> argparse.Namespace:
    """Parse deterministic delayed-evaluation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--suite", default="libero_spatial", help="Suite name, or comma-separated suite names"
    )
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-ids", default=None, help="Comma-separated task IDs; overrides --task-id")
    parser.add_argument("--all-tasks", action="store_true", help="Run every task in each requested suite")
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Episodes per task; 50 covers the complete standard LIBERO init-state set.",
    )
    parser.add_argument("--episode-start", type=int, default=0, help="First LIBERO init-state index")
    parser.add_argument("--delay", type=int, default=5)
    parser.add_argument("--delay-min", type=int, default=None, help="Random delay lower bound (inclusive)")
    parser.add_argument("--delay-max", type=int, default=None, help="Random delay upper bound (inclusive)")
    parser.add_argument(
        "--delay-seed",
        type=int,
        default=None,
        help="Base seed for stateless per-(suite, task, episode, chunk) delay sampling",
    )
    parser.add_argument("--dynamic-delay", action="store_true", help="Resample delay at every chunk switch")
    parser.add_argument(
        "--delay-sequence",
        default=None,
        help="Diagnostic comma-separated delays, consumed in order for every replan",
    )
    parser.add_argument(
        "--switch-step-min", type=int, default=1, help="Deprecated; ignored by continuous_online_v1"
    )
    parser.add_argument(
        "--switch-step-max", type=int, default=None, help="Deprecated; ignored by continuous_online_v1"
    )
    parser.add_argument(
        "--switch-seed", type=int, default=None, help="Deprecated; ignored by continuous_online_v1"
    )
    parser.add_argument("--replan-interval", type=int, default=10)
    parser.add_argument(
        "--allow-queue-underrun",
        action="store_true",
        help=(
            "Allow short-horizon baselines to exhaust their action chunk while inference is pending. "
            "For relative control, the controller sends zero motion while preserving the latest "
            "gripper command; for absolute control, it repeats the latest target pose."
        ),
    )
    parser.add_argument(
        "--control-fps",
        type=int,
        default=20,
        help="LIBERO/MuJoCo control rate; standard PI0.5 LIBERO evaluation uses 20 Hz",
    )
    parser.add_argument(
        "--control-mode",
        choices=("relative", "absolute"),
        default=None,
        help="Defaults to absolute for XVLA and relative for other policies.",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help="Optional LIBERO horizon override; the official XVLA protocol uses 800.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--disable-online-fast",
        action="store_true",
        help="Diagnostic only: use the checkpoint's cached slow/expert path and ignore fast adapters",
    )
    parser.add_argument(
        "--observation-age-seconds",
        type=float,
        default=None,
        help="Diagnostic override for the fast-condition age; use 0 for a synchronous clock",
    )
    parser.add_argument(
        "--disable-env-preprocessor",
        action="store_true",
        help="Diagnostic only: retain LIBERO state conversion but undo its 180-degree image flip",
    )
    parser.add_argument(
        "--max-control-steps",
        type=int,
        default=None,
        help="Optional rollout cap for smoke tests; capped results are not full benchmark scores",
    )
    parser.add_argument(
        "--online-observation-interval",
        type=int,
        default=1,
        help="Capture a new fast observation every N executed control steps during denoising",
    )
    return parser.parse_args()


def resolve_snapshot(path: Path) -> Path:
    """Resolve either a checkpoint directory or a local Hub cache root."""
    if (path / "config.json").is_file():
        return path
    snapshots = sorted((path / "snapshots").glob("*/config.json"))
    if len(snapshots) != 1:
        raise FileNotFoundError(f"expected one snapshot containing config.json under {path}")
    return snapshots[0].parent


def success_from_info(info: dict) -> bool:
    """Read LIBERO success from Gymnasium's current or final info layout."""
    if "is_success" in info:
        value = info["is_success"]
        return bool(np.asarray(value).reshape(-1)[0])
    if "final_info" in info:
        final = info["final_info"]
        if isinstance(final, dict):
            return bool(np.asarray(final.get("is_success", False)).reshape(-1)[0])
        if len(final) and isinstance(final[0], dict):
            return bool(final[0].get("is_success", False))
    return False


def sample_stateless_delay(
    base_seed: int,
    suite: str,
    task_id: int,
    episode: int,
    chunk_index: int,
    delay_min: int,
    delay_max: int,
) -> int:
    """Return a checkpoint-independent delay for one logical replan event."""
    key = f"{base_seed}:{suite}:{task_id}:{episode}:{chunk_index}".encode()
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
    return delay_min + value % (delay_max - delay_min + 1)


def annotate_frame(frame: np.ndarray, line_one: str, line_two: str) -> np.ndarray:
    """Draw a legible two-line chunk timeline overlay on an RGB frame."""
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    padding = 10
    line_height = 18
    draw.rectangle(
        (0, 0, image.width, padding * 2 + line_height * 2),
        fill=(0, 0, 0, 185),
    )
    draw.text((padding, padding), line_one, font=font, fill=(255, 255, 255, 255))
    draw.text(
        (padding, padding + line_height),
        line_two,
        font=font,
        fill=(255, 220, 80, 255),
    )
    return np.asarray(image)


def record_annotated_frame(
    env,
    frames: list[np.ndarray],
    *,
    step: int,
    active_chunk_executed: int,
    queue_remaining: int,
    new_chunk_status: str,
) -> None:
    """Render and append one frame with explicit timeline state."""
    rendered = np.asarray(env.call("render")[0])
    line_one = f"OLD chunk: ran={active_chunk_executed:02d} left={queue_remaining:02d} | step={step:03d}"
    frames.append(annotate_frame(rendered, line_one, new_chunk_status))


def run_episode(
    args,
    env_cfg,
    env,
    policy,
    preprocessor,
    postprocessor,
    env_preprocessor,
    env_postprocessor,
    suite,
    task_id,
    episode,
    delay_min,
    delay_max,
    switch_rng,
    dynamic_delay,
    output_dir,
):
    """Run one episode using an already-loaded policy and save its annotated video."""
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    raw, _ = env.reset(seed=[args.seed + episode], options={NEW_ROLLOUT_OPTION: True})

    def process(raw_obs):
        obs = preprocess_observation(raw_obs)
        obs["task"] = list(env.call("task_description"))
        obs = env_preprocessor(obs)
        if args.disable_env_preprocessor:
            for key, value in obs.items():
                if key.startswith("observation.images."):
                    obs[key] = torch.flip(value, dims=[2, 3])
        return preprocessor(obs)

    def postprocess_chunk(chunk):
        return postprocessor(chunk).squeeze(0).cpu()

    action_normalizer = next(
        (step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)), None
    )
    if isinstance(policy, OnlineFilteringPI05Policy) and (
        action_normalizer is None or policy.config.use_relative_actions
    ):
        raise ValueError(
            "executed-prefix conversion requires an action normalizer and non-relative LIBERO actions"
        )
    last_sent_model_action = None
    last_environment_action = None
    queue_underruns = 0

    def step_env(action):
        nonlocal last_sent_model_action, last_environment_action
        last_environment_action = action.clone()
        transition = env_postprocessor({ACTION: action.unsqueeze(0)})
        if isinstance(policy, OnlineFilteringPI05Policy):
            sent = transition[ACTION].to(device=policy.config.device)
            last_sent_model_action = normalize_executed_command(action_normalizer, sent)
        return env.step(transition[ACTION].cpu().numpy())

    def pop_action_or_safe_hold(action_queue):
        nonlocal queue_underruns
        if action_queue:
            return action_queue.popleft()
        if not args.allow_queue_underrun:
            raise RuntimeError("action queue exhausted while an inference result was pending")
        if last_environment_action is None:
            raise RuntimeError("cannot synthesize a safe hold before the first action")
        queue_underruns += 1
        if env_cfg.control_mode == "absolute":
            # Absolute actions encode a target pose, so repeating the latest target
            # holds position. Sending zeros would instead command an unrelated pose.
            return last_environment_action.clone()
        hold = torch.zeros_like(last_environment_action)
        # LIBERO uses delta end-effector actions and the final dimension is the
        # gripper command. Zero the motion delta instead of repeating it.
        hold[-1] = last_environment_action[-1]
        return hold

    def predict_cached_slow(batch):
        images, image_masks = policy._preprocess_images(batch)
        chunk = policy.model.sample_actions(
            images,
            image_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
        )
        action_dim = policy.config.output_features[ACTION].shape[0]
        return chunk[..., :action_dim]

    def predict_regular(batch):
        if isinstance(policy, OnlineFilteringPI05Policy):
            return policy.predict_action_chunk(
                batch, observation_age_override=args.observation_age_seconds
            )
        return policy.predict_action_chunk(batch)

    processed = process(raw)
    with torch.inference_mode():
        active_model_chunk = (
            predict_cached_slow(processed)
            if args.disable_online_fast and isinstance(policy, OnlineFilteringPI05Policy)
            else predict_regular(processed)
        )
        initial_environment_chunk = postprocess_chunk(active_model_chunk)
        queue = deque(initial_environment_chunk.unbind(0))
    action_diagnostics = {
        "model_shape": list(active_model_chunk.shape),
        "model_abs_mean": float(active_model_chunk.float().abs().mean().cpu()),
        "model_min": float(active_model_chunk.float().min().cpu()),
        "model_max": float(active_model_chunk.float().max().cpu()),
        "environment_shape": list(initial_environment_chunk.shape),
        "environment_abs_mean": float(initial_environment_chunk.float().abs().mean().cpu()),
        "environment_min": float(initial_environment_chunk.float().min().cpu()),
        "environment_max": float(initial_environment_chunk.float().max().cpu()),
        "environment_first_action": initial_environment_chunk[0].float().cpu().tolist(),
    }
    countdown = args.replan_interval
    total_reward = 0.0
    succeeded = False
    steps = 0
    inference_seconds = []
    simulated_pipeline_seconds = []
    frames = []
    events = []
    sampled_delays = []
    sampled_switch_steps = []
    active_chunk_executed = 0
    # Non-online policies (e.g. official PI05) do not sample a flow switch step,
    # but the common chunk-switch annotation still references this field.
    switch_step = None
    delay = None
    chunk_index = 0
    max_steps = int(env.call("_max_episode_steps")[0])
    environment_max_steps = max_steps
    if args.max_control_steps is not None:
        max_steps = min(max_steps, args.max_control_steps)
    record_annotated_frame(
        env,
        frames,
        step=steps,
        active_chunk_executed=active_chunk_executed,
        queue_remaining=len(queue),
        new_chunk_status=f"NEW chunk: IDLE | observe in {countdown} steps",
    )
    while steps < max_steps and not succeeded:
        if countdown == 0:
            delay_seed = args.seed if args.delay_seed is None else args.delay_seed
            delay_sequence = (
                [int(item) for item in args.delay_sequence.split(",")]
                if args.delay_sequence
                else None
            )
            delay = (
                delay_sequence[min(chunk_index, len(delay_sequence) - 1)]
                if delay_sequence is not None
                else sample_stateless_delay(
                    delay_seed,
                    suite,
                    task_id,
                    episode,
                    chunk_index,
                    delay_min,
                    delay_max,
                )
                if dynamic_delay
                else delay_min
            )
            chunk_index += 1
            sampled_delays.append(delay)
            initial = processed
            observation_step = steps
            executed_model_actions = []
            events.append({"type": "observation", "step": observation_step, "delay": delay})
            pending_elapsed = 0
            episode_finished = False
            callback_seconds = 0.0
            encoding_seconds = 0.0
            chunk_assimilations = []

            def advance_pending(
                target_elapsed,
                flow_step=None,
                generation=None,
                *,
                queue=queue,
                executed_model_actions=executed_model_actions,
                active_model_chunk=active_model_chunk,
                delay=delay,
                initial=initial,
            ):
                nonlocal raw, steps, active_chunk_executed, total_reward, succeeded
                nonlocal processed, pending_elapsed, episode_finished
                nonlocal encoding_seconds
                while pending_elapsed < target_elapsed and not episode_finished:
                    raw, reward, terminated, truncated, info = step_env(
                        pop_action_or_safe_hold(queue)
                    )
                    if isinstance(policy, OnlineFilteringPI05Policy):
                        executed_model_actions.append(last_sent_model_action.clone())
                    steps += 1
                    pending_elapsed += 1
                    active_chunk_executed += 1
                    total_reward += float(reward[0])
                    succeeded |= success_from_info(info)
                    episode_finished = (
                        succeeded or bool(terminated[0]) or bool(truncated[0]) or steps >= max_steps
                    )
                    processed = process(raw)
                    capture = (
                        generation is not None
                        and not episode_finished
                        and (
                            pending_elapsed % args.online_observation_interval == 0
                            or pending_elapsed == delay
                        )
                    )
                    if capture:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        encoding_start = time.perf_counter()
                        packet = policy.encode_fast_observation_packet(
                            processed,
                            initial_batch=initial,
                            capture_step=pending_elapsed,
                            capture_timestamp=time.monotonic(),
                            generation=generation,
                            executed_prefix=torch.stack(executed_model_actions, dim=1),
                        )
                        # Synthetic delay is measured by executed control steps. Do not let
                        # GPU contention between parallel eval workers turn synchronous
                        # observations into wall-clock-stale packets or alter the age gate.
                        packet = replace(packet, capture_timestamp=packet.ready_timestamp)
                        policy.push_fast_observation(packet)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        encoding_seconds += time.perf_counter() - encoding_start
                        events.append(
                            {
                                "type": "online_observation",
                                "step": steps,
                                "capture_step": pending_elapsed,
                                "flow_step": flow_step,
                            }
                        )
                    record_annotated_frame(
                        env,
                        frames,
                        step=steps,
                        active_chunk_executed=active_chunk_executed,
                        queue_remaining=len(queue),
                        new_chunk_status=f"NEW chunk: DENOISING k={flow_step} | elapsed={pending_elapsed}/{delay} obs={'READY' if capture else 'WAIT'}",
                    )

            def observation_callback(flow_step, num_steps, generation, delay=delay):
                # A deterministic control/flow clock: advance the simulator while the
                # old chunk executes, before each denoising query. Never advance to
                # t+delay before starting the old-condition flow.
                nonlocal callback_seconds
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                callback_start = time.perf_counter()
                advance_pending(flow_step * delay // (num_steps - 1), flow_step, generation)
                callback_seconds += time.perf_counter() - callback_start

            def assimilation_callback(
                flow_step,
                capture_step,
                updater_used,
                observation_step=observation_step,
                old_chunk_offset=active_chunk_executed,
                queued_count=len(queue),
            ):
                chunk_assimilations.append((flow_step, capture_step))
                events.append(
                    {
                        "type": "online_assimilation",
                        "step": observation_step + capture_step,
                        "capture_step": capture_step,
                        "flow_step": flow_step,
                        "updater_used": updater_used,
                    }
                )
                if frames:
                    frames[-1] = annotate_frame(
                        frames[-1],
                        f"OLD chunk: ran={old_chunk_offset + capture_step:02d} left={queued_count - capture_step:02d} | step={observation_step + capture_step:03d}",
                        f"NEW chunk: ASSIMILATED obs=+{capture_step} flow_k={flow_step} updater={'YES' if updater_used else 'NO'}",
                    )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                if isinstance(policy, OnlineFilteringPI05Policy) and not args.disable_online_fast:
                    chunk = policy.predict_action_chunk(
                        initial,
                        observation_callback=observation_callback,
                        assimilation_callback=assimilation_callback,
                        observation_age_override=args.observation_age_seconds,
                    )
                else:
                    # Official has no fast branch; apply the same elapsed control
                    # steps but keep its original slow observation.
                    callback_start = time.perf_counter()
                    advance_pending(delay)
                    callback_seconds += time.perf_counter() - callback_start
                    chunk = (
                        predict_cached_slow(initial)
                        if args.disable_online_fast and isinstance(policy, OnlineFilteringPI05Policy)
                        else predict_regular(initial)
                    )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pipeline_elapsed = time.perf_counter() - start
            simulated_pipeline_seconds.append(pipeline_elapsed)
            inference_seconds.append(max(0.0, pipeline_elapsed - callback_seconds + encoding_seconds))
            if episode_finished:
                break
            if isinstance(policy, OnlineFilteringPI05Policy) and not args.disable_online_fast:
                expected_capture_steps = []
                for flow_step in range(policy.config.num_inference_steps):
                    capture_step = flow_step * delay // (policy.config.num_inference_steps - 1)
                    if capture_step > 0 and (
                        not expected_capture_steps or capture_step != expected_capture_steps[-1]
                    ):
                        expected_capture_steps.append(capture_step)
                actual_capture_steps = [capture_step for _, capture_step in chunk_assimilations]
                if actual_capture_steps != expected_capture_steps:
                    raise RuntimeError(
                        "online assimilation trace does not match the causal delay clock: "
                        f"expected={expected_capture_steps}, actual={actual_capture_steps}, delay={delay}"
                    )
            active_model_chunk = chunk
            queue = deque(postprocess_chunk(chunk)[delay:].unbind(0))
            active_chunk_executed = delay
            events.append(
                {
                    "type": "chunk_switch",
                    "step": steps,
                    "observation_step": observation_step,
                    "delay": delay,
                    "new_chunk_start_index": delay,
                    "online_assimilations": sum(
                        e["type"] == "online_assimilation" and e["step"] >= observation_step for e in events
                    ),
                    "inference_switch_step": switch_step
                    if isinstance(policy, OnlineFilteringPI05Policy)
                    and policy.config.training_stage == "stage_b"
                    and delay > 0
                    else None,
                }
            )
            countdown = max(0, args.replan_interval - delay)
            record_annotated_frame(
                env,
                frames,
                step=steps,
                active_chunk_executed=active_chunk_executed,
                queue_remaining=len(queue),
                new_chunk_status=(
                    f"NEW chunk: SWITCHED | obs={observation_step:03d} start={delay}"
                    + (f" flow_k={switch_step}" if switch_step is not None else "")
                ),
            )
            continue
        raw, reward, terminated, truncated, info = step_env(pop_action_or_safe_hold(queue))
        steps += 1
        active_chunk_executed += 1
        countdown -= 1
        total_reward += float(reward[0])
        succeeded |= success_from_info(info)
        processed = process(raw)
        record_annotated_frame(
            env,
            frames,
            step=steps,
            active_chunk_executed=active_chunk_executed,
            queue_remaining=len(queue),
            new_chunk_status=f"NEW chunk: IDLE | observe in {countdown} steps",
        )
        if bool(terminated[0]) or bool(truncated[0]):
            break
    output_dir.mkdir(parents=True, exist_ok=True)
    status = "success" if succeeded else "failure"
    video_path = output_dir / f"episode_{episode:03d}_{status}_annotated.mp4"
    write_video(video_path, frames, fps=env_cfg.fps)
    return {
        "episode": episode,
        "seed": args.seed + episode,
        "delay": None if dynamic_delay else delay,
        "delays": sampled_delays,
        "inference_switch_steps": sampled_switch_steps,
        "success": succeeded,
        "reward": total_reward,
        "steps": steps,
        "mean_inference_s": float(np.mean(inference_seconds)) if inference_seconds else 0.0,
        "mean_simulated_pipeline_s": float(np.mean(simulated_pipeline_seconds))
        if simulated_pipeline_seconds
        else 0.0,
        "video_path": str(video_path),
        "events": events,
        "evaluation_protocol": "continuous_online_v1",
        "online_assimilations": sum(e["type"] == "online_assimilation" for e in events),
        "queue_underruns": queue_underruns,
        "initial_action_diagnostics": action_diagnostics,
        "smoke_test_capped": max_steps < environment_max_steps,
    }


def main() -> None:
    """Run one or many LIBERO suites while loading the policy only once."""
    args = parse_args()
    assets_path = Path(os.environ.get("LIBERO_ASSETS_PATH", "/data/zhoukai/dataset/libero-assets"))
    if (assets_path / "scenes/libero_tabletop_base_style.xml").is_file():
        import libero.libero as libero_resources

        # Process-local override; leave other users' LIBERO config/install untouched.
        libero_resources._assets_path_cache = str(assets_path)
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.episode_start < 0 or args.episode_start + args.episodes > 50:
        raise ValueError("episode range must stay within LIBERO's 50 standard init states")
    if args.online_observation_interval < 1:
        raise ValueError("online-observation-interval must be positive")
    if args.max_control_steps is not None and args.max_control_steps < 1:
        raise ValueError("max-control-steps must be positive")
    if args.control_fps < 1:
        raise ValueError("control-fps must be positive")
    suites = [item.strip() for item in args.suite.split(",") if item.strip()]
    if not suites:
        raise ValueError("at least one suite is required")
    if args.all_tasks and args.task_ids:
        raise ValueError("--all-tasks and --task-ids are mutually exclusive")
    requested_task_ids = [int(item) for item in args.task_ids.split(",")] if args.task_ids else [args.task_id]
    delay_min = args.delay if args.delay_min is None else args.delay_min
    delay_max = args.delay if args.delay_max is None else args.delay_max
    if delay_min < 0 or delay_min > delay_max:
        raise ValueError("delay range must satisfy 0 <= delay-min <= delay-max")
    if delay_max >= args.replan_interval and not args.allow_queue_underrun:
        raise ValueError("delay-max must be smaller than replan-interval")
    set_seed(args.seed)
    policy_path = resolve_snapshot(args.policy_path)
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = args.device
    policy_cfg.n_action_steps = policy_cfg.chunk_size
    baseline_root = Path("/data/yhr/pretrain/online_filtering_baselines")
    if policy_cfg.type == "pi0_fast":
        text_tokenizer = Path(
            "/data/yhr/pretrain/recovered_zhoukai/pretrain/paligemma-tokenizer"
        )
        action_tokenizer = baseline_root / "fast-action-tokenizer"
        if (text_tokenizer / "tokenizer.json").is_file():
            policy_cfg.text_tokenizer_name = str(text_tokenizer)
        if (action_tokenizer / "processor_config.json").is_file():
            policy_cfg.action_tokenizer_name = str(action_tokenizer)
    dino_paths = (
        Path("/data/yhr/pretrain/dinov3-vits16-pretrain-lvd1689m"),
        Path("/data/zhoukai/pretrain/dinov3-vits16-pretrain-lvd1689m"),
    )
    if isinstance(policy_cfg, PreTrainedConfig) and hasattr(policy_cfg, "dinov3_model_id"):
        for dino_path in dino_paths:
            if (dino_path / "config.json").is_file():
                policy_cfg.dinov3_model_id = str(dino_path)
                break
    if hasattr(policy_cfg, "num_inference_steps") and args.switch_step_max is None:
        args.switch_step_max = policy_cfg.num_inference_steps - 1
    if hasattr(policy_cfg, "max_inference_delay") and delay_max > policy_cfg.max_inference_delay:
        raise ValueError("delay-max exceeds the checkpoint training delay range")
    if hasattr(policy_cfg, "num_inference_steps") and policy_cfg.num_inference_steps < 2:
        raise ValueError("continuous evaluation requires at least two denoising steps")
    if args.replan_interval + delay_max > policy_cfg.chunk_size and not args.allow_queue_underrun:
        raise ValueError("old chunk cannot cover replan-interval + delay-max")
    switch_rng = random.Random(args.seed + 1 if args.switch_seed is None else args.switch_seed)
    control_mode = args.control_mode or ("absolute" if policy_cfg.type == "xvla" else "relative")
    episode_length = args.episode_length or (800 if policy_cfg.type == "xvla" else None)
    # Build the policy once. All LIBERO suites use the same feature contract.
    policy_env_cfg = LiberoEnv(
        task=suites[0],
        task_ids=[0],
        max_parallel_tasks=1,
        fps=args.control_fps,
        control_mode=control_mode,
        episode_length=episode_length,
    )
    policy = make_policy(cfg=policy_cfg, env_cfg=policy_env_cfg)
    policy.eval()
    if isinstance(policy, OnlineFilteringPI05Policy) and policy.config.training_stage != "online_flow":
        raise ValueError(
            "continuous two-condition evaluation requires training_stage='online_flow'; "
            f"checkpoint has {policy.config.training_stage!r}"
        )
    preprocessor_overrides = {"device_processor": {"device": args.device}}
    if isinstance(policy, OnlineFilteringPI05Policy):
        preprocessor_overrides["tokenizer_processor"] = {
            "tokenizer_name": "/data/yhr/pretrain/recovered_zhoukai/pretrain/paligemma-tokenizer"
        }
    elif policy_cfg.type in {"pi0", "pi05"}:
        # Official checkpoints retain the Hub tokenizer id in their processor
        # config. Evaluation is offline on this machine, so use the migrated
        # local tokenizer snapshot just like the online-filtering checkpoints.
        preprocessor_overrides["tokenizer_processor"] = {
            "tokenizer_name": "/data/yhr/pretrain/recovered_zhoukai/pretrain/paligemma-tokenizer"
        }
    elif policy_cfg.type == "xvla":
        bart_tokenizer = baseline_root / "bart-large-tokenizer"
        if (bart_tokenizer / "tokenizer.json").is_file():
            preprocessor_overrides["tokenizer_processor"] = {
                "tokenizer_name": str(bart_tokenizer)
            }
    elif policy_cfg.type == "pi0_fast":
        preprocessor_overrides["tokenizer_processor"] = {
            "tokenizer_name": policy_cfg.text_tokenizer_name
        }
        preprocessor_overrides["action_tokenizer_processor"] = {
            "action_tokenizer_name": policy_cfg.action_tokenizer_name,
            "paligemma_tokenizer_name": policy_cfg.text_tokenizer_name,
        }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    output_root = args.output_dir
    multi = args.all_tasks or len(suites) > 1 or len(requested_task_ids) > 1
    all_results = []
    try:
        for suite in suites:
            suite_cfg = LiberoEnv(
                task=suite,
                task_ids=None if args.all_tasks else requested_task_ids,
                max_parallel_tasks=1,
                fps=args.control_fps,
                control_mode=control_mode,
                episode_length=episode_length,
            )
            suite_envs = make_env(suite_cfg, n_envs=1, use_async_envs=False)
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(suite_cfg, policy_cfg)
            try:
                task_map = suite_envs[suite]
                for task_id, env in task_map.items():
                    task_output_dir = output_root / suite / f"task_{task_id}" if multi else output_root
                    results = []
                    for episode in range(args.episode_start, args.episode_start + args.episodes):
                        try:
                            result = run_episode(
                                args,
                                suite_cfg,
                                env,
                                policy,
                                preprocessor,
                                postprocessor,
                                env_preprocessor,
                                env_postprocessor,
                                suite,
                                task_id,
                                episode,
                                delay_min,
                                delay_max,
                                switch_rng,
                                args.dynamic_delay,
                                task_output_dir,
                            )
                        except AssertionError as error:
                            # PI0-FAST can occasionally emit a malformed textual action
                            # sequence.  This is a policy failure, not an evaluator failure:
                            # count the episode as failed and continue without retrying or
                            # attempting to reinterpret the generated tokens.
                            message = str(error)
                            if policy_cfg.type != "pi0_fast" or not message.startswith(
                                "Token sequence does not start with"
                            ):
                                raise
                            logging.warning(
                                "PI0-FAST produced an invalid action sequence for %s task %s "
                                "episode %s; counting the episode as failed",
                                suite,
                                task_id,
                                episode,
                            )
                            result = {
                                "episode": episode,
                                "seed": args.seed + episode,
                                "delay": None if args.dynamic_delay else delay_min,
                                "delays": [],
                                "inference_switch_steps": [],
                                "success": False,
                                "reward": 0.0,
                                "steps": 0,
                                "mean_inference_s": None,
                                "mean_simulated_pipeline_s": None,
                                "video_path": None,
                                "events": [],
                                "evaluation_protocol": "continuous_online_v1",
                                "online_assimilations": 0,
                                "queue_underruns": 0,
                                "failure_reason": "invalid_pi0_fast_action_sequence",
                                "smoke_test_capped": False,
                            }
                        results.append(result)
                    summary = {
                        "policy_path": str(policy_path),
                        "policy_type": policy_cfg.type,
                        "posterior_bridge": (
                            isinstance(policy, OnlineFilteringPI05Policy)
                            and policy.config.training_stage == "stage_b"
                        ),
                        "suite": suite,
                        "task_id": task_id,
                        "episodes_per_task": args.episodes,
                        "evaluation_protocol": "continuous_online_v1",
                        "online_observation_interval": args.online_observation_interval,
                        "delay_range": [delay_min, delay_max],
                        "dynamic_delay": args.dynamic_delay,
                        "replan_interval": args.replan_interval,
                        "control_fps": args.control_fps,
                        "control_mode": control_mode,
                        "episode_length": episode_length,
                        "inference_switch_step_range": None,
                        "flow_control_clock": "linear_over_sampled_delay",
                        "delay_sampling": "stateless_blake2b_by_suite_task_episode_chunk",
                        "episodes": results,
                        "success_rate": sum(item["success"] for item in results) / len(results),
                    }
                    task_output_dir.mkdir(parents=True, exist_ok=True)
                    with (task_output_dir / "summary.json").open("w") as file:
                        json.dump(summary, file, indent=2)
                    all_results.extend([{**item, "suite": suite, "task_id": task_id} for item in results])
                    print(json.dumps(summary, indent=2), flush=True)
            finally:
                close_envs(suite_envs)
    finally:
        pass
    if multi:
        aggregate = {
            "policy_path": str(policy_path),
            "posterior_bridge": (
                isinstance(policy, OnlineFilteringPI05Policy) and policy.config.training_stage == "stage_b"
            ),
            "episodes_per_task": args.episodes,
            "evaluation_protocol": "continuous_online_v1",
            "online_observation_interval": args.online_observation_interval,
            "delay_range": [delay_min, delay_max],
            "dynamic_delay": args.dynamic_delay,
            "replan_interval": args.replan_interval,
            "control_fps": args.control_fps,
            "control_mode": control_mode,
            "episode_length": episode_length,
            "inference_switch_step_range": None,
            "flow_control_clock": "linear_over_sampled_delay",
            "delay_sampling": "stateless_blake2b_by_suite_task_episode_chunk",
            "episodes": all_results,
            "success_rate": sum(item["success"] for item in all_results) / len(all_results),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "summary_all.json").open("w") as file:
            json.dump(aggregate, file, indent=2)
        print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
