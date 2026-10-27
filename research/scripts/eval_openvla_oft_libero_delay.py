#!/usr/bin/env python
"""Evaluate one combined OpenVLA-OFT checkpoint on LIBERO with synthetic latency."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

OPENVLA_ROOT = Path("/data/yhr/project/robot/openvla-oft")
sys.path.insert(0, str(OPENVLA_ROOT))

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
)
from experiments.robot.libero.run_libero_eval import (  # noqa: E402
    GenerateConfig,
    initialize_model,
    prepare_observation,
    process_action,
)
from experiments.robot.robot_utils import (  # noqa: E402
    get_action,
    get_image_resize_size,
    set_seed_everywhere,
)
from libero.libero import benchmark  # noqa: E402


# Match the exact control-step budgets used by the LeRobot PI0.5 evaluation.
PI05_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--suite",
        required=True,
        choices=("libero_spatial", "libero_object", "libero_goal", "libero_10"),
    )
    parser.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--delay-min", type=int, default=1)
    parser.add_argument("--delay-max", type=int, default=10)
    parser.add_argument("--replan-interval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--save-videos", action="store_true")
    return parser.parse_args()


def sample_delay(seed: int, suite: str, task_id: int, episode: int, chunk_index: int, lo: int, hi: int) -> int:
    payload = f"{seed}|{suite}|{task_id}|{episode}|{chunk_index}".encode()
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return lo + value % (hi - lo + 1)


def safe_hold(last_action: np.ndarray | None) -> np.ndarray:
    if last_action is None:
        return get_libero_dummy_action("openvla")
    action = np.zeros_like(last_action)
    action[-1] = last_action[-1]
    return action


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.delay_min < 0 or args.delay_max < args.delay_min:
        raise ValueError("invalid episode count or delay range")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed_everywhere(args.seed)

    cfg = GenerateConfig(
        pretrained_checkpoint=str(args.checkpoint),
        task_suite_name=args.suite,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=8,
        num_trials_per_task=args.episodes,
        use_wandb=False,
        seed=args.seed,
    )
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = [int(value) for value in args.task_ids.split(",")]
    results: list[dict] = []

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        task_dir = args.output_dir / args.suite / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)

        for episode in range(args.episode_start, args.episode_start + args.episodes):
            env.reset()
            obs = env.set_init_state(initial_states[episode])
            frames: list[np.ndarray] = []
            for _ in range(cfg.num_steps_wait):
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

            observation, frame = prepare_observation(obs, resize_size)
            frames.append(frame)
            start = time.perf_counter()
            initial_actions = get_action(
                cfg,
                model,
                observation,
                task_description,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=cfg.use_film,
            )
            inference_times = [time.perf_counter() - start]
            queue = deque(initial_actions)
            last_action = None
            steps = 0
            countdown = args.replan_interval
            chunk_index = 0
            delays: list[int] = []
            underruns = 0
            success = False
            max_steps = PI05_MAX_STEPS[args.suite]

            while steps < max_steps and not success:
                if countdown == 0:
                    # Match the five-way PI0.5/X-VLA evaluator's per-task worker
                    # seeds independently of how many OpenVLA models fit on one GPU.
                    delay_seed = args.seed + (task_id % 5) * 100000
                    delay = sample_delay(
                        delay_seed,
                        args.suite,
                        task_id,
                        episode,
                        chunk_index,
                        args.delay_min,
                        args.delay_max,
                    )
                    delays.append(delay)
                    chunk_index += 1
                    stale_observation, _ = prepare_observation(obs, resize_size)
                    for _ in range(delay):
                        if queue:
                            action = process_action(np.asarray(queue.popleft()).copy(), cfg.model_family)
                        else:
                            underruns += 1
                            action = safe_hold(last_action)
                        last_action = action
                        obs, _, done, _ = env.step(action.tolist())
                        steps += 1
                        _, frame = prepare_observation(obs, resize_size)
                        frames.append(frame)
                        if done or steps >= max_steps:
                            success = bool(done)
                            break
                    if success or steps >= max_steps:
                        break
                    start = time.perf_counter()
                    actions = get_action(
                        cfg,
                        model,
                        stale_observation,
                        task_description,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=cfg.use_film,
                    )
                    inference_times.append(time.perf_counter() - start)
                    queue = deque(actions[delay:])
                    countdown = max(0, args.replan_interval - delay)
                    continue

                if queue:
                    action = process_action(np.asarray(queue.popleft()).copy(), cfg.model_family)
                else:
                    underruns += 1
                    action = safe_hold(last_action)
                last_action = action
                obs, _, done, _ = env.step(action.tolist())
                steps += 1
                countdown -= 1
                _, frame = prepare_observation(obs, resize_size)
                frames.append(frame)
                success = bool(done)

            video_path = None
            if args.save_videos:
                status = "success" if success else "failure"
                video_path = task_dir / f"episode_{episode:03d}_{status}.mp4"
                imageio.mimsave(video_path, frames, fps=20)
            item = {
                "suite": args.suite,
                "task_id": task_id,
                "episode": episode,
                "success": success,
                "steps": steps,
                "delays": delays,
                "queue_underruns": underruns,
                "mean_inference_s": float(np.mean(inference_times)),
                "video_path": str(video_path) if video_path else None,
            }
            results.append(item)
            with (task_dir / f"episode_{episode:03d}.json").open("w") as file:
                json.dump(item, file, indent=2)
            print(json.dumps(item), flush=True)
        env.close()

    summary = {
        "checkpoint": str(args.checkpoint),
        "suite": args.suite,
        "delay_range": [args.delay_min, args.delay_max],
        "replan_interval": args.replan_interval,
        "max_steps": PI05_MAX_STEPS[args.suite],
        "episodes": results,
        "success_rate": sum(item["success"] for item in results) / len(results),
        "queue_underruns": sum(item["queue_underruns"] for item in results),
    }
    with (args.output_dir / f"summary_{args.suite}.json").open("w") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
