#!/usr/bin/env python
"""Compare online fast switching with stale conditioning and a full current-observation replan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data._utils.collate import default_collate

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.online_filtering_pi05.configuration_online_filtering_pi05 import (
    OnlineFilteringPI05Config,
)
from lerobot.policies.online_filtering_pi05.modeling_online_filtering_pi05 import (
    OnlineFilteringPI05Policy,
)
from lerobot.policies.pi05.online_filtering import DINOv3PatchEncoder, hard_commit_action_prefix
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def select_critical_events(root: Path, delay: int, max_events: int) -> list[dict]:
    """Select task-diverse gripper switches and large continuous-action changes."""
    gripper: dict[tuple[str, int], dict] = {}
    motion: dict[int, tuple[float, dict]] = {}
    for path in sorted(root.glob("data/**/*.parquet")):
        table = pq.read_table(path, columns=["index", "episode_index", "task_index", "action"])
        indices = np.asarray(table["index"])
        episodes = np.asarray(table["episode_index"])
        tasks = np.asarray(table["task_index"])
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        same_episode = episodes[1:] == episodes[:-1]
        for pos in np.flatnonzero((actions[1:, -1] * actions[:-1, -1] < 0) & same_episode) + 1:
            if pos < delay or pos + 50 >= len(indices) or episodes[pos - delay] != episodes[pos + 50]:
                continue
            kind = "gripper_open" if actions[pos, -1] > actions[pos - 1, -1] else "gripper_close"
            task = int(tasks[pos])
            gripper.setdefault(
                (kind, task),
                {
                    "index": int(indices[pos]) - delay,
                    "event_index": int(indices[pos]),
                    "task_index": task,
                    "kind": kind,
                    "score": float(abs(actions[pos, -1] - actions[pos - 1, -1])),
                },
            )
        continuous_delta = np.linalg.norm(actions[1:, :-1] - actions[:-1, :-1], axis=1)
        for pos in np.flatnonzero(same_episode) + 1:
            if pos < delay or pos + 50 >= len(indices) or episodes[pos - delay] != episodes[pos + 50]:
                continue
            task = int(tasks[pos])
            score = float(continuous_delta[pos - 1])
            event = {
                "index": int(indices[pos]) - delay,
                "event_index": int(indices[pos]),
                "task_index": task,
                "kind": "large_motion_change",
                "score": score,
            }
            if task not in motion or score > motion[task][0]:
                motion[task] = (score, event)

    gripper_events = sorted(gripper.values(), key=lambda row: (row["kind"], row["task_index"]))
    motion_events = [row for _, row in sorted(motion.values(), key=lambda item: -item[0])]
    selected = []
    # Reserve half the budget for discrete gripper events and half for continuous changes.
    for pool, budget in ((gripper_events, max_events // 2), (motion_events, max_events - max_events // 2)):
        if not pool:
            continue
        positions = np.linspace(0, len(pool) - 1, min(budget, len(pool)), dtype=int)
        selected.extend(pool[position] for position in positions)
    return selected[:max_events]


def temporal_step(batch: dict[str, torch.Tensor], cfg, step: int) -> dict[str, torch.Tensor]:
    current = dict(batch)
    for key in [*cfg.image_features, OBS_STATE]:
        current[key] = batch[key][:, step]
        pad_key = f"{key}_is_pad"
        if pad_key in batch and batch[pad_key].ndim == 2:
            current[pad_key] = batch[pad_key][:, step]
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/zhoukai/dataset/libero"))
    parser.add_argument("--delay", type=int, default=8)
    parser.add_argument("--max-events", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.set_num_threads(4)
    events = select_critical_events(args.dataset_root, args.delay, args.max_events)
    if not events:
        raise RuntimeError("no eligible critical events found")

    cfg = OnlineFilteringPI05Config.from_pretrained(args.checkpoint)
    if args.delay > cfg.max_inference_delay:
        raise ValueError("diagnostic delay exceeds checkpoint max_inference_delay")
    cfg.device = args.device
    encoder = DINOv3PatchEncoder.from_pretrained(
        "/data/zhoukai/pretrain/dinov3-vits16-pretrain-lvd1689m"
    )
    policy = OnlineFilteringPI05Policy.from_pretrained(
        args.checkpoint, config=cfg, fast_encoder=encoder
    ).eval()
    preprocessor, _ = make_pre_post_processors(
        cfg,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": "/data/zhoukai/pretrain/paligemma-tokenizer"},
            "device_processor": {"device": args.device},
        },
    )
    dataset = LeRobotDataset(
        "lerobot/libero",
        root=args.dataset_root,
        video_backend="pyav",
        delta_timestamps={
            **{key: [i / 10 for i in range(cfg.max_inference_delay + 1)] for key in [*cfg.image_features, OBS_STATE]},
            ACTION: [i / 10 for i in range(cfg.chunk_size)],
        },
    )

    results = []
    with torch.no_grad():
        for start in range(0, len(events), args.batch_size):
            selected = events[start : start + args.batch_size]
            batch = preprocessor(default_collate([dataset[event["index"]] for event in selected]))
            size = len(selected)
            actions = policy.prepare_action(batch)
            action_dim = cfg.output_features[ACTION].shape[0]
            initial = policy._initial_slow_batch(batch)
            images, image_masks = policy._preprocess_images(initial)
            tokens = batch[OBS_LANGUAGE_TOKENS]
            token_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
            prefix_embs, prefix_masks, prefix_cache = policy.model.prepare_prefix_cache(
                images, image_masks, tokens, token_masks
            )
            total_delays = torch.full((size,), args.delay, device=actions.device, dtype=torch.long)
            schedule = (
                torch.arange(cfg.num_inference_steps, device=actions.device)[None, :]
                * total_delays[:, None]
                // (cfg.num_inference_steps - 1)
            )

            stale_batch = dict(batch)
            for key in [*cfg.image_features, OBS_STATE]:
                stale_batch[key] = batch[key].clone()
                stale_batch[key][:] = batch[key][:, :1]
            correct_fast = {
                step: policy._make_fast_tokens(batch, schedule[:, step])
                for step in range(cfg.num_inference_steps)
            }
            stale_fast = {
                step: policy._make_fast_tokens(stale_batch, schedule[:, step])
                for step in range(cfg.num_inference_steps)
            }

            torch.manual_seed(424242 + start)
            noise = policy.model.sample_noise(actions.shape, actions.device)

            def online_rollout(fast: dict[int, torch.Tensor]) -> torch.Tensor:
                particle = noise.clone()
                committed = torch.zeros_like(particle)
                committed[..., :action_dim] = actions[..., :action_dim]
                for step in range(cfg.num_inference_steps):
                    old_step = max(step - 1, 0)
                    old_delays = schedule[:, old_step]
                    new_delays = schedule[:, step]
                    particle = hard_commit_action_prefix(particle, committed, new_delays)
                    condition = policy._make_online_condition_tokens(
                        fast[old_step], fast[step], old_delays, new_delays
                    )
                    flow_time = torch.full(
                        (size,), 1.0 - step / cfg.num_inference_steps, device=actions.device
                    )
                    velocity = policy.model.denoise_step(
                        prefix_pad_masks=prefix_masks,
                        past_key_values=prefix_cache,
                        x_t=particle,
                        timestep=flow_time,
                        prefix_embs=prefix_embs,
                        fast_adapters=policy.model.fast_adapters,
                        fast_tokens=condition,
                        fast_delays=new_delays,
                        committed_delays=new_delays,
                    )
                    particle = particle - velocity / cfg.num_inference_steps
                    particle = hard_commit_action_prefix(particle, committed, new_delays)
                return particle[..., :action_dim]

            online = online_rollout(correct_fast)
            stale = online_rollout(stale_fast)

            current = temporal_step(batch, cfg, args.delay)
            current_images, current_image_masks = policy._preprocess_images(current)
            current_prefix_embs, current_prefix_masks, current_prefix_cache = policy.model.prepare_prefix_cache(
                current_images, current_image_masks, tokens, token_masks
            )
            current_fast = policy._make_single_fast_tokens(current, delay=0)
            zero_delay = torch.zeros(size, device=actions.device, dtype=torch.long)
            current_condition = policy._make_online_condition_tokens(
                current_fast, current_fast, zero_delay, zero_delay
            )
            replan_noise = torch.randn_like(noise)
            replan_noise[:, : cfg.chunk_size - args.delay] = noise[:, args.delay :]
            replan = replan_noise
            for step in range(cfg.num_inference_steps):
                flow_time = torch.full(
                    (size,), 1.0 - step / cfg.num_inference_steps, device=actions.device
                )
                velocity = policy.model.denoise_step(
                    prefix_pad_masks=current_prefix_masks,
                    past_key_values=current_prefix_cache,
                    x_t=replan,
                    timestep=flow_time,
                    prefix_embs=current_prefix_embs,
                    fast_adapters=policy.model.fast_adapters,
                    fast_tokens=current_condition,
                    fast_delays=zero_delay,
                )
                replan = replan - velocity / cfg.num_inference_steps
            replan = replan[..., :action_dim]

            horizon = cfg.chunk_size - args.delay
            online_aligned = online[:, args.delay :]
            stale_aligned = stale[:, args.delay :]
            replan_aligned = replan[:, :horizon]
            gt_aligned = actions[:, args.delay :, :action_dim]
            final_fast_delta = (
                correct_fast[cfg.num_inference_steps - 1].float()
                - stale_fast[cfg.num_inference_steps - 1].float()
            ).square().mean(dim=(1, 2))
            final_fast_energy = correct_fast[cfg.num_inference_steps - 1].float().square().mean(
                dim=(1, 2)
            )

            def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return (a.float() - b.float()).square().mean(dim=(1, 2))

            online_replan = mse(online_aligned, replan_aligned)
            stale_replan = mse(stale_aligned, replan_aligned)
            for index, event in enumerate(selected):
                results.append(
                    {
                        **event,
                        "delay": args.delay,
                        "online_to_replan_mse": online_replan[index].item(),
                        "stale_to_replan_mse": stale_replan[index].item(),
                        "online_better_than_stale": bool(
                            online_replan[index] < stale_replan[index]
                        ),
                        "online_to_gt_mse": mse(online_aligned, gt_aligned)[index].item(),
                        "stale_to_gt_mse": mse(stale_aligned, gt_aligned)[index].item(),
                        "replan_to_gt_mse": mse(replan_aligned, gt_aligned)[index].item(),
                        "online_stale_mse": mse(online_aligned, stale_aligned)[index].item(),
                        "fast_token_relative_delta": (
                            final_fast_delta[index] / final_fast_energy[index].clamp_min(1e-12)
                        ).item(),
                    }
                )
            print(json.dumps({"completed": len(results), "total": len(events)}), flush=True)

    scalar_keys = [
        "online_to_replan_mse",
        "stale_to_replan_mse",
        "online_to_gt_mse",
        "stale_to_gt_mse",
        "replan_to_gt_mse",
        "online_stale_mse",
        "fast_token_relative_delta",
    ]
    report = {
        "checkpoint": str(args.checkpoint),
        "events": results,
        "aggregate": {
            **{key: sum(row[key] for row in results) / len(results) for key in scalar_keys},
            "online_better_than_stale_fraction": sum(
                row["online_better_than_stale"] for row in results
            )
            / len(results),
        },
        "by_kind": {},
        "note": (
            "Read-only normalized-action diagnostic. Full replan uses the same checkpoint with the "
            "observation at the critical event as its slow/fast condition and aligned noise."
        ),
    }
    for kind in sorted({row["kind"] for row in results}):
        rows = [row for row in results if row["kind"] == kind]
        report["by_kind"][kind] = {
            **{key: sum(row[key] for row in rows) / len(rows) for key in scalar_keys},
            "count": len(rows),
            "online_better_than_stale_fraction": sum(
                row["online_better_than_stale"] for row in rows
            )
            / len(rows),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
