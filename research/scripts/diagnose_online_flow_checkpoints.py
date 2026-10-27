#!/usr/bin/env python
"""Read-only PI0.5 FM and online-condition diagnostics for saved checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def masked_mse(value: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, action_dim: int):
    squared = (value[..., :action_dim].float() - target[..., :action_dim].float()).square()
    count = mask.sum(dim=1).clamp_min(1) * action_dim
    return (squared * mask.unsqueeze(-1)).sum(dim=(1, 2)) / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-root", default="/data/zhoukai/dataset/libero")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(314159)
    cfg = OnlineFilteringPI05Config.from_pretrained(args.checkpoint)
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
            **{key: [i / 10 for i in range(11)] for key in [*cfg.image_features, OBS_STATE]},
            ACTION: [i / 10 for i in range(cfg.chunk_size)],
        },
    )
    indices = [int((i + 0.5) * (len(dataset) - 60) / args.batch_size) for i in range(args.batch_size)]
    batch = preprocessor(default_collate([dataset[index] for index in indices]))
    valid = policy._observation_validity(batch)
    if not valid[:, : cfg.max_inference_delay + 1].all():
        raise RuntimeError("diagnostic samples include padded observations")

    with torch.no_grad():
        # Use identical canonical noise/time samples for every checkpoint process.
        torch.manual_seed(271828)
        canonical_loss, canonical_metrics = policy._forward_online_flow_canonical(batch, "mean")

        actions = policy.prepare_action(batch)
        action_dim = cfg.output_features[ACTION].shape[0]
        initial = policy._initial_slow_batch(batch)
        images, image_masks = policy._preprocess_images(initial)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        token_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        batch_size = actions.shape[0]
        # Cover the deployed total-delay range while keeping the same schedule across checkpoints.
        totals = 1 + torch.arange(batch_size, device=actions.device) % cfg.max_inference_delay
        schedules = (
            torch.arange(cfg.num_inference_steps, device=actions.device)[None, :] * totals[:, None]
            // (cfg.num_inference_steps - 1)
        )
        fast_tokens = {
            step: policy._make_fast_tokens(batch, schedules[:, step])
            for step in range(cfg.num_inference_steps)
        }
        torch.manual_seed(161803)
        particle = policy.model.sample_noise(actions.shape, actions.device)
        positions = torch.arange(cfg.chunk_size, device=actions.device)[None, :]
        per_step = []

        for step in range(cfg.num_inference_steps):
            old_step = max(step - 1, 0)
            old_delays = schedules[:, old_step]
            new_delays = schedules[:, step]
            old_fast = fast_tokens[old_step]
            new_fast = fast_tokens[step]
            rows = torch.arange(batch_size, device=actions.device)
            stale_batch = dict(batch)
            shuffled_batch = dict(batch)
            shuffled_rows = rows.roll(1)
            for key in [*cfg.image_features, OBS_STATE]:
                stale_batch[key] = batch[key].clone()
                shuffled_batch[key] = batch[key].clone()
                stale_batch[key][rows, new_delays] = batch[key][rows, old_delays]
                shuffled_batch[key][rows, new_delays] = batch[key][
                    shuffled_rows, new_delays[shuffled_rows]
                ]
            # Hold the externally supplied new delay fixed and alter only the observation content.
            stale_new_fast = policy._make_fast_tokens(stale_batch, new_delays)
            shuffled_new_fast = policy._make_fast_tokens(shuffled_batch, new_delays)
            time = torch.full(
                (batch_size,), 1.0 - step / cfg.num_inference_steps,
                device=actions.device,
                dtype=torch.float32,
            )
            suffix = positions >= new_delays[:, None]
            if "action_is_pad" in batch:
                suffix &= ~batch["action_is_pad"].to(device=actions.device, dtype=torch.bool)
            model_time = torch.where(suffix, time[:, None], torch.zeros_like(time[:, None]))

            def velocity(new_condition: torch.Tensor) -> torch.Tensor:
                condition = policy._make_online_condition_tokens(
                    old_fast, new_condition, old_delays, new_delays
                )
                return policy.model.predict_velocity(
                    images,
                    image_masks,
                    tokens,
                    token_masks,
                    particle,
                    model_time,
                    fast_adapters=policy.model.fast_adapters,
                    fast_tokens=condition,
                    fast_delays=new_delays,
                )

            correct = velocity(new_fast)
            stale = velocity(stale_new_fast)
            shuffled = velocity(shuffled_new_fast)
            target = (particle - actions) / time[:, None, None].clamp_min(1e-6)
            correct_mse = masked_mse(correct, target, suffix, action_dim)
            stale_mse = masked_mse(stale, target, suffix, action_dim)
            shuffled_mse = masked_mse(shuffled, target, suffix, action_dim)
            target_energy = masked_mse(target, torch.zeros_like(target), suffix, action_dim)
            switch = new_delays > old_delays
            correct_better_stale = correct_mse < stale_mse
            correct_better_shuffled = correct_mse < shuffled_mse
            fast_delta_energy = (new_fast.float() - stale_new_fast.float()).square().mean(dim=(1, 2))
            fast_token_energy = new_fast.float().square().mean(dim=(1, 2)).clamp_min(1e-12)
            per_step.append(
                {
                    "step": step,
                    "flow_time": float(time[0]),
                    "switch_fraction": switch.float().mean().item(),
                    "correct_velocity_mse": correct_mse.mean().item(),
                    "endpoint_equivalent_mse": (correct_mse * time.square()).mean().item(),
                    "stale_velocity_mse": stale_mse.mean().item(),
                    "shuffled_velocity_mse": shuffled_mse.mean().item(),
                    "correct_better_than_stale_fraction": correct_better_stale.float().mean().item(),
                    "correct_better_than_shuffled_fraction": correct_better_shuffled.float().mean().item(),
                    "switch_correct_better_than_stale_fraction": (
                        correct_better_stale[switch].float().mean().item() if switch.any() else None
                    ),
                    "switch_correct_better_than_shuffled_fraction": (
                        correct_better_shuffled[switch].float().mean().item() if switch.any() else None
                    ),
                    "switch_fast_token_delta_energy": (
                        fast_delta_energy[switch].mean().item() if switch.any() else 0.0
                    ),
                    "switch_fast_token_relative_delta": (
                        (fast_delta_energy[switch] / fast_token_energy[switch]).mean().item()
                        if switch.any()
                        else 0.0
                    ),
                    "target_velocity_energy": target_energy.mean().item(),
                    "correct_stale_velocity_delta_mse": masked_mse(
                        correct, stale, suffix, action_dim
                    ).mean().item(),
                }
            )
            proposal = particle - correct / cfg.num_inference_steps
            particle = hard_commit_action_prefix(proposal, actions, new_delays)

    report = {
        "checkpoint": str(args.checkpoint),
        "indices": indices,
        "schedule": "linear_deployment_clock",
        "total_delays": totals.cpu().tolist(),
        "canonical_flow_loss": canonical_loss.item(),
        "canonical_metrics": canonical_metrics,
        "per_step": per_step,
        "aggregate_recovery_steps_1_to_9": {
            key: sum(row[key] for row in per_step[1:]) / len(per_step[1:])
            for key in [
                "correct_velocity_mse",
                "endpoint_equivalent_mse",
                "stale_velocity_mse",
                "shuffled_velocity_mse",
                "correct_better_than_stale_fraction",
                "correct_better_than_shuffled_fraction",
                "target_velocity_energy",
                "correct_stale_velocity_delta_mse",
                "switch_fraction",
            ]
        },
        "note": (
            "Read-only dataset diagnostic. Each checkpoint rolls out its own particles from shared "
            "noise under the same linear delay schedules; it is not a task-success evaluation."
        ),
    }
    switch_rows = [row for row in per_step[1:] if row["switch_fraction"] > 0]
    report["aggregate_switch_only"] = {
        key: sum(row[key] for row in switch_rows if row[key] is not None)
        / sum(row[key] is not None for row in switch_rows)
        for key in [
            "switch_correct_better_than_stale_fraction",
            "switch_correct_better_than_shuffled_fraction",
            "switch_fast_token_delta_energy",
            "switch_fast_token_relative_delta",
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
