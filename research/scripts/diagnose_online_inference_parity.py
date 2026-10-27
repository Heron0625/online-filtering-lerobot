#!/usr/bin/env python
"""Check whether online PI0.5 training and inference vector-field paths agree."""

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


def stats(delta: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = delta.float()
    reference = reference.float()
    return {
        "mse": delta.square().mean().item(),
        "max_abs": delta.abs().max().item(),
        "relative_rms": (delta.square().mean().sqrt() / reference.square().mean().sqrt()).item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-root", default="/data/zhoukai/dataset/libero")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(20261005)
    cfg = OnlineFilteringPI05Config.from_pretrained(args.checkpoint)
    cfg.device = args.device
    encoder = DINOv3PatchEncoder.from_pretrained("/data/zhoukai/pretrain/dinov3-vits16-pretrain-lvd1689m")
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
    batch = preprocessor(default_collate([dataset[len(dataset) // 2]]))
    initial = policy._initial_slow_batch(batch)
    images, image_masks = policy._preprocess_images(initial)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    token_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.prepare_action(batch)
    prefix_embs, prefix_pad_masks, cache = policy.model.prepare_prefix_cache(
        images, image_masks, tokens, token_masks
    )
    particle = policy.model.sample_noise(actions.shape, actions.device)
    report = {}

    with torch.no_grad():
        for delay_value in (0, 1, 5, 10):
            old_value = max(delay_value - 1, 0)
            old_delay = torch.tensor([old_value], device=actions.device)
            delay = torch.tensor([delay_value], device=actions.device)
            old_fast = policy._make_fast_tokens(batch, old_delay)
            new_fast = policy._make_fast_tokens(batch, delay)
            condition = policy._make_online_condition_tokens(old_fast, new_fast, old_delay, delay)
            scalar_time = torch.tensor([0.6], device=actions.device)
            positions = torch.arange(cfg.chunk_size, device=actions.device)[None, :]
            model_time = torch.where(
                positions < delay[:, None], torch.zeros_like(scalar_time[:, None]), scalar_time[:, None]
            )
            training_path = policy.model.predict_velocity(
                images,
                image_masks,
                tokens,
                token_masks,
                particle,
                model_time,
                fast_adapters=policy.model.fast_adapters,
                fast_tokens=condition,
                fast_delays=delay,
            )
            inference_path = policy.model.denoise_step(
                prefix_pad_masks,
                cache,
                particle,
                scalar_time,
                prefix_embs=prefix_embs,
                fast_adapters=policy.model.fast_adapters,
                fast_tokens=condition,
                fast_delays=delay,
                committed_delays=delay,
            )
            cached_no_fast = policy.model.denoise_step(
                prefix_pad_masks, cache, particle, scalar_time
            )
            suffix = positions >= delay[:, None]
            suffix3 = suffix.unsqueeze(-1).expand_as(inference_path)
            age_sensitivity = {}
            for age_value in (0.1, 0.5, 1.0, 5.0):
                aged = policy.model.denoise_step(
                    prefix_pad_masks,
                    cache,
                    particle,
                    scalar_time,
                    prefix_embs=prefix_embs,
                    fast_adapters=policy.model.fast_adapters,
                    fast_tokens=condition,
                    fast_delays=delay,
                    fast_observation_ages=torch.tensor([age_value], device=actions.device),
                    committed_delays=delay,
                )
                age_sensitivity[str(age_value)] = stats(aged - inference_path, inference_path)
            report[str(delay_value)] = {
                "training_vs_online_inference": stats(inference_path - training_path, training_path),
                "fast_joint_vs_cached_no_fast": stats(inference_path - cached_no_fast, cached_no_fast),
                "fast_joint_vs_cached_no_fast_suffix": stats(
                    (inference_path - cached_no_fast)[suffix3], cached_no_fast[suffix3]
                ),
                "training_velocity_rms": training_path.float().square().mean().sqrt().item(),
                "cached_velocity_rms": cached_no_fast.float().square().mean().sqrt().item(),
                "age_sensitivity_vs_zero": age_sensitivity,
            }

        # Isolate accumulated integration drift at the identity condition (old=new, delay=0).
        zero = torch.zeros(1, dtype=torch.long, device=actions.device)
        fast0 = policy._make_fast_tokens(batch, zero)
        condition0 = policy._make_online_condition_tokens(fast0, fast0, zero, zero)
        joint_particle = particle.clone()
        cached_particle = particle.clone()
        per_step_endpoint = []
        for step in range(cfg.num_inference_steps):
            time = torch.full(
                (1,), 1.0 - step / cfg.num_inference_steps, device=actions.device
            )
            joint_velocity = policy.model.denoise_step(
                prefix_pad_masks,
                cache,
                joint_particle,
                time,
                prefix_embs=prefix_embs,
                fast_adapters=policy.model.fast_adapters,
                fast_tokens=condition0,
                fast_delays=zero,
            )
            cached_velocity = policy.model.denoise_step(
                prefix_pad_masks, cache, cached_particle, time
            )
            joint_particle -= joint_velocity / cfg.num_inference_steps
            cached_particle -= cached_velocity / cfg.num_inference_steps
            per_step_endpoint.append(stats(joint_particle - cached_particle, cached_particle))
        report["identity_rollout"] = {
            "per_step": per_step_endpoint,
            "joint_vs_cached_action": stats(joint_particle - cached_particle, cached_particle),
            "joint_vs_dataset_action": stats(joint_particle - actions, actions),
            "cached_vs_dataset_action": stats(cached_particle - actions, actions),
            "per_action_dimension": {
                "joint_mean": joint_particle[..., : cfg.output_features[ACTION].shape[0]]
                .float()
                .mean(dim=(0, 1))
                .cpu()
                .tolist(),
                "cached_mean": cached_particle[..., : cfg.output_features[ACTION].shape[0]]
                .float()
                .mean(dim=(0, 1))
                .cpu()
                .tolist(),
                "difference_rms": (joint_particle - cached_particle)[
                    ..., : cfg.output_features[ACTION].shape[0]
                ]
                .float()
                .square()
                .mean(dim=(0, 1))
                .sqrt()
                .cpu()
                .tolist(),
                "sign_disagreement_fraction": (
                    joint_particle[..., : cfg.output_features[ACTION].shape[0]].sign()
                    != cached_particle[..., : cfg.output_features[ACTION].shape[0]].sign()
                )
                .float()
                .mean(dim=(0, 1))
                .cpu()
                .tolist(),
            },
        }

        report["fixed_vs_online_rollout"] = {}
        for total_delay in (1, 5, 10):
            schedule = (
                torch.arange(cfg.num_inference_steps, device=actions.device) * total_delay
                // (cfg.num_inference_steps - 1)
            )
            online_particle = particle.clone()
            fixed_particle = hard_commit_action_prefix(
                particle.clone(), actions, torch.tensor([total_delay], device=actions.device)
            )
            final_delay = torch.tensor([total_delay], device=actions.device)
            final_fast = policy._make_fast_tokens(batch, final_delay)
            fixed_condition = policy._make_online_condition_tokens(
                final_fast, final_fast, final_delay, final_delay
            )
            for step in range(cfg.num_inference_steps):
                time = torch.full(
                    (1,), 1.0 - step / cfg.num_inference_steps, device=actions.device
                )
                new_value = int(schedule[step].item())
                old_value = int(schedule[max(step - 1, 0)].item())
                old_delay = torch.tensor([old_value], device=actions.device)
                new_delay = torch.tensor([new_value], device=actions.device)
                old_fast = policy._make_fast_tokens(batch, old_delay)
                new_fast = policy._make_fast_tokens(batch, new_delay)
                online_condition = policy._make_online_condition_tokens(
                    old_fast, new_fast, old_delay, new_delay
                )
                online_velocity = policy.model.denoise_step(
                    prefix_pad_masks,
                    cache,
                    online_particle,
                    time,
                    prefix_embs=prefix_embs,
                    fast_adapters=policy.model.fast_adapters,
                    fast_tokens=online_condition,
                    fast_delays=new_delay,
                    fast_observation_ages=torch.zeros_like(time),
                    committed_delays=new_delay,
                )
                online_particle -= online_velocity / cfg.num_inference_steps
                online_particle = hard_commit_action_prefix(online_particle, actions, new_delay)
                fixed_velocity = policy.model.denoise_step(
                    prefix_pad_masks,
                    cache,
                    fixed_particle,
                    time,
                    prefix_embs=prefix_embs,
                    fast_adapters=policy.model.fast_adapters,
                    fast_tokens=fixed_condition,
                    fast_delays=final_delay,
                    fast_observation_ages=torch.zeros_like(time),
                    committed_delays=final_delay,
                )
                fixed_particle -= fixed_velocity / cfg.num_inference_steps
                fixed_particle = hard_commit_action_prefix(fixed_particle, actions, final_delay)
            free = positions >= final_delay[:, None]
            free3 = free.unsqueeze(-1).expand_as(actions)
            report["fixed_vs_online_rollout"][str(total_delay)] = {
                "fixed_endpoint_vs_dataset": stats(
                    (fixed_particle - actions)[free3], actions[free3]
                ),
                "online_endpoint_vs_dataset": stats(
                    (online_particle - actions)[free3], actions[free3]
                ),
                "online_vs_fixed_endpoint": stats(
                    (online_particle - fixed_particle)[free3], fixed_particle[free3]
                ),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
