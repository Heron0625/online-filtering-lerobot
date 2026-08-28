#!/usr/bin/env python

"""One-batch Stage-A/Stage-B smoke test using local LIBERO and pretrained weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.online_filtering_pi05.configuration_online_filtering_pi05 import (
    OnlineFilteringPI05Config,
)
from lerobot.policies.online_filtering_pi05.modeling_online_filtering_pi05 import (
    OnlineFilteringPI05Policy,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def parse_args() -> argparse.Namespace:
    """Parse local artifact and smoke-test settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pi05-path", type=Path, required=True)
    parser.add_argument("--dinov3-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-index", type=int, default=100)
    parser.add_argument("--max-delay", type=int, default=2)
    parser.add_argument("--teacher-steps", type=int, default=1)
    parser.add_argument("--video-backend", default="pyav")
    return parser.parse_args()


def normalize_mean_std(value: torch.Tensor, stats: dict) -> torch.Tensor:
    """Apply the LIBERO mean/std normalization used by PI0.5 finetuning."""
    mean = torch.tensor(stats["mean"], dtype=value.dtype)
    std = torch.tensor(stats["std"], dtype=value.dtype)
    return (value - mean) / (std + 1e-8)


def load_batch(args: argparse.Namespace) -> tuple[dict[str, torch.Tensor], dict]:
    """Decode one real LIBERO sample with future observations and action chunk."""
    fps = 10
    delta_timestamps = {
        "observation.images.image": [index / fps for index in range(args.max_delay + 1)],
        "observation.images.image2": [index / fps for index in range(args.max_delay + 1)],
        OBS_STATE: [index / fps for index in range(args.max_delay + 1)],
        ACTION: [index / fps for index in range(50)],
    }
    dataset = LeRobotDataset(
        "lerobot/libero",
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        return_uint8=True,
        video_backend=args.video_backend,
    )
    sample = dataset[args.sample_index]
    with (args.dataset_root / "meta" / "stats.json").open() as stats_file:
        stats = json.load(stats_file)
    sample[OBS_STATE] = normalize_mean_std(sample[OBS_STATE], stats[OBS_STATE])
    sample[ACTION] = normalize_mean_std(sample[ACTION], stats[ACTION])
    batch = {
        key: value.unsqueeze(0).to(args.device) if isinstance(value, torch.Tensor) else value
        for key, value in sample.items()
    }
    batch[OBS_LANGUAGE_TOKENS] = torch.zeros(1, 200, dtype=torch.long, device=args.device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(
        1, 200, dtype=torch.bool, device=args.device
    )
    return batch, stats


def make_config(args: argparse.Namespace) -> OnlineFilteringPI05Config:
    """Build the small-delay local-weight smoke configuration."""
    return OnlineFilteringPI05Config(
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        pretrained_path=args.pi05_path,
        dinov3_model_id=str(args.dinov3_path),
        max_inference_delay=args.max_delay,
        teacher_integration_steps=args.teacher_steps,
        posterior_identity_dropout=0.0,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        dtype="bfloat16",
        device=args.device,
        gradient_checkpointing=True,
        train_expert_only=True,
        freeze_dinov3=True,
    )


def assert_finite_gradients(module: torch.nn.Module, prefix: str) -> int:
    """Validate and count gradient-bearing parameter tensors."""
    count = 0
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"non-finite gradient in {prefix}{name}")
        count += 1
    if count == 0:
        raise RuntimeError(f"no gradients found for {prefix}")
    return count


@torch.no_grad()
def apply_adapter_smoke_step(module: torch.nn.Module, learning_rate: float = 1e-3) -> None:
    """Make the Stage-A teacher condition-sensitive before testing shared-noise Stage B."""
    for parameter in module.parameters():
        if parameter.grad is not None:
            parameter.add_(parameter.grad, alpha=-learning_rate)


def main() -> None:
    """Run Stage-A and Stage-B forward/backward checks."""
    args = parse_args()
    batch, _ = load_batch(args)
    config = make_config(args)
    policy = OnlineFilteringPI05Policy.from_pretrained(args.pi05_path, config=config, strict=True)
    policy.train()

    stage_a_loss, stage_a_metrics = policy(batch)
    if not torch.isfinite(stage_a_loss):
        raise RuntimeError("Stage-A loss is not finite")
    stage_a_loss.backward()
    stage_a_gradients = assert_finite_gradients(policy.model.fast_adapters, "fast_adapters.")
    print(
        json.dumps(
            {
                "stage": "stage_a",
                "loss": stage_a_loss.item(),
                "mean_delay": stage_a_metrics["mean_delay"],
                "adapter_gradient_tensors": stage_a_gradients,
            }
        )
    )

    apply_adapter_smoke_step(policy.model.fast_adapters)
    policy.zero_grad(set_to_none=True)
    config.training_stage = "stage_b"
    policy._freeze_stage_a_teacher()
    policy.train()
    stage_b_loss, stage_b_metrics = policy(batch)
    if not torch.isfinite(stage_b_loss):
        raise RuntimeError("Stage-B loss is not finite")
    if stage_b_loss.item() <= 0:
        raise RuntimeError("Stage-B bridge target is still identity after the adapter smoke step")
    stage_b_loss.backward()
    updater_gradients = assert_finite_gradients(policy.posterior_updater, "posterior_updater.")
    frozen_teacher_gradients = sum(
        parameter.grad is not None
        for name, parameter in policy.named_parameters()
        if not name.startswith(
            ("posterior_updater.", "posterior_fast_proj.", "posterior_slow_resampler.")
        )
    )
    if frozen_teacher_gradients:
        raise RuntimeError("Stage-B produced gradients in the frozen Stage-A teacher")
    print(
        json.dumps(
            {
                "stage": "stage_b",
                "loss": stage_b_loss.item(),
                "endpoint": stage_b_metrics["posterior_endpoint_loss"],
                "mean_gap": stage_b_metrics["mean_gap"],
                "updater_gradient_tensors": updater_gradients,
                "frozen_teacher_gradient_tensors": frozen_teacher_gradients,
            }
        )
    )


if __name__ == "__main__":
    main()
