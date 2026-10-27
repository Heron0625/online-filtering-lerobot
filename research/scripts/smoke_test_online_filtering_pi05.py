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
    masked_particle_mse,
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-delay", type=int, default=2)
    parser.add_argument("--fast-num-tokens", type=int, default=64)
    parser.add_argument("--fast-spatial-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage-b-steps", type=int, default=10)
    parser.add_argument(
        "--regenerate-teacher",
        action="store_true",
        help="Regenerate the paired teacher every optimizer step instead of overfitting one cached pair.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--diagnostic-delays", default="1,2,5,10")
    parser.add_argument("--diagnostic-switch-steps", default="1,3,5,7,9")
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
    with (args.dataset_root / "meta" / "stats.json").open() as stats_file:
        stats = json.load(stats_file)
    samples = [dataset[args.sample_index + index] for index in range(args.batch_size)]
    for sample in samples:
        sample[OBS_STATE] = normalize_mean_std(sample[OBS_STATE], stats[OBS_STATE])
        sample[ACTION] = normalize_mean_std(sample[ACTION], stats[ACTION])
    batch = {
        key: torch.stack([sample[key] for sample in samples]).to(args.device)
        for key, value in samples[0].items()
        if isinstance(value, torch.Tensor)
    }
    batch[OBS_LANGUAGE_TOKENS] = torch.zeros(args.batch_size, 200, dtype=torch.long, device=args.device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(1, 200, dtype=torch.bool, device=args.device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = batch[OBS_LANGUAGE_ATTENTION_MASK].expand(args.batch_size, -1)
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
        fast_num_tokens=args.fast_num_tokens,
        fast_spatial_tokens=args.fast_spatial_tokens,
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
        training_stage="stage_b",
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


def main() -> None:
    """Overfit the updater briefly on a fixed online-teacher batch."""
    args = parse_args()
    batch, _ = load_batch(args)
    config = make_config(args)
    policy = OnlineFilteringPI05Policy.from_pretrained(args.pi05_path, config=config, strict=True)
    if args.diagnostic_only:
        policy.eval()
        delays_to_test = [int(value) for value in args.diagnostic_delays.split(",")]
        switch_steps_to_test = [int(value) for value in args.diagnostic_switch_steps.split(",")]
        if any(delay < 1 or delay > args.max_delay for delay in delays_to_test):
            raise ValueError("diagnostic delays must be within [1, max-delay]")
        if any(step < 1 or step >= config.num_inference_steps for step in switch_steps_to_test):
            raise ValueError("diagnostic switch steps must be strictly inside the inference grid")
        initial_batch = policy._initial_slow_batch(batch)
        images, img_masks = policy._preprocess_images(initial_batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = policy.prepare_action(batch)
        torch.manual_seed(args.seed)
        base_noise = policy.model.sample_noise(actions.shape, actions.device)
        action_dim = config.output_features[ACTION].shape[0]
        old_delays = torch.zeros(args.batch_size, dtype=torch.long, device=args.device)
        with torch.inference_mode():
            old_fast_tokens = policy._make_fast_tokens(batch, old_delays)
            for delay in delays_to_test:
                new_delays = torch.full_like(old_delays, delay)
                new_fast_tokens = policy._make_fast_tokens(batch, new_delays)
                suffix_mask = (
                    torch.arange(actions.shape[1], device=actions.device)[None, :] >= delay
                ).expand(args.batch_size, -1).clone()
                if "action_is_pad" in batch:
                    suffix_mask &= ~batch["action_is_pad"].to(dtype=torch.bool)
                for switch_step in switch_steps_to_test:
                    switch_steps = torch.full_like(old_delays, switch_step)
                    old_particle, new_particle = policy._integrate_paired_teacher_particles(
                        images=images,
                        img_masks=img_masks,
                        tokens=tokens,
                        masks=masks,
                        base_noise=base_noise,
                        executed_actions=actions,
                        switch_steps=switch_steps,
                        old_delays=old_delays,
                        new_delays=new_delays,
                        old_fast_tokens=old_fast_tokens,
                        new_fast_tokens=new_fast_tokens,
                    )
                    mse = masked_particle_mse(
                        new_particle.float(), old_particle.float(), suffix_mask, action_dim
                    )
                    print(
                        json.dumps(
                            {
                                "delay": delay,
                                "switch_step": switch_step,
                                "flow_time": 1 - switch_step / config.num_inference_steps,
                                "teacher_delta_mse": mse.item(),
                                "teacher_delta_rms": mse.sqrt().item(),
                            }
                        ),
                        flush=True,
                    )
        return
    policy.train()
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    history = []
    torch.manual_seed(args.seed)
    cached_pair = None if args.regenerate_teacher else policy._make_stage_b_pair(batch)
    if cached_pair is not None:
        # This is a deterministic fixed-target optimization check. Disabling
        # dropout prevents accidentally overfitting one repeated dropout mask;
        # eval mode does not disable gradients for the updater parameters.
        policy.eval()
    for step in range(args.stage_b_steps):
        # Reusing the seed fixes noise, filtration pair, switch time, and updater dropout. This is
        # an intentional overfit check: failure to improve indicates a broken learning path.
        torch.manual_seed(args.seed)
        optimizer.zero_grad(set_to_none=True)
        stage_b_loss, metrics = (
            policy(batch) if cached_pair is None else policy._stage_b_loss_from_pair(cached_pair)
        )
        if not torch.isfinite(stage_b_loss) or stage_b_loss.item() <= 0:
            raise RuntimeError("Stage-B online-teacher loss is invalid")
        stage_b_loss.backward()
        optimizer.step()
        history.append(metrics)
        if (step + 1) % args.log_every == 0 or step == 0 or step + 1 == args.stage_b_steps:
            print(json.dumps({"step": step + 1, **metrics}), flush=True)

    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    updater_gradients = assert_finite_gradients(policy.posterior_updater, "posterior_updater.")
    fast_projection_gradients = assert_finite_gradients(
        policy.posterior_fast_proj, "posterior_fast_proj."
    )
    slow_resampler_gradients = assert_finite_gradients(
        policy.posterior_slow_resampler, "posterior_slow_resampler."
    )
    if policy.posterior_condition_roles.grad is None or not torch.isfinite(
        policy.posterior_condition_roles.grad
    ).all():
        raise RuntimeError("missing or non-finite posterior condition-role gradients")
    frozen_teacher_gradients = sum(
        parameter.grad is not None
        for name, parameter in policy.named_parameters()
        if not name.startswith(
            (
                "posterior_updater.",
                "posterior_fast_proj.",
                "posterior_slow_resampler.",
                "posterior_condition_roles",
            )
        )
    )
    if frozen_teacher_gradients:
        raise RuntimeError("Stage-B produced gradients in the frozen Stage-A teacher")
    condition_ablation = {}
    if cached_pair is not None:
        policy.eval()
        with torch.no_grad():
            _, correct_metrics = policy._stage_b_loss_from_pair(cached_pair)
            stale_pair = dict(cached_pair)
            stale_pair["new_fast_tokens"] = cached_pair["old_fast_tokens"]
            _, stale_metrics = policy._stage_b_loss_from_pair(stale_pair)
            if args.batch_size > 1:
                shuffled_pair = dict(cached_pair)
                shuffled_pair["new_fast_tokens"] = cached_pair["new_fast_tokens"].roll(1, dims=0)
                _, shuffled_metrics = policy._stage_b_loss_from_pair(shuffled_pair)
            else:
                shuffled_metrics = None
        condition_ablation = {
            "correct_condition_bridge_gain": correct_metrics["bridge_gain"],
            "stale_condition_bridge_gain": stale_metrics["bridge_gain"],
            "shuffled_condition_bridge_gain": (
                None if shuffled_metrics is None else shuffled_metrics["bridge_gain"]
            ),
        }
    print(
        json.dumps(
            {
                "stage": "stage_b",
                "initial_endpoint": history[0]["posterior_endpoint_loss"],
                "final_endpoint": history[-1]["posterior_endpoint_loss"],
                "teacher_delta_mse": history[-1]["teacher_delta_mse"],
                "student_delta_mse": history[-1]["student_delta_mse"],
                "bridge_gain": history[-1]["bridge_gain"],
                "delta_cosine_similarity": history[-1]["delta_cosine_similarity"],
                "peak_memory_gb": peak_memory_gb,
                "mean_gap": history[-1]["mean_gap"],
                "updater_gradient_tensors": updater_gradients,
                "fast_projection_gradient_tensors": fast_projection_gradients,
                "slow_resampler_gradient_tensors": slow_resampler_gradients,
                "condition_role_gradients": True,
                "frozen_teacher_gradient_tensors": frozen_teacher_gradients,
                **condition_ablation,
            }
        )
    )


if __name__ == "__main__":
    main()
