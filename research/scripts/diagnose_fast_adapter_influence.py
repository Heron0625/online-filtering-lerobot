#!/usr/bin/env python
"""Trace fast-condition influence through every action-expert adapter layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

from diagnose_critical_full_replan import select_critical_events
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/zhoukai/dataset/libero"))
    parser.add_argument("--delay", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.set_num_threads(4)
    events = select_critical_events(args.dataset_root, args.delay, args.batch_size)[: args.batch_size]
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
            **{key: [i / 10 for i in range(cfg.max_inference_delay + 1)] for key in [*cfg.image_features, OBS_STATE]},
            ACTION: [i / 10 for i in range(cfg.chunk_size)],
        },
    )
    batch = preprocessor(default_collate([dataset[event["index"]] for event in events]))
    size = len(events)

    with torch.no_grad():
        actions = policy.prepare_action(batch)
        initial = policy._initial_slow_batch(batch)
        images, image_masks = policy._preprocess_images(initial)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        token_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        delays = torch.full((size,), args.delay, device=actions.device, dtype=torch.long)
        schedule = (
            torch.arange(cfg.num_inference_steps, device=actions.device)[None, :] * delays[:, None]
            // (cfg.num_inference_steps - 1)
        )
        fast = {
            step: policy._make_fast_tokens(batch, schedule[:, step])
            for step in range(cfg.num_inference_steps)
        }
        final_step = cfg.num_inference_steps - 1
        old_step = final_step - 1
        old_delays = schedule[:, old_step]
        new_delays = schedule[:, final_step]
        rows = torch.arange(size, device=actions.device)
        stale_batch = dict(batch)
        for key in [*cfg.image_features, OBS_STATE]:
            stale_batch[key] = batch[key].clone()
            stale_batch[key][rows, new_delays] = batch[key][rows, old_delays]
        stale_new = policy._make_fast_tokens(stale_batch, new_delays)

        torch.manual_seed(8675309)
        particle = policy.model.sample_noise(actions.shape, actions.device)
        committed = torch.zeros_like(particle)
        action_dim = cfg.output_features[ACTION].shape[0]
        committed[..., :action_dim] = actions[..., :action_dim]
        # Reach the last inference point under the real correct-condition rollout.
        for step in range(final_step):
            previous = max(step - 1, 0)
            step_old_delays = schedule[:, previous]
            step_new_delays = schedule[:, step]
            particle = hard_commit_action_prefix(particle, committed, step_new_delays)
            condition = policy._make_online_condition_tokens(
                fast[previous], fast[step], step_old_delays, step_new_delays
            )
            time = torch.full(
                (size,), 1.0 - step / cfg.num_inference_steps, device=actions.device
            )
            velocity = policy.model.predict_velocity(
                images,
                image_masks,
                tokens,
                token_masks,
                particle,
                torch.where(
                    torch.arange(cfg.chunk_size, device=actions.device)[None, :]
                    < step_new_delays[:, None],
                    torch.zeros_like(time[:, None]),
                    time[:, None],
                ),
                fast_adapters=policy.model.fast_adapters,
                fast_tokens=condition,
                fast_delays=step_new_delays,
            )
            particle = particle - velocity / cfg.num_inference_steps
            particle = hard_commit_action_prefix(particle, committed, step_new_delays)

        particle = hard_commit_action_prefix(particle, committed, new_delays)
        time = torch.full((size,), 0.1, device=actions.device)
        positions = torch.arange(cfg.chunk_size, device=actions.device)[None, :]
        model_time = torch.where(
            positions < new_delays[:, None], torch.zeros_like(time[:, None]), time[:, None]
        )
        conditions = {
            "correct": policy._make_online_condition_tokens(
                fast[old_step], fast[final_step], old_delays, new_delays
            ),
            "stale": policy._make_online_condition_tokens(
                fast[old_step], stale_new, old_delays, new_delays
            ),
        }

        traces: dict[str, dict[str, dict]] = {}

        def run(label: str, condition: torch.Tensor, use_adapters: bool = True) -> torch.Tensor:
            current: dict[str, dict] = {}
            handles = []

            def make_hook(layer_name: str):
                def hook(module, hook_args, hook_kwargs, output):
                    action_hidden = hook_args[0]
                    correction = output - action_hidden
                    flow_times = hook_kwargs["flow_times"]
                    adapter_delays = hook_kwargs["delays"]
                    ages = hook_kwargs.get("observation_ages")
                    if ages is None:
                        ages = torch.zeros_like(flow_times)
                    relative_position = (
                        positions - adapter_delays.unsqueeze(1)
                    ).clamp(min=0).to(action_hidden.dtype)
                    relative_position = relative_position / (
                        cfg.chunk_size - adapter_delays
                    ).clamp(min=1).unsqueeze(1).to(action_hidden.dtype)
                    gate_inputs = torch.stack(
                        [
                            flow_times.to(action_hidden.dtype).unsqueeze(1).expand(-1, cfg.chunk_size),
                            relative_position,
                            ages.to(action_hidden.dtype).unsqueeze(1).expand(-1, cfg.chunk_size),
                        ],
                        dim=-1,
                    )
                    gate = torch.sigmoid(module.gate_mlp(gate_inputs)).float()
                    current[layer_name] = {
                        "hidden_rms": action_hidden.float().square().mean().sqrt().item(),
                        "correction_rms": correction.float().square().mean().sqrt().item(),
                        "relative_correction_rms": (
                            correction.float().square().mean().sqrt()
                            / action_hidden.float().square().mean().sqrt().clamp_min(1e-12)
                        ).item(),
                        "gate_mean": gate.mean().item(),
                        "gate_min": gate.min().item(),
                        "gate_max": gate.max().item(),
                        "correction_tensor": correction.float().cpu(),
                    }

                return hook

            if use_adapters:
                for layer_name, adapter in policy.model.fast_adapters.items():
                    handles.append(adapter.register_forward_hook(make_hook(layer_name), with_kwargs=True))
            velocity = policy.model.predict_velocity(
                images,
                image_masks,
                tokens,
                token_masks,
                particle,
                model_time,
                fast_adapters=policy.model.fast_adapters if use_adapters else None,
                fast_tokens=condition,
                fast_delays=new_delays,
            )
            for handle in handles:
                handle.remove()
            traces[label] = current
            return velocity

        correct_velocity = run("correct", conditions["correct"])
        stale_velocity = run("stale", conditions["stale"])
        no_fast_velocity = run("no_fast", conditions["correct"], use_adapters=False)

        suffix = positions >= new_delays[:, None]

        def masked_rms(value: torch.Tensor) -> float:
            selected = value[..., :action_dim][suffix]
            return selected.float().square().mean().sqrt().item()

        layer_report = {}
        for layer_name in policy.model.fast_adapters:
            correct_row = traces["correct"][layer_name]
            stale_row = traces["stale"][layer_name]
            delta = correct_row.pop("correction_tensor") - stale_row.pop("correction_tensor")
            layer_report[layer_name] = {
                "correct": correct_row,
                "stale": stale_row,
                "correct_stale_correction_delta_rms": delta.square().mean().sqrt().item(),
            }

        correct_tokens = conditions["correct"].float()
        stale_tokens = conditions["stale"].float()
        normalized_shape = (correct_tokens.shape[-1],)
        # Measure the content contrast created by normalization in stable FP32;
        # the learned affine is shared by both conditions and is not needed here.
        normalized_correct = F.layer_norm(correct_tokens, normalized_shape)
        normalized_stale = F.layer_norm(stale_tokens, normalized_shape)
        report = {
            "checkpoint": str(args.checkpoint),
            "events": events,
            "probe_step": final_step,
            "flow_time": 0.1,
            "old_delay": int(old_delays[0]),
            "new_delay": int(new_delays[0]),
            "token_rms": correct_tokens.square().mean().sqrt().item(),
            "token_correct_stale_delta_rms": (
                correct_tokens - stale_tokens
            ).square().mean().sqrt().item(),
            "normalized_token_correct_stale_delta_rms": (
                normalized_correct.float() - normalized_stale.float()
            ).square().mean().sqrt().item(),
            "velocity_rms": masked_rms(correct_velocity),
            "correct_stale_velocity_delta_rms": masked_rms(correct_velocity - stale_velocity),
            "correct_no_fast_velocity_delta_rms": masked_rms(correct_velocity - no_fast_velocity),
            "layers": layer_report,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
