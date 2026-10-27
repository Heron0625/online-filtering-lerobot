"""Read-only condition sensitivity diagnosis near demonstrated gripper command transitions."""

# All loop-local closures are evaluated immediately, never retained across iterations.
# ruff: noqa: B023

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data._utils.collate import default_collate

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.common.vla_utils import pad_vector
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.online_filtering_pi05.configuration_online_filtering_pi05 import (
    OnlineFilteringPI05Config,
)
from lerobot.policies.online_filtering_pi05.modeling_online_filtering_pi05 import OnlineFilteringPI05Policy
from lerobot.policies.pi05.online_filtering import DINOv3PatchEncoder, hard_commit_action_prefix
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def select_events(root):
    """Select task-diverse rising/falling command transitions, not inferred physical contacts."""
    candidates = {"rising": [], "falling": []}
    seen = set()
    for path in sorted(Path(root).glob("data/**/*.parquet")):
        table = pq.read_table(path, columns=["index", "episode_index", "task_index", "action"])
        indices = np.asarray(table["index"])
        episodes = np.asarray(table["episode_index"])
        tasks = np.asarray(table["task_index"])
        actions = np.asarray(table["action"].to_pylist())
        gripper = actions[:, -1]
        switches = np.flatnonzero((gripper[1:] * gripper[:-1] < 0) & (episodes[1:] == episodes[:-1])) + 1
        for pos in switches:
            kind = "rising" if gripper[pos] > gripper[pos - 1] else "falling"
            key = (kind, int(tasks[pos]))
            if key in seen or pos < 12 or pos + 50 >= len(indices):
                continue
            if episodes[pos - 12] != episodes[pos + 50]:
                continue
            seen.add(key)
            candidates[kind].append(
                {
                    "event_index": int(indices[pos]),
                    "task_index": int(tasks[pos]),
                    "episode_index": int(episodes[pos]),
                    "kind": kind,
                    "old_gripper": float(gripper[pos - 1]),
                    "new_gripper": float(gripper[pos]),
                }
            )
    events = []
    for rows in candidates.values():
        chosen = np.linspace(0, len(rows) - 1, min(8, len(rows)), dtype=int)
        for j in chosen:
            event = rows[j]
            for flow_step in [3, 8]:
                # Half observe across a command transition; half shortly before it.
                offset = 4 if len(events) // 2 % 2 == 0 else 10
                events.append(
                    dict(
                        event,
                        index=event["event_index"] - offset,
                        event_relative_step=offset,
                        flow_step=flow_step,
                    )
                )
    return events


def main():
    """Run frozen teacher/updater probes and incrementally save measured results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="/data/zhoukai/dataset/libero")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(271828)
    events = select_events(args.dataset_root)
    if not events:
        raise RuntimeError("No eligible gripper transitions found")
    print(json.dumps({"selected_events": events}), flush=True)
    cfg = OnlineFilteringPI05Config.from_pretrained(args.checkpoint)
    cfg.device = "cuda"
    encoder = DINOv3PatchEncoder.from_pretrained("/data/zhoukai/pretrain/dinov3-vits16-pretrain-lvd1689m")
    policy = OnlineFilteringPI05Policy.from_pretrained(
        args.checkpoint, config=cfg, fast_encoder=encoder
    ).eval()
    pre, _ = make_pre_post_processors(
        cfg,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": "/data/zhoukai/pretrain/paligemma-tokenizer"}
        },
    )
    dataset = LeRobotDataset(
        "lerobot/libero",
        root=args.dataset_root,
        video_backend="pyav",
        delta_timestamps={
            **{k: [i / 10 for i in range(11)] for k in [*cfg.image_features, OBS_STATE]},
            ACTION: [i / 10 for i in range(cfg.chunk_size)],
        },
    )
    results = []
    with torch.no_grad():
        for start in range(0, len(events), 4):
            selected = events[start : start + 4]
            batch = pre(default_collate([dataset[e["index"]] for e in selected]))
            size = len(selected)
            old_d = torch.zeros(size, device="cuda", dtype=torch.long)
            new_d = torch.full_like(old_d, 8)
            steps = torch.tensor([e["flow_step"] for e in selected], device="cuda")
            if not policy._observation_validity(batch).gather(1, new_d[:, None]).all():
                raise RuntimeError("Padded observation")
            actions = policy.prepare_action(batch)
            images, image_masks = policy._preprocess_images(policy._initial_slow_batch(batch))
            rows = torch.arange(size, device="cuda")

            def fast_with(image_fresh, state_fresh, delay):
                altered = dict(batch)
                for key in [*cfg.image_features, OBS_STATE]:
                    altered[key] = batch[key].clone()
                    source = new_d if (state_fresh if key == OBS_STATE else image_fresh) else old_d
                    altered[key][rows, delay] = batch[key][rows, source]
                # old delay is zero: keep the slow/reference image fixed rather than overwriting it.
                # The current-at-old-delay tokens below are built directly by the resampler.
                return policy._make_fast_tokens(altered, delay)

            fast = {
                "stale": fast_with(False, False, new_d),
                "image_only": fast_with(True, False, new_d),
                "proprio_only": fast_with(False, True, new_d),
                "fresh": policy._make_fast_tokens(batch, new_d),
            }
            old_fast = policy._make_fast_tokens(batch, old_d)
            initial_pixels, current_pixels = policy._fast_pixels(batch, new_d)
            dtype = policy.model.fast_resampler.feature_proj.weight.dtype
            current_at_old = policy.model.fast_resampler(
                policy.fast_encoder(initial_pixels).to(dtype),
                policy.fast_encoder(current_pixels).to(dtype),
                delays=old_d,
                states=pad_vector(batch[OBS_STATE][rows, new_d], cfg.max_state_dim).to(dtype),
            )
            noise = policy.model.sample_noise(actions.shape, actions.device)
            # Early/late flow probes for each event must share the same noise realization.
            noise[1::2] = noise[::2].clone()

            def pair(da, db, fa, fb):
                return policy._integrate_paired_teacher_particles(
                    images=images,
                    img_masks=image_masks,
                    tokens=batch[OBS_LANGUAGE_TOKENS],
                    masks=batch[OBS_LANGUAGE_ATTENTION_MASK],
                    base_noise=noise,
                    executed_actions=actions,
                    switch_steps=steps,
                    old_delays=da,
                    new_delays=db,
                    old_fast_tokens=fa,
                    new_fast_tokens=fb,
                )

            old, target = pair(old_d, new_d, old_fast, fast["fresh"])
            obs_old, obs_new = pair(new_d, new_d, fast["stale"], fast["fresh"])
            control_old, control_new = pair(old_d, new_d, current_at_old, fast["fresh"])
            common = hard_commit_action_prefix(old, actions, new_d)

            def velocity(x, f, step):
                time = 1 - step.float() / cfg.num_inference_steps
                pos = torch.arange(cfg.chunk_size, device="cuda")[None]
                times = torch.where(pos < new_d[:, None], 0.0, time[:, None])
                return policy.model.predict_velocity(
                    images,
                    image_masks,
                    batch[OBS_LANGUAGE_TOKENS],
                    batch[OBS_LANGUAGE_ATTENTION_MASK],
                    x,
                    times,
                    fast_adapters=policy.model.fast_adapters,
                    fast_tokens=f,
                    fast_delays=new_d,
                )

            def finish(x, f):
                x = hard_commit_action_prefix(x, actions, new_d)
                for k in range(int(steps.min()), cfg.num_inference_steps):
                    v = velocity(x, f, torch.maximum(steps, torch.full_like(steps, k)))
                    x = torch.where((k >= steps)[:, None, None], x - v / cfg.num_inference_steps, x)
                    x = hard_commit_action_prefix(x, actions, new_d)
                return x

            velocities = {name: velocity(common, f, steps) for name, f in fast.items()}
            endpoints = {name: finish(common, f) for name, f in fast.items()}
            prefix, valid, _ = policy.model.embed_prefix(
                images, image_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
            )
            slow = policy.posterior_slow_resampler(prefix.float(), valid)
            prediction = policy.posterior_updater(
                old.float(),
                old_delays=old_d,
                new_delays=new_d,
                flow_times=1 - steps.float() / cfg.num_inference_steps,
                condition_tokens=policy._make_posterior_condition_tokens(slow, old_fast, fast["fresh"]),
                executed_actions=actions.float(),
            )
            final_student = finish(prediction, fast["fresh"])
            final_teacher = finish(target, fast["fresh"])
            for i, event in enumerate(selected):
                mask = torch.arange(cfg.chunk_size, device="cuda") >= 8
                if "action_is_pad" in batch:
                    mask &= ~batch["action_is_pad"][i].bool()
                dim = cfg.output_features[ACTION].shape[0]

                def per_dim(a, b):
                    return (a[i, mask, :dim].float() - b[i, mask, :dim].float()).square().mean(0)

                def mse(a, b):
                    return per_dim(a, b).mean().item()

                result = dict(
                    event,
                    teacher_delta_mse=mse(old, target),
                    observation_only_particle_mse=mse(obs_old, obs_new),
                    delay_commitment_only_particle_mse=mse(control_old, control_new),
                    updater_particle_mse=mse(prediction, target),
                    no_updater_final_teacher_mse=mse(endpoints["fresh"], final_teacher),
                    updater_final_teacher_mse=mse(final_student, final_teacher),
                )
                for name in fast:
                    result[name + "_velocity_change_per_dim"] = per_dim(
                        velocities[name], velocities["stale"]
                    ).tolist()
                    result[name + "_endpoint_change_per_dim"] = per_dim(
                        endpoints[name], endpoints["stale"]
                    ).tolist()
                    result[name + "_endpoint_gt_mse"] = mse(endpoints[name], actions)
                results.append(result)
            report = {
                "checkpoint": args.checkpoint,
                "seed": 271828,
                "samples": results,
                "note": "Gripper command sign transitions are proxies, NOT confirmed grasp/contact events. Observational ablations are sensitivity interventions; no optimizer update. Delay-only includes delay embedding, attention mask and committed prefix jointly. MSE is normalized-action suffix error, not task success.",
            }
            keys = [k for k, v in results[0].items() if isinstance(v, float) and k.endswith("mse")]
            report["aggregate"] = {k: sum(r[k] for r in results) / len(results) for k in keys}
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(report, indent=2))
            print(
                json.dumps(
                    {"completed": len(results), "total": len(events), "aggregate": report["aggregate"]}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
