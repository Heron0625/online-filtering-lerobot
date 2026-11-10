"""Audit local LIBERO trajectories against raw and replayed X-VLA HDF5 files.

Run from the repository root with PYTHONPATH=src. Local dataset paths below
match the 2026-10-30 handoff. Replay matches use length, full gripper sequence,
and nearest position goals; they are not source-demo IDs.
"""

import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from lerobot.policies.xvla.utils import libero_relative_action_to_xvla_ee6d

root = Path("/data/yhr/pretrain/recovered_zhoukai/dataset/libero")
base = Path("/data/yhr/pretrain/xvla_official")
df = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))], ignore_index=True)
tasks = pd.read_parquet(root / "meta/tasks.parquet")
episodes = {int(k): g.sort_values("frame_index") for k, g in df.groupby("episode_index")}
report = []
matches = []
for path in sorted((base / "libero_hdf5_raw").rglob("*.hdf5")):
    task = path.stem.removesuffix("_demo")
    language = re.sub(r"^(?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+_", "", task)
    language = language.replace("_", " ")
    if language not in tasks.index:
        continue
    ti = int(tasks.loc[language, "task_index"])
    eps = [(ei, g) for ei, g in episodes.items() if int(g.task_index.iloc[0]) == ti]
    with h5py.File(path) as f:
        raws = []
        for key in f["data"]:
            g = f["data"][key]
            a = g["actions"][()]
            keep = []
            prev = None
            for i, x in enumerate(a):
                if np.linalg.norm(x[:-1]) < 1e-4 and (prev is None or x[-1] == prev[-1]):
                    continue
                keep.append(i)
                prev = x
            raws.append((key, a[keep], g["obs/ee_states"][()][keep], np.array(keep)))
        count = 0
        errors = []
        for ei, g in eps:
            a = np.stack(g.action)
            s = np.stack(g["observation.state"])
            candidates = [
                (k, b, rs, ix) for k, b, rs, ix in raws if b.shape == a.shape and np.max(np.abs(b - a)) < 1e-6
            ]
            if len(candidates) != 1:
                continue
            k, b, rs, ix = candidates[0]
            count += 1
            errors.append(float(np.mean(np.abs(s[:, :6] - rs))))
            matches.append(
                {
                    "task": task,
                    "language": language,
                    "episode": ei,
                    "raw": str(path),
                    "demo": k,
                    "rows": len(a),
                    "raw_state_mae": errors[-1],
                }
            )
        report.append(
            {
                "task": task,
                "lerobot_episodes": len(eps),
                "action_exact_matches": count,
                "raw_state_mae_mean": float(np.mean(errors)) if errors else None,
            }
        )
print(json.dumps(report, indent=2))
Path("research/outputs/xvla_data_audit/raw_matches.json").write_text(json.dumps(matches, indent=2))
Path("research/outputs/xvla_data_audit/raw_summary.json").write_text(json.dumps(report, indent=2))
# Compare recovered relative+state targets with independently replayed controller goals.
for pp in sorted((base / "libero_xvla_preprocessed").rglob("*.hdf5")):
    task = pp.stem.removesuffix("_demo")
    m = [x for x in matches if x["task"] == task]
    if not m:
        continue
    with h5py.File(pp) as f:
        goals = [(k, f["data"][k]["abs_action_6d"][()]) for k in f["data"]]
        results = []
        for x in m:
            g = episodes[x["episode"]]
            a = np.stack(g.action)
            s = np.stack(g["observation.state"])
            pred = libero_relative_action_to_xvla_ee6d(torch.tensor(s), torch.tensor(a)).numpy()
            candidates = []
            for k, b in goals:
                if len(b) != len(a) or not np.array_equal(b[:, 9] > 0, a[:, 6] > 0):
                    continue
                err = np.mean(np.abs(b[:, :3] - pred[:, :3]))
                candidates.append((float(err), k, b))
            if not candidates:
                continue
            err, k, b = min(candidates, key=lambda z: z[0])
            p = pred[:, :3]
            q = b[:, :3]
            rot = Rotation.from_rotvec(s[:, 3:6]).as_matrix()
            delta = Rotation.from_rotvec(0.5 * np.clip(a[:, 3:6], -1, 1)).as_matrix()
            legacy = delta @ rot
            legacy6 = np.concatenate([legacy[:, :, 0], legacy[:, :, 1]], axis=-1)
            results.append(
                {
                    "episode": x["episode"],
                    "raw_demo": x["demo"],
                    "replay_demo": k,
                    "rows": len(a),
                    "position_mae": float(np.mean(abs(p - q))),
                    "position_max": float(np.max(abs(p - q))),
                    "corrected_rotation6d_mae": float(np.mean(abs(pred[:, 3:9] - b[:, 3:9]))),
                    "rotation6d_mae": float(np.mean(abs(legacy6 - b[:, 3:9]))),
                    "proprio_vs_previous_goal_mae": float(np.mean(abs(s[1:, :3] - q[:-1]))),
                    "position_shift_minus1_mae": float(np.mean(abs(p[1:] - q[:-1]))),
                    "position_shift_plus1_mae": float(np.mean(abs(p[:-1] - q[1:]))),
                }
            )
        print("REPLAY", task, json.dumps(results[:5], indent=2), "MATCHED", len(results))
        Path(f"research/outputs/xvla_data_audit/{task}.json").write_text(json.dumps(results, indent=2))
