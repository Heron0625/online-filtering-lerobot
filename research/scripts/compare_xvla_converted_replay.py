"""Compare corrected LeRobot targets with existing independently replayed HDF5 goals.

Uses the trajectory correspondences from diagnose_xvla_data_alignment.py.
These replay files have no source demo IDs, so the original match is approximate;
a same-state replay is required to separate conversion from rollout differences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def summarize(values: np.ndarray) -> dict:
    """Summarize per-step errors without hiding the tail."""
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "count": len(values),
    }


def compare(converted: Path, replay: Path, audit: Path, output: Path) -> None:
    """Compare every previously matched episode using fixed correspondences."""
    frames = pd.concat([pd.read_parquet(p) for p in sorted((converted / "data").rglob("*.parquet"))])
    episodes = {int(k): g.sort_values("frame_index") for k, g in frames.groupby("episode_index")}
    results = []
    for path in sorted(replay.glob("libero_*/*.hdf5")):
        task = path.stem.removesuffix("_demo")
        correspondence = audit / (task + ".json")
        if not correspondence.exists():
            continue
        with h5py.File(path) as f:
            positions = []
            angles = []
            rotation6 = []
            zero_angles = []
            grip_errors = []
            per_episode = []
            matches = json.loads(correspondence.read_text())
            for match in matches:
                predicted = np.stack(episodes[match["episode"]]["action"])
                target = f["data"][match["replay_demo"]]["abs_action_6d"][()]
                if len(predicted) != len(target):
                    raise ValueError("trajectory length changed after conversion")

                def matrix(x):
                    return np.stack((x[:, 3:6], x[:, 6:9], np.cross(x[:, 3:6], x[:, 6:9])), axis=-1)

                error = np.rad2deg(
                    Rotation.from_matrix(matrix(predicted) @ matrix(target).transpose(0, 2, 1)).magnitude()
                )
                distance = np.linalg.norm(predicted[:, :3] - target[:, :3], axis=-1)
                positions.extend(distance.tolist())
                per_episode.append(
                    {
                        "episode": match["episode"],
                        "raw_demo": match["raw_demo"],
                        "replay_demo": match["replay_demo"],
                        "position_l2_m": summarize(distance),
                        "rotation_deg": summarize(error),
                        "worst_rotation_frame": int(np.argmax(error)),
                        "worst_position_frame": int(np.argmax(distance)),
                    }
                )
                angles.extend(error.tolist())
                rotation6.extend(np.abs(predicted[:, 3:9] - target[:, 3:9]).mean(-1).tolist())
                grip_errors.extend((predicted[:, 9] != (target[:, 9] > 0)).tolist())
                # Repeated predicted goals exercise the history-retention path.
                repeated = np.r_[False, np.max(abs(np.diff(predicted[:, 3:9], axis=0)), axis=-1) == 0]
                zero_angles.extend(error[repeated].tolist())
            result = {
                "task": task,
                "episodes": len(matches),
                "position_l2_m": summarize(np.array(positions)),
                "rotation_deg": summarize(np.array(angles)),
                "rotation6d_mae": summarize(np.array(rotation6)),
                "gripper_mismatches": int(np.sum(grip_errors)),
                "per_episode": per_episode,
            }
            if zero_angles:
                result["repeated_goal_rotation_deg"] = summarize(np.array(zero_angles))
            results.append(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=Path("research/outputs/xvla_data_audit"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/outputs/xvla_data_audit/corrected_replay_comparison.json"),
    )
    args = parser.parse_args()
    compare(args.converted, args.replay, args.audit, args.output)
