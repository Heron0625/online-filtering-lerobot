"""Convert complete LIBERO LeRobot episodes to controller-frame X-VLA targets.

Videos and their 10 fps storage timebase are preserved. They contain consecutive
20 Hz control samples. Wrist orientation is explicitly tagged for the X-VLA
processor; simulator and already canonical replay images have no such tag.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.policies.xvla.processor_xvla import XVLA_WRIST_FLIP_KEY
from lerobot.policies.xvla.utils import libero_relative_action_to_xvla_ee6d, libero_state_to_xvla_proprio


def numeric_stats(values: np.ndarray) -> dict:
    """Compute feature statistics for the converted numeric columns."""
    result = {
        name: fn(values, axis=0).tolist()
        for name, fn in [("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std)]
    }
    result["count"] = [len(values)]
    for name, q in [("q01", 0.01), ("q10", 0.1), ("q50", 0.5), ("q90", 0.9), ("q99", 0.99)]:
        result[name] = np.quantile(values, q, axis=0).tolist()
    return result


def convert(source: Path, output: Path) -> None:
    """Write controller-frame targets without modifying the source dataset."""
    if output.exists():
        raise FileExistsError(output)
    info = json.loads((source / "meta/info.json").read_text())
    if info["features"]["action"]["shape"] != [7] or info["features"]["observation.state"]["shape"] != [8]:
        raise ValueError("expected original LIBERO relative 7D actions and body-frame 8D state")
    files = sorted((source / "data").rglob("*.parquet"))
    frames = [pd.read_parquet(path) for path in files]
    sizes = [len(frame) for frame in frames]
    data = pd.concat(frames, ignore_index=True)
    states = np.stack(data["observation.state"]).astype(np.float32)
    relative = np.stack(data["action"]).astype(np.float32)
    actions = np.empty((len(data), 20), dtype=np.float32)
    valid = np.ones(len(data), dtype=bool)
    proprio = libero_state_to_xvla_proprio(torch.from_numpy(states)).numpy()
    for episode, group in data.groupby("episode_index"):
        group = group.sort_values("frame_index")
        indices = group.index.to_numpy()
        if not np.array_equal(group.frame_index.to_numpy(), np.arange(len(group))):
            raise ValueError(f"episode {episode} has missing or duplicated frames")
        actions[indices] = libero_relative_action_to_xvla_ee6d(
            torch.from_numpy(states[indices]), torch.from_numpy(relative[indices])
        ).numpy()
        nonzero = np.flatnonzero((relative[indices, 3:6] != 0).any(axis=-1))
        leading = int(nonzero[0]) if len(nonzero) else len(indices)
        valid[indices[:leading]] = False
    output.mkdir(parents=True)
    shutil.copytree(source / "meta", output / "meta")
    (output / "videos").symlink_to((source / "videos").resolve(), target_is_directory=True)
    data["action"] = list(actions)
    data["observation.state"] = list(proprio)
    data[XVLA_WRIST_FLIP_KEY] = True
    data["observation.xvla_action_valid"] = valid
    start = 0
    for path, size in zip(files, sizes, strict=True):
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        data.iloc[start : start + size].to_parquet(target, index=False)
        start += size
    for key in ["action", "observation.state"]:
        info["features"][key]["shape"] = [20]
        info["features"][key]["names"] = ["xvla_controller_ee6d"]
    info["features"][XVLA_WRIST_FLIP_KEY] = {"dtype": "bool", "shape": [1], "names": ["flip180"]}
    info["features"]["observation.xvla_action_valid"] = {
        "dtype": "bool",
        "shape": [1],
        "names": ["target_valid"],
    }
    provenance = {
        "version": 2,
        "source": str(source.resolve()),
        "physical_control_hz": 20,
        "state_frame": "panda_grip_site",
        "zero_rotation": "retain_previous_episode_goal",
        "images": "original_videos_with_explicit_wrist_flip_flag",
        "initial_goal": "first_measured_site_orientation",
        "masked_leading_targets": int((~valid).sum()),
    }
    (output / "meta/xvla_conversion.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    stats = json.loads((output / "meta/stats.json").read_text())
    stats["action"] = numeric_stats(actions)
    stats["observation.state"] = numeric_stats(proprio)
    stats[XVLA_WRIST_FLIP_KEY] = numeric_stats(np.ones((len(data), 1), dtype=np.float32))
    stats["observation.xvla_action_valid"] = numeric_stats(valid[:, None].astype(np.float32))
    (output / "meta/stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "episodes": int(data.episode_index.nunique()),
                "frames": len(data),
                "zero_rotation_commands": int((relative[:, 3:6] == 0).all(-1).sum()),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source, args.output)
