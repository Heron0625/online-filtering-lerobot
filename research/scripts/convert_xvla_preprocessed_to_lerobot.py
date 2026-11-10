#!/usr/bin/env python
"""Convert replayed X-VLA LIBERO trajectories to a native 20 Hz LeRobot dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

FEATURES = {
    "observation.images.image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.image2": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {"dtype": "float32", "shape": (20,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (20,), "names": ["action"]},
}


def decode_jpeg(encoded: np.ndarray) -> np.ndarray:
    """Decode the replay writer's numerically RGB JPEG arrays."""
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid JPEG frame")
    # The replay writer intentionally round-trips RGB numerically through
    # OpenCV, so no additional BGR swap is needed here.
    return image


def absolute_action_20d(action_10d: np.ndarray) -> np.ndarray:
    """Pad the single-arm controller goal and convert gripper to binary."""
    result = np.zeros(20, dtype=np.float32)
    result[:9] = action_10d[:9]
    result[9] = float(action_10d[9] > 0)
    return result


def proprio_state_20d(action_10d: np.ndarray) -> np.ndarray:
    """Replay goals are already controller-frame; never reinterpret them as body rotvecs."""
    result = np.zeros(20, dtype=np.float32)
    result[:9] = action_10d[:9]
    return result


def convert(source_root: Path, output_root: Path, repo_id: str, expected_files: int = 40) -> None:
    """Convert complete replay files with already canonical images and rotations."""
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    files = sorted(source_root.glob("libero_*/*.hdf5"))
    if len(files) != expected_files:
        raise RuntimeError(f"expected {expected_files} preprocessed task files, found {len(files)}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        robot_type="panda",
        fps=20,
        features=FEATURES,
        use_videos=True,
        image_writer_threads=8,
        batch_encoding_size=10,
    )
    episode_count = 0
    for path in files:
        with h5py.File(path, "r") as f:
            language = str(f["data"].attrs["language_instruction"])
            demo_keys = sorted(f["data"], key=lambda key: int(key.split("_")[-1]))
            for demo_key in demo_keys:
                demo = f["data"][demo_key]
                absolute = demo["abs_action_6d"][()].astype(np.float32)
                agent = demo["agentview_rgb_jpeg"]
                wrist = demo["eye_in_hand_rgb_jpeg"]
                # Row t represents the post-action-t observation/current goal;
                # its supervised action is the next controller goal.  This
                # matches official X-VLA's abs_trajectory[0] / [1:] slicing.
                for index in range(len(absolute) - 1):
                    dataset.add_frame(
                        {
                            "observation.images.image": decode_jpeg(agent[index]),
                            "observation.images.image2": decode_jpeg(wrist[index]),
                            "observation.state": proprio_state_20d(absolute[index]),
                            "action": absolute_action_20d(absolute[index + 1]),
                            "task": language,
                        }
                    )
                dataset.save_episode()
                episode_count += 1
                if episode_count % 50 == 0:
                    print(f"converted {episode_count} episodes", flush=True)
    dataset.finalize()
    print(f"wrote {episode_count} episodes to {output_root}")


def main() -> None:
    """Parse conversion paths and validate the expected task count."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/xvla_libero_20hz_absolute")
    parser.add_argument("--expected-files", type=int, default=40)
    args = parser.parse_args()
    convert(args.source_root, args.output_root, args.repo_id, args.expected_files)


if __name__ == "__main__":
    main()
