"""Replay LIBERO tasks with per-demo restart support and controller provenance.

The canonical images are post-action: flip agent view, retain raw wrist view.
Existing completed files are left untouched by the watcher. New groups retain
source demo IDs and contain pre-action poses/actions for exact label audits.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def encode_frames(group: h5py.Group, key: str, frames: list[np.ndarray]) -> None:
    """Keep the numeric RGB convention of the existing replay JPEG writer."""
    dataset = group.create_dataset(key, shape=(len(frames),), dtype=h5py.vlen_dtype(np.dtype("uint8")))
    for i, frame in enumerate(frames):
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError("JPEG encoding failed")
        dataset[i] = encoded


def process(source: Path, output: Path, max_demos: int | None = None) -> None:
    """Write one task, flushing completed demos for restart after interruption."""
    import libero.libero as resources
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    resources._assets_path_cache = os.environ["LIBERO_ASSETS_PATH"]
    suite_name = source.parent.name
    name = source.stem.removesuffix("_demo")
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = next(suite.get_task(i) for i in range(suite.n_tasks) if suite.get_task(i).name == name)
    if output.exists():
        print(f"complete: {output}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".hdf5.incomplete")
    if temporary.exists():
        try:
            with h5py.File(temporary, "r") as f:
                resumable = f["data"].attrs.get("xvla_replay_version") == 2
        except OSError:
            resumable = False
        if not resumable:
            archive = temporary.with_name(
                temporary.name + ".legacy_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            )
            temporary.rename(archive)
            print(f"preserved legacy partial: {archive}", flush=True)
    env = OffScreenRenderEnv(
        bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
    )
    try:
        with h5py.File(source, "r") as src, h5py.File(temporary, "a") as dst:
            root = dst.require_group("data")
            root.attrs["language_instruction"] = task.language
            root.attrs["source_file"] = str(source.resolve())
            root.attrs["xvla_replay_version"] = 2
            root.attrs["control_freq"] = 20
            root.attrs["image_alignment"] = "post_action"
            root.attrs["agent_orientation"] = "simulator_rotated_180"
            root.attrs["wrist_orientation"] = "simulator_raw"
            root.attrs["renderer"] = os.environ.get("MESA_LOADER_DRIVER_OVERRIDE", "default")
            root.attrs["versions"] = json.dumps(
                {p: importlib.metadata.version(p) for p in ["mujoco", "robosuite", "numpy"]}
            )
            finished = set(json.loads(root.attrs.get("finished_source_demos", "[]")))
            # Only discard our incomplete per-demo write, never a committed demo.
            for key in list(root):
                if key.startswith("_pending_"):
                    del root[key]
            demo_keys = sorted(src["data"], key=lambda x: int(x.split("_")[-1]))
            if max_demos is not None:
                demo_keys = demo_keys[:max_demos]
            for key in demo_keys:
                if key in finished:
                    continue
                # A committed group also serves as a checkpoint if interruption
                # occurred between its atomic rename and the progress attribute.
                if key in root:
                    finished.add(key)
                    continue
                demo = src["data"][key]
                env.seed(0)
                env.reset()
                obs = env.set_init_state(demo["states"][0])
                for _ in range(10):
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                robot = env.env.robots[0]
                controller = robot.controller
                initial_goal = controller.goal_ori.copy()
                goals, images, wrist = [], [], []
                pre_states, pre_sites, relative, indices = [], [], [], []
                previous = None
                done = False
                for index, action in enumerate(demo["actions"][()]):
                    if np.linalg.norm(action[:-1]) < 1e-4 and (
                        previous is None or action[-1] == previous[-1]
                    ):
                        continue
                    previous = action
                    position = np.array(env.sim.data.site_xpos[robot.eef_site_id])
                    body = np.array(env.sim.data.get_body_xmat(robot.robot_model.eef_name))
                    site = np.array(env.sim.data.site_xmat[robot.eef_site_id]).reshape(3, 3)
                    pre_states.append(
                        np.r_[position, Rotation.from_matrix(body).as_rotvec(), obs["robot0_gripper_qpos"]]
                    )
                    pre_sites.append(np.r_[position, site[:, 0], site[:, 1], 0.0, np.zeros(10)])
                    relative.append(action)
                    indices.append(index)
                    obs, _, done, _ = env.step(action.tolist())
                    r = controller.goal_ori
                    goals.append(np.r_[controller.goal_pos, r[:, 0], r[:, 1], action[-1]])
                    images.append(np.flip(obs["agentview_image"], (0, 1)).copy())
                    wrist.append(obs["robot0_eye_in_hand_image"].copy())
                if done and len(goals) >= 52:
                    pending = "_pending_" + key
                    group = root.create_group(pending)
                    group.attrs["source_demo"] = key
                    group.attrs["success"] = True
                    for field, values in [
                        ("abs_action_6d", goals),
                        ("pre_action_state", pre_states),
                        ("pre_action_site_pose", pre_sites),
                        ("relative_action", relative),
                    ]:
                        group.create_dataset(field, data=np.asarray(values, dtype=np.float32))
                    group.create_dataset("source_action_index", data=indices)
                    group.create_dataset("initial_goal_ori", data=initial_goal)
                    group.create_dataset("initial_sim_state", data=demo["states"][0])
                    encode_frames(group, "agentview_rgb_jpeg", images)
                    encode_frames(group, "eye_in_hand_rgb_jpeg", wrist)
                    dst.flush()
                    root.move(pending, key)
                finished.add(key)
                root.attrs["finished_source_demos"] = json.dumps(sorted(finished))
                root.attrs["num_demos"] = len([k for k in root if k.startswith("demo_")])
                dst.flush()
                print(
                    f"{source.name} {key}: success={done} frames={len(goals)} processed={len(finished)}/{len(demo_keys)} kept={root.attrs['num_demos']}",
                    flush=True,
                )
            root.attrs["finished_source_demos"] = json.dumps(sorted(finished))
            root.attrs["num_demos"] = len([key for key in root if key.startswith("demo_")])
            dst.flush()
            if not root.attrs.get("num_demos", 0):
                raise RuntimeError(f"no successful demos: {source}")
        temporary.replace(output)
        print(f"finished: {output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-demos", type=int)
    args = parser.parse_args()
    process(args.source, args.output, args.max_demos)
