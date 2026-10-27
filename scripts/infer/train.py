"""Cloud training entry point using the same YAML. Never imports robot/ROS code."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

import yaml


def training_args(cfg: dict) -> list[str]:
    train = cfg["training"]
    for key in ("batch_size", "steps", "save_freq", "num_workers"):
        if isinstance(train[key], bool) or not isinstance(train[key], int) or train[key] <= 0:
            raise ValueError(f"training.{key} 必须是正整数")
    return [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={train['repo_id']}",
        f"--dataset.root={train['dataset_root']}",
        "--policy.type=pi05",
        f"--policy.pretrained_path={train['pretrained_path']}",
        "--policy.device=cuda",
        "--policy.dtype=bfloat16",
        "--policy.push_to_hub=false",
        "--policy.gradient_checkpointing=true",
        "--policy.train_expert_only=true",
        "--policy.chunk_size=50",
        f"--policy.n_action_steps={cfg['model']['execute_steps']}",
        # Explicit and carried in saved processors; do not substitute stats at deployment.
        '--policy.normalization_mapping={"STATE":"MEAN_STD","ACTION":"MEAN_STD","VISUAL":"IDENTITY"}',
        f"--output_dir={train['output_dir']}",
        "--job_name=pi05_rb75",
        f"--batch_size={train['batch_size']}",
        f"--steps={train['steps']}",
        f"--save_freq={train['save_freq']}",
        f"--num_workers={train['num_workers']}",
        "--wandb.enable=false",
        "--env_eval_freq=0",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--print", action="store_true", dest="print_only")
    args = parser.parse_args()
    with args.config.open() as handle:
        cfg = yaml.safe_load(handle)
    command = training_args(cfg)
    if args.print_only:
        print(shlex.join(command))
        return
    root = Path(cfg["training"]["dataset_root"]).expanduser()
    with (root / "meta/info.json").open() as handle:
        info = json.load(handle)
    if info["fps"] != cfg["control"]["action_hz"]:
        raise ValueError("数据集 FPS 与 control.action_hz 不一致，先修正推理时间尺度")
    from common import FEATURE_NAMES

    for name in ("observation.state", "action"):
        feature = info["features"][name]
        if feature["shape"] != [16] or feature.get("names") != FEATURE_NAMES:
            raise ValueError(f"{name} 的维度/关节顺序与 RB75 数采不一致")
    for name in cfg["cameras"]:
        if f"observation.images.{name}" not in info["features"]:
            raise ValueError(f"训练数据缺少相机 {name}")
    # No shell interpolation, and no local ROS imports or drivers.
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
