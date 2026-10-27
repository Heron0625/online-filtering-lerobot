#!/usr/bin/env python
"""Create a lightweight online-filtering init directory from an XVLA checkpoint."""

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    """Convert checkpoint metadata while reusing the immutable weights through a symlink."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-stage", choices=("stage_a", "stage_b"), default="stage_a")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    with (args.source / "config.json").open() as stream:
        config = json.load(stream)
    if config.get("type") != "xvla":
        raise ValueError(f"expected an xvla checkpoint, got {config.get('type')!r}")
    config["type"] = "online_filtering_xvla"
    config.update(
        training_stage=args.training_stage,
        max_inference_delay=25,
        online_flow_ema_decay=0.999,
        online_flow_rollout_steps=9,
        dinov3_model_id="/data/yhr/pretrain/dinov3-vits16-pretrain-lvd1689m",
        freeze_dinov3=True,
        fast_num_tokens=64,
        fast_adapter_layers=[3, 7, 11, 15, 19, 23],
        slow_visual_dropout_probability=0.0,
        slow_visual_dropout_final_probability=0.25,
        slow_visual_dropout_anneal_start_step=500,
        slow_visual_dropout_anneal_steps=2000,
        freeze_vision_encoder=False,
        freeze_language_encoder=False,
    )
    with (args.output / "config.json").open("w") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")

    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        shutil.copy2(args.source / name, args.output / name)
    os.symlink((args.source / "model.safetensors").resolve(), args.output / "model.safetensors")


if __name__ == "__main__":
    main()
