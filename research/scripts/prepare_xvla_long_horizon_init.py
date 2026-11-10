#!/usr/bin/env python
"""Create a correctly remapped long-horizon X-VLA initialization checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

from lerobot.policies.xvla.utils import remap_action_prefix_positional_embedding


def main() -> None:
    """Create a long-horizon checkpoint while preserving pretrained condition positions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads((args.source / "config.json").read_text())
    if config.get("type") != "xvla":
        raise ValueError(f"expected X-VLA source, got {config.get('type')!r}")
    old_horizon = int(config["chunk_size"])
    if args.chunk_size <= old_horizon:
        raise ValueError("new chunk size must exceed the source horizon")

    args.output.mkdir(parents=True)
    config["chunk_size"] = args.chunk_size
    config["n_action_steps"] = args.chunk_size
    # The official serialized processor uses 50 language tokens.  Its legacy
    # config says 1024, which would exceed max_len_seq if processors are rebuilt.
    config["tokenizer_max_length"] = 50
    # Preserve the official checkpoint's float32 flow integration. Training may
    # use AMP externally, but must not permanently cast the policy to bfloat16.
    config["dtype"] = "float32"
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        shutil.copy2(args.source / name, args.output / name)
    for artifact in args.source.iterdir():
        if artifact.name == "tokenizer" or artifact.name.startswith(
            ("policy_preprocessor_step_", "policy_postprocessor_step_")
        ):
            target = args.output / artifact.name
            target.symlink_to(artifact.resolve(), target_is_directory=artifact.is_dir())

    state = load_file(args.source / "model.safetensors")
    key = "model.transformer.pos_emb"
    if key not in state:
        raise KeyError(f"checkpoint is missing {key}")
    state[key] = remap_action_prefix_positional_embedding(
        state[key], old_horizon=old_horizon, new_horizon=args.chunk_size
    )
    save_file(state, args.output / "model.safetensors")


if __name__ == "__main__":
    main()
