#!/usr/bin/env python
"""Copy parallel LIBERO worker outputs into a single suite/task tree."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them. By default worker outputs are preserved.",
    )
    return parser.parse_args()


def copy_or_move(src: Path, dst: Path, *, move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    worker_dirs = sorted(args.input_dir.glob("worker_*"))
    if not worker_dirs:
        raise FileNotFoundError(f"no worker_* directories found under {args.input_dir}")

    count = 0
    for worker_dir in worker_dirs:
        for path in worker_dir.glob("libero_*/task_*/*"):
            if not path.is_file():
                continue
            rel = path.relative_to(worker_dir)
            copy_or_move(path, args.input_dir / rel, move=args.move)
            count += 1

    print(f"consolidated {count} files under {args.input_dir}")


if __name__ == "__main__":
    main()
