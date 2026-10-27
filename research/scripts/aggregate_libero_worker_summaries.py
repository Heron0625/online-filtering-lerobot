#!/usr/bin/env python
"""Aggregate parallel LIBERO worker summaries into one policy-level summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = sorted(args.input_dir.glob("worker_*/summary_all.json"))
    if not summaries:
        raise FileNotFoundError(f"no worker summary_all.json files found under {args.input_dir}")

    episodes = []
    worker_summaries = []
    base = None
    for path in summaries:
        with path.open() as file:
            summary = json.load(file)
        worker_name = path.parent.name
        worker_summaries.append(
            {
                "worker": worker_name,
                "summary_path": str(path),
                "episodes": len(summary["episodes"]),
                "success_rate": summary["success_rate"],
            }
        )
        for item in summary["episodes"]:
            episodes.append({**item, "worker": worker_name})
        if base is None:
            base = {
                key: value
                for key, value in summary.items()
                if key not in {"episodes", "success_rate"}
            }

    aggregate = {
        **(base or {}),
        "parallel_workers": len(summaries),
        "worker_summaries": worker_summaries,
        "episodes": episodes,
        "success_rate": sum(item["success"] for item in episodes) / len(episodes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        json.dump(aggregate, file, indent=2)
    print(json.dumps({"output": str(args.output), "success_rate": aggregate["success_rate"]}, indent=2))


if __name__ == "__main__":
    main()
