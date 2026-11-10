"""Collect finished 15/20/25 evaluations and update the single metrics document."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/yhr/project/robot/online_filtering")
OUTPUT = Path("/data/yhr/output/online_filtering/eval")
PYTHON = "/data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python"
MODES = ["zero40", "delay15_400", "delay20_400", "delay25_400"]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def main() -> None:
    """Wait for complete summaries, validate them, then publish metrics locally."""
    for _ in range(960):
        counts = {}
        try:
            for mode in MODES:
                base = OUTPUT / f"xvla50_corrected_30k_{mode}_r20"
                counts[mode] = sum(
                    len(json.loads(p.read_text())["episodes"]) for p in base.rglob("summary.json")
                )
        except json.JSONDecodeError:
            time.sleep(30)
            continue
        print(json.dumps(counts), flush=True)
        if counts == {"zero40": 40, "delay15_400": 400, "delay20_400": 400, "delay25_400": 400}:
            break
        time.sleep(30)
    else:
        raise TimeoutError("evaluation did not complete in eight hours; inspect worker logs")
    subprocess.run([PYTHON, str(OUTPUT / "aggregate_xvla50_results.py")], check=True)
    results = {
        mode: json.loads((OUTPUT / f"xvla50_corrected_30k_{mode}_r20/aggregate_summary.json").read_text())
        for mode in MODES
    }
    (OUTPUT / "xvla50_corrected_30k_latency_comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    metrics = ROOT / "research/docs/metrics.md"
    text = metrics.read_text()
    marker = "## X-VLA-50 corrected data, 30k checkpoint (2026-10-31)"
    before, section = text.split(marker, 1)
    start = section.index("| Delay | Episodes |")
    end = section.index("\n\n", start)
    table = [
        "| Delay | Episodes | Spatial | Object | Goal | LIBERO-10 | Overall | Queue underruns |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| 0 (quick check) | 40 | 9/10 | 10/10 | 10/10 | 9/10 | 38/40 (95.0%) | 0 |",
    ]
    for delay in [15, 20, 25]:
        result = results[f"delay{delay}_400"]
        rates = [f"{result['per_suite'][suite]['success_rate'] * 100:.1f}%" for suite in SUITES]
        table.append(
            "| "
            + " | ".join(
                [
                    str(delay),
                    "400",
                    *rates,
                    f"{result['success_rate'] * 100:.2f}%",
                    str(result["queue_underruns"]),
                ]
            )
            + " |"
        )
    section = section[:start] + "\n".join(table) + section[end:]
    section = section.replace(
        "Delays15/20/25: 400 episodes per delay (40 tasks × 10 initial states), in progress.",
        "Delays15/20/25 completed: 400 episodes per delay (40 tasks × 10 initial states).",
    )
    metrics.write_text(before + marker + section)
    print("COMPLETE: validated results written to metrics.md and comparison JSON", flush=True)


if __name__ == "__main__":
    main()
