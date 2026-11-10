"""Evaluate XVLA latency with bounded GPU workers and aggregate complete results.

Each work item loads one model and evaluates two tasks (10 episodes each).
GPU IDs, checkpoints, delays and output prefixes are configurable.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/data/yhr/project/robot/online_filtering")
OUTPUT = Path("/data/yhr/output/online_filtering/eval")
CHECKPOINT = "/data/yhr/output/online_filtering/ckpt/xvla50_corrected_v2_2gpu_b32_30k_20261031/checkpoints/030000/pretrained_model"
PYTHON = "/data/yhr/anaconda/anaconda3/envs/online_filtering/bin/python"
SUITES = [("libero_spatial", 280), ("libero_object", 280), ("libero_goal", 300), ("libero_10", 520)]


def main() -> None:
    """Run a bounded process pool and record each GPU/task assignment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--gpus", default="3,4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=5)
    parser.add_argument("--delays", default="15,20")
    parser.add_argument("--prefix", default="xvla50_corrected_30k")
    parser.add_argument("--xvla-observation-delivery", choices=["buffered", "direct"], default="buffered")
    args = parser.parse_args()
    gpus = [int(gpu) for gpu in args.gpus.split(",")]
    delays = [int(delay) for delay in args.delays.split(",")]
    if not gpus or args.workers_per_gpu < 1:
        raise ValueError("positive worker count and GPU list required")
    jobs = queue.Queue()
    for delay in delays:
        for suite, horizon in SUITES:
            for group in range(5):
                jobs.put((delay, suite, horizon, group))
    records = []
    lock = threading.Lock()

    def worker(index):
        gpu = gpus[index // args.workers_per_gpu]
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(ROOT / "src"),
                "CUDA_VISIBLE_DEVICES": "0" if gpu == 0 else f"{gpu},0",
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "MUJOCO_EGL_DEVICE_ID": "0",
                "LIBGL_ALWAYS_SOFTWARE": "1",
                "MESA_LOADER_DRIVER_OVERRIDE": "llvmpipe",
                "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "LP_NUM_THREADS": "1",
                "LIBERO_CONFIG_PATH": str(ROOT / "research/config/libero_runtime"),
                "LIBERO_ASSETS_PATH": "/data/yhr/pretrain/recovered_zhoukai/dataset/libero-assets",
            }
        )
        time.sleep((index % args.workers_per_gpu) * 10)
        while True:
            try:
                delay, suite, horizon, group = jobs.get_nowait()
            except queue.Empty:
                return
            out = OUTPUT / f"{args.prefix}_delay{delay}_400_r20" / f"worker_{group}" / suite
            out.mkdir(parents=True, exist_ok=True)
            command = [
                PYTHON,
                str(ROOT / "research/scripts/eval_libero_synthetic_delay.py"),
                "--policy-path",
                args.checkpoint,
                "--output-dir",
                str(out),
                "--suite",
                suite,
                "--task-ids",
                f"{group},{group + 5}",
                "--episodes",
                "10",
                "--episode-length",
                str(horizon),
                "--delay-min",
                str(delay),
                "--delay-max",
                str(delay),
                "--replan-interval",
                "20",
                "--control-fps",
                "20",
                "--allow-queue-underrun",
                "--seed",
                "1000",
                "--device",
                "cuda",
                "--xvla-observation-delivery",
                args.xvla_observation_delivery,
            ]
            print(
                f"START worker={index} gpu={gpu} delay={delay} suite={suite} tasks={group},{group + 5}",
                flush=True,
            )
            with (out / "worker.log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            record = {
                "worker": index,
                "gpu": gpu,
                "delay": delay,
                "suite": suite,
                "tasks": [group, group + 5],
                "returncode": result.returncode,
                "output": str(out),
            }
            with lock:
                records.append(record)
                (OUTPUT / f"{args.prefix}_pool_records.json").write_text(json.dumps(records, indent=2))
            print("FINISH " + json.dumps(record), flush=True)
            jobs.task_done()

    with ThreadPoolExecutor(max_workers=len(gpus) * args.workers_per_gpu) as pool:
        list(pool.map(worker, range(len(gpus) * args.workers_per_gpu)))
    if any(r["returncode"] for r in records):
        raise RuntimeError("evaluation worker failed; inspect pool records")
    comparison = {}
    for delay in delays:
        base = OUTPUT / f"{args.prefix}_delay{delay}_400_r20"
        episodes = []
        for path in sorted(base.rglob("summary.json")):
            data = json.loads(path.read_text())
            assert data["control_fps"] == 20 and data["replan_interval"] == 20
            assert data["delay_range"] == [delay, delay]
            for episode in data["episodes"]:
                assert not episode["smoke_test_capped"]
                assert episode["initial_action_diagnostics"]["model_shape"][1] == 50
                episodes.append({"suite": data["suite"], "task_id": data["task_id"], **episode})
        assert len(episodes) == 400
        assert len({(e["suite"], e["task_id"], e["episode"]) for e in episodes}) == 400
        per_suite = {}
        for suite, _ in SUITES:
            group = [e for e in episodes if e["suite"] == suite]
            assert len(group) == 100
            successes = sum(e["success"] for e in group)
            per_suite[suite] = {"successes": successes, "episodes": 100, "success_rate": successes / 100}
        successes = sum(e["success"] for e in episodes)
        summary = {
            "policy": args.checkpoint,
            "chunk_size": 50,
            "delay": delay,
            "control_hz": 20,
            "replan_interval": 20,
            "successes": successes,
            "episodes": 400,
            "success_rate": successes / 400,
            "queue_underruns": sum(e["queue_underruns"] for e in episodes),
            "per_suite": per_suite,
        }
        (base / "aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        comparison[str(delay)] = summary
    (OUTPUT / f"{args.prefix}_latency_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"COMPLETE delays {delays}: {400 * len(delays)} episodes", flush=True)


if __name__ == "__main__":
    main()
