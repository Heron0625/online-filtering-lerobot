"""Pilot matched baseline/online XVLA execution-time relative target conversion."""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from eval_xvla50_delay_pool import CHECKPOINT, PYTHON, ROOT, SUITES

OUTPUT = ROOT / "research/outputs/xvla_relative_execution_20261103"
ONLINE = "/data/yhr/output/online_filtering/ckpt/xvla50_phase_b_dropoutfix_2gpu_b32_30k_20261102/checkpoints/030000/pretrained_model"
ARMS = [
    (f"{label}_d{delay}", delay, source)
    for delay in [0, 15, 25]
    for label, source in [("baseline", CHECKPOINT), ("online", ONLINE)]
]


def worker(index):
    """Run one bounded GPU slot over its deterministic subset of jobs."""
    gpu = [2, 6, 7][index // 5]
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        CUDA_VISIBLE_DEVICES=f"{gpu},0",
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        MUJOCO_EGL_DEVICE_ID="0",
        LIBGL_ALWAYS_SOFTWARE="1",
        MESA_LOADER_DRIVER_OVERRIDE="llvmpipe",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        LP_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        LIBERO_CONFIG_PATH=str(ROOT / "research/config/libero_runtime"),
        LIBERO_ASSETS_PATH="/data/yhr/pretrain/recovered_zhoukai/dataset/libero-assets",
        __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json",
    )
    jobs = [
        (arm, delay, source, suite, horizon, group)
        for arm, delay, source in ARMS
        for suite, horizon in SUITES
        for group in range(5)
    ]
    for arm, delay, source, suite, horizon, group in jobs[index::15]:
        out = OUTPUT / arm / suite / str(group)
        out.mkdir(parents=True, exist_ok=True)
        command = [
            PYTHON,
            str(ROOT / "research/scripts/eval_libero_synthetic_delay.py"),
            "--policy-path",
            source,
            "--output-dir",
            str(out),
            "--suite",
            suite,
            "--task-ids",
            f"{group},{group + 5}",
            "--episodes",
            "1",
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
            "--diagnostic-paired-seed",
            "--xvla-relative-execution",
            "--xvla-observation-delivery",
            "direct" if source == ONLINE else "buffered",
        ]
        print(f"START gpu={gpu} arm={arm} suite={suite} tasks={group},{group + 5}", flush=True)
        with (out / "worker.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"DONE gpu={gpu} arm={arm} suite={suite} tasks={group},{group + 5}", flush=True)


def main():
    """Run all arms, validate task coverage and publish diagnostic-only counts."""
    with ThreadPoolExecutor(max_workers=15) as pool:
        list(pool.map(worker, range(15)))
    results = {}
    for arm, _, _ in ARMS:
        episodes = []
        for p in (OUTPUT / arm).rglob("summary.json"):
            data = json.loads(p.read_text())
            assert data["xvla_relative_execution"] and not data["xvla_chunk_delta"]
            assert data["control_mode"] == "relative"
            assert data["delay_range"] == [int(arm.rsplit("d", 1)[1])] * 2
            assert all(not e["smoke_test_capped"] for e in data["episodes"])
            episodes.extend({"suite": data["suite"], "task": data["task_id"], **e} for e in data["episodes"])
        assert len(episodes) == len({(e["suite"], e["task"]) for e in episodes}) == 40
        results[arm] = {
            "episodes": 40,
            "successes": sum(e["success"] for e in episodes),
            "queue_underruns": sum(e["queue_underruns"] for e in episodes),
            "delta_clipped_commands": sum(e["delta_clipped_commands"] for e in episodes),
            "results": episodes,
        }
    (OUTPUT / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    print("COMPLETE experimental relative-control pilot 240 episodes", flush=True)


if __name__ == "__main__":
    main()
