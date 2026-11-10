"""Full matched delay curves with a bounded, resumable multi-GPU evaluation pool."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval_xvla50_delay_pool import CHECKPOINT, PYTHON, ROOT, SUITES

ONLINE = "/data/yhr/output/online_filtering/ckpt/xvla50_phase_b_dropoutfix_2gpu_b32_30k_20261102/checkpoints/030000/pretrained_model"


def atomic_json(path, value):
    """Publish progress atomically so status readers never observe partial JSON."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def collect(out, suite, tasks, delay, episodes, *, allow_unselected=False):
    """Validate a completed work item, including the physical execution contract."""
    rows = []
    for p in out.rglob("summary.json"):
        d = json.loads(p.read_text())
        assert d["suite"] == suite
        if allow_unselected and d["task_id"] not in tasks:
            continue
        assert d["task_id"] in tasks
        assert d["xvla_precomputed_deltas"] and not d["xvla_relative_execution"]
        assert d["xvla_delta_execution"] == "accumulated" and d["control_mode"] == "absolute"
        assert d["control_fps"] == 20 and d["replan_interval"] == 25
        assert d["delay_range"] == [delay, delay] and d["diagnostic_paired_seed"]
        for e in d["episodes"]:
            assert not e["smoke_test_capped"]
            assert e["initial_action_diagnostics"]["model_shape"][1] == 50
            rows.append(
                {
                    "suite": suite,
                    "task": d["task_id"],
                    **{
                        k: e[k]
                        for k in ["episode", "success", "steps", "queue_underruns", "delta_clipped_commands"]
                    },
                }
            )
    assert len(rows) == len({(e["task"], e["episode"]) for e in rows}) == len(tasks) * episodes
    assert {(e["task"], e["episode"]) for e in rows} == {(t, i) for t in tasks for i in range(episodes)}
    return rows


def main():
    """Run all requested model/delay pairs and publish validated results and metrics."""
    global ONLINE
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="1,5")
    parser.add_argument("--workers-per-gpu", type=int, default=16)
    parser.add_argument("--gpu-workers", help="Explicit per-GPU counts, e.g. 1:4,5:12")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--delays", default="0,5,10,15,20,25")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--online-checkpoint", default=ONLINE)
    parser.add_argument("--only-online-arm", action="store_true")
    args = parser.parse_args()
    ONLINE = args.online_checkpoint
    gpus = [int(x) for x in args.gpus.split(",")]
    delays = [int(x) for x in args.delays.split(",")]
    counts = dict.fromkeys(gpus, args.workers_per_gpu)
    if args.gpu_workers:
        counts = {int(g): int(n) for g, n in (part.split(":") for part in args.gpu_workers.split(","))}
        assert set(counts) == set(gpus) and min(counts.values()) > 0
    slots = [g for level in range(max(counts.values())) for g in gpus if level < counts[g]]
    assert args.workers_per_gpu > 0 and 1 <= args.episodes <= 50 and max(delays) <= 25
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = queue.Queue()
    # Interleave models and delays so early progress covers the whole curve.
    for suite, horizon in SUITES:
        for group in range(5):
            for delay in delays:
                arms = [("online", ONLINE)] if args.only_online_arm else [("baseline", CHECKPOINT), ("online", ONLINE)]
                for label, source in arms:
                    jobs.put((label, source, delay, suite, horizon, group))
    lock = threading.Lock()
    stopped = threading.Event()
    children, records = {}, []
    total_jobs = jobs.qsize()
    atomic_json(
        args.output / "manifest.json",
        {
            "gpus": gpus,
            "workers_per_gpu": counts,
            "episodes_per_task": args.episodes,
            "delays": delays,
            "replan_interval": 25,
            "control_fps": 20,
            "action_execution": "precomputed_deltas_accumulated_from_switch_pose",
            "baseline": CHECKPOINT,
            "online": ONLINE,
            "videos": False,
            "total_jobs": total_jobs,
            "total_episodes": 40 * args.episodes * len(delays) * (1 if args.only_online_arm else 2),
        },
    )

    def terminate_children():
        with lock:
            processes = list(children.values())
        for process in processes:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)

    def interrupt(signum, frame):
        stopped.set()
        terminate_children()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    def worker(index):
        gpu = slots[index]
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(ROOT / "src"),
            CUDA_VISIBLE_DEVICES=str(gpu) if gpu == 0 else f"{gpu},0",
            MUJOCO_GL="egl",
            PYOPENGL_PLATFORM="egl",
            MUJOCO_EGL_DEVICE_ID="0",
            LIBGL_ALWAYS_SOFTWARE="1",
            MESA_LOADER_DRIVER_OVERRIDE="llvmpipe",
            __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json",
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
        )
        time.sleep((index // len(gpus)) * 2)
        while not stopped.is_set():
            try:
                job = jobs.get_nowait()
                label, source, delay, suite, horizon, group = job
            except queue.Empty:
                return
            out = args.output / f"{label}_d{delay}" / suite / str(group)
            out.mkdir(parents=True, exist_ok=True)
            tasks = [group, group + 5]
            try:
                rows = collect(out, suite, tasks, delay, args.episodes)
                resumed = True
            except (AssertionError, ValueError, KeyError):
                rows, resumed = None, False
            print(
                f"START worker={index} gpu={gpu} model={label} delay={delay} suite={suite} tasks={tasks} resumed={resumed}",
                flush=True,
            )
            rc = 0
            if not resumed:
                pending_tasks = []
                for task in tasks:
                    try:
                        collect(out, suite, [task], delay, args.episodes, allow_unselected=True)
                    except (AssertionError, ValueError, KeyError):
                        pending_tasks.append(task)
                assert pending_tasks
                cmd = [
                    PYTHON,
                    str(ROOT / "research/scripts/eval_libero_synthetic_delay.py"),
                    "--policy-path",
                    source,
                    "--output-dir",
                    str(out),
                    "--suite",
                    suite,
                    "--task-ids",
                    ",".join(map(str, pending_tasks)),
                    "--episodes",
                    str(args.episodes),
                    "--episode-length",
                    str(horizon),
                    "--delay-min",
                    str(delay),
                    "--delay-max",
                    str(delay),
                    "--replan-interval",
                    "25",
                    "--control-fps",
                    "20",
                    "--allow-queue-underrun",
                    "--seed",
                    "1000",
                    "--device",
                    "cuda",
                    "--diagnostic-paired-seed",
                    "--xvla-precomputed-deltas",
                    "--xvla-delta-execution",
                    "accumulated",
                    "--xvla-observation-delivery",
                    "direct" if label == "online" else "buffered",
                    "--no-video",
                    "--resume-episodes",
                    "--task-output-subdirs",
                ]
                with (out / "worker.log").open("w") as log:
                    process = subprocess.Popen(
                        cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                    )
                    with lock:
                        children[index] = process
                    rc = process.wait()
                    with lock:
                        children.pop(index, None)
                if rc == 0:
                    rows = collect(out, suite, tasks, delay, args.episodes)
            if rc:
                log_text = (out / "worker.log").read_text(errors="replace")[-16000:]
                if any(
                    token in log_text
                    for token in ["CUBLAS_STATUS_ALLOC_FAILED", "CUDA out of memory", "OutOfMemoryError"]
                ):
                    # A resource failure is not a failed policy episode. Keep the
                    # saved episode prefix and retry on a surviving, less busy slot.
                    with lock:
                        event_path = args.output / "resource_events.json"
                        events = json.loads(event_path.read_text()) if event_path.exists() else []
                        events.append(
                            {
                                "gpu": gpu,
                                "worker": index,
                                "output": str(out),
                                "returncode": rc,
                                "action": "requeue work item and retire slot",
                            }
                        )
                        atomic_json(event_path, events)
                    jobs.put(job)
                    jobs.task_done()
                    print(f"RESOURCE_RETRY retiring worker={index} gpu={gpu} output={out}", flush=True)
                    return
            record = {
                "worker": index,
                "gpu": gpu,
                "model": label,
                "delay": delay,
                "suite": suite,
                "tasks": tasks,
                "returncode": rc,
                "output": str(out),
                "episodes": len(rows) if rows else 0,
                "resumed": resumed,
            }
            with lock:
                records.append(record)
                atomic_json(args.output / "pool_records.json", records)
            print("FINISH " + json.dumps(record), flush=True)
            if rc:
                stopped.set()
                terminate_children()
                raise RuntimeError(f"worker failed: {out}/worker.log")
            jobs.task_done()

    pool = ThreadPoolExecutor(max_workers=len(slots))
    try:
        futures = [pool.submit(worker, i) for i in range(len(slots))]
        for future in as_completed(futures):
            future.result()
    except BaseException:
        stopped.set()
        terminate_children()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    assert len(records) == total_jobs and all(r["returncode"] == 0 for r in records)
    comparison = {}
    for label in ["baseline", "online"]:
        for delay in delays:
            rows = []
            for suite, _ in SUITES:
                for group in range(5):
                    rows.extend(
                        collect(
                            args.output / f"{label}_d{delay}" / suite / str(group),
                            suite,
                            [group, group + 5],
                            delay,
                            args.episodes,
                        )
                    )
            assert (
                len(rows) == len({(e["suite"], e["task"], e["episode"]) for e in rows}) == 40 * args.episodes
            )
            per_suite = {}
            for suite, _ in SUITES:
                subset = [e for e in rows if e["suite"] == suite]
                wins = sum(e["success"] for e in subset)
                per_suite[suite] = {
                    "successes": wins,
                    "episodes": len(subset),
                    "success_rate": wins / len(subset),
                }
            wins = sum(e["success"] for e in rows)
            result = {
                "successes": wins,
                "episodes": len(rows),
                "success_rate": wins / len(rows),
                "queue_underruns": sum(e["queue_underruns"] for e in rows),
                "delta_clipped_commands": sum(e["delta_clipped_commands"] for e in rows),
                "control_steps": sum(e["steps"] for e in rows),
                "per_suite": per_suite,
                "results": rows,
                "control_fps": 20,
                "replan_interval": 25,
                "actual_trigger_period": max(25, delay),
            }
            comparison[f"{label}_d{delay}"] = result
            atomic_json(args.output / f"{label}_d{delay}" / "aggregate_summary.json", result)
    atomic_json(args.output / "comparison.json", comparison)
    metrics = ROOT / "research/docs/metrics.md"
    text = metrics.read_text()
    marker = "Full accumulated-delta R25 evaluation pending; results will be published here."
    table = ["| Delay | Baseline | OF | OF minus baseline |", "|---|---:|---:|---:|"]
    for delay in delays:
        a, b = comparison[f"baseline_d{delay}"], comparison[f"online_d{delay}"]
        table.append(
            f"| {delay} | {a['successes']}/{a['episodes']} ({a['success_rate'] * 100:.2f}%) | {b['successes']}/{b['episodes']} ({b['success_rate'] * 100:.2f}%) | {(b['success_rate'] - a['success_rate']) * 100:+.2f} pp |"
        )
    if marker in text and not args.only_online_arm:
        metrics.write_text(text.replace(marker, "\n".join(table)))
    print(
        f"COMPLETE full R25 evaluation: {sum(v['episodes'] for v in comparison.values())} episodes",
        flush=True,
    )


if __name__ == "__main__":
    main()
