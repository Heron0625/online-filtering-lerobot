"""Read-only measurement of every configured input using recorder-equivalent QoS."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def measure(path: Path, duration: float, warmup: float) -> dict[str, Any]:
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rosidl_runtime_py.utilities import get_message

    with path.open() as handle:
        cfg = yaml.safe_load(handle)
    specs = {**cfg["streams"], **cfg["cameras"]}
    arrivals: dict[str, list[float]] = {name: [] for name in specs}
    headers: dict[str, list[int]] = {name: [] for name in specs}
    begin = float("inf")
    rclpy.init()
    node = rclpy.create_node("rb75_readonly_hz_probe")

    def callback(name: str, message: Any) -> None:
        now = time.monotonic()
        if now < begin:
            return
        arrivals[name].append(now)
        stamp = message.header.stamp
        headers[name].append(stamp.sec * 1_000_000_000 + stamp.nanosec)

    try:
        for name, spec in specs.items():
            node.create_subscription(
                get_message(spec.get("type", "sensor_msgs/msg/Image")),
                spec["topic"],
                lambda message, n=name: callback(n, message),
                qos_profile_sensor_data,
            )
        print(f"只订阅、不控制：发现话题 {warmup:g}s，然后同时测量全部输入 {duration:g}s。", flush=True)
        discovery_end = time.monotonic() + warmup
        while time.monotonic() < discovery_end:
            rclpy.spin_once(node, timeout_sec=0.05)
        begin = time.monotonic()
        while time.monotonic() - begin < duration:
            rclpy.spin_once(node, timeout_sec=0.02)
        elapsed = time.monotonic() - begin
        results = {}
        print("输入                         接收Hz   非零时间戳去重Hz   P95间隔ms   最大间隔ms", flush=True)
        for name, spec in specs.items():
            times = arrivals[name]
            gaps = np.diff(times) * 1000
            nonzero = [stamp for stamp in headers[name] if stamp > 0]
            entry = {
                "topic": spec["topic"],
                "target_hz": spec.get("fps", cfg["dataset"]["fps"]),
                "publishers": node.count_publishers(spec["topic"]),
                "messages": len(times),
                "hz": len(times) / elapsed,
                "unique_header_hz": len(set(nonzero)) / elapsed if nonzero else None,
                "p95_gap_ms": float(np.percentile(gaps, 95)) if len(gaps) else None,
                "max_gap_ms": float(gaps.max()) if len(gaps) else None,
            }
            results[name] = entry
            unique = f"{entry['unique_header_hz']:.2f}" if nonzero else "无有效时间戳"
            p95 = f"{entry['p95_gap_ms']:.2f}" if len(gaps) else "--"
            maximum = f"{entry['max_gap_ms']:.2f}" if len(gaps) else "--"
            print(f"{name:28s} {entry['hz']:7.2f} {unique:>16s} {p95:>12s} {maximum:>12s}", flush=True)
        return {"duration_s": elapsed, "config": str(path), "inputs": results}
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--warmup", type=float, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seconds <= 0 or args.warmup < 0:
        parser.error("seconds 必须大于 0，warmup 不能小于 0")
    result = measure(args.config, args.seconds, args.warmup)
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
