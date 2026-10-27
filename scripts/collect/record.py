#!/usr/bin/env python3
"""ROS2 RB75 staging recorder and offline LeRobot converter. See README.md."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import queue
import select
import shutil
import signal
import sqlite3
import subprocess
import sys
import termios
import threading
import time
import tty
import uuid
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import yaml

PARTS = ("left_joint", "left_gripper", "right_joint", "right_gripper")


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML 必须是 mapping")
    if not isinstance(cfg["dataset"]["fps"], int) or cfg["dataset"]["fps"] <= 0:
        raise ValueError("dataset.fps 必须是正整数")
    if cfg["robot"]["joints_per_arm"] != 7:
        raise ValueError("本采集器要求每臂 7 个关节 + 1 个夹爪")
    if set(cfg["streams"]) != {f"{kind}_{part}" for kind in ("state", "action") for part in PARTS}:
        raise ValueError("streams 必须完整定义 8 路 state/action 输入")
    for name, stream in cfg["streams"].items():
        expected = 7 if name.endswith("joint") else 1
        if stream["size"] != expected:
            raise ValueError(f"{name}: size 应为 {expected}")
        if not stream["topic"].startswith("/"):
            raise ValueError(f"{name}: 请使用绝对话题名")
    if len(cfg["cameras"]) != 4 or len({c["serial"] for c in cfg["cameras"].values()}) != 4:
        raise ValueError("必须配置四路不同序列号的相机")
    for camera in cfg["cameras"].values():
        if min(camera["width"], camera["height"], camera["fps"]) <= 0:
            raise ValueError("相机尺寸和 FPS 必须大于 0")
    if set(cfg["streams"]) & set(cfg["cameras"]):
        raise ValueError("相机名称不能与 streams 名称重复")
    for name in ("max_age_ms", "max_span_ms", "max_tick_lateness_ms", "queue_size", "min_frames"):
        if cfg["recording"][name] <= 0:
            raise ValueError(f"recording.{name} 必须大于 0")
    if not 1 <= cfg["dataset"]["jpeg_quality"] <= 100:
        raise ValueError("jpeg_quality 必须在 1..100")
    frequency = cfg["recording"].setdefault("frequency", {})
    if not isinstance(frequency.setdefault("allow_faster", False), bool):
        raise ValueError("recording.frequency.allow_faster 必须是 true/false")
    for key, default in {
        "window_s": 1.0,
        "tolerance_ratio": 0.05,
        "max_gap_periods": 1.8,
        "report_interval_s": 2.0,
    }.items():
        value = frequency.setdefault(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"recording.frequency.{key} 必须为有限正数")
    if frequency["tolerance_ratio"] >= 1 or frequency["max_gap_periods"] <= 1:
        raise ValueError("频率容差必须小于 1，最大消息间隔必须大于一个周期")
    for name, stream in cfg["streams"].items():
        fps = stream.setdefault("fps", cfg["dataset"]["fps"])
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(fps)
            or fps < cfg["dataset"]["fps"]
        ):
            raise ValueError(f"{name}.fps 必须不小于 dataset.fps，避免重复使用输入")
    for name, camera in cfg["cameras"].items():
        if camera["fps"] < cfg["dataset"]["fps"]:
            raise ValueError(f"{name}.fps 不得低于 dataset.fps")
    sync = cfg["recording"].setdefault("sync", {})
    for key, default in {"max_offset_ms": 15.0, "max_span_ms": 20.0, "lookahead_ms": 25.0}.items():
        value = sync.setdefault(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"recording.sync.{key} 必须为有限正数")
    limit = sync.setdefault("repeat_warning_steps", sync.pop("max_consecutive_drops", 3))
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("recording.sync.repeat_warning_steps 必须为正整数")
    if sync["lookahead_ms"] < sync["max_offset_ms"]:
        raise ValueError("sync.lookahead_ms 必须不小于 max_offset_ms，保证等待完整匹配窗口")
    if (
        sync["lookahead_ms"] + sync["max_offset_ms"] + cfg["recording"]["max_tick_lateness_ms"]
        > cfg["recording"]["max_age_ms"]
    ):
        raise ValueError("max_age_ms 必须覆盖同步等待、匹配偏移和调度容差")
    conversion = cfg.setdefault("conversion", {})
    if not isinstance(conversion.setdefault("parallel_encoding", True), bool):
        raise ValueError("conversion.parallel_encoding 必须是 true/false")
    for key, default in {"image_writer_threads": 4, "encoder_threads": 2}.items():
        value = conversion.setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"conversion.{key} 必须是正整数")
    return cfg


@dataclass(frozen=True)
class Sample:
    arrival: float
    header_ns: int
    value: Any


class Inputs:
    """Buffered inputs on one host monotonic arrival clock.

    snapshot() is causal/latest; match() pairs distinct messages around a grid
    timestamp with bounded lookahead. Source headers remain separate for audit.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.buffers: dict[str, deque[Sample]] = {
            name: deque(maxlen=32) for name in list(cfg["streams"]) + list(cfg["cameras"])
        }
        self.lock = threading.Lock()
        self.error: str | None = None
        self.frequency = cfg["recording"].get("frequency")
        self.arrivals: dict[str, deque[float]] = {name: deque() for name in self.buffers}

    def put(self, name: str, value: Any, header_ns: int = 0) -> None:
        with self.lock:
            now = time.monotonic()
            self.buffers[name].append(Sample(now, header_ns, value))
            if self.frequency:
                history = self.arrivals[name]
                history.append(now)
                # Retain one sample before the window for interval/gap checks.
                cutoff = now - self.frequency["window_s"]
                while len(history) > 2 and history[1] < cutoff:
                    history.popleft()

    def check_rates(self, now: float) -> dict[str, float]:
        """Check actual callback arrival rates and gaps; never infer Hz from saved ticks."""
        if not self.frequency:
            return {}
        window = self.frequency["window_s"]
        tolerance = self.frequency["tolerance_ratio"]
        rates = {}
        with self.lock:
            for name, history in self.arrivals.items():
                spec = self.cfg["streams"].get(name, self.cfg["cameras"].get(name))
                expected = spec["fps"]
                label = f"{name} ({spec['topic']})"
                if len(history) < 2 or history[0] > now - window:
                    raise RuntimeError(f"频率预热中：{label}，需连续接收至少 {window:g}s")
                stamps = [stamp for stamp in history if stamp >= now - window]
                # Use an interval estimate, avoiding a +/-1 message counting bias.
                rate = (
                    (len(stamps) - 1) / (stamps[-1] - stamps[0])
                    if len(stamps) >= 2 and stamps[-1] > stamps[0]
                    else 0.0
                )
                gap = max(
                    [now - history[-1], *[b - a for a, b in zip(history, list(history)[1:], strict=False)]]
                )
                max_gap = self.frequency["max_gap_periods"] / expected
                if gap > max_gap:
                    raise RuntimeError(
                        f"输入断帧：{label}，最大接收间隔 {gap * 1000:.1f}ms > {max_gap * 1000:.1f}ms，"
                        f"目标 {expected:g}Hz，实测 {rate:.2f}Hz"
                    )
                minimum = expected * (1 - tolerance)
                maximum = (
                    math.inf if self.frequency.get("allow_faster", False) else expected * (1 + tolerance)
                )
                if not minimum <= rate <= maximum:
                    allowed = (
                        f">= {minimum:.2f}Hz（允许更快输入）"
                        if math.isinf(maximum)
                        else f"{minimum:.2f}～{maximum:.2f}Hz"
                    )
                    raise RuntimeError(
                        f"接收频率异常：{label}，目标 {expected:g}Hz，实测 {rate:.2f}Hz，允许范围 {allowed}"
                    )
                rates[name] = rate
        return rates

    def check_new(self, snapshot: dict[str, Sample], previous: dict[str, Sample] | None) -> None:
        if previous is None:
            return
        for name, sample in snapshot.items():
            old = previous[name]
            if sample.arrival <= old.arrival or (
                sample.header_ns > 0 and old.header_ns > 0 and sample.header_ns <= old.header_ns
            ):
                spec = self.cfg["streams"].get(name, self.cfg["cameras"].get(name))
                raise RuntimeError(f"输入重复帧：{name} ({spec['topic']})，本采样周期没有新消息/新时间戳")

    def snapshot(self, tick: float) -> dict[str, Sample]:
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            result = {}
            for name, samples in self.buffers.items():
                sample = next((s for s in reversed(samples) if s.arrival <= tick), None)
                if sample is None or (tick - sample.arrival) * 1000 > self.cfg["recording"]["max_age_ms"]:
                    raise RuntimeError(f"{name} 缺失或超时")
                result[name] = sample
        arrivals = [s.arrival for s in result.values()]
        if (max(arrivals) - min(arrivals)) * 1000 > self.cfg["recording"]["max_span_ms"]:
            raise RuntimeError("输入接收时间跨度超限")
        return result

    def match(self, tick: float, previous: dict[str, Sample] | None, now: float) -> dict[str, Sample]:
        """Pair unused messages around a grid tick after bounded lookahead.

        All comparisons use host monotonic arrival timestamps. Never interpolate
        or duplicate source messages, or change the timestamp of the saved grid.
        """
        sync = self.cfg["recording"]["sync"]
        offset = sync["max_offset_ms"] / 1000
        span = sync["max_span_ms"] / 1000
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            candidates = {}
            for name, samples in self.buffers.items():
                old = previous[name] if previous else None
                candidates[name] = [
                    sample
                    for sample in samples
                    if abs(sample.arrival - tick) <= offset
                    and sample.arrival <= now
                    and (now - sample.arrival) * 1000 <= self.cfg["recording"]["max_age_ms"]
                    and (
                        old is None
                        or (
                            sample.arrival > old.arrival
                            and not (
                                sample.header_ns > 0
                                and old.header_ns > 0
                                and sample.header_ns <= old.header_ns
                            )
                        )
                    )
                ]
                if not candidates[name]:
                    spec = self.cfg["streams"].get(name, self.cfg["cameras"].get(name))
                    raise RuntimeError(
                        f"同步配对失败：{name} ({spec['topic']})，目标 ±{sync['max_offset_ms']:g}ms 内"
                        "没有未使用的新消息（缺帧、延迟或重复时间戳）"
                    )
        nearest = {
            name: min(samples, key=lambda sample: abs(sample.arrival - tick))
            for name, samples in candidates.items()
        }
        times = [sample.arrival for sample in nearest.values()]
        if max(times) - min(times) <= span:
            return nearest
        # Nearest-per-stream can straddle opposite edges. Search bounded windows
        # for an alternative group satisfying the skew and tick-distance bounds.
        best = None
        best_score = (math.inf, math.inf)
        for left in sorted({sample.arrival for samples in candidates.values() for sample in samples}):
            group = {}
            for name, stream_candidates in candidates.items():
                eligible = [sample for sample in stream_candidates if left <= sample.arrival <= left + span]
                if not eligible:
                    break
                group[name] = min(eligible, key=lambda sample: abs(sample.arrival - tick))
            if len(group) == len(candidates):
                distances = [abs(sample.arrival - tick) for sample in group.values()]
                score = (max(distances), sum(distances))
                if score < best_score:
                    best, best_score = group, score
        if best is None:
            raise RuntimeError(
                f"同步配对失败：各输入均有新消息，但无法满足组内最大时间差 {sync['max_span_ms']:g}ms"
            )
        return best

    def resample(self, tick: float, previous: dict[str, Sample] | None, now: float) -> dict[str, Sample]:
        """Prefer strict pairing; otherwise hold latest causal data, with bounded age.

        Old source headers do not refresh the age of the last accepted message.
        Reuse is allowed, but time reversal and stale inputs are not.
        """
        try:
            return self.match(tick, previous, now)
        except RuntimeError:
            pass
        result = {}
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            for name, samples in self.buffers.items():
                old = previous[name] if previous else None
                eligible = [
                    sample
                    for sample in samples
                    if sample.arrival <= tick
                    and (
                        old is None
                        or (
                            sample.arrival > old.arrival
                            and not (
                                sample.header_ns > 0
                                and old.header_ns > 0
                                and sample.header_ns <= old.header_ns
                            )
                        )
                    )
                ]
                sample = eligible[-1] if eligible else old
                if sample is None:
                    raise RuntimeError(f"【严重告警】{name} 没有可用消息")
                age = (now - sample.arrival) * 1000
                if age > self.cfg["recording"]["max_age_ms"]:
                    raise RuntimeError(f"【严重告警】{name} 数据过期 {age:.1f}ms，停止当前段")
                result[name] = sample
        return result

    def vector(self, snapshot: dict[str, Sample], kind: str) -> np.ndarray:
        return np.concatenate([snapshot[f"{kind}_{part}"].value for part in PARTS]).astype(np.float32)


class SamplingStats:
    """Episode-level quality counters; one reused source never drops the whole group."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.limit = cfg["recording"]["sync"]["repeat_warning_steps"]
        self.frames = 0
        self.repeats: dict[str, int] = {}
        self.consecutive: dict[str, int] = {}
        self.max_age: dict[str, float] = {}
        self.max_gap: dict[str, float] = {}
        self.max_skew = 0.0

    def observe(self, snapshot: dict[str, Sample], previous: dict[str, Sample] | None, now: float) -> None:
        self.frames += 1
        for name, sample in snapshot.items():
            reused = previous is not None and sample.arrival <= previous[name].arrival
            self.repeats[name] = self.repeats.get(name, 0) + int(reused)
            self.consecutive[name] = self.consecutive.get(name, 0) + 1 if reused else 0
            self.max_age[name] = max(self.max_age.get(name, 0.0), (now - sample.arrival) * 1000)
            if previous is not None and not reused:
                self.max_gap[name] = max(
                    self.max_gap.get(name, 0.0), (sample.arrival - previous[name].arrival) * 1000
                )
            if self.consecutive[name] == self.limit:
                print(f"【严重告警】{name} 连续 {self.limit} 步未更新；尚未过期，继续采集。", flush=True)
        times = [sample.arrival for sample in snapshot.values()]
        self.max_skew = max(self.max_skew, (max(times) - min(times)) * 1000)

    def summary(self) -> str:
        details = "；".join(
            f"{name}:重复 {self.repeats[name]}/{self.frames}"
            f"({self.repeats[name] / max(1, self.frames):.1%}),"
            f"最大数据年龄 {self.max_age[name]:.1f}ms,最大更新间隔 {self.max_gap.get(name, 0):.1f}ms"
            for name in self.repeats
        )
        return f"【采样质量】最大组内时间差 {self.max_skew:.1f}ms；{details}"


def numeric_value(message: Any, stream: dict[str, Any]) -> np.ndarray:
    value = np.asarray(getattr(message, stream["field"]), dtype=np.float32).reshape(-1)
    if len(value) < stream["size"]:
        raise ValueError("关节/夹爪消息维度不足")
    value = value[: stream["size"]] * stream.get("scale", 1.0) + stream.get("offset", 0.0)
    if not np.isfinite(value).all():
        raise ValueError("消息包含 NaN/Inf")
    return value


def image_bgr(message: Any, camera: dict[str, Any]) -> np.ndarray:
    """Decode raw ROS Image, including row padding, without cv_bridge ABI coupling."""
    if message.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"不支持图像编码 {message.encoding}，需要 rgb8/bgr8")
    if (message.width, message.height) != (camera["width"], camera["height"]):
        raise ValueError(f"相机尺寸与 YAML 不符: {message.width}x{message.height}")
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = rows[:, : message.width * 3].reshape(message.height, message.width, 3)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if message.encoding == "rgb8" else image


class EpisodeWriter:
    """Bounded staging queue; disk/JPEG failure invalidates the whole episode."""

    def __init__(self, session: Path, index: int, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.path = session / f"episode_{index:06d}.partial.sqlite3"
        self.queue: queue.Queue = queue.Queue(maxsize=cfg["recording"]["queue_size"])
        self.error: BaseException | None = None
        self.count = 0
        self.closed = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _encode(self, item: tuple[str, Sample]) -> tuple[str, bytes]:
        name, sample = item
        bgr = image_bgr(sample.value, self.cfg["cameras"][name])
        ok, data = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.cfg["dataset"]["jpeg_quality"]])
        if not ok:
            raise RuntimeError(f"{name} JPEG 编码失败")
        return name, data.tobytes()

    def _run(self) -> None:
        try:
            with (
                contextlib.closing(sqlite3.connect(self.path)) as db,
                ThreadPoolExecutor(max_workers=4) as pool,
            ):
                db.execute(
                    "CREATE TABLE frames (idx INTEGER PRIMARY KEY, tick REAL, state BLOB, action BLOB, stamps TEXT)"
                )
                db.execute(
                    "CREATE TABLE images (idx INTEGER, camera TEXT, jpeg BLOB, PRIMARY KEY(idx,camera))"
                )
                previous_samples: dict[str, Sample] = {}
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    tick, state, action, snapshot, sampled_at = item
                    encoded = list(pool.map(self._encode, [(n, snapshot[n]) for n in self.cfg["cameras"]]))
                    stamps = {
                        n: {
                            "arrival": s.arrival,
                            "header_ns": s.header_ns,
                            "age_ms": (sampled_at - s.arrival) * 1000,
                            "reused": n in previous_samples and s.arrival <= previous_samples[n].arrival,
                        }
                        for n, s in snapshot.items()
                    }
                    previous_samples = snapshot
                    db.execute(
                        "INSERT INTO frames VALUES (?,?,?,?,?)",
                        (self.count, tick, state.tobytes(), action.tobytes(), json.dumps(stamps)),
                    )
                    db.executemany(
                        "INSERT INTO images VALUES (?,?,?)", [(self.count, n, b) for n, b in encoded]
                    )
                    self.count += 1
                    if self.count % self.cfg["dataset"]["fps"] == 0:
                        db.commit()
                db.commit()
        except BaseException as exc:
            self.error = exc

    def submit(self, tick: float, inputs: Inputs, snapshot: dict[str, Sample]) -> None:
        if self.error:
            raise RuntimeError(f"暂存写入失败: {self.error}")
        try:
            self.queue.put_nowait(
                (
                    tick,
                    inputs.vector(snapshot, "state"),
                    inputs.vector(snapshot, "action"),
                    snapshot,
                    time.monotonic(),
                )
            )
        except queue.Full as exc:
            raise RuntimeError("暂存队列已满，无法维持目标频率") from exc

    def finish(self, accept: bool) -> Path | None:
        if self.closed:
            raise RuntimeError("episode 已关闭")
        self.closed = True
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self.thread.join()
        if self.error:
            # Keep partial data for diagnosis; never silently accept it.
            raise RuntimeError(f"暂存失败，保留 {self.path}: {self.error}")
        if not accept:
            self.path.unlink(missing_ok=True)
            return None
        if self.count < self.cfg["recording"]["min_frames"]:
            self.path.unlink(missing_ok=True)
            raise RuntimeError("episode 太短，已放弃")
        accepted = self.path.with_name(self.path.name.replace(".partial", ""))
        self.path.rename(accepted)
        return accepted


class Keyboard:
    def __enter__(self) -> Keyboard:
        if not sys.stdin.isatty():
            raise RuntimeError("请在交互终端启动，键盘需要 TTY")
        self.saved = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self.pending = ""
        return self

    def read(self, timeout: float) -> list[str]:
        if not select.select([sys.stdin], [], [], max(0.0, timeout))[0]:
            return []
        self.pending += os.read(sys.stdin.fileno(), 64).decode(errors="ignore")
        keys = []
        while self.pending:
            if self.pending.startswith("\x1b"):
                if self.pending in ("\x1b", "\x1b[", "\x1bO"):
                    break
                if self.pending.startswith(("\x1b[C", "\x1bOC")):
                    keys.append("right")
                self.pending = self.pending[3:]
            else:
                keys.append(self.pending[0].lower())
                self.pending = self.pending[1:]
        return keys

    def __exit__(self, *args: Any) -> None:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.saved)


def connect_ros(inputs: Inputs) -> tuple[Any, Any, threading.Thread]:
    # ROS Humble uses system Python; converter never imports these modules.
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data
    from rosidl_runtime_py.utilities import get_message

    rclpy.init()
    node = rclpy.create_node("lerobot_rb75_recorder")

    def callback(name: str, stream: dict[str, Any] | None, message: Any) -> None:
        try:
            value = numeric_value(message, stream) if stream else message
            stamp = message.header.stamp
            inputs.put(name, value, stamp.sec * 1_000_000_000 + stamp.nanosec)
        except Exception as exc:
            with inputs.lock:
                inputs.error = f"{name}: {exc}"

    for name, stream in inputs.cfg["streams"].items():
        node.create_subscription(
            get_message(stream["type"]),
            stream["topic"],
            lambda m, n=name, s=stream: callback(n, s, m),
            qos_profile_sensor_data,
        )
    for name, camera in inputs.cfg["cameras"].items():
        node.create_subscription(
            get_message("sensor_msgs/msg/Image"),
            camera["topic"],
            lambda m, n=name: callback(n, None, m),
            qos_profile_sensor_data,
        )

    def spin() -> None:
        with contextlib.suppress(ExternalShutdownException):
            rclpy.spin(node)

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    return rclpy, node, thread


@contextlib.contextmanager
def hardware(cfg: dict[str, Any], session: Path) -> Iterator[list[tuple[str, subprocess.Popen[bytes]]]]:
    """Only launch explicitly enabled local drivers; logs never steal the keyboard."""
    processes = []
    handles = []
    try:
        commands = []
        if cfg["runtime"]["start_slave_arms"]:
            for side in ("left", "right"):
                arm = cfg["robot"][side]
                commands.append(
                    (
                        f"arm_{side}",
                        [
                            cfg["runtime"]["slave_executable"],
                            "-a",
                            arm["ip"],
                            "-S",
                            side,
                            "-p",
                            "0",
                            "-P",
                            str(arm["tcp_port"]),
                            "-r",
                            str(arm["udp_port"]),
                            "-M",
                            "2",
                            "-j7",
                        ],
                    )
                )
        if cfg["runtime"]["start_cameras"]:
            for name, camera in cfg["cameras"].items():
                profile = f"{camera['width']}x{camera['height']}x{camera['fps']}"
                commands.append(
                    (
                        f"camera_{name}",
                        [
                            "ros2",
                            "run",
                            "realsense2_camera",
                            "realsense2_camera_node",
                            "--ros-args",
                            "-r",
                            f"__ns:={camera['namespace']}",
                            "-r",
                            "__node:=camera",
                            "-p",
                            f"serial_no:=_{camera['serial']}",
                            "-p",
                            "enable_color:=true",
                            "-p",
                            "enable_depth:=false",
                            "-p",
                            "enable_infra1:=false",
                            "-p",
                            "enable_infra2:=false",
                            "-p",
                            f"rgb_camera.color_profile:={profile}",
                        ],
                    )
                )
        for name, command in commands:
            handle = (session / f"{name}.log").open("w")
            handles.append(handle)
            processes.append(
                (
                    name,
                    subprocess.Popen(
                        command,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    ),
                )
            )
        yield processes
    finally:
        for _, process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for _, process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for handle in handles:
            handle.close()


def show_controls(recording: bool = False) -> None:
    if recording:
        print(
            "【采集中】\n"
            "  按 n：完成当前 episode 并暂存，等待下一段。\n"
            "  按 f：放弃当前 episode，重新采集。\n"
            "  要结束本次作业：先按 n 暂存（或 f 放弃），再按 s 统一转换。",
            flush=True,
        )
    else:
        print(
            "【等待操作】\n"
            "  按右方向键 →：开始一段新的 episode（需要所有输入就绪）。\n"
            "  按 s：结束本次采集作业，统一转换所有已暂存的 episode。",
            flush=True,
        )


def record(cfg: dict[str, Any]) -> Path | None:
    root = Path(cfg["dataset"]["save_dir"]).expanduser()
    folder_name = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%m%d%H%M")
    suffix = 0
    while True:
        session = root / (folder_name if suffix == 0 else f"{folder_name}_{suffix:02d}")
        try:
            session.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
    (session / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    inputs = Inputs(cfg)
    writer = None
    accepted = 0
    index = 0
    period = 1 / cfg["dataset"]["fps"]
    lookahead = cfg["recording"]["sync"]["lookahead_ms"] / 1000
    rclpy, node, thread = connect_ros(inputs)
    try:
        with hardware(cfg, session) as processes, Keyboard() as keyboard:
            print(f"\n作业目录：{session}\n采集频率：{cfg['dataset']['fps']} Hz。")
            print("Ctrl+C 保留已暂存段，当前段放弃；不会自动转换。")
            print(
                f"数据同步：{cfg['dataset']['fps']}Hz 固定时间栅格；"
                f"接收时间匹配 ±{cfg['recording']['sync']['max_offset_ms']:g}ms，"
                f"优先组内时间差 ≤{cfg['recording']['sync']['max_span_ms']:g}ms，"
                f"等待 {lookahead * 1000:g}ms 配对。",
                flush=True,
            )
            print("缺新消息时沿用最近数据；连续未更新告警，超过数据年龄上限停止当前段。", flush=True)
            show_controls()
            next_tick = time.monotonic()
            last_report = 0.0
            last_rate_report = 0.0
            previous = None
            stats = SamplingStats(cfg)
            while True:
                for name, process in processes:
                    if process.poll() is not None:
                        raise RuntimeError(f"{name} 驱动退出，请查看作业目录日志")
                now = time.monotonic()
                keys = keyboard.read(min(0.05, max(0, next_tick + lookahead - now)) if writer else 0.05)
                for key in keys:
                    if key == "right" and writer is None:
                        try:
                            inputs.snapshot(time.monotonic())
                            inputs.check_rates(time.monotonic())
                        except RuntimeError as exc:
                            print(f"无法开始：{exc}", flush=True)
                            print("请检查对应话题，输入就绪后按右方向键 → 重试。", flush=True)
                            show_controls()
                            continue
                        writer = EpisodeWriter(session, index, cfg)
                        index += 1
                        next_tick = time.monotonic()
                        previous = None
                        stats = SamplingStats(cfg)
                        last_rate_report = next_tick
                        print(f"已开始 episode {index - 1}。", flush=True)
                        show_controls(recording=True)
                    elif key in ("n", "f") and writer is not None:
                        current, writer = writer, None
                        print(stats.summary(), flush=True)
                        print(
                            "正在暂存当前 episode，请等待……"
                            if key == "n"
                            else "正在放弃当前 episode，请等待……",
                            flush=True,
                        )
                        try:
                            if key == "n":
                                try:
                                    inputs.snapshot(time.monotonic())
                                except RuntimeError as exc:
                                    current.finish(False)
                                    raise RuntimeError(f"当前段无效：{exc}；已放弃，请按 → 重采") from exc
                            path = current.finish(accept=key == "n")
                            accepted += int(path is not None)
                            if path is not None:
                                path.with_suffix(".quality.json").write_text(
                                    json.dumps(
                                        {
                                            "frames": stats.frames,
                                            "repeats": stats.repeats,
                                            "max_age_ms": stats.max_age,
                                            "max_update_gap_ms": stats.max_gap,
                                            "max_group_span_ms": stats.max_skew,
                                        },
                                        ensure_ascii=False,
                                        indent=2,
                                    )
                                )
                            print(
                                f"{'已暂存' if path else '已放弃'} {current.count} 帧；本次作业共暂存 {accepted} 段。",
                                flush=True,
                            )
                        except RuntimeError as exc:
                            print(str(exc), flush=True)
                        show_controls()
                    elif key == "s":
                        if writer:
                            print("当前仍在采集，尚未结束作业。", flush=True)
                            show_controls(recording=True)
                        else:
                            print(
                                f"本次采集结束，共 {accepted} 段，接下来统一转换，请等待。"
                                if accepted
                                else "本次没有已暂存的 episode，作业结束，无需转换。",
                                flush=True,
                            )
                            return session if accepted else None
                    elif key in ("right", "n", "f"):
                        print("该按键不适用于当前状态，请按以下提示操作。", flush=True)
                        show_controls(recording=writer is not None)
                if writer:
                    now = time.monotonic()
                    if now < next_tick + lookahead:
                        continue
                    try:
                        rates = {}
                        if now - last_rate_report >= cfg["recording"]["frequency"]["report_interval_s"]:
                            try:
                                rates = inputs.check_rates(now)
                            except RuntimeError as exc:
                                print(f"【频率告警】{exc}", flush=True)
                        if (now - next_tick - lookahead) * 1000 > cfg["recording"]["max_tick_lateness_ms"]:
                            raise RuntimeError("【严重告警】采样线程超时，不能维持完整固定时间轴")
                        snapshot = inputs.resample(next_tick, previous, now)
                        stats.observe(snapshot, previous, now)
                        writer.submit(next_tick, inputs, snapshot)
                        previous = snapshot
                        next_tick += period
                        if now - last_rate_report >= cfg["recording"]["frequency"]["report_interval_s"]:
                            last_rate_report = now
                            report = "；".join(f"{name}={rate:.1f}Hz" for name, rate in rates.items())
                            arrivals = [sample.arrival for sample in snapshot.values()]
                            skew = (max(arrivals) - min(arrivals)) * 1000
                            print(f"【采集中】组内时间差 {skew:.2f}ms；{report}", flush=True)
                            print(stats.summary(), flush=True)
                    except RuntimeError as exc:
                        current, writer = writer, None
                        print(stats.summary(), flush=True)
                        print(f"当前段异常：{exc}；停止录制，请排查后按 → 重采。", flush=True)
                        try:
                            abnormal = current.finish(True)
                            if abnormal is not None:
                                retained = abnormal.with_name("invalid_" + abnormal.name)
                                abnormal.rename(retained)
                                print(f"异常段保留用于排查，不自动转换：{retained}", flush=True)
                        except RuntimeError as finish_error:
                            print(f"异常段无法接受：{finish_error}", flush=True)
                        show_controls()
                elif time.monotonic() - last_report > 5:
                    last_report = time.monotonic()
                    try:
                        inputs.snapshot(last_report)
                        inputs.check_rates(last_report)
                        print("输入就绪，可以开始采集。", flush=True)
                    except RuntimeError as exc:
                        print(f"等待输入：{exc}", flush=True)
                    show_controls()
    finally:
        try:
            if writer:
                writer.finish(False)
        finally:
            rclpy.try_shutdown()
            thread.join(timeout=3)
            node.destroy_node()
            print(f"原始暂存保留在：{session}", flush=True)


def convert(session: Path) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cfg = load_config(session / "config.yaml")
    episodes = sorted(session.glob("episode_[0-9]*.sqlite3"))
    episodes = [p for p in episodes if ".partial." not in p.name]
    if not episodes:
        raise ValueError("没有已按 n 接受的 episode")
    destination = session / "lerobot"
    if destination.exists():
        raise FileExistsError(f"数据集已存在，拒绝覆盖：{destination}")
    temporary = session / f".lerobot_{uuid.uuid4().hex[:8]}.incomplete"
    names = [
        f"{side}_joint_{i + 1}" if i < 7 else f"{side}_gripper"
        for side in ("left", "right")
        for i in range(8)
    ]
    features = {
        key: {"dtype": "float32", "shape": (16,), "names": names} for key in ("observation.state", "action")
    }
    for name, camera in cfg["cameras"].items():
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (camera["height"], camera["width"], 3),
            "names": ["height", "width", "channels"],
        }
    dataset = LeRobotDataset.create(
        repo_id=cfg["dataset"]["repo_id"],
        root=temporary,
        fps=cfg["dataset"]["fps"],
        robot_type=cfg["robot"]["type"],
        features=features,
        image_writer_threads=cfg["conversion"]["image_writer_threads"],
        encoder_threads=cfg["conversion"]["encoder_threads"],
        use_videos=True,
    )
    try:
        for path in episodes:
            with contextlib.closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as db:
                count = 0
                previous_tick = None
                for idx, tick, state, action in db.execute(
                    "SELECT idx,tick,state,action FROM frames ORDER BY idx"
                ):
                    if idx != count or (previous_tick is not None and tick <= previous_tick):
                        raise ValueError(f"{path}: 暂存帧序号或时间倒退")
                    if previous_tick is not None:
                        steps = (tick - previous_tick) * cfg["dataset"]["fps"]
                        if abs(steps - round(steps)) > 0.05:
                            raise ValueError(f"{path}: 暂存时间不在采样栅格上")
                        if round(steps) != 1:
                            raise ValueError(f"{path}: 存在旧版丢帧缺口，拒绝压缩时间轴或自动切段")
                    previous_tick = tick
                    frame = {
                        "observation.state": np.frombuffer(state, dtype=np.float32).copy(),
                        "action": np.frombuffer(action, dtype=np.float32).copy(),
                        "task": cfg["dataset"]["task"],
                    }
                    images = dict(db.execute("SELECT camera,jpeg FROM images WHERE idx=?", (idx,)))
                    if set(images) != set(cfg["cameras"]):
                        raise ValueError(f"{path}: 第 {idx} 帧图像不完整")
                    for name, jpeg in images.items():
                        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if bgr is None:
                            raise ValueError(f"{path}: {name} JPEG 损坏")
                        frame[f"observation.images.{name}"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    dataset.add_frame(frame)
                    count += 1
                if count < cfg["recording"]["min_frames"]:
                    raise ValueError(f"{path}: episode 帧数不足")
                dataset.save_episode(parallel_encoding=cfg["conversion"]["parallel_encoding"])
                print(f"转换完成 {path.name}: {count} 帧", flush=True)
        dataset.finalize()
        temporary.rename(destination)
    except BaseException:
        print(f"转换未完成，原始数据不变；中间产物：{temporary}", file=sys.stderr)
        raise
    print(f"LeRobot 数据集已保存：{destination}", flush=True)
    return destination


def check(cfg: dict[str, Any]) -> None:
    from rosidl_runtime_py.utilities import get_message

    for stream in cfg["streams"].values():
        get_message(stream["type"])
    get_message("sensor_msgs/msg/Image")
    converter = Path(cfg["runtime"]["converter_python"])
    if not converter.is_file():
        raise FileNotFoundError(converter)
    if shutil.which("uv") is None:
        raise RuntimeError("请先安装 uv 或将其加入 PATH")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            str(converter),
            "python",
            "-c",
            "import cv2, yaml; from lerobot.datasets.lerobot_dataset import LeRobotDataset",
        ],
        env=env,
        check=True,
    )
    print("YAML、消息类型、转换解释器和 uv 检查通过；未连接或启动硬件。")
    print(f"采集 {cfg['dataset']['fps']} Hz；状态/动作各 16 维；四相机。")
    print("话题实际可用性、单位和时间抖动需通过现场采集验证。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "convert", "check"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--session", type=Path)
    args = parser.parse_args()
    if args.mode == "convert":
        if args.session is None:
            parser.error("convert 需要 --session")
        convert(args.session.resolve())
        return
    cfg = load_config(args.config.resolve())
    if args.mode == "check":
        check(cfg)
        return
    session = record(cfg)
    if session:
        print("作业结束，开始统一转换；原始暂存不会删除。", flush=True)
        env = os.environ.copy()
        # Strip ROS Python 3.10 paths in the separate training/conversion process.
        repo = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = str(repo / "src")
        subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                cfg["runtime"]["converter_python"],
                "python",
                str(Path(__file__).resolve()),
                "convert",
                "--session",
                str(session),
            ],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断。已接受的原始段仍在作业目录，可手动转换。", file=sys.stderr)
        sys.exit(130)
