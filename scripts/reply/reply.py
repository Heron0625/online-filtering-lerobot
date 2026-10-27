#!/usr/bin/env python3
"""Keyboard-driven playback of collect SQLite staging or local LeRobot datasets."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import select
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

JOINTS = [*range(7), *range(8, 15)]


def config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        cfg = yaml.safe_load(handle)
    if cfg["dataset"]["joint_source"] not in ("observation.state", "action"):
        raise ValueError("joint_source 只能是 observation.state 或 action")
    for section, names in {
        "dataset": ["speed_multiplier"],
        "alignment": ["speed_deg_s", "min_duration_s", "tolerance_deg", "settle_timeout_s"],
        "playback": [
            "command_hz",
            "feedback_timeout_s",
            "ready_timeout_s",
            "max_joint_speed_deg_s",
            "max_tracking_error_deg",
            "max_tick_lateness_s",
        ],
    }.items():
        for name in names:
            value = cfg[section][name]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{section}.{name} 必须是有限正数")
    for arm in cfg["playback"]["arms"].values():
        if any(not arm[key].startswith("/") for key in ("feedback_topic", "command_topic", "gripper_topic")):
            raise ValueError("请使用绝对 ROS 话题名")
    if set(cfg["playback"]["arms"]) != {"left", "right"}:
        raise ValueError("必须配置左右臂")
    if cfg["playback"]["max_tick_lateness_s"] >= 1 / cfg["playback"]["command_hz"]:
        raise ValueError("max_tick_lateness_s 必须小于一个控制周期，避免延迟后追赶指令")
    for name in ("gripper_scale", "gripper_offset", "gripper_command_min", "gripper_command_max"):
        if not math.isfinite(cfg["playback"][name]):
            raise ValueError(f"{name} 必须是有限数值")
    if cfg["playback"]["gripper_command_min"] > cfg["playback"]["gripper_command_max"]:
        raise ValueError("夹爪命令最小值不能大于最大值")
    return cfg


def validate_vectors(state: np.ndarray, action: np.ndarray) -> None:
    if state.ndim != 2 or state.shape[1] != 16 or action.shape != state.shape or len(state) < 2:
        raise ValueError(f"需要至少两帧、16 维的 state/action，得到 {state.shape}/{action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("数据包含 NaN/Inf")


def save_prepared(
    out: Path, index: int, label: str, state: np.ndarray, action: np.ndarray, fps: float, cfg: dict[str, Any]
) -> None:
    validate_vectors(state, action)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("数据 FPS 无效")
    source = state if cfg["dataset"]["joint_source"] == "observation.state" else action
    joints = np.rad2deg(source[:, JOINTS]).astype(np.float64)
    grippers = action[:, [7, 15]].astype(np.float64)
    grippers = grippers * cfg["playback"]["gripper_scale"] + cfg["playback"]["gripper_offset"]
    effective_fps = fps * cfg["dataset"]["speed_multiplier"]
    peak_speed = float(np.abs(np.diff(joints, axis=0)).max() * effective_fps)
    if peak_speed > cfg["playback"]["max_joint_speed_deg_s"]:
        raise ValueError(
            f"{label}: 轨迹关节速度 {peak_speed:.2f}°/s 超出配置上限，请检查数据或降低 speed_multiplier"
        )
    if cfg["playback"]["gripper_enabled"] and (
        (grippers < cfg["playback"]["gripper_command_min"]).any()
        or (grippers > cfg["playback"]["gripper_command_max"]).any()
    ):
        raise ValueError(f"{label}: 夹爪命令超出范围，请核对原始单位和 gripper_scale/offset")
    np.savez(
        out / f"episode_{index:06d}.npz", joints=joints, grippers=grippers, fps=effective_fps, label=label
    )
    print(
        f"已检查 {label}: {len(joints)} 帧，回放 {effective_fps:g} FPS，最大关节速度 {peak_speed:.2f}°/s",
        flush=True,
    )


def prepare(cfg: dict[str, Any], out: Path) -> None:
    path = Path(cfg["dataset"]["path"]).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"请在 config.yaml 的 dataset.path 填写实际作业目录：{path}")
    out.mkdir(parents=True, exist_ok=True)
    if list(out.iterdir()):
        raise ValueError("预处理输出目录必须为空")
    dataset = path / "lerobot" if (path / "lerobot" / "meta" / "info.json").exists() else path
    if (dataset / "meta" / "info.json").exists():
        import pyarrow.parquet as pq

        info = json.loads((dataset / "meta" / "info.json").read_text())
        expected_names = [
            f"{side}_joint_{i + 1}" if i < 7 else f"{side}_gripper"
            for side in ("left", "right")
            for i in range(8)
        ]
        for name in ("observation.state", "action"):
            feature = info["features"][name]
            if feature["shape"] != [16] or feature["names"] != expected_names:
                raise ValueError("数据特征顺序与 collect 的 16 维排列不一致")
        rows: dict[int, list[dict[str, Any]]] = {}
        for file in sorted((dataset / "data").rglob("*.parquet")):
            table = pq.read_table(
                file, columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"]
            )
            for row in table.to_pylist():
                rows.setdefault(row["episode_index"], []).append(row)
        if (
            not rows
            or set(rows) != set(range(info["total_episodes"]))
            or sum(len(frames) for frames in rows.values()) != info["total_frames"]
        ):
            raise ValueError("LeRobot 数据集 episode 不完整")
        for index, (episode, frames) in enumerate(sorted(rows.items())):
            frames.sort(key=lambda row: row["frame_index"])
            if [f["frame_index"] for f in frames] != list(range(len(frames))):
                raise ValueError(f"episode {episode} 帧编号不连续")
            if not np.allclose(
                [f["timestamp"] for f in frames], np.arange(len(frames)) / info["fps"], atol=0.001
            ):
                raise ValueError(f"episode {episode} 时间戳不连续")
            save_prepared(
                out,
                index,
                f"episode_{episode:06d}",
                np.asarray([f["observation.state"] for f in frames]),
                np.asarray([f["action"] for f in frames]),
                info["fps"],
                cfg,
            )
    else:
        with (path / "config.yaml").open() as handle:
            capture = yaml.safe_load(handle)
        episodes = [p for p in sorted(path.glob("episode_[0-9]*.sqlite3")) if ".partial." not in p.name]
        for index, file in enumerate(episodes):
            with contextlib.closing(sqlite3.connect(f"{file.as_uri()}?mode=ro", uri=True)) as db:
                raw_frames = db.execute("SELECT idx,tick,state,action FROM frames ORDER BY idx").fetchall()
            if not raw_frames or [f[0] for f in raw_frames] != list(range(len(raw_frames))):
                raise ValueError(f"{file}: 帧编号不连续")
            fps = capture["dataset"]["fps"]
            if not np.allclose(
                [f[1] - raw_frames[0][1] for f in raw_frames], np.arange(len(raw_frames)) / fps, atol=0.001
            ):
                raise ValueError(f"{file}: 时间戳不连续")
            save_prepared(
                out,
                index,
                file.stem,
                np.stack([np.frombuffer(f[2], dtype=np.float32) for f in raw_frames]),
                np.stack([np.frombuffer(f[3], dtype=np.float32) for f in raw_frames]),
                fps,
                cfg,
            )
    if not list(out.glob("episode_*.npz")):
        raise ValueError("没有可回放的 episode；仅使用完整段，不读取 partial/incomplete 数据")


class CancelledError(Exception):
    """User requested immediate end of playback."""


class KeyReader(Protocol):
    def read(self, timeout: float = 0) -> list[str]: ...


class Keyboard:
    def __enter__(self) -> Keyboard:
        if not sys.stdin.isatty():
            raise RuntimeError("请在交互终端启动")
        self.saved = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self.pending = ""
        return self

    def read(self, timeout: float = 0) -> list[str]:
        if not select.select([sys.stdin], [], [], max(0, timeout))[0]:
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


def wait_until(deadline: float, keyboard: KeyReader) -> None:
    while True:
        remaining = deadline - time.monotonic()
        if "f" in keyboard.read(min(0.02, max(0, remaining))):
            raise CancelledError
        if remaining <= 0:
            return


class RosRobot:
    def __init__(self, cfg: dict[str, Any]) -> None:
        import rclpy
        from rclpy.executors import ExternalShutdownException
        from rclpy.qos import qos_profile_sensor_data
        from trace_msgs.msg import GripperStatus, JointAngle

        self.cfg = cfg
        self.rclpy = rclpy
        self.JointAngle = JointAngle
        self.GripperStatus = GripperStatus
        self.lock = threading.Lock()
        self.feedback: dict[str, tuple[float, np.ndarray]] = {}
        self.error: str | None = None
        self.commanded = False
        self.last_grippers: dict[str, float] = {}
        self.driver_check: Callable[[], None] | None = None
        rclpy.init()
        self.node = rclpy.create_node("lerobot_rb75_replay")
        self.publishers = {}
        self.grippers = {}
        for side, arm in cfg["playback"]["arms"].items():
            self.publishers[side] = self.node.create_publisher(
                JointAngle, arm["command_topic"], qos_profile_sensor_data
            )
            self.grippers[side] = self.node.create_publisher(
                GripperStatus, arm["gripper_topic"], qos_profile_sensor_data
            )
            self.node.create_subscription(
                JointAngle,
                arm["feedback_topic"],
                lambda message, s=side: self._feedback(s, message),
                qos_profile_sensor_data,
            )

        def spin() -> None:
            with contextlib.suppress(ExternalShutdownException):
                rclpy.spin(self.node)

        self.thread = threading.Thread(target=spin, daemon=True)
        self.thread.start()

    def _feedback(self, side: str, message: Any) -> None:
        values = np.asarray(message.angle[:7], dtype=np.float64)
        with self.lock:
            if values.shape != (7,) or not np.isfinite(values).all():
                self.error = f"{side} 关节反馈无效"
            else:
                self.feedback[side] = (time.monotonic(), values)

    def current(self) -> np.ndarray:
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            for side in ("left", "right"):
                if (
                    side not in self.feedback
                    or time.monotonic() - self.feedback[side][0] > self.cfg["playback"]["feedback_timeout_s"]
                ):
                    raise RuntimeError(f"{side} 从臂反馈缺失/超时")
            return np.concatenate([self.feedback[side][1] for side in ("left", "right")])

    def check_publishers(self) -> None:
        if self.driver_check:
            self.driver_check()
        for side, arm in self.cfg["playback"]["arms"].items():
            topics = [arm["command_topic"]]
            if self.cfg["playback"]["gripper_enabled"]:
                topics.append(arm["gripper_topic"])
            for topic in topics:
                if self.cfg["playback"]["check_other_publishers"] and self.node.count_publishers(topic) > 1:
                    raise RuntimeError(f"{topic} 存在其他发布者，请先暂停主臂遥操作/推理控制")
                if self.node.count_subscribers(topic) == 0:
                    raise RuntimeError(f"{side}: {topic} 无订阅者，请检查从臂/夹爪驱动")

    def publish(self, joints: np.ndarray, grippers: np.ndarray | None = None) -> None:
        if not np.isfinite(joints).all() or (grippers is not None and not np.isfinite(grippers).all()):
            raise ValueError("拒绝发布非有限控制指令")
        self.commanded = True
        for i, side in enumerate(("left", "right")):
            message = self.JointAngle()
            message.header.stamp = self.node.get_clock().now().to_msg()
            message.angle = joints[i * 7 : (i + 1) * 7].tolist() + [0.0]
            self.publishers[side].publish(message)
            if grippers is not None and self.cfg["playback"]["gripper_enabled"]:
                status = float(grippers[i])
                # Match the existing executor: a gripper command persists until
                # changed. Do not repeat SDK/serial transactions at arm rate.
                if self.last_grippers.get(side) == status:
                    continue
                grip = self.GripperStatus()
                grip.header.stamp = message.header.stamp
                grip.status = status
                self.grippers[side].publish(grip)
                self.last_grippers[side] = status

    def hold(self) -> None:
        # Never send an old recorded target after cancellation.
        if not self.commanded:
            return
        with contextlib.suppress(RuntimeError):
            self.check_publishers()
            self.publish(self.current())

    def close(self) -> None:
        self.rclpy.try_shutdown()
        self.thread.join(timeout=3)
        self.node.destroy_node()


@dataclass
class Episode:
    joints: np.ndarray
    grippers: np.ndarray
    fps: float
    label: str

    @classmethod
    def load(cls, path: Path) -> Episode:
        with np.load(path, allow_pickle=False) as data:
            return cls(data["joints"], data["grippers"], float(data["fps"]), str(data["label"]))


def checked_publish(robot: Any, joints: np.ndarray, grippers: np.ndarray | None, cfg: dict[str, Any]) -> None:
    current = robot.current()
    errors = np.abs(current - joints)
    index = int(np.argmax(errors))
    limit = cfg["playback"]["max_tracking_error_deg"]
    if errors[index] > limit:
        side = "左臂" if index < 7 else "右臂"
        raise RuntimeError(
            f"实际关节与回放目标偏差超限：{side}第 {index % 7 + 1} 关节，"
            f"反馈 {current[index]:.2f}°，目标 {joints[index]:.2f}°，"
            f"偏差 {errors[index]:.2f}° > {limit:.2f}°，已停止发送轨迹"
        )
    robot.publish(joints, grippers)


def align(robot: Any, target: np.ndarray, cfg: dict[str, Any], keyboard: KeyReader) -> None:
    start = robot.current()
    duration = max(
        cfg["alignment"]["min_duration_s"],
        1.5 * float(np.abs(target - start).max()) / cfg["alignment"]["speed_deg_s"],
    )
    hz = cfg["playback"]["command_hz"]
    steps = math.ceil(duration * hz)
    print(f"正在平稳对齐到本段第一帧，预计 {steps / hz:.1f} 秒；按 f 结束播放。", flush=True)
    begin = time.monotonic()
    for i in range(steps + 1):
        wait_until(begin + i / hz, keyboard)
        if time.monotonic() - (begin + i / hz) > cfg["playback"]["max_tick_lateness_s"]:
            raise RuntimeError("对齐调度超时")
        if i % max(1, int(hz)) == 0:
            robot.check_publishers()
        u = i / steps
        checked_publish(robot, start + (target - start) * (u * u * (3 - 2 * u)), None, cfg)
    deadline = time.monotonic() + cfg["alignment"]["settle_timeout_s"]
    while np.abs(robot.current() - target).max() > cfg["alignment"]["tolerance_deg"]:
        if time.monotonic() > deadline:
            raise RuntimeError("对齐超时，实际关节未到达第一帧，取消播放")
        wait_until(time.monotonic() + 1 / hz, keyboard)
        robot.check_publishers()
        checked_publish(robot, target, None, cfg)
    print("已确认关节到位，开始回放本段；按 f 可随时结束播放。", flush=True)


def play_episode(robot: Any, episode: Episode, cfg: dict[str, Any], keyboard: KeyReader) -> None:
    align(robot, episode.joints[0], cfg, keyboard)
    hz = cfg["playback"]["command_hz"]
    duration = (len(episode.joints) - 1) / episode.fps
    steps = math.ceil(duration * hz)
    begin = time.monotonic()
    for i in range(steps + 1):
        elapsed = min(i / hz, duration)
        wait_until(begin + elapsed, keyboard)
        if time.monotonic() - (begin + elapsed) > cfg["playback"]["max_tick_lateness_s"]:
            raise RuntimeError("回放调度超时，取消播放；不会突发追赶命令")
        if i % max(1, int(hz)) == 0:
            robot.check_publishers()
        position = min(elapsed * episode.fps, len(episode.joints) - 1)
        lower = int(position)
        upper = min(lower + 1, len(episode.joints) - 1)
        fraction = position - lower
        joints = episode.joints[lower] * (1 - fraction) + episode.joints[upper] * fraction
        checked_publish(robot, joints, episode.grippers[lower], cfg)


def running_driver(command: list[str], side_flag: str | None = None) -> bool:
    """Reuse an existing executable/side, rejecting a different IP or serial port."""
    expected_side = command[command.index(side_flag) + 1] if side_flag else None
    for entry in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = [part.decode() for part in entry.read_bytes().split(b"\0") if part]
        except (OSError, UnicodeError):
            continue
        if not argv or Path(argv[0]).name != Path(command[0]).name:
            continue
        if side_flag and (side_flag not in argv or argv[argv.index(side_flag) + 1 :][:1] != [expected_side]):
            continue
        for flag in ("-a", "-P", "-r") if side_flag == "-S" else ("-p",) if side_flag == "-s" else ():
            configured = command[command.index(flag) + 1]
            actual = argv[argv.index(flag) + 1 :][:1] if flag in argv else []
            if actual != [configured]:
                raise RuntimeError(
                    f"{command[0]} {expected_side}: 已运行实例的 {flag} 与 YAML 不一致，请先核对配置"
                )
        return True
    return False


@contextlib.contextmanager
def drivers(cfg: dict[str, Any]) -> Iterator[Callable[[], None]]:
    runtime = cfg["runtime"]
    commands: list[tuple[str, list[str], str | None, str | None]] = []
    if runtime.get("start_bridge", False):
        commands.append(("zenoh_bridge", runtime["bridge_command"], None, None))
    if runtime["start_slave_arms"]:
        for side in ("left", "right"):
            arm = cfg["robot"][side]
            commands.append(
                (
                    f"slave_{side}",
                    [
                        runtime["slave_executable"],
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
                    "-S",
                    None,
                )
            )
    if runtime.get("start_grippers", False) and cfg["playback"]["gripper_enabled"]:
        for side in ("left", "right"):
            port = runtime["gripper_ports"][side]
            commands.append(
                (f"gripper_{side}", [runtime["gripper_executable"], "-s", side, "-p", port], "-s", port)
            )
    needed = []
    for name, command, side_flag, port in commands:
        if not command or not all(isinstance(arg, str) and arg for arg in command):
            raise ValueError(f"{name}: 启动命令无效")
        if running_driver(command, side_flag):
            print(f"复用已运行的 {name}，退出回放时不会关闭它。", flush=True)
            continue
        if shutil.which(command[0]) is None:
            raise FileNotFoundError(f"{name}: 找不到可执行程序 {command[0]}")
        if port and not Path(port).exists():
            raise FileNotFoundError(f"{name}: 找不到夹爪串口 {port}，请修改 runtime.gripper_ports")
        needed.append((name, command))
    processes: list[tuple[str, subprocess.Popen, Path]] = []
    with contextlib.ExitStack() as stack:
        try:
            if needed:
                log_root = Path(runtime["driver_log_dir"]).expanduser()
                log_root.mkdir(parents=True, exist_ok=True)
                log_dir = Path(tempfile.mkdtemp(prefix="run_", dir=log_root))
                print(f"驱动启动日志：{log_dir}", flush=True)
                for name, command in needed:
                    log_path = log_dir / f"{name}.log"
                    handle = stack.enter_context(log_path.open("w"))
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    processes.append((name, process, log_path))
                    print(f"已启动 {name}（PID {process.pid}），日志：{log_path}", flush=True)

            def check_drivers() -> None:
                for name, process, log_path in processes:
                    status = process.poll()
                    if status is not None:
                        raise RuntimeError(f"{name} 驱动退出（状态 {status}），请查看日志：{log_path}")

            yield check_drivers
        finally:
            # Stop only children created by this invocation, in reverse launch order.
            for _, process, _ in reversed(processes):
                if process.poll() is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def select_episode(paths: list[Path]) -> int:
    choices: dict[int, int] = {}
    print("\n可回放的 episode：", flush=True)
    for position, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            label = str(data["label"])
        number = int(label.rsplit("_", 1)[-1])
        if number in choices:
            raise ValueError(f"episode 编号重复：{number}")
        choices[number] = position
        print(f"  编号 {number}：{label}", flush=True)
    first = next(iter(choices))
    while True:
        try:
            value = input(f"请输入起始 episode 编号并回车（直接回车选 {first}；输入 f 回车结束）：").strip()
        except EOFError as exc:
            raise CancelledError from exc
        if value.lower() == "f":
            raise CancelledError
        if not value:
            value = str(first)
        if value.isascii() and value.isdigit() and int(value) in choices:
            position = choices[int(value)]
            print(f"已选择 episode {int(value)}；随后按 → 才会对齐并播放，后续按编号顺序继续。", flush=True)
            return position
        print("编号无效，请输入上方列表中的 episode 编号。", flush=True)


def play(cfg: dict[str, Any], prepared: Path) -> None:
    paths = sorted(prepared.glob("episode_*.npz"))
    if not paths:
        raise ValueError("没有预处理后的 episode")
    try:
        start_index = select_episode(paths)
    except CancelledError:
        print("已结束播放，未启动驱动或发送运动指令。", flush=True)
        return
    with drivers(cfg) as check_drivers, Keyboard() as keyboard:
        robot = RosRobot(cfg)
        robot.driver_check = check_drivers
        try:
            for index, path in enumerate(paths):
                if index < start_index:
                    continue
                episode = Episode.load(path)
                print(
                    f"\n等待播放第 {index + 1}/{len(paths)} 段：{episode.label}\n"
                    "按右方向键 →：先对齐到第一帧，然后播放本段。\n按 f：结束本次播放。",
                    flush=True,
                )
                while True:
                    check_drivers()
                    keys = keyboard.read(0.1)
                    if "f" in keys:
                        raise CancelledError
                    if "right" in keys:
                        break
                deadline = time.monotonic() + cfg["playback"]["ready_timeout_s"]
                print("正在等待从臂反馈和控制话题就绪；按 f 可结束播放。", flush=True)
                last_report = -math.inf
                while True:
                    wait_until(time.monotonic() + 0.02, keyboard)
                    try:
                        robot.current()
                        robot.check_publishers()
                        break
                    except RuntimeError as exc:
                        if time.monotonic() - last_report >= 1:
                            last_report = time.monotonic()
                            print(f"尚未就绪：{exc}；按 f 结束。", flush=True)
                        if time.monotonic() > deadline:
                            raise
                play_episode(robot, episode, cfg, keyboard)
                print(f"第 {index + 1}/{len(paths)} 段播放完成。", flush=True)
            print("所选 episode 及后续各段已播放完毕，本次播放结束。", flush=True)
        except CancelledError:
            print("已按 f 结束播放，不再发送后续轨迹。", flush=True)
        finally:
            try:
                robot.hold()
            finally:
                robot.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "play"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    cfg = config(args.config)
    if args.mode == "prepare":
        prepare(cfg, args.prepared)
    else:
        play(cfg, args.prepared)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n播放已中断。", file=sys.stderr)
        sys.exit(130)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"\n回放终止：{exc}", file=sys.stderr)
        sys.exit(1)
