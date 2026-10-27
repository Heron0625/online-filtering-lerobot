"""Keyboard-driven local RB75 PI05 rollout, with a separate policy worker."""

from __future__ import annotations

import argparse
import contextlib
import math
import socket
import struct
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from common import (
    JOINTS,
    MAX_PACKET_BYTES,
    decode_packet,
    encode_packet,
    load_config,
    replay_config,
    validate_actions,
)

# Reuse the recorder's exact units/image decoder and the tested replay control path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collect"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reply"))
from record import Inputs, image_bgr, numeric_value  # noqa: E402
from reply import CancelledError, Keyboard, RosRobot, checked_publish, drivers  # noqa: E402


def check_keys(keyboard: Any, timeout: float = 0) -> None:
    if "f" in keyboard.read(timeout):
        raise CancelledError


class Client:
    def __init__(self, address: str, timeout: float, poll: Callable[[], None]) -> None:
        self.connection = socket.socket(socket.AF_UNIX)
        self.connection.setblocking(False)
        deadline = time.monotonic() + timeout
        try:
            while True:
                poll()
                try:
                    self.connection.connect(address)
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() > deadline:
                        raise RuntimeError("模型启动超时，请查看 policy_worker.log") from None
            if not bool(self.receive(deadline, poll).get("ready", False)):
                raise RuntimeError("模型服务未就绪")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def read_exact(self, count: int, deadline: float, poll: Callable[[], None]) -> bytes:
        data = bytearray()
        while len(data) < count:
            poll()
            if time.monotonic() > deadline:
                raise RuntimeError("模型响应超时，停止本次推理，不追赶过期动作")
            try:
                part = self.connection.recv(min(count - len(data), 1024 * 1024))
            except BlockingIOError:
                continue
            if not part:
                raise RuntimeError("模型进程断开，请查看 policy_worker.log")
            data.extend(part)
        return bytes(data)

    def receive(self, deadline: float, poll: Callable[[], None]) -> dict[str, np.ndarray]:
        size = struct.unpack("!I", self.read_exact(4, deadline, poll))[0]
        if not 0 < size <= MAX_PACKET_BYTES:
            raise RuntimeError("模型响应长度无效")
        data = decode_packet(self.read_exact(size, deadline, poll))
        if "error" in data:
            raise RuntimeError(f"模型推理失败：{str(data['error'])}")
        return data

    def predict(
        self, request: dict[str, Any], timeout: float, poll: Callable[[], None]
    ) -> dict[str, np.ndarray]:
        payload = memoryview(encode_packet(**request))
        deadline = time.monotonic() + timeout
        while payload:
            poll()
            if time.monotonic() > deadline:
                raise RuntimeError("发送观测超时")
            try:
                sent = self.connection.send(payload)
            except BlockingIOError:
                continue
            if sent == 0:
                raise RuntimeError("模型进程断开")
            payload = payload[sent:]
        return self.receive(deadline, poll)


class Observations:
    def __init__(self, robot: Any, cfg: dict[str, Any]) -> None:
        from rclpy.qos import qos_profile_sensor_data
        from rosidl_runtime_py.utilities import get_message

        self.cfg = cfg
        self.inputs = Inputs(
            {
                "streams": cfg["streams"],
                "cameras": cfg["cameras"],
                "recording": cfg["observation"],
            }
        )

        def callback(name: str, stream: dict[str, Any] | None, message: Any) -> None:
            try:
                value = numeric_value(message, stream) if stream else message
                self.inputs.put(name, value)
            except Exception as exc:
                with self.inputs.lock:
                    self.inputs.error = f"{name}: {exc}"

        for name, stream in cfg["streams"].items():
            robot.node.create_subscription(
                get_message(stream["type"]),
                stream["topic"],
                lambda message, n=name, s=stream: callback(n, s, message),
                qos_profile_sensor_data,
            )
        for name, camera in cfg["cameras"].items():
            robot.node.create_subscription(
                get_message("sensor_msgs/msg/Image"),
                camera["topic"],
                lambda message, n=name: callback(n, None, message),
                qos_profile_sensor_data,
            )

    def state(self) -> np.ndarray:
        return self.inputs.vector(self.inputs.snapshot(time.monotonic()), "state")

    def request(self, reset: bool) -> dict[str, Any]:
        samples = self.inputs.snapshot(time.monotonic())
        result = {"state": self.inputs.vector(samples, "state"), "reset": np.array(reset)}
        for name, camera in self.cfg["cameras"].items():
            # Collector stores BGR JPEG and exports RGB; model gets the same RGB.
            result[name] = image_bgr(samples[name].value, camera)[:, :, ::-1].copy()
        return result


def execute_chunk(
    robot: Any, observations: Any, actions: np.ndarray, cfg: dict[str, Any], poll: Callable[[], None]
) -> None:
    state = observations.state()
    actions = validate_actions(actions, state, cfg)
    hz = cfg["control"]["command_hz"]
    action_hz = cfg["control"]["action_hz"]
    # Keep the data's 50 Hz timing; slower model computation adds an explicit
    # stationary wait between chunks rather than stretching or replaying old chunks.
    positions = np.vstack([state[JOINTS], actions[:, JOINTS]])
    duration = len(actions) / action_hz
    steps = math.ceil(duration * hz)
    begin = time.monotonic()
    rcfg = replay_config(cfg)
    for index in range(steps + 1):
        elapsed = min(index / hz, duration)
        deadline = begin + elapsed
        while time.monotonic() < deadline:
            poll()
        poll()
        if time.monotonic() - deadline > cfg["control"]["max_tick_lateness_s"]:
            raise RuntimeError("控制调度超时，停止推理，不突发追赶命令")
        position = elapsed * action_hz
        lower = min(int(position), len(actions) - 1)
        fraction = min(position - lower, 1.0)
        target = positions[lower] * (1 - fraction) + positions[lower + 1] * fraction
        if cfg["control"]["enabled"]:
            checked_publish(robot, np.rad2deg(target), actions[lower, [7, 15]], rcfg)


def run(cfg: dict[str, Any], address: str) -> None:
    enabled = cfg["control"]["enabled"]
    mode = "真机执行" if enabled else "只预测、不发送关节及夹爪指令"
    print(f"模式：{mode}\n等待模型加载；按 f 结束。", flush=True)
    with Keyboard() as keyboard:  # noqa: SIM117 -- Client polling uses this already-entered keyboard.
        with contextlib.closing(
            Client(
                address,
                cfg["model"]["startup_timeout_s"],
                lambda: check_keys(keyboard, 0.005),
            )
        ) as client:
            # A prediction-only run reuses already-running input drivers and does
            # not launch gripper initialisation or create commands on its own.
            rcfg = replay_config(cfg)
            if not enabled:
                rcfg["runtime"] = {
                    **cfg["runtime"],
                    "start_bridge": False,
                    "start_slave_arms": False,
                    "start_grippers": False,
                }
            with drivers(rcfg) as check_drivers:
                robot = RosRobot(rcfg)
                robot.driver_check = check_drivers
                try:
                    observations = Observations(robot, cfg)
                    last_graph_check = -math.inf

                    def healthy() -> None:
                        nonlocal last_graph_check
                        check_keys(keyboard)
                        check_drivers()
                        observations.state()
                        robot.current()
                        if enabled and time.monotonic() - last_graph_check >= 1:
                            robot.check_publishers()
                            last_graph_check = time.monotonic()

                    print("按 → 开始一轮推理；按 f 结束。请先将机械臂置于训练任务起始状态。", flush=True)
                    while True:
                        check_drivers()
                        keys = keyboard.read(0.05)
                        if "f" in keys:
                            raise CancelledError
                        if "right" in keys:
                            break
                    ready_deadline = time.monotonic() + cfg["control"]["ready_timeout_s"]
                    print("检查四相机和双臂状态；按 f 结束。", flush=True)
                    while True:
                        check_keys(keyboard, 0.02)
                        try:
                            healthy()
                            observations.request(True)
                            break
                        except RuntimeError as exc:
                            if time.monotonic() > ready_deadline:
                                raise
                            print(f"等待输入：{exc}", flush=True)
                    begin = time.monotonic()
                    reset = True

                    def poll() -> None:
                        healthy()
                        if time.monotonic() - begin >= cfg["control"]["max_duration_s"]:
                            raise CancelledError
                        # Bound per-poll wait below a control tick while also avoiding busy spins.
                        check_keys(keyboard, 0.001)

                    while True:
                        healthy()
                        if enabled:
                            robot.hold()
                        print("正在根据最新观测计算动作；按 f 结束。", flush=True)
                        response = client.predict(
                            observations.request(reset), cfg["model"]["prediction_timeout_s"], poll
                        )
                        reset = False
                        actions = validate_actions(response["actions"], observations.state(), cfg)
                        if len(actions) != cfg["model"]["execute_steps"]:
                            raise ValueError("模型返回动作块长度与 execute_steps 不一致")
                        print(
                            f"本块预测耗时 {float(response['latency_s']):.3f}s，"
                            f"{len(actions)} 步，动作时长 {len(actions) / cfg['control']['action_hz']:.3f}s；按 f 结束。",
                            flush=True,
                        )
                        execute_chunk(robot, observations, actions, cfg, poll)
                finally:
                    if enabled:
                        robot.hold()
                    robot.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--socket")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.check_config:
        print("配置检查通过；未加载权重、未连接机器人。")
        return
    if args.dry_run:
        cfg["control"]["enabled"] = False
    try:
        if not args.socket:
            parser.error("缺少模型 --socket")
        run(cfg, args.socket)
    except CancelledError:
        print("本轮推理结束（按键或时长上限），不再发送后续动作。", flush=True)
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"推理终止：{exc}") from exc


if __name__ == "__main__":
    main()
