"""Independent online client; reuses tested hardware helpers without changing their files."""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from smoothing import FinalCommandPublisher, JointActionSmoother
from transport import OnlineClient as Client


def observation_class(base):
    from infer import image_bgr

    class TimestampedObservations(base):
        def request(self, reset):
            samples = self.snapshot()
            # Existing drivers expose host arrival timestamps. Keep original ROS
            # headers for audit; do not claim hardware exposure timestamps are synced.
            result = {
                "state": self.state_vector(samples),
                "capture_client_s": min(sample.arrival for sample in samples.values()),
            }
            names = list(self.cfg["streams"]) + list(self.cfg["cameras"])
            result["arrival_client_s"] = np.array([samples[name].arrival for name in names])
            headers = []
            for name in names:
                header = getattr(samples[name].value, "header", None)
                stamp = getattr(header, "stamp", None)
                headers.append(0 if stamp is None else stamp.sec * 1_000_000_000 + stamp.nanosec)
            result["source_header_ns"] = np.array(headers, dtype=np.int64)
            for name, camera in self.cfg["cameras"].items():
                result[name] = image_bgr(samples[name].value, camera)[:, :, ::-1].copy()
            return result

    return TimestampedObservations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.execute:
        parser.error("Choose --dry-run or --execute")
    raw = yaml.safe_load(args.config.read_text())
    sys.path.insert(0, raw["online"]["legacy_infer_dir"])
    from common import load_config

    cfg = load_config(args.config)
    print(f"在线 client 配置：{args.config.resolve()}", flush=True)
    if cfg["control"]["execution_hz"] != 25 or cfg["control"]["swap_action_arms"]:
        raise ValueError("Online runtime requires 25 Hz and native action arm ordering")
    if cfg["control"]["gripper_mode"] != "continuous" or not cfg["control"]["gripper_enabled"]:
        raise ValueError("Online runtime requires continuous grippers for exact executed prefixes")
    if not 0 < cfg["online"]["fast_observation_hz"] <= 25:
        raise ValueError("fast_observation_hz must be in (0,25]")
    if not isinstance(cfg["online"].get("log_assimilated_details", False), bool):
        raise ValueError("online.log_assimilated_details must be true/false")
    JointActionSmoother(cfg["control"].get("action_smoothing", {}))
    if args.check_config:
        print("在线配置检查通过；未连接机器人或模型。")
        return
    cfg["control"]["enabled"] = args.execute
    from infer import CancelledError

    try:
        run(cfg, cfg["online"]["address"])
    except CancelledError:
        print("本轮在线推理结束。")
    except (RuntimeError, ValueError, OSError, EOFError) as exc:
        raise SystemExit(f"在线推理终止：{exc}") from exc


def run(cfg: dict[str, Any], address: str) -> None:
    from common import validate_actions
    from infer import (
        JOINTS,
        BinaryGripperFilter,
        CancelledError,
        InferenceRobot,
        InputUnavailable,
        Keyboard,
        Observations as BaseObservations,
        check_keys,
        checked_publish,
        drivers,
        initialize_grippers,
        initialize_pose,
        replay_config,
    )

    observations_type = observation_class(BaseObservations)
    enabled = cfg["control"]["enabled"]
    mode = "真机执行" if enabled else "只预测、不发送关节及夹爪指令"
    print(f"模式：{mode}\n等待模型加载；按 f 结束。", flush=True)
    if cfg["control"].get("swap_action_arms", False):
        print(
            "左右动作交换已开启：模型左臂/夹爪 → 实体右侧，模型右臂/夹爪 → 实体左侧；init 不交换。",
            flush=True,
        )
    with Keyboard() as keyboard:  # noqa: SIM117 -- Client polling uses this already-entered keyboard.
        with contextlib.closing(
            Client(
                address,
                cfg["model"]["startup_timeout_s"],
                lambda: check_keys(keyboard, 0.005),
                cfg,
            )
        ) as client:
            # A prediction-only run reuses already-running input drivers and does
            # not launch gripper initialisation or create commands on its own.
            initial = cfg.get("initial_pose", {})
            if initial.get("enabled", False) and initial["mode"] == "movej" and enabled:
                print("按 → 执行 MoveJ init；按 f 结束。init 后还需按 → 开始模型执行。", flush=True)
                while True:
                    keys = keyboard.read(0.05)
                    if "f" in keys:
                        raise CancelledError
                    if "right" in keys:
                        break
                initialize_pose(None, cfg, keyboard)
            rcfg = replay_config(cfg)
            realman_enabled = cfg.get("realman_gripper", {}).get("enabled", False)
            end_grippers = realman_enabled and cfg["control"]["gripper_mode"] == "binary"
            if realman_enabled:
                rcfg["runtime"] = {**rcfg["runtime"], "start_grippers": False}
            if end_grippers:
                rcfg["runtime"] = {**rcfg["runtime"], "start_grippers": False}
                rcfg["playback"]["gripper_enabled"] = False
            if not enabled:
                rcfg["runtime"] = {
                    **cfg["runtime"],
                    "start_bridge": False,
                    "start_slave_arms": False,
                    "start_grippers": False,
                }
            with drivers(rcfg) as check_drivers:
                robot = InferenceRobot(rcfg)
                robot.gripper_commands_enabled = enabled and cfg["control"]["gripper_enabled"]
                robot.driver_check = check_drivers
                try:
                    observations = observations_type(robot, cfg)
                    if end_grippers:
                        from realman_gripper import RealmanGrippers

                        robot.end_grippers = RealmanGrippers(cfg, observations.inputs)
                    last_graph_check = -math.inf

                    def healthy() -> None:
                        nonlocal last_graph_check
                        check_keys(keyboard)
                        check_drivers()
                        if robot.end_grippers is not None:
                            robot.end_grippers.check()
                        observations.state()
                        try:
                            robot.current()
                        except RuntimeError as exc:
                            if "反馈缺失/超时" in str(exc):
                                raise InputUnavailable(str(exc)) from exc
                            raise
                        if enabled and time.monotonic() - last_graph_check >= 1:
                            robot.check_publishers()
                            last_graph_check = time.monotonic()

                    if initial.get("enabled", False) and initial["mode"] == "servo" and enabled:
                        print(
                            "驱动已启动。按 → 平稳移动到 init 并闭合夹爪；按 f 退出。"
                            "完成后还需再按 → 开始模型执行。",
                            flush=True,
                        )
                        while True:
                            check_drivers()
                            keys = keyboard.read(0.05)
                            if "f" in keys:
                                raise CancelledError
                            if "right" in keys:
                                break
                        initialize_pose(robot, cfg, keyboard)
                    initialize_grippers(robot, observations, cfg, keyboard)
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
                            if enabled and cfg.get("initial_pose", {}).get("enabled", False):
                                initial = cfg["initial_pose"]
                                target = np.asarray(initial["left_degrees"] + initial["right_degrees"])
                                discrepancy = float(np.max(np.abs(robot.current() - target)))
                                if discrepancy > initial["tolerance_deg"]:
                                    raise RuntimeError(
                                        f"ROS 反馈与 init 目标相差 {discrepancy:.3f}°，"
                                        "尚未确认反馈同步，拒绝开始模型动作"
                                    )
                            observations.request(True)
                            break
                        except RuntimeError as exc:
                            if time.monotonic() > ready_deadline:
                                raise
                            print(f"等待输入：{exc}", flush=True)
                    begin = time.monotonic()

                    def poll() -> None:
                        healthy()
                        if time.monotonic() - begin >= cfg["control"]["max_duration_s"]:
                            raise CancelledError
                        # Bound per-poll wait below a control tick while also avoiding busy spins.
                        check_keys(keyboard, 0.001)

                    from execution import run_online as run_async

                    current_grips = np.array(
                        [robot.last_grippers.get(side, 0.0) for side in ("left", "right")]
                    )
                    if robot.end_grippers is not None:
                        sent, _ = robot.end_grippers.snapshot()
                        for i, side in enumerate(("left", "right")):
                            if side in sent:
                                current_grips[i] = sent[side]
                    gripper_filter = (
                        BinaryGripperFilter(cfg["control"], current_grips, time.monotonic())
                        if cfg["control"]["gripper_mode"] == "binary"
                        else None
                    )

                    def send_action(action: np.ndarray) -> None:
                        grips = (
                            gripper_filter.update(action[[7, 15]], time.monotonic())
                            if gripper_filter is not None
                            else action[[7, 15]]
                        )
                        if enabled:
                            try:
                                checked_publish(robot, np.rad2deg(action[JOINTS]), grips, rcfg)
                            except RuntimeError as exc:
                                if "反馈缺失/超时" in str(exc):
                                    raise InputUnavailable(str(exc)) from exc
                                raise

                    publish_action = FinalCommandPublisher(
                        cfg["control"].get("action_smoothing", {}),
                        observations.state,
                        lambda action, state: validate_actions(
                            action[None], state, cfg, binarize_grippers=False
                        )[0],
                        send_action,
                    )
                    smoothing = cfg["control"].get("action_smoothing", {})
                    print(f"最终下发平滑：{smoothing}", flush=True)

                    def recover_inputs(error: Exception) -> None:
                        timeout = float(cfg["control"].get("input_recovery_timeout_s", 10.0))
                        deadline = time.monotonic() + timeout
                        print(
                            f"输入异常：{error}；暂停下发动作，最多等待 {timeout:g} 秒恢复；按 f 退出。",
                            flush=True,
                        )
                        while True:
                            check_keys(keyboard, 0.02)
                            check_drivers()
                            if time.monotonic() - begin >= cfg["control"]["max_duration_s"]:
                                raise CancelledError
                            try:
                                healthy()
                                observations.request(True)
                                return
                            except InputUnavailable as exc:
                                if time.monotonic() >= deadline:
                                    raise RuntimeError(f"输入在 {timeout:g} 秒内未恢复：{exc}") from exc

                    if enabled:
                        robot.hold()
                    run_async(
                        client,
                        observations,
                        cfg,
                        poll,
                        publish_action,
                        robot.hold if enabled else lambda: None,
                        recover_inputs,
                    )
                finally:
                    if enabled:
                        robot.hold()
                    robot.close()


if __name__ == "__main__":
    main()
