"""Configuration, numeric checks and local IPC shared by ROS and policy Python."""

from __future__ import annotations

import io
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import yaml

JOINTS = [*range(7), *range(8, 15)]
PARTS = ("left_joint", "left_gripper", "right_joint", "right_gripper")
FEATURE_NAMES = [
    f"{side}_joint_{i + 1}" if i < 7 else f"{side}_gripper" for side in ("left", "right") for i in range(8)
]
MAX_PACKET_BYTES = 64 * 1024 * 1024


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        cfg = yaml.safe_load(handle)
    for section, keys in {
        "model": ("startup_timeout_s", "prediction_timeout_s", "execute_steps"),
        "control": (
            "action_hz",
            "command_hz",
            "max_joint_speed_deg_s",
            "max_first_step_deg",
            "max_tracking_error_deg",
            "max_tick_lateness_s",
            "max_duration_s",
            "feedback_timeout_s",
            "ready_timeout_s",
        ),
        "observation": ("max_age_ms", "max_span_ms"),
    }.items():
        for key in keys:
            value = cfg[section][key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{section}.{key} 必须为有限正数")
    if not isinstance(cfg["model"]["execute_steps"], int):
        raise ValueError("model.execute_steps 必须是整数")
    if cfg["model"]["dtype"] not in ("bfloat16", "float32"):
        raise ValueError("model.dtype 只支持 bfloat16/float32")
    if not isinstance(cfg["control"]["enabled"], bool):
        raise ValueError("control.enabled 必须是 true/false")
    if not isinstance(cfg["control"]["gripper_enabled"], bool):
        raise ValueError("control.gripper_enabled 必须是 true/false")
    if cfg["control"]["command_hz"] < cfg["control"]["action_hz"]:
        raise ValueError("command_hz 不能小于训练数据对应的 action_hz")
    if cfg["control"]["max_tick_lateness_s"] >= 1 / cfg["control"]["command_hz"]:
        raise ValueError("max_tick_lateness_s 必须小于控制周期")
    if not str(cfg["model"]["task"]).strip() or "请填写" in cfg["model"]["task"]:
        raise ValueError("请填写 model.task，使用与训练任务一致的具体指令")
    if set(cfg["streams"]) != {f"state_{part}" for part in PARTS}:
        raise ValueError("streams 必须按采集定义配置四路 state 输入")
    for name, stream in cfg["streams"].items():
        if stream["size"] != (7 if name.endswith("joint") else 1):
            raise ValueError(f"{name} size 不匹配")
        if not stream["topic"].startswith("/"):
            raise ValueError("stream topic 必须是绝对话题")
        for key, default in (("scale", 1.0), ("offset", 0.0)):
            if not math.isfinite(stream.get(key, default)):
                raise ValueError(f"{name}.{key} 无效")
    if set(cfg["cameras"]) != {"head", "chest", "left", "right"}:
        raise ValueError("必须配置 head/chest/left/right 四路相机")
    for camera in cfg["cameras"].values():
        if not camera["topic"].startswith("/") or min(camera["width"], camera["height"], camera["fps"]) <= 0:
            raise ValueError("相机话题或尺寸/FPS 无效")
    if set(cfg["arms"]) != {"left", "right"}:
        raise ValueError("arms 必须配置左右臂")
    for arm in cfg["arms"].values():
        if any(not arm[key].startswith("/") for key in ("command_topic", "feedback_topic", "gripper_topic")):
            raise ValueError("arm topic 必须是绝对话题")
    if not math.isfinite(cfg["control"]["gripper_min"]) or not math.isfinite(cfg["control"]["gripper_max"]):
        raise ValueError("夹爪范围必须有限")
    if cfg["control"]["gripper_min"] >= cfg["control"]["gripper_max"]:
        raise ValueError("夹爪范围无效")
    return cfg


def replay_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Adapt to the proven reply driver, publisher and tracking checks."""
    return {
        "runtime": cfg["runtime"],
        "robot": cfg["robot"],
        "playback": {
            **cfg["control"],
            "arms": cfg["arms"],
            "check_other_publishers": True,
        },
    }


def validate_actions(actions: np.ndarray, state: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Postprocessed actions are absolute radians + raw gripper commands, as in collect."""
    result = np.asarray(actions, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 16 or len(result) < 1 or not np.isfinite(result).all():
        raise ValueError(f"模型必须输出有限的 [T,16] 动作，得到 {result.shape}")
    state = np.asarray(state)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("当前 state 必须是有限 16 维")
    control = cfg["control"]
    # Check the entire received prefix before sending any of it; never clip a bad policy output.
    if ((result[:, [7, 15]] < control["gripper_min"]) | (result[:, [7, 15]] > control["gripper_max"])).any():
        raise ValueError("模型夹爪命令超出范围，拒绝执行整块动作")
    first_delta = np.abs(np.rad2deg(result[0, JOINTS] - state[JOINTS]))
    if first_delta.max() > control["max_first_step_deg"]:
        raise ValueError(f"首个动作离当前关节过远：{first_delta.max():.2f}°，不自动对齐模型目标")
    # The first action gets one action period to reach its target as well.
    joints = np.vstack([state[JOINTS], result[:, JOINTS]])
    speed = np.abs(np.rad2deg(np.diff(joints, axis=0))) * control["action_hz"]
    if speed.max() > control["max_joint_speed_deg_s"]:
        raise ValueError(f"模型动作关节速度 {speed.max():.2f}°/s 超限，拒绝执行整块动作")
    return result


def encode_packet(**arrays: Any) -> bytes:
    output = io.BytesIO()
    np.savez(output, **arrays)
    data = output.getvalue()
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError("IPC 数据包过大")
    return struct.pack("!I", len(data)) + data


def decode_packet(data: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}
