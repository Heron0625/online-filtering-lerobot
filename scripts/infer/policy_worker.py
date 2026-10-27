"""PI05/online-filtering inference worker, without ROS imports or robot I/O."""

from __future__ import annotations

import argparse
import socket
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
from common import FEATURE_NAMES, MAX_PACKET_BYTES, decode_packet, encode_packet, load_config


def load_native_weights(policy: Any, path: Path, device: str) -> None:
    """Native deployment exports must load strictly; never fall back to random weights."""
    from safetensors.torch import load_model

    load_model(policy, str(path / "model.safetensors"), strict=True, device=device)


class PolicyRunner:
    def __init__(self, cfg: dict[str, Any]) -> None:
        import torch

        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.online_filtering_pi05.configuration_online_filtering_pi05 import (
            OnlineFilteringPI05Config,  # Registers the checkpoint's config type.
        )
        from lerobot.policies.online_filtering_pi05.modeling_online_filtering_pi05 import (
            OnlineFilteringPI05Policy,
        )
        from lerobot.policies.online_filtering_pi05.processor_online_filtering_pi05 import (
            reconcile_online_filtering_pi05_pretrained_processors,
        )
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        self.torch = torch
        self.cfg = cfg
        path = Path(cfg["model"]["checkpoint"]).expanduser()
        for filename in (
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        ):
            if not (path / filename).is_file():
                raise FileNotFoundError(f"缺少 {path / filename}；请下载完整 pretrained_model 目录")
        pcfg = PreTrainedConfig.from_pretrained(path, local_files_only=True)
        if not isinstance(pcfg, PI05Config) or pcfg.type not in ("pi05", "online_filtering_pi05"):
            raise ValueError("checkpoint 必须是 pi05 或 online_filtering_pi05")
        if pcfg.input_features["observation.state"].shape != (16,) or pcfg.output_features[
            "action"
        ].shape != (16,):
            raise ValueError("模型 state/action 必须均为 16 维，与 RB75 数采一致")
        if pcfg.action_feature_names is not None and pcfg.action_feature_names != FEATURE_NAMES:
            raise ValueError("checkpoint 的 action_feature_names 与采集关节顺序不一致")
        expected = {f"observation.images.{name}" for name in cfg["cameras"]}
        if set(pcfg.image_features) != expected:
            raise ValueError(
                f"模型相机键 {set(pcfg.image_features)} 与四路数采键 {expected} 不符；训练时保留原键"
            )
        for name, camera in cfg["cameras"].items():
            shape = pcfg.image_features[f"observation.images.{name}"].shape
            if tuple(shape) != (3, camera["height"], camera["width"]):
                raise ValueError(f"{name}: checkpoint 图像尺寸 {shape} 与采集配置不符")
        if pcfg.rtc_config is not None and pcfg.rtc_config.enabled:
            raise ValueError("本接口使用分块停等推理，请使用未启用 RTC 的 checkpoint")
        if pcfg.use_visual_memory or pcfg.use_proprioceptive_memory:
            raise ValueError("本接口先支持标准单帧 pi05，请使用未启用 MEM 的 checkpoint")
        if cfg["model"]["execute_steps"] > pcfg.chunk_size:
            raise ValueError("execute_steps 不能超过模型 chunk_size")
        pcfg.device = cfg["model"]["device"]
        pcfg.dtype = cfg["model"]["dtype"]
        pcfg.compile_model = False
        if isinstance(pcfg, OnlineFilteringPI05Config):
            from transformers import DINOv3ViTConfig, DINOv3ViTModel

            from lerobot.policies.pi05.online_filtering import DINOv3PatchEncoder

            # Backbone weights are already in the native checkpoint. Only its exact
            # architecture config is needed here; avoid an unrelated Hub weight download.
            dino_path = cfg["model"].get("dinov3_config") or pcfg.dinov3_model_id
            backbone_config = DINOv3ViTConfig.from_pretrained(dino_path, local_files_only=True)
            encoder = DINOv3PatchEncoder(DINOv3ViTModel(backbone_config), freeze=pcfg.freeze_dinov3)
            pcfg.fast_encoder_device = pcfg.device
            self.policy = OnlineFilteringPI05Policy(pcfg, fast_encoder=encoder)
        else:
            self.policy = PI05Policy(pcfg)
        load_native_weights(self.policy, path, pcfg.device)
        self.policy.eval()
        overrides = {"device_processor": {"device": pcfg.device}}
        if cfg["model"].get("tokenizer_name"):
            overrides["tokenizer_processor"] = {"tokenizer_name": cfg["model"]["tokenizer_name"]}
        self.pre, self.post = make_pre_post_processors(
            pcfg,
            pretrained_path=path,
            preprocessor_overrides=overrides,
        )
        if isinstance(pcfg, OnlineFilteringPI05Config):
            self.pre, self.post = reconcile_online_filtering_pi05_pretrained_processors(
                pcfg, self.pre, self.post
            )
        print(f"{pcfg.type} 已严格加载：{path}，device={pcfg.device}，dtype={pcfg.dtype}", flush=True)
        print("运行模式：分块停等；每块使用初始观测，不接收块内异步更新。", flush=True)

    def predict(self, data: dict[str, np.ndarray]) -> np.ndarray:
        torch = self.torch
        state = data["state"]
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("模型输入 state 无效")
        if bool(data.get("reset", False)):
            self.policy.reset()
            self.pre.reset()
            self.post.reset()
        batch: dict[str, Any] = {
            "observation.state": torch.from_numpy(state.astype(np.float32)),
            "task": self.cfg["model"]["task"],
        }
        for name, camera in self.cfg["cameras"].items():
            rgb = data[name]
            if rgb.dtype != np.uint8 or rgb.shape != (camera["height"], camera["width"], 3):
                raise ValueError(f"{name}: 必须为 HWC uint8 RGB 图像")
            batch[f"observation.images.{name}"] = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
        with torch.inference_mode():
            processed = self.pre(batch)
            # Loaded processor pair reconnects relative/absolute steps. Postprocess
            # the whole chunk with the SAME observation anchor before returning it.
            actions = self.policy.predict_action_chunk(processed)
            actions = self.post(actions)
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != 16:
            raise ValueError(f"模型输出必须为 [1,T,16]，得到 {actions.shape}")
        if actions.shape[1] < self.cfg["model"]["execute_steps"] or not torch.isfinite(actions).all():
            raise ValueError("模型输出长度不足或含非有限数值")
        return actions[0, : self.cfg["model"]["execute_steps"]].detach().cpu().float().numpy()


def receive_exact(connection: socket.socket, count: int) -> bytes:
    data = bytearray()
    while len(data) < count:
        part = connection.recv(count - len(data))
        if not part:
            raise EOFError
        data.extend(part)
    return bytes(data)


def serve(cfg: dict[str, Any], address: str) -> None:
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(address)
        server.listen(1)
        runner = PolicyRunner(cfg)
        print("模型服务就绪，等待本机 ROS 控制端连接。", flush=True)
        connection, _ = server.accept()
        with connection:
            connection.sendall(encode_packet(ready=True))
            while True:
                try:
                    size = struct.unpack("!I", receive_exact(connection, 4))[0]
                    if not 0 < size <= MAX_PACKET_BYTES:
                        raise ValueError("无效 IPC 包长度")
                    data = decode_packet(receive_exact(connection, size))
                    begin = time.monotonic()
                    actions = runner.predict(data)
                    connection.sendall(encode_packet(actions=actions, latency_s=time.monotonic() - begin))
                except EOFError:
                    return
                except Exception as exc:
                    connection.sendall(encode_packet(error=str(exc)))
                    raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--socket")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.check:
        runner = PolicyRunner(cfg)
        # Offline smoke test uses zeros, NEVER sends these predictions to a robot.
        data = {name: np.zeros((c["height"], c["width"], 3), np.uint8) for name, c in cfg["cameras"].items()}
        actions = runner.predict({"state": np.zeros(16, np.float32), "reset": np.array(True), **data})
        if actions.shape != (cfg["model"]["execute_steps"], 16) or not np.isfinite(actions).all():
            raise ValueError(f"模型输出无效：{actions.shape}")
        print(f"真实模型与预处理试算通过：{actions.shape}；未连接机器人。", flush=True)
    elif args.socket:
        serve(cfg, args.socket)
    else:
        parser.error("需要 --socket 或 --check")


if __name__ == "__main__":
    main()
