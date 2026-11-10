"""Persistent online-filtering server; network receive, DINO and expert stay independent."""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from model import fast_packet, load_runner, raw_batch
from protocol import VERSION, address_socket, receive, send


class Session:
    def __init__(self, connection, runner, cfg):
        self.connection, self.runner, self.cfg = connection, runner, cfg
        self.condition = threading.Condition()
        self.send_lock = threading.Lock()
        self.pending = None
        self.active = None
        self.error = None
        self.closed = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ActionExpert")
        self.worker = threading.Thread(target=self.encode_loop, name="DINO", daemon=True)
        self.worker.start()

    def reply(self, **values):
        with self.send_lock:
            if not self.closed:
                send(self.connection, **values)

    def encode_loop(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                data, generation = self.pending
                self.pending = None
            try:
                packet = fast_packet(self.runner, data, generation)
                with self.condition:
                    if self.active == (int(data["request_id"]), generation):
                        self.runner.policy.push_fast_observation(packet)
                print(
                    json.dumps(
                        {
                            "event": "dino_ready",
                            "request_id": int(data["request_id"]),
                            "capture_step": packet.capture_step,
                            "age_s": packet.ready_timestamp - packet.capture_timestamp,
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:
                with self.condition:
                    if self.active == (int(data["request_id"]), generation):
                        self.error = exc

    def predict(self, data, generation):
        started = time.monotonic()
        assimilated = []
        try:
            torch = self.runner.torch

            def observe(step, total, gen):
                with self.condition:
                    if self.closed or self.active != (int(data["request_id"]), generation):
                        raise RuntimeError("Generation cancelled")
                    if self.error is not None:
                        raise RuntimeError(f"DINO failed: {self.error}")
                return None

            def record(step, capture_step, updater):
                assimilated.append([step, capture_step])
                print(
                    json.dumps(
                        {
                            "event": "assimilated",
                            "request_id": int(data["request_id"]),
                            "flow_step": step,
                            "capture_step": capture_step,
                        }
                    ),
                    flush=True,
                )

            with torch.inference_mode():
                processed = self.runner.pre(raw_batch(self.runner, data))
                chunk = self.runner.policy.predict_action_chunk(
                    processed,
                    online_generation=generation,
                    initial_capture_timestamp=float(data["capture_server_s"]),
                    observation_callback=observe,
                    assimilation_callback=record,
                )
                chunk = self.runner.post(chunk)[0].detach().cpu().float().numpy()
            with self.condition:
                if self.error is not None:
                    raise RuntimeError(f"DINO failed: {self.error}")
                if self.closed:
                    return
                self.active = None
                self.pending = None
            self.reply(
                kind="actions",
                request_id=data["request_id"],
                origin=data["origin"],
                actions=chunk,
                latency_s=time.monotonic() - started,
                assimilated=assimilated,
            )
        except Exception as exc:
            with self.condition:
                self.active = None
                self.pending = None
            with contextlib.suppress(OSError):
                self.reply(error=str(exc))

    def handle(self, data):
        kind = str(data["kind"])
        if kind == "ping":
            received = time.monotonic()
            self.reply(kind="pong", received=received, sent=time.monotonic())
            return
        if kind not in ("begin", "fast"):
            raise ValueError(f"Unknown message: {kind}")
        capture = float(data["capture_server_s"])
        now = time.monotonic()
        if not now - self.cfg["online"]["max_observation_age_s"] <= capture <= now + 0.025:
            raise ValueError("Observation stale or clock mapping invalid")
        if kind == "begin":
            with self.condition:
                if self.active is not None:
                    raise RuntimeError("Only one outstanding chunk is permitted")
                generation = self.runner.policy.begin_online_generation()
                self.active = (int(data["request_id"]), generation)
                self.error = None
                self.pending = None
            self.executor.submit(self.predict, data, generation)
        else:
            with self.condition:
                if self.active is None or self.active[0] != int(data["request_id"]):
                    return  # A late packet from an already completed generation.
                self.pending = (data, self.active[1])
                self.condition.notify()

    def close(self):
        with self.condition:
            self.closed = True
            self.pending = None
            self.condition.notify_all()
        with contextlib.suppress(OSError):
            self.connection.shutdown(socket.SHUT_RDWR)
        self.worker.join()
        self.executor.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument(
        "--check", action="store_true", help="Load and predict zeros; never connects to robot"
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg["control"]["execution_hz"] != 25:
        raise ValueError("This checkpoint uses 25 Hz")
    runner = load_runner(cfg)
    if args.check:
        import numpy as np

        batch = {
            name: np.zeros((cam["height"], cam["width"], 3), np.uint8) for name, cam in cfg["cameras"].items()
        }
        batch["state"] = np.zeros(14, np.float32)
        print(runner.predict(batch).shape)
        return
    family, address = address_socket(cfg["online"]["address"])
    if family == socket.AF_UNIX:
        Path(address).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if Path(address).exists():
            raise RuntimeError(f"Socket exists; check for an existing server: {address}")
    try:
        with socket.socket(family) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(address)
            if family == socket.AF_UNIX:
                Path(address).chmod(0o600)
            listener.listen(1)
            print(f"在线模型服务就绪：{cfg['online']['address']}", flush=True)
            while True:
                connection, _ = listener.accept()
                connection.settimeout(cfg["model"]["prediction_timeout_s"])
                session = Session(connection, runner, cfg)
                try:
                    session.reply(
                        kind="ready",
                        version=VERSION,
                        hz=25,
                        chunk_size=runner.policy.config.chunk_size,
                        max_delay=runner.policy.config.max_inference_delay,
                        task=cfg["model"]["task"],
                    )
                    while True:
                        session.handle(receive(connection))
                except (EOFError, OSError):
                    pass
                except Exception as exc:
                    session.reply(error=str(exc))
                finally:
                    with contextlib.suppress(OSError):
                        session.close()
                    connection.close()
    finally:
        if family == socket.AF_UNIX:
            Path(address).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
