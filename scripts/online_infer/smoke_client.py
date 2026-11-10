"""Synthetic socket smoke test. No ROS, robot imports, or robot commands."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml
from transport import OnlineClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    client = OnlineClient(cfg["online"]["address"], 300, lambda: None, cfg)
    try:
        for request_id in range(1, 3):
            start = time.monotonic()

            def observation(request_id=request_id):
                return {
                    "state": np.zeros(14, np.float32),
                    "capture_client_s": time.monotonic(),
                    "origin": 0,
                    "request_id": request_id,
                    **{
                        name: np.zeros((cam["height"], cam["width"], 3), np.uint8)
                        for name, cam in cfg["cameras"].items()
                    },
                }

            client.submit(observation(), begin=True)
            last_step = 0
            while True:
                result = client.take_response()
                if result is not None:
                    actions = result["actions"]
                    assert actions.shape == (50, 16) and np.isfinite(actions).all()
                    assert int(result["request_id"]) == request_id
                    print(
                        f"request={request_id} latency={float(result['latency_s']):.3f}s "
                        f"assimilated={result['assimilated'].tolist()}",
                        flush=True,
                    )
                    break
                elapsed = time.monotonic() - start
                step = int(elapsed * 25)
                if 0 < step <= int(client.metadata["max_delay"]) and step != last_step:
                    packet = observation()
                    packet.update(capture_step=step, executed_prefix=np.zeros((step, 16), np.float32))
                    client.submit(packet, begin=False)
                    last_step = step
                if elapsed > 300:
                    raise RuntimeError("Smoke test timeout")
                time.sleep(0.005)
    finally:
        client.close()


if __name__ == "__main__":
    main()
