"""25 Hz client execution; action history is owned by the control thread."""

from __future__ import annotations

import time

import numpy as np
from diagnostics import QueueDiagnostics
from timeline import Timeline, future_actions, merge_actions


def run_online(client, observations, cfg, poll, publish, hold, recover=None):
    diagnostics = QueueDiagnostics(cfg["control"]["execution_hz"])
    try:
        return _run_online(client, observations, cfg, poll, publish, diagnostics)
    finally:
        diagnostics.finish(time.monotonic())


def _run_online(client, observations, cfg, poll, publish, diagnostics):
    # An input fault exits this session. Reconnect to create a clean generation;
    # never reuse a partially consumed transport response after a sensor fault.
    from common import validate_actions

    timeline = Timeline()
    period = 1 / cfg["control"]["execution_hz"]
    remaining = np.empty((0, 16), np.float32)
    pending = None
    sequence = 0
    last_capture = None
    max_delay = int(client.metadata["max_delay"])
    last_action = None
    executed_since_result = 0
    next_observation = 0.0

    def submit_begin():
        nonlocal sequence, pending, last_capture
        packet = observations.request(False)
        origin = timeline.index_at(float(packet["capture_client_s"]))
        sequence += 1
        packet.update(request_id=sequence, origin=origin)
        client.submit(packet, begin=True)
        pending = (sequence, origin, time.monotonic())
        last_capture = float(packet["capture_client_s"])

    def accept(response, bootstrap=False):
        nonlocal pending, remaining, executed_since_result
        request_id, origin, started = pending
        if int(response["request_id"]) != request_id or int(response["origin"]) != origin:
            raise RuntimeError("Action response generation/origin mismatch")
        elapsed = len(timeline.actions) - origin
        if not bootstrap and elapsed > max_delay:
            raise RuntimeError("Inference delay exceeds the checkpoint's trained delay range")
        upcoming = future_actions(response["actions"], elapsed)
        upcoming = validate_actions(upcoming, observations.state(), cfg, binarize_grippers=False)
        old_remaining = len(remaining)
        remaining = validate_actions(
            merge_actions(remaining, upcoming),
            observations.state(),
            cfg,
            binarize_grippers=False,
        )
        now = time.monotonic()
        queue_fields = diagnostics.switch_fields(now, old_remaining, len(upcoming), bootstrap=bootstrap)
        assimilated = response["assimilated"].tolist()
        details = f" assimilated={assimilated}" if cfg["online"].get("log_assimilated_details", False) else ""
        print(
            f"chunk={request_id} latency={float(response['latency_s']):.3f}s "
            f"client_elapsed_s={now - started:.3f} trim={elapsed} {queue_fields} "
            f"assimilated_count={len(assimilated)}{details}",
            flush=True,
        )
        executed_since_result = 0
        pending = None

    submit_begin()
    while True:
        poll()
        response = client.take_response()
        if response is not None:
            accept(response, bootstrap=True)
            break
        if time.monotonic() - pending[2] > cfg["model"]["prediction_timeout_s"]:
            raise RuntimeError("Initial prediction timeout")
        time.sleep(0.001)

    deadline = time.monotonic() + period
    while True:
        while time.monotonic() < deadline:
            poll()
        poll()
        client.check()
        now = time.monotonic()
        if now - deadline > cfg["control"]["max_tick_lateness_s"]:
            raise RuntimeError("Control tick late; stop rather than replay overdue commands")
        response = client.take_response()
        if response is not None:
            accept(response)
        if pending is not None:
            elapsed = len(timeline.actions) - pending[1]
            if elapsed > max_delay:
                raise RuntimeError(f"Prediction took more than {max_delay} control steps; stop and reconnect")
            if now - pending[2] > cfg["model"]["prediction_timeout_s"]:
                raise RuntimeError("Prediction timeout")
        state = observations.state()
        holding = not len(remaining)
        if len(remaining):
            action = remaining[0]
            remaining = remaining[1:]
        else:
            # Keep the last issued target while waiting; do not reset the filter
            # to lagging feedback or continue toward an exhausted chunk's target.
            action = last_action.copy()
        action = validate_actions(action[None], state, cfg, binarize_grippers=False)[0]
        issued_action = publish(action)
        if issued_action is None:
            raise RuntimeError("Final publisher must return the action actually issued")
        issued_at = time.monotonic()
        timeline.append(issued_at, issued_action)
        if holding:
            diagnostics.hold_tick(issued_at, pending[0] if pending is not None else None)
        last_action = issued_action.copy()
        executed_since_result += 1
        if pending is None and (executed_since_result >= cfg["model"]["replan_steps"] or not len(remaining)):
            submit_begin()
        elif pending is not None and now >= next_observation:
            packet = observations.request(False)
            capture = float(packet["capture_client_s"])
            end = timeline.index_at(capture)
            step = end - pending[1]
            if capture > last_capture and 0 < step <= max_delay:
                packet.update(
                    request_id=pending[0],
                    origin=pending[1],
                    capture_step=step,
                    executed_prefix=timeline.prefix(pending[1], end),
                )
                client.submit(packet, begin=False)
                last_capture = capture
            next_observation = now + 1 / cfg["online"]["fast_observation_hz"]
        deadline += period
