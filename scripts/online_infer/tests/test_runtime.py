"""Timing and full-duplex contracts without model weights or robot drivers."""

from __future__ import annotations

import contextlib
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from protocol import MAX_BYTES, ClockMapping, receive, send
from timeline import Timeline, future_actions, merge_actions
from transport import OnlineClient


def test_timeline_uses_capture_instant_and_actual_commands():
    timeline = Timeline()
    for timestamp, command in [(10.0, 1), (10.04, 2), (10.08, 3)]:
        timeline.append(timestamp, np.full(16, command))
    assert timeline.index_at(10.06) == 2
    np.testing.assert_array_equal(timeline.prefix(1, timeline.index_at(10.06)), np.full((1, 16), 2))
    np.testing.assert_array_equal(timeline.prefix(1, 3), np.array([[2] * 16, [3] * 16]))


def test_late_result_trims_at_client_execution_time():
    actions = np.repeat(np.arange(50)[:, None], 16, axis=1)
    assert future_actions(actions, 11)[0, 0] == 11
    with pytest.raises(RuntimeError, match="expired"):
        future_actions(actions, 50)


@pytest.mark.parametrize("old_length,new_length", [(20, 30), (30, 20), (0, 30)])
def test_merge_blends_entire_overlap(old_length, new_length):
    result = merge_actions(np.zeros((old_length, 16)), np.ones((new_length, 16)))
    count = min(old_length, new_length)
    np.testing.assert_allclose(result[:count, 0], np.arange(1, count + 1) / (count + 1))
    np.testing.assert_array_equal(result[count:], 1)
    assert result.shape == (new_length, 16)


def test_clock_mapping_handles_unrelated_monotonic_origins():
    clock = ClockMapping.from_exchange(10, 1010.01, 1010.02, 10.03)
    assert clock.offset == pytest.approx(1000)
    assert clock.uncertainty == pytest.approx(0.01)
    assert 10.05 + clock.offset - clock.uncertainty == pytest.approx(1010.04)


def test_framing_rejects_unbounded_packets():
    a, b = socket.socketpair()
    try:
        a.sendall(struct.pack("!I", MAX_BYTES + 1))
        with pytest.raises(ValueError, match="size"):
            receive(b)
    finally:
        a.close()
        b.close()


def test_frame_roundtrip_no_pickle():
    a, b = socket.socketpair()
    try:
        send(a, kind="fast", state=np.arange(14, dtype=np.float32), executed_prefix=np.zeros((3, 16)))
        value = receive(b)
        assert str(value["kind"]) == "fast"
        np.testing.assert_array_equal(value["state"], np.arange(14))
    finally:
        a.close()
        b.close()


def test_full_duplex_dino_delivery_during_expert_and_reconnect(monkeypatch):
    encoded = threading.Event()
    running = threading.Event()
    failures = []
    packets = []

    class FakeTensor:
        def __getitem__(self, index):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return self

        def numpy(self):
            return np.zeros((50, 16), np.float32)

    class Policy:
        generation = 0

        def begin_online_generation(self):
            self.generation += 1
            packets.clear()
            encoded.clear()
            return self.generation

        def push_fast_observation(self, packet):
            packets.append(packet)
            encoded.set()

        def predict_action_chunk(self, batch, **kwargs):
            running.set()
            assert encoded.wait(3), "Expert blocked incoming fast observation"
            kwargs["observation_callback"](1, 10, self.generation)
            assert packets[-1].generation == self.generation
            kwargs["assimilation_callback"](1, packets[-1].capture_step, False)
            return FakeTensor()

    def encode(runner, data, generation):
        assert running.is_set()
        return SimpleNamespace(
            generation=generation,
            capture_step=int(data["capture_step"]),
            ready_timestamp=time.monotonic(),
            capture_timestamp=float(data["capture_server_s"]),
        )

    monkeypatch.setattr(server, "fast_packet", encode)
    monkeypatch.setattr(server, "raw_batch", lambda runner, data: data)
    runner = SimpleNamespace(
        policy=Policy(),
        torch=SimpleNamespace(inference_mode=contextlib.nullcontext),
        pre=lambda x: x,
        post=lambda x: x,
    )
    cfg = {"online": {"max_observation_age_s": 1, "clock_max_uncertainty_s": 0.02}, "model": {"task": "test"}}
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    address = f"tcp:127.0.0.1:{listener.getsockname()[1]}"

    def serve():
        try:
            for _ in range(2):
                connection, _ = listener.accept()
                session = server.Session(connection, runner, cfg)
                try:
                    session.reply(version=1, hz=25, chunk_size=50, max_delay=30, task="test")
                    while True:
                        session.handle(receive(connection))
                except (EOFError, OSError):
                    pass
                finally:
                    session.close()
                    connection.close()
        except BaseException as exc:
            failures.append(exc)
        finally:
            listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    for request_id in (1, 2):
        running.clear()
        client = OnlineClient(address, 3, lambda: None, cfg)
        try:
            client.submit(
                {"request_id": request_id, "origin": 5, "capture_client_s": time.monotonic()}, begin=True
            )
            assert running.wait(2)
            client.submit(
                {
                    "request_id": request_id,
                    "origin": 5,
                    "capture_step": 3,
                    "capture_client_s": time.monotonic(),
                },
                begin=False,
            )
            deadline = time.monotonic() + 3
            response = None
            while response is None and time.monotonic() < deadline:
                response = client.take_response()
                time.sleep(0.001)
            assert response is not None
            assert int(response["request_id"]) == request_id
            assert response["assimilated"].tolist() == [[1, 3]]
            assert response["actions"].shape == (50, 16)
        finally:
            client.close()
    thread.join(3)
    assert not thread.is_alive()
    assert not failures


@pytest.mark.parametrize("chunk_length", [50, 6])
@pytest.mark.parametrize("smoothing_enabled", [False, True])
def test_control_loop_sends_the_commands_it_really_published(
    monkeypatch, capsys, smoothing_enabled, chunk_length
):
    import execution

    clock = [100.0]
    published, fast_packets = [], []
    response_count = [0]
    monkeypatch.setattr(
        execution,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0], sleep=lambda delay: clock.__setitem__(0, clock[0] + delay)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "common",
        SimpleNamespace(
            JOINTS=[*range(7), *range(8, 15)], validate_actions=lambda actions, *args, **kwargs: actions
        ),
    )

    class StopTestError(Exception):
        pass

    class Observations:
        def request(self, reset):
            return {"capture_client_s": clock[0] - 0.005}

        def state(self):
            return np.zeros(14)

    class Client:
        metadata = {"max_delay": 30}
        pending = None

        def check(self):
            pass

        def submit(self, data, *, begin):
            if begin:
                self.pending = (data, clock[0])
            else:
                fast_packets.append(data)

        def take_response(self):
            if self.pending is None or clock[0] - self.pending[1] < 0.12:
                return None
            data, _ = self.pending
            self.pending = None
            response_count[0] += 1
            return {
                "request_id": data["request_id"],
                "origin": data["origin"],
                "latency_s": 0.12,
                "actions": np.full((chunk_length, 16), response_count[0], dtype=np.float32),
                "assimilated": np.empty((0, 2), dtype=int),
            }

    def poll():
        clock[0] += 0.001
        if clock[0] > 101.0:
            raise StopTestError

    cfg = {
        "control": {
            "execution_hz": 25,
            "max_tick_lateness_s": 0.015,
            "action_smoothing": {"enabled": smoothing_enabled, "alpha": 0.5},
        },
        "model": {"prediction_timeout_s": 1, "replan_steps": 2},
        "online": {"fast_observation_hz": 25},
    }
    from smoothing import FinalCommandPublisher

    observations = Observations()
    publisher = FinalCommandPublisher(
        cfg["control"]["action_smoothing"],
        observations.state,
        lambda action, state: action,
        lambda action: published.append(action.copy()),
    )
    with pytest.raises(StopTestError):
        execution.run_online(Client(), observations, cfg, poll, publisher, lambda: None)
    output = capsys.readouterr().out
    assert "old_remaining=" in output and "new_remaining=" in output
    assert "assimilated_count=0" in output and " assimilated=" not in output
    assert "queue_summary" in output
    if chunk_length == 6:
        assert "queue_hold_start" in output and "queue_hold_end" in output
    else:
        assert "queue_hold_start" not in output
        assert "queue_summary hold_count=0 hold_ticks=0 hold_total_s=0.000" in output
    assert response_count[0] >= 3
    assert len(fast_packets) >= 3
    for packet in fast_packets:
        start, count = packet["origin"], packet["capture_step"]
        np.testing.assert_array_equal(
            packet["executed_prefix"], np.asarray(published[start : start + count], dtype=np.float32)
        )


def test_joint_smoothing_seeds_from_feedback_and_preserves_grippers():
    from smoothing import JOINTS, JointActionSmoother

    smoother = JointActionSmoother({"enabled": True, "alpha": 0.5})
    target = np.ones(16)
    target[[7, 15]] = [0.2, 0.9]
    first = smoother.apply(target, np.zeros(14))
    second = smoother.apply(target, np.zeros(14))
    np.testing.assert_allclose(first[JOINTS], 0.5)
    np.testing.assert_allclose(second[JOINTS], 0.75)
    np.testing.assert_array_equal(first[[7, 15]], target[[7, 15]])
    np.testing.assert_array_equal(target[JOINTS], 1)
    smoother.reset(np.full(14, 0.2))
    np.testing.assert_allclose(smoother.apply(target, np.zeros(14))[JOINTS], 0.6)


@pytest.mark.parametrize("config", [{"enabled": False, "alpha": 0.5}, {"enabled": True, "alpha": 1.0}])
def test_smoothing_bypass(config):
    from smoothing import JointActionSmoother

    target = np.arange(16, dtype=float)
    np.testing.assert_array_equal(JointActionSmoother(config).apply(target, np.zeros(14)), target)


@pytest.mark.parametrize("alpha", [0, -1, 1.1, float("nan"), float("inf"), True])
def test_smoothing_rejects_invalid_alpha(alpha):
    from smoothing import JointActionSmoother

    with pytest.raises(ValueError, match="alpha"):
        JointActionSmoother({"enabled": True, "alpha": alpha})


def test_final_wire_commands_are_continuous_across_chunk_switch_and_hold():
    from smoothing import JOINTS, FinalCommandPublisher

    wire = []
    state = np.zeros(14)
    publisher = FinalCommandPublisher(
        {"enabled": True, "alpha": 0.5, "max_step_deg": 1.0},
        lambda: state,
        lambda action, feedback: action,
        lambda command: wire.append(np.rad2deg(command[JOINTS])),
    )
    first = publisher(np.full(16, np.deg2rad(90)))
    state[:] = np.deg2rad(-20)  # Feedback lag must not reset last-issued state.
    held = publisher(first)
    np.testing.assert_array_equal(held, first)
    # New chunk contains a large reversal; limit the actual outgoing command.
    switched = publisher(np.full(16, np.deg2rad(-90)))
    np.testing.assert_allclose(wire, [np.ones(14), np.ones(14), np.zeros(14)], atol=1e-12)
    np.testing.assert_allclose(np.rad2deg(switched[JOINTS]), wire[-1], atol=1e-12)
    assert max(np.abs(np.diff(np.vstack([np.zeros(14), *wire]), axis=0)).flat) <= 1.0 + 1e-12


def test_failed_final_send_does_not_advance_filter_state():
    from smoothing import FinalCommandPublisher

    def fail(command):
        raise OSError("send failed")

    publisher = FinalCommandPublisher(
        {"enabled": True, "alpha": 0.5}, lambda: np.zeros(14), lambda action, feedback: action, fail
    )
    with pytest.raises(OSError, match="send failed"):
        publisher(np.ones(16))
    assert publisher.smoother.previous is None


@pytest.mark.parametrize("limit", [0, -1, float("nan"), float("inf"), True])
def test_smoothing_rejects_invalid_command_step_limit(limit):
    from smoothing import JointActionSmoother

    with pytest.raises(ValueError, match="max_step_deg"):
        JointActionSmoother({"enabled": True, "max_step_deg": limit})


def test_queue_diagnostics_measure_starvation_without_counting_bootstrap(capsys):
    from diagnostics import QueueDiagnostics

    diagnostics = QueueDiagnostics(25)
    fields = diagnostics.switch_fields(1, 0, 50, bootstrap=True)
    assert diagnostics.hold_count == 0
    assert "old_remaining=0" in fields
    diagnostics.hold_tick(2, 3)
    diagnostics.hold_tick(2.04, 3)
    fields = diagnostics.switch_fields(2.12, 0, 27)
    assert diagnostics.hold_count == 1
    assert diagnostics.hold_ticks == 2
    assert diagnostics.hold_total_s == pytest.approx(0.12)
    assert "new_remaining=27" in fields
    diagnostics.hold_tick(3, 4)
    diagnostics.finish(3.08)
    assert diagnostics.hold_count == 2
    assert diagnostics.hold_total_s == pytest.approx(0.20)
    output = capsys.readouterr().out
    assert "reason=new_chunk duration_s=0.120 ticks=2" in output
    assert "reason=session_end duration_s=0.080 ticks=1" in output
    assert "hold_count=2 hold_ticks=3 hold_total_s=0.200" in output
