"""Full duplex client transport with bounded latest-only observation sending."""

from __future__ import annotations

import contextlib
import socket
import threading

from protocol import VERSION, address_socket, receive, send, synchronize


class OnlineClient:
    def __init__(self, address, timeout, poll, cfg):
        family, endpoint = address_socket(address)
        self.connection = socket.socket(family)
        self.connection.settimeout(timeout)
        self.connection.connect(endpoint)
        self.metadata = receive(self.connection)
        if int(self.metadata["version"]) != VERSION or int(self.metadata["hz"]) != 25:
            raise ValueError("Server protocol/frequency mismatch")
        if str(self.metadata["task"]) != cfg["model"]["task"]:
            raise ValueError("Client/server task mismatch; restart server with matching config")
        self.clock = synchronize(self.connection, cfg["online"]["clock_max_uncertainty_s"])
        print(f"模型已连接，时钟误差界约 {self.clock.uncertainty * 1000:.2f} ms", flush=True)
        self.condition = threading.Condition()
        self.begin_pending = None
        self.fast_pending = None
        self.response = None
        self.error = None
        self.closed = False
        self.sender = threading.Thread(target=self.send_loop, name="ObservationSender", daemon=True)
        self.receiver = threading.Thread(target=self.receive_loop, name="ActionReceiver", daemon=True)
        self.sender.start()
        self.receiver.start()

    def fail(self, error):
        with self.condition:
            self.error = error
            self.condition.notify_all()

    def send_loop(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(
                        lambda: self.closed or self.begin_pending is not None or self.fast_pending is not None
                    )
                    if self.closed:
                        return
                    if self.begin_pending is not None:
                        data, self.begin_pending = self.begin_pending, None
                    else:
                        data, self.fast_pending = self.fast_pending, None
                send(self.connection, **data)
        except Exception as exc:
            self.fail(exc)

    def receive_loop(self):
        try:
            while not self.closed:
                data = receive(self.connection)
                with self.condition:
                    if self.response is not None:
                        raise RuntimeError("Unexpected duplicate action response")
                    self.response = data
        except Exception as exc:
            if not self.closed:
                self.fail(exc)

    def check(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError(f"Online model connection failed: {self.error}") from self.error

    def submit(self, data, *, begin):
        self.check()
        data = dict(data)
        data["kind"] = "begin" if begin else "fast"
        # Lower bound mapping is conservative for age (never pretend RTT is zero).
        data["capture_server_s"] = (
            float(data["capture_client_s"]) + self.clock.offset - self.clock.uncertainty
        )
        with self.condition:
            if begin:
                if self.begin_pending is not None:
                    raise RuntimeError("An initial observation is already queued")
                self.fast_pending = None
                self.begin_pending = data
            else:
                self.fast_pending = data
            self.condition.notify()

    def take_response(self):
        self.check()
        with self.condition:
            result, self.response = self.response, None
            return result

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        with contextlib.suppress(OSError):
            self.connection.shutdown(socket.SHUT_RDWR)
        self.connection.close()
        self.sender.join(timeout=2)
        self.receiver.join(timeout=2)
