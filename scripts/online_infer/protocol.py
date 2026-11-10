"""Bounded numpy framing; no pickle. Shared by ROS Python and model Python."""

from __future__ import annotations

import io
import math
import socket
import struct
import time
from dataclasses import dataclass

import numpy as np

MAX_BYTES = 32 * 1024 * 1024
VERSION = 1


def send(connection, **values):
    buffer = io.BytesIO()
    np.savez(buffer, **values)
    payload = buffer.getvalue()
    if not 0 < len(payload) <= MAX_BYTES:
        raise ValueError("Packet too large")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def receive(connection):
    def exact(size):
        result = bytearray()
        while len(result) < size:
            part = connection.recv(size - len(result))
            if not part:
                raise EOFError("Model connection closed")
            result.extend(part)
        return result

    size = struct.unpack("!I", exact(4))[0]
    if not 0 < size <= MAX_BYTES:
        raise ValueError("Invalid packet size")
    with np.load(io.BytesIO(exact(size)), allow_pickle=False) as packet:
        result = {key: packet[key] for key in packet.files}
    if "error" in result:
        raise RuntimeError(str(result["error"]))
    return result


def address_socket(address):
    if address.startswith("unix:"):
        return socket.AF_UNIX, address[5:]
    if address.startswith("tcp:"):
        host, port = address[4:].rsplit(":", 1)
        return socket.AF_INET, (host, int(port))
    raise ValueError("Address must be unix:/path or tcp:host:port")


@dataclass(frozen=True)
class ClockMapping:
    offset: float
    uncertainty: float

    @classmethod
    def from_exchange(cls, sent, server_received, server_sent, received):
        values = (sent, server_received, server_sent, received)
        if not all(math.isfinite(v) for v in values) or received < sent or server_sent < server_received:
            raise ValueError("Invalid clock exchange")
        uncertainty = max(0.0, ((received - sent) - (server_sent - server_received)) / 2)
        return cls(((server_received - sent) + (server_sent - received)) / 2, uncertainty)


def synchronize(connection, max_uncertainty):
    candidates = []
    for _ in range(7):
        sent = time.monotonic()
        send(connection, kind="ping")
        response = receive(connection)
        received = time.monotonic()
        candidates.append(
            ClockMapping.from_exchange(sent, float(response["received"]), float(response["sent"]), received)
        )
    mapping = min(candidates, key=lambda value: value.uncertainty)
    if mapping.uncertainty > max_uncertainty:
        raise RuntimeError(f"Clock uncertainty {mapping.uncertainty:.4f}s exceeds limit")
    return mapping
