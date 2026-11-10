"""Client-owned action timeline. Counts commands actually issued, including holds."""

from __future__ import annotations

from bisect import bisect_right

import numpy as np


class Timeline:
    def __init__(self):
        self.times = []
        self.actions = []

    def append(self, timestamp, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (16,) or not np.isfinite(action).all():
            raise ValueError("Command must be finite action16")
        if self.times and timestamp < self.times[-1]:
            raise ValueError("Command timestamps must be monotonic")
        self.times.append(timestamp)
        self.actions.append(action.copy())

    def index_at(self, capture_timestamp):
        return bisect_right(self.times, capture_timestamp)

    def prefix(self, origin, end):
        if not 0 <= origin <= end <= len(self.actions):
            raise ValueError("Invalid executed prefix interval")
        return np.asarray(self.actions[origin:end], dtype=np.float32).reshape(-1, 16)


def future_actions(actions, elapsed):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 16 or not np.isfinite(actions).all():
        raise ValueError("Invalid predicted action chunk")
    if not 0 <= elapsed < len(actions):
        raise RuntimeError("Entire prediction expired; stop and replan from a fresh observation")
    return actions[elapsed:].copy()


def merge_actions(previous, upcoming):
    """Linearly blend every overlapping row, matching the original async client."""
    result = np.array(upcoming, dtype=np.float64, copy=True)
    count = min(len(previous), len(result))
    if count:
        weight = (np.arange(1, count + 1) / (count + 1))[:, None]
        result[:count] = previous[:count] * (1 - weight) + result[:count] * weight
    return result
