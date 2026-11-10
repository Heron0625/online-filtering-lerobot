"""Optional causal joint-command smoothing, independent of chunk replacement."""

from __future__ import annotations

import math

import numpy as np

JOINTS = [*range(7), *range(8, 15)]


class JointActionSmoother:
    def __init__(self, config):
        self.enabled = config.get("enabled", False)
        self.alpha = config.get("alpha", 0.5)
        if not isinstance(self.enabled, bool):
            raise ValueError("control.action_smoothing.enabled must be true/false")
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float))
            or not math.isfinite(self.alpha)
            or not 0 < self.alpha <= 1
        ):
            raise ValueError("control.action_smoothing.alpha must be in (0,1]")
        self.max_step_deg = config.get("max_step_deg")
        if self.max_step_deg is not None and (
            isinstance(self.max_step_deg, bool)
            or not isinstance(self.max_step_deg, (int, float))
            or not math.isfinite(self.max_step_deg)
            or self.max_step_deg <= 0
        ):
            raise ValueError("control.action_smoothing.max_step_deg must be positive or null")
        self.previous = None

    def reset(self, state):
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Smoothing requires finite state14")
        self.previous = state.copy()

    def apply(self, action, state):
        result = np.array(action, dtype=np.float64, copy=True)
        if result.shape != (16,) or not np.isfinite(result).all():
            raise ValueError("Smoothing requires finite action16")
        if self.enabled:
            if self.previous is None:
                self.reset(state)
            result[JOINTS] = self.alpha * result[JOINTS] + (1 - self.alpha) * self.previous
            if self.max_step_deg is not None:
                limit = np.deg2rad(self.max_step_deg)
                result[JOINTS] = self.previous + np.clip(result[JOINTS] - self.previous, -limit, limit)
            self.previous = result[JOINTS].copy()
        return result


class FinalCommandPublisher:
    """Filter immediately before hardware send; return exactly the issued action."""

    def __init__(self, config, read_state, validate, send):
        self.smoother = JointActionSmoother(config)
        self.read_state = read_state
        self.validate = validate
        self.send = send

    def __call__(self, action):
        state = self.read_state()
        raw = self.validate(np.array(action, dtype=np.float64, copy=True), state)
        previous = None if self.smoother.previous is None else self.smoother.previous.copy()
        try:
            command = self.smoother.apply(raw, state)
            command = self.validate(command, state)
            self.send(command)
            if self.smoother.enabled:
                self.smoother.previous = command[JOINTS].copy()
        except BaseException:
            # A failed send must never advance the last-issued command state.
            self.smoother.previous = previous
            raise
        return command.copy()
