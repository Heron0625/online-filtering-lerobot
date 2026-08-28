#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Analytic Gaussian reference implementation for posterior re-anchoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def linear_gaussian_update(mean: Tensor, covariance: Tensor, observation: Tensor, matrix: Tensor, noise: Tensor):
    """Return the exact posterior of a linear Gaussian observation model."""
    precision = torch.linalg.inv(covariance)
    noise_precision = torch.linalg.inv(noise)
    posterior_covariance = torch.linalg.inv(precision + matrix.mT @ noise_precision @ matrix)
    posterior_mean = posterior_covariance @ (
        precision @ mean + matrix.mT @ noise_precision @ observation
    )
    return posterior_mean, posterior_covariance


@dataclass(frozen=True)
class AffineGaussianFlow:
    """Linear interpolation flow from standard normal to ``N(mean, covariance)``."""

    mean: Tensor
    covariance: Tensor

    @property
    def scale(self) -> Tensor:
        return torch.linalg.cholesky(self.covariance)

    def map(self, base: Tensor, time: float | Tensor) -> Tensor:
        time_tensor = torch.as_tensor(time, dtype=base.dtype, device=base.device)
        target = self.mean.to(base) + base @ self.scale.to(base).mT
        return (1 - time_tensor) * base + time_tensor * target

    def inverse(self, value: Tensor, time: float | Tensor) -> Tensor:
        time_tensor = torch.as_tensor(time, dtype=value.dtype, device=value.device)
        identity = torch.eye(value.shape[-1], dtype=value.dtype, device=value.device)
        transform = (1 - time_tensor) * identity + time_tensor * self.scale.to(value)
        centered = value - time_tensor * self.mean.to(value)
        return torch.linalg.solve(transform, centered.mT).mT

    def velocity(self, value: Tensor, time: float | Tensor) -> Tensor:
        base = self.inverse(value, time)
        return self.mean.to(value) + base @ (self.scale.to(value) - torch.eye(
            value.shape[-1], dtype=value.dtype, device=value.device
        )).mT

    def integrate(self, base: Tensor, end_time: float, steps: int) -> Tensor:
        if steps < 1:
            raise ValueError("steps must be positive")
        value = base
        delta = end_time / steps
        for step in range(steps):
            time = step * delta
            value = value + delta * self.velocity(value, time)
        return value


def shared_base_reanchor(base: Tensor, new_flow: AffineGaussianFlow, time: float | Tensor) -> Tensor:
    """Recompute a particle at fixed flow time under a new condition using the same base noise."""
    return new_flow.map(base, time)


def naive_condition_switch(
    base: Tensor,
    old_flow: AffineGaussianFlow,
    new_flow: AffineGaussianFlow,
    switch_time: float,
    steps_before: int,
    steps_after: int,
) -> Tensor:
    """Integrate the old field, then change fields without posterior re-anchoring."""
    value = old_flow.integrate(base, switch_time, steps_before)
    delta = (1 - switch_time) / steps_after
    for step in range(steps_after):
        time = switch_time + step * delta
        value = value + delta * new_flow.velocity(value, time)
    return value
