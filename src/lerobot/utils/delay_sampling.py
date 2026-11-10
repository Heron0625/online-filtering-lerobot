"""Configurable control-step latency priors shared by real-robot training stages."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def validate_delay_distribution(
    ranges: Sequence[Sequence[int]] | None,
    probabilities: Sequence[float] | None,
    max_delay: int,
) -> None:
    if ranges is None and probabilities is None:
        return
    if ranges is None or probabilities is None or not ranges or len(ranges) != len(probabilities):
        raise ValueError("delay ranges and probabilities must be nonempty and have equal lengths")
    previous = 0
    for bounds in ranges:
        if len(bounds) != 2:
            raise ValueError("each delay range must contain an inclusive lower and upper bound")
        low, high = bounds
        if low != previous + 1 or high < low or high > max_delay:
            raise ValueError("delay ranges must partition all positive delays consecutively")
        previous = high
    if previous != max_delay:
        raise ValueError("delay ranges must cover max_inference_delay")
    if any(not math.isfinite(p) or p < 0 for p in probabilities) or not math.isclose(
        sum(probabilities), 1.0, abs_tol=1e-6
    ):
        raise ValueError("delay probabilities must be finite, nonnegative, and sum to one")


def sample_delay_distribution(
    valid_observations: Tensor,
    ranges: Sequence[Sequence[int]],
    probabilities: Sequence[float],
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample a band, then a valid step uniformly within it.

    At padded episode tails, empty bands have zero mass and remaining band masses
    are renormalized. Delay zero is used only when no positive observation exists.
    """
    valid = valid_observations.bool()
    if valid.ndim != 2 or valid.shape[1] < 1 or not valid[:, 0].all():
        raise ValueError("valid_observations must be (batch, window) with valid delay zero")
    validate_delay_distribution(ranges, probabilities, valid.shape[1] - 1)
    weights = torch.zeros_like(valid, dtype=torch.float32)
    for (low, high), probability in zip(ranges, probabilities, strict=True):
        candidates = valid[:, low : high + 1]
        weights[:, low : high + 1] = probability * candidates / candidates.sum(1, keepdim=True).clamp_min(1)
    empty = weights.sum(1) == 0
    # Includes the degenerate case where only zero-probability bands remain valid.
    fallback = valid.to(torch.float32)
    fallback[:, 0] = ~valid[:, 1:].any(1)
    weights[empty] = fallback[empty]
    return torch.multinomial(weights, 1, generator=generator).squeeze(1)
