#!/usr/bin/env python

"""CPU-only numerical validation of naive switching and shared-Z re-anchoring."""

from __future__ import annotations

import torch

from lerobot.policies.pi05.gaussian_reanchoring import (
    AffineGaussianFlow,
    naive_condition_switch,
    shared_base_reanchor,
)


def covariance(samples: torch.Tensor) -> torch.Tensor:
    """Compute an unbiased empirical covariance matrix."""
    centered = samples - samples.mean(0)
    return centered.mT @ centered / (samples.shape[0] - 1)


def main() -> None:
    """Run the analytic Gaussian posterior-transfer comparison."""
    dtype = torch.float64
    base = torch.randn(20_000, 2, dtype=dtype, generator=torch.Generator().manual_seed(2026))
    old_flow = AffineGaussianFlow(
        mean=torch.tensor([-1.5, 0.5], dtype=dtype),
        covariance=torch.tensor([[0.6, 0.1], [0.1, 1.2]], dtype=dtype),
    )
    new_flow = AffineGaussianFlow(
        mean=torch.tensor([1.5, -1.0], dtype=dtype),
        covariance=torch.tensor([[1.3, -0.25], [-0.25, 0.5]], dtype=dtype),
    )

    naive = naive_condition_switch(base, old_flow, new_flow, 0.55, steps_before=32, steps_after=32)
    reanchored = shared_base_reanchor(base, new_flow, 1.0)

    naive_mean_error = torch.linalg.vector_norm(naive.mean(0) - new_flow.mean).item()
    exact_mean_error = torch.linalg.vector_norm(reanchored.mean(0) - new_flow.mean).item()
    naive_covariance_error = torch.linalg.matrix_norm(covariance(naive) - new_flow.covariance).item()
    exact_covariance_error = torch.linalg.matrix_norm(covariance(reanchored) - new_flow.covariance).item()

    print(f"naive mean error:       {naive_mean_error:.6f}")
    print(f"shared-Z mean error:    {exact_mean_error:.6f}")
    print(f"naive covariance error: {naive_covariance_error:.6f}")
    print(f"shared-Z cov error:     {exact_covariance_error:.6f}")

    if not naive_mean_error > 0.1 or not naive_covariance_error > 0.05:
        raise RuntimeError("The counterexample did not separate naive switching from the new posterior")
    if not exact_mean_error < 0.03 or not exact_covariance_error < 0.05:
        raise RuntimeError("Shared-Z samples did not match the analytic new posterior")


if __name__ == "__main__":
    main()
