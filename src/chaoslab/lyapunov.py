"""Largest Lyapunov exponent via the Benettin renormalization method.

Two trajectories start a tiny distance ``d0`` apart in state space. After a
short interval their separation ``d`` is measured, the local expansion factor
``ln(d/d0)`` is accumulated, and the perturbed trajectory is rescaled back to
distance ``d0`` along the current separation direction. The largest Lyapunov
exponent is the long-time average expansion rate:

    lambda_max = (1 / (N tau)) * sum ln(d_i / d0).

Positive lambda is the quantitative signature of chaos. For the double
pendulum at high energy the suite requires lambda_max > 0; for a low-energy
(quasi-periodic) start it requires lambda_max ~ 0.
"""

from __future__ import annotations

import numpy as np

from .integrate import rk4_step
from .pendulum import DoublePendulum

__all__ = ["largest_lyapunov"]


def largest_lyapunov(
    pendulum: DoublePendulum,
    state0: np.ndarray,
    *,
    dt: float = 0.001,
    renorm_every: int = 100,
    n_renorm: int = 4000,
    d0: float = 1e-8,
    discard: int = 200,
) -> float:
    """Estimate the largest Lyapunov exponent (units: 1/time).

    ``discard`` initial renormalizations are dropped so the perturbation can
    align with the most-expanding direction before averaging begins.
    """
    state = np.array(state0, dtype=float)
    rng = np.random.default_rng(0)
    perturb = rng.standard_normal(4)
    perturb = state + d0 * perturb / np.linalg.norm(perturb)

    tau = renorm_every * dt
    log_sum = 0.0
    count = 0
    for i in range(n_renorm):
        for _ in range(renorm_every):
            state = rk4_step(pendulum, state, dt)
            perturb = rk4_step(pendulum, perturb, dt)
        diff = perturb - state
        dist = np.linalg.norm(diff)
        if dist == 0.0:
            continue
        if i >= discard:
            log_sum += np.log(dist / d0)
            count += 1
        # renormalize the perturbed trajectory back to distance d0
        perturb = state + (d0 / dist) * diff
    if count == 0:
        raise RuntimeError("no samples collected; increase n_renorm")
    return log_sum / (count * tau)
