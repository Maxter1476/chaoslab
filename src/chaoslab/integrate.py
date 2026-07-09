"""Integrators for the double pendulum.

RK4 for accuracy and a velocity-Verlet-style scheme are both offered; RK4 is
the default because it is simplest to reason about and the test suite pins
its energy drift to a tight bound over moderate horizons.
"""

from __future__ import annotations

import numpy as np

from .pendulum import DoublePendulum

__all__ = ["rk4_step", "integrate"]


def rk4_step(pendulum: DoublePendulum, state: np.ndarray, dt: float) -> np.ndarray:
    """One classic RK4 step, vectorized over leading axes of ``state``."""
    k1 = pendulum.derivatives(state)
    k2 = pendulum.derivatives(state + 0.5 * dt * k1)
    k3 = pendulum.derivatives(state + 0.5 * dt * k2)
    k4 = pendulum.derivatives(state + dt * k3)
    return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(
    pendulum: DoublePendulum,
    state0: np.ndarray,
    dt: float,
    n_steps: int,
    *,
    record_every: int = 1,
) -> dict[str, np.ndarray]:
    """Integrate and record the trajectory.

    ``state0`` may be a single state (shape ``(4,)``) or a batch
    (shape ``(n, 4)``); the recorded ``states`` array carries the batch axis.
    """
    state = np.array(state0, dtype=float)
    times, states = [], []
    for step in range(n_steps):
        state = rk4_step(pendulum, state, dt)
        if record_every and step % record_every == 0:
            times.append((step + 1) * dt)
            states.append(state.copy())
    return {"time": np.array(times), "states": np.array(states)}
