"""Double-pendulum dynamics and energy.

State is ``(theta1, theta2, omega1, omega2)`` — the two arm angles measured
from the downward vertical and their angular velocities. The equations of
motion are the exact Lagrangian result for two point masses on massless rods
(no small-angle approximation); see e.g. Landau & Lifshitz *Mechanics* §5.

Parameters use g = 9.81 by default; masses ``m1, m2`` and lengths
``l1, l2`` are configurable. Everything is vectorized so an array of states
(shape ``(..., 4)``) integrates in one call — the Lyapunov and fractal maps
rely on that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DoublePendulum"]


@dataclass(frozen=True)
class DoublePendulum:
    """Planar double pendulum with configurable masses, lengths, and gravity."""

    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = 9.81

    def __post_init__(self) -> None:
        if min(self.m1, self.m2, self.l1, self.l2) <= 0:
            raise ValueError("masses and lengths must be positive")

    def derivatives(self, state: np.ndarray) -> np.ndarray:
        """Time derivative of the state, vectorized over leading axes.

        ``state[..., :] = (theta1, theta2, omega1, omega2)``.
        """
        state = np.asarray(state, dtype=float)
        t1, t2, w1, w2 = (state[..., i] for i in range(4))
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g

        delta = t1 - t2
        cos_d, sin_d = np.cos(delta), np.sin(delta)
        denom = 2 * m1 + m2 - m2 * np.cos(2 * delta)

        dw1 = (
            -g * (2 * m1 + m2) * np.sin(t1)
            - m2 * g * np.sin(t1 - 2 * t2)
            - 2 * sin_d * m2 * (w2**2 * l2 + w1**2 * l1 * cos_d)
        ) / (l1 * denom)

        dw2 = (
            2
            * sin_d
            * (
                w1**2 * l1 * (m1 + m2)
                + g * (m1 + m2) * np.cos(t1)
                + w2**2 * l2 * m2 * cos_d
            )
        ) / (l2 * denom)

        out = np.empty_like(state)
        out[..., 0] = w1
        out[..., 1] = w2
        out[..., 2] = dw1
        out[..., 3] = dw2
        return out

    def energy(self, state: np.ndarray) -> np.ndarray:
        """Total mechanical energy (kinetic + potential), vectorized.

        Potential zero is taken at the common pivot, so a hanging pendulum has
        negative energy.
        """
        state = np.asarray(state, dtype=float)
        t1, t2, w1, w2 = (state[..., i] for i in range(4))
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g

        # Bob positions (y downward negative).
        y1 = -l1 * np.cos(t1)
        y2 = y1 - l2 * np.cos(t2)

        # Velocities of the two bobs.
        v1_sq = (l1 * w1) ** 2
        v2_sq = (
            (l1 * w1) ** 2
            + (l2 * w2) ** 2
            + 2 * l1 * l2 * w1 * w2 * np.cos(t1 - t2)
        )
        kinetic = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
        potential = m1 * g * y1 + m2 * g * y2
        return kinetic + potential

    def flip_time(self, state: np.ndarray, dt: float, t_max: float) -> float:
        """Time until the *lower* arm first flips over (|theta2| exceeds pi).

        Returns ``t_max`` if no flip occurs within the window. Uses the
        symplectic-free RK4 stepper from :mod:`chaoslab.integrate`, wrapped
        here for convenience.
        """
        from .integrate import rk4_step

        s = np.array(state, dtype=float)
        # unwrap theta2 continuously so a full rotation is detectable
        t = 0.0
        theta2_unwrapped = s[1]
        prev_t2 = s[1]
        while t < t_max:
            s = rk4_step(self, s, dt)
            theta2_unwrapped += _angle_delta(s[1], prev_t2)
            prev_t2 = s[1]
            t += dt
            if abs(theta2_unwrapped) > np.pi:
                return t
        return t_max


def _angle_delta(new: float, old: float) -> float:
    """Signed smallest-arc change between two angles, for unwrapping."""
    d = new - old
    return (d + np.pi) % (2 * np.pi) - np.pi
