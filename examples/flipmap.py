"""The fractal flip-time map — chaoslab's signature figure.

For a grid of initial angles (theta1, theta2) with the pendulum released from
rest, color each pixel by how long the lower arm takes to flip over. The
low-energy region where a flip is impossible forms a smooth lens; outside it
the boundary is fractal, a direct visualization of sensitive dependence on
initial conditions.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chaoslab import DoublePendulum, rk4_step


def flip_time_grid(n: int = 300, dt: float = 0.01, t_max: float = 40.0) -> np.ndarray:
    """Vectorized flip-time map over an n x n grid of (theta1, theta2)."""
    dp = DoublePendulum()
    grid = np.linspace(-np.pi, np.pi, n)
    t1, t2 = np.meshgrid(grid, grid)
    state = np.stack([t1, t2, np.zeros_like(t1), np.zeros_like(t1)], axis=-1).reshape(-1, 4)

    flip_time = np.full(state.shape[0], np.nan)
    theta2_unwrapped = state[:, 1].copy()
    prev = state[:, 1].copy()
    active = np.ones(state.shape[0], dtype=bool)

    steps = int(t_max / dt)
    for step in range(steps):
        state[active] = rk4_step(dp, state[active], dt)
        new = state[:, 1]
        delta = (new - prev + np.pi) % (2 * np.pi) - np.pi
        theta2_unwrapped += np.where(active, delta, 0.0)
        prev = new.copy()
        flipped = active & (np.abs(theta2_unwrapped) > np.pi)
        flip_time[flipped] = (step + 1) * dt
        active &= ~flipped
        if not active.any():
            break
    return flip_time.reshape(n, n)


def main() -> None:
    grid = flip_time_grid()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        grid,
        extent=(-np.pi, np.pi, -np.pi, np.pi),
        origin="lower",
        cmap="twilight_shifted",
        interpolation="nearest",
    )
    ax.set_xlabel(r"$\theta_1$ (upper arm)")
    ax.set_ylabel(r"$\theta_2$ (lower arm)")
    ax.set_title("Time for the lower arm to flip (released from rest)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("flip time [s]  (blank = no flip within 40 s)")
    fig.tight_layout()
    fig.savefig("docs/figures/flipmap.png", dpi=150)
    print("wrote docs/figures/flipmap.png")


if __name__ == "__main__":
    main()
