"""Sensitive dependence: three near-identical pendulums diverging."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chaoslab import DoublePendulum, integrate


def bob_positions(dp: DoublePendulum, states: np.ndarray) -> np.ndarray:
    t1, t2 = states[..., 0], states[..., 1]
    x2 = dp.l1 * np.sin(t1) + dp.l2 * np.sin(t2)
    y2 = -dp.l1 * np.cos(t1) - dp.l2 * np.cos(t2)
    return np.stack([x2, y2], axis=-1)


def main() -> None:
    dp = DoublePendulum()
    base = np.array([np.pi / 2, np.pi / 2, 0.0, 0.0])
    starts = np.array([base, base + [1e-3, 0, 0, 0], base + [2e-3, 0, 0, 0]])

    dt, n = 0.002, 6000
    traj = integrate(dp, starts, dt, n, record_every=2)
    tip = bob_positions(dp, traj["states"])  # (frames, 3, 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for k, c in enumerate(colors):
        ax1.plot(tip[:, k, 0], tip[:, k, 1], color=c, lw=0.4, alpha=0.8,
                 label=f"$\\theta_1(0)=\\pi/2 + {k}\\times10^{{-3}}$")
    ax1.set_aspect("equal")
    ax1.set_title("Lower-bob paths from three near-identical starts")
    ax1.legend(fontsize=8)

    sep = np.linalg.norm(tip[:, 1] - tip[:, 0], axis=-1)
    ax2.semilogy(traj["time"], sep, lw=1)
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("separation of bobs [m]")
    ax2.set_title("Exponential divergence (note the log scale)")

    fig.tight_layout()
    fig.savefig("docs/figures/divergence.png", dpi=150)
    print("wrote docs/figures/divergence.png")


if __name__ == "__main__":
    main()
