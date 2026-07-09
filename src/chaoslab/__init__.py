"""chaoslab — double-pendulum chaos, validated against analytic mechanics."""

from .integrate import integrate, rk4_step
from .lyapunov import largest_lyapunov
from .pendulum import DoublePendulum

__all__ = [
    "DoublePendulum",
    "integrate",
    "largest_lyapunov",
    "rk4_step",
]

__version__ = "0.1.0"
