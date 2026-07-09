import numpy as np
import pytest

from chaoslab import DoublePendulum, integrate, largest_lyapunov, rk4_step


def measure_period(times: np.ndarray, signal: np.ndarray) -> float:
    """Period from successive upward zero crossings of a mean-subtracted signal."""
    s = signal - signal.mean()
    crossings = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    if len(crossings) < 2:
        raise ValueError("not enough zero crossings")
    # linear interpolation of each crossing time
    def cross_time(i):
        frac = -s[i] / (s[i + 1] - s[i])
        return times[i] + frac * (times[i + 1] - times[i])

    cross_times = np.array([cross_time(i) for i in crossings])
    return float(np.mean(np.diff(cross_times)))


@pytest.mark.parametrize(
    ("ratio", "factor"),
    [
        (np.sqrt(2), 2 - np.sqrt(2)),   # in-phase (lower-frequency) mode
        (-np.sqrt(2), 2 + np.sqrt(2)),  # anti-phase (higher-frequency) mode
    ],
)
def test_small_angle_normal_modes(ratio, factor):
    """For the equal double pendulum the two normal-mode frequencies are
    omega^2 = (g/l)(2 ± sqrt(2)), with amplitude ratio theta2/theta1 = ∓sqrt(2).
    Excite each mode at tiny amplitude and recover its period."""
    dp = DoublePendulum(m1=1, m2=1, l1=1, l2=1, g=9.81)
    amp = 1e-3
    state0 = np.array([amp, ratio * amp, 0.0, 0.0])
    omega = np.sqrt(dp.g / dp.l1 * factor)
    analytic_period = 2 * np.pi / omega

    dt = 1e-4
    n = int(6 * analytic_period / dt)
    traj = integrate(dp, state0, dt, n, record_every=5)
    period = measure_period(traj["time"], traj["states"][:, 0])
    assert period == pytest.approx(analytic_period, rel=1e-3)


def test_energy_conserved_chaotic():
    """Even on a chaotic high-energy orbit RK4 must conserve energy tightly
    over a moderate horizon."""
    dp = DoublePendulum()
    state0 = np.array([np.pi / 2, np.pi / 2, 0.0, 0.0])
    e0 = float(dp.energy(state0))
    traj = integrate(dp, state0, 1e-4, 200_000, record_every=1000)
    drift = np.abs(dp.energy(traj["states"]) - e0)
    assert drift.max() < 1e-4


def test_energy_conserved_inverted_start():
    dp = DoublePendulum(m1=2.0, m2=1.0, l1=1.5, l2=0.8)
    state0 = np.array([2.5, -1.8, 1.0, -0.5])
    e0 = float(dp.energy(state0))
    traj = integrate(dp, state0, 5e-5, 100_000, record_every=1000)
    assert np.abs(dp.energy(traj["states"]) - e0).max() < 1e-4


def test_lyapunov_positive_when_chaotic():
    """High-energy double pendulum is chaotic: lambda_max > 0 decisively."""
    dp = DoublePendulum()
    state0 = np.array([np.pi / 2, np.pi / 2, 0.0, 0.0])
    lam = largest_lyapunov(dp, state0, n_renorm=3000)
    # The exponent for this energy is ~1 s^-1; assert a threshold that is
    # unambiguously positive yet robust to platform BLAS differences, and
    # that cleanly separates it from the regular case (which gives < 0.2).
    assert lam > 0.5


def test_lyapunov_near_zero_when_regular():
    """Tiny-amplitude motion is quasi-periodic: lambda_max ~ 0."""
    dp = DoublePendulum()
    state0 = np.array([1e-3, np.sqrt(2) * 1e-3, 0.0, 0.0])
    lam = largest_lyapunov(dp, state0, n_renorm=3000)
    assert abs(lam) < 0.2


def test_derivatives_vectorized_matches_loop():
    dp = DoublePendulum(m1=1.3, m2=0.7, l1=1.1, l2=0.9)
    rng = np.random.default_rng(0)
    batch = rng.uniform(-np.pi, np.pi, (16, 4))
    vectorized = dp.derivatives(batch)
    looped = np.array([dp.derivatives(s) for s in batch])
    assert np.allclose(vectorized, looped)


def test_batch_integration_matches_individual():
    dp = DoublePendulum()
    states = np.array([[0.5, 0.5, 0, 0], [1.0, -0.3, 0.2, 0.0]])
    batch = integrate(dp, states, 1e-3, 500)["states"][-1]
    individual = np.array(
        [integrate(dp, s, 1e-3, 500)["states"][-1] for s in states]
    )
    assert np.allclose(batch, individual)


def test_hanging_equilibrium_is_stationary():
    """Straight down with zero velocity is a fixed point."""
    dp = DoublePendulum()
    rest = np.zeros(4)
    assert np.allclose(dp.derivatives(rest), 0.0)
    after = integrate(dp, rest, 1e-3, 1000)["states"][-1]
    assert np.allclose(after, 0.0, atol=1e-12)


def test_energy_reference_point():
    """At rest hanging, E = -(m1+m2) g l1 - m2 g l2 (potential only)."""
    dp = DoublePendulum(m1=1, m2=1, l1=1, l2=1, g=9.81)
    expected = -(dp.m1 + dp.m2) * dp.g * dp.l1 - dp.m2 * dp.g * dp.l2
    assert float(dp.energy(np.zeros(4))) == pytest.approx(expected)


def test_flip_time_monotone_in_energy():
    """A pendulum released from higher up flips sooner (statistically): a
    near-vertical start flips fast, a modest start does not flip at all."""
    dp = DoublePendulum()
    fast = dp.flip_time(np.array([3.0, 3.0, 0, 0]), dt=1e-3, t_max=30.0)
    never = dp.flip_time(np.array([0.3, 0.3, 0, 0]), dt=1e-3, t_max=30.0)
    assert fast < 10.0
    assert never == 30.0  # not enough energy for the lower arm to go over


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        DoublePendulum(m1=-1.0)
    with pytest.raises(ValueError):
        DoublePendulum(l2=0.0)


def test_single_step_shape_preserved():
    dp = DoublePendulum()
    assert rk4_step(dp, np.zeros(4), 0.01).shape == (4,)
    assert rk4_step(dp, np.zeros((5, 4)), 0.01).shape == (5, 4)
