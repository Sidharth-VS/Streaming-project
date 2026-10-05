import numpy as np
import pytest

from approxiot.controller.adaptive_budget import AdaptiveBudgetController, ControllerConfig
from approxiot.controller.policies import BanditPolicy, PIDPolicy, ProportionalPolicy, build_policy


# ---------------------------------------------------------------- policies
def test_proportional_policy_direction():
    p = ProportionalPolicy()
    up = p.update(observed_error=0.02, target=0.01, current_fraction=0.1)
    down = p.update(observed_error=0.005, target=0.01, current_fraction=0.1)
    assert up > 0.1 > down


def test_proportional_policy_step_is_capped():
    p = ProportionalPolicy(kp=1.0, max_step=2.0)
    # Massive overshoot must at most double the fraction per update.
    assert p.update(100.0, 0.01, 0.05) == 0.1


def test_pid_policy_integral_speeds_convergence():
    """Under a persistent error the integral term must push harder than P-only."""
    model = lambda f: 0.01 / np.sqrt(10 * f)  # equilibrium at f = 0.4 for target 0.005
    p_only = PIDPolicy(kp=0.5, ki=0.0, kd=0.0)
    pid = PIDPolicy(kp=0.5, ki=0.2, kd=0.1)
    fp = fpid = 0.1
    for _ in range(25):
        fp = p_only.update(model(fp), 0.005, fp)
        fpid = pid.update(model(fpid), 0.005, fpid)
    # Both are still below the equilibrium (error persists), but the integral
    # term has moved the PID fraction further towards it.
    assert fp < 0.4 and fpid > fp


def test_bandit_policy_only_picks_known_arms():
    p = BanditPolicy(arms=(0.05, 0.2, 0.8), rng=np.random.default_rng(0))
    for _ in range(50):
        f = p.update(observed_error=0.01, target=0.01, current_fraction=0.2)
        assert f in (0.05, 0.2, 0.8)


def test_build_policy_dispatch():
    assert isinstance(build_policy("proportional"), ProportionalPolicy)
    assert isinstance(build_policy("pid"), PIDPolicy)
    assert isinstance(build_policy("bandit"), BanditPolicy)
    with pytest.raises(ValueError):
        build_policy("nope")


# ------------------------------------------------------------- controller
def _error_model(fraction, scale=0.005):
    """Standard error of a stratified sum under fraction f: ~ scale / sqrt(f).

    With scale=0.005 and target 0.01 the equilibrium sits at f = 0.25.
    """
    return scale / np.sqrt(max(fraction, 1e-9))


def test_controller_converges_to_target():
    """Closed loop: fraction should settle where observed error ~ target."""
    ctrl = AdaptiveBudgetController(
        ControllerConfig(target_error=0.01, policy="proportional", cadence_windows=2,
                         min_dwell_windows=4, dead_band=0.1)
    )
    ctrl.bind_fraction(0.9)
    f = 0.9
    for t in range(200):
        observed = _error_model(f)
        msg = ctrl.observe(t, observed)
        if msg is not None:
            f = msg.fraction
    final_observed = _error_model(f)
    assert abs(final_observed - 0.01) / 0.01 < 0.5  # within 50% of target
    assert 0.01 <= f <= 1.0


def test_controller_respects_dwell_time():
    ctrl = AdaptiveBudgetController(
        ControllerConfig(target_error=0.01, min_dwell_windows=5, cadence_windows=1, dead_band=0.0)
    )
    changes = []
    for t in range(60):
        msg = ctrl.observe(t, 0.05)  # always far above target
        if msg is not None:
            changes.append(t)
    gaps = np.diff(changes)
    assert all(g >= 5 for g in gaps), gaps


def test_controller_deadband_holds_fraction():
    ctrl = AdaptiveBudgetController(
        ControllerConfig(target_error=0.01, dead_band=0.2, min_dwell_windows=1, cadence_windows=1)
    )
    # Errors within +/-20% of target must never trigger a change.
    for t, obs in enumerate([0.009, 0.011, 0.0085, 0.0115, 0.012]):
        assert ctrl.observe(t, obs) is None
    assert all(e.action == "hold_deadband" for e in ctrl.history)


def test_controller_clamps_to_bounds():
    ctrl = AdaptiveBudgetController(
        ControllerConfig(target_error=0.001, f_min=0.05, f_max=0.8,
                         dead_band=0.0, min_dwell_windows=1, cadence_windows=1)
    )
    fractions = []
    for t in range(80):
        msg = ctrl.observe(t, 10.0)  # enormous error -> want to max out
        if msg is not None:
            fractions.append(msg.fraction)
    assert max(fractions) <= 0.8

    ctrl2 = AdaptiveBudgetController(
        ControllerConfig(target_error=100.0, f_min=0.05, f_max=0.8,
                         dead_band=0.0, min_dwell_windows=1, cadence_windows=1)
    )
    fractions2 = []
    for t in range(200):
        msg = ctrl2.observe(t, 0.0001)  # tiny error -> want to min out
        if msg is not None:
            fractions2.append(msg.fraction)
    assert min(fractions2) >= 0.05


def test_controller_emits_control_messages_and_history():
    ctrl = AdaptiveBudgetController(ControllerConfig(target_error=0.01, dead_band=0.0, cadence_windows=1, min_dwell_windows=2))
    messages = [m for t in range(50) if (m := ctrl.observe(t, 0.05 + t * 1e-4)) is not None]
    assert messages
    assert all(isinstance(m.fraction, float) for m in messages)
    assert ctrl.oscillation() >= 0.0
    assert len(ctrl.fraction_history) >= 1
