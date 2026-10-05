import numpy as np
import pytest

from approxiot.controller.adaptive_budget import ControllerConfig
from approxiot.datagen.synthetic import GAUSSIAN_TYPES, SyntheticStreamSpec
from approxiot.simulation import SimulationConfig, Simulation, LinkLatency, SyntheticSourceDriver


def _spec(rate_scale=0.004):
    return SyntheticStreamSpec(types=dict(GAUSSIAN_TYPES), rate_scale=rate_scale)


def _run(**overrides) -> object:
    cfg = SimulationConfig(num_windows=overrides.pop("num_windows", 30), **overrides)
    return Simulation(SyntheticSourceDriver(_spec()), cfg).run()


def test_native_execution_is_exact():
    result = _run(method="native", num_windows=12)
    rows = [r for r in result.rows if r["n_in_root"] > 0]
    assert rows
    for row in rows:
        assert row["approx"] == pytest.approx(row["exact"], rel=1e-12)
        assert row["realized_loss"] == pytest.approx(0.0)
    assert result.summary["accuracy_loss_mean"] == pytest.approx(0.0)


def test_approx_pipeline_pairs_windows_and_bounds_hold():
    """With the paper's latency and aligned windows every root result maps to
    exactly one source window; the 3-sigma bound should cover ~100% of pairs."""
    result = _run(method="approxiot", fraction=0.3, num_windows=40)
    rows = [r for r in result.rows if r["n_in_root"] > 0]
    # Pipeline depth = 3 hops -> results for source windows 0..T-4.
    assert len(rows) >= 36
    assert all(not np.isnan(r["exact"]) for r in rows)
    assert result.summary["bound_coverage"] >= 0.9  # z=3 confidence
    assert 0.0 < result.summary["accuracy_loss_mean"] < 0.05
    # Stratification forwarded strictly fewer items than it received.
    assert result.summary["bandwidth_saved"] > 0.0


def test_srs_pipeline_runs_and_is_lossy():
    result = _run(method="srs", fraction=0.3, num_windows=25)
    rows = [r for r in result.rows if r["n_in_root"] > 0]
    assert rows
    losses = [r["realized_loss"] for r in rows]
    assert all(np.isfinite(losses))
    assert result.summary["bandwidth_saved"] > 0.0


def test_async_jitter_exercises_eq9_without_breaking_accuracy():
    """Jittered items straddle window boundaries (paper Fig. 3), so a root
    window mixes items from several source windows.  Per-window pairing is
    therefore approximate; the cumulative sum comparison is the honest check
    that the Eq. 9 calibration keeps the estimator unbiased."""
    lat = LinkLatency(jitter_ms=400.0)  # 40% of items miss their window
    result = _run(method="approxiot", fraction=0.4, num_windows=35, latency=lat)
    rows = [r for r in result.rows if r["n_in_root"] > 0]
    assert rows
    # ~5% is the single-run cumulative noise floor at this scale (the heavy-
    # tailed D-type sum dominates); the point is there is no *systematic* bias.
    assert result.summary["cumulative_loss"] < 0.06
    # NOTE: per-window realized_loss is NOT asserted here: once windows mix
    # (n_src_windows_mixed > 1), per-window pairing compares estimates against
    # only the dominant source window, so it measures pairing noise, not error.
    # The cumulative comparison above is the honest yardstick under async.
    # Async handling actually engaged: some windows mix source windows.
    assert any(r["n_src_windows_mixed"] > 1 for r in rows)


def test_adaptive_controller_converges_in_simulation():
    """Phase 6 acceptance: the loop drives observed error toward the target.

    At this scale stderr(f=0.9) ~ 0.0035, so a target of 0.005 has a feasible
    equilibrium near f ~ 0.44: the controller should relax the over-provisioned
    start fraction down to it while the observed error stays near the target.
    """
    controller = ControllerConfig(target_error=0.005, policy="pid", cadence_windows=2,
                                  min_dwell_windows=3, dead_band=0.15, f_min=0.05)
    result = _run(method="approxiot", fraction=0.9, num_windows=60, controller=controller)
    assert result.controller_history, "controller must act"
    assert result.summary["controller_num_changes"] >= 1
    # After the loop settles, observed error should be near the target (the
    # dead-band holds it within ~15% once converged).
    tail = [r["stderr_relative"] for r in result.rows[-15:] if r["n_in_root"] > 0]
    median_tail = float(np.median(tail))
    assert 0.003 < median_tail < 0.008, median_tail
    # And the fraction must have moved off its (over-provisioned) start.
    assert result.summary["final_fraction_root"] < 0.9


def test_transport_stats_and_node_totals():
    result = _run(method="approxiot", fraction=0.5, num_windows=10)
    assert result.node_totals
    sources = [name for name in result.node_totals if name.startswith(("A", "B", "C", "D"))]
    for name in sources:
        totals = result.node_totals[name]
        assert totals["items_out"] <= totals["items_in"]
