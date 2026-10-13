"""EXP6 — The novelty: adaptive sampling-budget controller (plan Phase 6-7).

Compares three regimes on the same workloads:

1. **static-conservative**: a fraction the operator would pick to be safe
   (over-provisioned, wastes bandwidth);
2. **static-tuned**: the fraction a manual tuner would land on for the
   steady-state workload (violates the error target under rate/skew shifts);
3. **adaptive** (proportional and PID policies): closes the loop between the
   error-estimation module's relative standard error and per-node budgets.

Conditions: steady state, step-change in input rate mid-run, extreme-skew
workload, at two error targets (0.1% and 0.5%).

Metrics: bandwidth saved, error-target violation count, fraction oscillation,
time-to-reconverge after a step-change.
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from approxiot.controller.adaptive_budget import ControllerConfig
from approxiot.datagen.synthetic import FLUCTUATING_RATE_SETTINGS, SyntheticStreamSpec
from approxiot.simulation import LinkLatency, Simulation, SimulationConfig, SyntheticSourceDriver
from experiments.common import (
    NUM_WINDOWS,
    build_config,
    gaussian_spec,
    run_config,
    save_csv,
    save_fig,
    save_json,
    skew_spec,
)

STEP_WINDOWS = 20

REGIMES = (
    {"label": "static-conservative", "frac": 0.9, "controller": None},
    {"label": "static-tuned", "frac": 0.3, "controller": None},
    {"label": "adaptive-p", "frac": 0.9,
     "controller": ControllerConfig(target_error=0.005, policy="proportional",
                                    cadence_windows=2, min_dwell_windows=3,
                                    dead_band=0.3, confirmations_required=2,
                                    f_min=0.02, policy_kwargs={"max_step": 1.35})},
    {"label": "adaptive-pid", "frac": 0.9,
     "controller": ControllerConfig(target_error=0.005, policy="pid",
                                    cadence_windows=2, min_dwell_windows=3,
                                    dead_band=0.3, confirmations_required=2,
                                    f_min=0.02, policy_kwargs={"max_step": 1.35})},
)


def _schedule_with_step(window_idx: int):
    """Setting1 rates for the first STEP_WINDOWS, Setting3 after (step change)."""
    base = gaussian_spec()
    rates = (FLUCTUATING_RATE_SETTINGS["setting3"]
             if window_idx >= STEP_WINDOWS else FLUCTUATING_RATE_SETTINGS["setting1"])
    out = SyntheticStreamSpec(types=dict(base.types), rate_scale=base.rate_scale)
    out.rates = dict(rates)
    return out


def _run(cfg: SimulationConfig, spec, schedule=None):
    driver = SyntheticSourceDriver(spec, schedule=schedule)
    sim = Simulation(driver, cfg)
    return sim.run()


def _fraction_history(result) -> pd.DataFrame:
    ev = result.controller_history
    return pd.DataFrame([{"window": e.window, "frac": e.new_fraction, "action": e.action} for e in ev])


def main(seed: int = 42, windows: int = NUM_WINDOWS) -> dict:
    rows = []
    fraction_curves = []
    for scenario, spec, schedule in (
        ("steady", gaussian_spec(), None),
        ("step-change", gaussian_spec(), _schedule_with_step),
        ("skew", skew_spec(), None),
    ):
        for regime in REGIMES:
            cfg = build_config(method="approxiot", fraction=regime["frac"],
                               seed=seed, num_windows=windows,
                               latency=LinkLatency(),
                               controller=regime["controller"])
            result = _run(cfg, spec, schedule)
            df = pd.DataFrame(result.rows)
            df["scenario"] = scenario
            df["regime"] = regime["label"]
            df["target"] = regime["controller"].target_error if regime["controller"] else np.nan
            rows.append(df)
            fr = _fraction_history(result)
            if not fr.empty:
                fr["scenario"] = scenario
                fr["regime"] = regime["label"]
                fraction_curves.append(fr)
            summary = dict(result.summary)
            summary["scenario"] = scenario
            summary["regime"] = regime["label"]
            summary["violations"] = int(sum(
                1 for r in result.rows
                if np.isfinite(r["stderr_relative"]) and r["stderr_relative"] > 1.5 * (regime["controller"].target_error if regime["controller"] else 0.005)
            ))
    all_rows = pd.concat(rows, ignore_index=True)
    fraction_df = pd.concat(fraction_curves, ignore_index=True) if fraction_curves else pd.DataFrame()
    csv_path = save_csv(all_rows, "exp6_adaptive_controller.csv")
    frac_path = save_csv(fraction_df, "exp6_fraction_history.csv")

    agg = all_rows.groupby(["scenario", "regime"]).agg(
        median_loss=("realized_loss", "median"),
        bandwidth_saved=("bandwidth_saved", "mean"),
    ).reset_index()

    # Figure: fraction-over-time for the step-change scenario, adaptive vs static.
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    step = all_rows[all_rows["scenario"] == "step-change"]
    for ax, col in zip(axes, ("stderr_relative", "fraction_root")):
        for regime, g in step.groupby("regime"):
            g = g[g["n_in_root"] > 0]
            ax.plot(g["window"], g[col], marker=".", label=regime, ms=3)
        ax.axvline(STEP_WINDOWS, color="k", ls=":", alpha=0.5)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("relative stderr")
    axes[0].legend(fontsize="small")
    axes[0].set_title("EXP6: rate step-change — adaptive vs static regimes")
    axes[1].set_ylabel("sampling fraction (root)")
    axes[1].set_xlabel("window")
    fig_path = save_fig(fig, "exp6_adaptive_controller.png")

    return {
        "experiment": "exp6_adaptive_controller",
        "csv": str(csv_path),
        "fraction_history_csv": str(frac_path),
        "figure": str(fig_path),
        "aggregate": agg.to_dict(orient="records"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows", type=int, default=NUM_WINDOWS)
    args = parser.parse_args()
    r = main(seed=args.seed, windows=args.windows)
    print(r["aggregate"])
