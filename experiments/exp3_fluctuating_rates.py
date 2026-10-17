"""EXP3 — Fluctuating sub-stream rates (paper §V-D, Fig. 11(a,b) analogue).

Question: when the relative rate of each sub-stream type shifts over time
(A:B:C:D = Setting1 → 2 → 3 → back), how well do ApproxIoT and SRS keep up?

Both distributions from the paper are exercised: Fig. 11(a) Gaussian and
Fig. 11(b) Poisson.
"""

from __future__ import annotations

import argparse

import pandas as pd

from approxiot.datagen.synthetic import (
    FLUCTUATING_RATE_SETTINGS,
    GAUSSIAN_TYPES,
    POISSON_TYPES,
    SyntheticStreamSpec,
)
from approxiot.simulation import SyntheticSourceDriver
from experiments.common import (
    SIM_RATE_SCALE,
    build_config,
    run_config,
    save_csv,
    save_fig,
)

STEP_WINDOWS = 16  # each rate setting holds for this many windows
SETTINGS = ["setting1", "setting2", "setting3", "setting1"]


def schedule_builder(settings: list[str], base: SyntheticStreamSpec):
    """Per-window rate-setting schedule (paper §V-D: 1 → 2 → 3 → 1)."""
    cycle = [FLUCTUATING_RATE_SETTINGS[s] for s in settings]

    def schedule(window_idx: int) -> SyntheticStreamSpec:
        rates = cycle[(window_idx // STEP_WINDOWS) % len(cycle)]
        out = SyntheticStreamSpec(types=dict(base.types), rate_scale=base.rate_scale)
        out.rates = dict(rates)
        return out

    return schedule


def main(seed: int = 42, windows: int = 64) -> dict:
    rows = []
    for dist_name, types in (("gaussian", GAUSSIAN_TYPES), ("poisson", POISSON_TYPES)):
        base_spec = SyntheticStreamSpec(types=dict(types), rate_scale=SIM_RATE_SCALE)
        for method in ("approxiot", "srs"):
            cfg = build_config(method=method, fraction=0.5, seed=seed,
                               num_windows=windows)
            driver = SyntheticSourceDriver(
                base_spec, schedule=schedule_builder(SETTINGS, base_spec)
            )
            df, _summary = run_config(driver, cfg)
            df["method"] = method
            df["dist"] = dist_name
            df["phase"] = (df["window"] // STEP_WINDOWS) % 4 + 1
            rows.append(df)

    all_rows = pd.concat(rows, ignore_index=True)
    csv_path = save_csv(all_rows, "exp3_fluctuating_rates.csv")

    by_phase = all_rows.groupby(["dist", "method", "phase"])["realized_loss"].agg(
        ["mean", "median"]).reset_index()

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, dist in zip(axes, ("gaussian", "poisson")):
        sub = by_phase[by_phase["dist"] == dist]
        for method, g in sub.groupby("method"):
            ax.plot(g["phase"], g["median"].clip(lower=1e-8), marker="o", label=method)
        ax.set_yscale("log")
        ax.set_xlabel("rate-setting phase (1 -> 2 -> 3 -> 1)")
        ax.set_title(f"EXP3: fluctuating rates — {dist}")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    axes[0].set_ylabel("median accuracy loss")
    fig_path = save_fig(fig, "exp3_fluctuating_rates.png")

    checks = {}
    for dist in ("gaussian", "poisson"):
        pivoted = by_phase[by_phase["dist"] == dist].pivot(
            index="phase", columns="method", values="median").dropna()
        approx_worse = int((pivoted["approxiot"] > pivoted["srs"] * 1.5).sum())
        checks[dist] = {
            "median_loss_by_phase": pivoted.to_dict(orient="index"),
            "phases_where_approxiot_clearly_worse": approx_worse,
        }

    return {
        "experiment": "exp3_fluctuating_rates",
        "csv": str(csv_path),
        "figure": str(fig_path),
        "by_distribution": checks,
        "qualitative_check": {
            "note": "ApproxIoT median loss should stay at or below SRS's "
                    "through rate transitions (small statistical ties allowed).",
            "pass": all(c["phases_where_approxiot_clearly_worse"] == 0
                        for c in checks.values()),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows", type=int, default=64)
    args = parser.parse_args()
    print(main(seed=args.seed, windows=args.windows)["qualitative_check"])
