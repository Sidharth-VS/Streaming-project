"""EXP3 — Fluctuating sub-stream rates (paper §V-D analogue).

Question: when the relative rate of each sub-stream type shifts over time
(A:B:C:D = Setting1 → 2 → 3 → back), how well do ApproxIoT and SRS keep up?

Expected qualitative outcome: ApproxIoT maintains lower per-window accuracy
loss through the transitions because stratified reservoir sampling re-balances
per sub-stream within every window, while SRS relies on luck for sub-streams
that are momentarily rare.
"""

from __future__ import annotations

import argparse

import pandas as pd

from approxiot.datagen.synthetic import FLUCTUATING_RATE_SETTINGS
from approxiot.datagen.synthetic import SyntheticStreamSpec
from approxiot.simulation import SyntheticSourceDriver
from experiments.common import build_config, gaussian_spec, run_config, save_csv, save_json

STEP_WINDOWS = 16  # each rate setting holds for this many windows


def schedule_builder(settings: list[str]):
    """Per-window rate-setting schedule (paper §V-D: 1 → 2 → 3 → 1).

    Returns a callable mapping window index -> a SyntheticStreamSpec carrying
    that phase's per-type rates.
    """
    order = list(settings)
    cycle = [FLUCTUATING_RATE_SETTINGS[s] for s in order]
    base = gaussian_spec()

    def schedule(window_idx: int) -> SyntheticStreamSpec:
        rates = cycle[(window_idx // STEP_WINDOWS) % len(cycle)]
        out = SyntheticStreamSpec(types=dict(base.types), rate_scale=base.rate_scale)
        out.rates = dict(rates)
        return out

    return schedule


def main(seed: int = 42, windows: int = 64) -> dict:
    rows = []
    for method in ("approxiot", "srs"):
        base_spec = gaussian_spec()
        cfg = build_config(method=method, fraction=0.5, seed=seed,
                           num_windows=windows)
        driver = SyntheticSourceDriver(base_spec, schedule=schedule_builder(
            ["setting1", "setting2", "setting3", "setting1"]))
        df, summary = run_config(driver, cfg)
        df["method"] = method
        # Tag each row with the rate-setting phase it falls in.
        df["phase"] = (df["window"] // STEP_WINDOWS) % 4 + 1
        rows.append(df)

    all_rows = pd.concat(rows, ignore_index=True)
    csv_path = save_csv(all_rows, "exp3_fluctuating_rates.csv")

    by_phase = all_rows.groupby(["method", "phase"])["realized_loss"].agg(
        ["mean", "median"]).reset_index()
    pivoted = by_phase.pivot(index="phase", columns="method", values="median")

    approx_worse = int((pivoted["approxiot"] > pivoted["srs"] * 1.5).sum())

    return {
        "experiment": "exp3_fluctuating_rates",
        "csv": str(csv_path),
        "median_loss_by_phase": pivoted.to_dict(orient="index"),
        "phases_where_approxiot_clearly_worse": approx_worse,
        "qualitative_check": {
            "note": "ApproxIoT median loss should stay at or below SRS's "
                    "through rate transitions (small statistical ties allowed).",
            "pass": approx_worse == 0,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows", type=int, default=64)
    args = parser.parse_args()
    print(main(seed=args.seed, windows=args.windows)["qualitative_check"])
