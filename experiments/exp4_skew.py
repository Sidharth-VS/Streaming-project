"""EXP4 — Extreme sub-stream skew (paper §V-E analogue).

Question: when 80% of volume comes from type A (Poisson λ=10) and the
remaining value is dominated by type D (Poisson λ=10,000,000), how does
ApproxIoT's guaranteed per-sub-stream reservoir allocation compare to SRS?

Expected qualitative outcome (the paper's most dramatic): SRS loses the rare
but huge-value D sub-stream almost every window (coin flips drop it entirely),
so its loss is orders of magnitude worse.  ApproxIoT keeps a slot for D every
window and stays bounded.
"""

from __future__ import annotations

import argparse

import pandas as pd

from approxiot.simulation import SyntheticSourceDriver
from experiments.common import (
    build_config,
    run_config,
    save_csv,
    save_fig,
    save_json,
    skew_spec,
)


def main(seed: int = 42, windows: int = 64) -> dict:
    import matplotlib.pyplot as plt

    rows = []
    for method in ("approxiot", "srs"):
        for f in (0.1, 0.3, 0.5, 0.9):
            cfg = build_config(method=method, fraction=f, seed=seed,
                               num_windows=windows)
            driver = SyntheticSourceDriver(skew_spec())
            df, _ = run_config(driver, cfg)
            df["method"] = method
            df["fraction"] = f
            rows.append(df)

    all_rows = pd.concat(rows, ignore_index=True)
    csv_path = save_csv(all_rows, "exp4_skew.csv")

    agg = all_rows.groupby(["method", "fraction"])["realized_loss"].median().reset_index()

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method, g in agg.groupby("method"):
        ax.plot(g["fraction"], g["realized_loss"].clip(lower=1e-8), marker="o",
                label=method)
    ax.set_yscale("log")
    ax.set_xlabel("sampling fraction")
    ax.set_ylabel("median accuracy loss")
    ax.set_title("EXP4: extreme-skew workload — ApproxIoT vs SRS")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig_path = save_fig(fig, "exp4_skew.png")

    piv = agg.pivot(index="fraction", columns="method", values="realized_loss")
    approx_worse_count = int((piv["approxiot"] > piv["srs"]).sum())

    return {
        "experiment": "exp4_skew",
        "csv": str(csv_path),
        "figure": str(fig_path),
        "median_loss_by_fraction": piv.to_dict(orient="index"),
        "qualitative_check": {
            "note": "At every fraction ApproxIoT's loss should be at or below "
                    "SRS's; the gap should be widest at small fractions.",
            "fractions_approxiot_worse": approx_worse_count,
            "pass": approx_worse_count == 0,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows", type=int, default=64)
    args = parser.parse_args()
    print(main(seed=args.seed, windows=args.windows)["qualitative_check"])
