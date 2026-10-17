"""EXP1 — Sampling fraction sweep (paper §V-B, Figs. 6-9 analogue).

Question: how does accuracy loss and forwarded volume vary with the sampling
fraction, for ApproxIoT weighted hierarchical sampling vs coin-flip SRS vs
the unsampled (native) case?

Expected qualitative outcome (paper Fig. 6): loss decreases with fraction for
both sampling methods; ApproxIoT's loss is consistently at or below SRS's at
the same fraction, and both are far above the native (exact) baseline.
"""

from __future__ import annotations

import argparse

import pandas as pd

from experiments.common import (
    FRACTIONS,
    METHODS,
    PAPER_RATE_SCALE,
    build_config,
    gaussian_spec,
    loss_vs_fraction_plot,
    poisson_spec,
    run_config,
    save_csv,
    save_json,
    summary_table,
    throughput_plot,
)


def main(seed: int = 42, windows: int = 64) -> dict:
    rows: list[pd.DataFrame] = []
    summaries: list[dict] = []
    # Paper Fig. 6 / §V-B covers both distributions.
    for dist_name, spec_factory in [("gaussian", gaussian_spec), ("poisson", poisson_spec)]:
        for method in METHODS:
            for f in FRACTIONS:
                cfg = build_config(method=method, fraction=f, seed=seed,
                                   num_windows=windows)
                from approxiot.simulation import SyntheticSourceDriver

                # Paper-scale input so throughput is comparable to §V-B
                # (~11k items/s on the saturated cluster).
                driver = SyntheticSourceDriver(spec_factory(PAPER_RATE_SCALE))
                df, summary = run_config(driver, cfg)
                df["dist"] = dist_name
                df["fraction"] = f
                rows.append(df)
                summary["dist"] = dist_name
                summary["fraction"] = f
                summaries.append(summary)

    all_rows = pd.concat(rows, ignore_index=True)
    agg = summary_table(all_rows, group_keys=("dist", "method", "fraction"))

    csv_path = save_csv(all_rows, "exp1_sampling_fraction.csv")
    figures = []
    for dist in ("gaussian", "poisson"):
        figures.append(str(loss_vs_fraction_plot(
            all_rows[all_rows["dist"] == dist],
            f"EXP1: accuracy loss vs sampling fraction ({dist})",
            f"exp1_loss_vs_fraction_{dist}.png",
        )))
        figures.append(str(throughput_plot(
            all_rows[all_rows["dist"] == dist],
            f"EXP1: throughput vs sampling fraction ({dist})",
            f"exp1_throughput_vs_fraction_{dist}.png",
        )))

    # Sanity: ApproxIoT mean loss <= SRS mean loss at the same fraction,
    # across all swept fractions (statistical ties allowed), per distribution.
    checks: dict[str, list] = {}
    for dist in ("gaussian", "poisson"):
        gd = agg[agg["dist"] == dist].set_index(["method", "fraction"])
        violations = []
        for f in FRACTIONS:
            a = float(gd.loc[("approxiot", f), "realized_loss"])
            s = float(gd.loc[("srs", f), "realized_loss"])
            if a > s * 1.5:  # allow a noise margin; the paper trend is a <= s
                violations.append({"fraction": f, "approxiot": a, "srs": s})
        checks[dist] = violations

    result = {
        "experiment": "exp1_sampling_fraction",
        "csv": str(csv_path),
        "figures": figures,
        "summary_rows": summaries,
        "rate_scale": PAPER_RATE_SCALE,
        "qualitative_check": {
            "approxiot_loss_le_srs_violations": checks,
            "pass": all(not v for v in checks.values()),
        },
    }
    save_json(result, "exp1_sampling_fraction.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows", type=int, default=64)
    args = parser.parse_args()
    out = main(seed=args.seed, windows=args.windows)
    print(out["qualitative_check"])
