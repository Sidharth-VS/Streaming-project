"""EXP5 — Real-world datasets (paper §VI analogue).

Replays NYC TLC ``total_amount`` and OpenAQ pm25 value pools (each with a
distribution-matched proxy fallback per plan.md §1) as streams through the
weighted hierarchical sampler vs SRS, across sampling fractions.

The ``proxy_used`` flag is recorded explicitly in the summary so the dataset
substitution is visible, not hidden (plan Phase 8 requirement).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from approxiot.datagen.real_world import load_nyc_taxi, load_openaq
from approxiot.simulation import RealWorldSourceDriver, RealWorldSourceSpec
from experiments.common import (
    build_config,
    default_latency,
    run_config,
    save_csv,
    save_fig,
    save_json,
    summary_table,
)

FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
METHODS = ("approxiot", "srs", "native")


def main(seed: int = 42, windows: int = 32) -> dict:
    taxi_pool = load_nyc_taxi(download=False, rng=np.random.default_rng(seed))
    aq_pool = load_openaq(download=False, rng=np.random.default_rng(seed + 1))
    proxy_flags = {"nyc_taxi": taxi_pool.proxy_used, "openaq": aq_pool.proxy_used}

    def driver_factory() -> RealWorldSourceDriver:
        return RealWorldSourceDriver([
            RealWorldSourceSpec(name="taxi_src1", pool=taxi_pool, rate_per_sec=300.0),
            RealWorldSourceSpec(name="taxi_src2", pool=taxi_pool, rate_per_sec=300.0),
            RealWorldSourceSpec(name="aq_src1", pool=aq_pool, rate_per_sec=200.0),
            RealWorldSourceSpec(name="aq_src2", pool=aq_pool, rate_per_sec=200.0),
        ])

    rows = []
    summaries = []
    for method in METHODS:
        for f in FRACTIONS:
            cfg = build_config(method=method, fraction=f, seed=seed,
                               num_windows=windows, latency=default_latency())
            df, summary = run_config(driver_factory(), cfg)
            df["fraction"] = f
            rows.append(df)
            summary["fraction"] = f
            summaries.append(summary)
    all_rows = pd.concat(rows, ignore_index=True)
    agg = summary_table(all_rows, group_keys=("method", "fraction"))
    csv_path = save_csv(all_rows, "exp5_real_world.csv")

    piv = agg.pivot(index="fraction", columns="method", values="realized_loss")
    approx_worse = int((piv["approxiot"] > piv["srs"] * 1.3).sum())

    # Multi-seed re-run (mean over 3 seeds) to de-noise single-run sampling
    # variance before the trend verdict; recorded alongside the single-seed
    # table (paper Fig. 11 analogue).
    import numpy as _np
    multi_rows = []
    for s in range(3):
        t_pool = load_nyc_taxi(download=False, rng=_np.random.default_rng(100 + s))
        a_pool = load_openaq(download=False, rng=_np.random.default_rng(200 + s))
        def driver_factory_s():
            return RealWorldSourceDriver([
                RealWorldSourceSpec(name="taxi_src1", pool=t_pool, rate_per_sec=300.0),
                RealWorldSourceSpec(name="taxi_src2", pool=t_pool, rate_per_sec=300.0),
                RealWorldSourceSpec(name="aq_src1", pool=a_pool, rate_per_sec=200.0),
                RealWorldSourceSpec(name="aq_src2", pool=a_pool, rate_per_sec=200.0),
            ])
        for method in METHODS:
            for f in FRACTIONS:
                cfg = build_config(method=method, fraction=f, seed=seed + s,
                                   num_windows=windows, latency=default_latency())
                df, _ = run_config(driver_factory_s(), cfg)
                df["fraction"] = f
                df["method"] = method
                df["run_seed"] = seed + s
                multi_rows.append(df)
    multi_df = pd.concat(multi_rows, ignore_index=True)
    multi_agg = summary_table(multi_df, group_keys=("method", "fraction"))
    multi_path = save_csv(multi_df, "exp5_real_world_multiseed.csv")
    piv_multi = multi_agg.pivot(index="fraction", columns="method", values="realized_loss")
    approx_worse_multi = int((piv_multi["approxiot"] > piv_multi["srs"] * 1.3).sum())

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method, g in multi_agg.groupby("method"):
        g = g.sort_values("fraction")
        ax.plot(g["fraction"], g["realized_loss"].clip(lower=1e-8), marker="o",
                label=method)
    ax.set_yscale("log")
    ax.set_xlabel("sampling fraction")
    ax.set_ylabel("median accuracy loss (3 seeds)")
    proxy_note = " + ".join(f"{name}{'(proxy)' if proxy else ''}"
                            for name, proxy in proxy_flags.items())
    ax.set_title(f"EXP5: real-world workload ({proxy_note})")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig_path = save_fig(fig, "exp5_real_world.png")

    return {
        "experiment": "exp5_real_world",
        "csv": str(csv_path),
        "multiseed_csv": str(multi_path),
        "figure": str(fig_path),
        "proxy_datasets_used": proxy_flags,
        "summary_rows": summaries,
        "qualitative_check": {
            "approxiot_loss_le_srs_violations_single_seed": approx_worse,
            "approxiot_loss_le_srs_violations_multiseed": approx_worse_multi,
            "pass": approx_worse_multi == 0,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows", type=int, default=32)
    args = parser.parse_args()
    print(main(seed=args.seed, windows=args.windows)["qualitative_check"])
