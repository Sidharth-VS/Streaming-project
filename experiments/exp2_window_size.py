"""EXP2 — Window size sweep (paper §V-C, Fig. 10 analogue).

Question: how does the window size affect end-to-end latency?

The in-process model has an analytic latency floor ``3 * window_seconds +
total_link_ms`` (one window hop per tree level plus the paper's 20/40/80 ms
link latencies); this experiment measures throughput empirically and overlays
the analytic latency line to validate the emulated model.
"""

from __future__ import annotations

import argparse

import pandas as pd

from approxiot.simulation import SyntheticSourceDriver
from experiments.common import (
    build_config,
    gaussian_spec,
    run_config,
    save_csv,
    save_fig,
    save_json,
)

WINDOW_SIZES = (0.5, 1.0, 2.0, 3.5, 5.0)


def main(seed: int = 42, windows: int = 48) -> dict:
    import matplotlib.pyplot as plt

    rows = []
    for w in WINDOW_SIZES:
        cfg = build_config(method="approxiot", fraction=0.5, seed=seed,
                           num_windows=windows, window_seconds=w)
        driver = SyntheticSourceDriver(gaussian_spec())
        df, _summary = run_config(driver, cfg)
        df["window_seconds"] = w
        rows.append(df)

    all_rows = pd.concat(rows, ignore_index=True)
    agg = all_rows.groupby("window_seconds").agg(
        measured_latency_ms=("e2e_latency_ms", "mean"),
        throughput=("items_in_total", "mean"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.set_xlabel("window size (seconds)")
    ax1.set_ylabel("e2e latency (ms)")
    ax1.plot(agg["window_seconds"], agg["measured_latency_ms"], marker="o",
             color="C0", label="measured e2e latency")
    analytic = sorted({3 * w * 1000.0 + 140.0 for w in WINDOW_SIZES})
    ax1.plot([w for w in WINDOW_SIZES], [3 * w * 1000.0 + 140.0 for w in
             WINDOW_SIZES], "--", color="C1", label="analytic 3·window + 140 ms")
    ax2 = ax1.twinx()
    ax2.set_ylabel("items in / window")
    ax2.bar(agg["window_seconds"], agg["throughput"], alpha=0.15, color="C2")
    ax1.legend(loc="upper left")
    ax2.legend(loc="lower right")
    ax1.set_title("EXP2: e2e latency vs window size (paper Fig. 10 analogue)")
    fig_path = save_fig(fig, "exp2_window_size.png")

    csv_path = save_csv(agg, "exp2_window_size.csv")
    return {
        "experiment": "exp2_window_size",
        "csv": str(csv_path),
        "figure": str(fig_path),
        "aggregates": agg.to_dict(orient="records"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows", type=int, default=48)
    args = parser.parse_args()
    print(main(seed=args.seed, windows=args.windows)["experiment"])
