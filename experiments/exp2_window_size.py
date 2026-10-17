"""EXP2 — Window size sweep (paper §V-C, Fig. 10 analogue).

Question: how does the window size affect end-to-end latency?

Latency is measured from the per-item event trace (``approxiot.latency``):
each item is timestamped when created, waits for window boundaries
(ApproxIoT/native), spends n/capacity seconds in processing at each node, and
crosses the paper's 20/40/80 ms links.  SRS is **windowless** per §V-C, so its
latency should stay flat while ApproxIoT's grows with the window.

The experiment sweeps the paper's window sizes at a fixed 10% sampling
fraction and compares ApproxIoT with SRS and the native (no-sampling)
execution.  Throughput is reported in simulated items/second (paper §V-B
definition), not Python wall-clock speed, so it does not rise with window
size.
"""

from __future__ import annotations

import argparse

import pandas as pd

from approxiot.simulation import SyntheticSourceDriver
from experiments.common import (
    PAPER_RATE_SCALE,
    build_config,
    gaussian_spec,
    run_config,
    save_csv,
    save_fig,
)

WINDOW_SIZES = (0.5, 1.0, 2.0, 3.5, 5.0)
METHODS = ("approxiot", "srs", "native")
FRACTION = 0.10  # paper Fig. 10: fixed sampling fraction


def main(seed: int = 42, windows: int = 48) -> dict:
    import matplotlib.pyplot as plt

    rows = []
    for method in METHODS:
        for w in WINDOW_SIZES:
            cfg = build_config(method=method, fraction=FRACTION, seed=seed,
                               num_windows=windows, window_seconds=w)
            driver = SyntheticSourceDriver(gaussian_spec(PAPER_RATE_SCALE))
            df, _summary = run_config(driver, cfg)
            df["window_seconds"] = w
            df["method"] = method
            rows.append(df)

    all_rows = pd.concat(rows, ignore_index=True)
    agg = all_rows.groupby(["window_seconds", "method"]).agg(
        measured_latency_ms=("e2e_latency_ms", "mean"),
        measured_latency_p95_ms=("e2e_latency_p95_ms", "mean"),
        throughput=("ingest_throughput_items_per_sec", "mean"),
    ).reset_index()

    fig, (ax_lat, ax_thr) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for method, g in agg.groupby("method"):
        g = g.sort_values("window_seconds")
        ax_lat.plot(g["window_seconds"], g["measured_latency_ms"], marker="o",
                    label=method)
        ax_thr.plot(g["window_seconds"], g["throughput"], marker="o", label=method)
    ax_lat.set_xlabel("window size (seconds)")
    ax_lat.set_ylabel("measured e2e latency (ms)")
    ax_lat.set_title("EXP2: latency vs window size (paper Fig. 10 analogue)")
    ax_lat.legend()
    ax_lat.grid(alpha=0.3)
    ax_thr.set_xlabel("window size (seconds)")
    ax_thr.set_ylabel("items processed / simulated second")
    ax_thr.set_title("EXP2: simulated throughput (no window-size artifact)")
    ax_thr.legend()
    ax_thr.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = save_fig(fig, "exp2_window_size.png")

    csv_path = save_csv(agg, "exp2_window_size.csv")
    rows_path = save_csv(all_rows, "exp2_window_size_rows.csv")

    piv = agg.pivot(index="window_seconds", columns="method",
                    values="measured_latency_ms")
    approx_growth = float(piv["approxiot"].iloc[-1] / piv["approxiot"].iloc[0])
    srs_ratio = float(piv["srs"].iloc[-1] / piv["srs"].iloc[0])
    native_faster = int((piv["approxiot"] < piv["native"]).sum())
    return {
        "experiment": "exp2_window_size",
        "csv": str(csv_path),
        "rows_csv": str(rows_path),
        "figure": str(fig_path),
        "aggregates": agg.to_dict(orient="records"),
        "qualitative_check": {
            "note": "ApproxIoT latency should grow with the window; windowless "
                    "SRS should stay flat; sampling should beat native at 10%.",
            "approxiot_last_over_first": approx_growth,
            "srs_last_over_first": srs_ratio,
            "windows_where_approxiot_beats_native": native_faster,
            "pass": approx_growth > 2.0 and srs_ratio < 1.05
                    and native_faster == len(WINDOW_SIZES),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows", type=int, default=48)
    args = parser.parse_args()
    print(main(seed=args.seed, windows=args.windows)["qualitative_check"])
