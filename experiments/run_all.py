"""``python -m experiments.run_all`` — run the full experiment suite.

Executes exp1..exp6 in order, aggregates every experiment's JSON summary into
``results/summary.json``, and prints a compact per-experiment table with the
qualitative-check verdicts.  Recommended: ``make experiments``.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

from experiments.common import RESULTS_DIR, NUM_WINDOWS


def main(seed: int = 42, windows: int = NUM_WINDOWS) -> int:
    from experiments import (exp1_sampling_fraction, exp2_window_size,
                             exp3_fluctuating_rates, exp4_skew,
                             exp5_real_world, exp6_adaptive_controller)

    modules = [
        ("exp1_sampling_fraction", exp1_sampling_fraction),
        ("exp2_window_size", exp2_window_size),
        ("exp3_fluctuating_rates", exp3_fluctuating_rates),
        ("exp4_skew", exp4_skew),
        ("exp5_real_world", exp5_real_world),
        ("exp6_adaptive_controller", exp6_adaptive_controller),
    ]

    all_results = {}
    failed = []
    for name, mod in modules:
        t0 = time.perf_counter()
        try:
            kwargs = {}
            if name in ("exp1_sampling_fraction", "exp3_fluctuating_rates",
                        "exp4_skew", "exp5_real_world", "exp6_adaptive_controller"):
                kwargs = {"seed": seed, "windows": windows}
            elif name == "exp2_window_size":
                kwargs = {"seed": seed, "windows": max(windows // 2, 24)}
            result = mod.main(**kwargs)
            all_results[name] = result
            status = "ok"
        except Exception:
            tb = traceback.format_exc()
            all_results[name] = {"error": tb}
            failed.append(name)
            status = "FAILED"
        print(f"[{status:>6}] {name:30s} {time.perf_counter() - t0:6.1f}s")

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2, default=str))

    print(f"\nSummary written to {summary_path}")
    if failed:
        print(f"FAILED experiments: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows", type=int, default=NUM_WINDOWS)
    args = parser.parse_args()
    raise SystemExit(main(seed=args.seed, windows=args.windows))
