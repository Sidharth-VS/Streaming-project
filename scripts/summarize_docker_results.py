"""Extract the recorded results of the containerized ApproxIoT deployment.

The root node writes one CSV row per window to ``results/docker_results.csv``
(``RESULTS_PATH``) and prints a one-line JSON summary when it exits.  This
script turns those two raw artifacts into something recordable:

    python scripts/summarize_docker_results.py
    python scripts/summarize_docker_results.py --csv results/docker_results.csv --json-out results/docker_summary.json

Pure standard library, so it runs on the host or inside any node container.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path

# Root environment in docker-compose.yml; used to judge the accuracy columns.
TARGET_ERROR = 0.005
WARMUP_ROWS = 3  # first root windows have n_in_root == 0 (documented in RUNNING.md)


def load_rows(csv_path: Path) -> list[dict[str, float]]:
    with csv_path.open(newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row: dict[str, float] = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = float("nan")
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, float]], target_error: float) -> dict:
    measured = [r for r in rows if r["n_in_root"] > 0]
    errors = [r["stderr_relative"] for r in measured if r["stderr_relative"] == r["stderr_relative"]]
    latency = [r["e2e_latency_ms"] for r in measured if r["e2e_latency_ms"] == r["e2e_latency_ms"]]
    fractions = [r["fraction_root"] for r in measured]
    saved = [r["bandwidth_saved"] for r in measured]
    items_in = measured[-1]["items_in_total"] if measured else 0.0
    items_out = measured[-1]["items_out_total"] if measured else 0.0

    within_target = sum(1 for e in errors if e <= target_error)
    summary = {
        "csv_rows": len(rows),
        "warmup_rows_skipped": len(rows) - len(measured),
        "windows_with_results": len(measured),
        "target_error": target_error,
        "median_relative_stderr": round(statistics.median(errors), 6) if errors else None,
        "max_relative_stderr": round(max(errors), 6) if errors else None,
        "windows_within_target": within_target,
        "windows_within_target_pct": round(100.0 * within_target / len(errors), 1) if errors else None,
        "median_latency_ms": round(statistics.median(latency), 1) if latency else None,
        "final_fraction": round(fractions[-1], 6) if fractions else None,
        "fraction_min": round(min(fractions), 6) if fractions else None,
        "fraction_max": round(max(fractions), 6) if fractions else None,
        "fraction_changes": sum(
            1 for a, b in zip(fractions, fractions[1:]) if abs(a - b) > 1e-9
        ),
        "items_in_total": int(items_in),
        "items_out_total": int(items_out),
        "bandwidth_saved_total": round(1.0 - items_out / items_in, 6) if items_in else None,
        "median_bandwidth_saved": round(statistics.median(saved), 6) if saved else None,
    }
    # A run is only usable if the tree actually forwarded data and the
    # controller held the accuracy target.
    summary["checks"] = {
        "data_flowed": bool(measured) and items_in > 0,
        "accuracy_target_met": bool(errors) and summary["median_relative_stderr"] <= target_error,
        "bandwidth_saved_positive": (summary["bandwidth_saved_total"] or 0) > 0,
        "controller_acted": summary["fraction_changes"] > 0,
    }
    summary["pass"] = all(summary["checks"].values())
    return summary


def root_log_summary() -> dict | None:
    """Best-effort: the JSON line the root container printed when it exited."""
    try:
        proc = subprocess.run(
            ["docker", "compose", "logs", "root"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in reversed(proc.stdout.splitlines()):
        _, _, payload = line.partition("| ")
        payload = payload.strip()
        if payload.startswith("{") and '"node"' in payload:
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                continue
    return None


def print_table(rows: list[dict[str, float]]) -> None:
    header = f"{'win':>4} {'n_in':>7} {'sampled':>8} {'f':>7} {'rel_stderr':>11} {'bw_saved':>9} {'latency_ms':>11}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["n_in_root"] <= 0:
            continue
        print(
            f"{int(r['window']):>4} {int(r['n_in_root']):>7} {int(r['n_sampled_root']):>8} "
            f"{r['fraction_root']:>7.3f} {r['stderr_relative']:>11.5f} "
            f"{r['bandwidth_saved']:>9.3f} {r['e2e_latency_ms']:>11.1f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results/docker_results.csv")
    parser.add_argument("--json-out", default="results/docker_summary.json")
    parser.add_argument("--target-error", type=float, default=TARGET_ERROR)
    parser.add_argument("--no-log", action="store_true", help="skip reading the root container log")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"missing {csv_path} — run the stack first: docker compose up -d --build")
        return 1

    rows = load_rows(csv_path)
    if not rows:
        print(f"{csv_path} has no rows — the run produced nothing")
        return 1

    summary = summarize(rows, args.target_error)
    if not args.no_log:
        log = root_log_summary()
        if log:
            summary["root_log_summary"] = log

    print_table(rows)
    print()
    print(json.dumps(summary, indent=2))

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
