"""Record a finished Docker deployment as human-readable results.

The root node writes one CSV row per window to ``results/docker_results.csv``
(``RESULTS_PATH``) and prints a one-line JSON summary when it exits.  This
script turns both artifacts into something you can read, paste into a report
and show in a demo:

    python scripts/summarize_docker_results.py
    python scripts/summarize_docker_results.py --json-only          # machine view
    python scripts/summarize_docker_results.py --no-artifacts       # stdout only

Outputs (paths configurable):

    results/docker_summary.json    the recorded metrics + pass/fail checks
    results/docker_report.md       Markdown table for a README / write-up
    results/docker_report.html     self-contained charts for a demo (no server)

Only the standard library is used, so it runs on the host or in any node image.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

try:  # installed package
    from approxiot.runtime.report import (
        Painter,
        color_enabled,
        configure_stdout,
        html_report,
        markdown_report,
        render_report,
        summarize_rows,
    )
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from approxiot.runtime.report import (  # type: ignore[no-redef]
        Painter,
        color_enabled,
        configure_stdout,
        html_report,
        markdown_report,
        render_report,
        summarize_rows,
    )

# Root environment in docker-compose.yml; used when --target-error is omitted.
TARGET_ERROR = 0.005


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


def root_log_summary() -> dict | None:
    """Best-effort: the JSON line the root container printed when it exited."""
    try:
        proc = subprocess.run(
            ["docker", "compose", "logs", "root"],
            capture_output=True, timeout=30, check=False,
            # The root log is UTF-8 (table rules, checks); a Windows host would
            # otherwise try cp1252 and crash on the first box-drawing byte.
            encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if not proc.stdout:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results/docker_results.csv")
    parser.add_argument("--json-out", default="results/docker_summary.json")
    parser.add_argument("--md-out", default="results/docker_report.md")
    parser.add_argument("--html-out", default="results/docker_report.html")
    parser.add_argument("--target-error", type=float, default=TARGET_ERROR)
    parser.add_argument("--no-log", action="store_true",
                        help="do not read the root container log")
    parser.add_argument("--no-artifacts", action="store_true",
                        help="skip the Markdown/HTML reports")
    parser.add_argument("--json-only", action="store_true",
                        help="print the raw summary JSON instead of a table")
    parser.add_argument("--color", dest="color", action="store_true", default=None,
                        help="force ANSI colour")
    parser.add_argument("--no-color", dest="color", action="store_false",
                        help="never emit ANSI colour")
    args = parser.parse_args()
    configure_stdout()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"missing {csv_path} — run the stack first: docker compose up -d --build")
        return 1

    rows = load_rows(csv_path)
    if not rows:
        print(f"{csv_path} has no rows — the run produced nothing")
        return 1

    summary = summarize_rows(rows, args.target_error)
    if not args.no_log:
        log = root_log_summary()
        if log:
            summary["root_log_summary"] = log

    written: list[Path] = []
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    written.append(out)

    if not args.no_artifacts:
        for path, text in (
            (Path(args.md_out), markdown_report(rows, summary)),
            (Path(args.html_out), html_report(rows, summary)),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            written.append(path)

    if args.json_only:
        print(json.dumps(summary, indent=2))
    else:
        painter = Painter(color_enabled() if args.color is None else args.color)
        print(render_report(rows, summary, painter))

    print()
    for path in written:
        print(f"wrote {path}")
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
