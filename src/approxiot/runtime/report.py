"""Human-readable rendering of deployment results (standard library only).

The containerized run has two audiences:

* a person watching ``docker compose logs -f root`` during a demo, and
* the tooling that records the run (``scripts/summarize_docker_results.py``).

Both go through this module so the numbers on screen, in
``results/docker_summary.json``, in ``results/docker_report.md`` and in
``results/docker_report.html`` can never disagree.  Nothing here imports
numpy/pandas, so it works inside the slim node image and on a bare host
interpreter alike.

Layout: helpers -> aggregation -> console -> artifacts.
"""

from __future__ import annotations

import html
import math
import os
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

WIDTH = 78
THIN = "\u2500"  # ─
THICK = "\u2550"  # ═
ARROW = "\u2192"  # →
BULLET = "\u00b7"  # ·
OK = "\u2714"  # ✔
NO = "\u2718"  # ✘
MARK_OK = "\u2713"  # ✓
MARK_BAD = "\u2717"  # ✗
SPARKS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

# --------------------------------------------------------------------- style

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
CYAN = "\x1b[36m"


def configure_stdout() -> None:
    """Force UTF-8 output so the box-drawing glyphs survive a Windows console.

    ``docker compose logs`` and the containers are UTF-8 already; this is for a
    bare Windows host interpreter whose default code page is cp1252.
    """
    stream = sys.stdout
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        if (getattr(stream, "encoding", "") or "").lower() not in {"utf-8", "utf8"}:
            reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass


def color_enabled(stream: Any = None) -> bool:
    """ANSI on an interactive terminal only, unless LOG_COLOR says otherwise."""
    raw = (os.environ.get("LOG_COLOR") or "").strip().lower()
    if raw in {"1", "true", "yes", "always", "on"}:
        return True
    if raw in {"0", "false", "no", "never", "off"}:
        return False
    stream = stream if stream is not None else sys.stdout
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


class Painter:
    """Tiny ANSI wrapper; a no-op painter keeps plain text copy/pasteable."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def paint(self, text: str, code: str) -> str:
        if not self.enabled or not text:
            return text
        return f"{code}{text}{RESET}"

    def bold(self, text: str) -> str:
        return self.paint(text, BOLD)

    def dim(self, text: str) -> str:
        return self.paint(text, DIM)

    def green(self, text: str) -> str:
        return self.paint(text, GREEN)

    def yellow(self, text: str) -> str:
        return self.paint(text, YELLOW)

    def red(self, text: str) -> str:
        return self.paint(text, RED)

    def cyan(self, text: str) -> str:
        return self.paint(text, CYAN)


# ------------------------------------------------------------------- numbers


def is_number(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def human_int(value: Any) -> str:
    """``2178`` -> ``2,178``; anything unusable -> an em dash."""
    if not is_number(value):
        return "\u2014"
    return f"{int(round(float(value))):,}"


def pct(value: Any, digits: int = 1) -> str:
    """Fraction (0.662) -> ``66.2%``."""
    if not is_number(value):
        return "\u2014"
    return f"{100.0 * float(value):.{digits}f}%"


_UNITS = ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k"))


def _unit_for(scale: float) -> tuple[float, str]:
    for threshold, suffix in _UNITS:
        if scale >= threshold:
            return threshold, suffix
    return 1.0, ""


def si_pair(value: Any, bound: Any = None, digits: int = 2) -> str:
    """``20748180`` with bound ``255743`` -> ``20.75 M ± 0.26 M``.

    A shared unit keeps the pair readable and makes column widths stable.
    """
    if not is_number(value):
        return "\u2014"
    head, tail = si_parts(value, bound, digits)
    return head if not is_number(bound) else f"{head} \u00b1 {tail}"


def si_parts(value: Any, bound: Any = None, digits: int = 2) -> tuple[str, str]:
    """Estimate and error bound formatted in the *same* unit.

    Shared units keep separate table columns consistent:
    ``42.30 M`` / ``1.30 M`` rather than ``42.30 M`` / ``1.30 G``-style drift.
    """
    if not is_number(value):
        return "\u2014", "\u2014"
    value = float(value)
    scale = abs(value)
    if is_number(bound):
        scale = max(scale, abs(float(bound)))
    divisor, unit = _unit_for(scale)

    def fmt(number: float) -> str:
        return f"{number / divisor:,.{digits}f} {unit}".strip()

    return fmt(value), fmt(float(bound)) if is_number(bound) else "\u2014"


def latency(ms: Any) -> str:
    if not is_number(ms):
        return "\u2014"
    ms = float(ms)
    if ms >= 1000.0:
        return f"{ms / 1000.0:.2f} s"
    return f"{ms:.0f} ms"


def duration(seconds: Any) -> str:
    if not is_number(seconds):
        return "\u2014"
    seconds = int(round(float(seconds)))
    if seconds < 60:
        return f"{seconds} s"
    minutes, rest = divmod(seconds, 60)
    return f"{minutes} m {rest:02d} s"


def sparkline(values: Iterable[Any], width: int | None = None) -> str:
    """Unicode sparkline; evenly downsamples when ``width`` is smaller."""
    vals = [float(v) for v in values if is_number(v)]
    if not vals:
        return ""
    if width and len(vals) > width:
        step = len(vals) / float(width)
        buckets = []
        for i in range(width):
            chunk = vals[int(i * step) : max(int((i + 1) * step), int(i * step) + 1)]
            buckets.append(sum(chunk) / len(chunk))
        vals = buckets
    low, high = min(vals), max(vals)
    if high - low <= 1e-12:
        return SPARKS[len(SPARKS) // 2] * len(vals)
    # round(), not int(): float jitter between two near-identical values (e.g.
    # 0.675 vs 0.6750000000000001) must not make a flat run look oscillating.
    return "".join(
        SPARKS[min(len(SPARKS) - 1, max(0, int(round((v - low) / (high - low) * (len(SPARKS) - 1)))))]
        for v in vals
    )


def rule(width: int = WIDTH, char: str = THIN) -> str:
    return char * width


# --------------------------------------------------------------- aggregation


def summarize_rows(rows: Sequence[Mapping[str, Any]], target_error: float) -> dict:
    """Aggregate root CSV rows into the recorded run summary.

    Rows with ``n_in_root == 0`` are warm-up: the tree has not delivered a
    window's worth of samples yet, so they carry no estimate.
    """
    measured = [r for r in rows if float(r.get("n_in_root") or 0.0) > 0.0]
    errors = [float(r["stderr_relative"]) for r in measured if is_number(r.get("stderr_relative"))]
    latency_ms = [float(r["e2e_latency_ms"]) for r in measured if is_number(r.get("e2e_latency_ms"))]
    fractions = [float(r["fraction_root"]) for r in measured if is_number(r.get("fraction_root"))]
    # Budget changes are controller actions, so count them across every row:
    # the controller already moves during warm-up, before estimates exist.
    all_fractions = [float(r["fraction_root"]) for r in rows if is_number(r.get("fraction_root"))]
    saved = [float(r["bandwidth_saved"]) for r in measured if is_number(r.get("bandwidth_saved"))]
    items_in = float(measured[-1].get("items_in_total") or 0.0) if measured else 0.0
    items_out = float(measured[-1].get("items_out_total") or 0.0) if measured else 0.0

    within = sum(1 for e in errors if e <= target_error) if target_error > 0 else 0
    summary: dict[str, Any] = {
        "windows_total": len(rows),
        "csv_rows": len(rows),
        "warmup_rows_skipped": len(rows) - len(measured),
        "windows_with_results": len(measured),
        "target_error": target_error,
        "median_relative_stderr": round(statistics.median(errors), 6) if errors else None,
        "max_relative_stderr": round(max(errors), 6) if errors else None,
        "windows_within_target": within,
        "windows_within_target_pct": round(100.0 * within / len(errors), 1) if errors else None,
        "median_latency_ms": round(statistics.median(latency_ms), 1) if latency_ms else None,
        "fraction_start": round(all_fractions[0], 6) if all_fractions else None,
        "final_fraction": round(fractions[-1], 6) if fractions else None,
        "fraction_min": round(min(fractions), 6) if fractions else None,
        "fraction_max": round(max(fractions), 6) if fractions else None,
        "fraction_changes": sum(
            1 for a, b in zip(all_fractions, all_fractions[1:]) if abs(a - b) > 1e-9
        ),
        "items_in_total": int(items_in),
        "items_out_total": int(items_out),
        "bandwidth_saved_total": round(1.0 - items_out / items_in, 6) if items_in else None,
        "median_bandwidth_saved": round(statistics.median(saved), 6) if saved else None,
    }
    summary["checks"] = {
        "data_flowed": bool(measured) and items_in > 0,
        "accuracy_target_met": bool(errors)
        and target_error > 0
        and summary["median_relative_stderr"] <= target_error,
        "bandwidth_saved_positive": (summary["bandwidth_saved_total"] or 0.0) > 0.0,
        "controller_acted": summary["fraction_changes"] > 0,
    }
    summary["pass"] = all(summary["checks"].values())
    return summary


_CHECK_LABELS = (
    ("data_flowed", "data flowed"),
    ("accuracy_target_met", "accuracy"),
    ("bandwidth_saved_positive", "bandwidth"),
    ("controller_acted", "controller acted"),
)


# ------------------------------------------------------------------- console


def banner(
    title: str,
    facts: Sequence[str],
    detail: Sequence[str] = (),
    width: int = WIDTH,
    p: Painter | None = None,
) -> str:
    """Run header: one title, a few facts, then config detail."""
    p = p or Painter(False)
    lines = [rule(width, THICK), f" {p.bold(title)}"]
    for fact in facts:
        lines.append(f" {fact}")
    if detail:
        lines.append(p.dim(f" {BULLET} ".join(detail)))
    lines.append(rule(width, THICK))
    return "\n".join(lines)


def node_ready(name: str, role: str, parent: str | None, p: Painter | None = None) -> str:
    p = p or Painter(False)
    target = f" {ARROW} {parent}" if parent else ""
    return f"{p.cyan('[' + name + ']')} {role} ready{target}"


def table_header() -> str:
    return (
        f"{'win':>4} {'n_in':>7} {'sampled':>7} {'budget':>6} "
        f"{'estimate \u00b1 bound':>19} {'rel.err':>9} {'saved':>6} {'e2e':>7}"
    )


def window_row(
    window: Any,
    n_in: Any,
    n_sampled: Any,
    fraction: Any,
    approx: Any,
    error_bound: Any,
    stderr_relative: Any,
    bandwidth_saved: Any,
    latency_ms: Any,
    target_error: float | None = None,
    p: Painter | None = None,
) -> str:
    p = p or Painter(False)
    err_text = pct(stderr_relative, 2)
    ok: bool | None = None
    if target_error and target_error > 0 and is_number(stderr_relative):
        ok = float(stderr_relative) <= target_error
    cell = f"{err_text} {MARK_OK if ok else MARK_BAD}" if ok is not None else err_text
    plain_cell = cell.ljust(9)
    if ok is not None:
        mark = MARK_OK if ok else MARK_BAD
        plain_cell = plain_cell.replace(mark, p.green(mark) if ok else p.yellow(mark), 1)
    return (
        f"{int(window):>4} {human_int(n_in):>7} {human_int(n_sampled):>7} "
        f"{pct(fraction, 1):>6} {si_pair(approx, error_bound):>19} {plain_cell} "
        f"{pct(bandwidth_saved, 1):>6} {latency(latency_ms):>7}"
    )


def controller_change(
    window: Any,
    old_fraction: Any,
    new_fraction: Any,
    observed_error: Any,
    target_error: Any,
    p: Painter | None = None,
) -> str:
    p = p or Painter(False)
    arrow = p.cyan(f"{pct(old_fraction, 1)} {ARROW} {pct(new_fraction, 1)}")
    if is_number(observed_error) and float(observed_error) == 0.0:
        detail = "no estimate yet (warm-up)"
    elif is_number(observed_error) and is_number(target_error):
        detail = f"rel.err {pct(observed_error, 2)} vs target {pct(target_error, 2)}"
    else:
        detail = ""
    head = p.dim(f"        \u21b3 controller @ win {int(window)}:")
    return f"{head} budget {arrow}  {p.dim(detail)}".rstrip()


def node_done(summary: Mapping[str, Any], p: Painter | None = None) -> str:
    p = p or Painter(False)
    name = summary.get("node", "?")
    role = summary.get("role", "node")
    saved = summary.get("bandwidth_saved")
    return (
        f"{p.cyan('[' + str(name) + ']')} {role} done "
        f"{BULLET} {human_int(summary.get('windows'))} windows "
        f"{BULLET} {human_int(summary.get('items_in'))} in {ARROW} "
        f"{human_int(summary.get('items_out'))} out "
        f"{BULLET} saved {pct(saved, 1)} "
        f"{BULLET} budget {pct(summary.get('final_fraction'), 1)}"
    )


def _panel_row(label: str, value: str, width: int, p: Painter, status: str | None = None,
               color: str = GREEN) -> str:
    text = f"  {label:<13}{value}"
    if len(text) > width:
        text = text[: width - 1] + "\u2026"
    if status and p.enabled:
        idx = text.find(status)
        if idx >= 0:
            text = text[:idx] + p.paint(status, color) + text[idx + len(status) :]
    return text


def root_panel(
    summary: Mapping[str, Any],
    fraction_series: Sequence[Any] | None = None,
    stderr_series: Sequence[Any] | None = None,
    width: int = WIDTH,
    p: Painter | None = None,
) -> str:
    """Final dashboard for a run: numbers first, verdict last."""
    p = p or Painter(False)
    target = summary.get("target_error") or 0.0
    measured = summary.get("windows_with_results")
    warmup = summary.get("warmup_rows_skipped")
    within = summary.get("windows_within_target")
    within_pct = summary.get("windows_within_target_pct")
    median_err = summary.get("median_relative_stderr")
    max_err = summary.get("max_relative_stderr")
    checks = summary.get("checks") or {}
    passed = summary.get("pass")

    lines = [rule(width, THICK), f" {p.bold('ApproxIoT ' + str(summary.get('node', 'root')) + ' summary')}"]
    lines.append(rule(width, THIN))

    windows = human_int(measured)
    if is_number(warmup) and float(warmup or 0) > 0:
        windows += f" measured  ({human_int(warmup)} warm-up)"
    lines.append(_panel_row("windows", windows, width, p))

    acc = f"median rel. stderr {pct(median_err, 2)}"
    if target and float(target) > 0:
        acc += f"  vs target \u2264 {pct(target, 3)}"
    if checks:
        acc += f"   {OK if checks.get('accuracy_target_met') else NO} " + (
            "PASS" if checks.get("accuracy_target_met") else "FAIL"
        )
    lines.append(
        _panel_row(
            "accuracy", acc, width, p,
            status="PASS" if checks.get("accuracy_target_met") else "FAIL",
        )
    )
    if is_number(within) and is_number(within_pct):
        lines.append(
            _panel_row(
                "",
                f"within target {human_int(within)}/{human_int(measured)} ({pct(within_pct / 100.0, 1)})"
                + (f"  {BULLET} worst {pct(max_err, 2)}" if is_number(max_err) else ""),
                width,
                p,
            )
        )

    saved_total = summary.get("bandwidth_saved_total", summary.get("bandwidth_saved"))
    saved_median = summary.get("median_bandwidth_saved")
    bw = f"saved {pct(saved_total, 1)} total"
    if is_number(saved_median):
        bw += f"  {BULLET} median {pct(saved_median, 1)}"
    lines.append(_panel_row("bandwidth", bw, width, p))

    lines.append(
        _panel_row(
            "latency",
            f"median {latency(summary.get('median_latency_ms'))} end-to-end",
            width,
            p,
        )
    )

    changes = summary.get("fraction_changes", summary.get("controller_num_changes"))
    start = summary.get("fraction_start", summary.get("initial_fraction"))
    final = summary.get("final_fraction")
    ctrl = f"{human_int(changes)} change{'s' if (changes or 0) != 1 else ''}"
    if is_number(start) and is_number(final):
        ctrl += f"  {BULLET} {pct(start, 1)} {ARROW} {pct(final, 1)}"
    spark_values = [v for v in (fraction_series or []) if is_number(v)]
    if spark_values:
        ctrl += f"  {sparkline(spark_values, 24)}"
    lines.append(_panel_row("controller", ctrl, width, p))

    lines.append(
        _panel_row(
            "items",
            f"{human_int(summary.get('items_in_total', summary.get('items_in')))} in {ARROW} "
            f"{human_int(summary.get('items_out_total', summary.get('items_out')))} out",
            width,
            p,
        )
    )

    if stderr_series:
        errs = [v for v in stderr_series if is_number(v)]
        if errs:
            lines.append(_panel_row("error trace", f"{sparkline(errs, 24)}  (per window)", width, p))

    if checks:
        marks = "  ".join(
            f"{OK if checks.get(key) else NO} {label}" for key, label in _CHECK_LABELS
        )
        lines.append(rule(width, THIN))
        lines.append(
            _panel_row(
                "verdict",
                "PASS" if passed else "FAIL",
                width,
                p,
                status="PASS" if passed else "FAIL",
                color=GREEN if passed else RED,
            )
        )
        lines.append(_panel_row("checks", marks, width, p))
    lines.append(rule(width, THICK))
    return "\n".join(lines)


def render_report(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    p: Painter | None = None,
) -> str:
    """Console view of a finished run: per-window table + summary panel."""
    p = p or Painter(False)
    target = summary.get("target_error")
    measured = [r for r in rows if float(r.get("n_in_root") or 0.0) > 0.0]
    out = [table_header(), rule(len(table_header()))]
    previous_fraction = None
    for row in measured:
        out.append(
            window_row(
                row.get("window"),
                row.get("n_in_root"),
                row.get("n_sampled_root"),
                row.get("fraction_root"),
                row.get("approx"),
                row.get("error_bound"),
                row.get("stderr_relative"),
                row.get("bandwidth_saved"),
                row.get("e2e_latency_ms"),
                target_error=target,
                p=p,
            )
        )
        fraction = row.get("fraction_root")
        if is_number(fraction) and is_number(previous_fraction):
            if abs(float(fraction) - float(previous_fraction)) > 1e-9:
                out.append(
                    f"        \u21b3 budget change: {pct(previous_fraction, 1)} {ARROW} "
                    f"{pct(fraction, 1)}"
                )
        if is_number(fraction):
            previous_fraction = fraction
    out.append("")
    out.append(
        root_panel(
            summary,
            fraction_series=[r.get("fraction_root") for r in measured],
            stderr_series=[r.get("stderr_relative") for r in measured],
            p=p,
        )
    )
    return "\n".join(out)


# ----------------------------------------------------------------- artifacts


def _check_line(summary: Mapping[str, Any]) -> str:
    checks = summary.get("checks") or {}
    if not checks:
        return "\u2014"
    return "  ".join(f"{OK if checks.get(k) else NO} {label}" for k, label in _CHECK_LABELS)


def _metric_rows(summary: Mapping[str, Any]) -> list[tuple[str, str]]:
    saved_total = summary.get("bandwidth_saved_total", summary.get("bandwidth_saved"))
    return [
        ("windows", f"{human_int(summary.get('windows_with_results'))} measured"
                    f" ({human_int(summary.get('warmup_rows_skipped'))} warm-up)"),
        ("median rel. stderr", pct(summary.get("median_relative_stderr"), 3)),
        ("worst rel. stderr", pct(summary.get("max_relative_stderr"), 3)),
        ("target", pct(summary.get("target_error"), 3)),
        ("within target", f"{human_int(summary.get('windows_within_target'))} "
                          f"({pct((summary.get('windows_within_target_pct') or 0) / 100.0, 1)})"),
        ("bandwidth saved", pct(saved_total, 1)),
        ("median bandwidth saved", pct(summary.get("median_bandwidth_saved"), 1)),
        ("median latency", latency(summary.get("median_latency_ms"))),
        ("budget changes", human_int(summary.get("fraction_changes"))),
        ("budget range", f"{pct(summary.get('fraction_min'), 1)} \u2013 {pct(summary.get('fraction_max'), 1)}"),
        ("items in / out", f"{human_int(summary.get('items_in_total'))} / "
                           f"{human_int(summary.get('items_out_total'))}"),
    ]


def markdown_report(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> str:
    """Self-contained Markdown: verdict, metrics, per-window table."""
    measured = [r for r in rows if float(r.get("n_in_root") or 0.0) > 0.0]
    verdict = "PASS" if summary.get("pass") else "FAIL"
    lines = [
        "# ApproxIoT deployment run",
        "",
        f"**Verdict: {verdict}** \u2014 {_check_line(summary)}",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    lines += [f"| {name} | {value} |" for name, value in _metric_rows(summary)]
    if measured:
        fractions = [float(r["fraction_root"]) for r in measured]
        lines += [
            "",
            "## Adaptive budget",
            "",
            f"`{sparkline(fractions, 48)}`  ({pct(min(fractions), 1)} \u2013 {pct(max(fractions), 1)})",
        ]
    lines += [
        "",
        "## Per-window results",
        "",
        "| win | items in | sampled | budget | estimate | \u00b1 bound | rel. stderr | saved | e2e |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in measured:
        estimate, bound = si_parts(row.get("approx"), row.get("error_bound"))
        lines.append(
            "| {win} | {n_in} | {n_s} | {f} | {est} | {bound} | {err} | {saved} | {lat} |".format(
                win=int(row["window"]),
                n_in=human_int(row.get("n_in_root")),
                n_s=human_int(row.get("n_sampled_root")),
                f=pct(row.get("fraction_root"), 1),
                est=estimate,
                bound=bound,
                err=pct(row.get("stderr_relative"), 3),
                saved=pct(row.get("bandwidth_saved"), 1),
                lat=latency(row.get("e2e_latency_ms")),
            )
        )
    lines += [
        "",
        "_Generated by `scripts/summarize_docker_results.py`._",
        "",
    ]
    return "\n".join(lines)


def _svg_points(
    values: Sequence[float],
    x0: float,
    y0: float,
    w: float,
    h: float,
    vmin: float,
    vmax: float,
) -> str:
    if not values:
        return ""
    span = max(vmax - vmin, 1e-12)
    step = w / max(len(values) - 1, 1)
    return " ".join(
        f"{x0 + i * step:.1f},{y0 + h - (v - vmin) / span * h:.1f}"
        for i, v in enumerate(values)
    )


def _svg_chart(
    title: str,
    values: Sequence[float],
    unit: str,
    target_value: float | None = None,
    height: int = 150,
) -> str:
    width, left, right, top, bottom = 720, 56, 16, 18, 28
    plot_w, plot_h = width - left - right, height - top - bottom
    if not values:
        return f"<section class=chart><h2>{html.escape(title)}</h2><p>no data</p></section>"
    vmax = max(max(values), target_value or 0.0) * 1.05 or 1.0
    vmin = 0.0
    points = _svg_points(values, left, top, plot_w, plot_h, vmin, vmax)
    target = ""
    if target_value is not None:
        y = top + plot_h - (target_value - vmin) / max(vmax - vmin, 1e-12) * plot_h
        target = (
            f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}' "
            f"class='target'/><text x='{left + plot_w}' y='{y - 5:.1f}' class='note' "
            f"text-anchor='end'>target</text>"
        )
    labels = "".join(
        f"<text x='{left - 8}' y='{top + plot_h * (1.0 - step) + 4:.1f}' class='axis' "
        f"text-anchor='end'>{vmax * step * 100:.1f}%</text>"
        for step in (0.0, 0.5, 1.0)
    )
    return (
        f"<section class=chart><h2>{html.escape(title)}</h2>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' class='axis-line'/>"
        f"{labels}{target}<polyline points='{points}' class='series'/>"
        f"<text x='{left}' y='{height - 6}' class='axis'>window</text>"
        f"<text x='{left + plot_w}' y='{height - 6}' class='axis' text-anchor='end'>{html.escape(unit)}</text>"
        f"</svg></section>"
    )


_CSS = """
:root{color-scheme:light dark}
body{margin:0;padding:32px;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     background:#0f1116;color:#e6e8ee}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:0 0 8px;color:#9aa4b8;font-weight:600}
.wrap{max-width:960px;margin:0 auto}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:700;font-size:13px}
.pass{background:#12351f;color:#4ade80;border:1px solid #1f6b3a}
.fail{background:#3a1417;color:#f87171;border:1px solid #7f1d1d}
.sub{color:#8b93a7;margin:6px 0 22px;font-size:13px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:0 0 22px}
.card{background:#161922;border:1px solid #232838;border-radius:10px;padding:10px 12px}
.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#8b93a7}
.card .v{font-size:18px;font-weight:600;margin-top:2px}
.chart{background:#161922;border:1px solid #232838;border-radius:10px;padding:14px 16px;margin:0 0 14px}
svg{width:100%;height:auto;display:block}
.series{fill:none;stroke:#60a5fa;stroke-width:2;stroke-linejoin:round}
.target{stroke:#f59e0b;stroke-width:1;stroke-dasharray:4 4}
.axis-line{stroke:#2c3244;stroke-width:1}
.axis{fill:#8b93a7;font-size:11px}.note{fill:#f59e0b;font-size:11px}
table{width:100%;border-collapse:collapse;background:#161922;border:1px solid #232838;border-radius:10px}
th,td{padding:6px 9px;border-bottom:1px solid #232838;text-align:right;font-variant-numeric:tabular-nums;font-size:13px}
th{color:#9aa4b8;font-weight:600;text-align:right}
td:first-child,th:first-child{text-align:left}
tr:last-child td{border-bottom:none}
tr.change td{background:#1b2130;color:#93c5fd;font-style:italic;text-align:left}
"""


def html_report(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> str:
    """Self-contained HTML (inline CSS + SVG) for a demo or the write-up."""
    measured = [r for r in rows if float(r.get("n_in_root") or 0.0) > 0.0]
    verdict = bool(summary.get("pass"))
    cards = "".join(
        f"<div class=card><div class=k>{html.escape(name)}</div>"
        f"<div class=v>{html.escape(value)}</div></div>"
        for name, value in _metric_rows(summary)
        if name in {"median rel. stderr", "within target", "bandwidth saved", "median latency"}
    )
    stderr_values = [float(r["stderr_relative"]) for r in measured if is_number(r.get("stderr_relative"))]
    fraction_values = [float(r["fraction_root"]) for r in measured if is_number(r.get("fraction_root"))]
    charts = _svg_chart(
        "Relative standard error per window",
        stderr_values,
        "lower is better",
        target_value=float(summary.get("target_error") or 0.0) or None,
    ) + _svg_chart("Sampling budget per window", fraction_values, "fraction of items forwarded")

    body_rows = []
    previous = None
    for row in measured:
        estimate, bound = si_parts(row.get("approx"), row.get("error_bound"))
        body_rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}</td><td>{}</td><td>{}</td></tr>".format(
                int(row["window"]),
                human_int(row.get("n_in_root")),
                human_int(row.get("n_sampled_root")),
                pct(row.get("fraction_root"), 1),
                estimate,
                bound,
                pct(row.get("stderr_relative"), 3),
                pct(row.get("bandwidth_saved"), 1),
                latency(row.get("e2e_latency_ms")),
            )
        )
        fraction = row.get("fraction_root")
        if is_number(fraction) and is_number(previous) and abs(float(fraction) - float(previous)) > 1e-9:
            body_rows.append(
                f"<tr class=change><td colspan=9>controller: budget "
                f"{pct(previous, 1)} \u2192 {pct(fraction, 1)}</td></tr>"
            )
        if is_number(fraction):
            previous = fraction

    checks = " \u00b7 ".join(
        f"{OK if (summary.get('checks') or {}).get(k) else NO} {label}"
        for k, label in _CHECK_LABELS
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ApproxIoT deployment run</title><style>{_CSS}</style></head>
<body><div class=wrap>
<h1>ApproxIoT deployment run</h1>
<p class=sub><span class="badge {'pass' if verdict else 'fail'}">
{'PASS' if verdict else 'FAIL'}</span> &nbsp;{html.escape(checks)}
 &nbsp;\u00b7&nbsp; {len(measured)} windows &nbsp;\u00b7&nbsp; generated by
<code>scripts/summarize_docker_results.py</code></p>
<div class=cards>{cards}</div>
{charts}
<h2>Per-window results</h2>
<table><thead><tr><th>win</th><th>items in</th><th>sampled</th><th>budget</th>
<th>estimate</th><th>\u00b1 bound</th><th>rel. stderr</th><th>saved</th><th>e2e</th></tr></thead>
<tbody>{''.join(body_rows)}</tbody></table>
</div></body></html>
"""
