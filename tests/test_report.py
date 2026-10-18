"""Tests for the human-readable deployment reporting (runtime/report.py)."""

from __future__ import annotations

import json

import pytest

from approxiot.runtime import report


def _row(window: int, *, n_in: int = 1000, sampled: int = 500, fraction: float = 0.5,
         stderr: float = 0.004, latency_ms: float = 3800.0, saved: float = 0.5,
         approx: float = 20_000_000.0, bound: float = 250_000.0,
         items_in: int | None = None, items_out: int | None = None) -> dict:
    row = {
        "window": window,
        "n_in_root": n_in,
        "n_sampled_root": sampled,
        "fraction_root": fraction,
        "stderr_relative": stderr,
        "approx": approx,
        "error_bound": bound,
        "e2e_latency_ms": latency_ms,
        "bandwidth_saved": saved,
        "items_in_total": items_in if items_in is not None else 1000 * (window + 1),
        "items_out_total": items_out if items_out is not None else 500 * (window + 1),
    }
    return row


def test_si_pair_shares_one_unit_for_readability():
    assert report.si_pair(20_748_180, 255_743) == "20.75 M \u00b1 0.26 M"
    assert report.si_pair(993_022, 12_400) == "993.02 k \u00b1 12.40 k"
    assert report.si_pair(812.5, 22.3) == "812.50 \u00b1 22.30"
    assert report.si_pair(None) == "\u2014"
    assert report.si_pair(float("nan"), 1.0) == "\u2014"


def test_si_parts_keep_table_columns_in_the_same_unit():
    estimate, bound = report.si_parts(42_300_000, 648_330)
    assert estimate == "42.30 M"
    assert bound == "0.65 M"  # not "648.33 k": the pair shares the estimate's unit
    assert report.si_parts(None, 5) == ("\u2014", "\u2014")


def test_scalar_formatters():
    assert report.human_int(2178) == "2,178"
    assert report.human_int(None) == "\u2014"
    assert report.pct(0.6623, 1) == "66.2%"
    assert report.pct(0.00434, 2) == "0.43%"
    assert report.latency(3800.0) == "3.80 s"
    assert report.latency(435.2) == "435 ms"
    assert report.latency(float("nan")) == "\u2014"
    assert report.duration(105) == "1 m 45 s"


def test_sparkline_is_bounded_and_downsamples():
    line = report.sparkline([0.1, 0.2, 0.9])
    assert len(line) == 3
    assert set(line) <= set(report.SPARKS)
    assert len(report.sparkline(range(100), width=20)) == 20
    assert report.sparkline([]) == ""


def test_summarize_rows_skips_warmup_and_passes_a_good_run():
    rows = [
        _row(0, n_in=0, sampled=0, stderr=0.0, latency_ms=float("nan"), saved=0.0),
        _row(1, n_in=0, sampled=0, stderr=0.0, latency_ms=float("nan"), saved=0.0),
        _row(2),  # budget changes are counted across the whole run, warm-up too
        _row(3),
        _row(4, fraction=0.6),
    ]
    summary = report.summarize_rows(rows, target_error=0.005)
    assert summary["windows_with_results"] == 3
    assert summary["warmup_rows_skipped"] == 2
    assert summary["median_relative_stderr"] == pytest.approx(0.004)
    assert summary["fraction_changes"] == 1
    assert summary["windows_within_target"] == 3
    assert summary["bandwidth_saved_total"] == pytest.approx(0.5)
    assert summary["pass"] is True
    assert all(summary["checks"].values())
    json.dumps(summary)  # must stay serialisable for the recorded artifact


def test_summarize_rows_flags_a_missed_target():
    rows = [_row(0, stderr=0.02), _row(1, stderr=0.03)]
    summary = report.summarize_rows(rows, target_error=0.005)
    assert summary["checks"]["accuracy_target_met"] is False
    assert summary["pass"] is False
    assert summary["max_relative_stderr"] == pytest.approx(0.03)


def test_table_rows_are_aligned_and_carry_the_target_marker():
    header = report.table_header()
    rows = [
        report.window_row(3, 678, 251, 0.3703, 23_178_074, 993_022, 0.0143, 0.6298, 5090.0,
                          target_error=0.005),
        report.window_row(12, 2562, 1697, 0.6623, 20_748_180, 255_743, 0.0040, 0.445, 3.80,
                          target_error=0.005),
    ]
    assert all(len(row) == len(header) for row in rows)
    assert rows[0].endswith("5.09 s")
    assert report.MARK_BAD in rows[0]  # 1.43% is above the 0.50% target
    assert report.MARK_OK in rows[1]


def test_controller_change_line_is_readable():
    line = report.controller_change(12, 0.5, 0.6623, 0.0099, 0.005)
    assert "win 12" in line
    assert "50.0% \u2192 66.2%" in line
    assert "target 0.50%" in line


def test_root_panel_reports_every_check_only_once():
    summary = report.summarize_rows([_row(0), _row(1, fraction=0.6)], 0.005)
    summary["node"] = "root"
    summary["fraction_start"] = 0.5
    panel = report.root_panel(summary, fraction_series=[0.5, 0.6],
                              stderr_series=[0.004, 0.004])
    assert "root summary" in panel
    assert "verdict" in panel and "PASS" in panel
    for label in ("data flowed", "accuracy", "bandwidth", "controller acted"):
        assert label in panel
    assert all(len(line) <= report.WIDTH for line in panel.splitlines())


def test_render_report_marks_budget_changes():
    rows = [_row(0, fraction=0.5), _row(1, fraction=0.5), _row(2, fraction=0.7)]
    summary = report.summarize_rows(rows, 0.005)
    text = report.render_report(rows, summary)
    assert report.table_header() in text
    assert "budget change: 50.0% \u2192 70.0%" in text
    assert "verdict" in text


def test_markdown_and_html_artifacts_are_self_contained():
    rows = [_row(0), _row(1, fraction=0.7)]
    summary = report.summarize_rows(rows, 0.005)

    md = report.markdown_report(rows, summary)
    assert md.startswith("# ApproxIoT deployment run")
    assert "| win | items in | sampled |" in md
    assert "**Verdict: PASS**" in md
    assert md.count("\n| ") >= len(rows)

    html = report.html_report(rows, summary)
    assert html.startswith("<!doctype html>")
    assert "<svg" in html and "polyline" in html
    assert "http://" not in html and "https://" not in html  # no external assets
    assert html.count("<tr>") == 3  # header + one row per measured window
    assert html.count("class=change") == 1  # the budget step is called out


def test_html_escapes_untrusted_workload_text():
    rows = [_row(0)]
    summary = report.summarize_rows(rows, 0.005)
    summary["workload"] = "<script>alert(1)</script>"
    html = report.html_report(rows, summary)
    assert "<script>alert(1)</script>" not in html
