"""Tests for the event-based latency measurement (paper Figs. 9-10)."""

import numpy as np
import pytest

from approxiot.latency import LinkProfile, measure_latency


def _measure(method, window, fraction=0.1, n=10_000, capacity=12_000.0):
    return measure_latency(
        n_source_items=n,
        fraction=fraction,
        method=method,
        window_seconds=window,
        rng=np.random.default_rng(7),
        capacity_items_per_sec=capacity,
        links=LinkProfile(),
    )


def test_approxiot_latency_grows_with_window_size():
    small = _measure("approxiot", 0.5)
    large = _measure("approxiot", 5.0)
    assert small.n_completed > 0 and large.n_completed > 0
    # Roughly 3.5 windows of queueing plus service, so the growth is large.
    assert large.mean_ms > 5 * small.mean_ms


def test_srs_latency_is_flat_across_window_sizes():
    small = _measure("srs", 0.5)
    large = _measure("srs", 5.0)
    assert small.mean_ms == pytest.approx(large.mean_ms, rel=1e-12)
    # No window wait: link RTT + per-hop processing only.
    assert small.mean_ms == pytest.approx(140.0 + 4 / 12_000.0 * 1000.0, rel=1e-9)


def test_sampling_reduces_latency_versus_native():
    approx = _measure("approxiot", 1.0, fraction=0.1)
    native = _measure("native", 1.0)
    assert approx.mean_ms < native.mean_ms
    assert approx.n_completed < native.n_completed


def test_p95_is_at_least_mean_and_empty_windows_are_nan():
    m = _measure("approxiot", 1.0)
    assert m.p95_ms >= m.mean_ms
    empty = measure_latency(0, 0.5, "approxiot", 1.0, np.random.default_rng(0))
    assert np.isnan(empty.mean_ms)
