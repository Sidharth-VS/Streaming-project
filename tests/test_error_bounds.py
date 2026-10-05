import math

import numpy as np

from approxiot.error_estimation.error_bounds import (
    Confidence,
    SubStreamErrorStats,
    error_bound,
    estimate_variance_mean,
    estimate_variance_sum,
    sample_variance,
    standard_error_relative,
    summarize_substreams,
)


def test_sample_variance_eq12():
    # Hand-computed: mean=5, squared deviations 9+1+1+9=20, / (4-1) = 20/3.
    assert sample_variance([2.0, 4.0, 6.0, 8.0]) == 20.0 / 3.0


def test_sample_variance_edge_cases():
    assert sample_variance([3.0]) == 0.0
    assert sample_variance([]) == 0.0


def test_variance_of_sum_eq11_hand_computed():
    """Eq. 11: c_src(c_src - Y) s^2 / Y with c_src=8, Y=4, s^2=20/3 -> 160/3."""
    stats = SubStreamErrorStats(substream="A1", Y=4, sample_values=(2, 4, 6, 8), c_src=8.0)
    assert estimate_variance_sum([stats]) == 8.0 * 4.0 * (20.0 / 3.0) / 4.0


def test_census_has_zero_variance():
    """Sampling everything (Y = c_src) must give zero variance."""
    stats = SubStreamErrorStats(substream="A1", Y=4, sample_values=(2, 4, 6, 8), c_src=4.0)
    assert estimate_variance_sum([stats]) == 0.0


def test_variance_of_sum_eq10_sums_over_substreams():
    s1 = SubStreamErrorStats("A1", 4, (2, 4, 6, 8), 8.0)
    s2 = SubStreamErrorStats("B1", 5, (1, 2, 3, 4, 5), 10.0)
    assert estimate_variance_sum([s1, s2]) == estimate_variance_sum([s1]) + estimate_variance_sum([s2])


def test_variance_of_mean_eq14_hand_computed():
    """Eq. 14 for two sub-streams, values chosen to be hand-checkable.

    S1: Y=4, s^2=20/3, c_src=8;  S2: Y=5, s^2=2.5, c_src=10.
    phi_1 = 8/18, phi_2 = 10/18.
    Var = phi_1^2 * (20/3)/4 * (4/8) + phi_2^2 * (2.5/5) * (5/10)
        = (64/324)(5/6) + (100/324)(1/4) = 160/972 + 25/324.
    """
    s1 = SubStreamErrorStats("A1", 4, (2, 4, 6, 8), 8.0)
    s2 = SubStreamErrorStats("B1", 5, (1, 2, 3, 4, 5), 10.0)
    expected = 160.0 / 972.0 + 25.0 / 324.0
    assert math.isclose(estimate_variance_mean([s1, s2]), expected, rel_tol=1e-12)


def test_mean_variance_consistent_with_sum_variance():
    """Eq. 14 must equal Eq. 11 divided by (sum of c_src)^2."""
    s1 = SubStreamErrorStats("A1", 4, (2, 4, 6, 8), 8.0)
    s2 = SubStreamErrorStats("B1", 5, (1, 2, 3, 4, 5), 10.0)
    total_c = 18.0
    expected = (estimate_variance_sum([s1]) + estimate_variance_sum([s2])) / total_c**2
    assert math.isclose(estimate_variance_mean([s1, s2]), expected, rel_tol=1e-12)


def test_error_bound_68_95_997_rule():
    variance = 4.0  # std = 2
    assert error_bound(variance, Confidence.P68) == 2.0
    assert error_bound(variance, Confidence.P95) == 4.0
    assert error_bound(variance, Confidence.P997) == 6.0


def test_standard_error_relative():
    assert standard_error_relative(4.0, 200.0) == 0.01
    assert standard_error_relative(0.0, 10.0) == 0.0
    assert math.isinf(standard_error_relative(1.0, 0.0))


def test_summarize_substreams_recovers_csrc_via_weights():
    """c_src = Y * W_out (paper §III-D)."""
    sample = {"A1": [1.0, 2.0], "B1": [3.0]}
    stats = summarize_substreams(sample, {"A1": 2.5, "B1": 10.0})
    by_id = {s.substream: s for s in stats}
    assert by_id["A1"].c_src == 2 * 2.5
    assert by_id["B1"].c_src == 1 * 10.0
