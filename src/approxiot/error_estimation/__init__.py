"""Error estimation module (paper §III-D)."""

from approxiot.error_estimation.error_bounds import (
    Confidence,
    RunningVariance,
    SubStreamErrorStats,
    estimate_variance_mean,
    estimate_variance_sum,
    error_bound,
    sample_variance,
    standard_error_relative,
    summarize_substreams,
)

__all__ = [
    "Confidence",
    "RunningVariance",
    "SubStreamErrorStats",
    "estimate_variance_mean",
    "estimate_variance_sum",
    "error_bound",
    "sample_variance",
    "standard_error_relative",
    "summarize_substreams",
]
