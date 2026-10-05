"""Sampling primitives: reservoir sampling, stratification, weighted hierarchical sampling."""

from approxiot.sampling.reservoir import reservoir_sample
from approxiot.sampling.stratifier import stratify, sub_stream_counts
from approxiot.sampling.weighted_hierarchical import (
    WHSampResult,
    get_sample_size,
    whsamp,
)

__all__ = [
    "reservoir_sample",
    "stratify",
    "sub_stream_counts",
    "WHSampResult",
    "get_sample_size",
    "whsamp",
]
