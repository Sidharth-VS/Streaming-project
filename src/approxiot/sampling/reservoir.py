"""Reservoir sampling (paper §II-B, Vitter's Algorithm R).

Keeps the first ``R`` items, then for the i-th arriving item (i > R) keeps it
with probability ``R / i``, randomly evicting one resident.  Every item of the
(unbounded) stream therefore ends up in the reservoir with equal probability,
which is what makes the per-sub-stream weight ``c_i / N_i`` (Eq. 1) valid.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

import numpy as np

T = TypeVar("T")


def reservoir_sample(
    items: Sequence[T],
    size: int,
    rng: np.random.Generator | None = None,
) -> list[T]:
    """Draw a uniform random sample of at most ``size`` items.

    Args:
        items: the sub-stream items, in arrival order.
        size: reservoir size ``N`` (sample-size budget for this sub-stream).
        rng: numpy random generator (injectable for reproducibility).

    Returns:
        A list with ``min(len(items), size)`` sampled items.

    Raises:
        ValueError: if ``size`` is negative.
    """
    if size < 0:
        raise ValueError("reservoir size must be >= 0")
    if size == 0 or not items:
        return []
    if rng is None:
        rng = np.random.default_rng()

    n = len(items)
    if n <= size:
        return list(items)

    reservoir = list(items[:size])
    # Algorithm R: item index i (0-based) is kept with probability size / (i + 1).
    for i in range(size, n):
        j = int(rng.integers(0, i + 1))
        if j < size:
            reservoir[j] = items[i]
    return reservoir
