"""Variance estimation and error bounds for linear queries (paper §III-D).

Implements exactly the equations referenced in the plan:

- Eq. 10: ``Var(SUM*) = sum_i Var(SUM_i)`` (sub-streams sampled independently).
- Eq. 11: ``Var(SUM_i) = c_src_i * (c_src_i - Y_i) * s_i^2 / Y_i``.
- Eq. 12: ``s_i^2 = 1/(Y_i - 1) * sum_k (I_k - mean(I))^2`` (sample variance).
- Eq. 13: ``MEAN* = sum_i phi_i * MEAN_i`` with ``phi_i = c_src_i / sum_k c_src_k``.
- Eq. 14: ``Var(MEAN*) = sum_i phi_i^2 * (s_i^2 / Y_i) * ((c_src_i - Y_i) / c_src_i)``.
- Error bound: sqrt(Var) scaled by 1/2/3 for 68% / 95% / 99.7% confidence.

``c_src_i`` (the sub-stream's true arrival count at its source) is recovered
from the sampling metadata as ``c_src_i = Y_i * W_i^out``, exactly as the paper
suggests ("We can easily compute c_src by Y_ij * W_ij^out").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class Confidence(Enum):
    """The '68-95-99.7' rule (paper §III-D)."""

    P68 = 1  # one standard deviation
    P95 = 2
    P997 = 3


@dataclass(frozen=True)
class SubStreamErrorStats:
    """Per-sub-stream inputs to Eq. 11 / Eq. 14, measured at the root node.

    Attributes:
        Y: number of sampled items for this sub-stream at the root (``Y_i,j``).
        sample_values: the sampled item values ``I_i,j,k`` (used for Eq. 12).
        c_src: number of items the sub-stream produced at its source
            (``c_src_i``), recovered as ``Y * W_out``.
    """

    substream: str
    Y: int
    sample_values: tuple[float, ...]
    c_src: float
    #: Optional pre-computed Eq. 12 variance (e.g. a running estimate); when
    #: None, the window-local sample variance of ``sample_values`` is used.
    s2: float | None = None

    @classmethod
    def from_sample(
        cls,
        substream: str,
        sample_values: Sequence[float],
        W_out: float,
        s2: float | None = None,
        item_weights: Sequence[float] | None = None,
    ) -> "SubStreamErrorStats":
        """Build stats from a root sample plus its effective weight.

        ``c_src = Y * W_out`` (paper §III-D); if ``W_out`` degenerates below 1
        (never happens by construction) the count is still recoverable.
        """
        Y = len(sample_values)
        # With per-item effective weights, the Horvitz-Thompson-consistent
        # estimate of the sub-stream's arrival count uses the weight mean.
        if item_weights:
            w_mean = sum(float(w) for w in item_weights) / len(item_weights)
        else:
            w_mean = float(W_out)
        return cls(
            substream=substream,
            Y=Y,
            sample_values=tuple(float(v) for v in sample_values),
            c_src=Y * w_mean,
            s2=s2,
        )


def sample_variance(values: Sequence[float]) -> float:
    """Eq. 12: unbiased sample variance ``s^2`` with the ``Y - 1`` normalizer.

    Returns 0.0 for fewer than two samples (no spread information).
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)


class RunningVariance:
    """Welford online accumulator for Eq. 12 (``s^2``), Apache-Commons-Math style.

    The paper implements the error-estimation module with Apache Commons Math
    (§IV-B-III), whose ``Variance`` statistic is a running accumulator.  This
    matters at replication scale: when a window leaves the root only one or two
    sampled items per sub-stream, the window-local Eq. 12 degenerates to zero
    and the bound collapses.  Because each sub-stream's values are iid draws
    from a stationary distribution, accumulating ``s_i^2`` across windows is
    the statistically sound online estimator.
    """

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, values: Sequence[float]) -> None:
        for x in values:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            self._m2 += delta * (x - self.mean)

    @property
    def variance(self) -> float:
        """Unbiased sample variance (0 while fewer than two observations)."""
        if self.n < 2:
            return 0.0
        return self._m2 / (self.n - 1)

    @property
    def count(self) -> int:
        return self.n


def _squared_std(stats: SubStreamErrorStats) -> float:
    """Eq. 12, honoring a caller-supplied (e.g. running) variance estimate."""
    if stats.s2 is not None:
        return stats.s2
    return sample_variance(stats.sample_values)


def _variance_sum_per_substream(stats: SubStreamErrorStats) -> float:
    """Eq. 11 for a single sub-stream."""
    if stats.Y <= 0:
        return 0.0
    if stats.Y >= stats.c_src:
        return 0.0  # census of this sub-stream: no sampling error
    s2 = _squared_std(stats)  # Eq. 12
    return stats.c_src * (stats.c_src - stats.Y) * s2 / stats.Y


def estimate_variance_sum(stats: Sequence[SubStreamErrorStats]) -> float:
    """Eq. 10 + Eq. 11: variance of the approximate total sum."""
    return sum(_variance_sum_per_substream(s) for s in stats)


def estimate_variance_mean(stats: Sequence[SubStreamErrorStats]) -> float:
    """Eq. 13 + Eq. 14: variance of the approximate mean over all sub-streams."""
    total_c_src = sum(s.c_src for s in stats)
    if total_c_src <= 0:
        return 0.0
    var = 0.0
    for s in stats:
        phi_i = s.c_src / total_c_src  # Eq. 13
        if s.Y <= 0 or s.Y >= s.c_src:
            continue  # census: no sampling error contribution
        s2 = _squared_std(s)  # Eq. 12
        var += (phi_i ** 2) * (s2 / s.Y) * ((s.c_src - s.Y) / s.c_src)
    return var


def error_bound(variance: float, confidence: Confidence = Confidence.P997) -> float:
    """Absolute error bound: ``z * sqrt(variance)`` per the 68-95-99.7 rule."""
    return confidence.value * math.sqrt(max(variance, 0.0))


def standard_error_relative(variance: float, result: float) -> float:
    """Relative standard error ``sqrt(Var) / |result|`` (controller signal).

    This is the online, ground-truth-free proxy for the realized accuracy loss
    used by the adaptive budget controller.
    """
    if result == 0:
        return math.inf if variance > 0 else 0.0
    return math.sqrt(max(variance, 0.0)) / abs(result)


def summarize_substreams(
    sample: Mapping[str, Sequence],
    W_out: Mapping[str, float],
    s2_by_substream: Mapping[str, float] | None = None,
) -> list[SubStreamErrorStats]:
    """Build :class:`SubStreamErrorStats` from a root sample.

    Accepts samples of either plain floats or :class:`~approxiot.items.Item`
    (whose per-item ``eff_weight`` is used when present).
    """
    s2_by_substream = s2_by_substream or {}
    stats: list[SubStreamErrorStats] = []
    for sid, items in sample.items():
        if items and hasattr(items[0], "eff_weight"):
            item_weights = [float(getattr(it, "eff_weight")) for it in items]
            values = [float(it.value) for it in items]
            w_scalar = W_out.get(sid, 1.0)
        else:
            item_weights = None
            values = [float(v) for v in items]
            w_scalar = W_out.get(sid, 1.0)
        stats.append(
            SubStreamErrorStats.from_sample(
                sid,
                values,
                w_scalar,
                s2_by_substream.get(sid),
                item_weights,
            )
        )
    return stats
