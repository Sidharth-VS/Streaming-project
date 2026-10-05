"""Weighted hierarchical sampling (paper Algorithm 2) with Eq. 1 and Eq. 9.

This module implements the crux of ApproxIoT.  For every time interval a node:

1. stratifies its input into sub-streams ``S_i`` (``stratifier.py``),
2. splits the resource budget ``sampleSize`` across sub-streams
   (``getSampleSize``, Algorithm 2 line 7),
3. reservoir-samples each sub-stream (Algorithm 2 line 10),
4. computes the effective weight ``W_i^out`` (Eq. 1, and Eq. 9 when time
   intervals are not synchronized across nodes).

Effective weights are carried **per item** (``Item.eff_weight``).  For a
sub-stream whose every sampled item carries the same weight this is identical
to the paper's per-sub-stream sets (``W_i^out = W_i^in * c_i/N_i``, Eq. 1), and
the weighted sum estimator is exactly Eq. 2.  The per-item form is a strict
generalization: when asynchronous delivery mixes items of different batches in
one interval (paper Fig. 3), items with different sampling histories appear in
the same window and only per-item weights remain unbiased.

The paper-verbatim Eq. 9 calibration (``W^out = W^in * w_i * C^in/c_i``) is
available via ``eq9_boost=True``.  It is correct under the paper's model where a
node's interval receives a fraction ``alpha`` of *one* batch; under item-level
straddling it double-counts batches whose pieces arrive across several
intervals, which is why the per-item recursion is the default here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from approxiot.items import Item
from approxiot.sampling.reservoir import reservoir_sample
from approxiot.sampling.stratifier import stratify, sub_stream_counts

#: Minimum reservoir size granted to any non-empty sub-stream.  Proportional
#: allocation can round a small sub-stream down to zero; the paper's whole point
#: is to *never neglect* a sub-stream (Algorithm 2 line 22 keeps every S_i in
#: the returned sample), so we guarantee one slot per active sub-stream.
MIN_RESERVOIR_PER_SUBSTREAM = 1


@dataclass
class WHSampResult:
    """Output of one ``whsamp`` invocation (Algorithm 2 line 22)."""

    sample: dict[str, list[Item]] = field(default_factory=dict)
    #: ``W^out``: scalar effective weight per sub-stream.  With homogeneous
    #: per-item weights this equals the paper's W_i^out exactly; when a window
    #: mixes items with different histories it is the *max* item weight (the
    #: paper's max convention, Eq. 5) and ``W_item`` carries the exact values.
    W_out: dict[str, float] = field(default_factory=dict)
    #: ``C^out``: forwarded sample count per sub-stream.
    C_out: dict[str, int] = field(default_factory=dict)
    #: ``N_i``: reservoir size granted per sub-stream (diagnostics / controller).
    N_allocated: dict[str, int] = field(default_factory=dict)
    #: Per-item effective weights parallel to ``sample[sid]``.
    W_item: dict[str, list[float]] = field(default_factory=dict)

    @property
    def num_sampled(self) -> int:
        return sum(len(v) for v in self.sample.values())


def get_sample_size(
    sample_size: int,
    counts: Mapping[str, int],
) -> dict[str, int]:
    """Allocate ``sample_size`` reservoir slots across sub-streams (Algorithm 2 line 7).

    Policy: **proportional allocation** — the classic stratified-sampling choice
    (also Neyman-optimal when within-stratum variances are equal, which holds
    for the paper's homogeneous per-type sub-streams):

        ``N_i = sample_size * c_i / sum_j(c_j)``

    with largest-remainder rounding so that ``sum(N_i) <= max(sample_size, X)``.
    Every non-empty sub-stream receives at least one slot
    (``MIN_RESERVOIR_PER_SUBSTREAM``); if the budget is smaller than the number
    of sub-streams the effective budget becomes ``X`` — a documented, deliberate
    deviation that preserves the paper's "do not neglect any sub-stream" goal.
    """
    active = {sid: c for sid, c in counts.items() if c > 0}
    if not active:
        return {}

    total_items = sum(active.values())
    budget = max(sample_size, 0)

    # Proportional share with a floor of one slot per active sub-stream.
    quotas = {sid: budget * c / total_items for sid, c in active.items()}
    allocation = {sid: max(MIN_RESERVOIR_PER_SUBSTREAM, int(np.floor(q))) for sid, q in quotas.items()}

    remaining = budget - sum(allocation.values())
    if remaining > 0:
        # Largest-remainder: hand out leftover slots to the most under-served
        # sub-streams (largest fractional part first).
        order = sorted(active, key=lambda sid: (quotas[sid] - np.floor(quotas[sid]), active[sid]), reverse=True)
        idx = 0
        while remaining > 0:
            allocation[order[idx % len(order)]] += 1
            remaining -= 1
            idx += 1
    elif remaining < 0:
        # Budget below the number of sub-streams: the min-1 guarantee wins.
        # (Only reachable when sample_size < number of active sub-streams.)
        pass

    return allocation


def _incoming_weight(item: Item, sid: str, W_in: Mapping[str, float]) -> float:
    """Resolve the item's incoming effective weight.

    Items with ``eff_weight == 0`` (unset) inherit the per-sub-stream scalar
    ``W_in`` (default 1.0 at a source); otherwise the carried per-item weight is
    authoritative.
    """
    if item.eff_weight > 0:
        return item.eff_weight
    return float(W_in.get(sid, 1.0))


def whsamp(
    items: Iterable[Item],
    sample_size: int,
    W_in: Mapping[str, float] | None = None,
    C_in: Mapping[str, int] | None = None,
    rng: np.random.Generator | None = None,
    async_correction: bool = True,
    eq9_boost: bool = False,
) -> WHSampResult:
    """Run weighted hierarchical sampling over one interval's ``items``.

    Args:
        items: all items received in this time interval (all sub-streams mixed).
        sample_size: resource budget in items (``costFunction(budget)`` upstream).
        W_in: incoming scalar effective weights per sub-stream (default ``1.0``).
            Used for items whose ``eff_weight`` is unset.
        C_in: downstream forwarded counts per sub-stream; used only by the
            paper-verbatim Eq. 9 calibration (``eq9_boost=True``).
        rng: injectable RNG for reproducibility.
        async_correction: kept for API compatibility; the per-item effective
            weight recursion (active by default) supersedes the scalar
            asynchronous handling of §III-C.
        eq9_boost: apply the paper-verbatim Eq. 9 factor
            ``W^out = W^in * w_i * C^in/c_i`` on top of Eq. 1.

    Returns:
        :class:`WHSampResult` with the sample and the outgoing metadata.
    """
    if rng is None:
        rng = np.random.default_rng()
    W_in = dict(W_in or {})
    C_in = dict(C_in or {})

    sub_streams = stratify(items)
    counts = sub_stream_counts(sub_streams)
    allocation = get_sample_size(sample_size, counts)

    result = WHSampResult(N_allocated=dict(allocation))

    for sid, stream in sub_streams.items():
        c_i = counts[sid]  # c_i, Algorithm 2 line 9
        N_i = allocation[sid]  # N_i, Algorithm 2 line 7

        if c_i > N_i:
            # Eq. 1: w_i = c_i / N_i
            w_i = c_i / N_i
            if eq9_boost:
                # Eq. 9: W^out = W^in * w_i * C^in/c_i  (C^in defaults to c_i
                # when no metadata has arrived yet -> graceful sync fallback).
                C_i_in = float(C_in.get(sid, c_i))
                w_i *= C_i_in / c_i
            sample_i = reservoir_sample(stream, N_i, rng=rng)  # line 10
            weighted: list[Item] = []
            for item in sample_i:
                in_w = _incoming_weight(item, sid, W_in)
                weighted.append(item._replace(eff_weight=in_w * w_i))
            result.sample[sid] = weighted
            result.W_item[sid] = [it.eff_weight for it in weighted]
            # Scalar metadata: paper's max convention (Eq. 5).
            result.W_out[sid] = max(it.eff_weight for it in weighted)
            result.C_out[sid] = N_i  # line 15
        else:
            # c_i <= N_i: every item is kept with unchanged weight (line 18).
            weighted = []
            for item in stream:
                in_w = _incoming_weight(item, sid, W_in)
                weighted.append(item._replace(eff_weight=in_w))
            result.sample[sid] = weighted
            result.W_item[sid] = [it.eff_weight for it in weighted]
            result.W_out[sid] = max(it.eff_weight for it in weighted) if weighted else W_in.get(sid, 1.0)
            result.C_out[sid] = c_i  # line 19

        assert len(result.sample[sid]) == min(c_i, N_i)  # reservoir guarantee, Yi <= Ni

    return result
