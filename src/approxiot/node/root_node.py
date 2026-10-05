"""Root node: Algorithm 1 lines 15-20 (query + error estimation)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from approxiot.error_estimation.error_bounds import (
    Confidence,
    RunningVariance,
    error_bound,
    estimate_variance_mean,
    estimate_variance_sum,
    standard_error_relative,
    summarize_substreams,
)
from approxiot.items import Item
from approxiot.node.base_node import BaseNode, NodeBudget
from approxiot.queries import approximate_mean, approximate_sum


@dataclass
class RootResult:
    """One window's approximate output: ``result +/- error`` (Algorithm 1 line 20)."""

    window: int
    query: str
    approx: float
    error: float
    variance: float
    stderr_relative: float
    n_in: int
    n_sampled: int
    #: Dominant source window among the input items; pairs the result with the
    #: exact aggregate it estimates (see simulation.py).
    src_window: int = -1
    confidence: int = 3


class RootNode(BaseNode):
    def __init__(self, *args, query: str = "sum", confidence: Confidence = Confidence.P997, **kwargs):
        super().__init__(*args, **kwargs)
        if query not in {"sum", "mean"}:
            raise ValueError("query must be 'sum' or 'mean'")
        self.query = query
        self.confidence = confidence
        # Running Eq. 12 accumulators (Apache-Commons-Math style, paper §IV-B):
        # per sub-stream s_i^2, plus a global fallback for sub-streams whose
        # own sample has fewer than two observations so far.
        self._running_var: dict[str, RunningVariance] = {}
        self._global_var = RunningVariance()

    def execute(
        self,
        window: int,
        messages: list,
        budget: NodeBudget,
    ) -> RootResult:
        """Sample the arriving data, run the query, and estimate the error bound."""
        # Algorithm 1 line 10 happens at the root too (§IV-B: the root "also
        # makes use of the sampling module to take a sample of the input").
        result = self.process_window(window, messages, budget)

        # Update the running Eq. 12 statistics with this window's samples.
        s2_by_substream: dict[str, float] = {}
        for sid, items in result.sample.items():
            values = [float(i.value) for i in items]
            rv = self._running_var.setdefault(sid, RunningVariance())
            rv.update(values)
            self._global_var.update(values)
            s2_by_substream[sid] = (
                rv.variance if rv.count >= 2 else self._global_var.variance
            )
        stats = summarize_substreams(result.sample, result.W_out, s2_by_substream)

        if self.query == "sum":
            approx = approximate_sum(result.sample, result.W_out)
            variance = estimate_variance_sum(stats)
        else:
            approx = approximate_mean(result.sample, result.W_out)
            variance = estimate_variance_mean(stats)

        # Dominant source window of this window's input items.
        src_windows = [i.src_window for msg in messages for i in msg.items]
        src_window = max(set(src_windows), key=src_windows.count) if src_windows else -1

        n_in = len(src_windows)
        n_sampled = result.num_sampled
        bound = error_bound(variance, self.confidence)

        return RootResult(
            window=window,
            query=self.query,
            approx=approx,
            error=bound,
            variance=variance,
            stderr_relative=standard_error_relative(variance, approx),
            n_in=n_in,
            n_sampled=n_sampled,
            src_window=src_window,
            confidence=self.confidence.value,
        )
