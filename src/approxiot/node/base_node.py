"""Algorithm 1 — the per-node workflow shared by every topology node.

Per time interval (Algorithm 1, lines 2-21):

1. ``size <- costFunction(budget)``                              (line 3)
2. collect incoming items and the weight/count metadata sets
   ``W_in`` / ``C_in`` from downstream nodes                     (line 6)
3. ``{sample, W_out, C_out} <- WHSamp(items, size, W_in, C_in)`` (line 10)
4. forward to the parent, or run the query + error estimation at the root
   (lines 11-20)

Async support (§III-C): a node keeps the *most recent* ``W_in``/``C_in`` value
per sub-stream; arriving items without fresh metadata are sampled against the
cached values, and the Eq. 9 correction compensates for the count mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping

import numpy as np

from approxiot.items import Item
from approxiot.kafka_io.transport import WindowMessage
from approxiot.sampling.weighted_hierarchical import WHSampResult, whsamp


@dataclass
class NodeConfig:
    name: str
    parent: str | None = None
    #: Initial expected input volume per window (items) — used by the cost
    #: function until the node observes its own first window.
    expected_items_per_window: float = 1_000.0


@dataclass
class NodeBudget:
    """Resource budget: the sampling fraction this node must respect.

    ``costFunction`` maps it to an absolute sample size from the node's
    observed/expected input volume.  A static deployment sets it once; the
    adaptive controller rewrites it at runtime (plan §Phase 6).
    """

    fraction: float = 0.5


@dataclass
class WindowStats:
    """Per-window bookkeeping used by the metrics collector."""

    window: int
    items_in: int = 0
    items_out: int = 0


@dataclass
class BaseNode:
    """Every node samples on a sub-stream basis (paper §III-A)."""

    config: NodeConfig
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    #: Eq. 9 / §III-C: apply asynchronous-interval weight calibration.
    async_correction: bool = True
    #: "approxiot" (Algorithm 2), "srs" (coin flip), or "native" (no sampling).
    method: str = "approxiot"
    #: Paper-verbatim Eq. 9 scalar boost (see weighted_hierarchical docstring);
    #: the per-item effective-weight recursion is the unbiased default.
    eq9_boost: bool = False

    # -- runtime state -----------------------------------------------------
    W_latest: dict[str, float] = field(default_factory=dict)
    C_latest: dict[str, int] = field(default_factory=dict)
    last_input_count: int = 0
    total_in: int = 0
    total_out: int = 0
    fraction: float = 0.5  # current budget (updated by control plane)
    stats: List[WindowStats] = field(default_factory=list)

    # ------------------------------------------------------------------ API
    def cost_function(self, budget: NodeBudget, expected_items: int | None = None) -> int:
        """Map the budget (sampling fraction) to an absolute sample size.

        ``size = fraction * expected_input_volume`` (Algorithm 1 line 3).  The
        node self-tunes to its observed load: before the first window it uses
        the configured expectation, afterwards the previous window's input
        count.  Every active sub-stream is guaranteed at least one slot by
        ``get_sample_size`` even under a tiny budget.
        """
        expected = expected_items if expected_items is not None else max(
            1, self.last_input_count or int(self.config.expected_items_per_window)
        )
        return max(1, int(round(budget.fraction * expected)))

    def collect_input(self, messages: Iterable[WindowMessage]) -> List[Item]:
        """Algorithm 1 line 6: gather items and refresh cached metadata.

        Metadata updates are applied per sub-stream only when a new value
        arrives (§III-C), so items arriving out-of-band keep the last known
        ``W_in`` / ``C_in``.
        """
        items: List[Item] = []
        for msg in messages:
            items.extend(msg.items)
            for sid, w in msg.W.items():
                self.W_latest[sid] = w
            for sid, c in msg.C.items():
                self.C_latest[sid] = c
        return items

    def sample_window(self, items: List[Item], budget: NodeBudget) -> WHSampResult:
        """Algorithm 1 line 10, dispatched by ``method``."""
        size = self.cost_function(budget, expected_items=len(items) or None)
        self.last_input_count = len(items)

        if self.method == "native":
            # No sampling: everything is forwarded with unchanged weights.
            result = WHSampResult()
            sub_streams = sorted({i.substream for i in items})
            for sid in sub_streams:
                sid_items = [i for i in items if i.substream == sid]
                result.sample[sid] = sid_items
                result.W_item[sid] = [
                    i.eff_weight if i.eff_weight > 0 else self.W_latest.get(sid, 1.0)
                    for i in sid_items
                ]
                result.W_out[sid] = max(result.W_item[sid], default=1.0)
                result.C_out[sid] = len(sid_items)
            return result

        if self.method == "srs":
            from approxiot.baselines.srs import SRSSampler

            sampler = SRSSampler(budget.fraction)
            return sampler.process_window(items, W_in=self.W_latest, rng=self.rng)

        return whsamp(
            items,
            sample_size=size,
            W_in=self.W_latest,
            C_in=self.C_latest,
            rng=self.rng,
            async_correction=self.async_correction,
            eq9_boost=self.eq9_boost,
        )

    def process_window(
        self,
        window: int,
        messages: Iterable[WindowMessage],
        budget: NodeBudget,
        extra_items: Iterable[Item] | None = None,
    ) -> WHSampResult:
        """Run one interval of Algorithm 1 (sampling part) and update counters.

        ``extra_items`` lets a source node feed locally generated data into the
        same pipeline (the paper's sources produce *and* sample).
        """
        items = self.collect_input(messages)
        if extra_items is not None:
            items.extend(extra_items)
        result = self.sample_window(items, budget)

        n_out = result.num_sampled
        self.total_in += len(items)
        self.total_out += n_out
        self.stats.append(WindowStats(window=window, items_in=len(items), items_out=n_out))
        return result

    def apply_control(self, fraction: float) -> None:
        """Receive a budget update from the control plane."""
        self.fraction = float(fraction)

    def current_budget(self) -> NodeBudget:
        return NodeBudget(fraction=self.fraction)

    def make_message(self, window: int, result: WHSampResult) -> WindowMessage:
        items = [item for sample in result.sample.values() for item in sample]
        item_weights: list[float] = []
        for sid, sample in result.sample.items():
            item_weights.extend(result.W_item.get(sid, [1.0] * len(sample)))
        return WindowMessage(
            sender=self.config.name,
            sender_window=window,
            items=items,
            W=dict(result.W_out),
            C=dict(result.C_out),
            item_weights=item_weights,
        )
