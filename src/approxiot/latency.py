"""Event-based end-to-end latency measurement (paper §V-B/C, Figs. 9-10).

The paper measures latency on a 25-machine cluster; a pure-Python run cannot
observe WAN latency.  Instead of plugging the closed-form
``3 * window + link latency`` into the results (which makes every curve
trivially linear and cannot distinguish the sampling strategies), this module
runs a discrete-event trace of one source window's items through the same
pipeline:

1. item created at a uniformly random time inside the source window;
2. source waits for its window boundary, processes a batch at
   ``capacity_items_per_sec``, forwards it over the 20 ms link;
3. L1/L2 wait for their next window boundary (ApproxIoT/native), process,
   forward over 40/80 ms links;
4. the root processes the arriving batch and emits the query result.

**SRS is windowless**, exactly as the paper describes (§V-C: "the SRS-based
system does not require a window for sampling the input streams"): items are
sampled and forwarded immediately on arrival, so their latency is independent
of the window size.  This is what makes the two curves in Fig. 10 differ.

Latency is the per-item difference between creation and root completion,
measured from the event timestamps (mean and p95 over the window's items).
It is still a *simulated* measurement — no physical cluster is involved — but
it is not the old closed-form formula and it reacts to load, sampling fraction
and window size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinkProfile:
    """Paper §V-A WAN: 20/40/80 ms between adjacent layers."""

    source_to_l1_ms: float = 20.0
    l1_to_l2_ms: float = 40.0
    l2_to_root_ms: float = 80.0


@dataclass(frozen=True)
class LatencyMeasurement:
    mean_ms: float
    p95_ms: float
    n_completed: int
    method: str


def measure_latency(
    n_source_items: int,
    fraction: float,
    method: str,
    window_seconds: float,
    rng: np.random.Generator,
    capacity_items_per_sec: float = 12_000.0,
    links: LinkProfile = LinkProfile(),
) -> LatencyMeasurement:
    """Trace one source window's items and return measured latency statistics.

    ``n_source_items`` is the number of items the sources processed in the
    window being estimated (the exact count, used by the evaluation harness).
    """
    if n_source_items <= 0 or window_seconds <= 0 or capacity_items_per_sec <= 0:
        return LatencyMeasurement(float("nan"), float("nan"), 0, method)

    link_total_s = (
        links.source_to_l1_ms + links.l1_to_l2_ms + links.l2_to_root_ms
    ) / 1000.0

    if method == "srs":
        # Windowless: each item is forwarded as soon as it arrives, with no
        # window wait.  Sampling still happens independently at each hop.
        n = n_source_items
        for _ in range(3):
            n = int(rng.binomial(n, fraction))
        if n <= 0:
            return LatencyMeasurement(float("nan"), float("nan"), 0, method)
        # Four processing stages (source, L1, L2, root), one item each.
        latency_s = link_total_s + 4.0 / capacity_items_per_sec
        values = np.full(n, latency_s * 1000.0)
        return LatencyMeasurement(
            float(np.mean(values)), float(np.percentile(values, 95)), n, method
        )

    # Windowed execution (ApproxIoT and native).  A node can only forward at
    # the end of its interval; the batch then takes n_out / capacity seconds.
    if method == "native":
        n1 = n2 = n3 = n4 = n_source_items
    else:
        n1 = int(rng.binomial(n_source_items, fraction))
        n2 = int(rng.binomial(n1, fraction))
        n3 = int(rng.binomial(n2, fraction))
        n4 = int(rng.binomial(n3, fraction))
    if n4 <= 0:
        return LatencyMeasurement(float("nan"), float("nan"), 0, method)

    t = window_seconds + n1 / capacity_items_per_sec           # source window end
    t = max(t + links.source_to_l1_ms / 1000.0, 2 * window_seconds) + n2 / capacity_items_per_sec
    t = max(t + links.l1_to_l2_ms / 1000.0, 3 * window_seconds) + n3 / capacity_items_per_sec
    t = max(t + links.l2_to_root_ms / 1000.0, 4 * window_seconds) + n4 / capacity_items_per_sec

    created = rng.uniform(0.0, window_seconds, size=n4)
    values = (t - created) * 1000.0
    return LatencyMeasurement(
        float(np.mean(values)), float(np.percentile(values, 95)), n4, method
    )
