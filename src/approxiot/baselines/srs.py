"""Simple Random Sampling baseline (paper §IV-B-II, "coin flip sampling").

At every node, each arriving item is kept independently with probability ``f``
(the same sampling fraction ApproxIoT is given).  The unbiased hierarchical
estimator attaches an inverse-probability weight to every kept item:

    ``W_out = W_in * (1 / f)``      (per layer)

so that a value passing through ``L`` layers contributes with expected weight
``f^L * (1/f)^L = 1``.  Unlike ApproxIoT, SRS never stratifies, so low-rate
sub-streams can vanish from the sample entirely — the failure mode the paper
demonstrates in §V-E.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from approxiot.items import Item
from approxiot.sampling.weighted_hierarchical import WHSampResult


def coin_flip_sample(
    items: Iterable[Item],
    fraction: float,
    rng: np.random.Generator,
) -> list[Item]:
    """Keep each item independently with probability ``fraction``."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be within [0, 1]")
    if fraction >= 1.0:
        return list(items)
    if fraction <= 0.0:
        return []
    kept = [item for item in items if bool(rng.random() < fraction)]
    return kept


class SRSSampler:
    """Stateful per-node SRS processor with the same I/O contract as ``whsamp``."""

    def __init__(self, fraction: float):
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        self.fraction = float(fraction)

    def process_window(
        self,
        items: Iterable[Item],
        W_in: Mapping[str, float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> WHSampResult:
        """Coin-flip this window's items and produce outgoing metadata.

        Returns a :class:`WHSampResult` so the root node can treat ApproxIoT
        and SRS samples identically when running queries.
        """
        if rng is None:
            rng = np.random.default_rng()
        W_in = dict(W_in or {})

        items = list(items)
        kept = coin_flip_sample(items, self.fraction, rng)

        result = WHSampResult()
        weight_multiplier = 1.0 / self.fraction
        for item in kept:
            in_w = item.eff_weight if item.eff_weight > 0 else W_in.get(item.substream, 1.0)
            result.sample.setdefault(item.substream, []).append(
                item._replace(eff_weight=in_w * weight_multiplier)
            )

        sub_streams = {item.substream for item in items}
        for sid in sub_streams:
            # Inverse-probability weighting; identity when fraction == 1.
            result.W_out[sid] = float(W_in.get(sid, 1.0)) * weight_multiplier
            result.C_out[sid] = len(result.sample.get(sid, []))
            result.W_item[sid] = [it.eff_weight for it in result.sample.get(sid, [])]
        return result
