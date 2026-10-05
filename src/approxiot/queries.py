"""Linear queries over weighted samples (paper §III-B, Eq. 2-3 and Eq. 13).

Every estimator consumes the *per-item* effective weight when the sample
carries one (``Item.eff_weight``), falling back to the per-sub-stream scalar
``W_out`` otherwise — the two coincide whenever all items of a sub-stream share
one weight (the synchronized case the paper's Eq. 2 assumes).
"""

from __future__ import annotations

from typing import Mapping

from approxiot.items import Item


def _weight(item: Item, W_out: Mapping[str, float]) -> float:
    if item.eff_weight > 0:
        return item.eff_weight
    return W_out.get(item.substream, 1.0)


def approximate_sum(
    sample: Mapping[str, list[Item]],
    W_out: Mapping[str, float],
) -> float:
    """Eq. 2 + Eq. 3: ``SUM* = sum_i (sum_k I_i,k) * W_i^out`` (per-item form)."""
    total = 0.0
    for items in sample.values():
        total += sum(item.value * _weight(item, W_out) for item in items)
    return total


def approximate_mean(
    sample: Mapping[str, list[Item]],
    W_out: Mapping[str, float],
) -> float:
    """Eq. 13: ``MEAN* = sum_i phi_i * MEAN_i`` with ``phi_i = c_src_i / sum c_src``.

    ``c_src_i = Y_i * mean(W_i^out)`` is recovered from the metadata (with
    per-item weights the mean is the Horvitz-Thompson-consistent choice); the
    mean of sub-stream ``i`` is the plain average of its sampled values.
    """
    weighted = 0.0
    total_c = 0.0
    for sid, items in sample.items():
        if not items:
            continue
        y_i = len(items)
        weights = [item.eff_weight for item in items]
        if all(w > 0 for w in weights):
            w_mean = sum(weights) / y_i
        else:
            w_mean = W_out.get(sid, 1.0)
        c_src = y_i * w_mean
        mean_i = sum(item.value for item in items) / y_i
        weighted += c_src * mean_i
        total_c += c_src
    if total_c <= 0:
        return 0.0
    return weighted / total_c
