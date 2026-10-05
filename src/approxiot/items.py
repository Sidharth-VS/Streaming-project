"""The unit that flows through the topology: a data item tagged with its sub-stream.

Following the paper (§III-B), a *sub-stream* is the set of items that originate
from the same source (``substream`` = source id).  Every node stratifies its
input into sub-streams before sampling so that infrequent sub-streams are never
neglected.

``src_window`` records the source time interval the item was generated in; it is
only used by the evaluation harness to pair a root result with the exact
aggregate it approximates.

``eff_weight`` is the item's *effective weight* (the inverse probability of its
entire sampling path, Eq. 5-6).  ``0.0`` means "unset": the item has not been
weighted yet and should inherit the per-sub-stream scalar weight (``W_in``).
Carrying weights per item (rather than only as per-sub-stream metadata) keeps
the weighted estimators unbiased when asynchronous delivery mixes items from
different batches in one interval (paper Fig. 3).
"""

from __future__ import annotations

from typing import NamedTuple


class Item(NamedTuple):
    substream: str
    value: float
    src_window: int = 0
    eff_weight: float = 0.0  # 0.0 = unset -> inherit W_in scalar
