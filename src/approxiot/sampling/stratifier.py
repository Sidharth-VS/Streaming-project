"""Stratification of an input stream into per-source sub-streams (paper §III-B).

Algorithm 2 line 5: ``S <- Update(items)``.  The node groups the items received
within one time interval by their source, which lets the sampler allocate
reservoir budget per sub-stream instead of over the pooled stream.  This is the
property that prevents low-rate sub-streams from being overlooked.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from approxiot.items import Item


def stratify(items: Iterable[Item]) -> dict[str, list[Item]]:
    """Group ``items`` into sub-streams, preserving arrival order per group."""
    sub_streams: "OrderedDict[str, list[Item]]" = OrderedDict()
    for item in items:
        sub_streams.setdefault(item.substream, []).append(item)
    return dict(sub_streams)


def sub_stream_counts(sub_streams: dict[str, list[Item]]) -> dict[str, int]:
    """Return ``c_i = |S_i|`` for every sub-stream (Algorithm 2 line 9)."""
    return {sid: len(stream) for sid, stream in sub_streams.items()}
