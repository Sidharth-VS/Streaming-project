"""Pub/Sub transport abstraction.

The paper models inter-layer communication with Kafka topics (Fig. 5).  This
replication keeps the same topology semantics but makes the broker pluggable:

- :class:`InMemoryTransport` — deterministic, event-scheduled bus used by the
  simulation and all experiments (a message published for window ``t`` is
  delivered to consumers at ``arrival_window``).  This is the default because it
  makes experiments reproducible and runnable without infrastructure.
- ``KafkaTransport`` (``kafka_transport.py``) — same interface backed by
  ``confluent-kafka`` for the containerized deployment in ``docker-compose.yml``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List

from approxiot.items import Item


@dataclass
class WindowMessage:
    """One node's output for one time interval: sample + weight/count metadata.

    Mirrors Algorithm 1 line 13: ``Send(parent, W_out, C_out, sample)``.

    ``item_weights`` optionally carries the *per-item* effective weight (a list
    parallel to ``items``).  The paper communicates weights as per-sub-stream
    sets (``W``/``C``), which is sufficient when a message's items all belong to
    one interval; under asynchronous delivery (Fig. 3) items of different
    batches mix, and per-item weights are what keep the weighted estimators
    unbiased.  When absent, receivers fall back to the ``W`` set.
    """

    sender: str
    sender_window: int
    items: List[Item] = field(default_factory=list)
    W: Dict[str, float] = field(default_factory=dict)
    C: Dict[str, int] = field(default_factory=dict)
    item_weights: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.item_weights and len(self.item_weights) != len(self.items):
            raise ValueError("item_weights must be parallel to items")


@dataclass
class ControlMessage:
    """Controller -> node budget update (novel control plane, plan §Phase 6).

    Carried on a dedicated control topic so budget propagation never mixes with
    the data plane.
    """

    fraction: float
    generation: int
    issued_window: int
    target: float = 0.0
    policy: str = ""


class InMemoryTransport:
    """A deterministic topic bus with window-scheduled delivery.

    Topics are just FIFO queues of ``(arrival_window, message)``.  A consumer
    polling window ``t`` receives every message whose ``arrival_window <= t``,
    in publication order — exactly the "items arrive within an interval"
    semantics of the paper, with delivery delay standing in for network latency.
    """

    def __init__(self) -> None:
        self._topics: Dict[str, Deque[tuple[int, Any]]] = defaultdict(deque)
        self.published = 0
        self.delivered = 0

    def publish(self, topic: str, message: Any, arrival_window: int) -> None:
        self._topics[topic].append((int(arrival_window), message))
        self.published += 1

    def poll(self, topic: str, window: int) -> List[Any]:
        """Drain all messages due by ``window`` (FIFO order preserved)."""
        queue = self._topics.get(topic)
        if not queue:
            return []
        ready: List[Any] = []
        while queue and queue[0][0] <= window:
            ready.append(queue.popleft()[1])
        self.delivered += len(ready)
        return ready

    def pending(self, topic: str) -> int:
        return len(self._topics.get(topic, ()))
