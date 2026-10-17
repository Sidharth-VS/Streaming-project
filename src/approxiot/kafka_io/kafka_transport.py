"""Optional Kafka-backed transport for the containerized deployment.

Requires the ``kafka`` extra (``pip install -e ".[kafka]"``) and a running
broker (see ``docker-compose.yml``: Redpanda or ``confluent/cp-kafka``).

Wire format: JSON documents, one per window message::

    {"sender": "...", "sender_window": 12, "items": [["A1", 10.2, 12], ...],
     "W": {"A1": 2.0}, "C": {"A1": 3}}

This class implements the same ``publish``/``poll`` interface as
:class:`~approxiot.kafka_io.transport.InMemoryTransport`, so the node code is
transport-agnostic.  In real deployment the broker's own partitioning and the
containerized ``tc-netem`` rules provide the latency that the in-memory bus
models with ``arrival_window``.
"""

from __future__ import annotations

import json
from typing import Any, List

from approxiot.kafka_io.transport import ControlMessage, WindowMessage

try:  # pragma: no cover - optional dependency
    from confluent_kafka import Consumer as _KafkaConsumer
    from confluent_kafka import Producer as _KafkaProducer

    _KAFKA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _KAFKA_AVAILABLE = False


def _encode(message: Any) -> bytes:
    if isinstance(message, WindowMessage):
        payload = {
            "type": "window",
            "sender": message.sender,
            "sender_window": message.sender_window,
            "items": [
                [i.substream, i.value, i.src_window, i.created_at]
                for i in message.items
            ],
            "W": message.W,
            "C": message.C,
            "item_weights": list(message.item_weights),
        }
    elif isinstance(message, ControlMessage):
        payload = {
            "type": "control",
            "fraction": message.fraction,
            "generation": message.generation,
            "issued_window": message.issued_window,
            "target": message.target,
            "policy": message.policy,
        }
    else:  # pragma: no cover
        raise TypeError(f"unsupported message type {type(message)!r}")
    return json.dumps(payload).encode()


def _decode(raw: bytes) -> Any:
    from approxiot.items import Item

    payload = json.loads(raw)
    if payload["type"] == "window":
        items = []
        for row in payload["items"]:
            if len(row) >= 4:
                items.append(Item(row[0], float(row[1]), int(row[2]), created_at=float(row[3])))
            else:  # backwards-compatible 3-field encoding
                items.append(Item(row[0], float(row[1]), int(row[2])))
        weights = [float(w) for w in payload.get("item_weights", [])]
        if weights and len(weights) == len(items):
            items = [it._replace(eff_weight=w) for it, w in zip(items, weights)]
        return WindowMessage(
            sender=payload["sender"],
            sender_window=payload["sender_window"],
            items=items,
            W={k: float(v) for k, v in payload["W"].items()},
            C={k: int(v) for k, v in payload["C"].items()},
            item_weights=weights,
        )
    return ControlMessage(
        fraction=payload["fraction"],
        generation=payload["generation"],
        issued_window=payload["issued_window"],
        target=payload["target"],
        policy=payload["policy"],
    )


class KafkaTransport:
    """``confluent-kafka``-backed transport implementing the bus interface.

    One transport serves one node.  A node subscribes to several topics (its
    data topic and the ``control`` topic), so ``poll`` filters the records it
    drains by topic and buffers the rest: ``confluent-kafka``'s ``consume``
    returns records from *all* subscribed topics regardless of which topic the
    caller asked for.  ``subscribe`` also **replaces** the subscription list, so
    subscriptions are grown cumulatively; otherwise polling the data topic
    silently unsubscribes the control topic and budget updates are never seen.

    ``producer``/``consumer`` may be injected for tests.  Each node needs its
    own ``group_id`` when sharing the control topic, otherwise the group
    splits the broadcast instead of every node seeing each update.
    """

    def __init__(
        self,
        bootstrap_servers: str = "redpanda:9092",
        group_id: str = "approxiot",
        *,
        producer: Any | None = None,
        consumer: Any | None = None,
    ):
        if producer is not None and consumer is not None:
            self._producer = producer
            self._consumer = consumer
        else:
            if not _KAFKA_AVAILABLE:
                raise ImportError("install the 'kafka' extra: pip install -e \".[kafka]\"")
            self._producer = _KafkaProducer({"bootstrap.servers": bootstrap_servers})
            self._consumer = _KafkaConsumer(
                {
                    "bootstrap.servers": bootstrap_servers,
                    "group.id": group_id,
                    "auto.offset.reset": "earliest",
                }
            )
        self._subscribed: set[str] = set()
        self._buffers: dict[str, List[Any]] = {}

    def publish(self, topic: str, message: Any, arrival_window: int = 0) -> None:
        # arrival_window is ignored: real latency comes from the network
        # (``tc-netem`` in docker-compose.yml).
        self._producer.produce(topic, _encode(message))
        self._producer.poll(0)

    def _subscribe(self, topic: str) -> None:
        if topic in self._subscribed:
            return
        self._subscribed.add(topic)
        self._buffers.setdefault(topic, [])
        self._consumer.subscribe(sorted(self._subscribed))

    def poll(self, topic: str, window: int = 0) -> List[Any]:
        self._subscribe(topic)
        for record in self._consumer.consume(num_messages=500, timeout=0.1) or []:
            if record.error():
                continue
            self._buffers.setdefault(record.topic(), []).append(_decode(record.value()))
        messages = self._buffers.get(topic, [])
        self._buffers[topic] = []
        return messages

    def backlog(self, topic: str) -> int:
        return len(self._buffers.get(topic, []))

    def flush(self, timeout: float = 10.0) -> None:
        self._producer.flush(timeout)

    def close(self) -> None:
        try:
            self._producer.flush(5.0)
        except Exception:  # pragma: no cover - best-effort shutdown
            pass
        try:
            self._consumer.close()
        except Exception:  # pragma: no cover - best-effort shutdown
            pass
