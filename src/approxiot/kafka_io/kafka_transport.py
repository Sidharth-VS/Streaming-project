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
            "items": [[i.substream, i.value, i.src_window] for i in message.items],
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
        items = [Item(s, float(v), int(w)) for s, v, w in payload["items"]]
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


class KafkaTransport:  # pragma: no cover - requires a broker
    """``confluent-kafka``-backed transport implementing the bus interface."""

    def __init__(self, bootstrap_servers: str = "redpanda:9092", group_id: str = "approxiot"):
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

    def publish(self, topic: str, message: Any, arrival_window: int = 0) -> None:
        # arrival_window is ignored: real latency comes from the network.
        self._producer.produce(topic, _encode(message))
        self._producer.poll(0)

    def poll(self, topic: str, window: int = 0) -> List[Any]:
        if topic not in self._subscribed:
            self._consumer.subscribe([topic])
            self._subscribed.add(topic)
        messages: List[Any] = []
        for record in self._consumer.consume(num_messages=500, timeout=0.1):
            if record.error():
                continue
            messages.append(_decode(record.value()))
        return messages
