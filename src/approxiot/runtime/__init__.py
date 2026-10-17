"""Containerized node runtime (plan Phase 3: real Kafka/Redpanda deployment).

``approxiot.runtime.node`` is the entry point used by every container in
``docker-compose.yml``.  It wires Algorithm 1's per-window loop to a real
broker via :class:`~approxiot.kafka_io.kafka_transport.KafkaTransport` and, at
the root, runs the novel adaptive budget controller.
"""

from __future__ import annotations

from typing import Any

__all__ = ["NodeRuntime", "RuntimeConfig", "build_workload_spec", "main"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from approxiot.runtime import node

        return getattr(node, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
