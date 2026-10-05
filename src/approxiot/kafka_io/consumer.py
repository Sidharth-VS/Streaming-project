"""Consumer side of the pub/sub module (paper Fig. 5, module I)."""

from __future__ import annotations

from typing import Any, List

from approxiot.kafka_io.transport import InMemoryTransport


class Consumer:
    def __init__(self, transport: InMemoryTransport, topic: str) -> None:
        self.transport = transport
        self.topic = topic

    def poll(self, window: int) -> List[Any]:
        """Return every message that arrived by ``window`` (drains the queue)."""
        return self.transport.poll(self.topic, window)

    def backlog(self) -> int:
        return self.transport.pending(self.topic)
