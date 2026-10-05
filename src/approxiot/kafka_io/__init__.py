"""Transport layer: in-memory simulation bus (default) and optional Kafka."""

from approxiot.kafka_io.transport import (
    ControlMessage,
    InMemoryTransport,
    WindowMessage,
)

__all__ = ["ControlMessage", "InMemoryTransport", "WindowMessage"]
