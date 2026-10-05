"""Producer side of the pub/sub module (paper Fig. 5, module I).

Wraps a transport and converts link latency into an arrival window: a message
sent at the end of window ``t`` over a link with ``latency_ms`` and a window of
``window_ms`` arrives ``max(1, ceil(latency_ms / window_ms))`` windows later
(any non-zero latency costs at least one window boundary in the discrete model).
"""

from __future__ import annotations

import math

import numpy as np

from approxiot.kafka_io.transport import InMemoryTransport, WindowMessage


class Producer:
    def __init__(
        self,
        transport: InMemoryTransport,
        topic: str,
        latency_ms: float = 0.0,
        window_ms: float = 1000.0,
        jitter_ms: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.transport = transport
        self.topic = topic
        self.latency_ms = latency_ms
        self.window_ms = window_ms
        self.jitter_ms = jitter_ms
        self._rng = rng

    def delay_windows(self) -> int:
        latency = self.latency_ms
        if self.jitter_ms > 0 and self._rng is not None:
            latency += float(self._rng.uniform(0.0, self.jitter_ms))
        if latency <= 0:
            return 0
        return max(1, int(math.ceil(latency / self.window_ms)))

    def send(self, message: object, send_window: int) -> int:
        """Schedule a message for delivery and return its base arrival window.

        Without jitter the whole message (metadata + items) arrives together.
        With ``jitter_ms > 0`` the *items* are scheduled individually, mirroring
        the paper's Fig. 3: sampled items straddle interval boundaries while the
        weight/count metadata does not.  That mismatch (``c != C_in``) is what
        the Eq. 9 calibration corrects at the receiving node.
        """
        base = self.delay_windows()
        if (
            isinstance(message, WindowMessage)
            and self.jitter_ms > 0
            and self._rng is not None
            and message.items
        ):
            meta = WindowMessage(
                message.sender,
                message.sender_window,
                [],
                dict(message.W),
                dict(message.C),
            )
            self.transport.publish(self.topic, meta, send_window + base)
            jitter_prob = min(self.jitter_ms / self.window_ms, 1.0)
            weights = message.item_weights or [1.0] * len(message.items)
            for item, w in zip(message.items, weights):
                extra = 1 if self._rng.random() < jitter_prob else 0
                delayed = WindowMessage(
                    message.sender, message.sender_window, [item], {}, {}, [w]
                )
                self.transport.publish(self.topic, delayed, send_window + base + extra)
            return send_window + base
        arrival = send_window + base
        self.transport.publish(self.topic, message, arrival)
        return arrival
