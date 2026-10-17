"""Tests for the optional Kafka transport (plan Phase 3, Fig. 5 topics).

No broker is required: the confluent-kafka clients are injected as fakes, so
the wire-format codec and per-topic routing (a node subscribes to both its
data topic and the control topic) are exercised deterministically.
"""

from approxiot.items import Item
from approxiot.kafka_io.kafka_transport import KafkaTransport, _decode, _encode
from approxiot.kafka_io.transport import ControlMessage, WindowMessage


class FakeProducer:
    def __init__(self):
        self.sent = []
        self.flushed = 0

    def produce(self, topic, payload):
        self.sent.append((topic, payload))

    def poll(self, timeout):
        pass

    def flush(self, timeout):
        self.flushed += 1


class FakeRecord:
    def __init__(self, topic, payload):
        self._topic = topic
        self._payload = payload

    def error(self):
        return None

    def topic(self):
        return self._topic

    def value(self):
        return self._payload


class FakeConsumer:
    def __init__(self, records):
        self.records = list(records)
        self.subscriptions = []
        self.closed = False

    def subscribe(self, topics):
        self.subscriptions.append(list(topics))

    def consume(self, num_messages, timeout):
        batch, self.records = self.records[:num_messages], self.records[num_messages:]
        return batch

    def close(self):
        self.closed = True


def _window(sender="A1"):
    return WindowMessage(
        sender=sender,
        sender_window=3,
        items=[Item("A1", 10.5, 3, 2.0)],
        W={"A1": 2.0},
        C={"A1": 4},
        item_weights=[2.0],
    )


def test_codec_roundtrip_preserves_items_and_weights():
    message = _window()
    decoded = _decode(_encode(message))
    assert decoded.sender == "A1"
    assert decoded.sender_window == 3
    assert decoded.W == {"A1": 2.0}
    assert decoded.C == {"A1": 4}
    assert decoded.items[0] == Item("A1", 10.5, 3, 2.0)


def test_codec_roundtrip_control_message():
    message = ControlMessage(fraction=0.42, generation=7, issued_window=11,
                             target=0.005, policy="pid")
    assert _decode(_encode(message)) == message


def test_poll_filters_records_by_requested_topic_and_buffers_the_rest():
    window_payload = _encode(_window())
    control_payload = _encode(
        ControlMessage(fraction=0.3, generation=1, issued_window=5)
    )
    consumer = FakeConsumer([
        FakeRecord("in_L1_1", window_payload),
        FakeRecord("control", control_payload),
    ])
    transport = KafkaTransport(producer=FakeProducer(), consumer=consumer)

    data = transport.poll("in_L1_1")
    assert len(data) == 1 and isinstance(data[0], WindowMessage)
    assert transport.backlog("control") == 1

    control = transport.poll("control")
    # Subscriptions are cumulative: adding the control topic must not drop the
    # data topic (confluent-kafka's subscribe replaces the whole list).
    assert consumer.subscriptions[-1] == ["control", "in_L1_1"]
    assert len(control) == 1 and isinstance(control[0], ControlMessage)
    assert control[0].fraction == 0.3
    assert transport.backlog("control") == 0


def test_publish_encodes_and_flush_close_are_safe():
    consumer = FakeConsumer([])
    producer = FakeProducer()
    transport = KafkaTransport(producer=producer, consumer=consumer)

    transport.publish("in_L1_1", _window())
    topic, payload = producer.sent[0]
    assert topic == "in_L1_1"
    assert _decode(payload).sender == "A1"

    transport.flush()
    transport.close()
    assert producer.flushed >= 2
    assert consumer.closed
