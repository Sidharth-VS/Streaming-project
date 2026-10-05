"""Sampling (non-root) node: Algorithm 1 lines 11-14.

Subscribes to its upstream topic, samples the window's input, and publishes
``{sample, W_out, C_out}`` to the parent's topic.
"""

from __future__ import annotations

from approxiot.kafka_io.transport import WindowMessage
from approxiot.node.base_node import BaseNode, NodeBudget
from approxiot.sampling.weighted_hierarchical import WHSampResult


class SamplingNode(BaseNode):
    def forward(self, window: int, result: WHSampResult) -> WindowMessage:
        """Package the window's sample for the parent (Algorithm 1 line 13)."""
        return self.make_message(window, result)

    def process_and_forward(
        self,
        window: int,
        messages: list[WindowMessage],
        budget: NodeBudget,
    ) -> WindowMessage:
        result = self.process_window(window, messages, budget)
        return self.forward(window, result)
