"""Topology simulation: 8 sources -> 4 L1 -> 2 L2 -> 1 root (paper §V-A).

The paper's 25-machine testbed (8 sources, 4 first-layer, 2 second-layer edge
nodes, 1 datacenter node) is emulated in-process.  Kafka topics become
:class:`~approxiot.kafka_io.transport.InMemoryTransport` topics; the paper's
``tc-netem`` latencies (20 ms source->L1, 40 ms L1->L2, 80 ms L2->root) become
message arrival windows.  A node processes its window at the *end* of the
interval, so every hop costs one window; with deterministic latencies and a
fixed window size each root result therefore contains exactly one source
window's items (tagged via ``Item.src_window``), which the harness pairs with
the exact aggregate it estimates.

This is the documented substitution for the Docker/Kafka deployment: the plan
requires *qualitative* reproduction of the paper's trends, and an in-process
deterministic emulation makes every experiment reproducible with ``pytest`` and
plain Python.  ``docker-compose.yml`` + ``kafka_io/kafka_transport.py`` provide
the real-broker path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from collections import deque

import numpy as np

from approxiot.controller.adaptive_budget import AdaptiveBudgetController, ControllerConfig, ControllerEvent
from approxiot.datagen.real_world import RealWorldStream
from approxiot.datagen.synthetic import SyntheticStreamSpec, generate_window
from approxiot.error_estimation.error_bounds import Confidence
from approxiot.items import Item
from approxiot.kafka_io.consumer import Consumer
from approxiot.kafka_io.producer import Producer
from approxiot.kafka_io.transport import ControlMessage, InMemoryTransport
from approxiot.node.base_node import BaseNode, NodeConfig
from approxiot.node.root_node import RootNode, RootResult
from approxiot.node.sampling_node import SamplingNode

NUM_L1 = 4
NUM_L2 = 2


@dataclass
class LinkLatency:
    """Paper §V-A WAN emulation settings (ms)."""

    source_to_l1_ms: float = 20.0
    l1_to_l2_ms: float = 40.0
    l2_to_root_ms: float = 80.0
    control_ms: float = 20.0
    #: Extra uniform jitter per link (ms).  >0 exercises the asynchronous
    #: interval handling of §III-C / Eq. 9 (items and metadata straddle
    #: window boundaries).
    jitter_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.source_to_l1_ms + self.l1_to_l2_ms + self.l2_to_root_ms


class SourceDriver(Protocol):
    """Generates a source's items for one window and reports expected volume."""

    source_names: list[str]

    def generate(self, source_name: str, window_idx: int, window_seconds: float, rng: np.random.Generator) -> list[Item]: ...

    def expected_items(self, source_name: str, window_seconds: float) -> float: ...


class SyntheticSourceDriver:
    """Paper §V-A synthetic streams; optional per-window rate schedule (§V-D)."""

    def __init__(
        self,
        spec: SyntheticStreamSpec,
        schedule: Callable[[int], SyntheticStreamSpec] | None = None,
    ):
        self.spec = spec
        self.schedule = schedule
        self.source_names = list(spec.source_names)

    def _spec_for(self, window_idx: int) -> SyntheticStreamSpec:
        return self.schedule(window_idx) if self.schedule else self.spec

    def generate(self, source_name: str, window_idx: int, window_seconds: float, rng: np.random.Generator) -> list[Item]:
        return generate_window(self._spec_for(window_idx), source_name, window_idx, window_seconds, rng)

    def expected_items(self, source_name: str, window_seconds: float) -> float:
        return self.spec.source_rate(source_name, window_seconds)


@dataclass
class RealWorldSourceSpec:
    name: str
    pool: RealWorldStream
    rate_per_sec: float


class RealWorldSourceDriver:
    """Replays empirical value pools (NYC taxi / OpenAQ) at configured rates."""

    def __init__(self, sources: list[RealWorldSourceSpec], rate_scale: float = 1.0):
        self.sources = sources
        self.rate_scale = rate_scale
        self.source_names = [s.name for s in sources]
        self._by_name = {s.name: s for s in sources}

    def generate(self, source_name: str, window_idx: int, window_seconds: float, rng: np.random.Generator) -> list[Item]:
        src = self._by_name[source_name]
        expected = src.rate_per_sec * self.rate_scale * window_seconds
        n = int(rng.poisson(expected))
        values = src.pool.draw_window_values(n, rng)
        return [Item(substream=source_name, value=float(v), src_window=window_idx) for v in values]

    def expected_items(self, source_name: str, window_seconds: float) -> float:
        src = self._by_name[source_name]
        return src.rate_per_sec * self.rate_scale * window_seconds


@dataclass
class SimulationConfig:
    num_windows: int = 60
    window_seconds: float = 1.0
    method: str = "approxiot"  # approxiot | srs | native
    fraction: float = 0.5  # static sampling fraction / controller start point
    query: str = "sum"  # sum | mean
    seed: int = 42
    async_correction: bool = True
    latency: LinkLatency = field(default_factory=LinkLatency)
    controller: ControllerConfig | None = None
    confidence: Confidence = Confidence.P997


@dataclass
class SimulationResult:
    rows: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    controller_history: list[ControllerEvent] = field(default_factory=list)
    node_totals: dict[str, dict] = field(default_factory=dict)
    exact_by_window: dict[int, float] = field(default_factory=dict)


class Simulation:
    def __init__(self, driver: SourceDriver, config: SimulationConfig):
        self.driver = driver
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.transport = InMemoryTransport()
        self.exact_by_window: dict[int, float] = {}
        self.exact_count_by_window: dict[int, int] = {}
        self._last_rows: list[dict] = []
        self.controller: AdaptiveBudgetController | None = None
        self._build_topology()

    # ------------------------------------------------------------- topology
    def _build_topology(self) -> None:
        cfg = self.config
        lat = cfg.latency
        window_ms = cfg.window_seconds * 1000.0
        sources = list(self.driver.source_names)
        n_sources = len(sources)

        self.source_nodes: list[SamplingNode] = []
        self.l1_nodes: list[SamplingNode] = []
        self.l2_nodes: list[SamplingNode] = []
        self.nodes: dict[str, BaseNode] = {}
        self.producers: dict[str, Producer] = {}
        self.consumers: dict[str, Consumer] = {}

        # Expected volumes drive each node's cost function before its first window.
        total_source_items = sum(self.driver.expected_items(s, cfg.window_seconds) for s in sources)
        f = cfg.fraction if cfg.method != "native" else 1.0

        for i, name in enumerate(sources):
            parent = f"L1_{i % NUM_L1 + 1}"
            node = SamplingNode(
                config=NodeConfig(
                    name=name,
                    parent=parent,
                    expected_items_per_window=max(1.0, self.driver.expected_items(name, cfg.window_seconds)),
                ),
                rng=np.random.default_rng(cfg.seed + 1 + i),
                async_correction=cfg.async_correction,
                method=cfg.method,
            )
            node.fraction = f
            self.source_nodes.append(node)
            self.nodes[name] = node
            self.producers[name] = Producer(
                self.transport, f"in_{parent}", lat.source_to_l1_ms, window_ms, lat.jitter_ms, self.rng
            )

        for j in range(NUM_L1):
            name = f"L1_{j + 1}"
            parent = f"L2_{j % NUM_L2 + 1}"
            node = SamplingNode(
                config=NodeConfig(
                    name=name,
                    parent=parent,
                    expected_items_per_window=max(1.0, total_source_items / NUM_L1 * f),
                ),
                rng=np.random.default_rng(cfg.seed + 100 + j),
                async_correction=cfg.async_correction,
                method=cfg.method,
            )
            node.fraction = f
            self.l1_nodes.append(node)
            self.nodes[name] = node
            self.producers[name] = Producer(
                self.transport, f"in_{parent}", lat.l1_to_l2_ms, window_ms, lat.jitter_ms, self.rng
            )
            self.consumers[name] = Consumer(self.transport, f"in_{name}")

        for k in range(NUM_L2):
            name = f"L2_{k + 1}"
            node = SamplingNode(
                config=NodeConfig(
                    name=name,
                    parent="root",
                    expected_items_per_window=max(1.0, total_source_items / NUM_L2 * f * f),
                ),
                rng=np.random.default_rng(cfg.seed + 200 + k),
                async_correction=cfg.async_correction,
                method=cfg.method,
            )
            node.fraction = f
            self.l2_nodes.append(node)
            self.nodes[name] = node
            self.producers[name] = Producer(
                self.transport, "in_root", lat.l2_to_root_ms, window_ms, lat.jitter_ms, self.rng
            )
            self.consumers[name] = Consumer(self.transport, f"in_{name}")

        self.root = RootNode(
            config=NodeConfig(
                name="root",
                parent=None,
                expected_items_per_window=max(1.0, total_source_items * f ** 3),
            ),
            rng=np.random.default_rng(cfg.seed + 300),
            async_correction=cfg.async_correction,
            method=cfg.method,
            query=cfg.query,
            confidence=cfg.confidence,
        )
        self.root.fraction = f
        self.nodes["root"] = self.root
        self.consumers["root"] = Consumer(self.transport, "in_root")
        self.control_consumer = Consumer(self.transport, "control")
        self.control_producer = Producer(
            self.transport, "control", lat.control_ms, window_ms, 0.0, self.rng
        )

        if cfg.controller is not None:
            self.controller = AdaptiveBudgetController(cfg.controller, rng=np.random.default_rng(cfg.seed + 999))
            self.controller.bind_fraction(f)

    # ---------------------------------------------------------------- running
    def _apply_control(self, window: int) -> None:
        """Broadcast control-plane updates: one drain, applied to every node."""
        messages = self.control_consumer.poll(window)
        if not messages:
            return
        for message in messages:
            if isinstance(message, ControlMessage):
                for node in self.nodes.values():
                    node.apply_control(message.fraction)

    def run(self) -> SimulationResult:
        cfg = self.config
        result = SimulationResult()
        wall_start = time.perf_counter()
        cum_approx = 0.0
        # Controller signal: stderr aggregated over a trailing window of
        # observations (CLT over K windows), as a single-window stderr has a
        # sqrt(2/(Y-1)) relative noise floor that makes the loop chase noise
        # at replication scale.
        signal_len = 8
        recent_approx: deque[float] = deque(maxlen=signal_len)
        recent_var: deque[float] = deque(maxlen=signal_len)

        def aggregate_stderr() -> float:
            if not recent_approx:
                return 0.0
            total_var = sum(recent_var)
            total_approx = sum(recent_approx)
            if total_approx == 0:
                return 0.0
            return float(np.sqrt(total_var) / abs(total_approx))

        for t in range(cfg.num_windows):
            self._apply_control(t)

            # --- sources generate and sample -------------------------------
            for node in self.source_nodes:
                items = self.driver.generate(node.config.name, t, cfg.window_seconds, self.rng)
                self.exact_by_window[t] = self.exact_by_window.get(t, 0.0) + sum(i.value for i in items)
                self.exact_count_by_window[t] = self.exact_count_by_window.get(t, 0) + len(items)
                sampled = node.process_window(t, [], node.current_budget(), extra_items=items)
                self.producers[node.config.name].send(node.make_message(t, sampled), t)

            # --- first edge layer ------------------------------------------
            for node in self.l1_nodes:
                messages = self.consumers[node.config.name].poll(t)
                sampled = node.process_window(t, messages, node.current_budget())
                self.producers[node.config.name].send(node.make_message(t, sampled), t)

            # --- second edge layer ------------------------------------------
            for node in self.l2_nodes:
                messages = self.consumers[node.config.name].poll(t)
                sampled = node.process_window(t, messages, node.current_budget())
                self.producers[node.config.name].send(node.make_message(t, sampled), t)

            # --- root: query + error estimation -----------------------------
            messages = self.consumers["root"].poll(t)
            root_result: RootResult = self.root.execute(t, messages, self.root.current_budget())
            self._record(t, root_result, result, cum_approx, messages)
            cum_approx += root_result.approx

            # --- control loop ------------------------------------------------
            recent_approx.append(root_result.approx)
            recent_var.append(root_result.variance)
            if self.controller is not None:
                control = self.controller.observe(t, aggregate_stderr())
                if control is not None:
                    self.control_producer.send(control, t)

        wall_seconds = time.perf_counter() - wall_start
        self._last_rows = result.rows
        result.summary = self._summarize(wall_seconds)
        result.controller_history = self.controller.history if self.controller else []
        result.exact_by_window = dict(self.exact_by_window)
        result.node_totals = {
            name: {"items_in": node.total_in, "items_out": node.total_out}
            for name, node in self.nodes.items()
        }
        return result

    # ---------------------------------------------------------------- metrics
    def _record(
        self,
        window: int,
        root_result: RootResult,
        result: SimulationResult,
        cum_approx: float,
        messages: list,
    ) -> None:
        cfg = self.config
        exact = self.exact_by_window.get(root_result.src_window)
        realized_loss = None
        if exact not in (None, 0.0) and root_result.n_in > 0:
            realized_loss = abs(root_result.approx - exact) / abs(exact)

        # Cumulative pairing: under asynchronous arrivals (Fig. 3) a root window
        # mixes items from several source windows, so per-window pairing is only
        # an approximation; the cumulative sum comparison is exact once every
        # generated item has had time to traverse the pipeline.  Per-hop delay is
        # 1 window (2 under jitter), 3 hops total.
        max_delay = 3 * (2 if cfg.latency.jitter_ms > 0 else 1)
        settled = max(window - max_delay, -1)
        cum_exact = sum(v for w, v in self.exact_by_window.items() if w <= settled)
        cumulative_loss = (
            abs(cum_approx - cum_exact) / abs(cum_exact)
            if cum_exact not in (0.0,) and settled >= 0
            else float("nan")
        )
        n_mixture = len({i.src_window for msg in messages for i in msg.items})

        total_in = sum(n.total_in for n in self.nodes.values())
        total_out = sum(n.total_out for n in self.nodes.values())
        fractions = [n.fraction for n in self.nodes.values()]

        result.rows.append(
            {
                "method": cfg.method,
                "query": cfg.query,
                "seed": cfg.seed,
                "window": window,
                "src_window": root_result.src_window,
                "n_in_root": root_result.n_in,
                "n_sampled_root": root_result.n_sampled,
                "approx": root_result.approx,
                "exact": exact if exact is not None else float("nan"),
                "error_bound": root_result.error,
                "variance": root_result.variance,
                "stderr_relative": root_result.stderr_relative,
                "realized_loss": realized_loss if realized_loss is not None else float("nan"),
                "cumulative_loss": cumulative_loss,
                "n_src_windows_mixed": n_mixture,
                "fraction_root": self.root.fraction,
                "fraction_mean_nodes": sum(fractions) / len(fractions),
                "items_in_total": total_in,
                "items_out_total": total_out,
                "bandwidth_saved": 1.0 - (total_out / total_in) if total_in else 0.0,
                "e2e_latency_ms": 3 * cfg.window_seconds * 1000.0 + cfg.latency.total_ms,
            }
        )

    def _summarize(self, wall_seconds: float) -> dict:
        cfg = self.config
        rows = [r for r in self._last_rows if r["n_in_root"] > 0]
        losses = [r["realized_loss"] for r in rows if not np.isnan(r["realized_loss"])]
        covered = [
            abs(r["approx"] - r["exact"]) <= r["error_bound"]
            for r in rows
            if not np.isnan(r["exact"])
        ]
        total_in = sum(n.total_in for n in self.nodes.values())
        total_out = sum(n.total_out for n in self.nodes.values())
        summary = {
            "method": cfg.method,
            "query": cfg.query,
            "num_windows": cfg.num_windows,
            "window_seconds": cfg.window_seconds,
            "static_fraction": cfg.fraction,
            "paired_windows": len(losses),
            "accuracy_loss_mean": float(np.mean(losses)) if losses else float("nan"),
            "accuracy_loss_median": float(np.median(losses)) if losses else float("nan"),
            "accuracy_loss_p95": float(np.percentile(losses, 95)) if losses else float("nan"),
            "cumulative_loss": (
                float(result_rows[-1]["cumulative_loss"])
                if (result_rows := self._last_rows) and not np.isnan(result_rows[-1]["cumulative_loss"])
                else float("nan")
            ),
            "bound_coverage": float(np.mean(covered)) if covered else float("nan"),
            "e2e_latency_ms": 3 * cfg.window_seconds * 1000.0 + cfg.latency.total_ms,
            "bandwidth_saved": 1.0 - (total_out / total_in) if total_in else 0.0,
            "items_generated": sum(self.exact_count_by_window.values()),
            "wall_seconds": wall_seconds,
            "throughput_items_per_sec": (total_in / wall_seconds) if wall_seconds > 0 else float("nan"),
            "final_fraction_root": self.root.fraction,
        }
        if self.controller is not None:
            summary["controller_oscillation"] = self.controller.oscillation()
            summary["controller_num_changes"] = sum(
                1 for e in self.controller.history if e.action == "set"
            )
        return summary



def run_simulation(
    driver: SourceDriver,
    config: SimulationConfig | None = None,
    extra_columns: dict | None = None,
) -> SimulationResult:
    """Convenience wrapper: build, run, annotate and return the result."""
    config = config or SimulationConfig()
    sim = Simulation(driver, config)
    result = sim.run()
    if extra_columns:
        for row in result.rows:
            row.update(extra_columns)
        result.summary.update(extra_columns)
    return result
