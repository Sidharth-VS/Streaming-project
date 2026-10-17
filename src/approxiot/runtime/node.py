"""Per-container node runtime: Algorithm 1 over a real Kafka/Redpanda broker.

The simulation in :mod:`approxiot.simulation` is the reproducible in-process
path used by the experiments.  This module is the *deployment* path required by
plan Phase 3: one process per topology node, a broker carrying the paper's
Fig. 5 topics, and ``tc-netem`` (applied by the container entrypoint) supplying
the paper's 20/40/80 ms link latencies.

Topology (identical shape to the simulation)::

    8 sources -> 4 L1 -> 2 L2 -> 1 root

Every process runs the same loop: wait for the end of the current window,
adopt control-plane budget updates, read its input topic (sources generate
synthetic items instead), run Algorithm 1, and publish ``{sample, W, C}`` to
the parent's topic (the root writes a CSV row instead).

The root additionally feeds its relative standard error to the adaptive
controller and broadcasts budget updates on the dedicated ``control`` topic.

Configuration is environment-driven so ``docker-compose.yml`` can start 15
identical images (see that file for the per-node values).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from approxiot.controller.adaptive_budget import AdaptiveBudgetController, ControllerConfig
from approxiot.datagen.synthetic import (
    GAUSSIAN_TYPES,
    POISSON_TYPES,
    SKEW_TYPES,
    SyntheticStreamSpec,
    generate_window,
)
from approxiot.error_estimation.error_bounds import Confidence
from approxiot.kafka_io.transport import ControlMessage
from approxiot.node.base_node import BaseNode, NodeConfig
from approxiot.node.root_node import RootNode, RootResult
from approxiot.node.sampling_node import SamplingNode

ROLES = ("source", "edge", "root")

#: Root result CSV columns (one row per completed window).
RESULT_FIELDS = (
    "window",
    "src_window",
    "approx",
    "error_bound",
    "stderr_relative",
    "variance",
    "n_in_root",
    "n_sampled_root",
    "fraction_root",
    "items_in_total",
    "items_out_total",
    "bandwidth_saved",
    "e2e_latency_ms",
    "e2e_latency_p95_ms",
    "wall_time",
)

_CONFIDENCE_BY_NAME = {"p68": Confidence.P68, "p95": Confidence.P95, "p997": Confidence.P997}


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    return float(raw) if raw not in (None, "") else default


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    return int(raw) if raw not in (None, "") else default


def build_workload_spec(workload: str, rate_scale: float) -> SyntheticStreamSpec:
    """Build the paper's workload spec (plan §5) for the containerized run."""
    name = workload.lower()
    if name == "gaussian":
        return SyntheticStreamSpec(types=dict(GAUSSIAN_TYPES), rate_scale=rate_scale)
    if name == "poisson":
        return SyntheticStreamSpec(types=dict(POISSON_TYPES), rate_scale=rate_scale)
    if name == "skew":
        spec = SyntheticStreamSpec(types=dict(SKEW_TYPES), rate_scale=rate_scale)
        spec.rates = {"A": 80.0, "B": 19.89, "C": 0.1, "D": 0.01}
        return spec
    raise ValueError(f"unknown workload {workload!r} (use gaussian, poisson or skew)")


@dataclass
class RuntimeConfig:
    """Environment-driven configuration for one deployed node."""

    node_name: str
    role: str
    parent: str | None = None
    topic_in: str | None = None
    topic_out: str | None = None
    control_topic: str = "control"
    window_seconds: float = 2.0
    fraction: float = 0.5
    method: str = "approxiot"
    query: str = "sum"
    confidence: Confidence = Confidence.P997
    workload: str = "gaussian"
    rate_scale: float = 0.01
    seed: int = 42
    max_windows: int = 0
    start_delay_seconds: float = 0.0
    results_path: str | None = None
    target_error: float = 0.0
    policy: str = "pid"
    cadence_windows: int = 2
    min_dwell_windows: int = 3
    dead_band: float = 0.3
    bootstrap_servers: str = "redpanda:9092"

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"NODE_ROLE must be one of {ROLES}, got {self.role!r}")
        if self.role != "source" and not self.topic_in:
            raise ValueError(f"{self.role} node {self.node_name!r} needs TOPIC_IN")
        if self.role != "root" and not self.topic_out:
            raise ValueError(f"{self.role} node {self.node_name!r} needs TOPIC_OUT")
        if self.role == "source" and not self.parent:
            raise ValueError(f"source node {self.node_name!r} needs NODE_PARENT")
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError("FRACTION must be in (0, 1]")
        if self.window_seconds <= 0:
            raise ValueError("WINDOW_SECONDS must be positive")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RuntimeConfig":
        env = os.environ if env is None else env
        confidence_name = env.get("CONFIDENCE", "p997").lower()

        def required(key: str) -> str:
            value = env.get(key)
            if not value:
                raise ValueError(f"missing required environment variable {key}")
            return value

        return cls(
            node_name=required("NODE_NAME"),
            role=required("NODE_ROLE").lower(),
            parent=env.get("NODE_PARENT") or None,
            topic_in=env.get("TOPIC_IN") or None,
            topic_out=env.get("TOPIC_OUT") or None,
            control_topic=env.get("CONTROL_TOPIC", "control"),
            window_seconds=_env_float(env, "WINDOW_SECONDS", 2.0),
            fraction=_env_float(env, "FRACTION", 0.5),
            method=env.get("METHOD", "approxiot").lower(),
            query=env.get("QUERY", "sum").lower(),
            confidence=_CONFIDENCE_BY_NAME.get(confidence_name, Confidence.P997),
            workload=env.get("WORKLOAD", "gaussian").lower(),
            rate_scale=_env_float(env, "RATE_SCALE", 0.01),
            seed=_env_int(env, "SEED", 42),
            max_windows=_env_int(env, "MAX_WINDOWS", 0),
            start_delay_seconds=_env_float(env, "START_DELAY_SECONDS", 0.0),
            results_path=env.get("RESULTS_PATH") or None,
            target_error=_env_float(env, "TARGET_ERROR", 0.0),
            policy=env.get("POLICY", "pid").lower(),
            cadence_windows=_env_int(env, "CADENCE_WINDOWS", 2),
            min_dwell_windows=_env_int(env, "MIN_DWELL_WINDOWS", 3),
            dead_band=_env_float(env, "DEAD_BAND", 0.3),
            bootstrap_servers=env.get("KAFKA_BOOTSTRAP")
            or env.get("BOOTSTRAP_SERVERS")
            or "redpanda:9092",
        )


class NodeRuntime:
    """One deployed node: window scheduling + Algorithm 1 over a pub/sub bus."""

    def __init__(
        self,
        config: RuntimeConfig,
        transport: Any,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config
        self.transport = transport
        self.rng = rng or np.random.default_rng(config.seed + _stable_offset(config.node_name))
        self.spec = build_workload_spec(config.workload, config.rate_scale)
        self.total_expected = sum(
            self.spec.source_rate(name, config.window_seconds) for name in self.spec.source_names
        )
        self.node: BaseNode | RootNode = self._build_node()
        self.node.fraction = config.fraction

        self.controller = (
            AdaptiveBudgetController(
                ControllerConfig(
                    target_error=config.target_error,
                    policy=config.policy,
                    cadence_windows=config.cadence_windows,
                    min_dwell_windows=config.min_dwell_windows,
                    dead_band=config.dead_band,
                ),
                rng=np.random.default_rng(config.seed + 999),
            )
            if config.role == "root" and config.target_error > 0
            else None
        )
        if self.controller is not None:
            self.controller.bind_fraction(config.fraction)

        self.rows: list[dict] = []
        self.windows_completed = 0
        self._stop = False
        self._csv_file = None
        self._csv_writer: csv.DictWriter | None = None
        if config.role == "root" and config.results_path:
            self._open_results(config.results_path)

    # ------------------------------------------------------------- topology
    def _expected_input(self) -> float:
        cfg = self.config
        if cfg.role == "source":
            return max(1.0, self.spec.source_rate(cfg.node_name, cfg.window_seconds))
        depth = 1 if cfg.node_name.startswith("L1") else 2
        fanout = 4 if depth == 1 else 2
        return max(1.0, self.total_expected / fanout * cfg.fraction ** depth)

    def _build_node(self) -> BaseNode | RootNode:
        cfg = self.config
        node_rng = np.random.default_rng(cfg.seed + _stable_offset(cfg.node_name))
        if cfg.role == "root":
            return RootNode(
                config=NodeConfig(
                    name=cfg.node_name,
                    parent=None,
                    expected_items_per_window=max(1.0, self.total_expected * cfg.fraction ** 3),
                ),
                rng=node_rng,
                method=cfg.method,
                query=cfg.query,
                confidence=cfg.confidence,
            )
        return SamplingNode(
            config=NodeConfig(
                name=cfg.node_name,
                parent=cfg.parent,
                expected_items_per_window=self._expected_input(),
            ),
            rng=node_rng,
            method=cfg.method,
        )

    # ---------------------------------------------------------------- output
    def _open_results(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(target, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=RESULT_FIELDS)
        self._csv_writer.writeheader()
        self._csv_file.flush()

    def _write_row(self, row: dict) -> None:
        self.rows.append(row)
        if self._csv_writer is not None and self._csv_file is not None:
            self._csv_writer.writerow({k: row.get(k) for k in RESULT_FIELDS})
            self._csv_file.flush()

    # ---------------------------------------------------------------- window
    def apply_control(self) -> None:
        """Drain the control topic and adopt the latest broadcast budget."""
        for message in self.transport.poll(self.config.control_topic):
            if isinstance(message, ControlMessage):
                self.node.apply_control(message.fraction)

    def step(self, window: int) -> RootResult | None:
        """Run one window of Algorithm 1 (no sleeping; callers own the clock)."""
        cfg = self.config
        self.apply_control()

        if cfg.role == "source":
            items = generate_window(
                self.spec, cfg.node_name, window, cfg.window_seconds, self.rng
            )
            # Stamp items for the deployment path's real end-to-end latency
            # measurement (root records now - created_at).
            created = time.time()
            items = [item._replace(created_at=created) for item in items]
            result = self.node.process_window(
                window, [], self.node.current_budget(), extra_items=items
            )
            self.transport.publish(
                cfg.topic_out, self.node.make_message(window, result)
            )
        elif cfg.role == "edge":
            messages = self.transport.poll(cfg.topic_in)
            result = self.node.process_window(window, messages, self.node.current_budget())
            self.transport.publish(
                cfg.topic_out, self.node.make_message(window, result)
            )
        else:
            assert isinstance(self.node, RootNode)
            messages = self.transport.poll(cfg.topic_in)
            root_result = self.node.execute(window, messages, self.node.current_budget())
            self._record(window, root_result, messages)
            self._maybe_control(window, root_result)
            self.windows_completed += 1
            return root_result

        self.windows_completed += 1
        return None

    def _record(self, window: int, result: RootResult, messages: list) -> None:
        total_in = self.node.total_in
        total_out = self.node.total_out
        now = time.time()
        latencies = [
            (now - item.created_at) * 1000.0
            for msg in messages
            for item in msg.items
            if item.created_at > 0
        ]
        self._write_row(
            {
                "window": window,
                "src_window": result.src_window,
                "approx": result.approx,
                "error_bound": result.error,
                "stderr_relative": result.stderr_relative,
                "variance": result.variance,
                "n_in_root": result.n_in,
                "n_sampled_root": result.n_sampled,
                "fraction_root": self.node.fraction,
                "items_in_total": total_in,
                "items_out_total": total_out,
                "bandwidth_saved": 1.0 - (total_out / total_in) if total_in else 0.0,
                "e2e_latency_ms": float(np.mean(latencies)) if latencies else float("nan"),
                "e2e_latency_p95_ms": (
                    float(np.percentile(latencies, 95)) if latencies else float("nan")
                ),
                "wall_time": now,
            }
        )

    def _maybe_control(self, window: int, result: RootResult) -> None:
        if self.controller is None:
            return
        message = self.controller.observe(window, result.stderr_relative)
        if message is not None:
            self.transport.publish(self.config.control_topic, message)

    # ------------------------------------------------------------------ loop
    def run(self, max_windows: int | None = None) -> None:
        cfg = self.config
        total = cfg.max_windows if max_windows is None else max_windows
        if cfg.start_delay_seconds > 0:
            time.sleep(cfg.start_delay_seconds)
        started = time.monotonic()
        window = 0
        while not self._stop and (total <= 0 or window < total):
            deadline = started + (window + 1) * cfg.window_seconds
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            self.step(window)
            window += 1
        self.finish()

    def request_stop(self) -> None:
        self._stop = True

    def finish(self) -> dict:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        total_in = self.node.total_in
        total_out = self.node.total_out
        summary: dict[str, Any] = {
            "node": self.config.node_name,
            "role": self.config.role,
            "windows": self.windows_completed,
            "items_in": total_in,
            "items_out": total_out,
            "bandwidth_saved": round(1.0 - total_out / total_in, 6) if total_in else 0.0,
            "final_fraction": round(self.node.fraction, 6),
        }
        if self.config.role == "root":
            losses = [r["stderr_relative"] for r in self.rows if r["n_in_root"] > 0]
            summary["windows_with_results"] = len(losses)
            summary["median_relative_stderr"] = (
                round(float(np.median(losses)), 6) if losses else None
            )
            summary["results_path"] = self.config.results_path
            measured = [
                r["e2e_latency_ms"] for r in self.rows
                if np.isfinite(r["e2e_latency_ms"])
            ]
            summary["median_latency_ms"] = (
                round(float(np.median(measured)), 1) if measured else None
            )
            if self.controller is not None:
                summary["controller_num_changes"] = sum(
                    1 for e in self.controller.history if e.action == "set"
                )
                summary["final_controller_fraction"] = round(self.controller.fraction, 6)
        print(json.dumps(summary, default=str), flush=True)
        return summary


def _stable_offset(name: str) -> int:
    """Deterministic per-node RNG offset (Python's ``hash`` is salted)."""
    return sum(ord(c) * (i + 1) for i, c in enumerate(name))


def wait_for_broker(bootstrap_servers: str, timeout: float = 90.0) -> None:
    """Block until the broker's TCP port accepts connections."""
    host, _, port = bootstrap_servers.partition(":")
    port_num = int(port or 9092)
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port_num), timeout=2.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(f"broker {bootstrap_servers} not reachable: {last_error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one ApproxIoT topology node")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="override MAX_WINDOWS (0 = run until stopped)")
    parser.add_argument("--window-seconds", type=float, default=None,
                        help="override WINDOW_SECONDS")
    parser.add_argument("--bootstrap", default=None,
                        help="override KAFKA_BOOTSTRAP (host:port)")
    args = parser.parse_args(argv)

    config = RuntimeConfig.from_env()
    if args.max_windows is not None:
        config.max_windows = args.max_windows
    if args.window_seconds is not None:
        config.window_seconds = args.window_seconds
    if args.bootstrap:
        config.bootstrap_servers = args.bootstrap

    from approxiot.kafka_io.kafka_transport import KafkaTransport

    wait_for_broker(config.bootstrap_servers, timeout=120.0)
    transport = KafkaTransport(
        bootstrap_servers=config.bootstrap_servers,
        group_id=f"approxiot-{config.node_name}",
    )
    runtime = NodeRuntime(config, transport)

    def _handle(signum, frame):  # pragma: no cover - signal path
        runtime.request_stop()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    try:
        runtime.run()
    finally:
        transport.flush()
        transport.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
