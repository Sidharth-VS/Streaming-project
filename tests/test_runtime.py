"""Tests for the containerized node runtime (plan Phase 3)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pytest

from approxiot.runtime.node import NodeRuntime, RuntimeConfig, build_workload_spec


class FakeBus:
    def __init__(self):
        self.queues = defaultdict(list)

    def publish(self, topic, message, arrival_window=0):
        self.queues[topic].append(message)

    def poll(self, topic, window=0):
        out = list(self.queues[topic])
        self.queues[topic].clear()
        return out

    def flush(self, timeout=0.0):
        pass

    def close(self):
        pass


def _source_config(**overrides) -> RuntimeConfig:
    values = dict(
        node_name="A1", role="source", parent="L1_1", topic_out="in_root",
        workload="gaussian", rate_scale=0.01, window_seconds=1.0,
        fraction=0.5, max_windows=8, seed=7,
    )
    values.update(overrides)
    return RuntimeConfig(**values)


def _root_config(**overrides) -> RuntimeConfig:
    values = dict(
        node_name="root", role="root", topic_in="in_root",
        workload="gaussian", rate_scale=0.01, window_seconds=1.0,
        fraction=0.5, max_windows=8, seed=7, query="sum",
    )
    values.update(overrides)
    return RuntimeConfig(**values)


def test_source_to_root_pipeline_produces_estimates_and_csv(tmp_path: Path):
    bus = FakeBus()
    results = tmp_path / "root.csv"
    source = NodeRuntime(_source_config(), bus)
    root = NodeRuntime(_root_config(results_path=str(results)), bus)

    for window in range(10):
        source.step(window)
        root.step(window)

    non_empty = [r for r in root.rows if r["n_in_root"] > 0]
    assert non_empty, "root must receive sampled items"
    assert all(r["approx"] > 0 for r in non_empty)
    assert all(r["n_sampled_root"] <= r["n_in_root"] for r in non_empty)
    # Deployment-path latency is measured from item timestamps at the root.
    assert all(r["e2e_latency_ms"] >= 0 for r in non_empty)
    assert all(r["e2e_latency_ms"] == r["e2e_latency_ms"] for r in non_empty)

    root.finish()
    with open(results) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 10 and rows[-1]["approx"] != ""


def test_source_sampling_reduces_forwarded_volume():
    bus = FakeBus()
    source = NodeRuntime(_source_config(fraction=0.2, rate_scale=0.02), bus)
    source.step(0)
    assert source.node.total_in > 0
    assert source.node.total_out < source.node.total_in


def test_controller_at_root_broadcasts_budget_updates(tmp_path: Path):
    bus = FakeBus()
    root = NodeRuntime(
        _root_config(
            results_path=str(tmp_path / "root.csv"),
            target_error=1e-6, policy="proportional",
            cadence_windows=1, min_dwell_windows=1, dead_band=0.0,
            fraction=0.2,
        ),
        bus,
    )
    from approxiot.node.root_node import RootResult

    fake = RootResult(
        window=0, query="sum", approx=100.0, error=10.0, variance=11.1,
        stderr_relative=0.03, n_in=100, n_sampled=20,
    )
    root._maybe_control(0, fake)
    root._maybe_control(1, fake)  # confirmation-before-actuation
    assert bus.queues["control"], "controller must publish a ControlMessage"
    assert bus.queues["control"][0].fraction > 0.2


def test_config_from_env_and_validation():
    env = {
        "NODE_NAME": "L2_1", "NODE_ROLE": "edge", "NODE_PARENT": "root",
        "TOPIC_IN": "in_L2_1", "TOPIC_OUT": "in_root",
        "WINDOW_SECONDS": "1.5", "FRACTION": "0.3", "MAX_WINDOWS": "12",
        "TARGET_ERROR": "0.005",
    }
    cfg = RuntimeConfig.from_env(env)
    assert cfg.node_name == "L2_1" and cfg.role == "edge"
    assert cfg.window_seconds == 1.5 and cfg.max_windows == 12
    assert cfg.target_error == 0.005

    with pytest.raises(ValueError):
        RuntimeConfig(node_name="x", role="edge", topic_out="t")
    with pytest.raises(ValueError):
        RuntimeConfig(node_name="x", role="source", parent="L1_1", topic_out="t",
                      fraction=0.0)
    with pytest.raises(ValueError):
        RuntimeConfig.from_env({"NODE_NAME": "x", "NODE_ROLE": "nope"})


def test_workload_specs_match_paper_parameters():
    gauss = build_workload_spec("gaussian", 0.01)
    assert gauss.source_names == ["A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"]
    assert gauss.source_rate("A1", 1.0) == pytest.approx(50_000 / 2 * 0.01)

    skew = build_workload_spec("skew", 0.01)
    assert skew.rates == {"A": 80.0, "B": 19.89, "C": 0.1, "D": 0.01}

    with pytest.raises(ValueError):
        build_workload_spec("uniform", 0.01)
