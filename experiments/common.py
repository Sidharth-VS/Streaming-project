"""Shared infrastructure for the experiment modules.

Conventions (documented deviations from the original testbed, plan.md §1/§5):

- The paper's absolute rates (10^4–10^5 items/sec) are scaled down by
  ``SIM_RATE_SCALE`` so the pure-Python simulation stays fast; *relative*
  rates, value distributions, the tree shape and all qualitative trends are
  preserved (plan.md §5 explicitly permits this).
- Every run pins its RNG seeds; all results in ``results/`` are reproducible
  with the same command.
- Matplotlib uses the non-interactive 'Agg' backend: nothing pops up windows,
  everything is written to disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from approxiot.datagen.real_world import RealWorldStream  # noqa: E402
from approxiot.datagen.synthetic import (  # noqa: E402
    DEFAULT_RATES,
    SKEW_TYPES,
    SyntheticStreamSpec,
)
from approxiot.metrics.collector import MetricsCollector  # noqa: E402
from approxiot.simulation import (  # noqa: E402
    LinkLatency,
    RealWorldSourceDriver,
    RealWorldSourceSpec,
    Simulation,
    SimulationConfig,
    SyntheticSourceDriver,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

#: Global slowdown for the paper's item rates.  0.02 keeps one 40-window run
#: at a few thousand items/window (fast on CPU) while keeping A:B:C:D ≈ 80:40:20:1.
SIM_RATE_SCALE = 0.02

#: Number of windows per run (the paper's experiments run "about an hour";
#: 64 windows capture the same steady state at this scale).
NUM_WINDOWS = 64

#: Static sampling fractions swept by exp1/exp2-style experiments.
FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

METHODS = ("approxiot", "srs", "native")


# --------------------------------------------------------------------- specs
def gaussian_spec() -> SyntheticStreamSpec:
    """Paper §V-A Gaussian microbenchmark (A/B/C/D at DEFAULT_RATES)."""
    from approxiot.datagen.synthetic import GAUSSIAN_TYPES

    return SyntheticStreamSpec(types=dict(GAUSSIAN_TYPES), rate_scale=SIM_RATE_SCALE)


def poisson_spec() -> SyntheticStreamSpec:
    """Paper §V-A Poisson microbenchmark."""
    from approxiot.datagen.synthetic import POISSON_TYPES

    return SyntheticStreamSpec(types=dict(POISSON_TYPES), rate_scale=SIM_RATE_SCALE)


def skew_spec() -> SyntheticStreamSpec:
    """Paper §V-E extreme-skew workload (80 / 19.89 / 0.1 / 0.01 % volume)."""
    spec = SyntheticStreamSpec(types=dict(SKEW_TYPES), rate_scale=SIM_RATE_SCALE)
    spec.rates = {"A": 80.0, "B": 19.89, "C": 0.1, "D": 0.01}
    return spec


def rate_setting_spec(setting: str) -> SyntheticStreamSpec:
    """Paper §V-D fluctuating-rate settings 1/2/3 (A:B:C:D items/sec)."""
    from approxiot.datagen.synthetic import FLUCTUATING_RATE_SETTINGS

    rates = FLUCTUATING_RATE_SETTINGS[setting]
    spec = gaussian_spec()
    spec.rates = dict(rates)
    return spec


def default_latency() -> LinkLatency:
    """Paper §V-A WAN emulation: 20/40/80 ms source→L1→L2→root."""
    return LinkLatency()


# -------------------------------------------------------------------- runner
def build_config(**overrides) -> SimulationConfig:
    """Merge overrides over the standard experiment config."""
    cfg_kwargs = dict(
        num_windows=NUM_WINDOWS,
        window_seconds=1.0,
        fraction=0.5,
        query="sum",
        seed=42,
        latency=default_latency(),
    )
    cfg_kwargs.update(overrides)
    return SimulationConfig(**cfg_kwargs)


def run_config(driver, cfg: SimulationConfig) -> tuple[pd.DataFrame, dict]:
    """Run one simulation, return (window_rows_df, summary_dict)."""
    sim = Simulation(driver, cfg)
    result = sim.run()
    return pd.DataFrame(result.rows), dict(result.summary)


def sweep(
    driver_factory,
    methods=METHODS,
    fractions=FRACTIONS,
    cfg_overrides: dict | None = None,
) -> pd.DataFrame:
    """Run a (method, fraction) matrix; concatenate per-window rows."""
    frames = []
    for method in methods:
        for f in fractions:
            overrides = dict(cfg_overrides or {})
            overrides.update(method=method, fraction=f)
            cfg = build_config(**overrides)
            df, _ = run_config(driver_factory(), cfg)
            df.attrs["summary"] = None
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ exports
def save_csv(df: pd.DataFrame, name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    df.to_csv(path, index=False)
    return path


def save_json(obj, name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return path


def save_fig(fig: plt.Figure, name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ------------------------------------------------------------------ plotting
def loss_vs_fraction_plot(df: pd.DataFrame, title: str, fname: str) -> Path:
    """Accuracy loss vs sampling fraction (paper Fig. 6 analogue)."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method, g in df.groupby("method"):
        agg = g.groupby("fraction")["realized_loss"].mean().reset_index()
        ax.plot(agg["fraction"], agg["realized_loss"], marker="o", label=method)
    ax.set_xlabel("sampling fraction")
    ax.set_ylabel("accuracy loss (mean |err|/exact)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    return save_fig(fig, fname)


def throughput_plot(df: pd.DataFrame, title: str, fname: str) -> Path:
    """Throughput vs sampling fraction (paper Fig. 7 analogue)."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method, g in df.groupby("method"):
        agg = g.groupby("fraction")["items_out_total"].mean().reset_index()
        ax.plot(agg["fraction"], agg["items_out_total"], marker="o", label=method)
    ax.set_xlabel("sampling fraction")
    ax.set_ylabel("forwarded items / window (total)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return save_fig(fig, fname)


def summary_table(df: pd.DataFrame, group_keys=("method", "fraction")) -> pd.DataFrame:
    """Aggregate per-window rows into one row per group."""
    wanted = ["realized_loss", "cumulative_loss", "bound_coverage",
              "bandwidth_saved", "items_out_total", "e2e_latency_ms"]
    keep = [c for c in wanted if c in df.columns]
    agg = df.groupby(list(group_keys))[keep].mean().reset_index()
    return agg


__all__ = [
    "RESULTS_DIR",
    "NUM_WINDOWS",
    "FRACTIONS",
    "METHODS",
    "SIM_RATE_SCALE",
    "gaussian_spec",
    "poisson_spec",
    "skew_spec",
    "rate_setting_spec",
    "default_latency",
    "build_config",
    "run_config",
    "sweep",
    "save_csv",
    "save_json",
    "save_fig",
    "loss_vs_fraction_plot",
    "throughput_plot",
    "summary_table",
    "MetricsCollector",
    "RealWorldSourceDriver",
    "RealWorldSourceSpec",
    "RealWorldStream",
    "Simulation",
    "SimulationConfig",
    "SyntheticSourceDriver",
    "DEFAULT_RATES",
]
