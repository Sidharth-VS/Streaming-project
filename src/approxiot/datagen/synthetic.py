"""Synthetic data-stream generation (paper §V-A and §V-D/E parameters).

Sub-stream value distributions (exact parameters from the paper):

- Gaussian: A(mu=10, sigma=5), B(1000, 50), C(10000, 500), D(100000, 5000)
- Poisson:  A(lambda=10), B(100), C(1000), D(10000)
- Skew (§V-E): A(Poisson 10) at 80% of volume, B(100) at 19.89%,
  C(1000) at 0.1%, D(10_000_000) at 0.01%
- Fluctuating rates (§V-D), items/sec:
    Setting1 50k:25k:12.5k:625, Setting2 25k:25k:25k:25k, Setting3 625:12.5k:25k:50k

The replication scales absolute rates down (``rate_scale``) to keep the Python
simulation fast; relative rates, distributions and all *qualitative* trends are
preserved (plan.md §5 explicitly allows this).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from approxiot.items import Item

ValueDist = str  # "gauss" | "poisson"


@dataclass(frozen=True)
class SubStreamSpec:
    """One sub-stream type: value distribution + arrival rate (items/sec)."""

    name: str
    dist: ValueDist
    params: dict[str, float]
    rate: float  # items per second

    def draw_values(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.dist == "gauss":
            return rng.normal(self.params["mu"], self.params["sigma"], size=n)
        if self.dist == "poisson":
            return rng.poisson(self.params["lam"], size=n).astype(float)
        raise ValueError(f"unknown distribution {self.dist!r}")


#: Paper §V-A: default source rates for the microbenchmarks (items/sec per type).
DEFAULT_RATES: dict[str, float] = {"A": 50_000.0, "B": 25_000.0, "C": 12_500.0, "D": 625.0}

#: Paper §V-E skew scenario: volume shares + value distributions.
SKEW_TYPES: dict[str, SubStreamSpec] = {
    "A": SubStreamSpec("A", "poisson", {"lam": 10.0}, 0.80),
    "B": SubStreamSpec("B", "poisson", {"lam": 100.0}, 0.1989),
    "C": SubStreamSpec("C", "poisson", {"lam": 1000.0}, 0.001),
    "D": SubStreamSpec("D", "poisson", {"lam": 10_000_000.0}, 0.0001),
}

GAUSSIAN_TYPES: dict[str, SubStreamSpec] = {
    "A": SubStreamSpec("A", "gauss", {"mu": 10.0, "sigma": 5.0}, DEFAULT_RATES["A"]),
    "B": SubStreamSpec("B", "gauss", {"mu": 1000.0, "sigma": 50.0}, DEFAULT_RATES["B"]),
    "C": SubStreamSpec("C", "gauss", {"mu": 10_000.0, "sigma": 500.0}, DEFAULT_RATES["C"]),
    "D": SubStreamSpec("D", "gauss", {"mu": 100_000.0, "sigma": 5000.0}, DEFAULT_RATES["D"]),
}

POISSON_TYPES: dict[str, SubStreamSpec] = {
    "A": SubStreamSpec("A", "poisson", {"lam": 10.0}, DEFAULT_RATES["A"]),
    "B": SubStreamSpec("B", "poisson", {"lam": 100.0}, DEFAULT_RATES["B"]),
    "C": SubStreamSpec("C", "poisson", {"lam": 1000.0}, DEFAULT_RATES["C"]),
    "D": SubStreamSpec("D", "poisson", {"lam": 10_000.0}, DEFAULT_RATES["D"]),
}

#: Paper §V-D fluctuating-rate settings (items/sec, A:B:C:D).
FLUCTUATING_RATE_SETTINGS: dict[str, dict[str, float]] = {
    "setting1": {"A": 50_000.0, "B": 25_000.0, "C": 12_500.0, "D": 625.0},
    "setting2": {"A": 25_000.0, "B": 25_000.0, "C": 25_000.0, "D": 25_000.0},
    "setting3": {"A": 625.0, "B": 12_500.0, "C": 25_000.0, "D": 50_000.0},
}


@dataclass
class SyntheticStreamSpec:
    """A full synthetic workload: 4 value types x the paper's 8-source tree."""

    types: dict[str, SubStreamSpec]
    #: How many physical sources per type (2 -> 8 sources total, paper §V-A).
    sources_per_type: int = 2
    #: Global multiplier applied to every rate (keeps the sim fast; trends equal).
    rate_scale: float = 0.01
    #: Per-type rates (items/sec, unscaled).  Defaults to each spec's own rate.
    rates: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rates:
            self.rates = {name: spec.rate for name, spec in self.types.items()}

    def with_rates(self, rates: dict[str, float]) -> "SyntheticStreamSpec":
        """Return a copy with per-type rates replaced (for §V-D settings)."""
        return SyntheticStreamSpec(
            types=self.types,
            sources_per_type=self.sources_per_type,
            rate_scale=self.rate_scale,
            rates=dict(rates),
        )

    @property
    def source_names(self) -> list[str]:
        names: list[str] = []
        for type_name, spec in self.types.items():
            for k in range(self.sources_per_type):
                names.append(f"{spec.name}{k + 1}")
        return names

    def source_type(self, source_name: str) -> str:
        return source_name[0]

    def source_rate(self, source_name: str, window_seconds: float) -> float:
        """Per-source rate in items/window (split evenly across same-type sources)."""
        type_name = self.source_type(source_name)
        per_source = self.rates[type_name] / self.sources_per_type
        return per_source * self.rate_scale * window_seconds


def generate_window(
    spec: SyntheticStreamSpec,
    source_name: str,
    window_idx: int,
    window_seconds: float,
    rng: np.random.Generator,
) -> list[Item]:
    """Generate one source's items for one time interval (Poisson arrivals).

    The number of items in a window is Poisson-distributed with mean
    ``rate * window_seconds`` — a standard model for independent arrivals.
    """
    expected = spec.source_rate(source_name, window_seconds)
    n = int(rng.poisson(expected))
    type_name = spec.source_type(source_name)
    values = spec.types[type_name].draw_values(n, rng)
    return [Item(substream=source_name, value=float(v), src_window=window_idx) for v in values]


def build_topology_sources(
    spec: SyntheticStreamSpec,
) -> dict[str, str]:
    """Map each of the 8 source names to its parent L1 node (s1..s4 -> L1_1..L1_4)."""
    sources = spec.source_names
    return {name: f"L1_{i % 4 + 1}" for i, name in enumerate(sources)}
