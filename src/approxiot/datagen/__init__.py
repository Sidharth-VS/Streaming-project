"""Stream generators: synthetic (paper §V-A) and real-world datasets (§VI)."""

from approxiot.datagen.synthetic import (
    SubStreamSpec,
    SyntheticStreamSpec,
    GAUSSIAN_TYPES,
    POISSON_TYPES,
    SKEW_TYPES,
    FLUCTUATING_RATE_SETTINGS,
    build_topology_sources,
    generate_window,
)

__all__ = [
    "SubStreamSpec",
    "SyntheticStreamSpec",
    "GAUSSIAN_TYPES",
    "POISSON_TYPES",
    "SKEW_TYPES",
    "FLUCTUATING_RATE_SETTINGS",
    "build_topology_sources",
    "generate_window",
]
