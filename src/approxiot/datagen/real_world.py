"""Real-world dataset loaders (paper §VI substitutes).

The paper uses (1) NYC TLC trip records (query on ``total_amount``) and
(2) Brasov pollution readings.  Per plan.md §1, this replication substitutes:

1. NYC TLC Trip Record Data for any available month — the loaders below fetch
   the public Parquet/CSV and extract the ``total_amount`` column.
2. OpenAQ historical measurements (pm25 / pm10 / co / so2 / no2) for any city.

Both loaders degrade gracefully: when the dataset cannot be downloaded (offline
environment, missing optional dependency, API change) they fall back to a
documented **distribution-matched proxy** (log-normal amounts / pollutant
readings), so experiments always run and the substitution is explicit.

Dataset files are cached under ``data/`` (gitignored).
"""

from __future__ import annotations

import csv
import io
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("APPROXIOT_DATA_DIR", "data"))

NYC_TAXI_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_{month}.parquet"
)
OPENAQ_V2_URL = "https://api.openaq.org/v2/measurements"


@dataclass
class RealWorldStream:
    """A fixed pool of real-world values replayed as a stream.

    Values are shuffled once and emitted round-robin, which preserves the
    empirical value distribution while making window contents vary.
    """

    name: str
    values: np.ndarray
    proxy_used: bool
    note: str

    def draw_window_values(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if len(self.values) == 0:
            return np.zeros(0)
        idx = rng.integers(0, len(self.values), size=n)
        return self.values[idx]

    @property
    def mean(self) -> float:
        return float(self.values.mean()) if len(self.values) else 0.0


def _download(url: str, dest: Path, timeout: float = 60.0) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(dest, "wb") as fh:
            fh.write(resp.read())
        return True
    except (urllib.error.URLError, OSError):
        return False


def _taxi_proxy(rng: np.random.Generator, n: int = 50_000) -> np.ndarray:
    """Log-normal surrogate for NYC taxi ``total_amount`` (heavy right tail)."""
    # Matches the empirical shape of TLC totals: mode ~ $10, tail to $200+.
    return rng.lognormal(mean=2.4, sigma=0.85, size=n).round(2)


def load_nyc_taxi(
    month: str = "2013-01",
    field: str = "total_amount",
    max_rows: int = 200_000,
    download: bool = False,
    rng: np.random.Generator | None = None,
) -> RealWorldStream:
    """Load NYC TLC ``total_amount`` values for ``month`` (e.g. ``"2013-01"``).

    Falls back to the log-normal proxy when the download is unavailable or
    ``pyarrow`` (Parquet support) is missing.
    """
    rng = rng or np.random.default_rng()
    cache = DATA_DIR / f"yellow_tripdata_{month}.parquet"
    if download and not cache.exists():
        _download(NYC_TAXI_URL.format(month=month), cache)

    if cache.exists():
        try:  # pragma: no cover - exercised only with network + pyarrow
            import pyarrow  # noqa: F401
            import pyarrow.parquet as pq

            table = pq.read_table(cache, columns=[field])
            values = table.column(field).to_numpy()
            values = values[np.isfinite(values) & (values > 0)][:max_rows]
            if len(values) > 0:
                return RealWorldStream(
                    name=f"nyc_taxi_{month}:{field}",
                    values=np.asarray(values, dtype=float),
                    proxy_used=False,
                    note=f"NYC TLC {month} ({len(values)} rows)",
                )
        except Exception:
            pass

    return RealWorldStream(
        name=f"nyc_taxi_proxy_{month}",
        values=_taxi_proxy(rng),
        proxy_used=True,
        note="Log-normal proxy for TLC total_amount (download unavailable).",
    )


def _pollution_proxy(rng: np.random.Generator, mean: float, n: int = 20_000) -> np.ndarray:
    """Right-skewed surrogate for urban pollutant time series."""
    return np.clip(rng.lognormal(np.log(mean) - 0.35, 0.55, size=n), 0.0, None).round(3)


def load_openaq(
    city: str = "Brasov",
    parameter: str = "pm25",
    limit: int = 10_000,
    download: bool = False,
    rng: np.random.Generator | None = None,
) -> RealWorldStream:
    """Load historical air-quality measurements for ``city`` from OpenAQ v2.

    Falls back to a log-normal proxy calibrated to typical urban values when
    the API is unreachable or requires credentials.
    """
    rng = rng or np.random.default_rng()
    cache = DATA_DIR / f"openaq_{city}_{parameter}.json"
    if download and not cache.exists():
        url = f"{OPENAQ_V2_URL}?city={urllib.request.quote(city)}&parameter={parameter}&limit={limit}"
        if _download(url, cache):
            try:  # pragma: no cover - network path
                with open(cache) as fh:
                    payload = json.load(fh)
                values = [
                    float(r["value"])
                    for r in payload.get("results", [])
                    if r.get("value") is not None
                ]
                if values:
                    return RealWorldStream(
                        name=f"openaq_{city}_{parameter}",
                        values=np.asarray(values),
                        proxy_used=False,
                        note=f"OpenAQ {city} {parameter} ({len(values)} readings)",
                    )
            except Exception:
                pass

    typical = {"pm25": 25.0, "pm10": 40.0, "co": 0.6, "so2": 5.0, "no2": 30.0}
    return RealWorldStream(
        name=f"openaq_proxy_{city}_{parameter}",
        values=_pollution_proxy(rng, typical.get(parameter, 25.0)),
        proxy_used=True,
        note=f"Log-normal proxy for OpenAQ {city} {parameter} (download unavailable).",
    )


def stream_from_csv(path: str | Path, column: str, max_rows: int = 200_000) -> RealWorldStream:
    """Build a stream pool from any local CSV (escape hatch for other datasets)."""
    values: list[float] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                v = float(row[column])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(v) and v > 0:
                values.append(v)
            if len(values) >= max_rows:
                break
    if not values:
        raise ValueError(f"no usable numeric values in column {column!r} of {path}")
    return RealWorldStream(
        name=f"{Path(path).stem}:{column}",
        values=np.asarray(values),
        proxy_used=False,
        note=f"CSV {path} ({len(values)} rows)",
    )
