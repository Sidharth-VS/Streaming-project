"""Collects per-window experiment metrics and exports tidy CSV files.

Metrics follow the paper's definitions (§V-A):
- **accuracy loss** = ``|approx - exact| / exact`` (Fig. 6 analogue)
- **throughput**    = items processed per second (Fig. 7 analogue)
- **latency**       = end-to-end time from source to datacenter (Fig. 10 analogue)
- plus per-window bandwidth (items forwarded vs received) and, for the
  controller experiments, the sampling fraction over time.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class MetricsCollector:
    """A list-of-dicts table with uniform columns and CSV export."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, **kwargs: Any) -> None:
        self.rows.append(dict(kwargs))

    def extend(self, rows: list[dict[str, Any]]) -> None:
        self.rows.extend(dict(r) for r in rows)

    def to_dataframe(self):  # -> pandas.DataFrame (import kept local)
        import pandas as pd

        return pd.DataFrame(self.rows)

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            path.write_text("")
            return path
        fieldnames = sorted({key for row in self.rows for key in row})
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        return path

    def __len__(self) -> int:
        return len(self.rows)
