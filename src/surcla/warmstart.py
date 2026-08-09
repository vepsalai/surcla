"""Warm-start configurations: what tuning chose on corpus datasets like yours.

Each labelled corpus fit recorded the configuration its own tuning selected.
The shipped table distils those into one corpus-typical configuration per
family and (training-size, feature-count) band, taken from cells where the
family was competitive. These are starting points, not predictions: no
hyperparameter head was trained, and a short local search on your own data is
expected to improve on them. For the UQ families the configuration is the
variant that won internally, with `meaning` spelling it out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from .data import artifact_path


@dataclass
class WarmStart:
    config: dict | str      # parameter dict, or variant tag for the UQ families
    meaning: str | None     # plain-language reading of a variant tag
    band: str               # the (n, d) band this came from
    n_cells: int            # corpus cells behind it
    median_r2: float        # median R² those cells reached
    median_fit_s: float     # median fit seconds there

    def __repr__(self):
        what = self.meaning or json.dumps(self.config)
        return f"WarmStart({what}, from {self.n_cells} {self.band} cells)"


@lru_cache(maxsize=1)
def _table() -> dict:
    return json.loads(artifact_path("warm_start.json").read_text())


def _band_of(value: float, names: list[str], edges: list[float]) -> str:
    for name, edge in zip(names, edges):
        if value <= edge:
            return name
    return names[-1]


def lookup(family: str, n_train: int, n_features: int) -> WarmStart | None:
    """Corpus-typical starting configuration for one family at this size."""
    t = _table()
    fam = t["families"].get(family)
    if not fam:
        return None
    n_name = _band_of(n_train, t["selection"]["n_bands"], [100, 700, float("inf")])
    d_name = _band_of(n_features, t["selection"]["d_bands"], [5, 15, float("inf")])
    entry = fam.get(f"{n_name}|{d_name}")
    if entry is None:
        return None
    config = entry["config"]
    meaning = None
    if isinstance(config, str):
        meaning = t["variant_meaning"].get(family, {}).get(config)
    return WarmStart(config=config, meaning=meaning,
                     band=f"{n_name}, {d_name}", n_cells=entry["n_cells"],
                     median_r2=entry["median_r2"],
                     median_fit_s=entry["median_fit_s"])
