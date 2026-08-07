"""Calibration of the predicted R²: published constant shift and error band.

The raw decoder prediction transfers to real data with an arm-dependent bias;
a constant shift fitted once on the development suite's prediction-realization
pairs corrects it, and the sealed-suite median |error| of the shifted
prediction is quoted as the band. All values are published measurements
shipped in artifacts/published_metrics.json — nothing is re-fitted at query
time.
"""

import json

from .data import artifact_path


def published_metrics() -> dict:
    return json.loads(artifact_path("published_metrics.json").read_text())


def shift_and_band(arm: str, seed: int) -> tuple[float, float]:
    c = published_metrics()["calibration"][arm][str(seed)]
    return float(c["shift"]), float(c["band"])
