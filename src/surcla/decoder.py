"""Regret decoder: predict per-family R² surfaces, recommend the argmax.

Regression instead of winner classification because the top-2 family gap is
below 0.01 R² on roughly half of all cells: winner labels are coin flips
among near-ties, while the predicted surface keeps the target well posed.
Exact port of the research decoder that produced the published numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# id/label columns of the meta tables; everything else numeric (minus r2_*)
# is an encoder feature
ID_COLS = {"source", "dataset", "prior", "dataset_idx", "size_key", "seed_key",
           "idx_seed", "n_train", "n_test", "winner", "winner_r2"}


def family_columns(meta: pd.DataFrame) -> list[str]:
    return sorted(c for c in meta.columns if c.startswith("r2_"))


def feature_columns(meta: pd.DataFrame) -> list[str]:
    return [c for c in meta.columns
            if c not in ID_COLS and not c.startswith("r2_")
            and meta[c].dtype.kind in "fi"]


def regret_of(picks, meta: pd.DataFrame) -> np.ndarray:
    """Clipped regret of recommendations: winner_r2 - r2_of_pick in [0, 1];
    a pick whose family failed to fit (NaN) counts as regret 1."""
    out = []
    for pick, (_, row) in zip(picks, meta.iterrows()):
        r = row.get(f"r2_{pick}", np.nan)
        out.append(1.0 if np.isnan(r) else float(np.clip(row["winner_r2"] - r, 0.0, 1.0)))
    return np.asarray(out)


class RegretDecoder:
    """Multi-output RF over per-family R², recommendation = argmax prediction."""

    def __init__(self, fail_value: float = -1.0, clip: tuple = (-1.0, 1.0),
                 random_state: int = 0):
        self.fail_value = fail_value
        self.clip = clip
        self.random_state = random_state
        self.families_: list[str] = []
        self.features_: list[str] = []
        self.model_ = None

    def fit(self, meta: pd.DataFrame, features: list[str] | None = None) -> "RegretDecoder":
        """Fit on a meta-dataset; seed-averages targets per (dataset, size)."""
        meta = meta[meta["winner"].notna()]
        self.families_ = family_columns(meta)
        feats = features if features is not None else feature_columns(meta)
        self.features_ = sorted(set(feats) | {"n_train"})

        agg = meta.groupby(["dataset", "size_key"]).agg(
            {**{c: "mean" for c in self.families_},
             **{f: "first" for f in self.features_}}).reset_index()
        Y = agg[self.families_].to_numpy(float)
        Y = np.clip(np.nan_to_num(Y, nan=self.fail_value), *self.clip)

        self.model_ = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(random_state=self.random_state, n_jobs=-1))
        self.model_.fit(agg[self.features_], Y)
        return self

    def predict_r2(self, meta: pd.DataFrame) -> pd.DataFrame:
        """Predicted R² per family, one row per input row."""
        pred = self.model_.predict(meta[self.features_])
        cols = [c.removeprefix("r2_") for c in self.families_]
        return pd.DataFrame(pred, columns=cols, index=meta.index)

    def recommend(self, meta: pd.DataFrame) -> pd.DataFrame:
        """Recommended family + predicted attainable R² (reject-option score)."""
        r2 = self.predict_r2(meta)
        return pd.DataFrame({"family": r2.idxmax(axis=1),
                             "predicted_r2": r2.max(axis=1)}, index=meta.index)
