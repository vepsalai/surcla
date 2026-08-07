"""Failure-probability heads: P(fit fails | metafeatures) per fragile family.

Corpus fragility concentrates in Kriging (3.6% base rate, 14.7% at n >= 1000)
with MLP a distant second; every other family fails on under 0.2% of cells and
gets no head. Construction is the research one exactly: median imputation and
a 200-tree balanced-subsample random forest per fragile family, fit on the
corpus meta table's manual metafeatures with the family's failed-fit indicator
as the label. Measured discrimination (grouped five-fold out of fold):
Kriging AUC 0.970 in-corpus, 0.678 on development transfer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from .decoder import family_columns, feature_columns

MIN_FAIL_RATE = 0.002   # families below this corpus base rate get no head
TAU = 0.5               # demote a pick when P(fail) exceeds this


def make_head(seed: int = 0):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=200,
                               class_weight="balanced_subsample",
                               n_jobs=-1, random_state=seed))


class FailureHeads:
    """One binary classifier per fragile family, on the manual schema."""

    def __init__(self, tau: float = TAU, random_state: int = 0):
        self.tau = tau
        self.random_state = random_state
        self.features_: list[str] = []
        self.heads_: dict[str, object] = {}

    def fit(self, meta: pd.DataFrame) -> "FailureHeads":
        meta = meta[meta["winner"].notna()]
        self.features_ = sorted(feature_columns(meta))
        fams = [c.removeprefix("r2_") for c in family_columns(meta)]
        X = meta[self.features_]
        for f in fams:
            y = meta[f"r2_{f}"].isna().to_numpy()
            if y.mean() >= MIN_FAIL_RATE:
                self.heads_[f] = make_head(self.random_state).fit(X, y)
        return self

    def p_fail(self, query: pd.DataFrame) -> dict[str, float]:
        """Failure probability per fragile family for one query row."""
        q = query.reindex(columns=self.features_)
        return {f: float(h.predict_proba(q)[:, 1][0])
                for f, h in self.heads_.items()}
