"""The labelling run's input caps, as an estimator wrapper.

Kriging, PCE and PC-Kriging cost O(n^3) in training rows, so the labelling run
fitted them on at most 2000 rows and 30 features, the latter ranked by
random-forest importance. That bound is not only how the corpus stayed
affordable: it is part of what the labels mean. A predicted R2 for Kriging on
20000 rows is a prediction about a 2000-row fit, because that is the fit the
label measured, and an uncapped fit of the same family on the same data is a
different and far slower model. Measured on six features, one Kriging fit takes
1.5 s at n = 500, 81 s at 2000 and 343 s at 4000; cubic extrapolation puts
20000 rows near twelve hours, and cross-validation multiplies that by the fold
count.

`Capped` therefore reproduces those bounds around any estimator, and carries
the feature selection into `predict` so the caller keeps passing full-width
rows. The caps are defaults, not a policy: pass None to either and the fit uses
everything, at the price above.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

# The labelling protocol's values (research fit_surrogates.py).
MAX_ROWS = 2000
MAX_FEATURES = 30
CAPPED_FAMILIES = ("Kriging", "PCE", "PCK")

# Within the top-k by importance, a feature under this share of the largest
# importance is dropped as a straggler.
LOW_IMPORTANCE = 0.05


def select_features(X, y, max_features: int = MAX_FEATURES) -> np.ndarray:
    """Indices of the features the labelling run would have kept.

    A 50-tree forest ranks the features, the top `max_features` survive, and
    within those any whose importance is under 5% of the largest is dropped.
    At or below the limit every feature passes through untouched.
    """
    if X.shape[1] <= max_features:
        return np.arange(X.shape[1])
    rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    top = np.argsort(importances)[::-1][:max_features]
    kept = top[importances[top] >= LOW_IMPORTANCE * importances[top].max()]
    if len(kept) == 0:
        kept = top[:1]
    return np.sort(kept)


class Capped(BaseEstimator, RegressorMixin):
    """Fit `estimator` under the labelling run's row and feature caps.

    Feature selection happens on the rows given to `fit` and is remembered, so
    `predict` accepts the caller's full-width rows and narrows them itself. Row
    subsampling is seeded, so the same data yields the same subsample.
    """

    def __init__(self, estimator, max_rows: int | None = MAX_ROWS,
                 max_features: int | None = MAX_FEATURES, random_state: int = 0):
        self.estimator = estimator
        self.max_rows = max_rows
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]

        if self.max_features is not None:
            self.features_ = select_features(X, y, self.max_features)
        else:
            self.features_ = np.arange(X.shape[1])
        X_sel = X[:, self.features_]

        self.n_rows_used_ = len(y)
        if self.max_rows is not None and len(y) > self.max_rows:
            idx = np.random.default_rng(self.random_state).choice(
                len(y), size=self.max_rows, replace=False)
            X_sel, y = X_sel[idx], y[idx]
            self.n_rows_used_ = int(self.max_rows)

        self.estimator_ = self.estimator.fit(X_sel, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected rows of {self.n_features_in_} features, the width "
                f"fitted on, and got {X.shape}. Pass full-width rows: the "
                "feature cap is applied here, not by the caller.")
        return self.estimator_.predict(X[:, self.features_])

    def __getattr__(self, name):
        """Read through to the wrapped estimator, so the wrapper stays out of
        the way: `model.variant` and `model.solver` mean what they always did.

        Only consulted when normal lookup fails, so the wrapper's own
        attributes win, and never before `fit`, which keeps `check_is_fitted`
        honest and the delegation non-recursive.
        """
        if name.startswith("_") or "estimator_" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["estimator_"], name)

    def capped(self, n_rows: int, n_features: int) -> str:
        """What the caps actually removed here, or "" when they did nothing."""
        check_is_fitted(self, "estimator_")
        parts = []
        if self.n_rows_used_ < n_rows:
            parts.append(f"{self.n_rows_used_} of {n_rows} rows")
        if len(self.features_) < n_features:
            parts.append(f"{len(self.features_)} of {n_features} features")
        if not parts:
            return ""
        return ("fitted on " + " and ".join(parts)
                + ", the caps the labelling run used")
