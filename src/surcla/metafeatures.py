"""Hand-crafted metafeatures aimed at the x -> y relationship.

51 descriptors per dataset: per-feature statistics of each (x_i, y) pair
(rank correlations, mutual information, isotonic-regression residuals,
y-profile bins over x-quantiles) aggregated as mean/std/max, plus n and d.

Numerical parity contract: `xi_y_relationship` and `manual_metafeature_vector`
are byte-identical ports of the research extractor that built the training
tables. Any change to them is a new schema version and requires new artifact
tables (SCHEMA_VERSION below).
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, kurtosis, skew, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.isotonic import IsotonicRegression

# Metafeature schema of the corpus_v5_full training tables.
SCHEMA_VERSION = "manual-v5"

# Descriptors are estimated on at most this many rows (deterministic
# subsample), matching how the training tables were built.
MAX_ROWS_DEFAULT = 2000

_ARRAY_KEYS = ("y_mean5", "y_std5")


def xi_y_relationship(x: np.ndarray, y: np.ndarray) -> dict:
    """Per-feature descriptors for one (xi, y) pair."""
    out = {
        "mi": 0.0, "spearman_r": 0.0, "kendall_t": 0.0,
        "y_mean5": np.zeros(5), "y_std5": np.zeros(5),
        "iso_residual": 0.0, "n_sign_changes": 0, "tail_diff": 0.0, "quad_gain": 0.0,
        "y_kurtosis": float(kurtosis(y, nan_policy="omit")),
        "y_skew": float(skew(y, nan_policy="omit")),
        "mean_x": float(x.mean()), "std_x": 0.0, "skew_x": 0.0, "x_kurtosis": 0.0,
    }
    if x.std() == 0:
        return out

    n_bins = 5
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.digitize(x, edges[1:-1])
    y_mean5 = np.array([y[bin_idx == b].mean() if (bin_idx == b).any() else 0.0 for b in range(n_bins)])
    y_std5 = np.array([y[bin_idx == b].std() if (bin_idx == b).sum() > 1 else 0.0 for b in range(n_bins)])

    sp_r, _ = spearmanr(x, y)
    ke_t, _ = kendalltau(x, y)
    iso = IsotonicRegression().fit(x, y)
    lin_res = y - np.polyval(np.polyfit(x, y, 1), x)
    quad_res = y - np.polyval(np.polyfit(x, y, 2), x)
    slope_ch = np.diff(y_mean5)

    out.update({
        "mi":             float(mutual_info_regression(x.reshape(-1, 1), y, random_state=42)[0]),
        "spearman_r":     abs(float(sp_r)) if not np.isnan(sp_r) else 0.0,
        "kendall_t":      abs(float(ke_t)) if not np.isnan(ke_t) else 0.0,
        "y_mean5":        y_mean5,
        "y_std5":         y_std5,
        "iso_residual":   float(np.mean((y - iso.predict(x)) ** 2)),
        "n_sign_changes": int(np.sum(np.diff(np.sign(slope_ch)) != 0)),
        "tail_diff":      float(y_mean5[-1] - y_mean5[0]),
        "quad_gain":      float(np.mean(lin_res ** 2) - np.mean(quad_res ** 2)),
        "std_x":          float(x.std()),
        "skew_x":         float(skew(x, nan_policy="omit")),
        "x_kurtosis":     float(kurtosis(x, nan_policy="omit")),
    })
    return out


def manual_metafeature_vector(X: np.ndarray, y: np.ndarray) -> dict:
    """Aggregate per-feature descriptors into one fixed-length metafeature dict."""
    pf = pd.DataFrame([xi_y_relationship(X[:, j], y) for j in range(X.shape[1])])

    scalar_keys = [k for k in pf.columns if k not in _ARRAY_KEYS]
    stats = pf[scalar_keys].agg(["mean", "std", "max"])
    meta = {f"{key}__{stat}": float(stats.loc[stat, key]) for key in scalar_keys for stat in stats.index}

    for arr_key in _ARRAY_KEYS:
        profile = np.stack(pf[arr_key]).mean(axis=0)
        meta.update({f"{arr_key}__bin{i}": float(v) for i, v in enumerate(profile)})

    meta["n_samples"] = float(X.shape[0])
    meta["n_features"] = float(X.shape[1])
    return meta


def extract(X: np.ndarray, y: np.ndarray, max_rows: int = MAX_ROWS_DEFAULT,
            seed: int = 0) -> dict:
    """Query-time extraction: deterministic row cap, then the fixed vector.

    Mirrors how the training tables were built: descriptors (including
    `n_samples`) are computed on at most `max_rows` rows, subsampled with a
    seeded generator; the full training-set size travels separately as
    `n_train`, the decoder's always-included size feature.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2 or len(y) != X.shape[0]:
        raise ValueError(f"expected X (n, d) and y (n,); got {X.shape}, {y.shape}")
    n_full = len(y)
    if n_full > max_rows:
        sub = np.random.default_rng(seed).choice(n_full, size=max_rows, replace=False)
        X, y = X[sub], y[sub]
    meta = manual_metafeature_vector(X, y)
    meta["n_train"] = float(n_full)
    return meta
