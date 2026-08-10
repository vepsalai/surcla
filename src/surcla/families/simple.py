"""Constructors for the six families whose warm start is a parameter dict.

RF, LGBM and XGB were tuned by FLAML while the corpus was labelled, so their
mined configurations are FLAML *search-space* configurations, not native
constructor arguments. Each constructor below performs the translation FLAML
performed internally and returns an unfitted estimator. The three linear
families need no translation, only the fixed regularization the labelling run
used: Ridge alpha 1.0, Lasso alpha 0.01.

FLAML -> native, every one of them a silent-degradation trap if skipped:

    RF    max_leaves  -> max_leaf_nodes
    LGBM  log_max_bin -> max_bin = 2**log_max_bin - 1   (log2 scale: 8 -> 255)
    XGB   max_leaves  -> max_leaves, but only together with max_depth=0 and
                         grow_policy="lossguide"; XGBoost grows depth-wise to
                         max_depth=6 by default and then ignores max_leaves.

Keys a constructor does not recognize are dropped with a warning instead of
being forwarded, so a configuration from a newer artifact table cannot crash
a fit.

The translation is exact, the refit is not quite: warm-starting one of these
configurations reproduces a recorded corpus R² to about 1e-3 for RF and 5e-3
for XGB, not to the digit. Neither gap is a translation error. FLAML shuffled
the training rows before fitting, which moves the bootstrap draws, and XGB's
column subsampling makes the tree structure depend on the seed stream; on a
500-row test problem, shuffling moved RF by 2e-3 R² and sweeping the XGB seed
spanned 8e-3. The package leaves the caller's row order alone rather than
reproduce a shuffle for the sake of the digits.

lightgbm and xgboost are optional dependencies, imported when called.
"""

from __future__ import annotations

import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge

_LR_KEYS = ("fit_intercept",)
_RIDGE_KEYS = ("alpha", "fit_intercept", "solver")
_LASSO_KEYS = ("alpha", "fit_intercept", "max_iter", "tol", "selection")
_RF_KEYS = ("n_estimators", "max_leaves", "max_features")
_LGBM_KEYS = ("n_estimators", "num_leaves", "min_child_samples", "learning_rate",
              "log_max_bin", "colsample_bytree", "reg_alpha", "reg_lambda")
_XGB_KEYS = ("n_estimators", "max_leaves", "min_child_weight", "learning_rate",
             "subsample", "colsample_bylevel", "colsample_bytree",
             "reg_alpha", "reg_lambda")

# FLAML's own default seed for its random forest: keeping it makes a
# warm-started fit reproduce the labelled one.
RF_RANDOM_STATE = 12032022


def _accepted(config, allowed, family) -> dict:
    """Keep the configuration keys `family` understands; warn about the rest."""
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise TypeError(f"{family}: config must be a dict of parameters, got "
                        f"{type(config).__name__}")
    unknown = sorted(set(config) - set(allowed))
    if unknown:
        warnings.warn(f"{family}: ignoring unsupported config keys {unknown}",
                      stacklevel=3)
    return {k: v for k, v in config.items() if k in allowed}


def linear_regression(config=None) -> LinearRegression:
    """Ordinary least squares. Nothing was tuned; only `fit_intercept` applies."""
    return LinearRegression(**_accepted(config, _LR_KEYS, "LR"))


def ridge(config=None) -> Ridge:
    """Ridge at the labelling run's alpha 1.0 unless the config overrides it."""
    params = {"alpha": 1.0}
    params.update(_accepted(config, _RIDGE_KEYS, "Ridge"))
    return Ridge(**params)


def lasso(config=None) -> Lasso:
    """Lasso at the labelling run's alpha 0.01 and its 10k iteration cap.

    The sklearn default of 1000 iterations leaves coordinate descent short of
    convergence at this alpha on many corpus datasets, so the cap is part of
    the labelled configuration rather than a detail.
    """
    params = {"alpha": 0.01, "max_iter": 10_000}
    params.update(_accepted(config, _LASSO_KEYS, "Lasso"))
    return Lasso(**params)


def random_forest(config=None, random_state=RF_RANDOM_STATE
                  ) -> RandomForestRegressor:
    """Random forest from a FLAML config: `max_leaves` -> `max_leaf_nodes`.

    `n_estimators` and `max_features` mean the same thing on both sides and
    pass through; `max_features` is a fraction of the features in FLAML's
    search space, which sklearn accepts as a float.
    """
    params = _accepted(config, _RF_KEYS, "RF")
    if "max_leaves" in params:
        params["max_leaf_nodes"] = int(params.pop("max_leaves"))
    return RandomForestRegressor(n_jobs=-1, random_state=random_state, **params)


def lgbm(config=None):
    """LightGBM from a FLAML config: `log_max_bin` -> `max_bin = 2**k - 1`.

    FLAML searches the feature-binning resolution on a log2 scale, so a stored
    `log_max_bin` of 8 means 255 bins, not 8. Passing the raw value through
    would quantize every feature into a handful of bins. Every other key is
    LightGBM's own.
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("the LGBM family needs lightgbm: "
                          "pip install 'surcla[lgbm]'") from exc
    params = _accepted(config, _LGBM_KEYS, "LGBM")
    if "log_max_bin" in params:
        params["max_bin"] = 2 ** int(params.pop("log_max_bin")) - 1
    return LGBMRegressor(verbose=-1, **params)


def xgb(config=None):
    """XGBoost from a FLAML config, grown leaf-wise so `max_leaves` bites.

    FLAML tuned XGBoost through its unlimited-depth estimator, which sets
    `max_depth=0`, `grow_policy="lossguide"` and the histogram tree method
    before handing the config to `XGBRegressor`. Without those three the
    default depth-wise growth caps trees at depth 6 and ignores `max_leaves`
    entirely, fitting a different model from the labelled one without saying
    so. The remaining keys are XGBoost's own.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("the XGB family needs xgboost: "
                          "pip install 'surcla[xgb]'") from exc
    params = _accepted(config, _XGB_KEYS, "XGB")
    return XGBRegressor(max_depth=0, grow_policy="lossguide",
                        tree_method="hist", verbosity=0, **params)
