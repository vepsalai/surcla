"""MLP surrogate: warm-started architecture, standardized inputs, sklearn fit.

Port of the research `AutoEmulateMLP` fit path. There, AutoEmulate's torch MLP
was cross-validated to choose `layer_dims`, `lr`, `epochs` and `batch_size`,
and those four numbers were handed to a sklearn `MLPRegressor` that did the
actual fit. This package ships the outcome of that tuning instead of the
tuner: pass the configuration `warmstart.lookup("MLP", n, d)` returns and the
fit path below is the labelling one.

Configuration key -> `MLPRegressor` argument:

    layer_dims -> hidden_layer_sizes    list of ints, e.g. [64, 32, 16]
    lr         -> learning_rate_init
    epochs     -> max_iter
    batch_size -> batch_size

Everything else is fixed as it was at labelling time: ReLU activation, Adam,
and a `StandardScaler` in front, which MLP convergence depends on.

The parity holds for a complete configuration at the default seed, which is
every configuration the warm-start table can hand you. Two paths outside that
differ from the research class on purpose. Its network seed was hardcoded at
42 and its `random_state` argument seeded the HPO subsample instead; with no
HPO left to seed, `random_state` here seeds the network, so a seed other than
42 gives a model the labelling run never fitted. And a configuration missing
`lr`, `epochs` or `batch_size` made the research class build an invalid
`MLPRegressor` and crash; here the sklearn defaults fill in, which fits a
different model rather than none.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

# Rows above which the labelling run tuned on a random subsample. Past it the
# stored epoch count was chosen against a smaller training set and no longer
# transfers, so early stopping takes over.
HPO_SAMPLE_SIZE = 5000


def config_to_sklearn(config: dict) -> dict:
    """Map a mined MLP configuration to `MLPRegressor` keyword arguments."""
    kwargs = {}
    if "layer_dims" in config:
        kwargs["hidden_layer_sizes"] = tuple(config["layer_dims"])
    if "lr" in config:
        kwargs["learning_rate_init"] = float(config["lr"])
    if "epochs" in config:
        kwargs["max_iter"] = int(config["epochs"])
    if "batch_size" in config:
        kwargs["batch_size"] = int(config["batch_size"])
    return kwargs


class MLPSurrogate(BaseEstimator, RegressorMixin):
    """Standardize, then fit a sklearn MLP at a warm-started configuration.

    Parameters
    ----------
    config : dict or None
        Mined configuration, `{"layer_dims", "lr", "epochs", "batch_size"}`.
        Keys left out fall back to the sklearn defaults, except the
        architecture, which falls back to `fallback_hidden_layers`.
    fallback_hidden_layers : tuple
        Architecture used when the configuration carries no `layer_dims`.
    hpo_sample_size : int
        Training size above which validation-based early stopping replaces the
        configuration's epoch count (see `HPO_SAMPLE_SIZE`).
    random_state : int
        Seed of the network's weight initialization and batch shuffling. The
        default reproduces the labelling run.
    """

    def __init__(self, config=None, fallback_hidden_layers=(64, 32),
                 hpo_sample_size=HPO_SAMPLE_SIZE, random_state=42):
        self.config = config
        self.fallback_hidden_layers = fallback_hidden_layers
        self.hpo_sample_size = hpo_sample_size
        self.random_state = random_state

    def fit(self, X, y):
        if self.config is not None and not isinstance(self.config, dict):
            raise TypeError("config must be a dict of MLP parameters, got "
                            f"{type(self.config).__name__}")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        self.config_ = dict(self.config or {})
        kwargs = config_to_sklearn(self.config_)
        kwargs.setdefault("hidden_layer_sizes", tuple(self.fallback_hidden_layers))
        if len(y) > self.hpo_sample_size:
            kwargs["early_stopping"] = True

        mlp = MLPRegressor(activation="relu", solver="adam",
                           random_state=self.random_state, **kwargs)
        self.model_ = make_pipeline(StandardScaler(), mlp)
        self.model_.fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["model_"])
        return self.model_.predict(np.asarray(X, dtype=np.float32))
