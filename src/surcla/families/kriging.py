"""Kriging: a Gaussian process with an optional linear trend, four variants.

A Kriging model is a GP over the residual of a deterministic trend. Two axes
vary across the corpus fits and are exposed here as a variant tag:

  trend   O = ordinary, a constant mean absorbed by the GP itself
          L = linear, an ordinary-least-squares plane fitted on the raw inputs
              and subtracted before the GP sees the data, then added back at
              predict time
  kernel  G = squared exponential (RBF), infinitely smooth
          M = Matern with nu = 2.5, twice differentiable, tolerant of kinks

Both kernels are separable (ARD): one length scale per input dimension, so the
marginal-likelihood optimizer learns which inputs matter. The trailing S in the
tag records that. Hence Kriging_OGS, Kriging_OMS, Kriging_LGS, Kriging_LMS.

Numerical parity contract: defaults, kernel bounds, scaling order and the
detrend-on-raw-X choice reproduce the research fits that produced the corpus
labels. Changing any of them invalidates comparison against those labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

VARIANTS = ("Kriging_OGS", "Kriging_OMS", "Kriging_LGS", "Kriging_LMS")

# Broad but finite, so the optimizer can adapt to the dataset without wandering
# into ranges where the marginal likelihood is flat.
LENGTH_SCALE_BOUNDS = (1e-3, 1e4)
CONSTANT_VALUE_BOUNDS = (1e-3, 1e3)


class KrigingSurrogate(BaseEstimator, RegressorMixin):
    """Kriging surrogate with trend and kernel variants.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`. Read as Kriging_[O|L][G|M]S: ordinary or linear
        trend, Gaussian (RBF) or Matern kernel, separable anisotropy.
    scale_X : str
        Input scaling before the GP: "none", "standard" (default) or "minmax".
        A linear trend is always fitted on the unscaled inputs.
    alpha : float
        Noise variance added to the kernel diagonal (default 1e-8).
    normalize_y : bool
        Center and scale the GP target before fitting (default True).
    n_restarts_optimizer : int
        Restarts of the marginal-likelihood optimizer (default 8).
    random_state : int
        Seed for those restarts (default 42).
    """

    def __init__(
        self,
        variant="Kriging_OGS",
        scale_X="standard",
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=8,
        random_state=42,
    ):
        self.variant = variant
        self.scale_X = scale_X
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state

    def _parse_variant(self):
        """Split the variant tag into trend and kernel type."""
        tag = str(self.variant).upper()
        if not tag.startswith("KRIGING_") or len(tag) < 10:
            raise ValueError(
                f"Unknown variant '{self.variant}'. "
                "Use one of: Kriging_OGS, Kriging_OMS, Kriging_LGS, Kriging_LMS."
            )

        trend_type = {"O": "ordinary", "L": "linear"}.get(tag[8])
        kernel_type = {"G": "gaussian", "M": "matern"}.get(tag[9])
        if trend_type is None or kernel_type is None:
            raise ValueError(
                f"Invalid variant '{self.variant}'. "
                "Valid format: Kriging_[O|L][G|M]S."
            )
        return trend_type, kernel_type

    def _build_kernel(self, n_features, kernel_type):
        """Amplitude times a separable (one length scale per input) kernel."""
        length_scale = np.ones(n_features)
        if kernel_type == "gaussian":
            base = RBF(length_scale=length_scale,
                       length_scale_bounds=LENGTH_SCALE_BOUNDS)
        elif kernel_type == "matern":
            base = Matern(length_scale=length_scale, nu=2.5,
                          length_scale_bounds=LENGTH_SCALE_BOUNDS)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        return ConstantKernel(1.0,
                              constant_value_bounds=CONSTANT_VALUE_BOUNDS) * base

    def _scale(self, X):
        return X if self.scaler_ is None else self.scaler_.transform(X)

    def fit(self, X, y):
        """Fit the trend, then the GP on the residual."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_features = X.shape[1]
        self.n_features_in_ = n_features

        self.scaler_ = None
        if self.scale_X == "standard":
            self.scaler_ = StandardScaler()
        elif self.scale_X == "minmax":
            self.scaler_ = MinMaxScaler()
        elif self.scale_X != "none":
            raise ValueError(f"Unknown scale_X: {self.scale_X}")
        X_scaled = X.copy() if self.scaler_ is None else self.scaler_.fit_transform(X)

        trend_type, kernel_type = self._parse_variant()
        if trend_type == "linear":
            # Fitted on raw X, so the trend stays interpretable and the GP is
            # left with a zero-mean residual regardless of the input scaling.
            self.trend_ = LinearRegression(fit_intercept=True).fit(X, y)
            y_gp = y - self.trend_.predict(X)
        else:
            self.trend_ = None
            y_gp = y.copy()

        self.gp_ = GaussianProcessRegressor(
            kernel=self._build_kernel(n_features, kernel_type),
            alpha=self.alpha,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
        )
        self.gp_.fit(X_scaled, y_gp)
        return self

    def predict(self, X):
        """Posterior mean, with the trend added back."""
        check_is_fitted(self, ["gp_", "n_features_in_"])
        X = np.asarray(X, dtype=float)
        y_pred = self.gp_.predict(self._scale(X))
        if self.trend_ is not None:
            y_pred = y_pred + self.trend_.predict(X)
        return y_pred

    def predict_with_std(self, X):
        """Posterior mean and standard deviation.

        The trend is deterministic, so it shifts the mean and leaves the
        standard deviation untouched.
        """
        check_is_fitted(self, ["gp_", "n_features_in_"])
        X = np.asarray(X, dtype=float)
        y_pred, y_std = self.gp_.predict(self._scale(X), return_std=True)
        if self.trend_ is not None:
            y_pred = y_pred + self.trend_.predict(X)
        return y_pred, y_std


def make(variant: str, random_state: int = 42) -> KrigingSurrogate:
    """Build one registry variant with the settings used to label the corpus.

    The labelling run capped the marginal-likelihood restarts at two, below the
    class default, to keep the corpus affordable; a warm-started fit that used
    eight would not be the fit the corpus labels describe.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown Kriging variant '{variant}'; "
                         f"choose from {list(VARIANTS)}")
    return KrigingSurrogate(variant=variant, scale_X="standard",
                            n_restarts_optimizer=2, random_state=random_state)
