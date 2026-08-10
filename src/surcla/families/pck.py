"""PC-Kriging: an orthogonal PCE trend plus a Gaussian process on its residuals.

The fit is sequential. A polynomial chaos expansion (Legendre or Hermite basis,
distribution inferred per feature) is fitted by ridge regression on the full
design matrix and captures the global polynomial structure; a GP then fits what
the trend leaves behind, on standardized inputs, adding local correction.
Prediction is the sum of the two parts. Kernel hyperparameters are fixed, not
optimized: one unit length scale per feature and unit signal variance, so the
GP contributes a fixed-bandwidth interpolant of the residuals with `gp_alpha`
as nugget.

Four registry variants cross the trend degree with the kernel: PCK_2G, PCK_2M,
PCK_3G, PCK_3M, where G is the Gaussian RBF kernel and M is Matern nu = 5/2.

Numerical parity contract: `PCKrigingSurrogate` is a port of the research
implementation that produced the corpus labels, down to the degree cap, the
scaling, the nugget and the restart count. tests/test_family_pck.py pins it.
"""

import warnings

import numpy as np
from scipy.special import comb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .pce import build_pce_design_matrix, infer_distribution


class PCKrigingSurrogate(BaseEstimator, RegressorMixin):
    """PC-Kriging: orthogonal PCE trend, Gaussian process on the residuals.

    Parameters
    ----------
    degree : int
        PCE polynomial degree for the trend component.
    kernel_type : str
        Kriging kernel: "gaussian" (RBF) or "matern" (Matern nu = 2.5).
    scale_X : str
        Input scaling for the kriging component: "standard" or "none".
    pce_alpha : float
        Ridge regularization strength for the PCE trend fit.
    gp_alpha : float
        Noise variance added to the kriging kernel diagonal.
    normalize_y : bool
        Normalize the kriging targets (the residuals) before fitting.
    n_restarts_optimizer : int
        Restarts for the kriging hyperparameter search.
    significance : float
        p-value threshold for the per-feature distribution inference.
    max_terms : int or None
        Cap on PCE basis size; the degree is reduced until the basis fits.
    random_state : int
        Seed passed to the Gaussian process.
    """

    def __init__(
        self,
        degree=2,
        kernel_type="gaussian",
        scale_X="standard",
        pce_alpha=1e-6,
        gp_alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=2,
        significance=0.05,
        max_terms=500,
        random_state=42,
    ):
        self.degree = degree
        self.kernel_type = kernel_type
        self.scale_X = scale_X
        self.pce_alpha = pce_alpha
        self.gp_alpha = gp_alpha
        self.normalize_y = normalize_y
        self.n_restarts_optimizer = n_restarts_optimizer
        self.significance = significance
        self.max_terms = max_terms
        self.random_state = random_state

    def _safe_degree(self, n_features):
        """Reduce the PCE degree until the total-degree basis fits max_terms."""
        degree = self.degree
        if self.max_terms is not None:
            while (degree > 1
                   and int(comb(n_features + degree, degree)) > self.max_terms):
                degree -= 1
            if degree != self.degree:
                warnings.warn(
                    f"PCKrigingSurrogate: PCE degree reduced from {self.degree} "
                    f"to {degree} to stay within max_terms={self.max_terms}.",
                    UserWarning, stacklevel=3,
                )
        return degree

    def _build_gp_kernel(self, n_features):
        """Kriging kernel with one fixed length scale per feature."""
        length_scale = np.ones(n_features)
        if self.kernel_type == "gaussian":
            base = RBF(length_scale=length_scale, length_scale_bounds="fixed")
        elif self.kernel_type == "matern":
            base = Matern(length_scale=length_scale, nu=2.5,
                          length_scale_bounds="fixed")
        else:
            raise ValueError(
                f"Unknown kernel_type '{self.kernel_type}'. "
                "Use 'gaussian' or 'matern'."
            )
        return ConstantKernel(1.0, constant_value_bounds="fixed") * base

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.dist_types_ = []
        self.dist_params_ = []
        for j in range(n_features):
            dist_type, params = infer_distribution(X[:, j],
                                                   significance=self.significance)
            self.dist_types_.append(dist_type)
            self.dist_params_.append(params)

        self.effective_degree_ = self._safe_degree(n_features)
        Phi = build_pce_design_matrix(
            X, self.effective_degree_, self.dist_types_, self.dist_params_
        )
        self.n_pce_terms_ = Phi.shape[1]

        self.pce_ = Ridge(alpha=self.pce_alpha, fit_intercept=False)
        self.pce_.fit(Phi, y)
        residuals = y - self.pce_.predict(Phi)

        self.scaler_ = None
        X_scaled = X.copy()
        if self.scale_X == "standard":
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        elif self.scale_X != "none":
            raise ValueError(
                f"Unknown scale_X: '{self.scale_X}'. Use 'standard' or 'none'."
            )

        self.gp_ = GaussianProcessRegressor(
            kernel=self._build_gp_kernel(n_features),
            alpha=self.gp_alpha,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
        )
        self.gp_.fit(X_scaled, residuals)
        return self

    def _trend_and_scaled_inputs(self, X):
        Phi = build_pce_design_matrix(
            X, self.effective_degree_, self.dist_types_, self.dist_params_
        )
        X_scaled = X.copy()
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        return self.pce_.predict(Phi), X_scaled

    def predict(self, X):
        check_is_fitted(self, ["pce_", "gp_", "n_features_in_"])
        X = np.asarray(X, dtype=float)
        y_pce, X_scaled = self._trend_and_scaled_inputs(X)
        return y_pce + self.gp_.predict(X_scaled)

    def predict_with_std(self, X):
        """Prediction and the kriging standard deviation of the residual part."""
        check_is_fitted(self, ["pce_", "gp_", "n_features_in_"])
        X = np.asarray(X, dtype=float)
        y_pce, X_scaled = self._trend_and_scaled_inputs(X)
        y_residual, y_std = self.gp_.predict(X_scaled, return_std=True)
        return y_pce + y_residual, y_std


# Trend degree and kernel behind each registry key; the remaining settings are
# shared and applied by make().
VARIANTS = {
    "PCK_2G": {"degree": 2, "kernel_type": "gaussian"},
    "PCK_2M": {"degree": 2, "kernel_type": "matern"},
    "PCK_3G": {"degree": 3, "kernel_type": "gaussian"},
    "PCK_3M": {"degree": 3, "kernel_type": "matern"},
}


def make(variant: str, random_state: int = 42) -> PCKrigingSurrogate:
    """Build one registry variant with the settings used to label the corpus."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown PCK variant '{variant}'; "
                         f"choose from {sorted(VARIANTS)}")
    return PCKrigingSurrogate(scale_X="standard", n_restarts_optimizer=2,
                              random_state=random_state, **VARIANTS[variant])
