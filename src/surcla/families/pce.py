"""Polynomial chaos expansion with per-feature basis inference.

Every input feature is tested for normality and uniformity; the winning
marginal picks the orthogonal univariate family (probabilists' Hermite for
normal, Legendre for uniform, each evaluated after mapping the feature onto
the family's natural support). Tensor products of those univariate
polynomials, truncated by total degree with an optional hyperbolic q-norm and
an interaction-order cap, form the design matrix; a linear solver (ridge,
LARS, or orthogonal matching pursuit) fits the coefficients on it. With
`auto_select` the degree and truncation come from a K-fold search over the
candidate grid instead of the constructor arguments.

Numerical parity contract: a port of the research implementation that
produced the corpus PCE labels. `infer_distribution`, `generate_multi_indices`
and `build_pce_design_matrix` keep their research names because the PCK
surrogate builds its polynomial trend on them.
"""

import warnings
from itertools import combinations

import numpy as np
from numpy.polynomial.hermite_e import hermeval
from numpy.polynomial.legendre import legval
from scipy import stats
from scipy.special import comb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import (
    LassoLarsCV,
    LinearRegression,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    Ridge,
)
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted


def infer_distribution(x, significance=0.05):
    """Decide whether a 1-D sample reads as normal or uniform.

    Shapiro-Wilk against a Wilks-style KS test on the observed range; when
    both or neither pass at `significance` the larger p-value wins.

    Returns
    -------
    dist_type : str, "normal" or "uniform"
    params : dict, {"mu", "sigma"} or {"a", "b"}
    """
    x = np.asarray(x, dtype=float)

    _, p_normal = stats.shapiro(x)
    a, b = x.min(), x.max()
    _, p_uniform = stats.kstest(x, "uniform", args=(a, b - a))

    normal_ok = p_normal > significance
    uniform_ok = p_uniform > significance

    if normal_ok and not uniform_ok:
        dist_type = "normal"
    elif uniform_ok and not normal_ok:
        dist_type = "uniform"
    else:
        dist_type = "normal" if p_normal >= p_uniform else "uniform"

    if dist_type == "normal":
        params = {"mu": x.mean(), "sigma": x.std()}
    else:
        params = {"a": a, "b": b}

    return dist_type, params


def _legendre_basis(x, max_degree):
    """Legendre polynomials P_0 ... P_max_degree at x, with x in [-1, 1]."""
    out = []
    for d in range(max_degree + 1):
        coeffs = np.zeros(d + 1)
        coeffs[d] = 1.0
        out.append(legval(x, coeffs))
    return out


def _hermite_basis(x, max_degree):
    """Probabilists' Hermite polynomials He_0 ... He_max_degree at x."""
    out = []
    for d in range(max_degree + 1):
        coeffs = np.zeros(d + 1)
        coeffs[d] = 1.0
        out.append(hermeval(x, coeffs))
    return out


def _compositions_leq(total_max, k):
    """All k-tuples of degrees >= 1 summing to at most total_max, lexicographic."""
    if k == 0:
        yield ()
        return
    for first in range(1, total_max - k + 2):
        for rest in _compositions_leq(total_max - first, k - 1):
            yield (first,) + rest


def generate_multi_indices(n_features, degree, q_norm=1.0, max_interaction=None):
    """Multi-indices under total-degree or hyperbolic truncation.

    Admissible indices are enumerated directly (pick the interacting features,
    then their degree compositions), so cost scales with the number of terms
    kept rather than with (degree + 1) ** n_features, which is infeasible
    beyond roughly 15 features.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if q_norm <= 0:
        raise ValueError("q_norm must be positive")
    if q_norm > 1.0:
        raise ValueError("q_norm > 1 is not supported")

    # For q_norm <= 1 the q-ball lies inside the total-degree simplex, so
    # every admissible index has at most `degree` interacting features.
    max_k = degree if max_interaction is None else min(int(max_interaction), degree)
    max_k = min(max_k, n_features)

    multi_indices = [(0,) * n_features]
    for k in range(1, max_k + 1):
        for combo in combinations(range(n_features), k):
            for degs in _compositions_leq(degree, k):
                if q_norm != 1.0:
                    q_sum = sum(d ** q_norm for d in degs)
                    if q_sum ** (1.0 / q_norm) > degree + 1e-12:
                        continue
                idx = [0] * n_features
                for j, d in zip(combo, degs):
                    idx[j] = d
                multi_indices.append(tuple(idx))

    return multi_indices


def build_pce_design_matrix(
    X,
    degree,
    dist_types,
    dist_params,
    q_norm=1.0,
    max_interaction=None,
    multi_indices=None,
):
    """Tensor-product design matrix for the inferred per-feature bases.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    degree : int, maximum total polynomial degree
    dist_types : list[str], "normal" or "uniform" per feature
    dist_params : list[dict], as returned by `infer_distribution`
    q_norm : float, hyperbolic truncation parameter (1.0 = total degree)
    max_interaction : int or None, cap on interacting variables per term
    multi_indices : list[tuple[int]] or None, precomputed index set

    Returns
    -------
    Phi : ndarray of shape (n_samples, n_terms)
    """
    n_samples, n_features = X.shape

    univariate_bases = []
    for j, (dist, params) in enumerate(zip(dist_types, dist_params)):
        xj = X[:, j]
        if dist == "uniform":
            a, b = params["a"], params["b"]
            xj = 2.0 * (xj - a) / (b - a) - 1.0        # [a, b] -> [-1, 1]
            univariate_bases.append(_legendre_basis(xj, degree))
        else:
            mu, sigma = params["mu"], params.get("sigma", 1.0)
            xj = (xj - mu) / (sigma if sigma > 0 else 1.0)
            univariate_bases.append(_hermite_basis(xj, degree))

    if multi_indices is None:
        multi_indices = generate_multi_indices(
            n_features,
            degree,
            q_norm=q_norm,
            max_interaction=max_interaction,
        )

    Phi = []
    for degree_combo in multi_indices:
        term = np.ones(n_samples)
        for j, d in enumerate(degree_combo):
            term = term * univariate_bases[j][d]
        Phi.append(term)

    return np.column_stack(Phi)


class PCESurrogate(BaseEstimator, RegressorMixin):
    """Orthogonal polynomial chaos expansion, fitted by a linear solver.

    Parameters
    ----------
    degree : int
        Maximum total polynomial degree (default 2).
    alpha : float
        Ridge regularization strength (default 1e-6).
    significance : float
        p-value threshold for the distribution test (default 0.05).
    max_terms : int or None
        Cap on basis size; the degree is reduced until the expansion fits.
        None disables the cap.
    solver : str
        One of {"ridge", "lars", "omp"} (default "ridge").
    cv : int
        Cross-validation folds for LARS/OMP-CV and for auto-selection.
    max_nonzero : int or None
        Fixed sparsity for OMP; None uses OMP-CV.
    refit_selected : bool
        Refit the active terms with OLS after sparse selection.
    q_norm : float
        Hyperbolic truncation parameter (default 1.0).
    max_interaction : int or None
        Maximum interaction order in the basis.
    auto_select : bool
        Choose degree and truncation by CV over the candidate sets.
    degree_candidates : sequence[int] or None
        Candidate degrees; None means 1..degree when `auto_select`, else
        just `degree`.
    q_norm_candidates : sequence[float] or None
        Candidate q values; None means just `q_norm`.
    max_interaction_candidates : sequence[int | None] or None
        Candidate interaction caps; None means just `max_interaction`.
    """

    def __init__(
        self,
        degree=2,
        alpha=1e-6,
        significance=0.05,
        max_terms=500,
        solver="ridge",
        cv=5,
        max_nonzero=None,
        refit_selected=True,
        q_norm=1.0,
        max_interaction=None,
        auto_select=False,
        degree_candidates=None,
        q_norm_candidates=None,
        max_interaction_candidates=None,
    ):
        self.degree = degree
        self.alpha = alpha
        self.significance = significance
        self.max_terms = max_terms
        self.solver = solver
        self.cv = cv
        self.max_nonzero = max_nonzero
        self.refit_selected = refit_selected
        self.q_norm = q_norm
        self.max_interaction = max_interaction
        self.auto_select = auto_select
        self.degree_candidates = degree_candidates
        self.q_norm_candidates = q_norm_candidates
        self.max_interaction_candidates = max_interaction_candidates

    def _n_terms(self, n_features, degree, q_norm=None, max_interaction=None):
        if q_norm is None:
            q_norm = self.q_norm
        if max_interaction is None:
            max_interaction = self.max_interaction
        if q_norm == 1.0 and max_interaction is None:
            return int(comb(n_features + degree, degree, exact=True))
        return len(
            generate_multi_indices(
                n_features,
                degree,
                q_norm=q_norm,
                max_interaction=max_interaction,
            )
        )

    def _safe_degree(self, n_features, degree, q_norm=None, max_interaction=None,
                     warn=True):
        """Reduce the degree until the expansion fits under `max_terms`."""
        requested_degree = degree
        if q_norm is None:
            q_norm = self.q_norm
        if max_interaction is None:
            max_interaction = self.max_interaction
        if self.max_terms is not None:
            while degree > 1 and self._n_terms(
                n_features, degree, q_norm, max_interaction
            ) > self.max_terms:
                degree -= 1
            if warn and degree != requested_degree:
                requested_terms = self._n_terms(
                    n_features, requested_degree, q_norm, max_interaction
                )
                warnings.warn(
                    f"PCESurrogate: degree reduced from {requested_degree} to "
                    f"{degree} because n_features={n_features} would give "
                    f"{requested_terms} terms (max_terms={self.max_terms}).",
                    UserWarning, stacklevel=3,
                )
        return degree

    def _effective_cv(self, n_samples):
        return max(2, min(int(self.cv), int(n_samples)))

    def _refit_on_active_terms(self, Phi, y, coef):
        active = np.abs(coef) > 1e-12
        if not self.refit_selected or not np.any(active):
            return coef

        refit = LinearRegression(fit_intercept=False)
        refit.fit(Phi[:, active], y)
        coef_refit = np.zeros(Phi.shape[1], dtype=float)
        coef_refit[active] = refit.coef_
        return coef_refit

    def _fit_coefficients(self, Phi, y):
        n_samples = Phi.shape[0]
        solver = str(self.solver).lower()

        if solver == "ridge":
            model = Ridge(alpha=self.alpha, fit_intercept=False)
            model.fit(Phi, y)
            coef = np.asarray(model.coef_, dtype=float)
        elif solver == "lars":
            cv = self._effective_cv(n_samples)
            model = LassoLarsCV(cv=cv, fit_intercept=False)
            model.fit(Phi, y)
            coef = np.asarray(model.coef_, dtype=float)
            coef = self._refit_on_active_terms(Phi, y, coef)
        elif solver == "omp":
            if self.max_nonzero is None:
                cv = self._effective_cv(n_samples)
                model = OrthogonalMatchingPursuitCV(cv=cv, fit_intercept=False)
            else:
                model = OrthogonalMatchingPursuit(
                    n_nonzero_coefs=int(self.max_nonzero),
                    fit_intercept=False,
                )
            model.fit(Phi, y)
            coef = np.asarray(model.coef_, dtype=float)
            coef = self._refit_on_active_terms(Phi, y, coef)
        else:
            raise ValueError(
                f"Unsupported solver '{self.solver}'. Use one of: ridge, lars, omp."
            )

        return model, coef

    def _candidate_configs(self, n_features):
        degree_candidates = self.degree_candidates
        if degree_candidates is None:
            if self.auto_select:
                degree_candidates = range(1, int(self.degree) + 1)
            else:
                degree_candidates = [int(self.degree)]

        q_candidates = self.q_norm_candidates
        if q_candidates is None:
            q_candidates = [self.q_norm]

        interaction_candidates = self.max_interaction_candidates
        if interaction_candidates is None:
            interaction_candidates = [self.max_interaction]

        configs = []
        seen = set()
        for degree in degree_candidates:
            for q_norm in q_candidates:
                for max_interaction in interaction_candidates:
                    effective_degree = self._safe_degree(
                        n_features,
                        int(degree),
                        q_norm=q_norm,
                        max_interaction=max_interaction,
                        warn=False,
                    )
                    key = (effective_degree, float(q_norm), max_interaction)
                    if key not in seen:
                        seen.add(key)
                        configs.append(key)
        return configs

    def _fit_config(self, X, y, degree, q_norm, max_interaction):
        dist_types = []
        dist_params = []
        for j in range(X.shape[1]):
            dist_type, params = infer_distribution(X[:, j], self.significance)
            dist_types.append(dist_type)
            dist_params.append(params)

        multi_indices = generate_multi_indices(
            X.shape[1],
            degree,
            q_norm=q_norm,
            max_interaction=max_interaction,
        )
        Phi = build_pce_design_matrix(
            X,
            degree,
            dist_types,
            dist_params,
            q_norm=q_norm,
            max_interaction=max_interaction,
            multi_indices=multi_indices,
        )
        model, coef = self._fit_coefficients(Phi, y)
        return {
            "model": model,
            "coef": coef,
            "dist_types": dist_types,
            "dist_params": dist_params,
            "multi_indices": multi_indices,
            "Phi": Phi,
        }

    def _select_config(self, X, y):
        n_samples, n_features = X.shape
        configs = self._candidate_configs(n_features)
        if len(configs) == 1:
            degree, q_norm, max_interaction = configs[0]
            return {
                "degree": degree,
                "q_norm": q_norm,
                "max_interaction": max_interaction,
                "cv_rmse": np.nan,
            }

        splitter = KFold(
            n_splits=self._effective_cv(n_samples),
            shuffle=True,
            random_state=42,
        )
        results = []
        for degree, q_norm, max_interaction in configs:
            fold_rmses = []
            for train_idx, val_idx in splitter.split(X):
                fit_result = self._fit_config(
                    X[train_idx],
                    y[train_idx],
                    degree,
                    q_norm,
                    max_interaction,
                )
                Phi_val = build_pce_design_matrix(
                    X[val_idx],
                    degree,
                    fit_result["dist_types"],
                    fit_result["dist_params"],
                    q_norm=q_norm,
                    max_interaction=max_interaction,
                    multi_indices=fit_result["multi_indices"],
                )
                y_pred = Phi_val @ fit_result["coef"]
                fold_rmses.append(
                    float(np.sqrt(np.mean((y[val_idx] - y_pred) ** 2)))
                )

            results.append({
                "degree": degree,
                "q_norm": q_norm,
                "max_interaction": max_interaction,
                "cv_rmse": float(np.mean(fold_rmses)),
            })

        results.sort(key=lambda item: (item["cv_rmse"], item["degree"], item["q_norm"]))
        self.selection_results_ = results
        return results[0]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape

        selected = self._select_config(X, y)
        self.degree_ = int(selected["degree"])
        self.q_norm_ = float(selected["q_norm"])
        self.max_interaction_ = selected["max_interaction"]
        self.cv_rmse_ = selected["cv_rmse"]

        fit_result = self._fit_config(
            X,
            y,
            self.degree_,
            self.q_norm_,
            self.max_interaction_,
        )

        self.dist_types_ = fit_result["dist_types"]
        self.dist_params_ = fit_result["dist_params"]
        self.multi_indices_ = fit_result["multi_indices"]
        self.model_ = fit_result["model"]
        self.coef_ = fit_result["coef"]

        self.active_terms_ = np.flatnonzero(np.abs(self.coef_) > 1e-12)
        self.n_active_terms_ = int(self.active_terms_.size)

        self.n_features_in_ = n_features
        self.n_terms_ = len(self.multi_indices_)
        return self

    def predict(self, X):
        check_is_fitted(
            self,
            ["coef_", "dist_types_", "dist_params_", "degree_", "multi_indices_"],
        )
        X = np.asarray(X, dtype=float)
        Phi = build_pce_design_matrix(
            X,
            self.degree_,
            self.dist_types_,
            self.dist_params_,
            q_norm=self.q_norm_,
            max_interaction=self.max_interaction_,
            multi_indices=self.multi_indices_,
        )
        return Phi @ self.coef_

    def get_feature_distributions(self):
        """Inferred marginal and its parameters, per feature."""
        check_is_fitted(self, ["dist_types_", "dist_params_"])
        return list(zip(self.dist_types_, self.dist_params_))

    def get_sobol_indices(self):
        """Per-term share of the expansion variance, summing to about 1.

        Meaningful only because the basis is orthogonal: each term's squared
        coefficient is its own variance contribution.
        """
        check_is_fitted(self, ["coef_"])
        c2 = self.coef_ ** 2
        total = c2.sum()
        return c2 / total if total > 0 else c2


# The three registry variants differ in the solver alone; degree, truncation
# and the candidate grids they auto-select over are shared.
VARIANTS = {
    "PCE_Ridge_D5": "ridge",
    "PCE_LARS_D5": "lars",
    "PCE_OMP_D5": "omp",
}


def make(variant: str) -> PCESurrogate:
    """Build one registry variant with the settings used to label the corpus.

    `alpha` is passed for all three even though only the ridge solver reads it,
    so the line stays a transcription of the registry entry.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown PCE variant '{variant}'; "
                         f"choose from {list(VARIANTS)}")
    return PCESurrogate(degree=5, max_terms=500, solver=VARIANTS[variant],
                        alpha=1e-6, cv=5, refit_selected=True, auto_select=True,
                        q_norm=0.75, max_interaction=3,
                        degree_candidates=[2, 3, 4, 5],
                        q_norm_candidates=[1.0, 0.75, 0.5],
                        max_interaction_candidates=[2, 3])
