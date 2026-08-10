"""Parity: the packaged PCE reproduces the research implementation.

Loads `surrogates/pce_surrogate.py` from the research checkout next to this
repo (override with SURCLA_RESEARCH_REPO) and compares predictions on seeded
synthetic data. Skips when that checkout is absent.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from surcla.families.pce import PCESurrogate

_default_repo = Path(__file__).resolve().parents[2] / "surcla"
RESEARCH = Path(os.environ.get("SURCLA_RESEARCH_REPO", _default_repo))
RESEARCH_PCE = RESEARCH / "surrogates" / "pce_surrogate.py"

pytestmark = pytest.mark.skipif(
    not RESEARCH_PCE.exists(),
    reason=f"research implementation not found at {RESEARCH_PCE}",
)


def _research_module():
    spec = importlib.util.spec_from_file_location("_research_pce", RESEARCH_PCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _data(n=200, d=3, seed=0):
    """Mixed marginals so both the Legendre and the Hermite branch fire."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.uniform(-1.0, 1.0, n),
        rng.normal(0.0, 1.5, n),
        rng.uniform(0.0, 3.0, n),
    ])[:, :d]
    y = (X[:, 0] ** 2 - 0.5 * X[:, 1] + 0.3 * X[:, 0] * X[:, 1]
         + np.sin(X[:, 2]) + 0.05 * rng.normal(size=n))
    return X, y


@pytest.mark.parametrize("solver", ["ridge", "lars", "omp"])
def test_predictions_match_research(solver):
    research = _research_module()
    X, y = _data()
    X_test, _ = _data(n=64, seed=1)

    kwargs = dict(degree=3, solver=solver)
    ours = PCESurrogate(**kwargs).fit(X, y).predict(X_test)
    theirs = research.PCESurrogate(**kwargs).fit(X, y).predict(X_test)

    assert np.abs(ours - theirs).max() < 1e-10


def test_auto_select_matches_research():
    """The CV search over degree / q-norm / interaction cap picks the same
    configuration and the same coefficients."""
    research = _research_module()
    X, y = _data()
    X_test, _ = _data(n=64, seed=1)

    kwargs = dict(
        degree=4,
        solver="lars",
        auto_select=True,
        q_norm_candidates=[0.7, 1.0],
        max_interaction_candidates=[1, 2, None],
    )
    ours = PCESurrogate(**kwargs).fit(X, y)
    theirs = research.PCESurrogate(**kwargs).fit(X, y)

    assert (ours.degree_, ours.q_norm_, ours.max_interaction_) == (
        theirs.degree_, theirs.q_norm_, theirs.max_interaction_)
    assert np.abs(ours.predict(X_test) - theirs.predict(X_test)).max() < 1e-10


def test_fixed_sparsity_omp_matches_research():
    research = _research_module()
    X, y = _data()
    X_test, _ = _data(n=64, seed=1)

    kwargs = dict(degree=3, solver="omp", max_nonzero=8, refit_selected=False)
    ours = PCESurrogate(**kwargs).fit(X, y)
    theirs = research.PCESurrogate(**kwargs).fit(X, y)

    assert ours.n_active_terms_ == theirs.n_active_terms_
    assert np.abs(ours.predict(X_test) - theirs.predict(X_test)).max() < 1e-10


def test_helpers_match_research():
    """The symbols the PCK port imports from here behave identically."""
    research = _research_module()
    X, _ = _data()

    for j in range(X.shape[1]):
        from surcla.families.pce import infer_distribution
        ours = infer_distribution(X[:, j])
        theirs = research.infer_distribution(X[:, j])
        assert ours[0] == theirs[0]
        assert ours[1].keys() == theirs[1].keys()
        assert all(abs(ours[1][k] - theirs[1][k]) < 1e-12 for k in ours[1])

    from surcla.families.pce import build_pce_design_matrix, generate_multi_indices
    idx = generate_multi_indices(3, 3, q_norm=0.7, max_interaction=2)
    assert idx == research.generate_multi_indices(3, 3, q_norm=0.7,
                                                  max_interaction=2)

    types = ["uniform", "normal", "uniform"]
    params = [{"a": -1.0, "b": 1.0}, {"mu": 0.0, "sigma": 1.5},
              {"a": 0.0, "b": 3.0}]
    Phi = build_pce_design_matrix(X, 3, types, params, multi_indices=idx)
    Phi_ref = research.build_pce_design_matrix(X, 3, types, params,
                                               multi_indices=idx)
    assert np.abs(Phi - Phi_ref).max() == 0.0


def test_max_terms_reduces_degree():
    """fit() reduces the degree silently (candidate screening passes
    warn=False); the warning belongs to a direct _safe_degree call."""
    research = _research_module()
    X, y = _data()

    ours = PCESurrogate(degree=6, max_terms=20).fit(X, y)
    theirs = research.PCESurrogate(degree=6, max_terms=20).fit(X, y)
    assert ours.degree_ == theirs.degree_ < 6
    assert ours.n_terms_ == theirs.n_terms_ <= 20

    with pytest.warns(UserWarning, match="degree reduced"):
        assert PCESurrogate(degree=6, max_terms=20)._safe_degree(3, 6) == ours.degree_
