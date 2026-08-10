"""Parity: the shipped PC-Kriging reproduces the research implementation.

Both implementations are fitted on the same seeded synthetic data (n=150, d=3)
and their predictions must agree to 1e-10 for every registry variant. Skipped
when the research repo is not checked out beside this one; point
SURCLA_RESEARCH_REPO at its root to run it from elsewhere.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from surcla.families.pck import VARIANTS, make

_root = os.environ.get("SURCLA_RESEARCH_REPO")
RESEARCH = (Path(_root) if _root
            else Path(__file__).resolve().parents[2] / "surcla") / "surrogates"

pytestmark = pytest.mark.skipif(
    not (RESEARCH / "pck_surrogate.py").exists(),
    reason=f"research implementation not found under {RESEARCH}")


@pytest.fixture(scope="module")
def research_pck():
    """The research class, imported with its flat-module imports satisfied."""
    sys.path.insert(0, str(RESEARCH))
    try:
        from pck_surrogate import PCKrigingSurrogate
    finally:
        sys.path.remove(str(RESEARCH))
    return PCKrigingSurrogate


def _data(seed, n=150, d=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, d))
    y = (X[:, 0] ** 3 - 2.0 * X[:, 1] ** 2 + np.sin(3.0 * X[:, 2])
         + 0.5 * X[:, 0] * X[:, 1] + 0.05 * rng.standard_normal(n))
    return X, y


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_predict_matches_research(variant, research_pck):
    X, y = _data(0)
    X_test, _ = _data(1, n=40)
    ours = make(variant).fit(X, y)
    theirs = research_pck(scale_X="standard", n_restarts_optimizer=2,
                          random_state=42, **VARIANTS[variant]).fit(X, y)
    assert np.abs(ours.predict(X_test) - theirs.predict(X_test)).max() < 1e-10


def test_predict_with_std_matches_research(research_pck):
    X, y = _data(0)
    X_test, _ = _data(1, n=40)
    ours = make("PCK_3M").fit(X, y)
    theirs = research_pck(degree=3, kernel_type="matern", scale_X="standard",
                          n_restarts_optimizer=2, random_state=42).fit(X, y)
    mean, std = ours.predict_with_std(X_test)
    mean_ref, std_ref = theirs.predict_with_std(X_test)
    assert np.abs(mean - mean_ref).max() < 1e-10
    assert np.abs(std - std_ref).max() < 1e-10


def test_variant_wiring():
    X, y = _data(0)
    assert make("PCK_2G").fit(X, y).effective_degree_ == 2
    assert make("PCK_3G").fit(X, y).effective_degree_ == 3
    assert make("PCK_2M").fit(X, y).n_pce_terms_ == 10   # C(3 + 2, 2)
    with pytest.raises(ValueError, match="unknown PCK variant"):
        make("PCK_4G")
