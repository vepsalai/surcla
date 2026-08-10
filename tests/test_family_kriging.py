"""Parity: the shipped Kriging surrogate reproduces the research one.

The research tree is not distributed with the package, so this file skips
cleanly when it is absent. When present, both implementations are fitted on the
same seeded data and their predictions must agree to floating-point noise.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

from surcla.families.kriging import VARIANTS, KrigingSurrogate

RESEARCH_DIR = Path.home() / "claudeland/surcla_project/surcla/surrogates"

sys.path.insert(0, str(RESEARCH_DIR))
try:
    from gp_surrogate import GPSurrogate
except ImportError:                                  # end-user install
    GPSurrogate = None

needs_research = pytest.mark.skipif(
    GPSurrogate is None, reason="research tree not available")

TOL = 1e-10


def make_data(n=120, d=4, n_test=40, seed=0):
    """Smooth nonlinear response plus small noise, on [-1, 1]^d."""
    rng = np.random.default_rng(seed)

    def f(X):
        return (np.sin(np.pi * X[:, 0]) + 0.5 * X[:, 1] ** 2
                + X[:, 2] * X[:, 3] - 0.3 * X[:, 1])

    X = rng.uniform(-1.0, 1.0, size=(n, d))
    y = f(X) + 0.01 * rng.standard_normal(n)
    X_test = rng.uniform(-1.0, 1.0, size=(n_test, d))
    return X, y, X_test


def deviation(variant):
    """Max absolute gap in mean and in std between research and shipped fits."""
    X, y, X_test = make_data()
    research = GPSurrogate(variant=variant).fit(X, y)
    shipped = KrigingSurrogate(variant=variant).fit(X, y)
    r_mean, r_std = research.predict_with_std(X_test)
    s_mean, s_std = shipped.predict_with_std(X_test)
    assert np.allclose(r_mean, research.predict(X_test), rtol=0, atol=0)
    assert np.allclose(s_mean, shipped.predict(X_test), rtol=0, atol=0)
    return max(float(np.abs(r_mean - s_mean).max()),
               float(np.abs(r_std - s_std).max()))


@needs_research
@pytest.mark.parametrize("variant", VARIANTS)
def test_parity_with_research(variant):
    assert deviation(variant) < TOL


def test_variant_tag_is_validated():
    X, y, _ = make_data(n=30)
    for bad in ["Kriging_XGS", "Kriging_OXS", "GP_OGS", "Kriging_O"]:
        with pytest.raises(ValueError):
            KrigingSurrogate(variant=bad).fit(X, y)


def test_unknown_kernel_type_raises_rather_than_falling_back():
    """A typo must not silently become Matern, as it would under an else."""
    with pytest.raises(ValueError, match="Unknown kernel type"):
        KrigingSurrogate()._build_kernel(2, "gausian")


def test_linear_trend_is_recovered():
    """A pure plane is captured by the trend, leaving the GP near zero."""
    X, y, X_test = make_data(n=60)
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 1.0
    model = KrigingSurrogate(variant="Kriging_LGS").fit(X, y)
    truth = 2.0 * X_test[:, 0] - 3.0 * X_test[:, 1] + 1.0
    assert np.abs(model.predict(X_test) - truth).max() < 1e-6
