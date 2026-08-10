"""Guards on the cross-validated number and on refine's use of it.

Each test here pins a defect found by adversarial review of the first
implementation: a score inflated by replicate leakage, a fold count that
changed under the caller, an incoherent cv argument answered with a factually
wrong note, and a refine that scored one dataset to choose a configuration for
another.
"""

import warnings

import numpy as np
import pytest

from surcla import refine
from surcla.recommend import Candidate, cross_val_r2
from surcla.warmstart import lookup


def _candidate(family, n, d):
    return Candidate(family=family, predicted_r2=0.8, band=0.15, p_fail=None,
                     vetoed=False, warm_start=lookup(family, n, d))


def _replicated(n_points=20, reps=5, seed=0):
    """A replicated design: each point measured `reps` times."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(-1, 1, size=(n_points, 3))
    X = np.repeat(base, reps, axis=0)
    truth = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    y = truth + rng.normal(0, 0.05, size=len(X))
    groups = np.repeat(np.arange(n_points), reps)
    return X, y, groups


def test_replicates_split_across_folds_inflate_the_score():
    """Ungrouped folds read the noise floor as skill; grouping removes it."""
    X, y, groups = _replicated()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ungrouped, _, _ = cross_val_r2("RF", None, X, y, cv=5)
    grouped, _, _ = cross_val_r2("RF", None, X, y, cv=5, groups=groups)
    assert ungrouped > grouped + 0.1


def test_duplicate_rows_warn_when_no_grouping_is_given():
    X, y, groups = _replicated()
    with pytest.warns(UserWarning, match="duplicate rows"):
        cross_val_r2("LR", None, X, y, cv=5)
    with warnings.catch_warnings():           # grouped: no warning
        warnings.simplefilter("error")
        cross_val_r2("LR", None, X, y, cv=5, groups=groups)


def test_groups_travel_from_fit_into_refine():
    X, y, groups = _replicated()
    fitted = _candidate("Ridge", len(y), X.shape[1]).fit(X, y, cv=5,
                                                         groups=groups)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)   # no leakage warning
        refined = refine(fitted, X, y, budget=3)
    assert refined.cv_r2 >= fitted.cv_r2 - 1e-12


def test_incoherent_fold_counts_raise_rather_than_explain_wrongly():
    X = np.random.default_rng(0).normal(size=(40, 2))
    y = X[:, 0] * 2.0 + 0.1
    for bad in (0, 1, -3):
        with pytest.raises(ValueError, match="at least 2 folds"):
            cross_val_r2("LR", None, X, y, cv=bad)


def test_small_n_says_the_score_is_unstable():
    """The note must fire at the default cv, where the old guard could not."""
    X = np.random.default_rng(0).normal(size=(25, 3))
    y = X[:, 0] + 0.1 * X[:, 1]
    _, _, note = cross_val_r2("LR", None, X, y, cv=5)
    assert "moves substantially" in note


def test_refine_refuses_data_it_was_not_fitted_on():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] * 2 - X[:, 1] ** 2
    fitted = _candidate("LR", 60, 3).fit(X, y, cv=5)
    with pytest.raises(ValueError, match="rows the model was fitted on"):
        refine(fitted, X[:15], y[:15] * 100 + 7, budget=8)


def test_refine_rejects_a_configuration_of_the_wrong_shape():
    """A dict where a variant tag belongs used to raise AttributeError."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] + X[:, 1]
    fitted = _candidate("PCK", 40, 3).fit(X, y, cv=3)
    broken = type(fitted)(**{**fitted.__dict__, "config": {"degree": 3}})
    with pytest.raises(TypeError, match="variant tag"):
        refine(broken, X, y, budget=3, cv=3)
