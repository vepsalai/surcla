"""The labelling run's input caps, on by default for the UQ families.

The caps are not only an affordability measure: a predicted R² for Kriging on
a large dataset is a prediction about a 2000-row fit, so fitting uncapped
answers a different question, slowly. These tests pin that the default holds,
that it can be lifted, that the caller never has to know about it when
predicting, and that the other seven families are untouched.
"""

import time

import numpy as np
import pytest

from surcla.families import MAX_FEATURES, MAX_ROWS, Capped, build
from surcla.families.capped import select_features
from surcla.recommend import Candidate
from surcla.warmstart import lookup


def _data(n, d, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + rng.normal(0, 0.05, n)
    return X, y


@pytest.mark.parametrize("family", ["Kriging", "PCE", "PCK"])
def test_uq_families_are_capped_by_default(family):
    assert isinstance(build(family), Capped)


@pytest.mark.parametrize("family", ["RF", "LR", "Ridge", "Lasso", "MLP"])
def test_other_families_are_not_wrapped(family):
    assert not isinstance(build(family), Capped)


def test_caps_can_be_lifted():
    assert not isinstance(build("Kriging", None, None, None), Capped)


def test_row_cap_bounds_what_the_estimator_sees():
    X, y = _data(300, 3)
    model = build("Kriging", "_LMS", max_rows=100).fit(X, y)
    assert model.n_rows_used_ == 100
    assert model.estimator_.gp_.X_train_.shape[0] == 100


def test_feature_cap_survives_into_predict():
    """The caller keeps passing full-width rows; the wrapper narrows them."""
    X, y = _data(120, 12)
    model = build("PCE", "_Ridge_D5", max_features=4).fit(X, y)
    assert len(model.features_) <= 4
    assert model.predict(X[:5]).shape == (5,)          # 12 columns in, works
    # Narrowing is the wrapper's job: pre-narrowed rows are an error even when
    # the kept indices happen to fall inside them.
    with pytest.raises(ValueError, match="full-width rows"):
        model.predict(X[:5, :2])


def test_uncapped_fit_is_the_slower_one():
    """The cap is what keeps a cubic fit finishing; check it actually bites."""
    X, y = _data(700, 4)
    t0 = time.perf_counter()
    build("Kriging", "_LMS", max_rows=200).fit(X, y)
    capped = time.perf_counter() - t0
    t0 = time.perf_counter()
    build("Kriging", "_LMS", None, None).fit(X, y)
    full = time.perf_counter() - t0
    assert full > capped


def test_selection_passes_through_below_the_limit():
    X, y = _data(60, 5)
    assert np.array_equal(select_features(X, y, MAX_FEATURES), np.arange(5))


def test_fit_reports_the_cap_in_its_note():
    X, y = _data(400, 3)
    cand = Candidate(family="Kriging", predicted_r2=0.8, band=0.15,
                     p_fail=None, vetoed=False,
                     warm_start=lookup("Kriging", 400, 3))
    fitted = cand.fit(X, y, cv=2, max_rows=150)
    assert "150 of 400 rows" in fitted.cv_note
    assert fitted.caps == (150, MAX_FEATURES)
    assert fitted.predict(X[:4]).shape == (4,)


def test_default_caps_leave_small_data_alone():
    X, y = _data(80, 3)
    cand = Candidate(family="PCK", predicted_r2=0.8, band=0.15, p_fail=None,
                     vetoed=False, warm_start=lookup("PCK", 80, 3))
    fitted = cand.fit(X, y, cv=2)
    assert fitted.cv_note == "" or "rows" not in fitted.cv_note
    assert MAX_ROWS == 2000 and MAX_FEATURES == 30
