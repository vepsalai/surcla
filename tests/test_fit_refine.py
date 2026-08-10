"""Fitting a recommended candidate and refining it: the two accuracy numbers,
the small-n fold fallbacks, and the promise that refining never loses ground.

Candidates are built by hand from the warm-start table rather than through
`recommend`, so these tests cost a few fits instead of a decoder training run.
"""

import json

import numpy as np
import pytest

from surcla import FittedSurrogate, refine
from surcla.data import artifact_path
from surcla.families import AVAILABLE, FAMILIES, VARIANT_TAGS, build
from surcla.recommend import Candidate, cross_val_r2
from surcla.refine import _neighbours
from surcla.warmstart import lookup

N, D = 60, 3


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(N, D))
    y = (2.0 * X[:, 0] - X[:, 1] ** 2 + 0.5 * X[:, 0] * X[:, 2]
         + rng.normal(scale=0.05, size=N))
    return X, y


def candidate(family, n=N, d=D, vetoed=False):
    return Candidate(family=family, predicted_r2=0.8, band=0.1,
                     p_fail=0.9 if vetoed else None, vetoed=vetoed,
                     warm_start=lookup(family, n, d))


@pytest.mark.parametrize("family", ["RF", "MLP", "Kriging", "PCK"])
def test_fit_returns_both_numbers(family, data):
    fitted = candidate(family).fit(*data, cv=3)
    assert isinstance(fitted, FittedSurrogate)
    assert fitted.family == family
    assert fitted.n_train == N
    assert fitted.cv_folds == 3
    assert np.isfinite(fitted.cv_r2) and fitted.cv_r2 <= 1.0
    assert fitted.predicted_r2 == 0.8 and fitted.band == 0.1
    assert fitted.predict(data[0]).shape == (N,)
    assert fitted.predict(data[0][:4]).shape == (4,)


def test_pce_fits_through_the_registry(data):
    """The slow one: its warm start auto-selects degree and truncation."""
    fitted = candidate("PCE").fit(*data, cv=3)
    assert fitted.config == "_LARS_D5"
    assert fitted.model.solver == "lars"
    assert fitted.model.degree_ in (2, 3, 4, 5)
    assert np.isfinite(fitted.cv_r2)


def test_uq_warm_start_reaches_the_right_variant(data):
    fitted = candidate("Kriging").fit(*data, cv=3)
    assert fitted.config == "_LMS"                     # the table's tag
    assert fitted.model.variant == "Kriging_LMS"       # resolved by the registry
    assert fitted.model.n_restarts_optimizer == 2      # the labelling setting


def test_repr_shows_both_numbers(data):
    text = repr(candidate("Ridge").fit(*data, cv=3))
    assert "predicted R² 0.800 ± 0.10" in text
    assert "CV R²" in text
    assert "3-fold on your own data" in text


def test_vetoed_candidate_fits_but_warns(data):
    with pytest.warns(UserWarning, match="was vetoed"):
        fitted = candidate("Ridge", vetoed=True).fit(*data, cv=3)
    assert np.isfinite(fitted.cv_r2)


def test_tiny_n_falls_back_to_leave_one_out():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(8, 2))
    y = X[:, 0] + 0.1 * X[:, 1]
    fitted = candidate("LR", n=8, d=2).fit(X, y, cv=5)
    assert fitted.cv_folds == 8                        # 8 < 2 * 5
    assert "leave-one-out" in fitted.cv_note
    assert isinstance(fitted.cv_r2, float)


def test_cv_r2_is_undefined_not_fatal_on_constant_y():
    X = np.random.default_rng(2).uniform(size=(30, 2))
    r2, folds, note = cross_val_r2("LR", None, X, np.ones(30), cv=3)
    assert np.isnan(r2) and folds == 3 and "constant" in note


def test_cv_r2_recovers_a_linear_truth():
    rng = np.random.default_rng(3)
    X = rng.uniform(size=(80, 3))
    y = 1.0 + 2.0 * X[:, 0] - X[:, 2]
    r2, _, _ = cross_val_r2("LR", {"fit_intercept": True}, X, y, cv=5)
    assert r2 > 0.999


def test_neighbours_of_a_uq_family_are_its_siblings():
    assert set(_neighbours("Kriging", "_LMS")) == {"_OGS", "_OMS", "_LGS"}
    assert set(_neighbours("PCE", "_LARS_D5")) == {"_Ridge_D5", "_OMP_D5"}
    assert set(_neighbours("PCK", "_2M")) == {"_2G", "_3G", "_3M"}
    for family, tags in VARIANT_TAGS.items():
        assert len(_neighbours(family, tags[0])) == len(tags) - 1


def test_neighbours_exclude_the_incumbent_even_without_a_warm_start():
    """No warm start means the family's own default, which is still incumbent."""
    assert set(_neighbours("Kriging", None)) == {"_OMS", "_LGS", "_LMS"}
    assert 1.0 not in [m["alpha"] for m in _neighbours("Ridge", None)]
    assert 0.01 not in [m["alpha"] for m in _neighbours("Lasso", None)]


def test_neighbours_move_one_coordinate_at_a_time():
    config = {"n_estimators": 20, "max_leaves": 7, "max_features": 0.8}
    moves = _neighbours("RF", config)
    assert all(sum(v != config[k] for k, v in m.items()) == 1 for m in moves)
    assert {m["n_estimators"] for m in moves} == {10, 20, 40}
    assert all(m["max_features"] == 0.8 for m in moves)   # bounded, left alone
    assert _neighbours("LR", {"fit_intercept": True}) == []


def test_count_parameters_stay_at_least_one():
    assert {m["n_estimators"] for m in _neighbours("RF", {"n_estimators": 1})} \
        == {2}


def test_alpha_grid_for_the_penalized_linear_families():
    moves = _neighbours("Ridge", {"alpha": 1.0, "fit_intercept": True})
    assert [m["alpha"] for m in moves] == [0.001, 0.01, 0.1, 10.0]
    assert all(m["fit_intercept"] for m in moves)


@pytest.mark.parametrize("family", ["Ridge", "RF"])
def test_refine_never_returns_a_worse_cv(family, data):
    start = candidate(family).fit(*data, cv=3)
    better = refine(start, *data, budget=4, cv=3)
    assert better.n_configs_tried == 4
    assert better.cv_r2 >= start.cv_r2 - 1e-12
    assert better.cv_gain >= 0.0
    assert better.predict(data[0]).shape == (N,)
    assert better.predicted_r2 == start.predicted_r2   # carried over untouched


def test_refine_tries_the_sibling_variants(data):
    start = candidate("Kriging").fit(*data, cv=3)
    better = refine(start, *data, budget=4, cv=3)
    assert better.n_configs_tried == 4                 # incumbent plus three
    assert better.config in VARIANT_TAGS["Kriging"]
    assert better.cv_r2 >= start.cv_r2 - 1e-12
    assert "refined over 4 configurations" in repr(better)


def test_refine_budget_of_one_only_rescores_the_incumbent(data):
    start = candidate("RF").fit(*data, cv=3)
    same = refine(start, *data, budget=1, cv=3)
    assert same.n_configs_tried == 1
    assert same.config == start.config
    assert same.model is start.model                   # no needless refit
    with pytest.raises(ValueError):
        refine(start, *data, budget=0)


def test_refine_on_a_family_without_knobs(data):
    start = candidate("LR").fit(*data, cv=3)
    same = refine(start, *data, budget=8, cv=3)
    assert same.n_configs_tried == 1 and same.cv_gain == 0.0


def test_refine_scores_on_the_folds_that_produced_the_incoming_number(data):
    """Nothing to try, so the score must not move: with a fixed cv=5 default it
    did, by up to half an R² on some families, while reporting cv_gain 0."""
    start = candidate("LR").fit(*data, cv=3)
    same = refine(start, *data, budget=8)
    assert same.cv_folds == 3
    assert same.cv_r2 == pytest.approx(start.cv_r2, abs=1e-12)
    assert same.cv_gain == 0.0
    assert "rescored" not in same.cv_note


def test_refine_says_so_when_asked_for_different_folds(data):
    start = candidate("LR").fit(*data, cv=3)
    forced = refine(start, *data, budget=8, cv=5)
    assert forced.cv_folds == 5
    assert "rescored at 5 folds" in forced.cv_note


def test_refine_warns_that_a_small_n_gain_may_be_selection_noise():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 3))
    y = rng.normal(size=40)                            # nothing to learn
    start = candidate("Ridge", n=40, d=3).fit(X, y, cv=5)
    with pytest.warns(UserWarning, match="selection noise"):
        better = refine(start, X, y, budget=5, cv=5)
    assert better.cv_gain > 0.0


def test_refine_survives_tiny_n():
    rng = np.random.default_rng(4)
    X = rng.uniform(size=(9, 2))
    y = X[:, 0] - 0.5 * X[:, 1]
    start = candidate("Ridge", n=9, d=2).fit(X, y, cv=5)
    better = refine(start, X, y, budget=3, cv=5)
    assert better.cv_folds == 9
    assert better.cv_r2 >= start.cv_r2 - 1e-12


def _installed(package):
    try:
        __import__(package)
    except ImportError:
        return False
    return True


def test_available_tracks_the_optional_packages():
    assert set(AVAILABLE) >= {"Kriging", "PCE", "PCK", "MLP", "RF", "LR",
                              "Ridge", "Lasso"}
    for family, package in (("LGBM", "lightgbm"), ("XGB", "xgboost")):
        assert (family in AVAILABLE) == _installed(package)


@pytest.mark.parametrize("family, package", [("LGBM", "lightgbm"),
                                             ("XGB", "xgboost")])
def test_optional_boosters_fit_when_present(family, package, data):
    pytest.importorskip(package)
    fitted = candidate(family).fit(*data, cv=3)
    assert np.isfinite(fitted.cv_r2)
    assert refine(fitted, *data, budget=3, cv=3).cv_r2 >= fitted.cv_r2 - 1e-12


def test_registry_and_warm_start_table_agree():
    """A tag in the table with no constructor behind it would fail at fit time."""
    table = json.loads(artifact_path("warm_start.json").read_text())
    assert set(FAMILIES) == set(table["families"])
    for family, tags in VARIANT_TAGS.items():
        assert set(tags) == set(table["variant_meaning"][family])
    for family, cells in table["families"].items():
        for entry in cells.values():
            config = entry["config"]
            if isinstance(config, str):
                assert config in VARIANT_TAGS[family]


def test_build_rejects_nonsense():
    with pytest.raises(ValueError):
        build("NoSuchFamily")
    with pytest.raises(ValueError):
        build("Kriging", "_XYZ")
    with pytest.raises(TypeError):
        build("PCK", {"degree": 3})       # a dict where a tag belongs
