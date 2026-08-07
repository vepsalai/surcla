"""End-to-end tests of the recommendation report (slow: first call fits the
decoder and the failure heads from the bundled tables)."""

from pathlib import Path

import numpy as np
import pytest

from surcla import recommend

GOLD = np.load(Path(__file__).parent / "golden.npz")
FAMILIES = {"Kriging", "LGBM", "LR", "Lasso", "MLP", "PCE", "PCK", "RF",
            "Ridge", "XGB"}


@pytest.fixture(scope="module")
def report():
    return recommend(GOLD["cell_small_a_X"], GOLD["cell_small_a_y"], k=3)


def test_report_structure(report):
    assert len(report.candidates) == 10
    assert {c.family for c in report.candidates} == FAMILIES
    assert report.candidates[0].predicted_r2 >= report.candidates[1].predicted_r2 \
        or report.candidates[1].vetoed
    assert report.attainability == report.candidates[0].predicted_r2 \
        or report.candidates[0].vetoed
    assert set(report.regret_at_k) == {1, 2, 3}


def test_fragile_families_carry_p_fail(report):
    by_family = {c.family: c for c in report.candidates}
    assert by_family["Kriging"].p_fail is not None
    assert by_family["MLP"].p_fail is not None
    assert by_family["Ridge"].p_fail is None
    assert all(0.0 <= c.p_fail <= 1.0 for c in report.candidates
               if c.p_fail is not None)


def test_small_sample_advice(report):
    assert report.n_train < 100
    assert "no single pick" in report.advice


def test_deterministic(report):
    again = recommend(GOLD["cell_small_a_X"], GOLD["cell_small_a_y"], k=3)
    assert [c.family for c in again.candidates] == \
           [c.family for c in report.candidates]
    assert again.attainability == report.attainability


def test_input_validation():
    with pytest.raises(ValueError):
        recommend(np.ones((5, 2)), np.ones(5))          # too few rows
    with pytest.raises(ValueError):
        recommend(np.ones((20, 2)), np.ones(19))        # length mismatch
    X = np.random.default_rng(0).normal(size=(30, 3))
    y = np.full(30, np.nan)
    with pytest.raises(ValueError):
        recommend(X, y)                                 # non-finite
