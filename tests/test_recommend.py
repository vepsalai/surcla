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


def test_warm_starts(report):
    """Every candidate carries a starting configuration from a matching band."""
    from surcla.warmstart import lookup
    for c in report.candidates:
        assert c.warm_start is not None
        assert c.warm_start.n_cells > 0
        assert c.warm_start.band == "n<=100, d<=5"
    by_family = {c.family: c for c in report.candidates}
    assert isinstance(by_family["Kriging"].warm_start.config, str)   # variant tag
    assert by_family["Kriging"].warm_start.meaning                   # decoded
    assert isinstance(by_family["LGBM"].warm_start.config, dict)     # parameters
    # bands actually differ with size
    big = lookup("LGBM", 5000, 30)
    assert big.band == "n>700, d>15"
    assert big.config != by_family["LGBM"].warm_start.config
    assert lookup("NoSuchFamily", 100, 5) is None


def test_attain_r2_threshold(report):
    """The usefulness threshold is the caller's; the estimate does not move."""
    strict = recommend(GOLD["cell_small_a_X"], GOLD["cell_small_a_y"],
                       attain_r2=0.95)
    lenient = recommend(GOLD["cell_small_a_X"], GOLD["cell_small_a_y"],
                        attain_r2=0.0)
    assert strict.attainability == lenient.attainability == report.attainability
    assert strict.reject and not lenient.reject
    assert strict.attain_r2 == 0.95 and lenient.attain_r2 == 0.0
    assert "attain_r2=0.95" in strict.advice


def test_input_validation():
    with pytest.raises(ValueError):
        recommend(np.ones((5, 2)), np.ones(5))          # too few rows
    with pytest.raises(ValueError):
        recommend(np.ones((20, 2)), np.ones(19))        # length mismatch
    X = np.random.default_rng(0).normal(size=(30, 3))
    y = np.full(30, np.nan)
    with pytest.raises(ValueError):
        recommend(X, y)                                 # non-finite
    with pytest.raises(ValueError):
        recommend(GOLD["cell_small_a_X"], GOLD["cell_small_a_y"],
                  attain_r2=1.5)                        # out of R² range
