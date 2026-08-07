# SurCla

Surrogate model recommendation for regression datasets: given `(X, y)`, SurCla
predicts which of ten surrogate families (linear models, random forest,
LightGBM, XGBoost, MLP, polynomial chaos, Kriging, PC-Kriging) will fit it
best, how well, and how likely the fit is to fail — before any family is
fitted.

The recommender is trained entirely on generated data and was evaluated in a
single pre-registered pass on a sealed suite of 27 real problems. The paper
documents the method, the evaluation protocol, and the honest limits of the
recommendation (below roughly 100 training samples, trust the ranked shortlist
and the attainability warning, not any single pick).

**Status: work in progress — the package API is being assembled for the v1.0
release accompanying the paper.**

## Install

```
pip install git+https://github.com/vepsalai/surcla
```

## Quickstart

```python
from surcla import recommend

report = recommend(X, y, k=3)
report.candidates[0].family        # e.g. "Kriging"
report.candidates[0].predicted_r2  # calibrated attainability estimate
report.attainability               # best predicted R² across families
report.reject                      # True: no family is expected to fit this
```

## Citing

Paper reference and archived data DOI to appear here on publication.
