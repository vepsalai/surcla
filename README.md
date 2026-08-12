# SurCla

[![tests](https://github.com/vepsalai/surcla/actions/workflows/tests.yml/badge.svg)](https://github.com/vepsalai/surcla/actions/workflows/tests.yml)

Given a regression dataset `(X, y)`, SurCla predicts which of ten surrogate
families will fit it best, how well, and how likely the fit is to fail, before
any of them is fitted. The candidate pool spans both traditions engineers
actually choose between: linear models, random forest, LightGBM, XGBoost and a
neural network on one side, polynomial chaos, Kriging and PC-Kriging on the
other.

The recommender was trained entirely on 7,995 generated datasets, without a
single real meta-label, and evaluated in one pre-registered pass on a sealed
suite of 27 real problems. On that suite it reaches a median regret of 0.003
R², and 0.0003 once the two highest-ranked families are fitted: the pick is
rarely the exact winner and rarely costs anything, because the top families are
usually near-tied.

```
pip install git+https://github.com/vepsalai/surcla
pip install "surcla[lgbm,xgb] @ git+https://github.com/vepsalai/surcla"   # with the boosted families
```

On macOS the boosted families also need the OpenMP runtime: without it
XGBoost and LightGBM fail at import, and the error's most eye-catching
bullet (32-bit Python) is not the cause. In a plain venv, `brew install
libomp` provides it. In a conda environment, do not mix pip's native wheels
with conda's runtime: torch and scikit-learn wheels each bundle their own
libomp, and duplicate runtimes can segfault a Jupyter kernel mid-fit rather
than fail loudly. Install the native stack from conda-forge, so one runtime
serves everything, and add surcla on top without its pinned dependencies:

```
conda create -n surcla-env -c conda-forge python=3.12 pytorch scikit-learn numpy scipy pandas pyarrow xgboost lightgbm
conda activate surcla-env
pip install --no-deps "surcla @ git+https://github.com/vepsalai/surcla"
```

## Recommend, fit, refine

Runnable as is: the data loader ships with scikit-learn, already a dependency.

```python
from sklearn.datasets import fetch_california_housing
from surcla import recommend, refine

X, y = fetch_california_housing(return_X_y=True)    # 20,640 rows, 8 features

report = recommend(X, y)
print(report)
```

```
SurCla report (n=20640, arm=embed): attainability 0.851
  1. XGB: predicted R² 0.851 ± 0.15
  2. RF: predicted R² 0.816 ± 0.15
  3. MLP: predicted R² 0.801 ± 0.15  p_fail=0.01
  (± is the sealed-suite median |error| of the estimate, so roughly half of
   datasets fall outside it)
```

Fitting a candidate starts it from the configuration that tuning chose on
corpus datasets of the same size and width, and cross-validates it on your own
rows. The pick lands on XGBoost here, so this dataset wants the `[lgbm,xgb]`
install:

```python
fitted = report.candidates[0].fit(X, y)     # warm-started, cross-validated
print(fitted)
```

```
FittedSurrogate(XGB, n_train=20640)
  predicted R² 0.851 ± 0.15   recommender's estimate for data like yours
  CV R²        0.805          5-fold on your own data
```

```python
better = refine(fitted, X, y, budget=8)     # a few neighbouring configurations
print(better)
X_new  = X[:5]                              # stand-in for your new inputs
y_hat  = better.predict(X_new)
```

```
FittedSurrogate(XGB, n_train=20640)
  predicted R² 0.851 ± 0.15   recommender's estimate for data like yours
  CV R²        0.824          5-fold on your own data
  refined over 8 configurations, CV R² +0.019
```

The refine pass bought its +0.019 in about three seconds. `refine` is a
deployment convenience, not part of the paper's evaluation: no published
number used it. California housing is one of the 15 sealed CTR23 datasets the paper
validates on, which makes the demo checkable: in the labelling run the best
family reached 0.848 R² at the largest training slice and XGB reached 0.829,
so both the attainability estimate of 0.851 and the refined CV of 0.824 land
where the published labels say they should.

## Reading the two accuracy numbers

They answer different questions and are allowed to disagree. The **predicted
R²** is formed before any fitting, from the dataset's own characteristics, and
estimates what this family reaches on unseen data like yours; its band is the
sealed-suite median absolute error, not a confidence interval, so roughly half
of datasets land outside it. The **CV R²** is what this one configuration
actually reached out of fold on your rows. When they diverge, your dataset sits
away from the corpus, or the warm start suits it poorly, or n is small enough
that both are noisy. The prediction is the number that speaks to
generalization: it never saw your rows, so it cannot inherit their optimism,
while a few folds over a small n can. A high CV R² is a statement about these
rows under this split; for use beyond them, expect something nearer the
predicted value.

How many candidates to fit is your move: repeat `report.candidates[i].fit(X, y)`
down the ranking and stop when satisfied. `report.regret_at_k` prices the
stopping depths with the published sealed-suite regret: 0.003 median regret
at one fitted family, 0.0003 at two, zero at three.

## What it will not do

**Below roughly 100 training samples, no single pick is trustworthy** — ours or
any fixed default. In that regime the product is the ranked shortlist plus the
attainability estimate, and a low estimate is a signal to collect data rather
than to model harder. The report says so itself when it applies.

**The label is pool-relative.** Winner and regret are defined against these ten
families under the tuning budget the labelling run used. A family outside the
pool, or a much deeper search, is outside what these numbers describe.

**The problem class is tabular numeric input to a scalar response.** Fields,
constraints, gradients and multi-output responses are out of scope.

**Kriging, PCE and PC-Kriging fit under caps** of 2,000 rows and 30
importance-ranked features, the bounds the labelling run used. They are the
bounds the predicted R² describes, and they keep a cubic fit finishing: one
Kriging fit takes 81 s at 2,000 rows and 343 s at 4,000, so 20,000 rows
uncapped is a matter of hours. Pass `max_rows=None, max_features=None` to
`fit` to lift them, and expect a slower and usually better model than the
estimate refers to.

## Replicated designs

If rows repeat a design point, tell `fit` which rows belong together, or
cross-validation will put a replicate in the training fold and its twin in the
test fold and report the measurement noise floor as skill, by as much as
+0.5 R²:

```python
fitted = report.candidates[0].fit(X, y, groups=design_point_index)
```

Duplicate rows without a grouping raise a warning rather than a silent
inflation.

## What ships here

The package carries everything it needs: the corpus meta-table and per-seed
embeddings the decoder trains from, the three encoder checkpoints, the mined
warm-start table, and the published metrics every number in a report is quoted
from (`src/surcla/artifacts/`). Nothing is fetched at runtime and nothing is
re-measured on the fly. The first call fits the decoder and the failure heads
from those tables, about twenty seconds; later calls cost milliseconds plus the
encoder pass.

Two encodings are available. `arm="embed"`, the default, is the learned
Dataset2Vec v3 encoder that led the sealed evaluation. `arm="manual"` is the 51
hand-crafted metafeatures, which need no torch pass and are the fallback when
that matters.

## Reproducing the paper

`reproduce/paper_numbers.py` refits every arm from the shipped artifacts and
checks the result against what the paper reports, in about four minutes.
Under the scikit-learn that recorded the artifacts (1.5.2) every number
matches to 1e-9; under a different version a few near-tied cells flip their
pick, which the script reports as expected drift rather than failure and
bounds with per-metric bands. Anything beyond the bands exits non-zero. What it cannot cover, the corpus
generation, the labelling runs and the encoder training, is mapped in
[reproduce/README.md](reproduce/README.md).

## Citing

Paper reference and archived-data DOI to appear here on publication.

## License

MIT for the code. The research repository behind it, including the corpus
generation, the labelling harness and every script that produced the published
figures, is separate.
