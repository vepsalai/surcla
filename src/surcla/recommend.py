"""The recommendation report: ranked families, calibrated R², failure risk.

`recommend(X, y)` runs the deployed pipeline of the paper: the corpus-
trained Dataset2Vec v3 encoder (eight-draw embedding) feeding a random-forest
regret decoder trained on the shipped corpus table, with the published
constant-shift calibration, the failure heads' veto-by-demotion, and the
small-sample advice the sealed evaluation established. First call fits the
decoder and the heads from the bundled tables (around 20 seconds); later calls
cost milliseconds plus the embedding. Deterministic in everything a caller
reads off it, the ranking and the advice; the forests run threaded, so the
estimates themselves can differ in the last bit or two between calls.

`Candidate.fit` then turns one ranked candidate into a fitted model, warm
started from the corpus table and cross-validated on the caller's own rows.
That gives two accuracy numbers per candidate, a prediction and a measurement,
which `FittedSurrogate` keeps side by side rather than reconciling.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from . import data
from .calibration import published_metrics, shift_and_band
from .d2v import embed, load_encoder
from .decoder import RegretDecoder
from .failure import FailureHeads
from .families import AVAILABLE, MAX_FEATURES, MAX_ROWS, Capped, build
from .metafeatures import extract
from .warmstart import WarmStart, lookup

_PIPELINES: dict = {}

# Below this many rows the fold count is capped, because a fold of a handful of
# rows scores mostly its own sampling noise.
SMALL_N = 50


def _as_arrays(X, y):
    """Finite float arrays of matching length, X two-dimensional."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2 or X.shape[0] != len(y):
        raise ValueError(f"expected X (n, d) and y (n,); got {X.shape}, "
                         f"{y.shape}")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("X and y must be finite (drop or impute NaNs first)")
    return X, y


def cross_val_r2(family: str, config, X, y, cv: int = 5, groups=None,
                 max_rows: int | None = MAX_ROWS,
                 max_features: int | None = MAX_FEATURES):
    """Out-of-fold R² of one configuration, measured on the caller's own data.

    Every row is predicted by a model that never saw it, then all held-out
    predictions are scored at once. Pooling rather than averaging per-fold R²
    keeps the number defined when a fold holds a single row, which is what
    small datasets fall back to.

    Folds are capped at five below `SMALL_N` rows and become leave-one-out
    below `2 * cv` rows. Returns `(r2, n_splits, note)`, with r2 nan and the
    note saying why when the score is undefined: a fold that would not fit,
    a constant y, or fewer than two rows.

    `groups` labels rows that must not be split across folds. Repeated
    measurements of one design point are exactly that: without grouping, a
    replicate sitting in the training fold predicts its twin in the test fold,
    and the score reports the measurement noise floor instead of
    generalization, by as much as +0.5 R² on a fully replicated design. Pass
    one label per row, the design-point index, and folds are formed with
    GroupKFold. When groups is omitted and duplicate feature rows are present,
    this warns rather than guessing what they mean.

    `max_rows` and `max_features` reach the UQ families through `build`, so
    every fold is capped exactly as the labelling run was; see
    `families.capped`.
    """
    X, y = _as_arrays(X, y)
    if int(cv) < 2:
        raise ValueError(f"cv must be at least 2 folds, got {cv}")
    n = len(y)
    folds, notes = int(cv), []
    if n < SMALL_N and folds > 5:
        folds = 5
        notes.append(f"n < {SMALL_N}: fold count capped at 5")

    if groups is None:
        if n < 2 * folds:
            notes.append(f"n = {n}: leave-one-out instead of {folds}-fold")
            folds = n
        if folds < 2:
            return float("nan"), 0, "fewer than 2 rows: no cross-validation"
        if len(np.unique(X, axis=0)) < n:
            warnings.warn(
                f"{family}: X holds duplicate rows, which cross-validation "
                "will split across folds; if they are replicates of the same "
                "design point, pass groups= to keep them together, or cv_r2 "
                "will read the noise floor as skill", UserWarning,
                stacklevel=2)
        splits = KFold(n_splits=folds, shuffle=True, random_state=0).split(X)
    else:
        groups = np.asarray(groups).ravel()
        if len(groups) != n:
            raise ValueError(f"groups has {len(groups)} labels for {n} rows")
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            return float("nan"), 0, "fewer than 2 groups: no cross-validation"
        if n_groups < folds:
            notes.append(f"{n_groups} groups: {n_groups} folds, not {folds}")
            folds = n_groups
        splits = GroupKFold(n_splits=folds).split(X, y, groups)

    if n < SMALL_N:
        notes.append(f"n = {n}: this score moves substantially with the fold "
                     "split, so read it as an order of magnitude")

    oof = np.empty(n)
    for train, test in splits:
        try:
            model = build(family, config, max_rows,
                          max_features).fit(X[train], y[train])
            oof[test] = np.asarray(model.predict(X[test]), dtype=float).ravel()
        except Exception as exc:                       # any estimator, any cause
            warnings.warn(f"{family}: a cross-validation fold failed to fit "
                          f"({type(exc).__name__}: {exc}); cv_r2 is nan",
                          UserWarning, stacklevel=2)
            return float("nan"), folds, "a fold failed to fit"

    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        return float("nan"), folds, "y is constant: R² undefined"
    if not np.isfinite(oof).all():
        return float("nan"), folds, "a fold predicted non-finite values"
    r2 = 1.0 - float(((y - oof) ** 2).sum()) / ss_tot
    return r2, folds, "; ".join(notes)


@dataclass
class FittedSurrogate:
    """A fitted model and the two accuracy numbers that describe it.

    They answer different questions and legitimately disagree. `predicted_r2`
    is the recommender's calibrated estimate of what this family reaches on
    data like yours, formed before any fitting and carrying the published band
    `band`. `cv_r2` is what this one configuration actually reached out of fold
    on your rows. A gap means your dataset sits away from the corpus, or the
    warm start suits it poorly, or n is small enough that both numbers are
    noisy, or, for the three UQ families on a large dataset, that this fit saw
    data the labelled one was capped away from (see `Candidate.fit`).
    """

    family: str
    model: object                # the fitted estimator
    config: dict | str | None    # parameter dict or variant tag it was built at
    cv_r2: float                 # out-of-fold R² on the caller's own data
    predicted_r2: float          # the recommender's estimate, carried over
    band: float                  # published median |error| of that estimate
    n_train: int
    cv_folds: int = 0            # folds actually used, after the small-n rules
    cv_note: str = ""            # why the fold count or the score differs
    groups: object = field(default=None, repr=False)  # grouping cv_r2 honoured
    caps: tuple = field(default=(MAX_ROWS, MAX_FEATURES), repr=False)
    n_configs_tried: int = 1     # configurations evaluated (see refine)
    cv_gain: float = 0.0         # refine's improvement, scored on the folds
                                 # that chose the winner, so biased upward

    def predict(self, X_new):
        """Predictions for new rows, in the training feature order."""
        return np.asarray(self.model.predict(np.asarray(X_new,
                                                        dtype=np.float64)))

    def __repr__(self):
        tag = f" {self.config}" if isinstance(self.config, str) else ""
        predicted = f"{self.predicted_r2:.3f} ± {self.band:.2f}"
        lines = [
            f"FittedSurrogate({self.family}{tag}, n_train={self.n_train})",
            f"  predicted R² {predicted:<14} recommender's estimate for data "
            "like yours",
            f"  CV R²        {self.cv_r2:<14.3f} {self.cv_folds}-fold on your "
            "own data",
        ]
        if self.cv_note:
            lines.append(f"  note: {self.cv_note}")
        if self.n_configs_tried > 1:
            lines.append(f"  refined over {self.n_configs_tried} "
                         f"configurations, CV R² {self.cv_gain:+.3f}")
        return "\n".join(lines)


@dataclass
class Candidate:
    family: str
    predicted_r2: float          # calibrated estimate of attainable R²
    band: float                  # published median |error| of that estimate
    p_fail: float | None         # failure probability (fragile families only)
    vetoed: bool                 # p_fail > tau: demoted to the bottom
    warm_start: WarmStart | None = None   # configuration to fit first
    available: bool = True       # False when the family's optional package is absent

    def __repr__(self):
        veto = "  VETOED" if self.vetoed else ""
        pf = f"  p_fail={self.p_fail:.2f}" if self.p_fail is not None else ""
        missing = "" if self.available else "  (package not installed)"
        return (f"{self.family}: predicted R² {self.predicted_r2:.3f} "
                f"± {self.band:.2f}{pf}{veto}{missing}")

    def fit(self, X, y, cv: int = 5, groups=None,
            max_rows: int | None = MAX_ROWS,
            max_features: int | None = MAX_FEATURES) -> FittedSurrogate:
        """Fit this family on (X, y) and cross-validate it on the same data.

        The fit uses the warm start the recommender attached, so the model is
        the corpus-typical configuration for a dataset of this shape rather
        than a default one. The cross-validation refits that same
        configuration `cv` times to measure it on your data, which is the only
        number here that your rows actually produced; see `FittedSurrogate`
        for how it relates to the predicted R². Folds follow the small-n rules
        of `cross_val_r2`, and the returned object records which were used.

        Kriging, PCE and PCK are fitted under the labelling run's caps, at most
        `max_rows` rows and `max_features` features, because those bounds are
        what their predicted R² describes and what keeps a cubic fit from
        running for hours; `families.capped` gives the timings. When a cap
        bites, cv_note says by how much. Pass max_rows=None and
        max_features=None to use everything, and expect a slower and usually
        better model than the estimate refers to. The other seven families
        were labelled on full data and ignore both.

        Pass `groups` when rows repeat a design point, so replicates stay in
        one fold; `cross_val_r2` explains what omitting it costs.

        A vetoed candidate is fitted anyway, with a warning: the veto is a
        prediction about failure risk, not a refusal.
        """
        X, y = _as_arrays(X, y)
        if self.vetoed:
            pf = "" if self.p_fail is None else f" (p_fail={self.p_fail:.2f})"
            warnings.warn(f"{self.family} was vetoed{pf}: the failure heads "
                          "expect this fit to go wrong, so read cv_r2 before "
                          "trusting it", UserWarning, stacklevel=2)
        config = None if self.warm_start is None else self.warm_start.config
        model = build(self.family, config, max_rows, max_features).fit(X, y)
        r2, folds, note = cross_val_r2(self.family, config, X, y, cv, groups,
                                       max_rows, max_features)
        if isinstance(model, Capped):
            cap_note = model.capped(len(y), X.shape[1])
            note = "; ".join(n for n in (note, cap_note) if n)
        return FittedSurrogate(family=self.family, model=model, config=config,
                               cv_r2=r2, predicted_r2=self.predicted_r2,
                               band=self.band, n_train=int(len(y)),
                               cv_folds=folds, cv_note=note, groups=groups,
                               caps=(max_rows, max_features))


@dataclass
class Report:
    candidates: list[Candidate]  # all families, ranked; vetoed ones last
    attainability: float         # best calibrated predicted R²
    reject: bool                 # attainability below attain_r2
    attain_r2: float             # the usefulness threshold in force
    n_train: int
    regret_at_k: dict            # published sealed-suite regret at depth 1..3
    advice: str
    arm: str
    seed: int

    def __repr__(self):
        head = (f"SurCla report (n={self.n_train}, arm={self.arm}): "
                f"attainability {self.attainability:.3f}"
                + (f", REJECT (below attain_r2={self.attain_r2:g})"
                   if self.reject else ""))
        shown = self.candidates[:3]
        lines = [head] + [f"  {i + 1}. {c!r}" for i, c in enumerate(shown)]
        lines.append("  (± is the sealed-suite median |error| of the estimate, "
                     "so roughly half of datasets fall outside it)")
        if self.advice:
            lines.append(f"  note: {self.advice}")
        return "\n".join(lines)


def _pipeline(arm: str, seed: int):
    key = (arm, seed)
    if key not in _PIPELINES:
        meta = data.load_corpus_table()
        heads = FailureHeads(random_state=seed).fit(meta)
        if arm == "embed":
            embc = data.load_corpus_embeddings(seed)
            emb_cols = [c for c in embc.columns if c.startswith("emb_")]
            # table column emb_<i> holds embedding dimension i
            emb_cols_by_dim = sorted(emb_cols, key=lambda c: int(c.rsplit("_", 1)[1]))
            train = meta.merge(embc[data.JOIN + emb_cols], on=data.JOIN,
                               how="inner")
            dec = RegretDecoder(random_state=seed).fit(
                train, features=sorted(emb_cols))
            encoder, ck = load_encoder(data.encoder_checkpoint(seed))
            sample_rows = int(ck["config"].get("sample_rows", 512))
        elif arm == "manual":
            dec = RegretDecoder(random_state=seed).fit(meta)
            encoder, sample_rows, emb_cols_by_dim = None, None, None
        else:
            raise ValueError(f"arm must be 'embed' or 'manual', got {arm!r}")
        _PIPELINES[key] = (dec, heads, encoder, sample_rows, emb_cols_by_dim)
    return _PIPELINES[key]


def recommend(X, y, arm: str = "embed", seed: int = 0,
              attain_r2: float | None = None) -> Report:
    """Recommend surrogate families for (X, y) before fitting any.

    How many of the ranked candidates to fit is the caller's move: fit them
    in ranking order with `Candidate.fit` and stop when satisfied.
    `report.regret_at_k` carries the published sealed-suite regret of
    stopping after one, two, or three. Below roughly 100 training samples no
    single pick is trustworthy; fit at least the top three and read the
    attainability estimate.

    attain_r2 is the R² you consider useful for your application: the
    report is rejected when the best calibrated estimate falls below it.
    This is your decision threshold, not a validated one, so set it from what
    the surrogate is for; the default (0.5) is the low-attainability band the
    paper uses descriptively. Remember the estimate carries the band quoted on
    every candidate, so a threshold within that band of the estimate decides
    little.
    """
    X, y = _as_arrays(X, y)
    if X.shape[0] < 10 or X.shape[1] < 1:
        raise ValueError("need at least 10 rows and 1 feature")
    if attain_r2 is not None and not -1.0 <= float(attain_r2) <= 1.0:
        raise ValueError("attain_r2 must be an R² value in [-1, 1]")

    dec, heads, encoder, sample_rows, emb_cols_by_dim = _pipeline(arm, seed)
    mfs = extract(X, y)
    query_mf = pd.DataFrame([mfs])
    if arm == "embed":
        v = embed(X, y, encoder, sample_rows=sample_rows)
        query = pd.DataFrame([{c: float(v[i])
                               for i, c in enumerate(emb_cols_by_dim)}])
        query["n_train"] = float(len(y))
    else:
        query = query_mf

    raw = dec.predict_r2(query).iloc[0]
    shift, band = shift_and_band(arm, seed)
    calibrated = raw + shift
    p_fail = heads.p_fail(query_mf)

    cands = [Candidate(family=f, predicted_r2=float(calibrated[f]), band=band,
                       p_fail=p_fail.get(f),
                       vetoed=p_fail.get(f, 0.0) > heads.tau,
                       warm_start=lookup(f, len(y), X.shape[1]),
                       available=f in AVAILABLE)
             for f in calibrated.index]
    cands.sort(key=lambda c: (c.vetoed, -c.predicted_r2))

    pub = published_metrics()
    attain = max(c.predicted_r2 for c in cands)
    floor = (float(pub["advice"]["reject_attainability"])
             if attain_r2 is None else float(attain_r2))
    small_n = int(pub["advice"]["small_sample_n"])

    advice = ""
    if len(y) < small_n:
        advice = (f"n < {small_n}: no single pick is trustworthy in this "
                  "regime; fit at least the top 3 families and "
                  "treat a low attainability estimate as a signal to collect "
                  "data rather than to model harder.")
    absent = [c.family for c in cands[:3] if not c.available]
    if absent:
        advice = (f"{', '.join(absent)} rank in your top 3 but the "
                  "package is not installed: pip install 'surcla[lgbm,xgb]' "
                  "to fit them. " + advice).strip()
    if attain < floor:
        advice = (f"predicted attainability {attain:.2f} is below your "
                  f"attain_r2={floor:g}: no family is expected to reach a "
                  "useful fit on this data. " + advice).strip()

    return Report(candidates=cands, attainability=float(attain),
                  reject=bool(attain < floor), attain_r2=floor,
                  n_train=int(len(y)),
                  regret_at_k={int(kk): v for kk, v in
                               pub["sealed_v2"]["median_regret_at_k"].items()},
                  advice=advice, arm=arm, seed=seed)
