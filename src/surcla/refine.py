"""A short local search around a fitted configuration, scored on your data.

This is not hyperparameter optimization. There is no model of the search
space, no adaptive sampling and no convergence claim: `refine` evaluates a
handful of neighbours of the configuration `Candidate.fit` used, under the
same out-of-fold R², and returns whichever scored best, which is the incumbent
unless something beat it.

What the budget buys, per family:

    Kriging, PCE, PCK   the sibling variants (four, three, four). A budget
                        that covers them exhausts the neighbourhood; there is
                        nothing finer to try.
    RF, LGBM, XGB, MLP  one coordinate at a time, halved and doubled, over the
                        parameters whose sensible range is unbounded. This
                        locates the right order of magnitude, not the right
                        value.
    Ridge, Lasso        a five-point log grid over alpha.
    LR                  nothing. Ordinary least squares has no knob.

Cost is one cross-validation per configuration, so roughly `budget * cv` fits
plus one refit of the winner on all your rows. Asking for more budget than the
neighbourhood holds is free: the search stops when it runs out of neighbours.

What `cv_gain` is worth: it is the winner's score minus the incumbent's, both
measured on the folds the search itself used, so it is not an out-of-sample
estimate of anything. The winner was chosen by maximizing the number that
reports it, which biases that number upward, and the bias grows as n shrinks.
Refining a random forest on 60 rows of pure noise, where the true gain is zero
by construction, reports about +0.06 R² on average. The warning below `SMALL_N`
rows flags the worst of this, not all of it; a held-out set is the only way to
confirm a gain.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np

from .families import DEFAULT_VARIANT, FAMILIES, VARIANT_TAGS, build
from .recommend import SMALL_N, FittedSurrogate, _as_arrays, cross_val_r2

# Used only when the incoming model carries no usable fold count.
DEFAULT_CV = 5

# Unbounded parameters, where multiplying explores the order of magnitude.
# Bounded fractions (max_features, subsample, colsample_*) are left alone:
# doubling one leaves its valid range, and halving it is a shot in the dark.
SCALE_KEYS = ("learning_rate", "lr", "reg_alpha", "reg_lambda")
COUNT_KEYS = ("batch_size", "epochs", "max_leaves", "n_estimators",
              "num_leaves")
MULTIPLIERS = (0.5, 2.0)

ALPHA_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
ALPHA_FAMILIES = ("Ridge", "Lasso")


def _tag(family: str, config) -> str:
    """The variant tag a UQ configuration names, the family's default if none."""
    if config is None:
        config = DEFAULT_VARIANT[family]
    if not isinstance(config, str):
        raise TypeError(f"{family}: the configuration is a variant tag such as "
                        f"{VARIANT_TAGS[family][0]!r}, not a "
                        f"{type(config).__name__}")
    return config.removeprefix(family)


def _neighbours(family: str, config) -> list:
    """Configurations one step from `config`, in a deterministic order.

    The incumbent is never among them: it has already been scored.
    """
    if family in VARIANT_TAGS:
        current = _tag(family, config)
        return [t for t in VARIANT_TAGS[family] if t != current]

    if family in ALPHA_FAMILIES:
        base = dict(config or {})
        current = base.get("alpha", build(family, base or None).alpha)
        return [{**base, "alpha": a} for a in ALPHA_GRID if a != current]

    if not isinstance(config, dict):
        return []

    # Sweep every parameter once at one multiplier before returning to any of
    # them, so a budget smaller than the neighbourhood spreads across
    # coordinates instead of exhausting the alphabetically first one.
    out = []
    for multiplier in MULTIPLIERS:
        for key in sorted(config):
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if key in SCALE_KEYS and value > 0:
                out.append({**config, key: value * multiplier})
            elif key in COUNT_KEYS:
                moved = max(1, int(round(value * multiplier)))
                if moved != value:
                    out.append({**config, key: moved})
    return out


def _better(candidate: float, incumbent: float) -> bool:
    """A nan score never wins, and any finite score beats a nan incumbent."""
    if not np.isfinite(candidate):
        return False
    return not np.isfinite(incumbent) or candidate > incumbent


def refine(fitted: FittedSurrogate, X, y, budget: int = 8,
           cv: int | None = None) -> FittedSurrogate:
    """Try a few neighbours of `fitted`'s configuration and keep the best.

    A few extra fits, not a search: see the module docstring for what the
    budget buys in each family, and for why `cv_gain` is an optimistic number.

    Parameters
    ----------
    fitted : FittedSurrogate
        The starting point, as returned by `Candidate.fit`.
    X, y : array-like
        The same data the model was fitted on; a different row count is an
        error rather than a silent rescoring. Any grouping and any input caps
        given to `Candidate.fit` are reused, so replicates stay together and
        every configuration is judged under the same bounds.
    budget : int
        Maximum number of configurations evaluated, the incumbent included, so
        the default buys seven alternatives.
    cv : int or None
        Folds for the out-of-fold R², with the small-n fallbacks of
        `cross_val_r2`. The incumbent is rescored here rather than trusted from
        `fitted.cv_r2`, so every configuration is judged by the same scorer.
        None, the default, reuses the fold count behind `fitted.cv_r2`, which
        keeps the returned score comparable with the one you were shown; force
        a different count and the returned `cv_r2` moves for that reason alone,
        which `cv_note` then records.

    Returns
    -------
    FittedSurrogate
        Refitted on all of (X, y) when a neighbour won, otherwise the model
        passed in. Its `cv_r2` is never below the rescored incumbent's,
        `cv_gain` records that difference and `n_configs_tried` the number of
        configurations evaluated.
    """
    if budget < 1:
        raise ValueError("budget must be at least 1")
    X, y = _as_arrays(X, y)
    family = fitted.family
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from "
                         f"{list(FAMILIES)}")
    if fitted.n_train and len(y) != fitted.n_train:
        raise ValueError(
            f"refine needs the rows the model was fitted on: {family} was "
            f"fitted on {fitted.n_train} rows and {len(y)} were passed. "
            "Scoring one dataset to pick a configuration for another compares "
            "nothing; refit with Candidate.fit on the data you mean.")
    build(family, fitted.config)     # the registry's error, before any fitting
    if cv is None:
        cv = fitted.cv_folds if fitted.cv_folds >= 2 else DEFAULT_CV
    groups = fitted.groups
    max_rows, max_features = fitted.caps
    start_r2, folds, note = cross_val_r2(family, fitted.config, X, y, cv,
                                         groups, max_rows, max_features)
    best_r2, best_config, best_note, tried = start_r2, fitted.config, note, 1

    for candidate in _neighbours(family, fitted.config)[:budget - 1]:
        r2, _, cand_note = cross_val_r2(family, candidate, X, y, cv, groups,
                                        max_rows, max_features)
        tried += 1
        if _better(r2, best_r2):
            best_r2, best_config, best_note = r2, candidate, cand_note

    if not np.isfinite(best_r2):
        warnings.warn(f"{family}: no configuration produced a defined "
                      f"cross-validated R² ({best_note}); returning the model "
                      "unchanged", UserWarning, stacklevel=2)

    if best_config == fitted.config:
        model = fitted.model                # already fitted on all of (X, y)
    else:
        model = build(family, best_config, max_rows, max_features).fit(X, y)

    gain = (best_r2 - start_r2
            if np.isfinite(best_r2) and np.isfinite(start_r2) else 0.0)
    if gain > 0.0 and len(y) < SMALL_N:
        warnings.warn(f"{family}: the {gain:+.3f} cv_r2 gain was measured on "
                      "the same folds that chose the configuration, and at "
                      f"n = {len(y)} rows a gain of that size can be selection "
                      "noise on its own; confirm it on held-out data",
                      UserWarning, stacklevel=2)

    notes = [best_note] if best_note else []
    if fitted.cv_folds >= 2 and folds != fitted.cv_folds:
        notes.append(f"rescored at {folds} folds, not the {fitted.cv_folds} "
                     "behind the cv_r2 you were shown")
    return replace(fitted, model=model, config=best_config, cv_r2=best_r2,
                   cv_folds=folds, cv_note="; ".join(notes),
                   n_configs_tried=tried, cv_gain=float(gain))
