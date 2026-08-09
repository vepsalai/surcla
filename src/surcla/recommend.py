"""The recommendation report: ranked families, calibrated R², failure risk.

`recommend(X, y, k)` runs the deployed pipeline of the paper: the corpus-
trained Dataset2Vec v3 encoder (eight-draw embedding) feeding a random-forest
regret decoder trained on the shipped corpus table, with the published
constant-shift calibration, the failure heads' veto-by-demotion, and the
small-sample advice the sealed evaluation established. First call fits the
decoder and the heads from the bundled tables (roughly a minute); later calls
cost milliseconds plus the embedding. Deterministic: the same (X, y, k) on
the same artifact version yields the same report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data
from .calibration import published_metrics, shift_and_band
from .d2v import embed, load_encoder
from .decoder import RegretDecoder
from .failure import FailureHeads
from .metafeatures import extract
from .warmstart import WarmStart, lookup

_PIPELINES: dict = {}


@dataclass
class Candidate:
    family: str
    predicted_r2: float          # calibrated estimate of attainable R²
    band: float                  # published median |error| of that estimate
    p_fail: float | None         # failure probability (fragile families only)
    vetoed: bool                 # p_fail > tau: demoted to the bottom
    warm_start: WarmStart | None = None   # configuration to fit first

    def __repr__(self):
        veto = "  VETOED" if self.vetoed else ""
        pf = f"  p_fail={self.p_fail:.2f}" if self.p_fail is not None else ""
        return (f"{self.family}: predicted R² {self.predicted_r2:.3f} "
                f"± {self.band:.2f}{pf}{veto}")


@dataclass
class Report:
    candidates: list[Candidate]  # all families, ranked; vetoed ones last
    attainability: float         # best calibrated predicted R²
    reject: bool                 # attainability below attain_r2
    attain_r2: float             # the usefulness threshold in force
    n_train: int
    k: int
    regret_at_k: dict            # published sealed-suite regret at k = 1..3
    advice: str
    arm: str
    seed: int

    def __repr__(self):
        head = (f"SurCla report (n={self.n_train}, arm={self.arm}): "
                f"attainability {self.attainability:.3f}"
                + (f", REJECT (below attain_r2={self.attain_r2:g})"
                   if self.reject else ""))
        lines = [head] + [f"  {i + 1}. {c!r}"
                          for i, c in enumerate(self.candidates[:max(self.k, 3)])]
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


def recommend(X, y, k: int = 3, arm: str = "embed", seed: int = 0,
              attain_r2: float | None = None) -> Report:
    """Recommend surrogate families for (X, y) before fitting any.

    k is the regret appetite: how many top-ranked families you intend to fit
    (the published sealed-suite regret at each k ships in the report). Below
    roughly 100 training samples no single pick is trustworthy; fit at least
    the top ranked families and read the attainability estimate.

    attain_r2 is the $R^2$ you consider useful for your application: the
    report is rejected when the best calibrated estimate falls below it.
    This is your decision threshold, not a validated one, so set it from what
    the surrogate is for; the default (0.5) is the low-attainability band the
    paper uses descriptively. Remember the estimate carries the band quoted on
    every candidate, so a threshold within that band of the estimate decides
    little.
    """
    if k not in (1, 2, 3):
        raise ValueError("k must be 1, 2, or 3")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2 or X.shape[0] != len(y):
        raise ValueError(f"expected X (n, d) and y (n,); got {X.shape}, {y.shape}")
    if X.shape[0] < 10 or X.shape[1] < 1:
        raise ValueError("need at least 10 rows and 1 feature")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("X and y must be finite (drop or impute NaNs first)")
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
                       warm_start=lookup(f, len(y), X.shape[1]))
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
                  f"regime; fit at least the top {max(k, 3)} families and "
                  "treat a low attainability estimate as a signal to collect "
                  "data rather than to model harder.")
    if attain < floor:
        advice = (f"predicted attainability {attain:.2f} is below your "
                  f"attain_r2={floor:g}: no family is expected to reach a "
                  "useful fit on this data. " + advice).strip()

    return Report(candidates=cands, attainability=float(attain),
                  reject=bool(attain < floor), attain_r2=floor,
                  n_train=int(len(y)), k=k,
                  regret_at_k={int(kk): v for kk, v in
                               pub["sealed_v2"]["median_regret_at_k"].items()},
                  advice=advice, arm=arm, seed=seed)
