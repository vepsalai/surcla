"""Recompute the paper's sealed-suite numbers from the artifacts shipped here.

Everything below is rebuilt from scratch: the decoder is refitted on the corpus
table, run against validation_v2, and the results are compared against the
values the paper reports. Nothing is read from a cached result except as the
thing being checked.

    python reproduce/paper_numbers.py

Under the scikit-learn that recorded the artifacts (1.5.2) every number
reproduces to 1e-9, being the same computation on the same inputs. A different
scikit-learn builds marginally different forests from the same seed, and on a
suite where the top families are near-tied on half the cells, a marginally
different forest flips a few picks: accuracies then move in steps of 1/486
and the rank metrics in the third decimal. Differences inside the per-metric
drift bands below are printed as '~' and the script still exits zero; anything
beyond them exits non-zero, which keeps this a regression test on the
artifacts as much as a reproduction script.

What this cannot check is upstream of the artifacts: the corpus generation,
the labelling runs and the encoder training. Those need the research
repository and a cluster; see reproduce/README.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr

from surcla import data
from surcla.calibration import published_metrics
from surcla.decoder import RegretDecoder, feature_columns, regret_of

ART = Path(__file__).parent / "artifacts"
SEEDS = [0, 1, 2]
JOIN = ["dataset", "size_key", "seed_key"]
ARMS = [("synthetic", "manual"), ("synthetic", "embed"),
        ("real-v1", "manual"), ("real-v1", "embed")]
EXACT = 1e-9
RECORDED_SKLEARN = "1.5.2"

# Acceptable drift when scikit-learn differs from the recorded version. The
# bands sit well above the drift measured across 1.5.2 -> 1.9.0 (at most
# 7/486 in the accuracies, 0.011 in Spearman, 0.005 in mean regret) and well
# below the seed-to-seed spread the paper reports, so a real artifact problem
# still fails loudly.
DRIFT = {"acc": 0.025, "top3_acc": 0.025, "acc_within_0p01": 0.025,
         "rank_spearman": 0.02, "mean_regret": 0.01, "median_regret": 0.002}


def load_validation(tag: str, seed: int) -> pd.DataFrame:
    """Validation meta table joined with that seed's embeddings."""
    meta = pd.read_parquet(ART / f"meta_manual_validation_{tag}.parquet")
    emb = pd.read_parquet(ART / f"d2v_emb_validation_{tag}_s{seed}.parquet")
    emb_cols = [c for c in emb.columns if c.startswith("emb_")]
    joined = meta.merge(emb[JOIN + emb_cols], on=JOIN, how="inner",
                        validate="1:1")
    assert len(joined) == len(emb), "join lost rows"
    return joined


def load_corpus(seed: int) -> pd.DataFrame:
    meta = data.load_corpus_table()
    emb = data.load_corpus_embeddings(seed)
    emb_cols = [c for c in emb.columns if c.startswith("emb_")]
    return meta.merge(emb[JOIN + emb_cols], on=JOIN, how="inner")


def rank_correlation(pred: pd.DataFrame, meta: pd.DataFrame) -> float:
    """Mean per-cell Spearman between predicted and true family R2 vectors."""
    fams = list(pred.columns)
    rhos = []
    for (_, p), (_, row) in zip(pred.iterrows(), meta.iterrows()):
        true = np.array([row.get(f"r2_{f}", np.nan) for f in fams], dtype=float)
        valid = ~np.isnan(true)
        if valid.sum() >= 3:
            rho = spearmanr(p.to_numpy()[valid], true[valid]).statistic
            if not np.isnan(rho):
                rhos.append(rho)
    return float(np.mean(rhos))


def best_of_k_regret(pred: pd.DataFrame, meta: pd.DataFrame, k: int):
    """Regret when the k highest-ranked families are fitted and the best kept."""
    fams = np.array(pred.columns)
    order = np.argsort(-pred.to_numpy(), axis=1)
    out, hit = [], []
    for row_order, (_, row) in zip(order, meta.iterrows()):
        picks = fams[row_order[:k]]
        r2s = np.array([row.get(f"r2_{p}", np.nan) for p in picks], dtype=float)
        out.append(1.0 if np.all(np.isnan(r2s))
                   else float(np.clip(row["winner_r2"] - np.nanmax(r2s), 0, 1)))
        hit.append(row["winner"] in set(picks))
    return np.asarray(out), float(np.mean(hit))


def evaluate(dec: RegretDecoder, val: pd.DataFrame) -> dict:
    """The metrics the paper's final table reports, for one fitted arm."""
    pred = dec.predict_r2(val)
    rec = pred.idxmax(axis=1)
    reg = regret_of(rec, val)
    out = {"acc": float(np.mean(rec.to_numpy() == val["winner"].to_numpy())),
           "median_regret": float(np.median(reg)),
           "mean_regret": float(reg.mean()),
           "rank_spearman": rank_correlation(pred, val),
           "acc_within_0p01": float(np.mean(reg <= 0.01))}
    for k in (2, 3):
        regk, acck = best_of_k_regret(pred, val, k)
        out[f"top{k}_acc"] = acck
        out[f"top{k}_median_regret"] = float(np.median(regk))
        out[f"top{k}_mean_regret"] = float(regk.mean())
    return out, rec, reg


def check(label: str, got: float, want: float, drift: float,
          failures: list, drifted: list, exact: float = EXACT) -> None:
    """Three verdicts: exact ('ok'), within the drift band ('~'), or FAIL.

    `exact` widens only the first tier, for section-2 checks against the
    paper's rounded figures, where agreement to the printed digits is the
    strongest statement available.
    """
    delta = abs(got - want)
    if delta <= exact:
        flag = "ok "
    elif delta <= drift:
        flag = " ~ "
        drifted.append(label)
    else:
        flag = "FAIL"
        failures.append(f"{label}: {got} vs published {want} "
                        f"(drift band {drift})")
    print(f"  {flag} {label:<34} {got:>9.4f}  published {want:>9.4f}")


def main() -> int:
    failures: list[str] = []
    drifted: list[str] = []
    reference = pd.read_csv(ART / "final_table_v2.csv")
    published = published_metrics()

    print(f"scikit-learn {sklearn.__version__} "
          f"(artifacts recorded with {RECORDED_SKLEARN})")
    if sklearn.__version__ != RECORDED_SKLEARN:
        print("A different scikit-learn builds marginally different forests\n"
              "from the same seed, so a few near-tied cells flip their pick:\n"
              "small differences are expected, printed as '~', and explained\n"
              "in reproduce/README.md.")
    print("\nRefitting every arm on the shipped artifacts "
          f"({len(ARMS)} arms x {len(SEEDS)} seeds)\n")
    rows, per_cell = [], {}
    for seed in SEEDS:
        corpus = load_corpus(seed)
        emb_cols = sorted(c for c in corpus.columns if c.startswith("emb_"))
        for tr_name in ("synthetic", "real-v1"):
            train = (corpus if tr_name == "synthetic"
                     else load_validation("v1", seed))
            val = load_validation("v2", seed)
            val = val[val["winner"].notna()].reset_index(drop=True)
            manual_feats = sorted(
                (set(feature_columns(train)) & set(feature_columns(val)))
                - set(emb_cols))
            for enc, feats in (("manual", manual_feats), ("embed", emb_cols)):
                dec = RegretDecoder(random_state=seed).fit(train, features=feats)
                res, rec, reg = evaluate(dec, val)
                rows.append({"train": tr_name, "encoder": enc, "seed": seed,
                             **res})
                per_cell[(tr_name, enc, seed)] = (val, reg)
        print(f"  seed {seed} done", flush=True)
    got = pd.DataFrame(rows)

    print("\n1. Sealed table, per arm and seed, against the recorded run")
    for _, ref in reference.iterrows():
        mine = got[(got.train == ref["train"]) & (got.encoder == ref["encoder"])
                   & (got.seed == ref["seed"])].iloc[0]
        for metric in ("acc", "median_regret", "mean_regret", "rank_spearman",
                       "top3_acc"):
            if metric in ref:
                check(f"{ref['train']}/{ref['encoder']} s{ref['seed']} {metric}",
                      mine[metric], float(ref[metric]), DRIFT[metric],
                      failures, drifted)

    print("\n2. Headline figures of the deployed arm, against the paper")
    dep = got[(got.train == "synthetic") & (got.encoder == "embed")]
    sealed = published["sealed_v2"]
    check("top-1 accuracy", dep.acc.mean(), sealed["acc_top1"]["mean"],
          DRIFT["acc"], failures, drifted, exact=5e-4)
    check("top-3 accuracy", dep.top3_acc.mean(), sealed["acc_top3"]["mean"],
          DRIFT["top3_acc"], failures, drifted, exact=5e-4)
    check("median regret k=1", dep.median_regret.mean(),
          sealed["median_regret_at_k"]["1"], DRIFT["median_regret"],
          failures, drifted, exact=5e-4)
    check("median regret k=2", dep.top2_median_regret.mean(),
          sealed["median_regret_at_k"]["2"], DRIFT["median_regret"],
          failures, drifted, exact=5e-4)
    check("rank Spearman", dep.rank_spearman.mean(),
          sealed["rank_spearman"]["mean"], DRIFT["rank_spearman"],
          failures, drifted, exact=5e-4)
    check("accuracy within 0.01 R2", dep.acc_within_0p01.mean(),
          sealed["acc_within_0p01_r2"]["mean"], DRIFT["acc_within_0p01"],
          failures, drifted, exact=5e-3)

    print("\n3. CTR23 versus VLSE split of the deployed arm")
    split_ref = pd.read_csv(ART / "sealed_source_split.csv")
    for source in ("CTR23", "VLSE"):
        med = []
        for seed in SEEDS:
            val, reg = per_cell[("synthetic", "embed", seed)]
            vlse = val["dataset"].str.startswith("vlse_").to_numpy()
            mask = vlse if source == "VLSE" else ~vlse
            med.append(float(np.median(reg[mask])))
        want = split_ref[(split_ref.train == "synthetic")
                         & (split_ref.encoder == "embed")
                         & (split_ref.source == source)]["median_regret"].mean()
        check(f"{source} median regret", float(np.mean(med)), want,
              DRIFT["median_regret"], failures, drifted)

    print("\n4. Suite shape, against the per-dataset table")
    per_ds_ref = pd.read_csv(ART / "per_dataset_v2.csv")
    val0, _ = per_cell[("synthetic", "embed", 0)]
    check("datasets covered", float(val0["dataset"].nunique()),
          float(len(per_ds_ref)), 0.0, failures, drifted)
    check("cells scored", float(len(val0)), 486.0, 0.0, failures, drifted)

    print()
    if failures:
        print(f"{len(failures)} MISMATCH(ES) beyond the drift bands:")
        for f in failures:
            print(f"  - {f}")
        return 1
    if drifted:
        print(f"no mismatches: {len(drifted)} number(s) drifted within the "
              "near-tie band\nexpected across scikit-learn versions, the "
              "rest match exactly")
        return 0
    print("every recomputed number matches what the paper reports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
