# Reproducing the paper

The paper's pipeline has four stages, and they cost wildly different amounts to
re-run. This directory reproduces the last one from artifacts that ship in the
repository; the earlier three need the research repository, and two of them
need a cluster.

| Stage | What it produces | Cost | Reproducible here |
|---|---|---|---|
| Corpus generation | 7,995 synthetic datasets from three generator priors (structural causal models, a fitted expression grammar, tree-based) | hours | no, see below |
| Labelling | every family fitted on every cell, 23,985 corpus cells plus the validation suites | ~10k CPU-hours, SLURM array | no |
| Encoder training | the three Dataset2Vec v3 checkpoints | ~1 GPU-day each | no, but the checkpoints ship |
| Evaluation | the sealed table, the splits, the bootstrap intervals | ~4 minutes | **yes** |

## The sealed evaluation

```
python reproduce/paper_numbers.py
```

Refits all four arms on all three seeds from the shipped corpus table and
embeddings, runs them against validation_v2, and compares every result against
the recorded run and against the figures the paper reports. Under the
scikit-learn that recorded the artifacts (1.5.2) every number agrees to 1e-9,
being the same computation on the same inputs. A different scikit-learn builds
marginally different random forests from the same seed, and because the top
families are near-tied on roughly half the cells, a marginally different
forest flips a few picks: accuracies move in steps of 1/486, the rank metrics
in the third decimal. That variation is the paper's near-tie structure showing
itself, not an artifact problem, so the script prints those differences as
`~` within per-metric drift bands and still exits zero. Only differences
beyond the bands exit non-zero, which keeps the script a regression test on
the artifacts. Across 1.5.2 to 1.9.0 we measured at most 7/486 in the
accuracies and 0.011 in Spearman, against bands of 0.025 and 0.02.

It checks the four-arm sealed table, the deployed arm's headline figures
(top-1 and top-3 accuracy, median regret at k = 1 and 2, rank Spearman,
accuracy within 0.01 R²), the CTR23 versus VLSE split, and the shape of the
suite.

## What ships for it

`reproduce/artifacts/` holds the validation side: the metafeature-and-label
tables for validation_v1 (360 cells, the development suite) and validation_v2
(486 cells, sealed), each seed's embeddings for both, and the recorded results
to compare against. The corpus side lives in `src/surcla/artifacts/`, since the
package needs it at runtime anyway.

These are derived tables, metafeatures and per-family R² measurements. The
underlying datasets are not redistributed here: the 15 CTR23 tasks come from
OpenML under their own licences, and the 12 VLSE functions are analytic and
listed in the paper, sampled at 10,000 points with seed 0.

## What is not here, and where it lives

Corpus generation, the labelling harness, the SLURM scripts, encoder training,
and the scripts behind every figure stay in the research repository, which is
the lab notebook the paper pins as a submodule. That repository is also where
the label caps live that this package now reproduces at fit time: Kriging, PCE
and PC-Kriging were labelled on at most 2,000 rows and 30 features.

Two upstream properties are worth stating plainly, because they bound what
reproduction can mean. Labelling is stochastic in the sampling seed, so a fresh
labelling run of the same corpus would not return bit-identical R² values; the
paper's three-seed reporting exists for that reason. And encoder training was
run once per seed under a pre-registered protocol, with misses recorded rather
than retried, so re-running it explores the same distribution rather than
retracing the same path.
