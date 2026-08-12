# Reproducing the paper

The paper's pipeline has four stages, and they cost wildly different amounts to
re-run. This directory reproduces the last one from artifacts that ship in the
repository; the earlier three need the research repository, and two of them
need a cluster.

| Stage | What it produces | Cost | Reproducible here |
|---|---|---|---|
| Corpus generation | 7,995 synthetic datasets from three generator priors (structural causal models, a fitted expression grammar, tree-based) | hours | no |
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
suite. It runs from artifacts shipped in this repository; the validation
datasets themselves are not redistributed (the CTR23 tasks come from OpenML
under their own licences, the VLSE functions are analytic and listed in the
paper), and everything upstream of the artifacts lives in the research
repository.
