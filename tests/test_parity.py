"""Golden tests: the shipped pipeline reproduces the research pipeline.

golden.npz holds two small validation slices (n <= sample_rows, the regime
with no row sampling and therefore bit-exact embeddings) with the embeddings
the research extraction recorded for them.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from surcla import data
from surcla.d2v import embed, load_encoder
from surcla.decoder import RegretDecoder
from surcla.metafeatures import extract

GOLD = np.load(Path(__file__).parent / "golden.npz")


def test_encoder_checkpoint_loads():
    model, ck = load_encoder(data.encoder_checkpoint(0))
    assert ck["config"]["arch"] == "v3"
    assert sum(p.numel() for p in model.parameters()) > 100_000


@pytest.mark.parametrize("cell", ["cell_small_a", "cell_small_b"])
def test_embedding_golden(cell):
    model, ck = load_encoder(data.encoder_checkpoint(0))
    v = embed(GOLD[f"{cell}_X"], GOLD[f"{cell}_y"], model,
              sample_rows=int(ck["config"]["sample_rows"]),
              cache_seed=int(GOLD[f"{cell}_seed"]))
    assert np.abs(v - GOLD[f"{cell}_emb"]).max() < 1e-5


def test_metafeature_schema():
    meta = extract(GOLD["cell_small_a_X"], GOLD["cell_small_a_y"])
    assert len(meta) == 52          # 51 descriptors + n_train
    assert meta["n_train"] == float(len(GOLD["cell_small_a_y"]))
    assert "spearman_r__max" in meta and "y_mean5__bin0" in meta


def test_decoder_end_to_end():
    """Deployed configuration: corpus table + embeddings s0, recommend a toy
    dataset. Slow (~20 s: one forest fit)."""
    meta = data.load_corpus_table()
    embc = data.load_corpus_embeddings(0)
    emb_cols = [c for c in embc.columns if c.startswith("emb_")]
    train = meta.merge(embc[data.JOIN + emb_cols], on=data.JOIN, how="inner")
    dec = RegretDecoder(random_state=0).fit(train, features=sorted(emb_cols))

    model, ck = load_encoder(data.encoder_checkpoint(0))
    v = embed(GOLD["cell_small_a_X"], GOLD["cell_small_a_y"], model,
              sample_rows=int(ck["config"]["sample_rows"]),
              cache_seed=int(GOLD["cell_small_a_seed"]))
    q = pd.DataFrame([{**dict(zip(emb_cols, v)),
                       "n_train": float(len(GOLD["cell_small_a_y"]))}])
    rec = dec.recommend(q)
    assert rec["family"].iloc[0] in {"Kriging", "PCE", "PCK", "LGBM", "XGB",
                                     "RF", "MLP", "LR", "Ridge", "Lasso"}
    assert -1.0 <= rec["predicted_r2"].iloc[0] <= 1.0
