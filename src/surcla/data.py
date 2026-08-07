"""Bundled artifact access: training tables, embeddings, encoder checkpoints."""

from importlib import resources
from pathlib import Path

import pandas as pd

JOIN = ["dataset", "size_key", "seed_key"]


def artifact_path(name: str) -> Path:
    return Path(resources.files("surcla") / "artifacts" / name)


def load_corpus_table() -> pd.DataFrame:
    """The corpus_v5_full meta table: metafeatures + per-family R² labels."""
    return pd.read_parquet(artifact_path("meta_manual_corpus_v5_full.parquet"))


def load_corpus_embeddings(seed: int = 0) -> pd.DataFrame:
    """Corpus embeddings of the deployed encoder (join on JOIN columns)."""
    return pd.read_parquet(artifact_path(f"d2v_emb_corpus_s{seed}.parquet"))


def encoder_checkpoint(seed: int = 0) -> Path:
    return artifact_path(f"d2v_v3_s{seed}.pt")
