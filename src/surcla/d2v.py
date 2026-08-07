"""Dataset2Vec v3 encoder, inference only.

The architecture is an exact port of the research encoder (cross-feature
attention within rows, 4-channel value+rank tokens, K-query attention pooling
over rows, a 32-dimensional bottleneck, and a [log n, log d, log n/d]
side-channel at the head); training code stays in the research repository.
`embed` replicates the deployed eight-draw evaluation bit for bit: per-slice
standardization, clip to ±10, cache of up to 1024 rows, eight seeded draws of
`sample_rows` rows, ranks computed within each draw, embeddings averaged.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata

CACHE_ROWS = 1024
DRAWS = 8


def mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(),
                         nn.Linear(d_hidden, d_out), nn.GELU())


class AttentionBlock(nn.Module):
    """Pre-norm transformer encoder block: masked self-attention + FFN."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                 nn.Linear(2 * width, width))

    def forward(self, s, key_padding_mask):
        h = self.norm1(s)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        s = s + a
        return s + self.ffn(self.norm2(s))


class Dataset2VecV3(nn.Module):
    """Set encoder over (value, target, rank, target-rank) cell tokens."""

    def __init__(self, n_families: int, width: int = 64, blocks: int = 2,
                 heads: int = 4, k_queries: int = 4, bottleneck: int = 32):
        super().__init__()
        self.f = mlp(4, width, width)
        self.row_blocks = nn.ModuleList(AttentionBlock(width, heads)
                                        for _ in range(blocks))
        self.row_queries = nn.Parameter(torch.randn(1, k_queries, width) / width ** 0.5)
        self.row_pool = nn.MultiheadAttention(width, heads, batch_first=True)
        self.row_proj = nn.Linear(k_queries * width, width)
        self.feat_block = AttentionBlock(width, heads)
        self.pool_query = nn.Parameter(torch.randn(1, 1, width) / width ** 0.5)
        self.pool_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.pool_norm = nn.LayerNorm(width)
        self.neck = nn.Sequential(nn.Linear(width, bottleneck), nn.GELU())
        self.head = nn.Sequential(nn.Linear(bottleneck + 3, width), nn.GELU(),
                                  nn.Linear(width, n_families))

    def _embed(self, x, y, xr, yr, row_mask, feat_mask, log_n):
        B, n, d = x.shape
        tokens = torch.stack([x, y.unsqueeze(-1).expand_as(x),
                              xr, yr.unsqueeze(-1).expand_as(xr)], dim=-1)
        t = self.f(tokens)                                     # (B, n, d, w)

        # Cross-feature attention within each row.
        t = t.reshape(B * n, d, -1)
        pad = (feat_mask == 0).repeat_interleave(n, dim=0)     # (B*n, d)
        for block in self.row_blocks:
            t = block(t, key_padding_mask=pad)
        t = t.reshape(B, n, d, -1)

        # K-query attention pool over rows, per feature.
        t = t.permute(0, 2, 1, 3).reshape(B * d, n, -1)        # (B*d, n, w)
        row_pad = (row_mask == 0).unsqueeze(1).expand(B, d, n).reshape(B * d, n)
        q = self.row_queries.expand(B * d, -1, -1)
        u, _ = self.row_pool(q, t, t, key_padding_mask=row_pad,
                             need_weights=False)               # (B*d, K, w)
        u = self.row_proj(u.reshape(B, d, -1))                 # (B, d, w)

        u = self.feat_block(u, key_padding_mask=(feat_mask == 0))
        q = self.pool_query.expand(B, -1, -1)
        v, _ = self.pool_attn(q, u, u, key_padding_mask=(feat_mask == 0),
                              need_weights=False)
        v = self.neck(self.pool_norm(v.squeeze(1)))            # (B, bottleneck)

        log_d = torch.log10(feat_mask.sum(1, keepdim=True).clamp(min=1.0))
        return v, torch.cat([log_n, log_d, log_n - log_d], dim=1)

    def forward(self, *args):
        v, side = self._embed(*args)
        return self.head(torch.cat([v, side], dim=1))


def load_encoder(path: str | Path) -> tuple[nn.Module, dict]:
    """Load a v3 checkpoint; returns (model in eval mode, checkpoint dict)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    config = ck["config"]
    if config.get("arch") != "v3":
        raise ValueError(f"expected a v3 checkpoint, got arch={config.get('arch')!r}")
    model = Dataset2VecV3(n_families=len(ck["fam_cols"]),
                          width=config.get("width", 64),
                          blocks=config.get("blocks", 2),
                          heads=config.get("heads", 4),
                          k_queries=config.get("k_queries", 4),
                          bottleneck=config.get("bottleneck", 32))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def embed(X: np.ndarray, y: np.ndarray, model: nn.Module, sample_rows: int,
          draws: int = DRAWS, cache_rows: int = CACHE_ROWS,
          cache_seed: int = 0) -> np.ndarray:
    """Deployed embedding of one dataset: eight-draw averaged bottleneck vector.

    Replicates the research pipeline exactly: cache up to `cache_rows` rows
    (seeded subsample), standardize per slice, clip to ±10; per draw, sample
    `sample_rows` rows with generator seed 1000+draw (first rows if draws=1),
    compute within-draw normalized ranks, and run the encoder; average the
    draws. `log n` uses the full training-set size, not the cache.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n_full = len(y)
    if n_full > cache_rows:
        sub = np.random.default_rng(cache_seed).choice(n_full, size=cache_rows,
                                                       replace=False)
        X, y = X[sub], y[sub]
    X_sd = X.std(axis=0)
    X_sd[X_sd == 0] = 1.0
    y_sd = y.std() or 1.0
    Xy = np.column_stack([(X - X.mean(axis=0)) / X_sd,
                          (y - y.mean()) / y_sd]).astype(np.float32)
    Xy = np.clip(Xy, -10.0, 10.0)

    vs = []
    model.eval()
    with torch.no_grad():
        for b in range(draws):
            rng = None if draws == 1 else np.random.default_rng(1000 + b)
            M = Xy
            if len(M) > sample_rows:
                take = (rng.choice(len(M), size=sample_rows, replace=False)
                        if rng is not None else np.arange(sample_rows))
                M = M[take]
            R = ((rankdata(M, axis=0, method="average") - 1.0)
                 / max(M.shape[0] - 1, 1)).astype(np.float32)
            xt = torch.from_numpy(M[None, :, :-1])
            yt = torch.from_numpy(M[None, :, -1])
            xr = torch.from_numpy(R[None, :, :-1])
            yr = torch.from_numpy(R[None, :, -1])
            row_mask = torch.ones(1, M.shape[0])
            feat_mask = torch.ones(1, M.shape[1] - 1)
            log_n = torch.log10(torch.tensor([[float(n_full)]],
                                             dtype=torch.float32))
            v, _ = model._embed(xt, yt, xr, yr, row_mask, feat_mask, log_n)
            vs.append(v)
    return torch.stack(vs).mean(0).squeeze(0).numpy()
