"""Registry: a family name plus its warm start in, an unfitted estimator out.

`recommend` names families ("Kriging", "RF", ...) and returns a `WarmStart`
whose `config` takes one of two forms. For the tree, network and linear
families it is a parameter dict. For the three UQ families it is a variant tag
naming which of that family's competing internal variants won on corpus
datasets of this shape, spelled as in warm_start.json ("_LMS", "_LARS_D5",
"_2M"). `build` resolves both:

    cand = report.candidates[0]
    model = build(cand.family, cand.warm_start.config).fit(X, y)

Tag to constructor arguments is the mapping the research registry used when it
labelled the corpus: Kriging_LMS is a linear trend under a Matern 5/2 kernel,
PCE_LARS_D5 the LARS solver with degree and truncation auto-selected up to 5,
PCK_2M a degree-2 PCE trend with Matern kriging. Each family module owns its
own variant table and `make`, so those settings stay next to the numerics they
pin; this module only dispatches.

LGBM and XGB need optional packages. `AVAILABLE` lists the families this
interpreter can build; building one of the others raises ImportError naming the
extra to install.
"""

from __future__ import annotations

from importlib.util import find_spec

from . import kriging, pce, pck
from .kriging import KrigingSurrogate
from .mlp import MLPSurrogate
from .pce import PCESurrogate
from .pck import PCKrigingSurrogate
from .simple import lasso, lgbm, linear_regression, random_forest, ridge, xgb

FAMILIES = ("Kriging", "PCE", "PCK", "MLP", "RF", "LGBM", "XGB", "LR", "Ridge",
            "Lasso")

# A warm-start tag is the registry key minus the family prefix, which is how
# warm_start.json's variant_meaning block spells it.
VARIANT_TAGS = {
    "Kriging": tuple(k.removeprefix("Kriging") for k in kriging.VARIANTS),
    "PCE": tuple(k.removeprefix("PCE") for k in pce.VARIANTS),
    "PCK": tuple(k.removeprefix("PCK") for k in pck.VARIANTS),
}

# Used when no warm start is supplied: each family's own constructor defaults.
DEFAULT_VARIANT = {"Kriging": "Kriging_OGS", "PCE": "PCE_Ridge_D5",
                   "PCK": "PCK_2G"}

_OPTIONAL_PACKAGE = {"LGBM": "lightgbm", "XGB": "xgboost"}


def _variant_key(family: str, config) -> str:
    """Full registry key from a tag ("_LMS") or from a key ("Kriging_LMS")."""
    if config is None:
        return DEFAULT_VARIANT[family]
    if not isinstance(config, str):
        raise TypeError(f"{family}: the warm start is a variant tag such as "
                        f"{VARIANT_TAGS[family][0]!r}, not a "
                        f"{type(config).__name__}")
    return config if config.startswith(family) else family + config


def _kriging(config=None) -> KrigingSurrogate:
    """Kriging at the variant the tag names."""
    return kriging.make(_variant_key("Kriging", config))


def _pce(config=None) -> PCESurrogate:
    """Polynomial chaos expansion at the solver the tag names."""
    return pce.make(_variant_key("PCE", config))


def _pck(config=None) -> PCKrigingSurrogate:
    """PC-Kriging at the trend degree and kernel the tag names."""
    return pck.make(_variant_key("PCK", config))


CONSTRUCTORS = {
    "Kriging": _kriging,
    "PCE": _pce,
    "PCK": _pck,
    "MLP": MLPSurrogate,
    "RF": random_forest,
    "LGBM": lgbm,
    "XGB": xgb,
    "LR": linear_regression,
    "Ridge": ridge,
    "Lasso": lasso,
}

AVAILABLE = tuple(f for f in FAMILIES
                  if f not in _OPTIONAL_PACKAGE
                  or find_spec(_OPTIONAL_PACKAGE[f]) is not None)


def build(family: str, config=None):
    """Unfitted estimator for one family at the warm start `config`.

    `config` is a parameter dict, a variant tag for Kriging, PCE and PCK, or
    None for the family's own defaults. Dict keys the family does not
    understand are dropped with a warning, so a configuration from a newer
    artifact table cannot crash a fit; an unrecognized variant tag raises,
    because there is no sane way to guess which variant was meant.
    """
    if family not in CONSTRUCTORS:
        raise ValueError(f"unknown family {family!r}; choose from "
                         f"{list(FAMILIES)}")
    return CONSTRUCTORS[family](config)


__all__ = ["AVAILABLE", "CONSTRUCTORS", "DEFAULT_VARIANT", "FAMILIES",
           "VARIANT_TAGS", "build",
           "KrigingSurrogate", "MLPSurrogate", "PCESurrogate",
           "PCKrigingSurrogate", "lasso", "lgbm", "linear_regression",
           "random_forest", "ridge", "xgb"]
