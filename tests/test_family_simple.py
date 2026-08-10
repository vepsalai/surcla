"""Family constructors: real warm starts must fit, and FLAML keys must be
translated rather than forwarded (a wrong translation degrades fits silently)."""

import json

import numpy as np
import pytest

from surcla.data import artifact_path
from surcla.families import CONSTRUCTORS, simple

BAND = "n<=100|d<=5"
N, D = 80, 4


@pytest.fixture(scope="module")
def warm_start():
    table = json.loads(artifact_path("warm_start.json").read_text())
    return {f: table["families"][f][BAND]["config"] for f in CONSTRUCTORS}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, D))
    y = (np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2 - X[:, 2]
         + rng.normal(scale=0.1, size=N))
    return X, y


def _fit_predict(estimator, data):
    X, y = data
    pred = np.asarray(estimator.fit(X, y).predict(X))
    assert pred.shape == (N,)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("family", ["LR", "Ridge", "Lasso", "RF", "MLP"])
def test_warm_start_fits(family, warm_start, data):
    _fit_predict(CONSTRUCTORS[family](warm_start[family]), data)


def test_lgbm_warm_start_fits(warm_start, data):
    pytest.importorskip("lightgbm")
    _fit_predict(simple.lgbm(warm_start["LGBM"]), data)


def test_xgb_warm_start_fits(warm_start, data):
    pytest.importorskip("xgboost")
    _fit_predict(simple.xgb(warm_start["XGB"]), data)


def test_labelling_run_defaults():
    """No config means the fixed settings the corpus was labelled with."""
    assert simple.ridge().alpha == 1.0
    assert simple.lasso().alpha == 0.01
    assert simple.lasso().max_iter == 10_000
    assert simple.ridge({"alpha": 5.0}).alpha == 5.0


def test_rf_max_leaves_becomes_max_leaf_nodes(warm_start):
    rf = simple.random_forest(warm_start["RF"])
    assert rf.max_leaf_nodes == warm_start["RF"]["max_leaves"]
    assert "max_leaves" not in rf.get_params()


def test_lgbm_log_max_bin_is_a_log2_exponent(warm_start):
    pytest.importorskip("lightgbm")
    config = warm_start["LGBM"]
    params = simple.lgbm(config).get_params()
    assert params["max_bin"] == 2 ** config["log_max_bin"] - 1 == 255
    assert "log_max_bin" not in params


def test_xgb_grows_leaf_wise(warm_start):
    pytest.importorskip("xgboost")
    config = warm_start["XGB"]
    params = simple.xgb(config).get_params()
    assert params["grow_policy"] == "lossguide"   # else max_leaves is ignored
    assert params["max_depth"] == 0
    assert params["max_leaves"] == config["max_leaves"]


def test_mlp_config_maps_onto_the_network(warm_start, data):
    config = warm_start["MLP"]
    model = CONSTRUCTORS["MLP"](config).fit(*data)
    net = model.model_[-1]
    assert net.hidden_layer_sizes == tuple(config["layer_dims"])
    assert net.learning_rate_init == config["lr"]
    assert net.max_iter == config["epochs"]
    assert net.batch_size == config["batch_size"]
    assert not net.early_stopping        # n is far below hpo_sample_size


@pytest.mark.parametrize("build, config, kept", [
    (simple.ridge, {"alpha": 2.0, "max_leaves": 8}, ("alpha", 2.0)),
    (simple.random_forest, {"n_estimators": 5, "learning_rate": 0.1},
     ("n_estimators", 5)),
])
def test_unknown_keys_are_dropped_not_forwarded(build, config, kept, data):
    with pytest.warns(UserWarning, match="ignoring unsupported config keys"):
        estimator = build(config)
    key, value = kept
    assert estimator.get_params()[key] == value
    assert config.keys() - estimator.get_params().keys()   # something was dropped
    _fit_predict(estimator, data)


def test_non_dict_config_is_rejected():
    with pytest.raises(TypeError):
        simple.ridge("_OGS")             # a UQ variant tag, not a parameter dict
    with pytest.raises(TypeError):
        CONSTRUCTORS["MLP"]("_2G").fit(np.zeros((10, 2)), np.zeros(10))
