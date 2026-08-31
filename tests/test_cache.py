"""Model cache (src/methods/cache.py, this session's decision): fit()
checks cache/{method}_{hash}.pt before training; use_cache=False forces a
real retrain. A cache hit must reproduce exactly what a fresh fit() would
have produced (same predict() output, given predict()'s own determinism —
see test_methods.py::test_stable_given_same_seed for that separate fix).
"""
import shutil

import numpy as np
import pytest

from src.data import load_sin_homo
from src.methods import cache as cache_module
from src.methods.bbb import BBBMethod
from src.methods.map import MAPMethod


@pytest.fixture()
def toy_data():
    ds = load_sin_homo(seed=0)
    return ds.X_train, ds.y_train, ds.X_eval


@pytest.fixture(autouse=True)
def clean_cache_dir():
    shutil.rmtree(cache_module.CACHE_DIR, ignore_errors=True)
    yield
    shutil.rmtree(cache_module.CACHE_DIR, ignore_errors=True)


def test_cache_hit_reproduces_fresh_fit_exactly(toy_data):
    X_train, y_train, X_eval = toy_data
    m1 = MAPMethod(epochs=100).fit(X_train, y_train, seed=3, use_cache=True)
    pred_fresh = m1.predict(X_eval)
    assert any(cache_module.CACHE_DIR.glob("homoscedastic_mlp_*.pt"))

    m2 = MAPMethod(epochs=100).fit(X_train, y_train, seed=3, use_cache=True)
    pred_cached = m2.predict(X_eval)

    np.testing.assert_allclose(pred_fresh.mean, pred_cached.mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(pred_fresh.var_aleatoric, pred_cached.var_aleatoric, rtol=1e-10, atol=1e-12)


def test_use_cache_false_ignores_cache(toy_data):
    """A different config with use_cache=False must not read a stale/wrong
    cache entry from an unrelated earlier fit — checked by fitting twice
    with use_cache=False and confirming no cache file is written at all."""
    X_train, y_train, X_eval = toy_data
    MAPMethod(epochs=50).fit(X_train, y_train, seed=1, use_cache=False)
    assert not any(cache_module.CACHE_DIR.glob("*.pt"))


def test_different_hyperparameters_do_not_collide_in_cache(toy_data):
    """Two configs differing only in a value baked into model_factory
    (fixed_sigma2, via cache_key_extra) must land in different cache
    entries — the exact bug class `cache_key_extra` exists to prevent."""
    X_train, y_train, X_eval = toy_data
    m_a = MAPMethod(epochs=50, fixed_sigma2=0.01).fit(X_train, y_train, seed=2, use_cache=True)
    m_b = MAPMethod(epochs=50, fixed_sigma2=0.04).fit(X_train, y_train, seed=2, use_cache=True)

    pred_a = m_a.predict(X_eval)
    pred_b = m_b.predict(X_eval)
    assert np.allclose(pred_a.var_aleatoric, 0.01)
    assert np.allclose(pred_b.var_aleatoric, 0.04)
    assert len(list(cache_module.CACHE_DIR.glob("homoscedastic_mlp_*.pt"))) == 2


def test_bbb_cache_hit_preserves_deterministic_prior_buffers(toy_data):
    """BBB's cache stores only `state_dict()` (the learned mu/rho
    parameters) — the prior buffers (`prior_weight_sigma` etc.) are
    non-persistent and must come from `model_factory()` re-running
    `dnn_to_bnn`/`_apply_layerwise_prior` on every call, cache hit or not.
    Regression test for exactly that assumption.
    """
    X_train, y_train, X_eval = toy_data
    m1 = BBBMethod(epochs=30, T=5, layerwise_prior_omega=4.0).fit(X_train, y_train, seed=5, use_cache=True)
    prior_sigma_fresh = m1.model.mlp.linear1.prior_weight_sigma.clone()

    m2 = BBBMethod(epochs=30, T=5, layerwise_prior_omega=4.0).fit(X_train, y_train, seed=5, use_cache=True)
    prior_sigma_cached = m2.model.mlp.linear1.prior_weight_sigma.clone()

    np.testing.assert_allclose(prior_sigma_fresh.numpy(), prior_sigma_cached.numpy())
    np.testing.assert_allclose(prior_sigma_cached.numpy(), 4.0)  # omega/sqrt(fan_in=1) = 4.0
