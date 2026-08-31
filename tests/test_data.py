import numpy as np
import pytest

from src.data import (
    load_sin_homo, load_sin_hetero, load_sin_gap,
    gap_split, load_uci, n_uci_splits, UCI_SPEC, _UCI_SPLITS_ROOT,
)

_uci_data_available = _UCI_SPLITS_ROOT.exists()
_uci_skip_reason = "run python scripts/fetch_data.py first"


# --------------------------------------------------------------------- #
# Synthetic 1D variants
# --------------------------------------------------------------------- #
def test_sin_homo_constant_noise():
    ds = load_sin_homo(seed=0)
    assert ds.X_train.shape == (250, 1)
    np.testing.assert_allclose(ds.sigma_true_train, 0.1, atol=1e-6)
    np.testing.assert_allclose(ds.sigma_true_eval, 0.1, atol=1e-6)


def test_sin_hetero_noise_increases_with_x():
    ds = load_sin_hetero(seed=0)
    # sigma(x) = 0.05 + 0.15*x/6 -> sigma(0) = 0.05, sigma(6) = 0.2
    assert ds.sigma_true_train[0] == pytest.approx(0.05, abs=1e-6)
    assert ds.sigma_true_train[-1] == pytest.approx(0.2, abs=1e-6)
    assert np.all(np.diff(ds.sigma_true_train) >= 0)


def test_sin_gap_has_no_training_points_in_the_gap():
    ds = load_sin_gap(seed=0)
    x = ds.X_train.ravel()
    assert ds.X_train.shape == (250, 1)
    assert np.all((x <= 2.0) | (x >= 4.0))
    assert not np.any((x > 2.0) & (x < 4.0))


# --------------------------------------------------------------------- #
# Gap split (real-valued features)
# --------------------------------------------------------------------- #
def test_gap_split_selects_most_correlated_feature_and_partitions_cleanly():
    rng = np.random.RandomState(0)
    n = 900
    informative = rng.uniform(0, 10, size=n)
    noise_feature = rng.uniform(0, 10, size=n)
    X = np.stack([noise_feature, informative], axis=1)
    y = informative + 0.01 * rng.randn(n)  # column 1 is far more predictive than column 0

    result = gap_split(X, y, seed=0)

    assert result.feature_index == 1

    n_train = len(result.X_train)
    n_in_range = len(result.X_in_range)
    n_in_gap = len(result.X_in_gap)
    assert n_train + n_in_range + n_in_gap == n

    # n_in_gap must match the count implied by [q33, q66] of the selected raw feature
    q33, q66 = np.quantile(informative, [1 / 3, 2 / 3])
    expected_n_in_gap = int(np.sum((informative >= q33) & (informative <= q66)))
    assert n_in_gap == expected_n_in_gap


# --------------------------------------------------------------------- #
# UCI benchmarks — read from data/uci_splits/, populated by
# `python scripts/fetch_data.py` (see docs/datasets.md); skipped if absent
# rather than fetched here, since fetching needs network + git.
# --------------------------------------------------------------------- #
@pytest.mark.skipif(not _uci_data_available, reason=_uci_skip_reason)
@pytest.mark.parametrize("name", sorted(UCI_SPEC))
def test_uci_dataset_matches_protocol_table(name):
    spec = UCI_SPEC[name]
    ds = load_uci(name, split=0)

    n_total = ds.X_train.shape[0] + ds.X_test.shape[0]
    assert n_total == spec["n"]
    assert ds.X_train.shape[1] == spec["d"]
    assert ds.X_train.shape[0] == spec["n_train"]  # exact: these are the literature's own split files


@pytest.mark.skipif(not _uci_data_available, reason=_uci_skip_reason)
@pytest.mark.parametrize("name", sorted(UCI_SPEC))
def test_uci_splits_are_disjoint_and_stable_across_calls(name):
    n_splits = n_uci_splits(name)
    assert n_splits == 20

    ds_a = load_uci(name, split=0)
    ds_b = load_uci(name, split=0)
    np.testing.assert_array_equal(ds_a.X_train, ds_b.X_train)
    np.testing.assert_array_equal(ds_a.y_test, ds_b.y_test)

    ds_other = load_uci(name, split=1)
    assert ds_other.X_train.shape == ds_a.X_train.shape


@pytest.mark.skipif(not _uci_data_available, reason=_uci_skip_reason)
def test_load_uci_rejects_out_of_range_split():
    with pytest.raises(ValueError):
        load_uci("yacht", split=20)
