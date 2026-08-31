import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.metrics import (
    rmse, mae, nll, ll, crps, picp, mpiw, interval_score,
    calibration_curve, ece_reg, epi_ratio, epistemic_growth, median_by_mask,
    ratio_by_mask, sigma_rmse, value_at,
)


def test_rmse_mae_known_values():
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    mean = np.array([0.0, 1.0, 2.0, 5.0])  # single error of 2 at the last point
    assert rmse(y_true, mean) == pytest.approx(np.sqrt(4 / 4))
    assert mae(y_true, mean) == pytest.approx(2 / 4)


def test_ll_is_negative_nll():
    rng = np.random.RandomState(0)
    y_true = rng.randn(200)
    mean = rng.randn(200) * 0.1
    std = np.full(200, 0.9)
    assert ll(y_true, mean, std) == pytest.approx(-nll(y_true, mean, std))


def test_ll_prefers_well_fitted_model_over_baseline():
    rng = np.random.RandomState(1)
    y_true = rng.randn(5000) * 2.0 + 1.0

    # "well fitted": correct mean and std
    ll_good = ll(y_true, mean=np.full_like(y_true, 1.0), std=np.full_like(y_true, 2.0))
    # baseline: unit Gaussian, ignoring the true mean/scale
    ll_baseline = ll(y_true, mean=np.zeros_like(y_true), std=np.ones_like(y_true))

    assert ll_good > ll_baseline


def test_crps_standard_normal_closed_form_at_zero():
    # CRPS(N(0,1), y=0) = 2*phi(0) - 1/sqrt(pi), a known reference value.
    expected = 2 * (1 / np.sqrt(2 * np.pi)) - 1 / np.sqrt(np.pi)
    value = crps(y_true=np.array([0.0]), mean=np.array([0.0]), std=np.array([1.0]))
    assert value == pytest.approx(expected, rel=1e-9)


def test_crps_shrinks_to_absolute_error_as_std_to_zero():
    y_true = np.array([2.0])
    mean = np.array([0.0])
    value = crps(y_true, mean, std=np.array([1e-6]))
    assert value == pytest.approx(2.0, abs=1e-3)


def test_picp_mpiw_known_coverage():
    # mean=0, std=1: y=+-1 sits inside a 95% interval (z~1.96), y=+-3 sits outside.
    y_true = np.array([1.0, -1.0, 3.0, -3.0])
    mean = np.zeros(4)
    std = np.ones(4)
    assert picp(y_true, mean, std, confidence=0.95) == pytest.approx(0.5)
    assert mpiw(std, confidence=0.95) == pytest.approx(2 * 1.959964, rel=1e-4)


def test_interval_score_penalises_misses():
    mean = np.zeros(1)
    std = np.ones(1)
    covered = interval_score(np.array([0.0]), mean, std, confidence=0.95)
    missed = interval_score(np.array([10.0]), mean, std, confidence=0.95)
    assert missed > covered


def test_calibration_curve_well_calibrated_model():
    rng = np.random.RandomState(2)
    n = 200_000
    mean = rng.randn(n)
    std = np.full(n, 1.5)
    y_true = mean + std * rng.randn(n)  # y_true is exactly N(mean, std^2)

    alphas, empirical = calibration_curve(y_true, mean, std)
    np.testing.assert_allclose(empirical, alphas, atol=0.01)
    assert ece_reg(y_true, mean, std) < 0.01


# --------------------------------------------------------------------- #
# Standardisation round-trip contract (section 5.3): sigma must be scaled
# by y_scaler.scale_, not just the mean. Forgetting this gives a correct
# RMSE and a wrong NLL — the brief calls this out as the most common silent
# bug in the protocol.
# --------------------------------------------------------------------- #
def test_destandardising_sigma_preserves_nll():
    rng = np.random.RandomState(3)
    n = 300

    y_original = rng.uniform(-5, 25, size=n)
    mean_original = rng.uniform(-5, 25, size=n)
    sigma_original = rng.uniform(0.5, 3.0, size=n)  # known, heteroscedastic

    y_scaler = StandardScaler().fit(y_original.reshape(-1, 1))
    scale = y_scaler.scale_[0]

    # Same predictive distribution, expressed in standardised space.
    y_std = y_scaler.transform(y_original.reshape(-1, 1)).ravel()
    mean_std = y_scaler.transform(mean_original.reshape(-1, 1)).ravel()
    sigma_std = sigma_original / scale

    # Invert per the documented convention (src/data.py module docstring):
    # sigma_original = sigma_standardised * y_scaler.scale_
    mean_recovered = y_scaler.inverse_transform(mean_std.reshape(-1, 1)).ravel()
    sigma_recovered = sigma_std * scale

    np.testing.assert_allclose(mean_recovered, mean_original, rtol=1e-5)
    np.testing.assert_allclose(sigma_recovered, sigma_original, rtol=1e-5)

    nll_direct = nll(y_original, mean_original, sigma_original)
    nll_via_inversion = nll(y_original, mean_recovered, sigma_recovered)
    assert nll_via_inversion == pytest.approx(nll_direct, rel=1e-6)

    # Negative control: forgetting to scale sigma (the bug this guards
    # against) must actually change the NLL, otherwise the test above would
    # pass vacuously regardless of correct scaling.
    nll_without_sigma_scaling = nll(y_original, mean_recovered, sigma_std)
    assert nll_without_sigma_scaling != pytest.approx(nll_direct, rel=1e-6)


def test_sigma_rmse_known_value():
    sigma_true = np.array([0.1, 0.1, 0.1, 0.1])
    sigma_hat = np.array([0.1, 0.1, 0.1, 0.5])  # single error of 0.4 at the last point
    assert sigma_rmse(sigma_hat, sigma_true) == pytest.approx(np.sqrt(0.16 / 4))
    assert sigma_rmse(sigma_true, sigma_true) == pytest.approx(0.0)


def test_epi_ratio_and_ratio_by_mask():
    var_epi = np.array([1.0, 1.0, 1.0, 1.0])
    var_alea = np.array([1.0, 1.0, 1.0, 1.0])
    var_total = var_epi + var_alea
    assert epi_ratio(var_epi, var_total) == pytest.approx(0.5)

    values = np.array([10.0, 10.0, 1.0, 1.0])
    mask_a = np.array([True, True, False, False])
    mask_b = np.array([False, False, True, True])
    assert ratio_by_mask(values, mask_a, mask_b) == pytest.approx(10.0)


def test_epistemic_growth_median_reference_survives_a_narrow_dip():
    """Regression test for the D14d/D14g reference-point bug (D14i).

    Reproduces the failure shape exactly: a profile that is flat at 0.14
    across the whole grid apart from one narrow dip to 0.04, with the old
    convention's probe point (x=3) sitting inside the dip. Anchoring on
    that point reports ~3.9x growth where the true growth is ~1.1x; the
    median over the training support must not be movable that way.
    """
    x = np.linspace(-2.0, 8.0, 1001)
    std_epi = np.full_like(x, 0.14)
    std_epi[np.abs(x - 3.0) < 0.15] = 0.04          # the narrow dip
    std_epi[x > 6.0] = 0.156                         # mild real growth outside the data
    in_range = (x >= 0) & (x <= 6)

    dip_value, _ = value_at(std_epi, x, 3.0)
    edge_value, x_used = value_at(std_epi, x, 8.0)
    assert dip_value == pytest.approx(0.04)
    assert x_used == pytest.approx(8.0, abs=0.01)
    # what the old single-point convention would have reported
    assert edge_value / dip_value == pytest.approx(3.9, abs=0.1)

    growth = epistemic_growth(std_epi, x, in_range, probes=(8.0,))
    assert growth["median_in_range"] == pytest.approx(0.14)   # dip is ~3% of the region
    assert growth["ratio_at_8"] == pytest.approx(0.156 / 0.14, rel=1e-6)
    assert growth["ratio_at_8"] < 1.2                          # the honest number


def test_median_by_mask_rejects_empty_mask():
    values = np.array([1.0, 2.0, 3.0])
    assert median_by_mask(values, np.array([True, True, False])) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        median_by_mask(values, np.zeros(3, dtype=bool))


def test_evaluate_on_synthetic_gap_ratio_is_nan_for_a_zero_epistemic_method():
    """`map`'s var_epistemic is an exact zero array by construction, so its
    gap_ratio is undefined rather than infinite. Regression test: the first
    E5 run died with ZeroDivisionError on exactly this row.
    """
    from types import SimpleNamespace
    from src.methods.base import Prediction
    from src.sweeps import evaluate_on_synthetic

    x = np.linspace(-2.0, 8.0, 501)
    ds = SimpleNamespace(
        X_eval=x.reshape(-1, 1),
        y_eval_noisy=np.sin(x) + 0.1 * np.random.RandomState(0).randn(x.size),
    )
    pred = Prediction(
        mean=np.sin(x),
        var_aleatoric=np.full_like(x, 0.01),
        var_epistemic=np.zeros_like(x),
    )
    out = evaluate_on_synthetic(pred, ds, "sin_gap")
    assert np.isnan(out["gap_ratio"])
    assert np.isnan(out["ratio_at_8"])
    assert np.isfinite(out["rmse_in_range"])
