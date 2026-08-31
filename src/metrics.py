import numpy as np
from scipy.stats import norm


# --------------------------------------------------------------------- #
# 7.1 Accuracy
# --------------------------------------------------------------------- #
def rmse(y_true, mean):
    return float(np.sqrt(np.mean((y_true - mean) ** 2)))


def mae(y_true, mean):
    return float(np.mean(np.abs(y_true - mean)))


# --------------------------------------------------------------------- #
# 7.2 Predictive distribution quality
# --------------------------------------------------------------------- #
def nll(y_true, mean, std):
    """Gaussian negative log-likelihood — internal use, minimise."""
    var = std ** 2
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + (y_true - mean) ** 2 / (2 * var)))


def ll(y_true, mean, std):
    """Average test log-likelihood in the literature's convention: higher is
    better, values negative. This is the column comparable with published
    tables (RMSE and LL are the only two)."""
    return -nll(y_true, mean, std)


def crps(y_true, mean, std):
    """Continuous ranked probability score, closed form for a Gaussian
    predictive distribution. Proper scoring rule, less tail-sensitive than
    NLL. Minimise."""
    z = (y_true - mean) / std
    return float(np.mean(std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def picp(y_true, mean, std, confidence=0.95):
    z = norm.ppf(0.5 + confidence / 2)
    lower = mean - z * std
    upper = mean + z * std
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def mpiw(std, confidence=0.95):
    z = norm.ppf(0.5 + confidence / 2)
    return float(np.mean(2 * z * std))


def interval_score(y_true, mean, std, confidence=0.95):
    """Winkler interval score: interval width plus a penalty for missing the
    interval, scaled by 2/alpha. Resolves the fact that MPIW alone is
    meaningless when coverage differs between methods. Minimise."""
    alpha = 1 - confidence
    z = norm.ppf(0.5 + confidence / 2)
    lower = mean - z * std
    upper = mean + z * std
    width = upper - lower
    below = np.maximum(lower - y_true, 0) * (2 / alpha)
    above = np.maximum(y_true - upper, 0) * (2 / alpha)
    return float(np.mean(width + below + above))


# --------------------------------------------------------------------- #
# 7.3 Calibration
# --------------------------------------------------------------------- #
def calibration_curve(y_true, mean, std, alphas=None):
    """Quantile calibration curve: for each alpha, the fraction of test
    points falling at or below the alpha-quantile of their predicted
    Gaussian. A perfectly calibrated model has empirical(alpha) == alpha."""
    if alphas is None:
        alphas = np.arange(0.05, 1.0, 0.05)
    alphas = np.asarray(alphas)
    empirical = np.array([
        np.mean(y_true <= mean + std * norm.ppf(a)) for a in alphas
    ])
    return alphas, empirical


def ece_reg(y_true, mean, std, alphas=None):
    """Expected calibration error for regression: mean absolute deviation of
    the calibration curve from the diagonal. Single number for the table."""
    alphas, empirical = calibration_curve(y_true, mean, std, alphas)
    return float(np.mean(np.abs(empirical - alphas)))


# --------------------------------------------------------------------- #
# L0 — exact aleatoric ground truth, synthetic data only (brief section 6)
# --------------------------------------------------------------------- #
def sigma_rmse(sigma_hat, sigma_true):
    """RMSE(sigma_hat(x), sigma_true(x)) — only computable where the true
    noise level is known by construction (synthetic data). Homoscedastic
    methods are expected to score poorly on `sin_hetero` (brief section 6):
    that is the point of the test, not a bug."""
    sigma_hat = np.asarray(sigma_hat)
    sigma_true = np.asarray(sigma_true)
    return float(np.sqrt(np.mean((sigma_hat - sigma_true) ** 2)))


# --------------------------------------------------------------------- #
# 7.4 Uncertainty decomposition — always computed, selectively reported
# --------------------------------------------------------------------- #
def mean_var_aleatoric(var_aleatoric):
    return float(np.mean(var_aleatoric))


def mean_var_epistemic(var_epistemic):
    return float(np.mean(var_epistemic))


def epi_ratio(var_epistemic, var_total):
    return float(np.mean(var_epistemic) / np.mean(var_total))


def ratio_by_mask(values, mask_a, mask_b):
    """mean(values[mask_a]) / mean(values[mask_b]).

    Shared building block for `epi_extrap_ratio`, `epi_gap_ratio` and
    `alea_gap_ratio` (section 7.4) — same computation, different masks
    (extrapolation vs. interpolation region, or gap vs. dense region).
    """
    return float(np.mean(values[mask_a]) / np.mean(values[mask_b]))


# --------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------- #
def summary(y_true, mean, std):
    return {
        "rmse": rmse(y_true, mean),
        "mae": mae(y_true, mean),
        "ll": ll(y_true, mean, std),
        "nll": nll(y_true, mean, std),
        "crps": crps(y_true, mean, std),
        "picp95": picp(y_true, mean, std),
        "mpiw95": mpiw(std),
        "interval_score95": interval_score(y_true, mean, std),
        "ece_reg": ece_reg(y_true, mean, std),
    }


# --------------------------------------------------------------------- #
# Epistemic growth away from the training data (D14d/D14g, corrected)
# --------------------------------------------------------------------- #
def value_at(values, x, x0):
    """`values` at the grid point nearest `x0`.

    Reports the grid point actually used alongside the value, because a
    1000-point grid over [-2, 8] never lands exactly on a requested probe:
    silently returning "the value at x=8" when the nearest node is 7.994
    is how a probe drifts without anyone noticing.
    """
    values = np.asarray(values)
    x = np.asarray(x)
    idx = int(np.argmin(np.abs(x - x0)))
    return float(values[idx]), float(x[idx])


def median_by_mask(values, mask):
    """Median of `values` over `mask`. Empty mask raises rather than
    returning NaN — every call site here passes a region that must be
    non-empty by construction, so an empty one is a bug in the mask, not a
    missing measurement."""
    values = np.asarray(values)[np.asarray(mask)]
    if values.size == 0:
        raise ValueError("mask selects no points")
    return float(np.median(values))


def epistemic_growth(std_epi, x, in_range_mask, probes=(-2.0, 8.0)):
    """How much a method's epistemic std grows from inside the training
    support out to each probe point.

    **The in-range reference is the MEDIAN over the whole training support,
    not the value at one hand-picked x.** The single-point version
    (`std_epi(8) / std_epi(3)`, used in earlier revisions of D14d/D14g)
    is not robust: on `sin_homo` at seed=0, MC dropout's profile is
    essentially flat at 0.13-0.15 across the entire eval grid but has one
    narrow local dip to 0.040 at x~3.2 — and x=3 landed in it. That single
    unlucky node turned a genuine ratio of ~1.2x into a reported 3.9x, and
    the "MCD's uncertainty grows nearly 4x in extrapolation" reading built
    on top of it. A median over ~600 in-range grid points cannot be moved
    by one narrow feature, which is the entire reason to use it here.

    The dip is a real property of the fitted network, not noise, so it is
    not being "cleaned away": it stays visible in
    `make_epistemic_profile_figure`. It just must not be what the
    comparison is anchored to.

    `in_range_mask` comes from `src.data.range_masks(...)["in_range"]`, so
    `sin_gap`'s reference excludes its held-out (2, 4) gap.
    """
    std_epi = np.asarray(std_epi)
    x = np.asarray(x)
    reference = median_by_mask(std_epi, in_range_mask)

    out = {
        "median_in_range": reference,
        "min_in_range": float(np.min(std_epi[np.asarray(in_range_mask)])),
        "max_in_range": float(np.max(std_epi[np.asarray(in_range_mask)])),
    }
    for x0 in probes:
        value, x_used = value_at(std_epi, x, x0)
        key = f"{x0:g}".replace("-", "m").replace(".", "p")
        out[f"std_epi_at_{key}"] = value
        out[f"x_used_{key}"] = x_used
        out[f"ratio_at_{key}"] = value / reference if reference > 0 else float("nan")
    return out
