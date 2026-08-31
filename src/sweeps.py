"""Shared evaluation for the 1D hyperparameter sweeps.

`experiments/ensemble_epochs_sweep.py` (O8) and
`experiments/mcd_dropout_sweep.py` both ask the same question of a fitted
method — "what are its fit metrics per region, and how much does its
epistemic std grow away from the training data" — so the answer is
computed in one place. Without this, the two scripts would each restate
the mask handling and the growth metric, and the second one to be edited
would quietly start measuring something slightly different from the first.

Deliberately NOT merged into `experiments/e1_synthetic.py`: that script
owns `results/e1_synthetic.csv`, whose schema is fixed by brief section 8
and must not grow sweep-specific columns. Sweeps write their own files.
"""
import numpy as np

from src.data import range_masks
from src.metrics import epistemic_growth, median_by_mask, summary

PROBES = (-2.0, 8.0)

# E1's protocol constants, defined here as the SINGLE source and imported by
# `experiments/e1_synthetic.py` as well as by every sweep script. Previously
# each sweep restated them; three copies of "sigma_o is fixed to 0.01 on the
# homoscedastic sets" is three chances for a sweep to silently stop being
# comparable with the E1 results it is supposed to extend.
#
# D-sigma-E1: fix sigma_o to its true, known-by-construction VARIANCE for the
# NN methods on synthetic data only (E2's real UCI data always learns it).
# The value is a variance, not a std — `HomoscedasticMLP(fixed_sigma2=...)`
# stores it as `log(fixed_sigma2)`. Guarded by
# tests/test_methods.py::test_e1_known_homoscedastic_sigma_is_variance_not_std.
# `sin_hetero` is absent deliberately: its true sigma varies with x, so there
# is no single value to fix a homoscedastic backbone to.
KNOWN_HOMOSCEDASTIC_SIGMA = {"sin_homo": 0.1 ** 2, "sin_gap": 0.1 ** 2}

# D14d/D14e: elbo_samples=32 for BBB, E1 scale only (~90s/fit at N=250; at
# E2's N=8611 the same setting costs ~32h for one dataset's 20 splits).
BBB_ELBO_SAMPLES_E1 = 32


def e1_method_kwargs(method_name: str, dataset_name: str) -> dict:
    """The E1 protocol's per-method keyword arguments, minus `--quick`.

    Mirrors `experiments/e1_synthetic.py::_method_kwargs` (which adds the
    quick-mode epoch overrides on top of this). Any sweep that wants its
    numbers comparable with `results/e1_synthetic.csv` must build its
    methods through here rather than by hand.
    """
    kwargs = {}
    if method_name != "gp" and dataset_name in KNOWN_HOMOSCEDASTIC_SIGMA:
        kwargs["fixed_sigma2"] = KNOWN_HOMOSCEDASTIC_SIGMA[dataset_name]
    if method_name == "bbb":
        kwargs["elbo_samples"] = BBB_ELBO_SAMPLES_E1
    return kwargs


def evaluate_on_synthetic(pred, ds, dataset_name: str, probes=PROBES) -> dict:
    """Flat `{column: value}` for one fitted method on one synthetic dataset.

    Per-region fit metrics are suffixed with the region name
    (`rmse_in_range`, `ll_extrapolation`, ...) so that one sweep row holds
    every region, rather than the long-format one-row-per-region shape
    `results/e1_synthetic.csv` uses. A sweep is read as "what did this
    hyperparameter do", which means comparing regions across a row.

    Metrics are computed against `ds.y_eval_noisy`, never `ds.y_eval` — the
    clean function has no noise, and scoring predictive intervals against
    it is what once produced `PICP@95 = 1.00` for all six methods at once
    (see docs/chapter4_notes.md, part E).
    """
    x = ds.X_eval.ravel()
    masks = range_masks(dataset_name, x)
    std_epi = np.sqrt(pred.var_epistemic)

    out = {}
    for region, mask in masks.items():
        stats = summary(ds.y_eval_noisy[mask], pred.mean[mask], pred.std_total[mask])
        for key in ("rmse", "ll", "picp95", "mpiw95"):
            out[f"{key}_{region}"] = stats[key]
        out[f"median_std_epi_{region}"] = median_by_mask(std_epi, mask)

    growth = epistemic_growth(std_epi, x, masks["in_range"], probes=probes)
    out["median_std_epi_in_range"] = growth["median_in_range"]
    for x0 in probes:
        key = f"{x0:g}".replace("-", "m").replace(".", "p")
        out[f"std_epi_at_{key}"] = growth[f"std_epi_at_{key}"]
        out[f"ratio_at_{key}"] = growth[f"ratio_at_{key}"]
    if "in_gap" in masks:
        # `map` reports var_epistemic as an exact zero array by construction
        # (src/methods/base.py), so its in-range reference is 0 and the ratio
        # is undefined, not infinite — the same guard `epistemic_growth`
        # already applies to `ratio_at_*`. Without it a `map` row on sin_gap
        # raises ZeroDivisionError mid-sweep (it did, in E5's first run).
        reference = growth["median_in_range"]
        out["gap_ratio"] = out["median_std_epi_in_gap"] / reference if reference > 0 else float("nan")
    return out


def aggregate(df, group_cols, value_cols):
    """mean and std across seeds, for the summary table.

    `std` uses `ddof=1`: with 3 seeds the population formula would
    understate the between-seed spread, and that spread is the thing these
    sweeps are read against — D14h's `M` ablation was decided by a
    difference being smaller than it.
    """
    agg = df.groupby(list(group_cols))[list(value_cols)].agg(["mean", lambda s: s.std(ddof=1)])
    agg.columns = [f"{c}_{'mean' if k == 'mean' else 'std'}" for c, k in agg.columns]
    return agg.reset_index()
