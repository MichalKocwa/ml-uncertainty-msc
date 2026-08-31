"""E1 — Synthetic 1D (brief section 9): six methods x three variants
(sin_homo, sin_hetero, sin_gap). Deterministic, single seed, single split —
unlike E2's 20 UCI splits, section 5.1's synthetic sets exist precisely
because they have known ground truth, not to be resampled.

Outputs:
  results/e1_synthetic.csv           -- section 8's fixed schema, one row per (dataset, method, split_type):
                                         "in_range" (where the training data is; sin_gap's two intervals),
                                         "extrapolation" (outside [0,6]), and for sin_gap only "in_gap"
                                         (the held-out (2,4) interval) -- not one row mixing all of them,
                                         which buries in-range fit quality under extrapolation error
                                         (this session's decision -- see docs/chapter4_notes.md)
  results/e1_sigma_calibration.csv   -- L0 (section 6): RMSE(sigma_hat, sigma_true) on the full eval grid
  results/predictions_1d/{dataset}_{method}.csv -- per-point (x, mean, std_alea, std_epi, y_true),
                                                     so figures redraw without retraining (section 8)
  results/predictions_1d/{dataset}_train.csv    -- (x, y) training points, same reason

Usage:
  python experiments/e1_synthetic.py [--quick] [--methods map,gp,...] [--datasets sin_homo,...] [--seed 0]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import SYNTHETIC_DATASETS, range_masks
from src.metrics import epi_ratio, mean_var_aleatoric, mean_var_epistemic, sigma_rmse, summary
from src.methods import METHODS
from src.results import (
    RESULTS_DIR, append_generic_csv, append_predictions_1d, append_result_row,
    git_commit_short, now_iso, save_train_points,
)
from src.seeding import set_seed
from src.style import SEED as FIGURE_SEED
from src.sweeps import BBB_ELBO_SAMPLES_E1 as _BBB_ELBO_SAMPLES_E1
from src.sweeps import KNOWN_HOMOSCEDASTIC_SIGMA as _KNOWN_HOMOSCEDASTIC_SIGMA

EXPERIMENT_ID = "e1_synthetic"
QUICK_EPOCHS = 100
QUICK_MARGLIK_STEPS = 20

# D-sigma-E1 / D14d-D14e: both constants now live in `src/sweeps.py` as the single
# source, so that the depth/epoch/dropout sweeps cannot drift from the protocol the
# E1 results in `results/e1_synthetic.csv` were produced under. Re-exported here
# under their historical names because the surrounding comments in this file, the
# notes and the tests all refer to them by these names.
KNOWN_HOMOSCEDASTIC_SIGMA = _KNOWN_HOMOSCEDASTIC_SIGMA
BBB_ELBO_SAMPLES_E1 = _BBB_ELBO_SAMPLES_E1


def _method_kwargs(method_name: str, dataset_name: str, quick: bool) -> dict:
    kwargs = {}
    if quick:
        if method_name == "laplace":
            kwargs.update(epochs=QUICK_EPOCHS, marglik_steps=QUICK_MARGLIK_STEPS)
        elif method_name in ("map", "mcd", "ensemble", "bbb"):
            kwargs.update(epochs=QUICK_EPOCHS)
        # gp has no epoch count
    if method_name != "gp" and dataset_name in KNOWN_HOMOSCEDASTIC_SIGMA:
        kwargs["fixed_sigma2"] = KNOWN_HOMOSCEDASTIC_SIGMA[dataset_name]
    if method_name == "bbb":
        kwargs["elbo_samples"] = BBB_ELBO_SAMPLES_E1
    return kwargs


def _range_masks(dataset_name: str, x) -> dict:
    """One row per `split_type` instead of one row mixing fit quality with
    extrapolation (this session's decision — a single RMSE over the whole
    [-2, 10] eval grid was dominated by extrapolation error even for
    methods that fit the training range well, e.g. MAP's RMSE=0.76 despite
    tracking the data closely inside [0,6]).

    The mask definitions themselves now live in `src.data.range_masks`, so
    that the diagnostics computed on top of `results/predictions_1d/`
    (`scripts/epistemic_growth.py` and the sweep scripts) use exactly the
    same notion of `in_range` as the rows in `results/e1_synthetic.csv` —
    previously this function was the only definition and anything else
    needing it would have had to restate it.
    """
    return range_masks(dataset_name, x)


def run_one(dataset_name: str, method_name: str, seed: int, quick: bool, use_cache: bool = False) -> None:
    """`use_cache` (src/methods/cache.py) defaults OFF here, deliberately —
    a cache HIT makes the recorded `train_time_s` for that row near-zero
    (state-dict load, not a real train), and this function records
    `train_time_s` as a result metric (D23). Pass `use_cache=True` (CLI:
    `--use-cache`) only for prediction/figure exploration where the timing
    column doesn't matter — never for a run whose numbers are meant to be
    trusted (docs/chapter4_notes.md D-width-E5 documents this actually
    happening once, for Laplace, before this default was flipped).
    """
    set_seed(seed)
    ds = SYNTHETIC_DATASETS[dataset_name](seed=seed)

    method = METHODS[method_name](**_method_kwargs(method_name, dataset_name, quick))

    t0 = time.perf_counter()
    method.fit(ds.X_train, ds.y_train, seed=seed, use_cache=use_cache)
    train_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = method.predict(ds.X_eval)
    predict_time_s = time.perf_counter() - t0
    n_test_total = len(ds.X_eval)
    predict_time_ms_per_1k = predict_time_s * 1000.0 / n_test_total * 1000.0

    x_eval = ds.X_eval.ravel()
    masks = _range_masks(dataset_name, x_eval)
    for split_type, mask in masks.items():
        y_sub = ds.y_eval_noisy[mask]  # metrics against noisy observations, not the clean function (this session's fix)
        mean_sub = pred.mean[mask]
        std_sub = pred.std_total[mask]
        var_alea_sub = pred.var_aleatoric[mask]
        var_epi_sub = pred.var_epistemic[mask]
        var_total_sub = pred.var_total[mask]

        row = dict(
            experiment_id=EXPERIMENT_ID,
            dataset=dataset_name,
            method=method_name,
            config_id="quick" if quick else "default",
            split_index=0,
            init_seed=seed,
            n_train=len(ds.X_train),
            n_test=int(mask.sum()),
            split_type=split_type,
            **summary(y_sub, mean_sub, std_sub),
            mean_var_aleatoric=mean_var_aleatoric(var_alea_sub),
            mean_var_epistemic=mean_var_epistemic(var_epi_sub),
            epi_ratio=epi_ratio(var_epi_sub, var_total_sub),
            train_time_s=train_time_s,
            predict_time_ms_per_1k=predict_time_ms_per_1k,
            n_parameters=method.n_parameters,
            timestamp=now_iso(),
            git_commit=git_commit_short(),
        )
        append_result_row(RESULTS_DIR / f"{EXPERIMENT_ID}.csv", row)

        print(f"  {dataset_name:12s} {method_name:10s} {split_type:14s} "
              f"n={row['n_test']:4d} rmse={row['rmse']:.4f} ll={row['ll']:.4f} "
              f"picp95={row['picp95']:.4f} mpiw95={row['mpiw95']:.4f} "
              f"epi_ratio={row['epi_ratio']:.4f}")

    append_generic_csv(RESULTS_DIR / "e1_sigma_calibration.csv", dict(
        experiment_id=EXPERIMENT_ID,
        dataset=dataset_name,
        method=method_name,
        sigma_rmse=sigma_rmse(pred.std_total, ds.sigma_true_eval),
        timestamp=now_iso(),
        git_commit=git_commit_short(),
    ))

    append_predictions_1d(
        RESULTS_DIR / "predictions_1d" / f"{dataset_name}_{method_name}.csv",
        x=ds.X_eval.ravel(), mean=pred.mean,
        std_alea=pred.var_aleatoric ** 0.5, std_epi=pred.var_epistemic ** 0.5,
        y_true=ds.y_eval,
    )
    # shared across all six methods' panels for this dataset; harmless to rewrite identically each time
    save_train_points(
        RESULTS_DIR / "predictions_1d" / f"{dataset_name}_train.csv",
        x=ds.X_train.ravel(), y=ds.y_train,
    )
    print(f"  {dataset_name:12s} {method_name:10s} sigma_rmse={sigma_rmse(pred.std_total, ds.sigma_true_eval):.4f} "
          f"train_time_s={train_time_s:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="reduced epochs, for smoke-testing the pipeline")
    parser.add_argument("--methods", type=str, default=",".join(METHODS.keys()))
    parser.add_argument("--datasets", type=str, default=",".join(SYNTHETIC_DATASETS.keys()))
    parser.add_argument(
        "--seed", type=int, default=FIGURE_SEED,
        help=f"default ({FIGURE_SEED}) matches src/style.py's SEED, which every E1 figure assumes",
    )
    parser.add_argument(
        "--use-cache", action="store_true",
        help="read/write cache/ (src/methods/cache.py) for a faster rerun -- OFF by default: this script "
             "records train_time_s as a result metric (D23), and a cache hit makes it near-zero instead of "
             "a genuine measurement (see docs/chapter4_notes.md D-width-E5, where this bit once). Opt-in only "
             "for prediction/figure exploration where the timing column doesn't matter.",
    )
    args = parser.parse_args()

    methods = args.methods.split(",")
    datasets = args.datasets.split(",")
    for name in methods:
        if name not in METHODS:
            raise ValueError(f"unknown method '{name}', expected one of {sorted(METHODS)}")
    for name in datasets:
        if name not in SYNTHETIC_DATASETS:
            raise ValueError(f"unknown dataset '{name}', expected one of {sorted(SYNTHETIC_DATASETS)}")

    for dataset_name in datasets:
        print(f"{dataset_name}:")
        for method_name in methods:
            run_one(dataset_name, method_name, seed=args.seed, quick=args.quick, use_cache=args.use_cache)


if __name__ == "__main__":
    main()
