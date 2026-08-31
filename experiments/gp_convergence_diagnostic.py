"""Do sklearn's GP convergence warnings change the E2 numbers, or only the log?

The E2 run emitted 71 `ConvergenceWarning`s from `GaussianProcessRegressor`
on `energy` and `wine_quality_red`: `lbfgs failed to converge (ABNORMAL)`
after 17-38 iterations, `constant_value` close to its upper bound 1e5, and
`noise_level` close to its lower bound 1e-5. A warning is not by itself a
wrong answer - marginal-likelihood optimisation is restarted five times and
the best restart is kept, so a failed restart can be discarded harmlessly -
so the question is whether the splits that warn report different metrics
from the splits that do not.

What is recorded per split:
  * every warning raised during `fit`, by category and by which parameter
    it names;
  * the fitted hyperparameters (`constant_value`, `length_scale`,
    `noise_level`), each tested against sklearn's default bounds (1e-5,
    1e5) on the log scale, since a value pinned to a bound is the
    optimiser reporting that it wanted to leave the admissible region;
  * the log-marginal-likelihood of the retained fit, which is the only
    direct measure of whether a warned fit is actually worse;
  * RMSE and LL, so the comparison can be made on the reported quantities.

Then the restart check the author asked for: one `energy` split refitted
with `n_restarts_optimizer` 5 -> 20. If more restarts move neither the
log-marginal-likelihood nor the metrics, the warnings are noise from
discarded restarts and a footnote settles it.

`--collapse-check` answers the other half, and belongs here rather than in
`duplicate_ll_diagnostic.py` because it is a statement about the FIT, not
about the metrics: each split is fitted twice, once on the training set as
E2 uses it and once with repeated feature rows collapsed to their first
occurrence, and the two sets of kernel hyperparameters are put side by
side. `wine`'s repeated rows carry identical targets, which is exactly the
evidence a marginal likelihood reads as "the observation noise is zero";
if that is the cause of the floored `noise_level`, collapsing them must
lift it. `energy` has no repeated rows at all and therefore acts as the
control: there, collapsing must change nothing.

**Nothing here changes a default.** `n_restarts_optimizer=5` stays what
`GPMethod` uses, the collapsed fit is a diagnostic and never a reported
cell, and E2's `gp`/`wine` row is left exactly as it is (author's
decision, 2026-08-30: report two numbers with a footnote, do not
deduplicate).

Writes:
  results/gp_convergence_diagnostic.csv
  results/gp_restarts_check.csv
  results/gp_duplicate_collapse.csv   (--collapse-check)

Usage:
  python experiments/gp_convergence_diagnostic.py
  python experiments/gp_convergence_diagnostic.py --datasets energy --restart-split 3
  python experiments/gp_convergence_diagnostic.py --collapse-check --skip-warnings
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import metrics
from src.data import load_uci, n_uci_splits
from src.methods.gp import GPMethod
from src.results import RESULTS_DIR, append_generic_csv, git_commit_short, now_iso
from src.seeding import set_seed

DEFAULT_DATASETS = ("energy", "wine_quality_red")
OUT_PATH = RESULTS_DIR / "gp_convergence_diagnostic.csv"
RESTARTS_PATH = RESULTS_DIR / "gp_restarts_check.csv"
COLLAPSE_PATH = RESULTS_DIR / "gp_duplicate_collapse.csv"

# Features are standardised per split (an affine per-column map), so an exact
# duplicate in raw space is one here too; the rounding guards the last bits.
ROUND_DECIMALS = 9

# sklearn's defaults for the three kernel hyperparameters in `GPMethod`
# (ConstantKernel * RBF + WhiteKernel), all (1e-5, 1e5).
BOUNDS = (1e-5, 1e5)
# "on the bound" in the optimiser's own coordinates: within 1% of the log
# range of either end. The warning sklearn prints uses a similar test.
BOUND_TOL_LOG = 0.01 * (np.log10(BOUNDS[1]) - np.log10(BOUNDS[0]))


def _at_bound(value: float) -> str:
    log_v = np.log10(max(value, 1e-300))
    if log_v <= np.log10(BOUNDS[0]) + BOUND_TOL_LOG:
        return "lower"
    if log_v >= np.log10(BOUNDS[1]) - BOUND_TOL_LOG:
        return "upper"
    return "no"


def _collapse_repeats(X: np.ndarray, y: np.ndarray):
    """Keep the first occurrence of every repeated feature row. Diagnostic
    only — the reported cells are always fitted on the split as it comes."""
    _, first = np.unique(np.round(X, ROUND_DECIMALS), axis=0, return_index=True)
    keep = np.sort(first)
    return X[keep], y[keep]


def _fit_and_measure(dataset: str, split: int, n_restarts: int, collapse: bool = False) -> dict:
    import warnings
    import torch
    torch.set_num_threads(1)

    ds = load_uci(dataset, split=split)
    X_train, y_train = ds.X_train, ds.y_train
    n_train_full = len(X_train)
    if collapse:
        X_train, y_train = _collapse_repeats(X_train, y_train)

    set_seed(split)
    method = GPMethod(n_restarts_optimizer=n_restarts)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        method.fit(X_train, y_train, seed=split, use_cache=False)
    pred = method.predict(ds.X_test)

    scale = float(ds.y_scaler.scale_[0])
    mean = ds.y_scaler.inverse_transform(pred.mean.reshape(-1, 1)).ravel()
    y_test = ds.y_scaler.inverse_transform(ds.y_test.reshape(-1, 1)).ravel()
    std = pred.std_total * scale

    kernel = method.gp.kernel_
    constant_value = float(kernel.k1.k1.constant_value)
    length_scale = float(np.ravel(kernel.k1.k2.length_scale)[0])
    noise_level = float(kernel.k2.noise_level)

    messages = [str(w.message) for w in caught]
    return dict(
        dataset=dataset, split_index=split, n_restarts_optimizer=n_restarts,
        collapsed=collapse, n_train=len(X_train), n_train_full=n_train_full,
        n_repeats_removed=n_train_full - len(X_train), n_test=len(ds.X_test),
        n_warnings=len(messages),
        lbfgs_abnormal=sum("failed to converge" in m for m in messages),
        bound_warnings=sum("close to the specified" in m for m in messages),
        warned=len(messages) > 0,
        constant_value=constant_value, length_scale=length_scale, noise_level=noise_level,
        constant_value_at_bound=_at_bound(constant_value),
        length_scale_at_bound=_at_bound(length_scale),
        noise_level_at_bound=_at_bound(noise_level),
        log_marginal_likelihood=float(method.gp.log_marginal_likelihood_value_),
        rmse=metrics.rmse(y_test, mean), ll=metrics.ll(y_test, mean, std),
        picp95=metrics.picp(y_test, mean, std), mpiw95=metrics.mpiw(std),
        first_message=messages[0].splitlines()[0] if messages else "",
    )


def _cell(args) -> dict:
    return _fit_and_measure(*args)


def collapse_check(datasets, n_splits_override, workers) -> pd.DataFrame:
    """Each split fitted twice: as E2 has it, and with repeats collapsed.

    The comparison is of the FITTED KERNEL, not of the metrics — the two
    fits see different training sets, so their RMSE and LL are recorded but
    are not a like-for-like comparison. `noise_level` is the quantity the
    question is about.
    """
    cells = [
        (dataset, split, 5, collapse)
        for dataset in datasets
        for split in range(n_splits_override or n_uci_splits(dataset))
        for collapse in (False, True)
    ]
    print(f"\ncollapse check: {len(cells)} GP fits on {workers} workers", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(_cell, cells):
            rows.append(row)
            append_generic_csv(COLLAPSE_PATH, dict(
                {k: v for k, v in row.items() if k != "first_message"},
                timestamp=now_iso(), git_commit=git_commit_short(),
            ))
    df = pd.DataFrame(rows)

    for dataset, sub in df.groupby("dataset"):
        removed = sub[sub.collapsed].n_repeats_removed.mean()
        print(f"\n{dataset}: {removed:.1f} repeated training rows collapsed per split "
              f"(of {sub.n_train_full.iloc[0]})")
        print(f"  {'training set':16s} {'noise_level':>12s} {'length_scale':>13s} "
              f"{'constant':>11s} {'on bound':>9s} {'RMSE':>8s} {'LL':>8s}")
        for collapsed, label in ((False, "as in E2"), (True, "repeats collapsed")):
            g = sub[sub.collapsed == collapsed]
            on_bound = (g.noise_level_at_bound == "lower").sum()
            print(f"  {label:16s} {g.noise_level.median():12.5g} {g.length_scale.median():13.4f} "
                  f"{g.constant_value.median():11.4g} {on_bound:6d}/{len(g):2d} "
                  f"{g.rmse.mean():8.4f} {g.ll.mean():8.4f}")
    print(f"\nwrote {COLLAPSE_PATH}")
    return df


def _summarise(df: pd.DataFrame) -> None:
    for dataset, sub in df.groupby("dataset"):
        warned, clean = sub[sub.warned], sub[~sub.warned]
        print(f"\n{dataset}: {len(warned)}/{len(sub)} splits warned")
        print(f"  lbfgs ABNORMAL on {int(sub.lbfgs_abnormal.astype(bool).sum())} splits, "
              f"bound warnings on {int(sub.bound_warnings.astype(bool).sum())}")
        for param in ("constant_value", "length_scale", "noise_level"):
            counts = sub[f"{param}_at_bound"].value_counts().to_dict()
            print(f"  {param:16s} median {sub[param].median():12.5g}   at bound: {counts}")
        if len(warned) and len(clean):
            print(f"  {'group':8s} {'n':>3s} {'RMSE':>16s} {'LL':>16s} {'LML':>10s}")
            for name, g in (("warned", warned), ("clean", clean)):
                print(f"  {name:8s} {len(g):3d} {g.rmse.mean():8.4f} +/-{g.rmse.sem():6.4f} "
                      f"{g.ll.mean():8.4f} +/-{g.ll.sem():6.4f} "
                      f"{g.log_marginal_likelihood.mean():10.1f}")
            print(f"  difference (warned - clean): RMSE {warned.rmse.mean() - clean.rmse.mean():+.4f}, "
                  f"LL {warned.ll.mean() - clean.ll.mean():+.4f}")
            print("  NOTE: splits differ in difficulty, so this is not a paired test — "
                  "the restart check below is.")
        elif len(clean) == 0:
            print("  every split warned; there is no clean group to compare against")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--restart-split", type=int, default=None,
                        help="energy split for the 5-vs-20 restart check (default: the first that warns)")
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--collapse-check", action="store_true",
                        help="also fit each split with repeated feature rows collapsed "
                             "(diagnostic for the floored noise_level on wine)")
    parser.add_argument("--skip-warnings", action="store_true",
                        help="run only --collapse-check, skipping the warning survey and restart check")
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    if args.skip_warnings:
        if not args.collapse_check:
            parser.error("--skip-warnings leaves nothing to do without --collapse-check")
        collapse_check(datasets, args.splits, args.workers)
        return

    cells = [
        (dataset, split, 5)
        for dataset in datasets for split in range(args.splits or n_uci_splits(dataset))
    ]
    print(f"{len(cells)} GP fits on {args.workers} workers (n_restarts_optimizer=5, as in E2)", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(_cell, cells):
            rows.append(row)
            append_generic_csv(OUT_PATH, dict(row, timestamp=now_iso(), git_commit=git_commit_short()))
    df = pd.DataFrame(rows)
    _summarise(df)
    print(f"\nwrote {OUT_PATH}")

    # Paired restart check on one energy split: same data, same seed, more restarts.
    energy = df[df.dataset == "energy"]
    if energy.empty:
        return
    split = args.restart_split
    if split is None:
        warned = energy[energy.warned].split_index
        split = int(warned.iloc[0]) if len(warned) else int(energy.split_index.iloc[0])
    base = energy[energy.split_index == split].iloc[0].to_dict()
    print(f"\nrestart check on energy split {split} "
          f"({'warned' if base['warned'] else 'did not warn'} at 5 restarts): "
          f"refitting with {args.restarts}", flush=True)
    more = _fit_and_measure("energy", split, args.restarts)
    for row in (base, more):
        append_generic_csv(RESTARTS_PATH, dict(
            {k: v for k, v in row.items() if k != "first_message"},
            timestamp=now_iso(), git_commit=git_commit_short(),
        ))
    print(f"  {'restarts':>9s} {'warnings':>8s} {'LML':>12s} {'RMSE':>9s} {'LL':>9s} "
          f"{'constant':>12s} {'length':>10s} {'noise':>12s}")
    for row in (base, more):
        print(f"  {row['n_restarts_optimizer']:9d} {row['n_warnings']:8d} "
              f"{row['log_marginal_likelihood']:12.2f} {row['rmse']:9.4f} {row['ll']:9.4f} "
              f"{row['constant_value']:12.4g} {row['length_scale']:10.4f} {row['noise_level']:12.4g}")
    print(f"  delta: LML {more['log_marginal_likelihood'] - base['log_marginal_likelihood']:+.4f}, "
          f"RMSE {more['rmse'] - base['rmse']:+.5f}, LL {more['ll'] - base['ll']:+.5f}")
    print(f"\nwrote {RESTARTS_PATH}")

    if args.collapse_check:
        collapse_check(datasets, args.splits, args.workers)


if __name__ == "__main__":
    main()
