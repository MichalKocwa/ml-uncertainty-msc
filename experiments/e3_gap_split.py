"""E3 — gap splits: does a method's epistemic term notice a hole in the data?

Brief section 5.4 removes the middle third of ONE feature from training and
evaluates in-gap against in-range. Two changes, both decided by the author
(2026-08-31) and both recorded in docs/chapter4_notes.md:

**1. Every dimension, not one (foong2019's protocol).** Section 5.4 picks a
single feature "with the largest variance" — a criterion that does not
survive standardisation (afterwards every feature has variance 1) and, before
it, ranks features by the units they happen to be recorded in. Any
single-feature rule then needs defending in chapter 4; running all `d`
dimensions needs no defence, because it is what the reference work does, and
it reports a DISTRIBUTION over dimensions rather than one number that
depends on the choice. Correlation with the target is kept as the criterion
where a single dimension is unavoidable (`sin_gap`, and the chapter-4
description).

**2. A random-removal control, which the brief does not have.** Removing
`[q33, q66]` of a feature removes about a third of the training rows, so
`epi_gap_ratio > 1` on its own is ambiguous: the model has both a structured
hole AND less data. The control fits the same methods on a training set with
the same NUMBER of rows deleted at random, and is scored on the same test
partitions. The comparison that means something is gap against control, not
gap against E2. The control is fitted once per (dataset, split) and scored
against every dimension's partition — the removal is not dimension-specific,
so refitting it `d` times would cost eight times as much and measure the
same thing.

Design held from the accepted plan:
  * quantiles `q33`, `q66` computed on the FULL dataset in raw units, so the
    same hole means the same thing in every split;
  * the 20 literature splits are kept and crossed with the `d` dimensions,
    so each dimension has 20 paired measurements and the test rows stay the
    ones E2's `random` rows were scored on;
  * test rows are partitioned, never dropped: `split_type` is `gap_in` for
    test rows inside the removed band and `gap_out` for the rest;
  * `sin_gap` in the same file, where the hole is in the data by
    construction (E1's `sin_homo` at the same N is its natural control, so
    no control is refitted here).

Writes:
  results/e3_gap_split.csv   — section 8's schema, one row per (dataset, method, config_id, split, split_type)
  results/e3_gap_ratio.csv   — per fit: epi_gap_ratio and the partition sizes behind it

Usage:
  python experiments/e3_gap_split.py --workers 8
  python experiments/e3_gap_split.py --quick            # 2 splits, 2 dimensions
  python experiments/e3_gap_split.py --datasets sin_gap
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import metrics
from src.data import load_sin_gap, load_uci_raw, n_uci_splits, uci_split_indices
from src.methods import METHODS
from src.results import (
    RESULTS_DIR, append_generic_csv, append_result_row, git_commit_short, now_iso,
)
from src.seeding import set_seed

EXPERIMENT_ID = "e3_gap_split"
RESULTS_PATH = RESULTS_DIR / "e3_gap_split.csv"
RATIO_PATH = RESULTS_DIR / "e3_gap_ratio.csv"
EPOCHS_TABLE = "uci_epochs_sweep_final_combined.csv"

UCI_DATASETS = ("energy", "concrete")   # both d=8, both with a complete six-method row in E2
SYNTHETIC_DATASETS = ("sin_gap",)
METHOD_ORDER = ("mcd", "map", "laplace", "ensemble", "bbb", "gp")
BBB_ELBO_SAMPLES = 32                   # as in E2
# "middle third" by RANK (foong2019). Brief 5.4's value-based `[q33, q66]`
# does not survive tied features on these datasets — see middle_third_membership.
SYNTHETIC_SEEDS = (0, 1, 2)
SYNTHETIC_GAP = (2.0, 4.0)              # load_sin_gap trains on [0,2] u [4,6]
QUICK_SPLITS, QUICK_DIMS = 2, 2
CONTROL_SPLITS = 5                      # author, 2026-08-31 — see --control-splits
TIE_BREAK_SEED = 12345                  # fixed, so the middle third is reproducible

# `elbo_samples` is deliberately NOT reduced for E3, even though BBB is 70% of
# its cost. D14e measured `K` changing the very quantity E3 reports: the MPIW
# growth ratio runs 1.03 at K=1 against 1.21 at K=32, so a cheaper estimator
# flattens the band it is supposed to measure. P10 predicts
# `epi_gap_ratio[bbb] ~ 1`, so the bias would push the result TOWARDS the
# literature's prediction — the direction a reader is least likely to
# question. Estimator settings stay identical across experiments that measure
# the same quantity (docs/chapter4_notes.md, D31b).


def _epochs_by_dataset() -> dict:
    return {r.dataset: int(r.epochs) for r in pd.read_csv(RESULTS_DIR / EPOCHS_TABLE).itertuples()}


def _build(method_name: str, epochs: int):
    if method_name == "gp":
        return METHODS["gp"]()
    kwargs = dict(epochs=epochs)
    if method_name == "bbb":
        kwargs["elbo_samples"] = BBB_ELBO_SAMPLES
    return METHODS[method_name](**kwargs)


def _standardise(X_train, y_train, X_test):
    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
    x_std = np.where(x_std == 0, 1.0, x_std)
    y_mean, y_std = float(y_train.mean()), float(y_train.std())
    return ((X_train - x_mean) / x_std, (y_train - y_mean) / y_std,
            (X_test - x_mean) / x_std, y_mean, y_std)


def middle_third_membership(X: np.ndarray, dimension: int) -> np.ndarray:
    """Boolean over the FULL dataset: is this row in feature `dimension`'s middle third?

    **By rank, not by value** — the correction that made this experiment
    runnable (2026-08-31). Brief 5.4 removes rows whose feature value falls in
    `[q33, q66]`; with tied values that interval does not contain a third of
    the rows. Measured on the two datasets used here: of 16 feature columns
    only 4 have a `[q33, q66]` band holding ~1/3 of the rows. `energy`'s
    overall height takes 2 distinct values, so its band holds **100%** and the
    training set comes out empty (`ValueError: n must be positive, got 0`);
    `concrete`'s zero-inflated columns (fly ash, slag, superplasticiser) have
    `q33 = 0`, so their bands hold 67%.

    foong2019 sorts by the feature and takes the middle third of the sorted
    rows, which is tie-proof, and the author's decision was to follow that
    protocol. Ranks are computed once on the FULL dataset (the accepted design
    point: the same hole in every split), ties broken by original row order so
    the partition is deterministic.

    Degeneracy does not disappear, it only stops crashing: for a two-valued
    feature the "hole" is a set of rows sharing a value, and rows with the same
    value stay in training. `gap_leak_fraction` in the ratio file measures
    exactly that, per dimension, so such columns can be separated in analysis
    instead of silently averaged in.
    """
    # Ties broken at RANDOM, not by row order. Breaking them by row order would
    # make the hole follow whatever the file's ordering encodes — and `energy`
    # is a designed experiment whose rows are ordered by the design, so a
    # two-valued column's "hole" came out as a hole in a different variable
    # entirely (measured: epi_gap_ratio 35.9 for a column whose gap leaks 100%,
    # i.e. where there is no hole to find). With a random tie-break the removed
    # set inside a tied group is a random subset, so a degenerate column
    # honestly reports "no hole" instead of borrowing structure from row order.
    n = X.shape[0]
    tie_break = np.random.RandomState(TIE_BREAK_SEED).permutation(n)
    order = np.lexsort((tie_break, X[:, dimension]))
    membership = np.zeros(n, dtype=bool)
    membership[order[n // 3:2 * n // 3]] = True
    return membership


def gap_leak_fraction(values_kept: np.ndarray, values_removed: np.ndarray) -> float:
    """Fraction of KEPT training rows whose feature value lies inside the
    removed rows' value range — 0 for a clean hole, ~1 for a categorical one."""
    if len(values_removed) == 0 or len(values_kept) == 0:
        return float("nan")
    low, high = float(values_removed.min()), float(values_removed.max())
    return float(np.mean((values_kept >= low) & (values_kept <= high)))


def pin_threads() -> int:
    """One thread on every path (D23b): the count is part of the configuration,
    not an implementation detail — see experiments/thread_determinism_check.py."""
    import torch
    torch.set_num_threads(1)
    return torch.get_num_threads()


def _fit_and_predict(method_name, epochs, X_train, y_train, X_test, seed):
    """One fit, returned in ORIGINAL target units. Separate from scoring so a
    single control fit can be scored against every dimension's partition."""
    X_tr, y_tr, X_te, y_mean, y_std = _standardise(X_train, y_train, X_test)
    set_seed(seed)
    method = _build(method_name, epochs)
    method.fit(X_tr, y_tr, seed=seed, use_cache=False)
    pred = method.predict(X_te)
    return dict(
        mean=pred.mean * y_std + y_mean,
        std_total=pred.std_total * y_std,
        std_epi=np.sqrt(pred.var_epistemic) * y_std,
        var_alea=pred.var_aleatoric * y_std ** 2,
        var_epi=pred.var_epistemic * y_std ** 2,
        n_train=len(X_tr), n_parameters=method.n_parameters,
    )


def _partition_rows(dataset, method_name, config_id, split, fit, y_test, in_gap, seed):
    """Score one fit's two test partitions separately."""
    mean, std_total, std_epi = fit["mean"], fit["std_total"], fit["std_epi"]
    var_alea, var_epi = fit["var_alea"], fit["var_epi"]

    rows, partition_stats = [], {}
    for label, mask in (("gap_in", in_gap), ("gap_out", ~in_gap)):
        if not mask.any():
            continue
        rows.append(dict(
            experiment_id=EXPERIMENT_ID, dataset=dataset, method=method_name,
            config_id=config_id, split_index=split, init_seed=seed,
            n_train=fit["n_train"], n_test=int(mask.sum()), split_type=label,
            **metrics.summary(y_test[mask], mean[mask], std_total[mask]),
            mean_var_aleatoric=metrics.mean_var_aleatoric(var_alea[mask]),
            mean_var_epistemic=metrics.mean_var_epistemic(var_epi[mask]),
            epi_ratio=metrics.epi_ratio(var_epi[mask], var_alea[mask] + var_epi[mask]),
            n_parameters=fit["n_parameters"],
            torch_threads="not_applicable" if method_name == "gp" else pin_threads(),
            timestamp=now_iso(), git_commit=git_commit_short(),
        ))
        partition_stats[label] = dict(
            n=int(mask.sum()), mean_std_epi=float(np.mean(std_epi[mask])),
            rmse=metrics.rmse(y_test[mask], mean[mask]),
            ll=metrics.ll(y_test[mask], mean[mask], std_total[mask]),
            picp95=metrics.picp(y_test[mask], mean[mask], std_total[mask]),
        )
    return rows, partition_stats


def _ratio_row(dataset, method_name, variant, dimension, split, stats, n_train_full, n_train,
               n_distinct=None, leak=None):
    """`epi_gap_ratio` (brief 7.4): mean epistemic SIGMA in the gap over out of it."""
    inside, outside = stats.get("gap_in"), stats.get("gap_out")
    if inside is None or outside is None:
        return None
    return dict(
        dataset=dataset, method=method_name, variant=variant, dimension=dimension,
        split_index=split, n_train_full=n_train_full, n_train=n_train,
        n_removed=n_train_full - n_train,
        # `feature_n_distinct` and `gap_leak_fraction` say whether this
        # dimension's "hole" is a hole at all — see middle_third_membership.
        feature_n_distinct=n_distinct, gap_leak_fraction=leak,
        n_gap_in=inside["n"], n_gap_out=outside["n"],
        mean_std_epi_gap_in=inside["mean_std_epi"], mean_std_epi_gap_out=outside["mean_std_epi"],
        epi_gap_ratio=inside["mean_std_epi"] / outside["mean_std_epi"]
        if outside["mean_std_epi"] > 0 else np.nan,
        rmse_gap_in=inside["rmse"], rmse_gap_out=outside["rmse"],
        ll_gap_in=inside["ll"], ll_gap_out=outside["ll"],
        picp95_gap_in=inside["picp95"], picp95_gap_out=outside["picp95"],
        timestamp=now_iso(), git_commit=git_commit_short(),
    )


def _uci_cell(args):
    """One (dataset, split, variant, dimension, method) fit."""
    dataset, split, variant, dimension, method_name, epochs, n_dims = args
    pin_threads()
    X, y = load_uci_raw(dataset)
    train_idx, test_idx = uci_split_indices(dataset, split)
    X_train_full, y_train_full = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Middle thirds computed on the FULL dataset, so the hole is the same in
    # every split; membership is by rank, see middle_third_membership.
    membership = {j: middle_third_membership(X, j) for j in range(n_dims)}

    if variant == "gap":
        keep = ~membership[dimension][train_idx]
        config_id = f"gap,dim={dimension}"
    else:
        # Same number of rows removed, chosen at random: the control that
        # separates "the method sees a hole" from "the method has less data".
        # The count is the mean over dimensions, since one control fit is
        # scored against all of them.
        removed = int(round(np.mean([
            membership[j][train_idx].sum() for j in range(n_dims)
        ])))
        rng = np.random.RandomState(1000 + split)
        keep = np.ones(len(X_train_full), dtype=bool)
        keep[rng.choice(len(X_train_full), size=removed, replace=False)] = False
        config_id = "control,random_removal"

    X_train, y_train = X_train_full[keep], y_train_full[keep]

    # ONE fit. The gap variant is scored on its own dimension's partition; the
    # control is scored on every dimension's partition — the random removal is
    # not dimension-specific, so refitting it `d` times would measure the same
    # thing `d` times.
    fit = _fit_and_predict(method_name, epochs, X_train, y_train, X_test, seed=split)
    dimensions = [dimension] if variant == "gap" else list(range(n_dims))

    all_rows, all_ratios = [], []
    for j in dimensions:
        in_gap = membership[j][test_idx]
        scored_id = config_id if variant == "gap" else f"{config_id},scored_dim={j}"
        rows, stats = _partition_rows(dataset, method_name, scored_id, split, fit,
                                      y_test, in_gap, seed=split)
        all_rows.extend(rows)
        removed_values = X_train_full[membership[j][train_idx], j]
        kept_values = X_train_full[~membership[j][train_idx], j]
        ratio = _ratio_row(dataset, method_name, variant, j, split, stats,
                           len(X_train_full), len(X_train),
                           n_distinct=int(len(np.unique(X[:, j]))),
                           leak=gap_leak_fraction(kept_values, removed_values))
        if ratio is not None:
            all_ratios.append(ratio)
    return all_rows, all_ratios


def _synthetic_cell(args):
    """`sin_gap`: the hole is in the data by construction, so there is nothing
    to remove — the test grid is partitioned by the training gap's edges."""
    dataset, seed, method_name, epochs = args
    ds = load_sin_gap(seed=seed)
    # 1-D and continuous: the training gap's own edges, no rank trick needed.
    #
    # `gap_out` is the TRAINING SUPPORT, not "everything that is not the gap".
    # The evaluation grid runs over [-2, 8] while training data live in
    # [0, 2] u [4, 6], so "not the gap" would include the extrapolation tails,
    # where the epistemic term is largest by design — and the ratio would then
    # measure gap against extrapolation instead of gap against dense data.
    # Measured cost of getting this wrong: the GP came out at 0.286 (its
    # uncertainty in the gap being SMALLER than in the tails), against 3.266
    # for the same quantity in E1's `epistemic_growth.csv`, which compares with
    # the in-range region. The extrapolation points are excluded from both
    # partitions here; E5 and `epistemic_growth.csv` are where extrapolation is
    # the subject.
    x_eval = ds.X_eval.ravel()
    train_low, train_high = float(ds.X_train.min()), float(ds.X_train.max())
    in_gap = (x_eval >= SYNTHETIC_GAP[0]) & (x_eval <= SYNTHETIC_GAP[1])
    in_support = (x_eval >= train_low) & (x_eval <= train_high) & ~in_gap

    set_seed(seed)
    method = _build(method_name, epochs)
    method.fit(ds.X_train, ds.y_train, seed=seed, use_cache=False)
    pred = method.predict(ds.X_eval)

    std_epi = np.sqrt(pred.var_epistemic)
    rows, stats = [], {}
    for label, mask in (("gap_in", in_gap), ("gap_out", in_support)):
        rows.append(dict(
            experiment_id=EXPERIMENT_ID, dataset=dataset, method=method_name,
            config_id="gap,dim=0", split_index=seed, init_seed=seed,
            n_train=len(ds.X_train), n_test=int(mask.sum()), split_type=label,
            **metrics.summary(ds.y_eval_noisy[mask], pred.mean[mask], pred.std_total[mask]),
            mean_var_aleatoric=metrics.mean_var_aleatoric(pred.var_aleatoric[mask]),
            mean_var_epistemic=metrics.mean_var_epistemic(pred.var_epistemic[mask]),
            epi_ratio=metrics.epi_ratio(pred.var_epistemic[mask],
                                        pred.var_aleatoric[mask] + pred.var_epistemic[mask]),
            n_parameters=method.n_parameters,
            torch_threads="not_applicable" if method_name == "gp" else pin_threads(),
            timestamp=now_iso(), git_commit=git_commit_short(),
        ))
        stats[label] = dict(
            n=int(mask.sum()), mean_std_epi=float(np.mean(std_epi[mask])),
            rmse=metrics.rmse(ds.y_eval_noisy[mask], pred.mean[mask]),
            ll=metrics.ll(ds.y_eval_noisy[mask], pred.mean[mask], pred.std_total[mask]),
            picp95=metrics.picp(ds.y_eval_noisy[mask], pred.mean[mask], pred.std_total[mask]),
        )
    ratio = _ratio_row(dataset, method_name, "gap", 0, seed, stats,
                       len(ds.X_train), len(ds.X_train),
                       n_distinct=len(np.unique(ds.X_train)), leak=0.0)
    return rows, ([ratio] if ratio else [])


def _done_cells() -> set:
    """`(dataset, method, variant, dimension, split)` already in the ratio file.

    Keyed off `e3_gap_ratio.csv` rather than the schema file because that is
    the one row per FIT; a control fit writes `d` of them, and any of those
    present means the fit happened.
    """
    if not RATIO_PATH.exists():
        return set()
    df = pd.read_csv(RATIO_PATH)
    return {
        (r.dataset, r.method, r.variant,
         None if r.variant == "control" else int(r.dimension), int(r.split_index))
        for r in df.itertuples()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=",".join(SYNTHETIC_DATASETS + UCI_DATASETS))
    parser.add_argument("--methods", type=str, default=",".join(METHOD_ORDER))
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--dims", type=int, default=None, help="use only the first N feature dimensions")
    parser.add_argument("--variants", type=str, default="gap,control")
    parser.add_argument("--control-splits", type=int, default=CONTROL_SPLITS,
                        help="splits for the random-removal control (it is a methodological "
                             "control, not a headline number, so it needs fewer than the gap arm)")
    parser.add_argument("--quick", action="store_true", help=f"{QUICK_SPLITS} splits, {QUICK_DIMS} dimensions")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    pin_threads()
    datasets = args.datasets.split(",")
    methods = args.methods.split(",")
    variants = args.variants.split(",")
    n_splits = QUICK_SPLITS if args.quick else args.splits
    n_dims_limit = QUICK_DIMS if args.quick else args.dims
    epochs_by_dataset = _epochs_by_dataset()
    done = _done_cells()

    uci_cells, synthetic_cells = [], []
    for dataset in datasets:
        if dataset in SYNTHETIC_DATASETS:
            synthetic_cells += [
                (dataset, seed, method, 2000)
                for seed in (SYNTHETIC_SEEDS[:n_splits] if n_splits else SYNTHETIC_SEEDS)
                for method in methods
                if (dataset, method, "gap", 0, seed) not in done
            ]
            continue
        X, _ = load_uci_raw(dataset)
        n_dims = X.shape[1]
        dims = range(min(n_dims_limit, n_dims) if n_dims_limit else n_dims)
        splits = range(n_splits or n_uci_splits(dataset))
        for split in splits:
            for method in methods:
                if "gap" in variants:
                    uci_cells += [
                        (dataset, split, "gap", j, method, epochs_by_dataset[dataset], n_dims)
                        for j in dims if (dataset, method, "gap", j, split) not in done
                    ]
                if ("control" in variants and split < args.control_splits
                        and (dataset, method, "control", None, split) not in done):
                    uci_cells.append((dataset, split, "control", None, method,
                                      epochs_by_dataset[dataset], n_dims))

    print(f"{len(synthetic_cells)} synthetic + {len(uci_cells)} UCI fits on {args.workers} workers",
          flush=True)

    def _write(result):
        rows, ratios = result
        for row in rows:
            append_result_row(RESULTS_PATH, row)
        for ratio in ratios:
            append_generic_csv(RATIO_PATH, ratio)
        if ratios:
            r = ratios[0]
            print(f"  {r['dataset']:9s} {r['method']:9s} {r['variant']:8s} dim={r['dimension']} "
                  f"split={r['split_index']:2d}  epi_gap_ratio={r['epi_gap_ratio']:7.3f} "
                  f"(n_train {r['n_train']}/{r['n_train_full']})", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(_synthetic_cell, synthetic_cells):
            _write(result)
        for result in pool.map(_uci_cell, uci_cells):
            _write(result)

    if RATIO_PATH.exists():
        df = pd.read_csv(RATIO_PATH)
        print("\nepi_gap_ratio, mean over splits and dimensions:")
        print(df.pivot_table(index="method", columns=["dataset", "variant"],
                             values="epi_gap_ratio").round(3).to_string())
    print(f"\nwrote {RESULTS_PATH} and {RATIO_PATH}")


if __name__ == "__main__":
    main()
