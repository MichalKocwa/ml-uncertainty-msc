"""Does the GP's `wine_quality_red` log-likelihood come from memorised duplicates?

E2's `gp` / `wine_quality_red` cell reports LL = +0.424 while every other
method on the same rows reports about -0.96 at an indistinguishable RMSE
(0.63-0.65). The mechanism found on split 0: `wine` contains 240 exactly
repeated feature rows out of 1599, the fitted `noise_level` sits on
sklearn's lower bound (1e-5), and the GP therefore interpolates any test
row that also appears in training - median |error| 1e-5, median sigma
0.0036, LL +4.73 on those points against -1.02 on the rest.

This script measures that split by split, for ALL SIX methods, so the
comparison in chapter 5 is "the interpolator memorises training points,
the parametric methods do not" with a number attached rather than an
assertion about one method. Test points are partitioned by whether their
FEATURE vector appears in the training split (and, among those, whether
the target agrees), and the log-likelihood is averaged inside each part.

The `dropout_p`/`noise_level`/anything-else defaults are untouched: this
is the E2 configuration re-run with the per-point likelihood kept instead
of only its mean. `ll_all` is checked against `results/e2_uci.csv` and the
two must agree to floating-point noise - if they do not, the diagnostic is
measuring a different model from the table and says so.

Writes:
  results/duplicate_ll_diagnostic.csv  — one row per (dataset, method, split)
  results/dataset_duplicates.csv       — duplicate census for all six datasets

Usage:
  python experiments/duplicate_ll_diagnostic.py
  python experiments/duplicate_ll_diagnostic.py --datasets wine_quality_red --workers 8
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data import load_uci, n_uci_splits
from src.methods import METHODS
from src.results import RESULTS_DIR, append_generic_csv, git_commit_short, now_iso
from src.seeding import set_seed

METHOD_ORDER = ("mcd", "map", "laplace", "ensemble", "bbb", "gp")
ALL_DATASETS = ("yacht", "energy", "concrete", "wine_quality_red", "kin8nm", "power_plant")
DEFAULT_DATASETS = ("wine_quality_red",)
BBB_ELBO_SAMPLES = 32  # as in E2
EPOCHS_TABLE = "uci_epochs_sweep_final_combined.csv"
OUT_PATH = RESULTS_DIR / "duplicate_ll_diagnostic.csv"
CENSUS_PATH = RESULTS_DIR / "dataset_duplicates.csv"
UCI_ROOT = Path(__file__).resolve().parent.parent / "data" / "uci_splits"

# Features are standardised per split, an affine per-column map, so an exact
# duplicate in raw space is an exact duplicate here too. Rounding guards
# against the last-bit differences that affine rescaling can introduce.
ROUND_DECIMALS = 9


def _build(method_name: str, epochs: int):
    if method_name == "gp":
        return METHODS["gp"]()
    kwargs = dict(epochs=epochs)
    if method_name == "bbb":
        kwargs["elbo_samples"] = BBB_ELBO_SAMPLES
    return METHODS[method_name](**kwargs)


def _pointwise_ll(y, mean, std):
    return -0.5 * np.log(2 * np.pi * std ** 2) - 0.5 * ((y - mean) / std) ** 2


def _duplicate_mask(X_train, y_train, X_test, y_test):
    """`(is_duplicate, target_agrees)` for every test row.

    A test row is a duplicate when its feature vector occurs in the
    training split. `target_agrees` distinguishes the two cases that matter
    for an interpolator: repeating a training point with the SAME target
    (the GP reproduces it and is rewarded) from repeating it with a
    DIFFERENT one (the GP reproduces the training target confidently and is
    punished) - `wine`'s quality score is an integer rating, so both occur.
    """
    train_keys = {}
    for row, target in zip(np.round(X_train, ROUND_DECIMALS), y_train):
        train_keys.setdefault(tuple(row), []).append(float(target))
    is_dup, agrees = [], []
    for row, target in zip(np.round(X_test, ROUND_DECIMALS), y_test):
        matches = train_keys.get(tuple(row))
        is_dup.append(matches is not None)
        agrees.append(
            bool(matches is not None and np.any(np.isclose(matches, float(target), atol=1e-9)))
        )
    return np.array(is_dup), np.array(agrees)


def _cell(args) -> dict:
    dataset, method_name, split, epochs = args
    import warnings
    import torch
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")

    ds = load_uci(dataset, split=split)
    set_seed(split)
    method = _build(method_name, epochs)
    method.fit(ds.X_train, ds.y_train, seed=split, use_cache=False)
    pred = method.predict(ds.X_test)

    scale = float(ds.y_scaler.scale_[0])
    mean = ds.y_scaler.inverse_transform(pred.mean.reshape(-1, 1)).ravel()
    y_test = ds.y_scaler.inverse_transform(ds.y_test.reshape(-1, 1)).ravel()
    y_train = ds.y_scaler.inverse_transform(ds.y_train.reshape(-1, 1)).ravel()
    std = pred.std_total * scale

    ll = _pointwise_ll(y_test, mean, std)
    is_dup, agrees = _duplicate_mask(ds.X_train, y_train, ds.X_test, y_test)
    uniq = ~is_dup
    dup_same = is_dup & agrees
    dup_diff = is_dup & ~agrees

    def part(mask, prefix):
        if not mask.any():
            return {f"{prefix}_n": 0, f"{prefix}_ll": np.nan,
                    f"{prefix}_rmse": np.nan, f"{prefix}_median_std": np.nan}
        return {
            f"{prefix}_n": int(mask.sum()),
            f"{prefix}_ll": float(ll[mask].mean()),
            f"{prefix}_rmse": float(np.sqrt(np.mean((y_test[mask] - mean[mask]) ** 2))),
            f"{prefix}_median_std": float(np.median(std[mask])),
        }

    row = dict(
        dataset=dataset, method=method_name, split_index=split, epochs=epochs,
        n_test=len(y_test), n_train=len(y_train),
        ll_all=float(ll.mean()),
        rmse_all=float(np.sqrt(np.mean((y_test - mean) ** 2))),
        **part(is_dup, "dup"), **part(uniq, "uniq"),
        **part(dup_same, "dup_same_y"), **part(dup_diff, "dup_diff_y"),
    )
    row["ll_gap_dup_minus_uniq"] = row["dup_ll"] - row["uniq_ll"]
    # How much of the reported mean LL is contributed by the duplicates, i.e.
    # what the cell would lose if those rows scored like the unique ones.
    row["ll_shift_from_dups"] = row["ll_all"] - row["uniq_ll"]
    return row


def census(datasets) -> pd.DataFrame:
    """Duplicate rates per dataset: within the file, and across the 20 splits."""
    rows = []
    for dataset in datasets:
        root = UCI_ROOT / dataset
        data = np.loadtxt(root / "data.txt")
        feature_idx = np.loadtxt(root / "index_features.txt").astype(int)
        X = np.round(data[:, feature_idx], ROUND_DECIMALS)
        n_unique = len(np.unique(X, axis=0))
        shares = []
        for split in range(n_uci_splits(dataset)):
            tr = np.loadtxt(root / f"index_train_{split}.txt").astype(int)
            te = np.loadtxt(root / f"index_test_{split}.txt").astype(int)
            train_keys = {tuple(r) for r in X[tr]}
            shares.append(float(np.mean([tuple(r) in train_keys for r in X[te]])))
        rows.append(dict(
            dataset=dataset, n_rows=len(X), n_unique_feature_rows=n_unique,
            n_repeated_rows=len(X) - n_unique,
            repeated_fraction=1.0 - n_unique / len(X),
            test_rows_seen_in_train_mean=float(np.mean(shares)),
            test_rows_seen_in_train_min=float(np.min(shares)),
            test_rows_seen_in_train_max=float(np.max(shares)),
            timestamp=now_iso(), git_commit=git_commit_short(),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(CENSUS_PATH, index=False)
    return df


def _check_against_e2(df: pd.DataFrame) -> None:
    path = RESULTS_DIR / "e2_uci.csv"
    if not path.exists():
        return
    e2 = pd.read_csv(path).set_index(["dataset", "method", "split_index"]).ll
    diffs = [
        abs(r.ll_all - e2[(r.dataset, r.method, r.split_index)])
        for r in df.itertuples() if (r.dataset, r.method, r.split_index) in e2.index
    ]
    if diffs:
        print(f"\nreproduction check against e2_uci.csv: {len(diffs)} cells, "
              f"max |delta LL| = {max(diffs):.2e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--methods", type=str, default=",".join(METHOD_ORDER))
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--census-only", action="store_true",
                        help="only count duplicates in all six datasets, fit nothing")
    args = parser.parse_args()

    print("duplicate census (all six datasets):")
    print(census(ALL_DATASETS).drop(columns=["timestamp", "git_commit"]).round(4).to_string(index=False))
    print(f"wrote {CENSUS_PATH}")
    if args.census_only:
        return

    datasets = args.datasets.split(",")
    methods = args.methods.split(",")
    epochs_by_dataset = {
        r.dataset: int(r.epochs) for r in pd.read_csv(RESULTS_DIR / EPOCHS_TABLE).itertuples()
    }
    cells = [
        (dataset, method, split, epochs_by_dataset[dataset])
        for dataset in datasets for method in methods
        for split in range(args.splits or n_uci_splits(dataset))
    ]
    print(f"\n{len(cells)} cells on {args.workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(_cell, cells):
            rows.append(row)
            append_generic_csv(OUT_PATH, dict(row, timestamp=now_iso(), git_commit=git_commit_short()))
    df = pd.DataFrame(rows)

    for (dataset,), sub in df.groupby(["dataset"]):
        print(f"\n{dataset}: mean over splits, log-likelihood by partition of the test set")
        print(f"  {'method':9s} {'LL all':>8s} {'LL dup':>8s} {'LL uniq':>8s} {'gap':>8s} "
              f"{'shift':>8s} {'n dup':>6s} {'n uniq':>6s} {'RMSE dup':>9s} {'RMSE uniq':>9s}")
        for method in [m for m in METHOD_ORDER if m in set(sub.method)]:
            g = sub[sub.method == method]
            print(f"  {method:9s} {g.ll_all.mean():8.3f} {g.dup_ll.mean():8.3f} "
                  f"{g.uniq_ll.mean():8.3f} {g.ll_gap_dup_minus_uniq.mean():8.3f} "
                  f"{g.ll_shift_from_dups.mean():8.3f} {g.dup_n.mean():6.1f} "
                  f"{g.uniq_n.mean():6.1f} {g.dup_rmse.mean():9.4f} {g.uniq_rmse.mean():9.4f}")
        same, diff = sub.dup_same_y_n.mean(), sub.dup_diff_y_n.mean()
        print(f"  duplicated test rows per split: {same:.1f} with the same target, "
              f"{diff:.1f} with a different one")

    _check_against_e2(df)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
