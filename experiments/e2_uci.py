"""E2 — the main UCI benchmark table (brief sections 5.2, 8, 9).

Six datasets x six methods x 20 literature splits. Epoch counts are
per-dataset (D30); `elbo_samples=32` for BBB (author, 2026-08-28); every
other hyperparameter is the shared default from `src/methods/`.

**Method order: mcd, map, laplace, ensemble, bbb, gp** (author's decision).
MC dropout runs FIRST despite not being the cheapest: it is the only method
with published per-split reference numbers, so P13 — the single external
check on whether this protocol is right at all — resolves on it. A protocol
error should surface after an hour, not after fifteen.

**GP runs last, separately, datasets ascending in N, with a 20-minute
per-split limit.** E0 measured exact-GP cost growing as ~N^2.3 with a run
at N=8000 that once finished in 42 minutes and once did not finish in over
two hours; 20 minutes is a bit over twice the good run's time at the
largest N we attempt. The limit is checked AFTER each split, not enforced
mid-fit — sklearn's optimiser cannot be interrupted from here, so a single
split may overrun before the dataset is abandoned. On exceeding it, the
remaining splits of that dataset are skipped and the cell becomes "—" with
a footnote, exactly like `power_plant` (D5) — that is a result to describe,
not a failure. Skips are logged to `results/e2_gp_skipped.csv` so the
footnote has a measured number behind it.

**One thread everywhere (2026-08-31).** `pin_threads()` runs on the worker
path, the sequential path and `--probe`, and every row records the count in
`torch_threads`. The first version of this table did not: the parallel worker
pinned one thread while the sequential path inherited torch's default of 8,
and `experiments/thread_determinism_check.py` measured that the same seed
then produces a different network — up to 6.75e-4 in RMSE and 3.1e-3 in LL,
small against the between-split spread but not reproducible. The mixed-thread
table is kept as `results/e2_uci_mixed_threads.csv`.

**No model cache, deliberately.** `train_time_s` is a reported result
(D23, P11) and a cache hit would turn it into a state-dict load time.
Interruption is handled by resuming from the CSV instead: every
(dataset, method, split_index) already present is skipped, and each row is
appended the moment its fit finishes, so a kill costs at most one fit.

Writes:
  results/e2_uci.csv                — section 8's schema, one row per (dataset, method, split)
  results/calibration_curves.csv    — (experiment_id, dataset, method, split_index, alpha, empirical)
  results/literature_comparison.csv — P13, paired per split against Gal's own files
  results/e2_gp_skipped.csv         — GP cells abandoned on the time limit

Usage:
  python experiments/e2_uci.py --probe        # timing probe only (see PROBE)
  python experiments/e2_uci.py                # full run, resuming
  python experiments/e2_uci.py --quick        # 2 splits, for smoke-testing the plumbing
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import metrics
from src.data import UCI_SPEC, load_uci, n_uci_splits
from src.methods import METHODS
from src.results import (
    RESULTS_DIR, append_generic_csv, append_result_row, git_commit_short, now_iso,
)
from src.seeding import set_seed

EXPERIMENT_ID = "e2_uci"
RESULTS_PATH = RESULTS_DIR / "e2_uci.csv"
CALIBRATION_PATH = RESULTS_DIR / "calibration_curves.csv"
LITERATURE_PATH = RESULTS_DIR / "literature_comparison.csv"
GP_SKIPPED_PATH = RESULTS_DIR / "e2_gp_skipped.csv"

METHOD_ORDER = ("mcd", "map", "laplace", "ensemble", "bbb", "gp")
# Ascending in N, so the cheapest evidence arrives first and GP's limit is
# hit (if at all) only after the smaller datasets are already banked.
DATASET_ORDER = ("yacht", "energy", "concrete", "wine_quality_red", "kin8nm", "power_plant")

# D5: exact GP is not run on the largest set — the cell is "—" with a
# footnote, never a value borrowed from a different N.
GP_EXCLUDED = {"power_plant"}
GP_TIME_LIMIT_S = 20 * 60

BBB_ELBO_SAMPLES = 32  # author, 2026-08-28 (D14e closed)
EPOCHS_TABLE = "uci_epochs_sweep_final_combined.csv"  # D30

QUICK_SPLITS = 2

# The timing probe (author's decision): the two methods whose cost is not
# yet measured, on the most expensive dataset either will attempt, plus one
# MC-dropout cell cheap enough to give P13 a first reading before the full
# run starts.
PROBE = (("kin8nm", "laplace"), ("kin8nm", "gp"), ("wine_quality_red", "mcd"))

# `init_seed = split_index`: the split files fix the data partition, so the
# seed's only job is the network initialisation (brief section 5.2). Tying
# it to the split index gives every split a different init while keeping
# the whole table reproducible from the split number alone.


def _epochs_by_dataset() -> dict:
    df = pd.read_csv(RESULTS_DIR / EPOCHS_TABLE)
    return {r.dataset: int(r.epochs) for r in df.itertuples()}


def _build(method_name: str, epochs: int):
    if method_name == "gp":
        return METHODS["gp"]()
    kwargs = dict(epochs=epochs)
    if method_name == "bbb":
        kwargs["elbo_samples"] = BBB_ELBO_SAMPLES
    return METHODS[method_name](**kwargs)


def _done_cells() -> set:
    if not RESULTS_PATH.exists():
        return set()
    df = pd.read_csv(RESULTS_PATH)
    return {(r.dataset, r.method, int(r.split_index)) for r in df.itertuples()}


def _gp_abandoned() -> set:
    if not GP_SKIPPED_PATH.exists():
        return set()
    return set(pd.read_csv(GP_SKIPPED_PATH)["dataset"])


def pin_threads() -> int:
    """One thread, on EVERY path — worker, sequential and `--probe` alike.

    `set_seed` pins this too (src/seeding.py), which covers model construction;
    this call covers everything before the first `set_seed` in a process and
    makes the setting greppable from the experiment script rather than only
    implied by a library. Both were added 2026-08-31 after the mixed-thread
    table was found (see `experiments/thread_determinism_check.py`).
    """
    import torch
    torch.set_num_threads(1)
    return torch.get_num_threads()


def compute_cell(args) -> tuple:
    """Worker entry point: one cell, computed and RETURNED, never written.

    Writing stays with the parent process so the CSV has a single writer —
    concurrent appends from 8 processes would interleave mid-line — and so
    that resume semantics (`_done_cells`) keep working unchanged.

    `torch.set_num_threads(1)`: the shared backbone is a 1x50 network on
    batches of 128, far too small for intra-op threading to pay for itself.
    Measured on BBB/`yacht`/K=32: 7.06 s single-threaded against 8.01 s at
    torch's default of 8 threads — one thread is both faster per fit and
    leaves the other cores free for other fits. Thread count perturbs
    float64 reduction order (measured max |delta| 1.4e-15 on a `concrete`
    prediction), so it is pinned rather than left to the default.
    """
    dataset, method_name, split, epochs = args
    pin_threads()
    return _compute_cell(dataset, method_name, split, epochs)


def run_cell(dataset: str, method_name: str, split: int, epochs: int) -> dict:
    """One (dataset, method, split), computed AND written. Sequential path."""
    row, calibration = _compute_cell(dataset, method_name, split, epochs)
    _write_cell(row, calibration)
    return row


def _write_cell(row: dict, calibration: list) -> None:
    append_result_row(RESULTS_PATH, row)
    for cal_row in calibration:
        append_generic_csv(CALIBRATION_PATH, cal_row)


def _compute_cell(dataset: str, method_name: str, split: int, epochs: int) -> tuple:
    """One (dataset, method, split). Returns `(row, calibration_rows)`.

    Metrics are computed after inverting the target standardisation, on the
    WHOLE predictive distribution and not just its mean (brief section 5.3:
    `sigma_original = sigma_standardised * scaler.scale_` — forgetting this
    gives a correct RMSE and a wrong NLL).
    """
    ds = load_uci(dataset, split=split)
    seed = split

    set_seed(seed)
    method = _build(method_name, epochs)
    t0 = time.perf_counter()
    method.fit(ds.X_train, ds.y_train, seed=seed, use_cache=False)
    train_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = method.predict(ds.X_test)
    predict_time_ms_per_1k = (time.perf_counter() - t0) * 1000.0 / len(ds.X_test) * 1000.0

    scale = float(ds.y_scaler.scale_[0])
    mean = ds.y_scaler.inverse_transform(pred.mean.reshape(-1, 1)).ravel()
    y_test = ds.y_scaler.inverse_transform(ds.y_test.reshape(-1, 1)).ravel()
    std_total = pred.std_total * scale
    var_alea = pred.var_aleatoric * scale ** 2
    var_epi = pred.var_epistemic * scale ** 2

    row = dict(
        experiment_id=EXPERIMENT_ID, dataset=dataset, method=method_name,
        config_id=f"epochs={epochs}" + (f",K={BBB_ELBO_SAMPLES}" if method_name == "bbb" else ""),
        split_index=split, init_seed=seed,
        n_train=len(ds.X_train), n_test=len(ds.X_test), split_type="random",
        **metrics.summary(y_test, mean, std_total),
        mean_var_aleatoric=metrics.mean_var_aleatoric(var_alea),
        mean_var_epistemic=metrics.mean_var_epistemic(var_epi),
        epi_ratio=metrics.epi_ratio(var_epi, var_alea + var_epi),
        train_time_s=train_time_s,
        predict_time_ms_per_1k=predict_time_ms_per_1k,
        n_parameters=method.n_parameters,
        # `gp` is sklearn: `torch.set_num_threads` does not reach it, and it was
        # measured identical to 12 significant figures at 1 and 8 BLAS threads
        # (concrete, wine). Recording the fact rather than the ambient torch
        # value keeps the column meaning one thing. NOT the string "n/a": that
        # is in pandas' default NA list and reads back as NaN, i.e. as "never
        # filled in", which is a different statement.
        torch_threads="not_applicable" if method_name == "gp" else pin_threads(),
        timestamp=now_iso(), git_commit=git_commit_short(),
    )
    alphas, empirical = metrics.calibration_curve(y_test, mean, std_total)
    calibration = [dict(
        experiment_id=EXPERIMENT_ID, dataset=dataset, method=method_name,
        split_index=split, alpha=float(alpha), empirical=float(emp),
    ) for alpha, emp in zip(alphas, empirical)]
    return row, calibration


def _cells(datasets, methods, splits_by_dataset, done, gp_abandoned):
    """(dataset, method, split) still to do, in the decided order: methods
    outermost so a whole method finishes before the next starts, GP last."""
    for method_name in methods:
        for dataset in datasets:
            if method_name == "gp" and (dataset in GP_EXCLUDED or dataset in gp_abandoned):
                continue
            for split in range(splits_by_dataset[dataset]):
                if (dataset, method_name, split) in done:
                    continue
                yield dataset, method_name, split


def run_parallel(datasets, methods, n_splits_override, epochs_by_dataset, workers):
    """Same cells, same order, `workers` processes.

    Cells are submitted METHOD BY METHOD and drained before the next method
    starts, so the ordering the author asked for (mcd first, so P13 resolves
    early) survives parallelisation. GP is never parallelised: its
    per-split time limit is a sequential decision — abandoning a dataset
    after one overrun is meaningless if seven more of its splits are
    already in flight.

    `train_time_s` recorded here is NOT a cost measurement: 8 concurrent
    single-threaded fits each take ~1.6x their solo time (measured: 7.06 s
    solo, ~11.3 s under 8-way concurrency). The cost column comes from
    `--timing-pass`, a separate exclusive sequential run — the same
    separation D14k had to impose after the fact.
    """
    done = _done_cells()
    gp_abandoned = _gp_abandoned()
    splits_by_dataset = {d: (n_splits_override or n_uci_splits(d)) for d in datasets}

    for method_name in methods:
        if method_name == "gp":
            run(datasets, ["gp"], n_splits_override, epochs_by_dataset)
            continue
        cells = [
            (dataset, method_name, split, epochs_by_dataset[dataset])
            for dataset in datasets
            for split in range(splits_by_dataset[dataset])
            if (dataset, method_name, split) not in done
        ]
        if not cells:
            continue
        print(f"\n{method_name}: {len(cells)} cells on {workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for row, calibration in pool.map(compute_cell, cells):
                _write_cell(row, calibration)
                print(f"  {row['method']:8s} {row['dataset']:17s} split={row['split_index']:2d}  "
                      f"rmse={row['rmse']:9.4f} ll={row['ll']:9.4f}", flush=True)


def timing_pass(datasets, methods, epochs_by_dataset):
    """The cost column (D23, P11), measured on its own terms.

    Split 0 only, one method at a time, one thread, nothing else running —
    so `train_time_s` here is a measurement rather than a by-product of
    whatever else shared the CPU. Written to `results/e2_cost.csv`, not to
    the main results file, because it is a different quantity from the
    `train_time_s` column produced by a parallel run.
    """
    pin_threads()
    out = RESULTS_DIR / "e2_cost.csv"
    for method_name in methods:
        for dataset in datasets:
            if method_name == "gp" and (dataset in GP_EXCLUDED or dataset in _gp_abandoned()):
                continue
            row, _ = _compute_cell(dataset, method_name, 0, epochs_by_dataset[dataset])
            append_generic_csv(out, dict(
                experiment_id=EXPERIMENT_ID, dataset=dataset, method=method_name,
                split_index=0, n_train=row["n_train"], epochs=epochs_by_dataset[dataset],
                train_time_s=row["train_time_s"],
                predict_time_ms_per_1k=row["predict_time_ms_per_1k"],
                n_parameters=row["n_parameters"], torch_threads=1, workers=1,
                timestamp=now_iso(), git_commit=git_commit_short(),
            ))
            print(f"  {method_name:8s} {dataset:17s} train={row['train_time_s'] / 60:7.3f} min  "
                  f"predict={row['predict_time_ms_per_1k']:8.1f} ms/1k", flush=True)
    print(f"\nwrote {out}")


def run(datasets, methods, n_splits_override, epochs_by_dataset):
    done = _done_cells()
    gp_abandoned = _gp_abandoned()
    splits_by_dataset = {
        d: (n_splits_override or n_uci_splits(d)) for d in datasets
    }
    gp_over_limit = set()

    for dataset, method_name, split in _cells(datasets, methods, splits_by_dataset, done, gp_abandoned):
        if method_name == "gp" and dataset in gp_over_limit:
            continue
        epochs = epochs_by_dataset[dataset]
        row = run_cell(dataset, method_name, split, epochs)
        print(f"  {method_name:8s} {dataset:17s} split={split:2d}  "
              f"rmse={row['rmse']:9.4f} ll={row['ll']:9.4f} "
              f"({row['train_time_s'] / 60:6.2f} min)", flush=True)

        if method_name == "gp" and row["train_time_s"] > GP_TIME_LIMIT_S:
            gp_over_limit.add(dataset)
            append_generic_csv(GP_SKIPPED_PATH, dict(
                dataset=dataset, n_train=row["n_train"], split_index=split,
                train_time_s=row["train_time_s"], limit_s=GP_TIME_LIMIT_S,
                reason="exact GP exceeded the per-split time limit; remaining splits skipped, "
                       "cell reported as an em dash with a footnote (D5)",
                timestamp=now_iso(), git_commit=git_commit_short(),
            ))
            print(f"    GP over the {GP_TIME_LIMIT_S / 60:.0f}-minute limit on {dataset} "
                  f"({row['train_time_s'] / 60:.1f} min) — abandoning its remaining splits", flush=True)


def build_literature_comparison(datasets=None) -> pd.DataFrame:
    """P13 as a PAIRED comparison (brief section 11).

    We use the same split index files as Gal, so split `i` is literally the
    same test rows for both — the comparison is per split, not two means
    with error bars, which removes the (large, on these datasets) between-
    split variance from the test.

    `test_MC_rmse_*`, not `test_rmse_*`: the latter is a single
    deterministic forward pass and does not reproduce the published table
    (see docs/datasets.md). BBB and deep-ensemble reference rows are the
    author's to fill in by hand from the papers — this function only emits
    the MC-dropout rows it can source from files.
    """
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} does not exist yet")
    own = pd.read_csv(RESULTS_PATH)
    own = own[own.method == "mcd"]
    if datasets:
        own = own[own.dataset.isin(datasets)]

    published_files = {
        "rmse": "test_MC_rmse_100_xepochs_1_hidden_layers.txt",
        "ll": "test_ll_100_xepochs_1_hidden_layers.txt",
    }
    root = Path(__file__).resolve().parent.parent / "data" / "uci_splits"

    rows = []
    for dataset, sub in own.groupby("dataset"):
        for metric, filename in published_files.items():
            path = root / dataset / "results" / filename
            if not path.exists():
                continue
            published = np.loadtxt(path).ravel()
            for r in sub.itertuples():
                i = int(r.split_index)
                if i >= len(published):
                    continue
                own_value = float(getattr(r, metric))
                rows.append(dict(
                    dataset=dataset, method="mcd", metric=metric, split_index=i,
                    own_value=own_value, published_value=float(published[i]),
                    difference=own_value - float(published[i]),
                ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["dataset", "metric", "split_index"]).reset_index(drop=True)
        df.to_csv(LITERATURE_PATH, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=",".join(DATASET_ORDER))
    parser.add_argument("--methods", type=str, default=",".join(METHOD_ORDER))
    parser.add_argument("--splits", type=int, default=None, help="use only the first N splits")
    parser.add_argument("--quick", action="store_true", help=f"--splits {QUICK_SPLITS}")
    parser.add_argument("--probe", action="store_true",
                        help="run only the timing probe cells (split 0) and stop")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes for the NN methods (GP always runs sequentially). "
                             ">1 invalidates train_time_s — use --timing-pass for the cost column.")
    parser.add_argument("--timing-pass", action="store_true",
                        help="split 0 only, sequential, one thread: produces results/e2_cost.csv")
    args = parser.parse_args()

    pin_threads()
    epochs_by_dataset = _epochs_by_dataset()

    if args.probe:
        print("timing probe (split 0):")
        done = _done_cells()
        for dataset, method_name in PROBE:
            if (dataset, method_name, 0) in done:
                print(f"  {method_name} / {dataset}: already in {RESULTS_PATH.name}, skipping")
                continue
            row = run_cell(dataset, method_name, 0, epochs_by_dataset[dataset])
            print(f"  {method_name:8s} {dataset:17s} "
                  f"train={row['train_time_s'] / 60:6.2f} min  "
                  f"predict={row['predict_time_ms_per_1k']:8.1f} ms/1k  "
                  f"rmse={row['rmse']:9.4f} ll={row['ll']:9.4f}", flush=True)
        comparison = build_literature_comparison(datasets=[d for d, m in PROBE if m == "mcd"])
        if not comparison.empty:
            print("\nP13, this split only (own - published; same test rows by construction):")
            for _, r in comparison.iterrows():
                print(f"  {r['dataset']:17s} {r['metric']:5s} split={int(r['split_index'])}  "
                      f"own={r['own_value']:9.4f}  published={r['published_value']:9.4f}  "
                      f"diff={r['difference']:+9.4f}")
        return

    datasets = [d for d in args.datasets.split(",")]
    methods = [m for m in args.methods.split(",")]
    n_splits = QUICK_SPLITS if args.quick else args.splits
    if args.timing_pass:
        timing_pass(datasets, methods, epochs_by_dataset)
        return
    if args.workers > 1:
        run_parallel(datasets, methods, n_splits, epochs_by_dataset, args.workers)
    else:
        run(datasets, methods, n_splits, epochs_by_dataset)

    comparison = build_literature_comparison()
    if not comparison.empty:
        print(f"\nwrote {LITERATURE_PATH} ({len(comparison)} paired rows)")
        summary = comparison.groupby(["dataset", "metric"])["difference"].agg(
            ["mean", "sem", "count"]
        )
        summary["same_sign_frac"] = comparison.groupby(["dataset", "metric"])["difference"].apply(
            lambda s: max((s > 0).mean(), (s < 0).mean())
        )
        print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
