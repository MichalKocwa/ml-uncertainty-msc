"""How much of E2 depends on the thread count rather than on the seed?

`docs/chapter4_notes.md` promises every result is reproducible from its
seed. That is false as written: `torch`'s intra-op thread count changes the
order of float64 reductions, and over tens of thousands of optimiser steps
the ~1e-15 per-operation difference amplifies. Measured on `map` / `kin8nm`
/ 1000 epochs / seed 0: max |parameter delta| 0.124 between one thread and
eight, RMSE 0.0732589 against 0.0732018.

E2 was produced by a mix of both settings — `compute_cell` (the parallel
worker) pins one thread, while the sequential path and `--probe` do not and
therefore inherit the default (8 on this machine). This script measures what
that mix cost: it recomputes the same cells at BOTH thread counts and
compares each against the value stored in `results/e2_uci.csv`, so the
comparison also identifies which setting produced each stored row.

Three datasets spanning the range of optimiser-step counts, which is what
the amplification depends on: `yacht` (277 rows, 3 steps/epoch), `concrete`
(927, 8) and `kin8nm` (7373, 58). GP is excluded — it is sklearn, so its
threading is controlled by the BLAS environment rather than by
`torch.set_num_threads`, and it is a separate question.

Writes results/thread_determinism_check.csv.

Usage:
  python experiments/thread_determinism_check.py --workers 8
  python experiments/thread_determinism_check.py --datasets yacht --methods map,mcd
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from experiments.e2_uci import _compute_cell, _epochs_by_dataset
from src.results import RESULTS_DIR, append_generic_csv, git_commit_short, now_iso
from src.seeding import TORCH_THREADS_ENV

DATASETS = ("yacht", "concrete", "kin8nm")
METHODS = ("map", "mcd", "laplace", "ensemble", "bbb")
THREAD_COUNTS = (1, 8)  # 8 is torch's default on this machine, i.e. what the sequential path used
SPLIT = 0
OUT_PATH = RESULTS_DIR / "thread_determinism_check.csv"
E2_PATH = RESULTS_DIR / "e2_uci.csv"


def _cell(args) -> dict:
    dataset, method, epochs, threads = args
    import os
    import torch
    # Through the environment switch, not `torch.set_num_threads` directly:
    # `set_seed` pins the count on every call, so a direct call here would be
    # overridden the moment the method builds its model.
    os.environ[TORCH_THREADS_ENV] = str(threads)
    torch.set_num_threads(threads)
    row, _ = _compute_cell(dataset, method, SPLIT, epochs)
    return dict(dataset=dataset, method=method, split_index=SPLIT, epochs=epochs,
                torch_threads=threads, rmse=row["rmse"], ll=row["ll"],
                picp95=row["picp95"], mpiw95=row["mpiw95"],
                mean_var_epistemic=row["mean_var_epistemic"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
    parser.add_argument("--methods", type=str, default=",".join(METHODS))
    parser.add_argument("--threads", type=str, default=",".join(map(str, THREAD_COUNTS)))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    epochs_by_dataset = _epochs_by_dataset()
    cells = [
        (dataset, method, epochs_by_dataset[dataset], threads)
        for dataset in args.datasets.split(",")
        for method in args.methods.split(",")
        for threads in [int(t) for t in args.threads.split(",")]
    ]
    print(f"{len(cells)} cells on {args.workers} workers "
          f"(each process pins its own thread count, so concurrency does not affect the arithmetic)",
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(_cell, cells):
            rows.append(row)
            append_generic_csv(OUT_PATH, dict(row, timestamp=now_iso(), git_commit=git_commit_short()))
            print(f"  {row['method']:9s} {row['dataset']:9s} threads={row['torch_threads']}  "
                  f"rmse={row['rmse']:.8f} ll={row['ll']:.8f}", flush=True)

    df = pd.DataFrame(rows)
    stored = pd.read_csv(E2_PATH)
    stored = stored[stored.split_index == SPLIT].set_index(["dataset", "method"])

    print(f"\n{'dataset':10s} {'method':9s} {'RMSE @1':>12s} {'RMSE @8':>12s} {'|delta|':>10s} "
          f"{'stored':>12s} {'matches':>8s}")
    summary = []
    for (dataset, method), sub in df.groupby(["dataset", "method"]):
        by_threads = sub.set_index("torch_threads")
        if not set(by_threads.index) >= {1, 8}:
            continue
        r1, r8 = float(by_threads.loc[1, "rmse"]), float(by_threads.loc[8, "rmse"])
        l1, l8 = float(by_threads.loc[1, "ll"]), float(by_threads.loc[8, "ll"])
        stored_rmse = float(stored.loc[(dataset, method), "rmse"])
        stored_ll = float(stored.loc[(dataset, method), "ll"])
        # Which setting reproduces the stored row exactly? That identifies how
        # that row was produced, which the CSV itself does not record.
        matches = ("1 thread" if abs(r1 - stored_rmse) < 1e-12
                   else "8 threads" if abs(r8 - stored_rmse) < 1e-12 else "neither")
        print(f"{dataset:10s} {method:9s} {r1:12.8f} {r8:12.8f} {abs(r1 - r8):10.2e} "
              f"{stored_rmse:12.8f} {matches:>8s}")
        summary.append(dict(dataset=dataset, method=method,
                            rmse_delta=abs(r1 - r8), ll_delta=abs(l1 - l8),
                            rmse_vs_stored=min(abs(r1 - stored_rmse), abs(r8 - stored_rmse)),
                            ll_vs_stored=min(abs(l1 - stored_ll), abs(l8 - stored_ll)),
                            matches=matches))

    s = pd.DataFrame(summary)
    print(f"\nmax |RMSE(1 thread) - RMSE(8 threads)| = {s.rmse_delta.max():.3e} "
          f"({s.loc[s.rmse_delta.idxmax(), 'dataset']}/{s.loc[s.rmse_delta.idxmax(), 'method']})")
    print(f"max |LL(1 thread)   - LL(8 threads)|   = {s.ll_delta.max():.3e} "
          f"({s.loc[s.ll_delta.idxmax(), 'dataset']}/{s.loc[s.ll_delta.idxmax(), 'method']})")
    print("stored rows reproduced by: " + ", ".join(
        f"{k} x{v}" for k, v in s.matches.value_counts().items()))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
