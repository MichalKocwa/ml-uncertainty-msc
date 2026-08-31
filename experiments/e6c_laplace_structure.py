"""E6c — Laplace's covariance structure and prior precision (Figure 3.9, P5, P6).

Two predictions from the literature rest on this ablation and both are
`pending` in `results/expectations_check.csv` until it runs:

  * **P5** (`ritter2018`): a full Laplace with no regularisation overestimates
    uncertainty when the parameter count is comparable to `N`
    — `mpiw95[unregularised] >> mpiw95[tuned]`.
  * **P6** (`ritter2018`): the diagonal approximation behaves like dropout,
    while KFAC gives higher uncertainty out of distribution
    — `epi_extrap_ratio[kron] > epi_extrap_ratio[diag]`.

Grid: `hessian_structure ∈ {full, kron, diag}` x `prior_precision_mode ∈
{fixed, marglik, unregularised}`, on `sin_homo` and `sin_gap`, three seeds.
`fixed` is E1/E2's default (`prior_precision = 1/gamma^2`, the same prior as
every other method, D11); `marglik` is the tuned variant P7 speaks of;
`unregularised` is P5's.

Metrics are split by region, as in E1: in-range against extrapolation, plus
`epi_extrap_ratio` — the mean epistemic sigma outside the training range over
the mean inside it, which is the quantity both predictions are about. A single
number over the whole evaluation grid would be dominated by extrapolation and
would answer neither.

Writes results/e6c_laplace_structure.csv.

Usage:
  python experiments/e6c_laplace_structure.py --workers 6
  python experiments/e6c_laplace_structure.py --quick
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import metrics
from src.data import SYNTHETIC_DATASETS
from src.methods.laplace import LaplaceMethod
from src.results import RESULTS_DIR, append_generic_csv, git_commit_short, now_iso
from src.seeding import set_seed

OUT_PATH = RESULTS_DIR / "e6c_laplace_structure.csv"
STRUCTURES = ("full", "kron", "diag")
PRIOR_MODES = ("fixed", "marglik", "unregularised")
DATASETS = ("sin_homo", "sin_gap")
SEEDS = (0, 1, 2)
TRAIN_RANGE = (0.0, 6.0)   # brief 5.1: the synthetic datasets train on [0, 6]


def _cell(args) -> dict:
    dataset, structure, prior_mode, seed = args
    import warnings
    warnings.filterwarnings("ignore")

    ds = SYNTHETIC_DATASETS[dataset](seed=seed)
    set_seed(seed)
    method = LaplaceMethod(epochs=2000, hessian_structure=structure,
                           prior_precision_mode=prior_mode)
    # A cell that cannot be computed is recorded, not skipped and not crashed
    # on. `prior_precision_mode="unregularised"` is expected to be fragile —
    # that is what P5 is about — and "the posterior precision does not
    # factorise" is a result about the variant, not a failure of the run. The
    # alternative (raising the prior precision until it works) would answer a
    # different question.
    try:
        method.fit(ds.X_train, ds.y_train, seed=seed, use_cache=False)
        pred = method.predict(ds.X_eval)
    except Exception as error:  # noqa: BLE001 — the point is to record whatever it was
        return dict(
            experiment_id="e6c", dataset=dataset, hessian_structure=structure,
            prior_precision_mode=prior_mode, seed=seed, status="failed",
            failure=f"{type(error).__name__}: {str(error).splitlines()[0][:200]}",
            timestamp=now_iso(), git_commit=git_commit_short(),
        )

    x = ds.X_eval.ravel()
    in_range = (x >= TRAIN_RANGE[0]) & (x <= TRAIN_RANGE[1])
    std_epi = np.sqrt(pred.var_epistemic)
    row = dict(
        experiment_id="e6c", dataset=dataset, hessian_structure=structure,
        prior_precision_mode=prior_mode, seed=seed, status="ok", failure="",
        prior_precision=float(getattr(method, "prior_precision_", np.nan)),
        n_parameters=method.n_parameters,
        mpiw95=metrics.mpiw(pred.std_total),
        mpiw95_in_range=metrics.mpiw(pred.std_total[in_range]),
        mpiw95_extrapolation=metrics.mpiw(pred.std_total[~in_range]),
        picp95_in_range=metrics.picp(ds.y_eval_noisy[in_range], pred.mean[in_range],
                                     pred.std_total[in_range]),
        picp95_extrapolation=metrics.picp(ds.y_eval_noisy[~in_range], pred.mean[~in_range],
                                          pred.std_total[~in_range]),
        rmse_in_range=metrics.rmse(ds.y_eval_noisy[in_range], pred.mean[in_range]),
        ll_in_range=metrics.ll(ds.y_eval_noisy[in_range], pred.mean[in_range],
                               pred.std_total[in_range]),
        mean_std_epi_in_range=float(np.mean(std_epi[in_range])),
        mean_std_epi_extrapolation=float(np.mean(std_epi[~in_range])),
        # P6's quantity: how much the epistemic term grows off the training range.
        epi_extrap_ratio=float(np.mean(std_epi[~in_range]) / np.mean(std_epi[in_range])),
        timestamp=now_iso(), git_commit=git_commit_short(),
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
    parser.add_argument("--structures", type=str, default=",".join(STRUCTURES))
    parser.add_argument("--prior-modes", type=str, default=",".join(PRIOR_MODES))
    parser.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)))
    parser.add_argument("--quick", action="store_true", help="one dataset, one seed")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    datasets = ["sin_homo"] if args.quick else args.datasets.split(",")
    seeds = [0] if args.quick else [int(s) for s in args.seeds.split(",")]
    cells = [
        (dataset, structure, prior_mode, seed)
        for dataset in datasets
        for structure in args.structures.split(",")
        for prior_mode in args.prior_modes.split(",")
        for seed in seeds
    ]
    print(f"{len(cells)} Laplace fits on {args.workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(_cell, cells):
            rows.append(row)
            append_generic_csv(OUT_PATH, row)
            if row["status"] == "ok":
                print(f"  {row['dataset']:9s} {row['hessian_structure']:5s} "
                      f"{row['prior_precision_mode']:14s} seed={row['seed']}  "
                      f"mpiw95={row['mpiw95']:8.4f}  epi_extrap_ratio={row['epi_extrap_ratio']:8.3f}",
                      flush=True)
            else:
                print(f"  {row['dataset']:9s} {row['hessian_structure']:5s} "
                      f"{row['prior_precision_mode']:14s} seed={row['seed']}  "
                      f"FAILED: {row['failure']}", flush=True)

    df = pd.DataFrame(rows)
    failed = df[df.status == "failed"]
    if not failed.empty:
        print(f"\n{len(failed)}/{len(df)} cells did not produce a posterior:")
        for (structure, mode), g in failed.groupby(["hessian_structure", "prior_precision_mode"]):
            print(f"  {structure:5s} / {mode:14s}: {len(g)} of "
                  f"{len(df[(df.hessian_structure == structure) & (df.prior_precision_mode == mode)])} "
                  f"— {g.failure.iloc[0]}")
    df = df[df.status == "ok"]
    if df.empty:
        print("\nno cell produced a posterior; nothing to summarise")
        print(f"\nwrote {OUT_PATH}")
        return
    for dataset, sub in df.groupby("dataset"):
        print(f"\n{dataset}: mean over seeds")
        print(sub.pivot_table(index="hessian_structure", columns="prior_precision_mode",
                              values=["mpiw95", "epi_extrap_ratio", "picp95_in_range"])
              .round(3).to_string())

    print("\nP5 (unregularised full Laplace overestimates): "
          "mpiw95[full/unregularised] vs mpiw95[full/fixed]")
    for dataset, sub in df.groupby("dataset"):
        full = sub[sub.hessian_structure == "full"].set_index("prior_precision_mode").mpiw95
        if {"unregularised", "fixed"} <= set(full.index):
            print(f"  {dataset:9s} {full['unregularised']:.4f} vs {full['fixed']:.4f} "
                  f"(ratio {full['unregularised'] / full['fixed']:.2f}x)")

    print("\nP6 (kron > diag out of distribution): epi_extrap_ratio at the shared prior (fixed)")
    for dataset, sub in df.groupby("dataset"):
        fixed = sub[sub.prior_precision_mode == "fixed"].groupby("hessian_structure").epi_extrap_ratio.mean()
        print(f"  {dataset:9s} " + ", ".join(f"{k} {v:.3f}" for k, v in fixed.items()))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
