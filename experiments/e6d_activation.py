"""E6d — ReLU against TanH for MC dropout (P4).

`gal2016` predicts that MC dropout's uncertainty grows without bound outside
the training range under ReLU, and stays bounded under TanH:
`epi_extrap_ratio[relu] >> epi_extrap_ratio[tanh]`. That is P4, and it is
`pending` in `results/expectations_check.csv` until this runs.

Why it needs its own run even though D7b compared the two activations: D7b
was a decision sweep (it is why `DEFAULT_ACTIVATION` is `tanh`) and it
reported coverage and band asymmetry, not the growth ratio P4 is stated in,
and it left no results CSV. Nothing here is transcribed from it.

The measured quantity is `epi_extrap_ratio` — mean epistemic sigma outside
the training range over mean inside it — computed on the same evaluation grid
and the same regions as E1, so the numbers sit next to `e1_synthetic.csv`
rather than in a scale of their own. `map` is included as the reference that
has no epistemic term at all, and the other four methods because the same
grid answers the question for them at no extra cost — P4 is about MC dropout,
but "does this hold only for MC dropout" is worth one column.

Writes results/e6d_activation.csv.

Usage:
  python experiments/e6d_activation.py
  python experiments/e6d_activation.py --methods mcd --seeds 0
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
from src.methods import METHODS
from src.results import RESULTS_DIR, append_generic_csv, git_commit_short, now_iso
from src.seeding import set_seed

OUT_PATH = RESULTS_DIR / "e6d_activation.csv"
ACTIVATIONS = ("relu", "tanh")
DATASETS = ("sin_homo", "sin_gap")
METHOD_ORDER = ("map", "mcd", "bbb", "laplace", "ensemble")   # gp has no activation
SEEDS = (0, 1, 2)
TRAIN_RANGE = (0.0, 6.0)


def _build(method_name: str, activation: str):
    kwargs = dict(epochs=2000, activation=activation)
    if method_name == "bbb":
        kwargs["elbo_samples"] = 32
    return METHODS[method_name](**kwargs)


def _cell(args) -> dict:
    dataset, method_name, activation, seed = args
    import warnings
    warnings.filterwarnings("ignore")

    ds = SYNTHETIC_DATASETS[dataset](seed=seed)
    set_seed(seed)
    method = _build(method_name, activation)
    method.fit(ds.X_train, ds.y_train, seed=seed, use_cache=False)
    pred = method.predict(ds.X_eval)

    x = ds.X_eval.ravel()
    in_range = (x >= TRAIN_RANGE[0]) & (x <= TRAIN_RANGE[1])
    std_epi = np.sqrt(pred.var_epistemic)
    inside = float(np.mean(std_epi[in_range]))
    outside = float(np.mean(std_epi[~in_range]))
    return dict(
        experiment_id="e6d", dataset=dataset, method=method_name, activation=activation,
        seed=seed, n_parameters=method.n_parameters,
        mean_std_epi_in_range=inside, mean_std_epi_extrapolation=outside,
        # `map`'s epistemic term is zero by construction, so the ratio is 0/0 —
        # reported as NaN rather than forced to a number.
        epi_extrap_ratio=(outside / inside) if inside > 0 else float("nan"),
        # The far edge on its own: P4 is about growth "without bound", which a
        # mean over the whole outside region understates if it saturates.
        std_epi_at_edge=float(std_epi[np.argmax(x)]),
        mpiw95_in_range=metrics.mpiw(pred.std_total[in_range]),
        mpiw95_extrapolation=metrics.mpiw(pred.std_total[~in_range]),
        picp95_in_range=metrics.picp(ds.y_eval_noisy[in_range], pred.mean[in_range],
                                     pred.std_total[in_range]),
        picp95_extrapolation=metrics.picp(ds.y_eval_noisy[~in_range], pred.mean[~in_range],
                                          pred.std_total[~in_range]),
        rmse_in_range=metrics.rmse(ds.y_eval_noisy[in_range], pred.mean[in_range]),
        ll_in_range=metrics.ll(ds.y_eval_noisy[in_range], pred.mean[in_range],
                               pred.std_total[in_range]),
        timestamp=now_iso(), git_commit=git_commit_short(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
    parser.add_argument("--methods", type=str, default=",".join(METHOD_ORDER))
    parser.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    cells = [
        (dataset, method, activation, int(seed))
        for dataset in args.datasets.split(",")
        for method in args.methods.split(",")
        for activation in ACTIVATIONS
        for seed in args.seeds.split(",")
    ]
    print(f"{len(cells)} fits on {args.workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(_cell, cells):
            rows.append(row)
            append_generic_csv(OUT_PATH, row)
            print(f"  {row['dataset']:9s} {row['method']:9s} {row['activation']:5s} "
                  f"seed={row['seed']}  epi_extrap_ratio={row['epi_extrap_ratio']:8.3f}  "
                  f"std_epi_at_edge={row['std_epi_at_edge']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    print("\nepi_extrap_ratio, mean over seeds:")
    print(df.pivot_table(index="method", columns=["dataset", "activation"],
                         values="epi_extrap_ratio").round(3).to_string())
    print("\nP4 (relu >> tanh for MC dropout):")
    for dataset, sub in df[df.method == "mcd"].groupby("dataset"):
        by_activation = sub.groupby("activation").epi_extrap_ratio.agg(["mean", "sem"])
        relu, tanh = by_activation.loc["relu", "mean"], by_activation.loc["tanh", "mean"]
        print(f"  {dataset:9s} relu {relu:.3f} +/-{by_activation.loc['relu', 'sem']:.3f}, "
              f"tanh {tanh:.3f} +/-{by_activation.loc['tanh', 'sem']:.3f} "
              f"(ratio {relu / tanh:.2f}x)")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
