"""P13 diagnosis: is the gap to Gal's numbers explained by an untuned `dropout_p`?

E2's MC-dropout rows differ from the published per-split values with the
SAME SIGN on essentially every split of every dataset (our RMSE higher, our
LL lower), which brief section 11 flags as a protocol error rather than a
method property. The size of the gap tracks `n_train` inversely:

    wine (1439)        +1.8%      concrete (927)   +17%
    power_plant (8611) +6.7%      energy (691)    +121%
    kin8nm (7373)      +18%       yacht (277)     +155%

A defect acting identically everywhere (standardisation, LL sign, wrong
index files) cannot produce that ordering. `dropout_p` can: Gal grid-searches
it (and `tau`) inside every one of the 20 folds, while we hold `p = 0.1`
fixed (D18) — a fixed value can be near-optimal where there is enough data
to wash the choice out and badly wrong where there is not.

This script tests that on the two worst datasets, at the epoch counts E2
actually uses, over all 20 splits. If some `p` brings `yacht` near Gal's
0.666 / -1.250, the hypothesis holds and P13 stays as planned: a separate
validation run reproducing Gal's full protocol (relu, 4000 epochs, grid
search per fold) on the smallest datasets, with the main table left on our
own protocol as a documented difference.

**`dropout_p` is NOT changed anywhere by this script.** It sweeps a local
value and reports; D18's `p = 0.1` stays the default (author's instruction).

Writes results/p13_dropout_diagnostic.csv.

Usage:
  python experiments/p13_dropout_diagnostic.py
  python experiments/p13_dropout_diagnostic.py --datasets energy --dropout 0.005,0.01,0.05
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import metrics
from src.data import load_uci
from src.methods.mcd import MCDropoutMethod
from src.results import RESULTS_DIR, append_generic_csv, git_commit_short, now_iso
from src.seeding import set_seed

DEFAULT_DROPOUT = (0.005, 0.01, 0.05, 0.1, 0.2)
DEFAULT_DATASETS = ("yacht",)
N_SPLITS = 20
EPOCHS_TABLE = "uci_epochs_sweep_final_combined.csv"
OUT_PATH = RESULTS_DIR / "p13_dropout_diagnostic.csv"

# Gal's published means, read from his per-split files at run time rather
# than typed in here (docs/datasets.md: the README's rounded cells are not
# all in the same convention, so only the 20-line files are trusted).
PUBLISHED = {
    "rmse": "test_MC_rmse_100_xepochs_1_hidden_layers.txt",
    "ll": "test_ll_100_xepochs_1_hidden_layers.txt",
}


def _published(dataset: str) -> dict:
    root = Path(__file__).resolve().parent.parent / "data" / "uci_splits" / dataset / "results"
    return {k: np.loadtxt(root / v).ravel() for k, v in PUBLISHED.items() if (root / v).exists()}


def _cell(args) -> dict:
    dataset, dropout_p, split, epochs = args
    import torch
    torch.set_num_threads(1)

    ds = load_uci(dataset, split=split)
    set_seed(split)
    method = MCDropoutMethod(epochs=epochs, dropout_p=dropout_p)
    method.fit(ds.X_train, ds.y_train, seed=split, use_cache=False)
    pred = method.predict(ds.X_test)

    scale = float(ds.y_scaler.scale_[0])
    mean = ds.y_scaler.inverse_transform(pred.mean.reshape(-1, 1)).ravel()
    y_test = ds.y_scaler.inverse_transform(ds.y_test.reshape(-1, 1)).ravel()
    std = pred.std_total * scale
    return dict(
        dataset=dataset, dropout_p=dropout_p, split_index=split, epochs=epochs,
        rmse=metrics.rmse(y_test, mean), ll=metrics.ll(y_test, mean, std),
        picp95=metrics.picp(y_test, mean, std), mpiw95=metrics.mpiw(std),
        mean_var_aleatoric=float(np.mean(pred.var_aleatoric)) * scale ** 2,
        mean_var_epistemic=float(np.mean(pred.var_epistemic)) * scale ** 2,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--dropout", type=str, default=",".join(map(str, DEFAULT_DROPOUT)))
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    dropouts = [float(v) for v in args.dropout.split(",")]
    epochs_by_dataset = {
        r.dataset: int(r.epochs)
        for r in pd.read_csv(RESULTS_DIR / EPOCHS_TABLE).itertuples()
    }

    cells = [
        (dataset, p, split, epochs_by_dataset[dataset])
        for dataset in datasets for p in dropouts for split in range(args.splits)
    ]
    print(f"{len(cells)} cells on {args.workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(_cell, cells):
            rows.append(row)
            append_generic_csv(OUT_PATH, dict(row, timestamp=now_iso(), git_commit=git_commit_short()))
    df = pd.DataFrame(rows)

    for dataset, sub in df.groupby("dataset"):
        published = _published(dataset)
        print(f"\n{dataset} (epochs={epochs_by_dataset[dataset]}, {args.splits} splits, "
              f"mean +/- standard error over splits)")
        print(f"  {'p':>6s} {'RMSE':>16s} {'LL':>16s} {'PICP95':>7s} {'MPIW95':>8s}")
        for p, g in sub.groupby("dropout_p"):
            print(f"  {p:6.3f} {g.rmse.mean():8.4f} +/-{g.rmse.sem():5.4f} "
                  f"{g.ll.mean():8.4f} +/-{g.ll.sem():5.4f} "
                  f"{g.picp95.mean():7.3f} {g.mpiw95.mean():8.4f}")
        if published:
            print(f"  {'Gal':>6s} {published['rmse'].mean():8.4f}        "
                  f"{published['ll'].mean():8.4f}")
            best_rmse = sub.groupby("dropout_p").rmse.mean().idxmin()
            best_ll = sub.groupby("dropout_p").ll.mean().idxmax()
            print(f"  best RMSE at p={best_rmse}, best LL at p={best_ll}; "
                  f"E2 uses p=0.1 (D18)")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
