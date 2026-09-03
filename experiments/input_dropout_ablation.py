"""Input dropout on against off, for MC dropout (D15).

D15 removed dropout from the input layer, departing from a literal reading of
gal2016 ("dropout before every weighted layer"). The reason is a measurement,
not a preference: at `d=1` input dropout zeroes the network's *only* feature in
`dropout_p` of forward passes, at training and prediction time alike
(`AlwaysOnDropout` samples regardless of `model.train()`). During training that
corrupts a fraction of batches to `x=0`, and the network covers the resulting
loss spikes by inflating `log_sigma2` — so the aleatoric term absorbs an
artefact of the mask rather than the noise in the data.

That decision is recorded in `docs/chapter4_notes.md` D15 and in `MLP`'s
docstring, **but it never left a results CSV**. Chapter 4 has to state the
departure and justify it, and the rule of this repository is that a number in
the thesis comes from a file. This script re-runs the measurement and writes
one. Nothing here is transcribed from the notes; the comparison is recomputed.

**One deliberate departure from the E1 protocol, and the whole measurement
depends on it: `fixed_sigma2` is NOT applied.** E1 pins the noise variance of
`sin_homo` and `sin_gap` to the known truth (`src/sweeps.py:35`), which is what
makes its aleatoric/epistemic split interpretable. Here it must stay free: the
quantity under test is precisely whether the network inflates its own
`log_sigma2` to cover the dropped feature, and a fixed variance cannot inflate.
With `fixed_sigma2` applied, both arms would report `var_aleatoric = 0.01`
exactly and the question could not be asked.

The true variance is 0.01 on both datasets, so `mean_var_aleatoric` is readable
against a known target rather than only against the other arm.

Writes results/input_dropout_ablation.csv.

Usage:
  python experiments/input_dropout_ablation.py
  python experiments/input_dropout_ablation.py --datasets sin_homo --seeds 0
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

OUT_PATH = RESULTS_DIR / "input_dropout_ablation.csv"
DATASETS = ("sin_homo", "sin_gap")
SEEDS = (0, 1, 2)
EPOCHS = 2000            # the E1 protocol's epoch count for the synthetic scale
TRAIN_RANGE = (0.0, 6.0)

# Both datasets are generated with sigma = 0.1 (`_sigma_homo`), so the target
# the fitted variance should be reproducing is 0.1**2. Stated here rather than
# read from `KNOWN_HOMOSCEDASTIC_SIGMA`, because that constant is the value E1
# *pins* the model to and this run deliberately does not pin anything — reusing
# it would blur the distinction the docstring above draws.
TRUE_VAR_ALEATORIC = 0.1 ** 2


def _cell(args) -> dict:
    dataset, input_dropout, seed = args
    import warnings
    warnings.filterwarnings("ignore")

    ds = SYNTHETIC_DATASETS[dataset](seed=seed)
    set_seed(seed)
    method = METHODS["mcd"](epochs=EPOCHS, input_dropout=input_dropout)
    method.fit(ds.X_train, ds.y_train, seed=seed, use_cache=False)
    pred = method.predict(ds.X_eval)

    x = ds.X_eval.ravel()
    in_range = (x >= TRAIN_RANGE[0]) & (x <= TRAIN_RANGE[1])
    std_epi = np.sqrt(pred.var_epistemic)
    mean_var_alea = metrics.mean_var_aleatoric(pred.var_aleatoric)
    return dict(
        experiment_id="input_dropout_ablation", dataset=dataset, method="mcd",
        input_dropout=input_dropout, seed=seed, dropout_p=method.dropout_p,
        epochs=EPOCHS, n_parameters=method.n_parameters,
        # The headline: how far the fitted noise sits from the truth.
        mean_var_aleatoric=mean_var_alea,
        true_var_aleatoric=TRUE_VAR_ALEATORIC,
        var_aleatoric_ratio_to_true=mean_var_alea / TRUE_VAR_ALEATORIC,
        # Fit quality inside the training range, where the corrupted batches do
        # their damage. D15 reports both alongside the variance.
        rmse_in_range=metrics.rmse(ds.y_eval_noisy[in_range], pred.mean[in_range]),
        ll_in_range=metrics.ll(ds.y_eval_noisy[in_range], pred.mean[in_range],
                               pred.std_total[in_range]),
        mpiw95_in_range=metrics.mpiw(pred.std_total[in_range]),
        picp95_in_range=metrics.picp(ds.y_eval_noisy[in_range], pred.mean[in_range],
                                     pred.std_total[in_range]),
        # Reported so that "the bands got narrower" cannot be mistaken for "the
        # epistemic term collapsed": the two move for different reasons.
        mean_std_epi_in_range=float(np.mean(std_epi[in_range])),
        mean_std_epi_extrapolation=float(np.mean(std_epi[~in_range])),
        timestamp=now_iso(), git_commit=git_commit_short(),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
    parser.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cells = [
        (dataset, input_dropout, int(seed))
        for dataset in args.datasets.split(",")
        for input_dropout in (False, True)
        for seed in args.seeds.split(",")
    ]
    print(f"{len(cells)} fits on {args.workers} workers "
          f"(sigma^2 fitted, not pinned — see the module docstring)", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(_cell, cells):
            rows.append(row)
            append_generic_csv(OUT_PATH, row)
            print(f"  {row['dataset']:9s} input_dropout={str(row['input_dropout']):5s} "
                  f"seed={row['seed']}  var_alea={row['mean_var_aleatoric']:.4f} "
                  f"({row['var_aleatoric_ratio_to_true']:5.1f}x true)  "
                  f"rmse={row['rmse_in_range']:.4f}  mpiw={row['mpiw95_in_range']:.4f}",
                  flush=True)

    df = pd.DataFrame(rows)
    print("\nmean over seeds (true var_aleatoric = 0.01):")
    for column in ("mean_var_aleatoric", "rmse_in_range", "mpiw95_in_range"):
        table = df.pivot_table(index="dataset", columns="input_dropout", values=column)
        table.columns = [f"input_dropout={c}" for c in table.columns]
        print(f"\n  {column}")
        print(table.round(4).to_string().replace("\n", "\n  "))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
