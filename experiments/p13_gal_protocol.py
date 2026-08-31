"""P13: does our MC dropout reproduce gal2016's published UCI numbers?

E2's `mcd` rows sit above the published RMSE and below the published LL on
essentially every split of every dataset, by a margin that grows as the
training set shrinks (`wine` +1.8%, `power_plant` +6.7%, `kin8nm` +18%,
`concrete` +17%, `energy` +121%, `yacht` +155%). `p13_dropout_diagnostic.py`
showed `dropout_p` accounts for most of it — at `p = 0.005` rather than our
fixed 0.1, `yacht`'s RMSE gap falls from +155% to +43% and `energy`'s from
+121% to +9% — which points at the one protocol difference that matters:
gal2016 grid-searches his hyperparameters inside every fold, we hold one
value for the whole table (D18).

This run equalises the protocol instead of arguing about it. Everything
below is gal2016's `experiment.py` / `net/net.py` at the commit our data
comes from (`6eb4497`), reproduced here on OUR MC dropout implementation:

  * relu, 1 hidden layer of `n_hidden.txt` (50) units, dropout before the
    first Dense AND before the output (`input_dropout=True`);
  * `40 * 100 = 4000` epochs, batch 128, Adam at Keras's default `lr=1e-3`;
  * grid over `dropout_rates.txt` x `tau_values.txt` — BOTH, read from the
    pinned commit's own files, never typed in here (the tau grid is not the
    same on every dataset: 0.25/0.5/0.75 on `yacht` and `energy`, an order
    of magnitude smaller on `concrete`);
  * selection by validation log-likelihood on the LAST 20% of the training
    rows in file order, then a refit on the whole training split with the
    winning pair;
  * `tau` as the noise model — his network has no learned variance, so
    ours is fixed to `sigma^2 = 1/(tau * scale^2)` (standardised units) and
    his L2 coefficient `reg = lengthscale^2 (1-p) / (2 N tau)` is converted
    to our penalty as `reg / (2 sigma^2)` on weights and 0 on biases. That
    conversion is what `tests/test_p13_gal_protocol.py` verifies against
    his objective's gradients, rather than being asserted here;
  * `T = 10000` stochastic passes, MC RMSE on their mean, and his own
    log-likelihood estimator (a `logsumexp` mixture at precision `tau`, in
    ORIGINAL units) — not our moment-matched Gaussian LL.

Three datasets: `yacht`, `energy` (the two largest gaps) and `concrete`
(+17%, the intermediate point — with only the two extremes a closed gap
cannot distinguish "the protocol explains it" from "the protocol explains
it only on the smallest sets").

What this run is NOT: it does not change any default. `dropout_p`, the
activation, the epoch counts and the noise model of the main table are
untouched; E2 stays on our own protocol and the difference between the two
is a documented result, not a defect.

Writes:
  results/p13_gal_protocol.csv       — one row per (dataset, split): chosen p and tau, test RMSE/LL, published values
  results/p13_gal_protocol_grid.csv  — every (p, tau) cell's validation score, for the selection to be auditable

Usage:
  python experiments/p13_gal_protocol.py --workers 8
  python experiments/p13_gal_protocol.py --datasets yacht --splits 2 --epochs 200   # smoke test
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from src.data import load_uci_raw, n_uci_splits, uci_split_indices
from src.methods.backbone import DEFAULT_BATCH_SIZE, DEFAULT_GAMMA
from src.methods.mcd import MCDropoutMethod
from src.results import RESULTS_DIR, append_generic_csv, git_commit_short, now_iso
from src.seeding import set_seed

DATASETS = ("yacht", "energy", "concrete")
UCI_ROOT = Path(__file__).resolve().parent.parent / "data" / "uci_splits"
OUT_PATH = RESULTS_DIR / "p13_gal_protocol.csv"
GRID_PATH = RESULTS_DIR / "p13_gal_protocol_grid.csv"

# gal2016 `net.py` / `experiment.py`, all four fixed by his code rather than chosen here.
LENGTHSCALE = 1e-2
EPOCHS_MULTIPLIER = 100      # --epochx of the published run; 40 * 100 = 4000 epochs
T_SAMPLES = 10000
LEARNING_RATE = 1e-3         # Keras Adam default
VALIDATION_FRACTION = 0.2

PUBLISHED = {
    "rmse": "test_MC_rmse_100_xepochs_1_hidden_layers.txt",
    "ll": "test_ll_100_xepochs_1_hidden_layers.txt",
}


def _read_grid(dataset: str):
    """`(dropout_rates, tau_values)` from the pinned commit's own files."""
    root = UCI_ROOT / dataset
    rates = np.atleast_1d(np.loadtxt(root / "dropout_rates.txt")).tolist()
    taus = np.atleast_1d(np.loadtxt(root / "tau_values.txt")).tolist()
    return rates, taus


def _protocol_constants(dataset: str):
    root = UCI_ROOT / dataset
    n_epochs = int(np.loadtxt(root / "n_epochs.txt")) * EPOCHS_MULTIPLIER
    n_hidden = int(np.loadtxt(root / "n_hidden.txt"))
    return n_epochs, n_hidden


def _published(dataset: str) -> dict:
    root = UCI_ROOT / dataset / "results"
    return {k: np.loadtxt(root / v).ravel() for k, v in PUBLISHED.items()}


def _standardise(X_train, y_train, X_eval):
    """gal2016's `net.py` normalisation, including its zero-variance guard
    (`std[std == 0] = 1`), fitted on whatever training set it is handed —
    which during grid search is the 80% sub-train, not the full split."""
    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
    x_std = np.where(x_std == 0, 1.0, x_std)
    y_mean, y_std = float(y_train.mean()), float(y_train.std())
    return (
        (X_train - x_mean) / x_std,
        (y_train - y_mean) / y_std,
        (X_eval - x_mean) / x_std,
        y_mean, y_std,
    )


def _penalty_coefficients(dropout_p: float, n: int, tau: float, sigma2_standardised: float) -> dict:
    """gal2016's `reg`, expressed in our loss's units.

    His Keras objective is `mean((y - yhat)^2) + reg * sum(W^2)` on WEIGHTS
    only; ours is a Gaussian NLL at fixed `sigma^2`, which is his data term
    scaled by `1/(2 sigma^2)`. Scaling the penalty by the same factor makes
    the two objectives proportional, and Adam is invariant to a constant
    factor on the loss — see `tests/test_p13_gal_protocol.py`.
    """
    reg = LENGTHSCALE ** 2 * (1.0 - dropout_p) / (2.0 * n * tau)
    coefficient = reg / (2.0 * sigma2_standardised)
    return {
        "linear1.weight": coefficient, "linear1.bias": 0.0,
        "mean_head.weight": coefficient, "mean_head.bias": 0.0,
    }


def _gal_log_likelihood(y_true, samples, tau: float) -> float:
    """gal2016's test LL: a `T`-component Gaussian mixture at precision `tau`.

        ll = logsumexp_t(-0.5 tau (y - yhat_t)^2) - log T - 0.5 log(2 pi) + 0.5 log tau

    `samples` is `(T, n)` in ORIGINAL target units, as is `y_true`. This is
    not the moment-matched Gaussian LL the rest of the project reports —
    P13 compares against his published column, so it has to be his estimator.
    """
    residual = y_true[None, :] - samples
    per_point = (
        logsumexp(-0.5 * tau * residual ** 2, axis=0)
        - np.log(samples.shape[0])
        - 0.5 * np.log(2 * np.pi)
        + 0.5 * np.log(tau)
    )
    return float(np.mean(per_point))


def _fit_and_score(X_train, y_train, X_eval, y_eval, dropout_p, tau, epochs, hidden, seed, T):
    """One (p, tau) cell: fit on `X_train`, score on `X_eval` his way."""
    X_tr, y_tr, X_ev, y_mean, y_std = _standardise(X_train, y_train, X_eval)
    sigma2 = 1.0 / (tau * y_std ** 2)

    set_seed(seed)
    method = MCDropoutMethod(
        hidden=hidden, dropout_p=dropout_p, T=T, epochs=epochs, lr=LEARNING_RATE,
        batch_size=DEFAULT_BATCH_SIZE, activation="relu", input_dropout=True,
        fixed_sigma2=sigma2, gamma=DEFAULT_GAMMA,  # gamma unused: the penalty override replaces it
        penalty_override=_penalty_coefficients(dropout_p, len(X_tr), tau, sigma2),
    )
    method.fit(X_tr, y_tr, seed=seed, use_cache=False)
    pred = method.predict(X_ev)

    samples = pred.samples * y_std + y_mean          # (T, n), original units
    mc_mean = samples.mean(axis=0)
    return dict(
        rmse=float(np.sqrt(np.mean((y_eval - mc_mean) ** 2))),
        ll=_gal_log_likelihood(y_eval, samples, tau),
    )


def run_fold(args) -> tuple:
    """One split: grid search on the validation slice, refit, score on test."""
    dataset, split, epochs_override, t_samples = args
    import torch
    torch.set_num_threads(1)

    X, y = load_uci_raw(dataset)
    train_idx, test_idx = uci_split_indices(dataset, split)
    X_train_full, y_train_full = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # gal2016 takes the LAST 20% of the training rows, in file order, as
    # validation — not a random draw, so this is reproducible from the split
    # files alone.
    n_sub = int((1.0 - VALIDATION_FRACTION) * len(X_train_full))
    X_sub, y_sub = X_train_full[:n_sub], y_train_full[:n_sub]
    X_val, y_val = X_train_full[n_sub:], y_train_full[n_sub:]

    epochs, hidden = _protocol_constants(dataset)
    if epochs_override:
        epochs = epochs_override
    rates, taus = _read_grid(dataset)

    grid_rows, best = [], None
    for dropout_p in rates:
        for tau in taus:
            scored = _fit_and_score(X_sub, y_sub, X_val, y_val, dropout_p, tau,
                                    epochs, hidden, split, t_samples)
            grid_rows.append(dict(
                dataset=dataset, split_index=split, dropout_p=dropout_p, tau=tau,
                n_train=len(X_sub), n_validation=len(X_val), epochs=epochs,
                validation_rmse=scored["rmse"], validation_ll=scored["ll"],
            ))
            # `>`, so the first of equal scores wins — gal2016's own tie-break.
            if best is None or scored["ll"] > best["validation_ll"]:
                best = dict(dropout_p=dropout_p, tau=tau, validation_ll=scored["ll"],
                            validation_rmse=scored["rmse"])

    final = _fit_and_score(X_train_full, y_train_full, X_test, y_test,
                           best["dropout_p"], best["tau"], epochs, hidden, split, t_samples)
    published = _published(dataset)
    row = dict(
        dataset=dataset, split_index=split, epochs=epochs, hidden=hidden,
        n_train=len(X_train_full), n_validation=len(X_val), n_test=len(X_test),
        chosen_dropout_p=best["dropout_p"], chosen_tau=best["tau"],
        validation_ll=best["validation_ll"], validation_rmse=best["validation_rmse"],
        rmse=final["rmse"], ll=final["ll"],
        published_rmse=float(published["rmse"][split]), published_ll=float(published["ll"][split]),
        rmse_difference=final["rmse"] - float(published["rmse"][split]),
        ll_difference=final["ll"] - float(published["ll"][split]),
        t_samples=t_samples,
    )
    return row, grid_rows


def _summarise(df: pd.DataFrame) -> None:
    for dataset, sub in df.groupby("dataset"):
        print(f"\n{dataset} ({len(sub)} splits, epochs={int(sub.epochs.iloc[0])}, "
              f"n_train={int(sub.n_train.iloc[0])})")
        print(f"  {'':14s} {'RMSE':>18s} {'LL':>18s}")
        print(f"  {'ours (Gal protocol)':14s} {sub.rmse.mean():9.4f} +/-{sub.rmse.sem():6.4f} "
              f"{sub.ll.mean():9.4f} +/-{sub.ll.sem():6.4f}")
        print(f"  {'published':14s} {sub.published_rmse.mean():9.4f} +/-{sub.published_rmse.sem():6.4f} "
              f"{sub.published_ll.mean():9.4f} +/-{sub.published_ll.sem():6.4f}")
        # Paired: the split files make split i literally the same test rows.
        print(f"  {'paired diff':14s} {sub.rmse_difference.mean():+9.4f} +/-{sub.rmse_difference.sem():6.4f} "
              f"{sub.ll_difference.mean():+9.4f} +/-{sub.ll_difference.sem():6.4f}")
        print(f"  {'same sign':14s} {max((sub.rmse_difference > 0).mean(), (sub.rmse_difference < 0).mean()):9.2f} "
              f"{'':7s} {max((sub.ll_difference > 0).mean(), (sub.ll_difference < 0).mean()):9.2f}")
        relative = sub.rmse_difference.mean() / sub.published_rmse.mean() * 100
        print(f"  RMSE gap: {relative:+.1f}% of the published mean")
        print("  chosen p:   " + ", ".join(
            f"{p}x{int(c)}" for p, c in sub.chosen_dropout_p.value_counts().sort_index().items()))
        print("  chosen tau: " + ", ".join(
            f"{t}x{int(c)}" for t, c in sub.chosen_tau.value_counts().sort_index().items()))
        print("  per fold (split: p, tau): " + ", ".join(
            f"{int(r.split_index)}: {r.chosen_dropout_p}/{r.chosen_tau}" for r in sub.itertuples()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
    parser.add_argument("--splits", type=int, default=None, help="use only the first N splits")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the protocol's 4000 epochs (smoke tests only)")
    parser.add_argument("--samples", type=int, default=T_SAMPLES,
                        help=f"stochastic passes at predict time (protocol: {T_SAMPLES})")
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    folds = [
        (dataset, split, args.epochs, args.samples)
        for dataset in datasets
        for split in range(args.splits or n_uci_splits(dataset))
    ]
    for dataset in datasets:
        rates, taus = _read_grid(dataset)
        print(f"{dataset}: grid {len(rates)} x {len(taus)} = {len(rates) * len(taus)} cells per fold "
              f"(p={rates}, tau={taus})")
    print(f"{len(folds)} folds on {args.workers} workers", flush=True)

    rows, grid_rows = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row, cells in pool.map(run_fold, folds):
            rows.append(row)
            grid_rows.extend(cells)
            append_generic_csv(OUT_PATH, dict(row, timestamp=now_iso(), git_commit=git_commit_short()))
            for cell in cells:
                append_generic_csv(GRID_PATH, dict(cell, timestamp=now_iso(), git_commit=git_commit_short()))
            print(f"  {row['dataset']:9s} split={row['split_index']:2d}  "
                  f"p={row['chosen_dropout_p']:<6} tau={row['chosen_tau']:<6} "
                  f"rmse={row['rmse']:8.4f} (published {row['published_rmse']:8.4f})  "
                  f"ll={row['ll']:8.4f} (published {row['published_ll']:8.4f})", flush=True)

    _summarise(pd.DataFrame(rows))
    print(f"\nwrote {OUT_PATH} and {GRID_PATH}")


if __name__ == "__main__":
    main()
