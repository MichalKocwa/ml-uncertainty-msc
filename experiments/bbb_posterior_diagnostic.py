"""Does BBB's variational posterior actually train, and how far does `elbo_samples` move it?

**Part 1 (author, 2026-08-28): does the posterior train at all on UCI?**
`frac_posterior_var_below_prior` reads 78-100% in every combination of
dataset, epoch count and `elbo_samples` measured at E2 scale
(`results/bbb_elbo_samples_cost.csv`). An untrained network scores exactly
100% (`posterior_rho_init = -3.0` gives `sigma_post = 0.0486`, a variance of
0.236% of a `gamma = 1` prior's), so that reading is consistent both with a
posterior that trains and settles low, and with one that never moves —
which would make BBB in the main table a deterministic network with noise.
The four quantities below separate those: the `sigma_post` distribution
against its initial value, the same for E1 as a reference, `std_epistemic`
from `predict()`, and the KL term against the data term.

**Part 2 (author, 2026-08-28): is D14d's `elbo_samples` argument
reproducible from current code?** D14d justified `elbo_samples=32` with
"43% of weights below 1% of prior at K=1, 0.66% at K=32" — i.e. with the
claim that this parameter controls whether the posterior trains at all,
not merely the gradient estimator's variance. Neither of the two plausible
threshold conventions reproduces 0.66% on the current E1 configuration, and
that configuration is not the one D14d measured (it predates D-sigma-E1, so
`sigma_o` was learned, not fixed). `--e1-sigma learned` with an
`--e1-elbo-samples` grid re-runs D14d's actual configuration from today's
code, on three seeds, reporting BOTH threshold conventions.

E1's fixed-sigma row is built through `src.sweeps.e1_method_kwargs`, the
same function `experiments/e1_synthetic.py` uses, so the reference is the
configuration that produced `results/e1_synthetic.csv` rather than a copy.

Writes results/bbb_posterior_diagnostic.csv (appended, not overwritten).

Usage:
  python experiments/bbb_posterior_diagnostic.py
  python experiments/bbb_posterior_diagnostic.py --datasets "" --e1-sigma learned \\
      --e1-elbo-samples 1,8,32 --e1-seeds 0,1,2
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from bayesian_torch.models.dnn_to_bnn import get_kl_loss

from src.data import SYNTHETIC_DATASETS, load_uci
from src.methods.backbone import DEFAULT_BATCH_SIZE, DTYPE, gaussian_nll
from src.methods.bbb import BBBMethod, frac_posterior_var_below_prior
from src.results import RESULTS_DIR, git_commit_short, now_iso, upsert_csv
from src.seeding import set_seed
from src.sweeps import e1_method_kwargs

ELBO_SAMPLES = 32  # the value decided for E2 (author, 2026-08-28); E1 already uses it
SEED = 0
SPLIT_INDEX = 0
E1_DATASET = "sin_homo"
E1_EPOCHS = 2000  # e1_synthetic.py's non-quick default

# softplus(posterior_rho_init) — every variational sigma starts here, so
# "did the posterior move" is a comparison against this number.
INIT_SIGMA = math.log1p(math.exp(-3.0))

DEFAULT_DATASETS = ("yacht", "concrete")
QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)

# Rows are keyed by these; re-measuring one replaces it rather than
# appending a duplicate.
KEYS = ["dataset", "protocol", "epochs", "elbo_samples", "seed"]


def _posterior_sigmas(model):
    """`(weight_sigmas, all_sigmas, prior_sigmas)` as flat numpy arrays."""
    layers = [model.mlp.linear1, *model.mlp.extra_hidden, model.mlp.mean_head]
    weights, biases, priors = [], [], []
    with torch.no_grad():
        for layer in layers:
            weights.append(torch.log1p(torch.exp(layer.rho_weight)).flatten().numpy())
            priors.append(layer.prior_weight_sigma.flatten().numpy())
            if getattr(layer, "rho_bias", None) is not None:
                biases.append(torch.log1p(torch.exp(layer.rho_bias)).flatten().numpy())
                priors.append(layer.prior_bias_sigma.flatten().numpy())
    w = np.concatenate(weights)
    return w, np.concatenate([w, *biases] if biases else [w]), np.concatenate(priors)


def _elbo_terms(model, X, y, n_samples, batch_size=DEFAULT_BATCH_SIZE):
    """Epoch-level data term and KL term, in the training loop's own units.

    `_train_bbb` minimises `sum_nll(batch) + KL/n_batches` per step, so over
    a full epoch the KL enters exactly once and the data term is the summed
    NLL over the training set — which is what these two numbers are. The
    data term is averaged over `n_samples` weight draws, because a single
    draw of it is a Monte Carlo estimate, not a fixed quantity.
    """
    X_t = torch.as_tensor(X, dtype=DTYPE)
    y_t = torch.as_tensor(y, dtype=DTYPE).reshape(-1, 1)
    n = X.shape[0]
    bs = min(batch_size, n)
    totals = []
    with torch.no_grad():
        for _ in range(n_samples):
            total = 0.0
            for start in range(0, n, bs):
                sl = slice(start, start + bs)
                mu, log_var = model(X_t[sl])
                total += float(gaussian_nll(mu, log_var, y_t[sl])) * (min(start + bs, n) - start)
            totals.append(total)
        kl = float(get_kl_loss(model.mlp))
    return float(np.mean(totals)), kl


def _row(label, protocol, epochs, elbo_samples, seed, model, pred, X_fit, y_fit):
    w, all_sigmas, priors = _posterior_sigmas(model)
    data_term, kl = _elbo_terms(model, X_fit, y_fit, n_samples=max(elbo_samples, 8))
    std_epi = np.sqrt(pred.var_epistemic)
    below = frac_posterior_var_below_prior(model)
    # Second convention, to settle which one D14d's "43% / 0.66%" used
    # (D14e-uwaga: the code that produced those numbers is not in the repo).
    # `threshold=1e-4` on the VARIANCE is exactly `sigma_post < 0.01 *
    # sigma_prior`, i.e. reading "1% of the prior" as a ratio of standard
    # deviations rather than of variances.
    below_sigma = frac_posterior_var_below_prior(model, threshold=1e-4)
    row = dict(
        dataset=label, protocol=protocol, epochs=epochs, elbo_samples=elbo_samples,
        seed=seed, n_fit=len(X_fit), init_sigma=INIT_SIGMA,
        prior_sigma_min=float(priors.min()), prior_sigma_max=float(priors.max()),
        frac_var_below_prior_weights=below["weights"],
        frac_var_below_prior_all=below["all"],
        frac_sigma_below_1pct_prior_weights=below_sigma["weights"],
        frac_sigma_below_1pct_prior_all=below_sigma["all"],
        n_variational_weights=below["n_weights"],
        n_variational_params=below["n_all"],
        frac_sigma_above_init_weights=float(np.mean(w > INIT_SIGMA)),
        sigma_w_mean=float(w.mean()),
        std_epi_median=float(np.median(std_epi)),
        std_epi_min=float(std_epi.min()), std_epi_max=float(std_epi.max()),
        mean_var_aleatoric=float(np.mean(pred.var_aleatoric)),
        data_term_sum_nll=data_term, kl_term=kl, kl_over_data=kl / abs(data_term),
    )
    for q in QUANTILES:
        row[f"sigma_w_q{int(q * 100)}"] = float(np.quantile(w, q))
        row[f"sigma_all_q{int(q * 100)}"] = float(np.quantile(all_sigmas, q))
    return row


def run_uci(datasets, epochs_by_dataset):
    rows = []
    for name in datasets:
        ds = load_uci(name, split=SPLIT_INDEX)
        epochs = epochs_by_dataset[name]
        set_seed(SEED)
        method = BBBMethod(epochs=epochs, elbo_samples=ELBO_SAMPLES)
        method.fit(ds.X_train, ds.y_train, seed=SEED, use_cache=False)
        pred = method.predict(ds.X_test)
        rows.append(_row(name, "E2 (UCI)", epochs, ELBO_SAMPLES, SEED, method.model, pred,
                         ds.X_train, ds.y_train))
        print(f"  {name} K={ELBO_SAMPLES} done", flush=True)
    return rows


def run_e1(elbo_grid, seeds, sigma_mode):
    """`sigma_mode='fixed'` reproduces the CURRENT E1 protocol (D-sigma-E1,
    `fixed_sigma2=0.01`); `'learned'` reproduces the configuration D14d
    actually measured, which predates that decision.
    """
    protocol = f"E1 (synthetic, sigma_o {sigma_mode})"
    rows = []
    for seed in seeds:
        ds = SYNTHETIC_DATASETS[E1_DATASET](seed=seed)
        for elbo_samples in elbo_grid:
            kwargs = dict(e1_method_kwargs("bbb", E1_DATASET))  # fixed_sigma2 + elbo_samples, as E1 ran
            if sigma_mode == "learned":
                kwargs.pop("fixed_sigma2", None)
            kwargs["elbo_samples"] = elbo_samples
            set_seed(seed)
            method = BBBMethod(epochs=E1_EPOCHS, **kwargs)
            method.fit(ds.X_train, ds.y_train, seed=seed, use_cache=False)
            pred = method.predict(ds.X_eval)
            rows.append(_row(E1_DATASET, protocol, E1_EPOCHS, elbo_samples, seed,
                             method.model, pred, ds.X_train, ds.y_train))
            print(f"  {E1_DATASET} sigma_o={sigma_mode} K={elbo_samples} seed={seed} done", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS),
                        help="UCI datasets for the part-1 rows; pass an empty string to skip them")
    parser.add_argument("--epochs-from", type=str, default="uci_epochs_sweep_final_combined.csv")
    parser.add_argument("--no-e1", action="store_true", help="skip the E1 rows")
    parser.add_argument("--e1-sigma", type=str, default="fixed", choices=("fixed", "learned"))
    parser.add_argument("--e1-elbo-samples", type=str, default=str(ELBO_SAMPLES))
    parser.add_argument("--e1-seeds", type=str, default=str(SEED))
    args = parser.parse_args()

    datasets = [d for d in args.datasets.split(",") if d]
    rows = []
    if datasets:
        table = pd.read_csv(RESULTS_DIR / args.epochs_from)
        rows += run_uci(datasets, {r.dataset: int(r.epochs) for r in table.itertuples()})
    if not args.no_e1:
        rows += run_e1([int(v) for v in args.e1_elbo_samples.split(",")],
                       [int(v) for v in args.e1_seeds.split(",")],
                       args.e1_sigma)

    df = pd.DataFrame(rows)
    df["timestamp"] = now_iso()
    df["git_commit"] = git_commit_short()

    out = RESULTS_DIR / "bbb_posterior_diagnostic.csv"
    # Appended, not overwritten (brief section 1): a run over one
    # configuration must not wipe the others' rows.
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
        df = df.drop_duplicates(subset=KEYS, keep="last")
    df = df.sort_values(KEYS).reset_index(drop=True)
    upsert_csv(out, df, KEYS)
    print(f"\nwrote {out}")

    shown = df[df[KEYS].apply(tuple, axis=1).isin([tuple(r[k] for k in KEYS) for r in rows])]
    print(f"\nsigma_post = softplus(rho), weights only. Initial value: {INIT_SIGMA:.4f}")
    print(f"  {'dataset':10s} {'protocol':28s} {'K':>3s} {'seed':>4s} {'min':>8s} {'q25':>8s} "
          f"{'median':>8s} {'q75':>8s} {'max':>8s} {'>init':>7s}")
    for _, r in shown.iterrows():
        print(f"  {r['dataset']:10s} {r['protocol']:28s} {r['elbo_samples']:3d} {r['seed']:4d} "
              f"{r['sigma_w_q0']:8.4f} {r['sigma_w_q25']:8.4f} {r['sigma_w_q50']:8.4f} "
              f"{r['sigma_w_q75']:8.4f} {r['sigma_w_q100']:8.4f} "
              f"{r['frac_sigma_above_init_weights']:6.1%}")

    print("\nthreshold conventions, std_epistemic, ELBO terms")
    print(f"  {'dataset':10s} {'K':>3s} {'seed':>4s} {'var<1%pri':>10s} {'sig<1%pri':>10s} "
          f"{'std_epi_med':>11s} {'sum NLL':>10s} {'KL':>8s} {'KL/|NLL|':>9s}")
    for _, r in shown.iterrows():
        print(f"  {r['dataset']:10s} {r['elbo_samples']:3d} {r['seed']:4d} "
              f"{r['frac_var_below_prior_weights']:10.2%} {r['frac_sigma_below_1pct_prior_weights']:10.2%} "
              f"{r['std_epi_median']:11.5f} {r['data_term_sum_nll']:10.2f} {r['kl_term']:8.2f} "
              f"{r['kl_over_data']:9.4f}")


if __name__ == "__main__":
    main()
