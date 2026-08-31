"""D14e — what does BBB's `elbo_samples` cost at E2 scale, at E2's epoch count?

D14e settled `elbo_samples=32` for E1 (N=250: ~90s vs ~17s per fit, same
order of magnitude) and explicitly deferred E2's value, because at N=8611
and 2000 epochs the same setting measured ~96.3 min per fit — ~32h for one
dataset's 20 splits. That measurement was taken at 2000 epochs; O4/D12 has
since replaced 2000 with a per-dataset epoch count
(`experiments/uci_epochs_sweep.py`), so the cost has to be re-measured at
the count that will actually run.

Two datasets, the extremes of the E2 range: `yacht` (n_train=277) and
`power_plant` (n_train=8611). Grid `{1, 8, 16, 32}` — `1` is the class
default and the reference row the cost multiplier is expressed against;
`{8, 16, 32}` are the candidates.

Reported per run: fit time, and the fraction of variational parameters
whose posterior variance sits below 1% of their prior's
(`frac_posterior_var_below_prior`, both conventions — see its docstring),
which is the quantity D14d used to motivate raising `elbo_samples`. Read
its initialisation caveat before interpreting the numbers: at
`posterior_rho_init = -3.0` an UNTRAINED network already scores 100%, so a
high value at a short epoch count says "the posterior has barely moved",
not necessarily "the posterior collapsed". Test RMSE/LL go to the CSV too,
since the prediction is computed anyway, but they are ONE split at ONE seed
and are not a basis for choosing anything.

**Timing hygiene (D23, D14k's `train_time_s` caveat).** The cache is off by
default, and this script must be the only compute running: D14k's epoch
sweep shares its CPU with a concurrent sweep and its time column had to be
marked "do not cite" as a result.

Writes results/bbb_elbo_samples_cost.csv.

Usage:
  python experiments/bbb_elbo_samples_cost.py
  python experiments/bbb_elbo_samples_cost.py --datasets yacht --epochs yacht=200
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import metrics
from src.data import load_uci
from src.methods.backbone import DEFAULT_BATCH_SIZE
from src.methods.bbb import BBBMethod, frac_posterior_var_below_prior
from src.results import RESULTS_DIR, git_commit_short, now_iso, upsert_csv
from src.seeding import set_seed

DEFAULT_DATASETS = ("yacht", "power_plant")
DEFAULT_ELBO_SAMPLES = (1, 8, 16, 32)
SPLIT_INDEX = 0
SEED = 0
N_SPLITS_E2 = 20  # for the projected "hours for one dataset's full E2 row" column


def _chosen_epochs(path: Path) -> dict:
    """`{dataset: epochs}` from `uci_epochs_sweep.py`'s COMBINED table — the
    per-dataset epoch count O4/D12 settled on (the max over methods, D30).
    Read rather than hard-coded so the two measurements cannot silently
    disagree.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"'{path}' not found — run `python experiments/uci_epochs_sweep.py` first, "
            "or pass --epochs dataset=N explicitly."
        )
    df = pd.read_csv(path)
    return {r.dataset: int(r.epochs) for r in df.itertuples()}


def run(datasets, elbo_grid, epochs_by_dataset, use_cache: bool) -> pd.DataFrame:
    rows = []
    for name in datasets:
        ds = load_uci(name, split=SPLIT_INDEX)
        epochs = epochs_by_dataset[name]
        print(f"\n{name}: n_train={len(ds.X_train)} epochs={epochs} "
              f"(batch_size={DEFAULT_BATCH_SIZE}, split={SPLIT_INDEX}, seed={SEED})")
        for elbo_samples in elbo_grid:
            set_seed(SEED)
            method = BBBMethod(epochs=epochs, elbo_samples=elbo_samples)
            t0 = time.perf_counter()
            method.fit(ds.X_train, ds.y_train, seed=SEED, use_cache=use_cache)
            train_time_s = time.perf_counter() - t0

            var_below_prior = frac_posterior_var_below_prior(method.model)
            pred = method.predict(ds.X_test)
            # metrics in the target's original units (D21): the whole
            # predictive distribution is un-standardised, not just the mean
            scale = float(ds.y_scaler.scale_[0])
            mean = ds.y_scaler.inverse_transform(pred.mean.reshape(-1, 1)).ravel()
            std = pred.std_total * scale
            y_test = ds.y_scaler.inverse_transform(ds.y_test.reshape(-1, 1)).ravel()

            row = dict(
                dataset=name, n_train=len(ds.X_train), epochs=epochs,
                batch_size=DEFAULT_BATCH_SIZE, split_index=SPLIT_INDEX, init_seed=SEED,
                elbo_samples=elbo_samples,
                train_time_s=train_time_s,
                train_time_min=train_time_s / 60.0,
                projected_hours_20_splits=train_time_s * N_SPLITS_E2 / 3600.0,
                frac_var_below_prior_weights=var_below_prior["weights"],
                frac_var_below_prior_all=var_below_prior["all"],
                n_variational_weights=var_below_prior["n_weights"],
                n_variational_params=var_below_prior["n_all"],
                test_rmse=metrics.rmse(y_test, mean),
                test_ll=metrics.ll(y_test, mean, std),
                timestamp=now_iso(), git_commit=git_commit_short(),
            )
            rows.append(row)
            print(f"  elbo_samples={elbo_samples:3d}  fit={train_time_s / 60.0:7.2f} min  "
                  f"({row['projected_hours_20_splits']:6.2f} h / 20 splits)  "
                  f"var<1% prior: weights={var_below_prior['weights']:6.2%} all={var_below_prior['all']:6.2%}  "
                  f"test_rmse={row['test_rmse']:.4f} test_ll={row['test_ll']:.4f}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--elbo-samples", type=str, default=",".join(map(str, DEFAULT_ELBO_SAMPLES)))
    parser.add_argument("--epochs", type=str, default="",
                        help="override the epoch count per dataset, e.g. 'yacht=200,power_plant=100'. "
                             "Default: read from the combined O4/D12 table (see --epochs-from).")
    parser.add_argument("--epochs-from", type=str, default="uci_epochs_sweep_final_combined.csv",
                        help="file in results/ holding the per-dataset epoch counts")
    parser.add_argument("--use-cache", action="store_true",
                        help="reuse cache/ across reruns. OFF by default: this script measures fit time, "
                             "which a cache hit reduces to a state-dict load (D23, D14k).")
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    elbo_grid = [int(v) for v in args.elbo_samples.split(",")]

    epochs_by_dataset = _chosen_epochs(RESULTS_DIR / args.epochs_from)
    for item in filter(None, args.epochs.split(",")):
        key, value = item.split("=")
        epochs_by_dataset[key] = int(value)
    missing = [d for d in datasets if d not in epochs_by_dataset]
    if missing:
        raise SystemExit(f"no epoch count for {missing} — pass --epochs {missing[0]}=N")

    df = run(datasets, elbo_grid, epochs_by_dataset, args.use_cache)
    out = RESULTS_DIR / "bbb_elbo_samples_cost.csv"
    # Appended, not overwritten (brief section 1): a run over one dataset
    # must not wipe another dataset's rows. Re-measuring the same
    # configuration replaces its row rather than duplicating it.
    keys = ["dataset", "epochs", "elbo_samples", "split_index", "init_seed"]
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
        df = df.drop_duplicates(subset=keys, keep="last")
    df = df.sort_values(["dataset", "elbo_samples"]).reset_index(drop=True)
    upsert_csv(out, df, ["dataset", "elbo_samples", "epochs"])
    print(f"\nwrote {out}")

    print(f"\n  {'dataset':13s} {'epochs':>6s} {'K':>3s} {'fit [min]':>9s} {'x(K=1)':>7s} "
          f"{'20 splits [h]':>13s} {'var<pri(w)':>10s} {'var<pri(all)':>12s}")
    for name, sub in df.groupby("dataset", sort=False):
        base = float(sub[sub.elbo_samples == 1]["train_time_s"].iloc[0]) if (sub.elbo_samples == 1).any() else None
        for _, r in sub.iterrows():
            ratio = f"{r['train_time_s'] / base:6.2f}x" if base else "     --"
            print(f"  {r['dataset']:13s} {r['epochs']:6d} {r['elbo_samples']:3d} "
                  f"{r['train_time_min']:9.2f} {ratio:>7s} {r['projected_hours_20_splits']:13.2f} "
                  f"{r['frac_var_below_prior_weights']:10.2%} {r['frac_var_below_prior_all']:12.2%}")


if __name__ == "__main__":
    main()
