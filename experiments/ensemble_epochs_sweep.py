"""O8 — does deep ensembles' epistemic band widen at fewer epochs?

Open question O8 (docs/chapter4_notes.md): at 2000 epochs with N=250 and a
1x50 backbone, ensemble members may all converge to the same minimum,
collapsing the between-member variance that IS the method's epistemic
term. Raised by the author at D7b, never measured.

Why ensemble and not the other two: of bbb/mcd/ensemble it is the only one
whose epistemic profile already has the RIGHT SHAPE. On sin_homo at seed=0
its std_epi runs 0.020 (median in-range) to 0.099 at x=-2 — a 4.9x rise
that tracks data density — but the whole curve sits ~5x below sigma_o=0.1,
so it is invisible in a +/-2*sigma_total figure. bbb (ratio 0.99 to the
left) and mcd (1.27) have no shape to amplify; ensemble has shape and no
amplitude. Epoch count is the parameter that plausibly controls exactly
that, and D14h has already ruled out `M` (it changes the PRECISION of the
estimate, not its value).

Not named `e*_`: those ids are reserved for the brief's own experiments
(section 9). This is a diagnostic answering an open question, and claiming
an E-number for it would imply a place in the thesis's experiment list
that the brief does not give it.

Writes results/ensemble_epochs_sweep.csv (one row per epochs x dataset x
seed) — never results/e1_synthetic.csv, whose schema is fixed.

Usage:
  python experiments/ensemble_epochs_sweep.py
  python experiments/ensemble_epochs_sweep.py --epochs 300,2000 --seeds 0
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data import SYNTHETIC_DATASETS
from src.methods.ensemble import DeepEnsembleMethod
from src.results import RESULTS_DIR, git_commit_short, now_iso, upsert_csv
from src.seeding import set_seed
from src.sweeps import aggregate, evaluate_on_synthetic

# Same E1 protocol the figures were produced under (D-sigma-E1): sigma_o
# fixed to its true value on these two datasets, so that anything moving
# between epoch settings is the epistemic term and not the noise estimate
# silently re-fitting itself.
KNOWN_HOMOSCEDASTIC_SIGMA = {"sin_homo": 0.1 ** 2, "sin_gap": 0.1 ** 2}

DEFAULT_EPOCHS = (300, 600, 1000, 2000)
DEFAULT_DATASETS = ("sin_homo", "sin_gap")
DEFAULT_SEEDS = (0, 1, 2)

REPORT_COLS = [
    "median_std_epi_in_range", "std_epi_at_m2", "std_epi_at_8",
    "ratio_at_m2", "ratio_at_8",
    "rmse_in_range", "ll_in_range", "picp95_in_range",
    "rmse_extrapolation", "ll_extrapolation", "picp95_extrapolation",
]


def run(epochs_grid, datasets, seeds, use_cache: bool):
    rows = []
    for dataset in datasets:
        for epochs in epochs_grid:
            for seed in seeds:
                set_seed(seed)
                ds = SYNTHETIC_DATASETS[dataset](seed=seed)
                kwargs = {}
                if dataset in KNOWN_HOMOSCEDASTIC_SIGMA:
                    kwargs["fixed_sigma2"] = KNOWN_HOMOSCEDASTIC_SIGMA[dataset]

                method = DeepEnsembleMethod(epochs=epochs, **kwargs)
                t0 = time.perf_counter()
                method.fit(ds.X_train, ds.y_train, seed=seed, use_cache=use_cache)
                train_time_s = time.perf_counter() - t0
                pred = method.predict(ds.X_eval)

                row = dict(dataset=dataset, epochs=epochs, seed=seed,
                           train_time_s=train_time_s,
                           **evaluate_on_synthetic(pred, ds, dataset),
                           timestamp=now_iso(), git_commit=git_commit_short())
                rows.append(row)
                print(f"  {dataset:9s} epochs={epochs:5d} seed={seed}  "
                      f"med_epi={row['median_std_epi_in_range']:.4f} "
                      f"@-2={row['std_epi_at_m2']:.4f} @8={row['std_epi_at_8']:.4f} "
                      f"r(-2)={row['ratio_at_m2']:5.2f} r(8)={row['ratio_at_8']:5.2f} "
                      f"rmse_in={row['rmse_in_range']:.4f} ll_in={row['ll_in_range']:.3f} "
                      f"picp_in={row['picp95_in_range']:.3f} ({train_time_s:.1f}s)")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=str, default=",".join(map(str, DEFAULT_EPOCHS)))
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--use-cache", action="store_true",
                        help="reuse cache/ across reruns. OFF by default: this sweep records train_time_s, "
                             "which a cache hit reduces to a state-dict load (D23, D-width-E5).")
    args = parser.parse_args()

    epochs_grid = [int(v) for v in args.epochs.split(",")]
    datasets = args.datasets.split(",")
    seeds = [int(v) for v in args.seeds.split(",")]

    df = run(epochs_grid, datasets, seeds, args.use_cache)
    out = RESULTS_DIR / "ensemble_epochs_sweep.csv"
    upsert_csv(out, df, ["dataset", "epochs", "seed"])
    print(f"\nwrote {out}")

    summary = aggregate(df, ["dataset", "epochs"], REPORT_COLS)
    print("\nmean +/- std across seeds:")
    for dataset in datasets:
        sub = summary[summary.dataset == dataset]
        print(f"\n{dataset}")
        print(f"  {'epochs':>6s} {'med_epi':>16s} {'@x=-2':>16s} {'@x=8':>16s} "
              f"{'ratio(-2)':>14s} {'ratio(8)':>14s}")
        for _, r in sub.iterrows():
            print(f"  {int(r.epochs):6d} "
                  f"{r.median_std_epi_in_range_mean:8.4f}+/-{r.median_std_epi_in_range_std:.4f} "
                  f"{r.std_epi_at_m2_mean:8.4f}+/-{r.std_epi_at_m2_std:.4f} "
                  f"{r.std_epi_at_8_mean:8.4f}+/-{r.std_epi_at_8_std:.4f} "
                  f"{r.ratio_at_m2_mean:6.2f}+/-{r.ratio_at_m2_std:5.2f} "
                  f"{r.ratio_at_8_mean:6.2f}+/-{r.ratio_at_8_std:5.2f}")
        print(f"  {'epochs':>6s} {'rmse_in':>16s} {'ll_in':>16s} {'picp_in':>16s} "
              f"{'rmse_extrap':>16s} {'ll_extrap':>16s} {'picp_extrap':>16s}")
        for _, r in sub.iterrows():
            print(f"  {int(r.epochs):6d} "
                  f"{r.rmse_in_range_mean:8.4f}+/-{r.rmse_in_range_std:.4f} "
                  f"{r.ll_in_range_mean:8.3f}+/-{r.ll_in_range_std:.3f} "
                  f"{r.picp95_in_range_mean:8.3f}+/-{r.picp95_in_range_std:.3f} "
                  f"{r.rmse_extrapolation_mean:8.4f}+/-{r.rmse_extrapolation_std:.4f} "
                  f"{r.ll_extrapolation_mean:8.3f}+/-{r.ll_extrapolation_std:.3f} "
                  f"{r.picp95_extrapolation_mean:8.3f}+/-{r.picp95_extrapolation_std:.3f}")


if __name__ == "__main__":
    main()
