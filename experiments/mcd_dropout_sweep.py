"""MC dropout: what `dropout_p` does, and what it cannot do.

**This sweep is not an attempt to fix MC dropout.** It is designed to
demonstrate a negative result for chapter 5: `dropout_p` is a pure
AMPLITUDE control on MCD's epistemic term, with no effect on its SHAPE.
Gal's dropout posterior is a Bernoulli mask over hidden units — nothing in
it is a function of where the training data lies, so the variance it
produces has no mechanism by which to respond to data density (Verdoja &
Kyrki 2020, arXiv 2008.02627: MCD's epistemic estimate is unaffected by
the amount of training data or its variance, and is set by the dropout
rate).

The measurement that shows this: as `dropout_p` grows, the epistemic
profile's LEVEL (`median_std_epi_in_range`) should scale up, while its
normalised growth (`ratio_at_m2`, `ratio_at_8`) stays put — and the
correlation between the log profiles at different `dropout_p` should stay
near 1, i.e. the same curve moved up, not a different curve. Raising
`dropout_p` to widen the band therefore buys coverage in extrapolation
only by also inflating uncertainty where the data is dense, which
`picp95_in_range` / `mpiw95_in_range` price directly.

Not named `e*_`: the brief's E-numbers (section 9) are its own
experiments; this is a diagnostic for the discussion chapter. Related but
distinct from P13, which must reproduce gal2016's protocol literally
(relu, learned sigma_o) — this one stays on the project's own E1 protocol
so its numbers are comparable with every other E1 result.

Writes results/mcd_dropout_sweep.csv and
results/mcd_dropout_profiles.csv (per-x std_epi, for the shape test).

Usage:
  python experiments/mcd_dropout_sweep.py
  python experiments/mcd_dropout_sweep.py --dropout-p 0.1,0.3 --seeds 0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data import SYNTHETIC_DATASETS
from src.methods.mcd import MCDropoutMethod
from src.results import RESULTS_DIR, git_commit_short, now_iso, upsert_csv
from src.seeding import set_seed
from src.sweeps import aggregate, evaluate_on_synthetic

KNOWN_HOMOSCEDASTIC_SIGMA = {"sin_homo": 0.1 ** 2, "sin_gap": 0.1 ** 2}

DEFAULT_DROPOUT_P = (0.05, 0.1, 0.2, 0.3)
DEFAULT_DATASETS = ("sin_homo",)
DEFAULT_SEEDS = (0, 1, 2)

REPORT_COLS = [
    "median_std_epi_in_range", "std_epi_at_m2", "std_epi_at_8",
    "ratio_at_m2", "ratio_at_8",
    "rmse_in_range", "ll_in_range", "picp95_in_range", "mpiw95_in_range",
    "rmse_extrapolation", "ll_extrapolation", "picp95_extrapolation", "mpiw95_extrapolation",
]


def run(dropout_grid, datasets, seeds, use_cache: bool):
    rows, profiles = [], []
    for dataset in datasets:
        for p in dropout_grid:
            for seed in seeds:
                set_seed(seed)
                ds = SYNTHETIC_DATASETS[dataset](seed=seed)
                kwargs = {}
                if dataset in KNOWN_HOMOSCEDASTIC_SIGMA:
                    kwargs["fixed_sigma2"] = KNOWN_HOMOSCEDASTIC_SIGMA[dataset]

                method = MCDropoutMethod(dropout_p=p, **kwargs)
                method.fit(ds.X_train, ds.y_train, seed=seed, use_cache=use_cache)
                pred = method.predict(ds.X_eval)

                rows.append(dict(dataset=dataset, dropout_p=p, seed=seed,
                                 **evaluate_on_synthetic(pred, ds, dataset),
                                 timestamp=now_iso(), git_commit=git_commit_short()))
                profiles.append(pd.DataFrame(dict(
                    dataset=dataset, dropout_p=p, seed=seed,
                    x=ds.X_eval.ravel(), std_epi=np.sqrt(pred.var_epistemic),
                )))
                r = rows[-1]
                print(f"  {dataset:9s} p={p:.2f} seed={seed}  "
                      f"med_epi={r['median_std_epi_in_range']:.4f} "
                      f"r(-2)={r['ratio_at_m2']:5.2f} r(8)={r['ratio_at_8']:5.2f} "
                      f"picp_in={r['picp95_in_range']:.3f} mpiw_in={r['mpiw95_in_range']:.3f} "
                      f"picp_ex={r['picp95_extrapolation']:.3f}")
    return pd.DataFrame(rows), pd.concat(profiles, ignore_index=True)


def shape_correlations(profiles: pd.DataFrame, dataset: str, seed: int) -> pd.DataFrame:
    """Pearson correlation between log-`std_epi` profiles at different
    `dropout_p`, same dataset and seed.

    On logs, because the claim under test is multiplicative: "the same
    shape scaled by a constant" is exactly `log s_p(x) = log s_q(x) + c`,
    which is correlation 1 on logs regardless of the constant. A high
    correlation here says `dropout_p` moved the curve without reshaping it.
    """
    sub = profiles[(profiles.dataset == dataset) & (profiles.seed == seed)]
    wide = sub.pivot(index="x", columns="dropout_p", values="std_epi")
    return np.log(wide).corr()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dropout-p", type=str, default=",".join(map(str, DEFAULT_DROPOUT_P)))
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()

    dropout_grid = [float(v) for v in args.dropout_p.split(",")]
    datasets = args.datasets.split(",")
    seeds = [int(v) for v in args.seeds.split(",")]

    df, profiles = run(dropout_grid, datasets, seeds, args.use_cache)
    out = RESULTS_DIR / "mcd_dropout_sweep.csv"
    upsert_csv(out, df, ["dataset", "dropout_p", "seed"])
    profiles.to_csv(RESULTS_DIR / "mcd_dropout_profiles.csv", index=False)
    print(f"\nwrote {out} and {RESULTS_DIR / 'mcd_dropout_profiles.csv'}")

    summary = aggregate(df, ["dataset", "dropout_p"], REPORT_COLS)
    for dataset in datasets:
        sub = summary[summary.dataset == dataset]
        print(f"\n{dataset} — mean +/- std across seeds")
        print(f"  {'p':>5s} {'med_epi (LEVEL)':>18s} {'ratio(-2) (SHAPE)':>20s} {'ratio(8) (SHAPE)':>20s}")
        for _, r in sub.iterrows():
            print(f"  {r.dropout_p:5.2f} "
                  f"{r.median_std_epi_in_range_mean:9.4f}+/-{r.median_std_epi_in_range_std:.4f} "
                  f"{r.ratio_at_m2_mean:11.2f}+/-{r.ratio_at_m2_std:5.2f} "
                  f"{r.ratio_at_8_mean:11.2f}+/-{r.ratio_at_8_std:5.2f}")
        print(f"  {'p':>5s} {'picp_in':>15s} {'mpiw_in':>15s} {'picp_extrap':>15s} "
              f"{'mpiw_extrap':>15s} {'ll_in':>15s} {'rmse_in':>15s}")
        for _, r in sub.iterrows():
            print(f"  {r.dropout_p:5.2f} "
                  f"{r.picp95_in_range_mean:7.3f}+/-{r.picp95_in_range_std:.3f} "
                  f"{r.mpiw95_in_range_mean:7.3f}+/-{r.mpiw95_in_range_std:.3f} "
                  f"{r.picp95_extrapolation_mean:7.3f}+/-{r.picp95_extrapolation_std:.3f} "
                  f"{r.mpiw95_extrapolation_mean:7.3f}+/-{r.mpiw95_extrapolation_std:.3f} "
                  f"{r.ll_in_range_mean:7.3f}+/-{r.ll_in_range_std:.3f} "
                  f"{r.rmse_in_range_mean:7.4f}+/-{r.rmse_in_range_std:.4f}")

        print(f"\n  shape test — corr(log std_epi profiles) across dropout_p, seed={seeds[0]}:")
        corr = shape_correlations(profiles, dataset, seeds[0])
        print("   " + "\n   ".join(corr.round(3).to_string().split("\n")))


if __name__ == "__main__":
    main()
