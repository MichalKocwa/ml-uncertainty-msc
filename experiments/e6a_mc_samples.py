"""E6a — how many stochastic passes `T` does a predictive estimate need?

`T = 100` is the default for MC dropout and BBB and it was chosen
arbitrarily (`src/methods/mcd.py`: "arbitrary default — E6a is the ablation
that justifies T"). This is that ablation, and it produces Figure 3.7.

What varies and what does not: the network is trained ONCE per method and
seed, and only the predictive sampling is repeated. That is the whole point
— `T` is a property of the estimator, not of the posterior, so training
again per `T` would mix estimator variance with optimisation variance. Each
`T` is evaluated `REPEATS` times with different sampling streams, and the
quantity of interest is the SPREAD across those repeats: the smallest `T`
whose spread is negligible against the differences between methods is the
`T` the main table needs.

Reported per (method, dataset, T, repeat): the band width (`mpiw95`),
coverage, the epistemic term, and `ll` — a value AND its repeat-to-repeat
standard deviation, since a stable-looking mean at small `T` can still hide
a band that moves by a factor of two between draws.

Writes results/e6a_mc_samples.csv.

Usage:
  python experiments/e6a_mc_samples.py
  python experiments/e6a_mc_samples.py --datasets sin_homo --repeats 3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import metrics
from src.data import SYNTHETIC_DATASETS
from src.methods.bbb import BBBMethod
from src.methods.mcd import MCDropoutMethod
from src.results import RESULTS_DIR, append_generic_csv, git_commit_short, now_iso
from src.seeding import set_seed
from src.style import SEED

OUT_PATH = RESULTS_DIR / "e6a_mc_samples.csv"
T_VALUES = (2, 5, 10, 20, 50, 100, 200, 500)   # brief section 9, E6a
REPEATS = 10                                    # "po 10 losowań masek na punkt"
DATASETS = ("sin_homo", "sin_gap")
METHODS = ("mcd", "bbb")
DEFAULT_T = 100                                 # the value under test


def _build(method_name: str, T: int):
    if method_name == "mcd":
        return MCDropoutMethod(T=T, epochs=2000)
    return BBBMethod(T=T, epochs=2000, elbo_samples=32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
    parser.add_argument("--methods", type=str, default=",".join(METHODS))
    parser.add_argument("--t-values", type=str, default=",".join(map(str, T_VALUES)))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    t_values = [int(v) for v in args.t_values.split(",")]
    rows = []
    for dataset in args.datasets.split(","):
        ds = SYNTHETIC_DATASETS[dataset](seed=args.seed)
        for method_name in args.methods.split(","):
            # Train once; every `T` and every repeat reuses this network.
            set_seed(args.seed)
            method = _build(method_name, max(t_values))
            method.fit(ds.X_train, ds.y_train, seed=args.seed, use_cache=False)
            print(f"{dataset}/{method_name}: trained, sweeping T", flush=True)

            for T in t_values:
                method.T = T
                for repeat in range(args.repeats):
                    # `predict` re-seeds from the fit seed so that a result is
                    # reproducible; overriding it here is what makes the repeats
                    # different sampling streams rather than the same one.
                    method._seed = args.seed + 1000 * repeat
                    pred = method.predict(ds.X_eval)
                    row = dict(
                        experiment_id="e6a", dataset=dataset, method=method_name,
                        T=T, repeat=repeat, init_seed=args.seed,
                        predict_seed=method._seed,
                        mpiw95=metrics.mpiw(pred.std_total),
                        picp95=metrics.picp(ds.y_eval_noisy, pred.mean, pred.std_total),
                        ll=metrics.ll(ds.y_eval_noisy, pred.mean, pred.std_total),
                        rmse=metrics.rmse(ds.y_eval_noisy, pred.mean),
                        mean_std_epistemic=float(np.mean(np.sqrt(pred.var_epistemic))),
                        mean_var_epistemic=float(np.mean(pred.var_epistemic)),
                        timestamp=now_iso(), git_commit=git_commit_short(),
                    )
                    rows.append(row)
                    append_generic_csv(OUT_PATH, row)
            method._seed = args.seed

    df = pd.DataFrame(rows)
    for (dataset, method_name), sub in df.groupby(["dataset", "method"]):
        print(f"\n{dataset}/{method_name}: spread across {args.repeats} repeats "
              f"(sd/mean, i.e. the estimator's own noise)")
        print(f"  {'T':>5s} {'mpiw95':>10s} {'sd':>9s} {'rel sd':>8s} "
              f"{'mean_std_epi':>13s} {'rel sd':>8s} {'picp95':>8s}")
        reference = sub[sub["T"] == max(t_values)].mpiw95.mean()
        for T, g in sub.groupby("T"):
            print(f"  {T:5d} {g.mpiw95.mean():10.4f} {g.mpiw95.std():9.4f} "
                  f"{g.mpiw95.std() / g.mpiw95.mean():8.4f} "
                  f"{g.mean_std_epistemic.mean():13.5f} "
                  f"{g.mean_std_epistemic.std() / g.mean_std_epistemic.mean():8.4f} "
                  f"{g.picp95.mean():8.3f}")
        at_default = sub[sub["T"] == DEFAULT_T]
        if not at_default.empty:
            print(f"  bias of the default T={DEFAULT_T} against T={max(t_values)}: "
                  f"{at_default.mpiw95.mean() - reference:+.4f} mpiw95 "
                  f"({(at_default.mpiw95.mean() / reference - 1) * 100:+.2f}%)")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
