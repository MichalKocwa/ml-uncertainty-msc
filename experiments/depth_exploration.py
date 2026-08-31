"""Topology exploration for visual comparison — depth x width, seed=0.

Follow-up to `experiments/e5_depth.py`, and NOT a decision: E5's `{1,2}` arm
showed depth=2 fixing the in-between uncertainty shape (bbb `gap_ratio`
1.13 -> 1.66, mcd 0.70 -> 1.12) while extrapolation collapsed for every
method including MAP (`sin_homo` extrapolation RMSE 0.233 -> 0.559 for map,
which has no epistemic term at all — so that half is a property of the mean
function, not of any uncertainty estimate).

`hidden=20` is in the grid to separate those two effects. At 2x50 the
network has p=2701 against N=250; at 2x20 it has p=481, much closer to the
current 1x50's p=151. If the extrapolation collapse tracks parameter count
rather than depth, 2x20 should keep depth's benefit to the uncertainty
shape without the cost — and if it collapses anyway, the cost is depth's,
not capacity's. Either answer is informative; that is why both widths are
here rather than only the one that might work.

**Nothing here changes a default and nothing here writes into the thesis
directories.** Predictions go to `results/depth_exploration/`, figures to
`figures/depth_exploration/`; `results/predictions_1d/` and
`figures/rodzial3_rys/` are untouched.

Y axis: one range shared by every configuration, method and dataset,
computed from the fitted results by `plotting.exploration_band_extent` and
reported by this script. Never per figure — a per-figure range would make a
collapsed band and a huge band look the same, which is the one thing this
comparison exists to distinguish.

Seed 0 only, TanH, otherwise the E1 protocol (`src.sweeps.e1_method_kwargs`).
Single seed is deliberate and matches the sweep methodology fixed in D14d:
exploration is one seed, confirmation is three, and this is exploration.

Writes:
  results/depth_exploration/{dataset}_{method}_d{depth}h{hidden}.csv
  results/depth_exploration/{dataset}_train.csv
  results/depth_exploration_summary.csv
  figures/depth_exploration/{dataset}_{method}_d{depth}h{hidden}[_epistemic].png

Usage:
  python experiments/depth_exploration.py
  python experiments/depth_exploration.py --figures-only   # redraw from saved predictions
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data import SYNTHETIC_DATASETS
from src.methods import METHODS
from src.plotting import (
    EXPLORATION_FIGURES_DIR, EXPLORATION_PREDICTIONS_DIR, config_tag,
    exploration_band_extent, make_exploration_figure,
)
from src.results import RESULTS_DIR, upsert_csv
from src.seeding import set_seed
from src.style import METHOD_ORDER, SEED, Y_RANGE
from src.sweeps import e1_method_kwargs, evaluate_on_synthetic

# (depth, hidden). 1x50 first: it is the current default and the reference
# every other row is read against.
CONFIGS = [(1, 50), (2, 50), (3, 50), (2, 20), (3, 20), (1, 20)]
DATASETS = ("sin_homo", "sin_gap")
SUMMARY_PATH = RESULTS_DIR / "depth_exploration_summary.csv"


SUMMARY_KEYS = ["config", "dataset", "method"]


def upsert_summary(rows) -> pd.DataFrame:
    """Merge `rows` into SUMMARY_PATH, keyed on (config, dataset, method).

    Written after EVERY fit, not once at the end. This grid takes ~40
    minutes end to end and has been interrupted mid-run more than once;
    with an end-of-run write, an interruption left the prediction CSVs on
    disk but no metrics at all, so the completed fits could not be reported
    without redoing them.
    """
    df = pd.DataFrame(rows)
    if SUMMARY_PATH.exists():
        existing = pd.read_csv(SUMMARY_PATH)
        incoming = set(map(tuple, df[SUMMARY_KEYS].values))
        keep = [tuple(v) not in incoming for v in existing[SUMMARY_KEYS].values]
        df = pd.concat([existing[keep], df], ignore_index=True)
    upsert_csv(SUMMARY_PATH, df, SUMMARY_KEYS)
    return df


def n_parameters_for(method_name: str, dataset: str, depth: int, hidden: int, in_dim: int = 1) -> int:
    """Parameter count without training, for rows recovered from saved
    predictions. Built from the real method classes rather than an arithmetic
    formula so it cannot drift from what the classes actually construct.

    `fixed_sigma2` has to come from `e1_method_kwargs` here, not be left at
    its default: under the E1 protocol `log_sigma2` is a registered buffer
    rather than an `nn.Parameter`, so `count_parameters` excludes it. Without
    this the recovered rows report one parameter more than the freshly
    fitted ones (152 vs 151 at 1x50) — the same configuration showing two
    different sizes depending on whether its row happened to be recomputed.
    """
    from src.methods.backbone import HomoscedasticMLP, count_parameters
    if method_name == "gp":
        return 3
    fixed_sigma2 = e1_method_kwargs(method_name, dataset).get("fixed_sigma2")
    n = count_parameters(HomoscedasticMLP(
        in_dim=in_dim, hidden=hidden, depth=depth, fixed_sigma2=fixed_sigma2))
    return n * METHODS["ensemble"]().M if method_name == "ensemble" else n


def _row_from_prediction(pred, ds, dataset, method_name, depth, hidden, seed, n_parameters) -> dict:
    metrics = evaluate_on_synthetic(pred, ds, dataset)
    return dict(
        config=f"{depth}x{hidden}", depth=depth, hidden=hidden,
        dataset=dataset, method=method_name, seed=seed, n_parameters=n_parameters,
        median_std_epi_in_range=metrics["median_std_epi_in_range"],
        gap_ratio=metrics.get("gap_ratio", np.nan),
        rmse_in_range=metrics["rmse_in_range"],
        rmse_extrapolation=metrics["rmse_extrapolation"],
        ll_extrapolation=metrics["ll_extrapolation"],
        picp95_extrapolation=metrics["picp95_extrapolation"],
    )


def fit_all(configs, datasets, methods, seed: int, refit: bool = False):
    """Fit every (config, dataset, method), skipping any whose predictions
    are already on disk unless `refit`.

    Resumable by design: a skipped cell still has its metrics recomputed
    from the saved predictions (cheap — no training), so an interrupted run
    followed by a second invocation produces the same complete summary as
    one uninterrupted run would have.
    """
    from src.methods.base import Prediction

    for dataset in datasets:
        ds = SYNTHETIC_DATASETS[dataset](seed=seed)
        EXPLORATION_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"x": ds.X_train.ravel(), "y": ds.y_train}).to_csv(
            EXPLORATION_PREDICTIONS_DIR / f"{dataset}_train.csv", index=False)

        for depth, hidden in configs:
            for method_name in methods:
                path = (EXPLORATION_PREDICTIONS_DIR
                        / f"{dataset}_{method_name}_{config_tag(depth, hidden)}.csv")
                if path.exists() and not refit:
                    saved = pd.read_csv(path)
                    pred = Prediction(
                        mean=saved["mean"].values,
                        var_aleatoric=saved.std_alea.values ** 2,
                        var_epistemic=saved.std_epi.values ** 2,
                    )
                    n_parameters = n_parameters_for(
                        method_name, dataset, depth, hidden, ds.X_train.shape[1])
                    status = "cached"
                else:
                    set_seed(seed)
                    ds = SYNTHETIC_DATASETS[dataset](seed=seed)
                    kwargs = e1_method_kwargs(method_name, dataset)
                    if method_name != "gp":
                        kwargs.update(depth=depth, hidden=hidden)
                    method = METHODS[method_name](**kwargs)
                    method.fit(ds.X_train, ds.y_train, seed=seed, use_cache=False)
                    pred = method.predict(ds.X_eval)
                    pd.DataFrame({
                        "x": ds.X_eval.ravel(), "mean": pred.mean,
                        "std_alea": np.sqrt(pred.var_aleatoric), "std_epi": np.sqrt(pred.var_epistemic),
                        "y_true": ds.y_eval,
                    }).to_csv(path, index=False)
                    n_parameters = method.n_parameters
                    status = "fitted"

                row = _row_from_prediction(
                    pred, ds, dataset, method_name, depth, hidden, seed, n_parameters)
                df = upsert_summary([row])
                print("  {:9s} {:>6s} {:9s} p={:6d} med_epi={:.4f} gap={:>6s} "
                      "rmse_in={:.4f} rmse_ex={:.4f} ll_ex={:8.3f} picp_ex={:.3f}  [{}]".format(
                          dataset, f"{depth}x{hidden}", method_name, n_parameters,
                          row["median_std_epi_in_range"],
                          "-" if np.isnan(row["gap_ratio"]) else "{:.2f}".format(row["gap_ratio"]),
                          row["rmse_in_range"], row["rmse_extrapolation"],
                          row["ll_extrapolation"], row["picp95_extrapolation"], status), flush=True)
    return df


def draw_all(configs, datasets, methods):
    """Both band versions, one shared y range each, reported before drawing."""
    ranges = {}
    for epistemic in (False, True):
        column = "std_epi" if epistemic else "std_total"
        lo, hi = exploration_band_extent(configs, datasets, methods, column)
        ranges[column] = (lo, hi)
        label = "epistemic" if epistemic else "total"
        fits = "fits inside" if (lo >= Y_RANGE[0] and hi <= Y_RANGE[1]) else "DOES NOT FIT inside"
        print(f"\nshared y range ({label} band): ({lo:.3f}, {hi:.3f}) — {fits} Y_RANGE={Y_RANGE}", flush=True)

    for epistemic in (False, True):
        column = "std_epi" if epistemic else "std_total"
        y_range = ranges[column]
        for depth, hidden in configs:
            for dataset in datasets:
                for method in methods:
                    suffix = "_epistemic" if epistemic else ""
                    name = f"{dataset}_{method}_{config_tag(depth, hidden)}{suffix}.png"
                    make_exploration_figure(
                        dataset, method, depth, hidden,
                        EXPLORATION_FIGURES_DIR / name, y_range, epistemic=epistemic)
    n = len(configs) * len(datasets) * len(methods) * 2
    print(f"wrote {n} figures to {EXPLORATION_FIGURES_DIR}", flush=True)
    return ranges


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== summary (seed=0) ===", flush=True)
    print("  {:7s} {:9s} {:>7s} {:>10s} {:>10s} {:>10s} {:>11s} {:>10s} {:>9s}".format(
        "config", "method", "p", "med_epi", "gap_ratio", "rmse_in", "rmse_extrap",
        "ll_extrap", "picp_ex"), flush=True)
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    for (config, depth, hidden), g in df.groupby(["config", "depth", "hidden"], sort=False):
        for _, r in g.assign(_o=g.method.map(order)).sort_values(["_o", "dataset"]).iterrows():
            gap = "-" if np.isnan(r.gap_ratio) else f"{r.gap_ratio:.2f}"
            med = "-" if r.median_std_epi_in_range == 0 else f"{r.median_std_epi_in_range:.4f}"
            print("  {:7s} {:9s} {:7d} {:>10s} {:>10s} {:10.4f} {:11.4f} {:10.3f} {:9.3f}   [{}]".format(
                config, r.method, int(r.n_parameters), med, gap,
                r.rmse_in_range, r.rmse_extrapolation, r.ll_extrapolation,
                r.picp95_extrapolation, r.dataset), flush=True)

    print("\n=== the column that matters most: map's extrapolation RMSE ===", flush=True)
    print("  (map has no epistemic term, so this isolates damage to the MEAN "
          "function from anything to do with uncertainty)", flush=True)
    m = df[df.method == "map"]
    print("  {:7s} {:>7s} {:>16s} {:>16s}".format("config", "p", "sin_homo", "sin_gap"), flush=True)
    for (config, _, _), g in m.groupby(["config", "depth", "hidden"], sort=False):
        by_ds = g.set_index("dataset").rmse_extrapolation
        print("  {:7s} {:7d} {:16.4f} {:16.4f}".format(
            config, int(g.n_parameters.iloc[0]),
            by_ds.get("sin_homo", np.nan), by_ds.get("sin_gap", np.nan)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figures-only", action="store_true",
                        help="redraw from results/depth_exploration/ without refitting")
    parser.add_argument("--fit-only", action="store_true",
                        help="fit and write predictions, no figures — for running the grid in chunks")
    parser.add_argument("--refit", action="store_true",
                        help="ignore saved predictions and retrain (default: resume, reusing them)")
    parser.add_argument("--configs", type=str, default=None,
                        help="subset as 'DxH,DxH' (e.g. '2x20,3x20'); default: the full grid")
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
    parser.add_argument("--methods", type=str, default=",".join(METHOD_ORDER))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.configs:
        configs = [tuple(int(v) for v in c.strip().split("x")) for c in args.configs.split(",")]
    else:
        configs = CONFIGS
    datasets = args.datasets.split(",")
    methods = args.methods.split(",")

    if not args.figures_only:
        df = fit_all(configs, datasets, methods, args.seed, refit=args.refit)
        print(f"\nwrote {SUMMARY_PATH}", flush=True)
    else:
        df = pd.read_csv(SUMMARY_PATH)

    if args.fit_only:
        return

    # Figures and the summary table always cover the FULL grid, not the
    # subset just fitted — a chunked run must still produce one comparison.
    draw_all(CONFIGS, DATASETS, list(METHOD_ORDER))
    print_summary(pd.read_csv(SUMMARY_PATH))


if __name__ == "__main__":
    main()
