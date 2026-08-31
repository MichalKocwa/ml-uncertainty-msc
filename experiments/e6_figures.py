"""Figures 3.7 and 3.9 from E6a and E6c.

  * `figures/rodzial3_rys/img3_7.png` — predictive-estimate stability against
    the number of stochastic passes `T`, drawn from `results/e6a_mc_samples.csv`.
    Justifies the `T = 100` default, which was arbitrary until E6a ran.
  * `figures/rodzial3_rys/img3_9.png` — Laplace's three covariance structures
    side by side on identical axes, the chapter-3 convention (`src/style.py`'s
    `X_RANGE`/`Y_RANGE`, forced, never set per panel), with `epi_extrap_ratio`
    read from `results/e6c_laplace_structure.csv` printed on each panel.

3.9 refits the three structures rather than reading a CSV: `e6c`'s results
file stores summary metrics, not per-point predictions, and three fits on
`sin_homo` cost about fifteen seconds. Everything is `seed = SEED` (0), the
same seed as every other chapter-3 figure, and stated in both captions.

Usage:
  python experiments/e6_figures.py
  python experiments/e6_figures.py --only 3.7
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import SYNTHETIC_DATASETS
from src.methods.laplace import LaplaceMethod
from src.plotting import FIGURES_DIR, _save, _shade_training_support
from src.results import RESULTS_DIR
from src.seeding import set_seed
from src.style import METHOD_COLORS, METHOD_LABELS, SEED, X_RANGE, Y_RANGE

FIG_37 = FIGURES_DIR / "rodzial3_rys" / "img3_7.png"
FIG_39 = FIGURES_DIR / "rodzial3_rys" / "img3_9.png"
STRUCTURES = ("full", "kron", "diag")
DATASET_STYLES = {"sin_homo": "-", "sin_gap": "--"}


def figure_3_7() -> None:
    df = pd.read_csv(RESULTS_DIR / "e6a_mc_samples.csv")
    grouped = df.groupby(["dataset", "method", "T"]).mpiw95.agg(["mean", "std"]).reset_index()
    grouped["rel_sd"] = grouped["std"] / grouped["mean"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # Left: the estimate itself, mean +/- one repeat-to-repeat sd.
    ax = axes[0]
    for (dataset, method), sub in grouped.groupby(["dataset", "method"]):
        sub = sub.sort_values("T")
        ax.errorbar(sub["T"], sub["mean"], yerr=sub["std"], marker="o", markersize=3.5,
                    linewidth=1.3, capsize=2.5, color=METHOD_COLORS[method],
                    linestyle=DATASET_STYLES[dataset],
                    label=f"{METHOD_LABELS[method]}, {dataset}")
    ax.set_xscale("log")
    ax.set_xlabel("T (stochastic forward passes)")
    ax.set_ylabel("MPIW@95")
    ax.set_title("Estimate and its spread over 10 repeats")
    ax.legend(fontsize=7.5, loc="lower right")

    # Right: the estimator's own noise, against the 1/sqrt(T) it should follow.
    ax = axes[1]
    for (dataset, method), sub in grouped.groupby(["dataset", "method"]):
        sub = sub.sort_values("T")
        ax.plot(sub["T"], sub.rel_sd, marker="o", markersize=3.5, linewidth=1.3,
                color=METHOD_COLORS[method], linestyle=DATASET_STYLES[dataset],
                label=f"{METHOD_LABELS[method]}, {dataset}")
    reference_t = np.array(sorted(grouped["T"].unique()), dtype=float)
    anchor = grouped[(grouped.method == "bbb") & (grouped["T"] == reference_t[0])].rel_sd.max()
    ax.plot(reference_t, anchor * np.sqrt(reference_t[0] / reference_t), color="black",
            linewidth=0.9, linestyle=":", alpha=0.7, label=r"$1/\sqrt{T}$ reference")
    ax.axvline(100, color="black", linewidth=0.8, alpha=0.35)
    ax.annotate("default T = 100", xy=(100, ax.get_ylim()[1]), xytext=(105, anchor * 0.75),
                fontsize=7.5, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("T (stochastic forward passes)")
    ax.set_ylabel("relative sd of MPIW@95")
    ax.set_title("Estimator noise: BBB needs ~10x the samples")
    ax.legend(fontsize=7.5, loc="lower left")

    fig.suptitle(
        f"Predictive-estimate stability against T (one trained network per method, "
        f"10 resampled predictions per T, seed = {SEED})", fontsize=10,
    )
    fig.tight_layout()
    _save(fig, FIG_37)
    print(f"wrote {FIG_37}")


def figure_3_9(dataset: str = "sin_homo") -> None:
    summary = pd.read_csv(RESULTS_DIR / "e6c_laplace_structure.csv")
    summary = summary[(summary.dataset == dataset) & (summary.status == "ok")
                      & (summary.prior_precision_mode == "fixed")]
    ratios = summary.groupby("hessian_structure").epi_extrap_ratio.mean()

    ds = SYNTHETIC_DATASETS[dataset](seed=SEED)
    x = ds.X_eval.ravel()
    colour = METHOD_COLORS["laplace"]

    fig, axes = plt.subplots(1, len(STRUCTURES), figsize=(12.0, 4.0), sharey=True)
    for ax, structure in zip(axes, STRUCTURES):
        set_seed(SEED)
        method = LaplaceMethod(epochs=2000, hessian_structure=structure,
                               prior_precision_mode="fixed")
        method.fit(ds.X_train, ds.y_train, seed=SEED, use_cache=False)
        pred = method.predict(ds.X_eval)

        _shade_training_support(ax, ds.X_train.ravel())
        ax.fill_between(x, pred.mean - 2 * pred.std_total, pred.mean + 2 * pred.std_total,
                        color=colour, alpha=0.20, label=r"$\pm 2\sigma_{\mathrm{total}}$")
        ax.fill_between(x, pred.mean - 2 * np.sqrt(pred.var_epistemic),
                        pred.mean + 2 * np.sqrt(pred.var_epistemic),
                        color=colour, alpha=0.38, label=r"$\pm 2\sigma_{\mathrm{epistemic}}$")
        ax.plot(x, pred.mean, color=colour, linewidth=1.3, label="mean")
        ax.plot(x, np.sin(x), color="black", linestyle="--", linewidth=1.0, alpha=0.6,
                label="true f(x)")
        ax.scatter(ds.X_train.ravel(), ds.y_train, s=5, color="black", alpha=0.30, zorder=5)
        # Identical axes across panels, forced from src/style.py — the chapter-3
        # requirement, and the only way the three structures are comparable by eye.
        ax.set_xlim(*X_RANGE)
        ax.set_ylim(*Y_RANGE)
        ax.set_xlabel("x")
        ax.set_title(f"{structure}  (epi. growth {ratios[structure]:.1f}x)")

    axes[0].set_ylabel("y")
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle(
        f"Laplace posterior under three covariance structures, shared prior "
        f"(prior_precision = 1/$\\gamma^2$), {dataset}, seed = {SEED}; identical axes", fontsize=10,
    )
    fig.tight_layout()
    _save(fig, FIG_39)
    print(f"wrote {FIG_39}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", type=str, default=None, choices=["3.7", "3.9"])
    parser.add_argument("--dataset", type=str, default="sin_homo", help="dataset for figure 3.9")
    args = parser.parse_args()

    if args.only in (None, "3.7"):
        figure_3_7()
    if args.only in (None, "3.9"):
        figure_3_9(args.dataset)


if __name__ == "__main__":
    main()
