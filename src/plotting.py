"""Figure generation for E1 (brief section 10).

One function, `_draw_posterior`, draws every per-method posterior panel —
the thesis figures on `sin_homo` (one per method, `img3_{method}.png`) and
every multi-method figure (`make_overlay_figure`, `make_grid_figure`) all
call it, so "identical axes and scale" is a property of sharing code, not
a convention to remember per figure. Axis limits, seed, colours and
eval-grid resolution all come from `src/style.py` and are never set per
call.

Reads only `results/predictions_1d/{dataset}_{method}.csv` and
`results/predictions_1d/{dataset}_train.csv` (both written by
`experiments/e1_synthetic.py`) — no `src.data`, no `src.methods`, no
retraining, per brief section 8. The one exception is
`make_img2_2_function_samples`, which needs genuine GP function draws
(`gp.sample_y`) not stored in `predictions_1d`; it imports `src.methods.gp`
locally and is not part of `figures/redraw.py`'s no-training contract.

Not generated here (brief section 10, this session's instructions):
`img2_3_1.png` and `img3_4.png` are conceptual, drawn by hand, not from
data. `img2_1.png` needs the two-head heteroscedastic network ruled out in
Etap 2 and belongs in a separate script.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.style import (
    METHOD_COLORS, METHOD_LABELS, METHOD_ORDER, SEED, X_RANGE, Y_RANGE, Y_RANGE_EPISTEMIC,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
PREDICTIONS_DIR = REPO_ROOT / "results" / "predictions_1d"

# Topology exploration (E5 follow-up): a SEPARATE predictions directory and a
# SEPARATE figures directory from the thesis figures. Nothing here writes into
# `figures/rodzial3_rys/` or `results/predictions_1d/` — this is visual
# exploration of architectures that are not the default, and mixing it into the
# directories the thesis reads from would make "which figure came from the
# configuration we actually use" a question someone has to check rather than see.
EXPLORATION_PREDICTIONS_DIR = REPO_ROOT / "results" / "depth_exploration"
EXPLORATION_FIGURES_DIR = FIGURES_DIR / "depth_exploration"


def config_tag(depth: int, hidden: int) -> str:
    """`d{depth}h{hidden}` — the filename tag for one topology."""
    return f"d{depth}h{hidden}"


def _load_predictions(dataset: str, method: str) -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_DIR / f"{dataset}_{method}.csv")
    df["std_total"] = np.sqrt(df.std_alea ** 2 + df.std_epi ** 2)
    return df


def _load_train_points(dataset: str) -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_DIR / f"{dataset}_train.csv")


def load_exploration_predictions(dataset: str, method: str, depth: int, hidden: int) -> pd.DataFrame:
    df = pd.read_csv(EXPLORATION_PREDICTIONS_DIR / f"{dataset}_{method}_{config_tag(depth, hidden)}.csv")
    df["std_total"] = np.sqrt(df.std_alea ** 2 + df.std_epi ** 2)
    return df


def load_exploration_train_points(dataset: str) -> pd.DataFrame:
    return pd.read_csv(EXPLORATION_PREDICTIONS_DIR / f"{dataset}_train.csv")


def exploration_band_extent(configs, datasets, methods, std_column: str, margin: float = 0.1):
    """The tightest `(lo, hi)` containing every panel's band across EVERY
    configuration, method and dataset in the exploration, plus a margin.

    Computed once over the whole grid and then passed to every panel, which
    is the only way a visual comparison across topologies means anything: a
    per-figure y range would make a configuration with a collapsed band and
    one with a huge band look identical. Returned rather than stored as a
    module constant because these are exploratory architectures whose range
    is not known until they have been fitted — unlike `Y_RANGE`, which is
    pinned to the default configuration's known results.
    """
    lo, hi = np.inf, -np.inf
    for depth, hidden in configs:
        for dataset in datasets:
            train = load_exploration_train_points(dataset)
            lo, hi = min(lo, train.y.min()), max(hi, train.y.max())
            for method in methods:
                df = load_exploration_predictions(dataset, method, depth, hidden)
                lo = min(lo, float((df["mean"] - 2 * df[std_column]).min()))
                hi = max(hi, float((df["mean"] + 2 * df[std_column]).max()))
    span = hi - lo
    return lo - margin * span, hi + margin * span


def make_exploration_figure(
    dataset: str, method: str, depth: int, hidden: int, out_path: Path, y_range, epistemic: bool = False,
) -> None:
    """One exploration panel, drawn through the same `_draw_band` as every
    thesis figure — only the predictions directory and the y range differ.
    """
    std_column = "std_epi" if epistemic else "std_total"
    band_label = (r"predictive $\pm 2\sigma_{\mathrm{epistemic}}$" if epistemic
                  else r"predictive $\pm 2\sigma$")
    suffix = " — epistemic only" if epistemic else ""
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    _draw_band(
        ax,
        load_exploration_predictions(dataset, method, depth, hidden),
        load_exploration_train_points(dataset),
        METHOD_COLORS[method], std_column, band_label, y_range,
        title=f"{METHOD_LABELS[method]} — {depth}x{hidden} ({dataset}, seed={SEED}){suffix}",
    )
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, out_path)


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _training_intervals(train_x, gap_factor: float = 10.0):
    """Contiguous intervals covered by the training points, as
    `[(lo, hi), ...]`.

    Derived from the training scatter itself rather than imported from
    `src.data`, deliberately: this module's contract (see the module
    docstring) is that it reads `results/predictions_1d/` and nothing else,
    so that `figures/redraw.py` can redraw every figure without importing
    any method or data code. Splitting on gaps in the sorted x values keeps
    that contract and handles `sin_gap`'s two training halves with the same
    code path as `sin_homo`'s single interval — no per-dataset branch, and
    nothing to update if a future variant has three islands.

    `gap_factor` x the median spacing is the split threshold. On `sin_gap`
    the ratio it has to separate is ~124 (a 2.0-wide hole against ~0.016
    spacing), so 10 is not a tuned constant — anything from ~2 to ~100
    gives the same answer on every variant here.
    """
    xs = np.sort(np.asarray(train_x))
    if xs.size < 2:
        return [(float(xs[0]), float(xs[0]))] if xs.size else []
    diffs = np.diff(xs)
    threshold = gap_factor * float(np.median(diffs))
    breaks = np.flatnonzero(diffs > threshold)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [xs.size - 1]])
    return [(float(xs[a]), float(xs[b])) for a, b in zip(starts, ends)]


def _shade_training_support(ax, train_x, label: str = "training support") -> None:
    """Grey band behind everything else marking where training data exists.

    Without it the reader has to infer the boundary from where the scatter
    stops, which is exactly the judgement these figures are asking them to
    make — the whole question is what each method's band does on either
    side of that line, so the line should be drawn, not inferred. Matches
    the shading already used by the diagnostic raw-path figures.
    """
    for i, (lo, hi) in enumerate(_training_intervals(train_x)):
        ax.axvspan(lo, hi, color="0.85", alpha=0.45, linewidth=0, zorder=0,
                   label=label if i == 0 else None)


def _draw_band(ax, df, train, color, std_column: str, band_label: str, y_range, title=None) -> None:
    """The one posterior panel implementation: shaded training support, the
    +/-2 sigma band of `std_column`, the predictive mean, true f(x), the
    training scatter, and forced (X_RANGE, y_range) axes.

    Takes already-loaded frames rather than (dataset, method) strings so
    that the depth-exploration figures — which read a different predictions
    directory and use their own shared y range — go through exactly this
    code rather than a near-copy of it. `y_range` is a parameter, but every
    caller passes a constant shared across all of its own panels; none
    computes one per figure.
    """
    _shade_training_support(ax, train.x)
    ax.fill_between(
        df.x, df["mean"] - 2 * df[std_column], df["mean"] + 2 * df[std_column],
        color=color, alpha=0.25, linewidth=0, label=band_label,
    )
    ax.plot(df.x, df["mean"], color=color, linewidth=1.6, label="predictive mean")
    ax.plot(df.x, df.y_true, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="true f(x)")
    ax.scatter(train.x, train.y, s=6, color="black", alpha=0.35, zorder=5, label="train")

    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*y_range)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is not None:
        ax.set_title(title)


def _draw_posterior(ax, dataset: str, method: str, title: str = None) -> None:
    """One panel: train scatter, true f(x), predictive mean, +/-2 sigma band.
    Forced shared (X_RANGE, Y_RANGE) axes — the only path any thesis figure
    in this module uses to draw a total-band posterior, so "identical axes
    and scale" cannot drift between calls.
    """
    _draw_band(
        ax, _load_predictions(dataset, method), _load_train_points(dataset),
        METHOD_COLORS[method], "std_total", r"predictive $\pm 2\sigma$", Y_RANGE, title,
    )


def make_single_method_figure(dataset: str, method: str, out_path: Path) -> None:
    """One method, one dataset — e.g. img3_gp.png, img3_bbb.png, e1_map.png."""
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    _draw_posterior(ax, dataset, method, title=f"{METHOD_LABELS[method]} ({dataset}, seed={SEED})")
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, out_path)


def _draw_posterior_epistemic(ax, dataset: str, method: str, title: str = None) -> None:
    """Same panel as `_draw_posterior`, but the band is +/-2*std_epi around
    the mean, not +/-2*std_total (D14f, docs/chapter4_notes.md). Only
    meaningful because E1's aleatoric term is now fixed to the same true
    value across every method on sin_homo/sin_gap (D-sigma-E1) — with a
    shared constant subtracted out, the total band's differences between
    methods ARE the epistemic term already, but at E1's scale (sigma_o~0.1)
    that term is visually swamped in a +/-2*sigma_total plot for the three
    sampling-based methods (bbb/mcd/ensemble), whose in-range epistemic std
    is comparable to sigma_o itself.

    **Y axis: `Y_RANGE_EPISTEMIC`, not `Y_RANGE`** — corrected this
    session. These panels previously shared `Y_RANGE=(-5, 5)` with their
    `+/-2*sigma_total` counterparts, on the reading that brief section 10
    requires one scale for every figure. It does not: the requirement is
    that no METHOD be drawn on a different scale from another method, so
    that band widths can be compared by eye across the five panels. Two
    figure types plotting two different quantities were never what it was
    protecting, and forcing them onto one axis cost these panels most of
    their vertical resolution for nothing. `Y_RANGE_EPISTEMIC` is still
    forced identically across all five methods here, so the between-method
    comparability the section actually asks for is intact.
    """
    _draw_band(
        ax, _load_predictions(dataset, method), _load_train_points(dataset),
        METHOD_COLORS[method], "std_epi",
        r"predictive $\pm 2\sigma_{\mathrm{epistemic}}$",
        Y_RANGE_EPISTEMIC,  # NOT Y_RANGE — see src/style.py's Y_RANGE_EPISTEMIC
        title,
    )


def make_single_method_figure_epistemic(dataset: str, method: str, out_path: Path) -> None:
    """Epistemic-only counterpart of `make_single_method_figure` — e.g.
    img3_gp_epistemic.png. See `_draw_posterior_epistemic`."""
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    _draw_posterior_epistemic(ax, dataset, method, title=f"{METHOD_LABELS[method]} ({dataset}, seed={SEED}) — epistemic only")
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, out_path)


def make_epistemic_profile_figure(dataset: str, out_path: Path, methods=None) -> None:
    """`std_epi(x)` for every method on one log-scaled axis.

    The panel figures cannot show what this shows. On `sin_homo` the
    epistemic term spans 0.009 (ensemble, mid-data) to 0.89 (laplace, at
    the extrapolation edge) — two orders of magnitude. On any linear y
    axis wide enough to contain laplace and gp, the bbb/mcd/ensemble
    curves are three flat lines within a hair of the x axis; on one narrow
    enough to resolve those three, laplace and gp leave the figure. A log
    axis is not a presentational preference here, it is the only scale on
    which "gp grows 40x and ensemble grows 5x" is a visible statement
    rather than an assertion the reader has to take on trust from a table.

    Plotting the epistemic std DIRECTLY, rather than as a band around the
    predictive mean, also removes the mean's own shape from the picture —
    on the panel figures a band's apparent width changes as the mean
    curves through it, which is exactly the confound when the question is
    whether the width itself responds to data density.

    `map` is excluded by default: its `var_epistemic` is identically zero
    by construction (`src/methods/base.py`), which has no place on a log
    axis and is not a measurement of anything.
    """
    if methods is None:
        methods = [m for m in METHOD_ORDER if m != "map"]

    train = _load_train_points(dataset)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    _shade_training_support(ax, train.x)

    for method in methods:
        df = _load_predictions(dataset, method)
        ax.semilogy(df.x, df.std_epi, color=METHOD_COLORS[method],
                    linewidth=1.6, label=METHOD_LABELS[method])

    ax.set_xlim(*X_RANGE)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\sigma_{\mathrm{epistemic}}(x)$  (log scale)")
    ax.set_title(f"Epistemic uncertainty profile ({dataset}, seed={SEED})")
    ax.grid(True, which="both", axis="y", alpha=0.25, linewidth=0.5)
    # Legend below the axes, not inside them: ensemble's curve dips to its
    # global minimum (~0.007 on sin_homo) right where an in-axes legend
    # would sit, and on a log axis there is no spare whitespace to move it
    # into without changing the limits and flattening everything else.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8, ncol=3, frameon=False)
    _save(fig, out_path)


def make_overlay_figure(dataset: str, methods, out_path: Path, title: str = None) -> None:
    """Several methods' bands overlaid on one set of axes — for direct
    visual comparison of band width/shape at a glance. Shared (X_RANGE,
    Y_RANGE) via the same per-method data as `_draw_posterior`, drawn
    without going through it (one shared "train"/"true f(x)" pair, not one
    per method).
    """
    train = _load_train_points(dataset)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(train.x, train.y, s=6, color="black", alpha=0.3, zorder=5, label="train")

    y_true_plotted = False
    for method in methods:
        df = _load_predictions(dataset, method)
        color = METHOD_COLORS[method]
        ax.fill_between(
            df.x, df["mean"] - 2 * df.std_total, df["mean"] + 2 * df.std_total,
            color=color, alpha=0.15, linewidth=0,
        )
        ax.plot(df.x, df["mean"], color=color, linewidth=1.6, label=METHOD_LABELS[method])
        if not y_true_plotted:
            ax.plot(df.x, df.y_true, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="true f(x)")
            y_true_plotted = True

    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower left", fontsize=8)
    if title is not None:
        ax.set_title(title)
    _save(fig, out_path)


def make_grid_figure(dataset: str, methods, out_path: Path, title: str = None) -> None:
    """One panel per method, up to 3 columns, forced shared axes.
    `make_comparison_grid`-equivalent, generalised to any method subset
    (not only all six) — brief section 10's 2x3 grid is `make_grid_figure(
    dataset, METHOD_ORDER, ...)`.
    """
    n = len(methods)
    ncols = min(3, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, method in zip(axes, methods):
        _draw_posterior(ax, dataset, method, title=METHOD_LABELS[method])
    for ax in axes[len(methods):]:
        ax.axis("off")
    if title is not None:
        fig.suptitle(title)
    _save(fig, out_path)


def make_img2_2_function_samples(out_path: Path, n_samples: int = 8) -> None:
    """"próbki funkcji z posteriora, rozchodzące się poza [0,6]" (brief section 10)
    — a chapter-2, method-agnostic illustration, drawn with GP: the exact
    posterior, and the only method here with a literal `.sample_y()` (a true
    function draw), unlike the T stochastic forward passes MCD/BBB/ensemble use.
    Retrains a fresh GP (cheap, ~1s) — `sample_y` draws are not part of the
    `predictions_1d` schema, so this one figure is not redrawable from CSVs
    alone the way the others are (see module docstring).
    """
    from src.data import SYNTHETIC_DATASETS
    from src.methods.gp import GPMethod

    ds = SYNTHETIC_DATASETS["sin_homo"](seed=SEED)
    method = GPMethod().fit(ds.X_train, ds.y_train, seed=SEED)
    x_grid = np.linspace(*X_RANGE, 300).reshape(-1, 1)
    samples = method.gp.sample_y(x_grid, n_samples=n_samples, random_state=SEED)  # (300, n_samples)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for i in range(n_samples):
        label = "posterior function samples" if i == 0 else None
        ax.plot(x_grid.ravel(), samples[:, i], color=METHOD_COLORS["gp"], alpha=0.5, linewidth=1.0, label=label)
    ax.plot(x_grid.ravel(), np.sin(x_grid.ravel()), color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="true f(x)")
    train = _load_train_points("sin_homo")
    ax.scatter(train.x, train.y, s=6, color="black", alpha=0.35, zorder=5, label="train")
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Posterior function samples (GP, sin_homo, seed={SEED})")
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, out_path)


def make_img2_3_nested_bands(out_path: Path) -> None:
    """"dwa zagnieżdżone pasma: aleatoryczne i całkowite" (brief section 10)
    — inner = +/-2*std_alea, outer = +/-2*std_total. GP, same reasoning as
    img2_2: a chapter-2-general illustration via the exact reference method,
    before chapter 3 introduces the five approximations.
    """
    train = _load_train_points("sin_homo")
    df = _load_predictions("sin_homo", "gp")
    color = METHOD_COLORS["gp"]

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.fill_between(
        df.x, df["mean"] - 2 * df.std_total, df["mean"] + 2 * df.std_total,
        color=color, alpha=0.15, linewidth=0, label=r"$\pm2\sigma_{\mathrm{total}}$",
    )
    ax.fill_between(
        df.x, df["mean"] - 2 * df.std_alea, df["mean"] + 2 * df.std_alea,
        color=color, alpha=0.35, linewidth=0, label=r"$\pm2\sigma_{\mathrm{aleatoric}}$",
    )
    ax.plot(df.x, df["mean"], color=color, linewidth=1.6, label="predictive mean")
    ax.plot(df.x, df.y_true, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="true f(x)")
    ax.scatter(train.x, train.y, s=6, color="black", alpha=0.35, zorder=5, label="train")
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Aleatoric vs total uncertainty (GP, sin_homo, seed={SEED})")
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, out_path)
