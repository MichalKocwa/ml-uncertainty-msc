"""O4/D12 — how many epochs does the main UCI table train for, per dataset?

`batch_size = 128` is shared by every NN method (D18), so a single declared
"2000 epochs" means 34x more optimizer steps on `power_plant` (n_train=8611,
68 steps/epoch, 136k steps) than on `yacht` (n_train=277, 3 steps/epoch,
6k steps). That is the same defect D18 removed from the batch-size axis,
reintroduced through the epoch axis: one declared constant covering two
different optimisation regimes.

Decision (author): the unit stays EPOCHS, not gradient steps — Hernandez-
Lobato & Adams' and Gal's protocol declares epochs, and P13 compares our
numbers against that protocol; a fixed step budget would give
`power_plant` 89 passes over the data instead of 2000 and would cost the
only external correctness check the implementation has. The NUMBER of
epochs is determined per dataset, here.

**Selection rule: the MAXIMUM over methods, per dataset ("variant B",
author's decision 2026-08-27).** A first pass measured `map` alone, on the
argument that it is the cheapest proxy. That failed on exactly the datasets
where the question mattered: on `concrete`, `map` chose 100 epochs and
`mcd` 1000, a 10x disagreement, because `map` overfits there (validation
NLL rising while validation RMSE stands still — `sigma^2` collapsing) while
dropout and the KL term keep the other two improving. Since brief section 4
requires ONE epoch count shared by every method on a dataset, some method's
optimum has to become everyone's budget; "nobody is undertrained" is the
only rule that does not have to be defended by explaining why one
particular method got to set it. The per-method disagreement is kept and
reported — it is a measured fact for chapter 4, not an assumption.

Measurement protocol:
  - ONE split (index 0) of each of the six datasets
  - 20% of the training fold held out as validation, EXCLUSIVELY for this
    measurement (see `_inner_split` — it never enters E2)
  - grid {5, 10, 20, 50, 100, 200, 500, 1000, 2000}. The low end exists
    because the first pass found `wine_quality_red`'s optimum at ~15 epochs
    and `concrete`'s at ~60, i.e. off the bottom of the original grid.
  - criterion: the smallest grid value whose validation NLL is within an
    ABSOLUTE 0.02 nats of that method's grid minimum. Absolute, not "1% of
    the minimum": at NLL ~ -1.3 a 1% rule gives a 0.013-nat tolerance,
    below the between-seed spread, so it would pick 2000 almost every time.
  - three methods: `map`, `mcd`, `bbb` (the two extremes of regularisation
    plus the baseline); the dataset's epoch count is the max of the three.

**One instrumented run per (dataset, method, seed), not one per grid
point.** The training trajectory is deterministic given the seed, so the
model after epoch `e` of a 2000-epoch run is the model a separate
`epochs=e` run would produce. Verified against the first pass, which did
refit per grid point: the two agree to every printed digit. This makes the
9-point grid cost the same as its largest entry instead of their sum.

Cost limits (author's decision — without them BBB alone is 4-5 hours):
  - BBB at `elbo_samples=8`, not 32. This looks for where the curve flattens,
    not for final quality, and a noisier gradient estimator can only need
    the same number of epochs or more — so the answer is an upper bound.
  - BBB on one seed; `map` and `mcd` on three.
  - BBB on `kin8nm`/`power_plant` stops at 500 epochs if the curve is flat
    within 0.02 nats over the preceding 200 (both have flat `map` curves,
    so the remaining 1500 epochs would very likely add nothing). An
    early-stopped run's answer is marked as a lower bound in the output.

Not named `e{N}_`: those ids belong to the brief's own experiments
(section 9). This is a diagnostic answering open question O4.

Writes results/uci_epochs_sweep{,_chosen,_combined}.csv and
figures/uci_epochs_sweep.png.

Usage:
  python experiments/uci_epochs_sweep.py --out-suffix _variantB
  python experiments/uci_epochs_sweep.py --datasets concrete --methods map,mcd,bbb
"""
import argparse
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src import metrics
from src.data import UCI_SPEC, load_uci_raw, uci_split_indices
from src.methods.backbone import DEFAULT_BATCH_SIZE
from src.methods.bbb import BBBMethod
from src.methods.map import MAPMethod
from src.methods.mcd import MCDropoutMethod
from src.plotting import FIGURES_DIR
from src.results import RESULTS_DIR, git_commit_short, now_iso
from src.seeding import set_seed

DEFAULT_EPOCHS_GRID = (5, 10, 20, 50, 100, 200, 500, 1000, 2000)
DEFAULT_DATASETS = ("yacht", "energy", "concrete", "wine_quality_red", "kin8nm", "power_plant")
DEFAULT_METHODS = ("map", "mcd", "bbb")
DEFAULT_SEEDS = (0, 1, 2)
SPLIT_INDEX = 0

# Absolute tolerance in nats on the validation NLL (author's decision, see
# module docstring). Not a fraction of the minimum.
NLL_TOLERANCE_NATS = 0.02

VALIDATION_FRACTION = 0.2
# Fixed, independent of the init seed: the validation rows must be the same
# for every epoch and every seed, or the curves would not be comparable
# point to point.
VALIDATION_SPLIT_SEED = 0

# Per-method seed budget (see the cost limits in the module docstring).
METHOD_SEEDS = {"map": (0, 1, 2), "mcd": (0, 1, 2), "bbb": (0,)}

# BBB's ELBO-sample count HERE ONLY. Not a statement about E2's value —
# that is decision 2 (D14e), still open. See the cost limits above for why
# this measurement runs at 8 and why the resulting epoch count is an upper
# bound rather than an estimate.
BBB_ELBO_SAMPLES = 8

# Early stop, per the cost limits: (method, dataset) pairs allowed to stop
# at ABORT_EPOCH when the curve has been flat within ABORT_TOL nats over
# the preceding ABORT_WINDOW epochs.
EARLY_ABORT = {("bbb", "kin8nm"), ("bbb", "power_plant")}
ABORT_EPOCH = 500
ABORT_WINDOW = 200
ABORT_TOL = NLL_TOLERANCE_NATS
# Extra evaluation points inside the abort window, so the trend is judged on
# more than the single grid entry at 500.
ABORT_PROBES = (300, 400)

# The rule tests the TREND over the second half of the window (400 -> 500),
# not the spread across the whole window. The spread version was tried first
# and let through a curve that was still descending: BBB on `kin8nm` moved
# 0.2693 -> 0.2475 between epochs 200 and 500, monotonically, while every
# pairwise gap inside the window stayed under 0.02 — so "flat" was satisfied
# by a curve with no flat part in it. A still-falling curve has not
# converged however small each step is, which is what a trend test asks and
# a spread test does not.


# Hard ceiling on the epoch count a dataset may be assigned, applied when
# the tables are built (the measurements above the ceiling are still taken
# and still saved — they are the evidence for the ceiling, not waste).
#
# `yacht` (author's decision, 2026-08-28): `map` peaks at 500 and collapses
# afterwards (-1.00 at 500, -0.08 at 2000, +1.35 +/- 1.27 at 5000), while
# `mcd` was still improving at the end of every grid tried (-0.306 at 2000,
# -0.364 at 5000). With two methods moving in OPPOSITE directions the
# max-over-methods rule has no fixed point: every extension of the grid
# raises the shared budget and costs `map` more (its penalty against its own
# optimum goes 0.92 nats at 2000 to 2.35 at 5000), so the grid, not the
# data, would be setting the answer. 5000 was measured and NOT adopted;
# `yacht` is capped at 2000 — which is also BBB's genuine interior optimum
# there — and `mcd`'s optimum on `yacht` is recorded as unmeasured, a stated
# limitation of the protocol rather than a number pretending to be one.
EPOCH_CEILING = {"yacht": 2000}


class _EarlyStop(Exception):
    """Raised from the epoch callback to end a run early; caught in `run`."""


def _make_method(method_name: str, epochs: int):
    if method_name == "map":
        return MAPMethod(epochs=epochs)
    if method_name == "mcd":
        return MCDropoutMethod(epochs=epochs)
    if method_name == "bbb":
        return BBBMethod(epochs=epochs, elbo_samples=BBB_ELBO_SAMPLES)
    raise ValueError(f"unknown method '{method_name}'")


def _inner_split(name: str, split: int = SPLIT_INDEX):
    """Training fold of literature split `split`, cut 80/20 into a fit fold
    and a validation fold, each standardised on the FIT fold only.

    The scaler is re-fitted on the inner 80% rather than reused from
    `load_uci`: `load_uci` standardises `y` on the whole 90% training fold,
    and `y` enters the validation NLL directly, so reusing that scaler
    would bias the validation loss downwards — systematically, and harder
    on the small datasets.

    **The validation fold exists only here.** E2 trains on the full 90%
    training fold, per the literature protocol.
    """
    X, y = load_uci_raw(name)
    train_idx, _ = uci_split_indices(name, split)
    X_train, y_train = X[train_idx], y[train_idx]

    rng = np.random.RandomState(VALIDATION_SPLIT_SEED)
    perm = rng.permutation(len(X_train))
    n_val = int(round(VALIDATION_FRACTION * len(perm)))
    val_rows, fit_rows = perm[:n_val], perm[n_val:]

    X_fit, y_fit = X_train[fit_rows], y_train[fit_rows]
    X_val, y_val = X_train[val_rows], y_train[val_rows]

    x_scaler = StandardScaler().fit(X_fit)
    y_scaler = StandardScaler().fit(y_fit.reshape(-1, 1))
    return (
        x_scaler.transform(X_fit).astype(np.float64),
        y_scaler.transform(y_fit.reshape(-1, 1)).ravel().astype(np.float64),
        x_scaler.transform(X_val).astype(np.float64),
        y_scaler.transform(y_val.reshape(-1, 1)).ravel().astype(np.float64),
    )


def _predict_isolated(method, model, X_val):
    """`method.predict` on a mid-training `model`, with every RNG stream
    left exactly as it was found.

    Necessary, not defensive: `MCDropoutMethod.predict` and
    `BBBMethod.predict` both call `set_seed(self._seed)` (so that a
    prediction depends on the seed alone, not on ambient state), and MC
    dropout's training-time masks are drawn from that same global stream —
    reseeding it mid-run would change the rest of the trajectory and break
    the equivalence with separate per-grid-point fits that this whole
    approach rests on.
    """
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()
    was_training = model.training
    try:
        method.model = model
        return method.predict(X_val)
    finally:
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
        random.setstate(py_state)
        model.train(was_training)


def _run_one(name, method_name, seed, epochs_grid, max_epochs, X_fit, y_fit, X_val, y_val,
             allow_abort: bool = True):
    """One instrumented training run; one record per evaluated epoch."""
    steps_per_epoch = math.ceil(len(X_fit) / DEFAULT_BATCH_SIZE)
    may_abort = allow_abort and (method_name, name) in EARLY_ABORT
    probes = set(epochs_grid) | (set(ABORT_PROBES) | {ABORT_EPOCH} if may_abort else set())

    records = []

    def callback(epoch, model):
        if epoch not in probes:
            return
        pred = _predict_isolated(method, model, X_val)
        std = pred.std_total
        records.append(dict(
            epoch=epoch,
            on_grid=epoch in epochs_grid,
            val_nll=metrics.nll(y_val, pred.mean, std),
            val_ll=metrics.ll(y_val, pred.mean, std),
            val_rmse=metrics.rmse(y_val, pred.mean),
            mean_var_aleatoric=float(np.mean(pred.var_aleatoric)),
            mean_var_epistemic=float(np.mean(pred.var_epistemic)),
        ))
        if may_abort and epoch == ABORT_EPOCH:
            half = [r for r in records if ABORT_EPOCH - ABORT_WINDOW // 2 <= r["epoch"] <= ABORT_EPOCH]
            half.sort(key=lambda r: r["epoch"])
            if len(half) >= 2 and (half[0]["val_nll"] - half[-1]["val_nll"]) <= ABORT_TOL:
                raise _EarlyStop

    set_seed(seed)
    method = _make_method(method_name, max_epochs)
    t0 = time.perf_counter()
    aborted = False
    try:
        method.fit(X_fit, y_fit, seed=seed, use_cache=False, epoch_callback=callback)
    except _EarlyStop:
        aborted = True
    train_time_s = time.perf_counter() - t0

    # probe rows are kept: they are what the abort rule was judged on, and
    # dropping them would leave that decision unauditable. Everything
    # downstream selects `on_grid` explicitly.
    df = pd.DataFrame(records)
    df.insert(0, "dataset", name)
    df.insert(1, "method", method_name)
    df.insert(2, "seed", seed)
    df["n_fit"] = len(X_fit)
    df["n_val"] = len(X_val)
    df["n_train_full"] = UCI_SPEC[name]["n_train"]
    df["batch_size"] = DEFAULT_BATCH_SIZE
    df["steps_per_epoch"] = steps_per_epoch
    df["total_steps"] = steps_per_epoch * df["epoch"]
    df["elbo_samples"] = BBB_ELBO_SAMPLES if method_name == "bbb" else None
    df["run_aborted"] = aborted
    df["max_epoch_measured"] = int(df.loc[df["on_grid"], "epoch"].max())
    df["run_train_time_s"] = train_time_s
    print(f"  {name:17s} {method_name:4s} seed={seed}  "
          f"{'ABORTED at ' + str(int(df['epoch'].max())) if aborted else 'full ' + str(max_epochs)}"
          f"  best_val_nll={df['val_nll'].min():8.4f}  ({train_time_s / 60:5.1f} min)", flush=True)
    return df


def run(datasets, methods, epochs_grid, seeds_by_method, max_epochs, allow_abort=True) -> pd.DataFrame:
    frames = []
    for name in datasets:
        X_fit, y_fit, X_val, y_val = _inner_split(name)
        print(f"\n{name}: n_fit={len(X_fit)} n_val={len(X_val)} "
              f"steps/epoch={math.ceil(len(X_fit) / DEFAULT_BATCH_SIZE)}")
        for method_name in methods:
            for seed in seeds_by_method[method_name]:
                frames.append(_run_one(name, method_name, seed, epochs_grid, max_epochs,
                                       X_fit, y_fit, X_val, y_val, allow_abort=allow_abort))
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = now_iso()
    df["git_commit"] = git_commit_short()
    return df


def choose_epochs(df: pd.DataFrame, tolerance: float = NLL_TOLERANCE_NATS) -> pd.DataFrame:
    """Smallest grid value whose MEAN validation NLL is within `tolerance`
    nats of that (dataset, method)'s grid minimum. Off-grid probe rows
    (`on_grid == False`) exist only for the abort rule and never take part
    in the selection."""
    df = df[df["on_grid"]]
    out = []
    for (dataset, method), sub in df.groupby(["dataset", "method"], sort=False):
        ceiling = EPOCH_CEILING.get(dataset)
        above_ceiling = sub[sub["epoch"] > ceiling] if ceiling else sub.iloc[:0]
        if ceiling:
            sub = sub[sub["epoch"] <= ceiling]
        curve = sub.groupby("epoch")["val_nll"].agg(["mean", "std"]).sort_index()
        best = curve["mean"].min()
        chosen = int(curve.index[curve["mean"] <= best + tolerance].min())
        n_train_full = int(sub["n_train_full"].iloc[0])
        steps_per_epoch_full = math.ceil(n_train_full / DEFAULT_BATCH_SIZE)
        max_measured = int(sub["epoch"].max())
        out.append(dict(
            dataset=dataset, method=method,
            n_train_full=n_train_full,
            steps_per_epoch_full=steps_per_epoch_full,
            chosen_epochs=chosen,
            total_steps_full=steps_per_epoch_full * chosen,
            best_epochs=int(curve["mean"].idxmin()),
            best_val_nll=float(best),
            chosen_val_nll=float(curve["mean"].loc[chosen]),
            excess_nats=float(curve["mean"].loc[chosen] - best),
            n_seeds=int(sub["seed"].nunique()),
            run_aborted=bool(sub["run_aborted"].any()),
            max_epoch_measured=max_measured,
            # The criterion cannot look past what it was allowed to see, so a
            # choice sitting on the top of that range is normally a lower
            # bound — UNLESS the ceiling truncated a range we did measure
            # and the method was worse above it, which makes the choice a
            # genuine interior optimum (BBB on `yacht`: best at 2000, worse
            # at 5000).
            chosen_at_measured_edge=bool(
                chosen == max_measured
                and not (len(above_ceiling) and above_ceiling.groupby("epoch")["val_nll"].mean().min() >= best)
            ),
            epoch_ceiling=ceiling,
            # measured above the ceiling and better there: the ceiling, not
            # the data, is what stopped this method's count from rising
            better_above_ceiling=bool(
                len(above_ceiling) and above_ceiling.groupby("epoch")["val_nll"].mean().min() < best
            ),
        ))
    return pd.DataFrame(out)


def combine_over_methods(chosen: pd.DataFrame) -> pd.DataFrame:
    """One epoch count per dataset: the maximum over methods (variant B)."""
    out = []
    for dataset, sub in chosen.groupby("dataset", sort=False):
        winner = sub.loc[sub["chosen_epochs"].idxmax()]
        by_method = {r.method: int(r.chosen_epochs) for r in sub.itertuples()}
        spread = max(by_method.values()) / max(1, min(by_method.values()))
        out.append(dict(
            dataset=dataset,
            n_train_full=int(winner["n_train_full"]),
            steps_per_epoch_full=int(winner["steps_per_epoch_full"]),
            epochs=int(winner["chosen_epochs"]),
            total_steps_full=int(winner["total_steps_full"]),
            set_by=winner["method"],
            method_spread=spread,
            is_lower_bound=bool(winner["chosen_at_measured_edge"]),
            capped=bool(sub["better_above_ceiling"].any()),
            **{f"epochs_{m}": v for m, v in by_method.items()},
        ))
    return pd.DataFrame(out)


def consolidate(paths) -> pd.DataFrame:
    """Union of several result CSVs, later files winning on (dataset,
    method, seed, epoch).

    The final chapter-4 table is assembled from more than one run: the main
    sweep, plus the re-measurements the author ordered where the first pass
    hit the edge of its grid (`yacht` extended to 5000; `kin8nm`/BBB rerun
    to 2000 without the early stop). Recomputing the selection from the
    union — rather than editing numbers into a table by hand — keeps the
    published table derived from files that are all still on disk.
    """
    frames = [pd.read_csv(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    # `on_grid` was added after the first sweep was already on disk; those
    # files stored grid rows only, so a missing value means True.
    if "on_grid" not in df.columns:
        df["on_grid"] = True
    df["on_grid"] = df["on_grid"].fillna(True).astype(bool)
    before = len(df)
    df = df.drop_duplicates(subset=["dataset", "method", "seed", "epoch"], keep="last")
    print(f"consolidated {len(paths)} files: {before} rows -> {len(df)} after "
          f"dropping {before - len(df)} superseded")
    return df.reset_index(drop=True)


def make_figure(df: pd.DataFrame, chosen: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    from src.style import METHOD_COLORS, METHOD_LABELS

    df = df[df["on_grid"]]
    datasets = list(dict.fromkeys(df["dataset"]))
    methods = list(dict.fromkeys(df["method"]))
    ncols = min(3, len(datasets))
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), squeeze=False)

    for ax, name in zip(axes.ravel(), datasets):
        sub_ds = df[df.dataset == name]
        for method_name in methods:
            sub = sub_ds[sub_ds.method == method_name]
            if sub.empty:
                continue
            curve = sub.groupby("epoch")["val_nll"].agg(["mean", "std"]).sort_index()
            colour = METHOD_COLORS[method_name]
            ax.errorbar(curve.index, curve["mean"], yerr=curve["std"].fillna(0.0), marker="o",
                        ms=4, color=colour, capsize=3, label=METHOD_LABELS[method_name])
            ax.axhline(curve["mean"].min() + NLL_TOLERANCE_NATS, color=colour, ls=":", lw=1)
            pick = chosen[(chosen.dataset == name) & (chosen.method == method_name)]
            if not pick.empty:
                e = int(pick["chosen_epochs"].iloc[0])
                ax.plot([e], [curve["mean"].loc[e]], marker="*", ms=15, color=colour,
                        mec="black", mew=0.6, ls="none", zorder=5)
        ax.set_xscale("log")
        ax.set_xticks(sorted(df["epoch"].unique()))
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(f"{name} (n_train={UCI_SPEC[name]['n_train']})")
        ax.set_xlabel("epochs")
        ax.set_ylabel("validation NLL [nats, standardised y]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes.ravel()[len(datasets):]:
        ax.set_visible(False)

    fig.suptitle(f"Validation NLL vs epochs (split 0, dotted = that method's min + "
                 f"{NLL_TOLERANCE_NATS} nats, star = chosen)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--epochs", type=str, default=",".join(map(str, DEFAULT_EPOCHS_GRID)))
    parser.add_argument("--seeds", type=str, default="",
                        help="override the per-method seed budget for ALL methods, e.g. '0'. "
                             "Default: map/mcd on 0,1,2 and bbb on 0 (see METHOD_SEEDS).")
    parser.add_argument("--out-suffix", type=str, default="")
    parser.add_argument("--no-abort", action="store_true",
                        help="ignore EARLY_ABORT and always run the full grid")
    parser.add_argument("--consolidate", type=str, default="",
                        help="skip training: build the tables and figure from the union of these "
                             "result CSVs (comma-separated, later files win on duplicate rows)")
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    methods = args.methods.split(",")
    epochs_grid = tuple(int(v) for v in args.epochs.split(","))
    max_epochs = max(epochs_grid)
    if args.seeds:
        override = tuple(int(v) for v in args.seeds.split(","))
        seeds_by_method = {m: override for m in methods}
    else:
        seeds_by_method = {m: METHOD_SEEDS[m] for m in methods}

    if args.consolidate:
        df = consolidate([RESULTS_DIR / f for f in args.consolidate.split(",")])
    else:
        df = run(datasets, methods, epochs_grid, seeds_by_method, max_epochs,
                 allow_abort=not args.no_abort)
    stem = f"uci_epochs_sweep{args.out_suffix}"
    df.to_csv(RESULTS_DIR / f"{stem}.csv", index=False)

    chosen = choose_epochs(df)
    chosen.to_csv(RESULTS_DIR / f"{stem}_chosen.csv", index=False)
    combined = combine_over_methods(chosen)
    combined.to_csv(RESULTS_DIR / f"{stem}_combined.csv", index=False)
    make_figure(df, chosen, FIGURES_DIR / f"{stem}.png")
    print(f"\nwrote {RESULTS_DIR / f'{stem}.csv'} (+ _chosen, _combined) and "
          f"{FIGURES_DIR / f'{stem}.png'}")

    print(f"\nvalidation NLL, mean over seeds:")
    for (dataset, method), sub in df.groupby(["dataset", "method"], sort=False):
        curve = sub.groupby("epoch")["val_nll"].agg(["mean", "std"]).sort_index()
        print(f"\n{dataset} / {method}")
        for epoch, r in curve.iterrows():
            std = "" if np.isnan(r["std"]) else f" +/- {r['std']:.4f}"
            print(f"  {epoch:5d}  {r['mean']:8.4f}{std}")

    print(f"\nper-method choice (min + {NLL_TOLERANCE_NATS} nats rule):")
    print(f"  {'dataset':18s} {'method':6s} {'epochs':>6s} {'best@':>6s} {'val_nll':>9s} "
          f"{'excess':>7s} {'seeds':>5s} {'note':>12s}")
    for _, r in chosen.iterrows():
        note = ("CAPPED" if r["better_above_ceiling"]
                else "LOWER BOUND" if r["chosen_at_measured_edge"]
                else "aborted" if r["run_aborted"] else "")
        print(f"  {r['dataset']:18s} {r['method']:6s} {r['chosen_epochs']:6d} {r['best_epochs']:6d} "
              f"{r['chosen_val_nll']:9.4f} {r['excess_nats']:7.4f} {r['n_seeds']:5d} {note:>12s}")

    print("\nCHAPTER 4 TABLE — max over methods (variant B):")
    print(f"  {'dataset':18s} {'n_train':>7s} {'steps/ep':>8s} {'epochs':>6s} {'steps':>8s} "
          f"{'set by':>7s} {'spread':>7s}")
    for _, r in combined.iterrows():
        print(f"  {r['dataset']:18s} {r['n_train_full']:7d} {r['steps_per_epoch_full']:8d} "
              f"{r['epochs']:6d} {r['total_steps_full']:8d} {r['set_by']:>7s} "
              f"{r['method_spread']:6.1f}x {'  CAPPED' if r['capped'] else ''}")


if __name__ == "__main__":
    main()
