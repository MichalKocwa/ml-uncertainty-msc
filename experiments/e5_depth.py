"""E5 — depth ablation (brief section 9), `depth in {1, 2}` arm.

The brief specifies E5 as "depth ablation, depth in {1, 2, 4} x all methods
x 2 datasets". This script runs the `{1, 2}` arm, which is the one that
answers the question raised by `foong2020` (docs/chapter4_notes.md D14j):
that paper's non-expressiveness theorem covers exactly the single-hidden-
layer case this project's backbone uses, for exactly the two methods whose
uncertainty has the wrong shape here (mean-field VI and MC dropout), and
its universality result starts at two hidden layers. `depth=4` remains for
Etap 5 to complete E5.

**This does NOT change the default.** `backbone.DEFAULT_DEPTH` stays 1, and
`depth=1` is bit-identical to the backbone as it stood before the parameter
existed (`tests/test_methods.py::test_depth_one_is_bit_identical_to_pre_depth_backbone`),
so this sweep's `depth=1` rows are a re-derivation of the E1 numbers rather
than a separate configuration — which also makes them a check on the harness.

**Success criteria, fixed in advance (author, 2026-08-26)** — recorded here
so the result cannot be read against criteria chosen after seeing it:
  (a) `gap_ratio` for bbb and mcd on `sin_gap` clearly above 1
      (currently 1.06 and 0.68). `sin_gap` is the deciding dataset, not
      `sin_homo`: its geometry is the one foong2020's theorem speaks about
      (uncertainty BETWEEN separated clusters), whereas `sin_homo` is
      extrapolation beyond a single island, which the theorem does not cover.
  (b) Laplace's needle does not return — `max |var_epi[i+1] - var_epi[i]|`
      over the eval grid stays at the D-width-E5 scale (~0.007-0.011), not
      the N=50 scale that motivated N=250 and float64.
  (c) In-range fit does not degrade materially.
If (a) holds, a backbone change gets considered. If not, this is a negative
result with theory behind it — the strongest of the set, because it is the
only one that tested a route the literature actually predicted might work.

`use_cache=False` (not exposed as a flag): this script records
`train_time_s`, and D-width-E5 documents a cache hit silently reducing
Laplace's recorded training time to a state-dict load, because its backbone
is the same code MAP had just trained and cached in the same run.
**Run it with nothing else competing for the CPU** — D14k's timings were
spoiled exactly that way.

Writes results/e5_depth.csv incrementally (one row per method x depth x
dataset x seed), so an interrupted run keeps what it already measured.

Usage:
  python experiments/e5_depth.py
  python experiments/e5_depth.py --depths 2 --methods bbb,mcd --datasets sin_gap --seeds 0
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data import SYNTHETIC_DATASETS
from src.methods import METHODS
from src.results import RESULTS_DIR, git_commit_short, now_iso
from src.seeding import set_seed
from src.sweeps import aggregate, e1_method_kwargs, evaluate_on_synthetic

DEFAULT_DEPTHS = (1, 2)
DEFAULT_METHODS = ("map", "gp", "bbb", "mcd", "laplace", "ensemble")
DEFAULT_DATASETS = ("sin_homo", "sin_gap")
DEFAULT_SEEDS = (0, 1, 2)

# `gp` has no hidden layers at all, so "depth" is not a property it has. It is
# fitted ONCE per (dataset, seed) and recorded with this sentinel rather than
# duplicated across the depth grid — a duplicated row would read as two
# independent measurements of a quantity depth cannot move.
GP_DEPTH_SENTINEL = 0

NOT_APPLICABLE = "-"

OUT_PATH = RESULTS_DIR / "e5_depth.csv"


KEY_COLUMNS = ["method", "depth", "dataset", "seed"]


def write_merged(rows) -> None:
    """Upsert `rows` into OUT_PATH, keyed on (method, depth, dataset, seed).

    NOT a blind overwrite. This sweep is long enough that it gets run in
    pieces — one dataset now, the missing seeds later — and a plain
    `to_csv` makes the second run silently destroy the first run's rows
    (it did: E5's `--datasets sin_gap` follow-up wiped the completed
    `sin_homo` half, which was only recoverable from a manual copy).
    Re-running a key that already exists replaces it, so a redone seed
    updates in place rather than appearing twice.
    """
    df = pd.DataFrame(rows)
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        incoming_keys = set(map(tuple, df[KEY_COLUMNS].values))
        keep = [tuple(v) not in incoming_keys for v in existing[KEY_COLUMNS].values]
        df = pd.concat([existing[keep], df], ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(KEY_COLUMNS).to_csv(OUT_PATH, index=False)


def needle_metric(var_epi, x) -> dict:
    """`max |var_epi[i+1] - var_epi[i]|` over the sorted eval grid.

    The diagnostic D-width-E5 and the Laplace module docstring both use for
    the `var_epistemic` needle spikes: a single-grid-step jump in the
    linearised predictive variance. Grid-adjacent differences, not a
    derivative, because the spikes being detected are ~0.02 wide on a grid
    of spacing ~0.01 — narrow enough that any smoothed estimate would hide
    exactly what this is looking for.
    """
    order = np.argsort(x)
    values, xs = np.asarray(var_epi)[order], np.asarray(x)[order]
    deltas = np.abs(np.diff(values))
    i = int(np.argmax(deltas))
    return {"max_abs_delta_var_epi": float(deltas[i]), "max_abs_delta_var_epi_at_x": float(xs[i])}


def laplace_conditioning(method) -> dict:
    """Condition number of Laplace's posterior precision, via symmetric
    eigenvalues rather than `np.linalg.cond`'s SVD: the matrix is symmetric
    positive definite by construction (GGN + prior_precision*I), `eigvalsh`
    exploits that, and the smallest eigenvalue is itself the quantity of
    interest — a posterior precision going near-singular is the mechanism
    behind the needle spikes, so reporting lambda_min separately says more
    than the ratio alone.

    Only defined for `hessian_structure="full"`, where `posterior_precision`
    is a dense matrix; `kron`/`diag` return factored/vector forms.
    """
    if getattr(method, "la", None) is None or method.hessian_structure != "full":
        return {}
    matrix = method.la.posterior_precision
    matrix = matrix.detach().cpu().numpy() if hasattr(matrix, "detach") else np.asarray(matrix)
    eigenvalues = np.linalg.eigvalsh(matrix)
    lam_min, lam_max = float(eigenvalues[0]), float(eigenvalues[-1])
    return {
        "posterior_precision_lambda_min": lam_min,
        "posterior_precision_lambda_max": lam_max,
        "posterior_precision_cond": lam_max / lam_min if lam_min > 0 else float("inf"),
    }


def run_one(method_name: str, depth: int, dataset: str, seed: int) -> dict:
    set_seed(seed)
    ds = SYNTHETIC_DATASETS[dataset](seed=seed)

    kwargs = e1_method_kwargs(method_name, dataset)
    if method_name != "gp":
        kwargs["depth"] = depth
    method = METHODS[method_name](**kwargs)

    t0 = time.perf_counter()
    method.fit(ds.X_train, ds.y_train, seed=seed, use_cache=False)
    train_time_s = time.perf_counter() - t0
    pred = method.predict(ds.X_eval)

    row = dict(
        method=method_name,
        depth=GP_DEPTH_SENTINEL if method_name == "gp" else depth,
        dataset=dataset, seed=seed,
        n_parameters=method.n_parameters,
        train_time_s=train_time_s,
        **evaluate_on_synthetic(pred, ds, dataset),
        **needle_metric(pred.var_epistemic, ds.X_eval.ravel()),
        timestamp=now_iso(), git_commit=git_commit_short(),
    )
    if method_name == "laplace":
        row.update(laplace_conditioning(method))
    return row


def depth_label(depth) -> str:
    return NOT_APPLICABLE if int(depth) == GP_DEPTH_SENTINEL else str(int(depth))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--depths", type=str, default=",".join(map(str, DEFAULT_DEPTHS)))
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    args = parser.parse_args()

    depths = [int(v) for v in args.depths.split(",")]
    methods = args.methods.split(",")
    datasets = args.datasets.split(",")
    seeds = [int(v) for v in args.seeds.split(",")]

    rows = []
    for dataset in datasets:
        for seed in seeds:
            for method_name in methods:
                # gp is depth-invariant: fit once per (dataset, seed), not once per depth
                method_depths = [depths[0]] if method_name == "gp" else depths
                for depth in method_depths:
                    row = run_one(method_name, depth, dataset, seed)
                    rows.append(row)
                    write_merged(rows)  # incremental, upsert not overwrite

                    extra = ""
                    if method_name == "laplace":
                        extra = (" needle={:.5f} cond={:.3e}".format(
                            row["max_abs_delta_var_epi"],
                            row.get("posterior_precision_cond", float("nan"))))
                    gap = " gap={:5.2f}".format(row["gap_ratio"]) if "gap_ratio" in row else ""
                    print("  {:9s} seed={} depth={:>2s} {:9s} p={:5d} med_epi={:.4f} "
                          "r(-2)={:6.2f} r(8)={:6.2f}{} rmse_in={:.4f} ll_in={:6.3f} ({:.1f}s){}".format(
                              dataset, seed, depth_label(row["depth"]), method_name,
                              row["n_parameters"], row["median_std_epi_in_range"],
                              row["ratio_at_m2"], row["ratio_at_8"], gap,
                              row["rmse_in_range"], row["ll_in_range"], row["train_time_s"], extra),
                          flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print("\nwrote {}".format(OUT_PATH), flush=True)
    report(df, datasets)


def report(df, datasets) -> None:
    cols = ["median_std_epi_in_range", "ratio_at_m2", "ratio_at_8", "gap_ratio",
            "rmse_in_range", "ll_in_range", "picp95_in_range",
            "rmse_extrapolation", "ll_extrapolation", "picp95_extrapolation",
            "max_abs_delta_var_epi", "train_time_s"]
    summary = aggregate(df, ["dataset", "method", "depth"], [c for c in cols if c in df.columns])

    for dataset in datasets:
        sub = summary[summary.dataset == dataset].sort_values(["method", "depth"])
        if sub.empty:
            continue
        print("\n=== {} — mean +/- std across seeds ===".format(dataset), flush=True)
        header = "  {:9s} {:>5s} {:>15s} {:>15s} {:>15s}".format(
            "method", "depth", "med_epi", "ratio(-2)", "ratio(8)")
        has_gap = "gap_ratio_mean" in sub.columns and sub.gap_ratio_mean.notna().any()
        if has_gap:
            header += " {:>15s}".format("gap_ratio")
        print(header, flush=True)
        for _, r in sub.iterrows():
            line = "  {:9s} {:>5s} {:7.4f}+/-{:6.4f} {:7.2f}+/-{:6.2f} {:7.2f}+/-{:6.2f}".format(
                r.method, depth_label(r.depth),
                r.median_std_epi_in_range_mean, r.median_std_epi_in_range_std,
                r.ratio_at_m2_mean, r.ratio_at_m2_std,
                r.ratio_at_8_mean, r.ratio_at_8_std)
            if has_gap:
                line += " {:7.2f}+/-{:6.2f}".format(r.gap_ratio_mean, r.gap_ratio_std)
            print(line, flush=True)

        print("  {:9s} {:>5s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s}".format(
            "method", "depth", "rmse_in", "ll_in", "picp_in", "rmse_ex", "ll_ex", "picp_ex"), flush=True)
        for _, r in sub.iterrows():
            print("  {:9s} {:>5s} {:7.4f}+/-{:6.4f} {:7.3f}+/-{:6.3f} {:7.3f}+/-{:6.3f} "
                  "{:7.4f}+/-{:6.4f} {:7.3f}+/-{:6.3f} {:7.3f}+/-{:6.3f}".format(
                      r.method, depth_label(r.depth),
                      r.rmse_in_range_mean, r.rmse_in_range_std,
                      r.ll_in_range_mean, r.ll_in_range_std,
                      r.picp95_in_range_mean, r.picp95_in_range_std,
                      r.rmse_extrapolation_mean, r.rmse_extrapolation_std,
                      r.ll_extrapolation_mean, r.ll_extrapolation_std,
                      r.picp95_extrapolation_mean, r.picp95_extrapolation_std), flush=True)

    lap = df[df.method == "laplace"]
    if not lap.empty:
        print("\n=== laplace diagnostics (criterion b) ===", flush=True)
        print("  {:9s} {:>5s} {:>22s} {:>24s} {:>14s}".format(
            "dataset", "depth", "max|dvar_epi|", "cond(post. precision)", "lambda_min"), flush=True)
        for (dataset, depth), g in lap.groupby(["dataset", "depth"]):
            cond = g.posterior_precision_cond if "posterior_precision_cond" in g else pd.Series([np.nan])
            lmin = g.posterior_precision_lambda_min if "posterior_precision_lambda_min" in g else pd.Series([np.nan])
            print("  {:9s} {:5d} {:10.6f}+/-{:9.6f} {:11.3e}+/-{:10.3e} {:14.4f}".format(
                dataset, int(depth),
                g.max_abs_delta_var_epi.mean(), g.max_abs_delta_var_epi.std(ddof=1),
                cond.mean(), cond.std(ddof=1), lmin.mean()), flush=True)

    print("\n=== training time (exclusive run, no competing jobs) ===", flush=True)
    for (method_name, depth), g in df.groupby(["method", "depth"]):
        print("  {:9s} depth={:>2s} p={:5d} {:7.2f}s +/- {:.2f}".format(
            method_name, depth_label(depth), int(g.n_parameters.mean()),
            g.train_time_s.mean(), g.train_time_s.std(ddof=1)), flush=True)


if __name__ == "__main__":
    main()
