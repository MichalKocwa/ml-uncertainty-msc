"""Compact per-experiment tables for writing chapters 4 and 5.

`results/` holds one row per fit: 681 for E2's main table, 4836 for E3, 72 574
across `depth_exploration/`. Those files are the evidence and stay the source
of truth, but a chapter needs the aggregate — the mean over the repeat axis
(split, seed or repeat) and the standard error that says how much of the
difference between two methods is noise rather than signal.

Only E3 had such an aggregate written to a file (`e3_summary.py`). This script
builds one for every other experiment and copies through the tables that were
already aggregated, so that `results/summary/` alone is enough to write from.

Two rules this script keeps:

1. **Every number is computed from a CSV in `results/`.** Nothing is typed in,
   nothing is carried over from a note. Re-running it after a re-run of an
   experiment reproduces the tables from the new rows.
2. **The repeat axis is never silently averaged away.** Each aggregated table
   carries `n` — how many fits went into the cell — so a cell backed by three
   seeds cannot be mistaken for one backed by twenty splits.

`sem` is the standard error of the mean, `std(ddof=1) / sqrt(n)`; it is empty
where `n = 1`, because one observation has no spread. This is the convention
the reference tables in `docs/datasets.md` use, not the standard deviation.

Known defect handled here: `results/e6c_laplace_structure.csv` has six rows
with 9 fields against a 21-field header (the failure path in
`experiments/e6c_laplace_structure.py` returns a dict without the metric keys,
and `append_generic_csv` does not align columns). Read naively, those rows put
a timestamp in `prior_precision` and a commit hash in `n_parameters`. This
script aggregates the `status == "ok"` rows only and reports the failed cells
by count, which is what E6c is about anyway.

Usage:
  python scripts/make_summary_tables.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = RESULTS_DIR / "summary"

# Author's decision, from experiments/e2_uci.py.
METHOD_ORDER = ("mcd", "map", "laplace", "ensemble", "bbb", "gp")
DATASET_ORDER = ("yacht", "energy", "concrete", "wine_quality_red", "kin8nm",
                 "power_plant", "sin_homo", "sin_hetero", "sin_gap")

# Written by every experiment, of no use in a table.
BOOKKEEPING = ("experiment_id", "timestamp", "git_commit", "torch_threads",
               "workers", "failure")


@dataclass
class Table:
    """One output table. `group_by` empty means pass the source through."""
    name: str
    source: str
    title: str
    note: str
    group_by: Sequence[str] = ()
    metrics: Sequence[str] = ()
    carry: Sequence[str] = ()          # constant within a group, copied as-is
    row_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    extra: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame] | None = None
    drop: Sequence[str] = field(default_factory=tuple)


def _ok_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == "ok"].copy()


def _e6c_failure_counts(raw: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """E6c's result is partly *which* cells produced no posterior at all."""
    counts = (raw.assign(failed=(raw["status"] == "failed").astype(int))
                 .groupby(["dataset", "hessian_structure", "prior_precision_mode"],
                          observed=True)["failed"].sum().rename("n_failed"))
    return out.merge(counts, on=["dataset", "hessian_structure",
                                 "prior_precision_mode"], how="left")


TABLES: list[Table] = [
    Table(
        name="e1_synthetic", source="e1_synthetic.csv",
        title="E1 — synthetic datasets, six methods, seed 0",
        note="One fit per cell (`seed=0`), so there is no repeat axis and no "
             "`sem`. Split by `split_type`: metrics over the training range, "
             "over the extrapolation region, and — for `sin_gap` — inside the "
             "hole. `qice` is empty throughout (unimplemented, brief O2).",
        drop=("config_id", "split_index", "init_seed", "qice"),
    ),
    Table(
        name="e1_sigma_calibration", source="e1_sigma_calibration.csv",
        title="E1 — aleatoric calibration (L0)",
        note="`sigma_rmse` between the fitted noise scale and the generating "
             "`sigma_fn`, per dataset and method.",
    ),
    Table(
        name="e2_main", source="e2_uci.csv",
        title="E2 — main UCI table, mean ± sem over 20 splits",
        note="**The table chapter 5 is built on.** Averaged over the 20 fixed "
             "train/test splits of the reference protocol, so a cell is "
             "directly comparable with the published numbers. GP is absent on "
             "`power_plant` (skipped, D5) and `kin8nm` (abandoned after 31.1 "
             "min) — see `e2_gp_skipped.csv`. `config_id` records the epoch "
             "count chosen per dataset (D30).",
        group_by=("dataset", "method"),
        metrics=("rmse", "mae", "ll", "nll", "crps", "picp95", "mpiw95",
                 "interval_score95", "ece_reg", "mean_var_aleatoric",
                 "mean_var_epistemic", "epi_ratio"),
        carry=("config_id", "n_train", "n_test"),
    ),
    Table(
        name="e2_cost", source="e2_cost.csv",
        title="E2 — cost, split 0, sequential, one thread",
        note="The cost column for chapter 5. Measured in an exclusive run; "
             "`train_time_s` in `e2_uci.csv` is a by-product of a parallel "
             "run and is not a measurement (D23a).",
        drop=("split_index",),
    ),
    Table(
        name="e3_gap_summary", source="e3_gap_summary.csv",
        title="E3 — gap splits on UCI, already aggregated by e3_summary.py",
        note="Copied through unchanged. Rows are grouped by `group`: "
             "`real_gap` is the result E3 exists to report, `negative_control` "
             "and `control` are the two controls, `duplicate_axis` the "
             "dimensions counted once. Mixing the groups into one average is "
             "not reportable — see part E.1b/E.1c of the notes.",
    ),
    Table(
        name="e3_gap_by_dimension", source="e3_gap_ratio.csv",
        title="E3 — per dimension, mean ± sem over the splits in `n`",
        note="`e3_gap_summary.csv` averages within a group and drops which "
             "feature each hole belongs to. Part E.1c needs that back: its "
             "Table 1 names the clean axes individually (`energy` dim 0, "
             "`concrete` dims 0, 5, 6) and its Table 2 is explicitly per "
             "dimension **with no average taken**. `gap_leak_fraction` is the "
             "quantity that assigns a row to a group: below 0.05 a real hole, "
             "0.05-0.9 partial, near 1 the negative control. `variant=gap` "
             "removes the middle third by rank, `variant=control` removes as "
             "many rows at random. **Duplicate axes are not merged here** — "
             "`energy` dims 0 and 1 are the same hole (E.1b), so do not "
             "average them together.\n\n"
             "**`n` is not the same everywhere, by design, not by omission:** "
             "`variant=gap` runs the full 20 splits, `variant=control` runs 5 "
             "(`CONTROL_SPLITS = 5`, author's decision 2026-08-31, exposed as "
             "`--control-splits` in `experiments/e3_gap_split.py:82`), and "
             "`sin_gap` runs 3 seeds. A `sem` computed from 5 or 3 "
             "observations is a weak estimate — read it as an order of "
             "magnitude, and do not compare a control's `sem` with a gap's as "
             "if they carried the same weight.",
        group_by=("dataset", "method", "variant", "dimension"),
        metrics=("gap_leak_fraction", "epi_gap_ratio", "mean_std_epi_gap_in",
                 "mean_std_epi_gap_out", "rmse_gap_in", "rmse_gap_out",
                 "ll_gap_in", "ll_gap_out", "picp95_gap_in", "picp95_gap_out"),
        carry=("feature_n_distinct", "n_train_full", "n_removed"),
    ),
    Table(
        name="input_dropout_ablation", source="input_dropout_ablation.csv",
        title="Input dropout on against off, MC dropout, mean ± sem over 3 seeds",
        note="The measurement behind D15, the departure from gal2016's "
             "\"dropout before every weighted layer\". With input dropout the "
             "fitted noise variance sits an order of magnitude above the truth "
             "(`true_var_aleatoric = 0.01`): the network covers the batches "
             "that dropout corrupted to `x = 0` by inflating `log_sigma2`, so "
             "the aleatoric term absorbs an artefact of the mask. "
             "**`sigma^2` is fitted here, not pinned** — E1 pins it on these "
             "two datasets, and a pinned variance cannot inflate, so the "
             "question would be unaskable under the E1 protocol.",
        group_by=("dataset", "input_dropout"),
        metrics=("mean_var_aleatoric", "var_aleatoric_ratio_to_true",
                 "rmse_in_range", "ll_in_range", "mpiw95_in_range",
                 "picp95_in_range", "mean_std_epi_in_range",
                 "mean_std_epi_extrapolation"),
        carry=("dropout_p", "epochs", "n_parameters", "true_var_aleatoric"),
    ),
    Table(
        name="uci_epochs_chosen", source="uci_epochs_sweep_final_chosen.csv",
        title="Chosen epoch count per dataset and method (D30)",
        note="The evidence behind the epoch column of chapter 4. The number "
             "used in a run is the **maximum over methods** for that dataset "
             "(D30), not the per-method value in `chosen_epochs`. "
             "`chosen_at_measured_edge` and `better_above_ceiling` flag where "
             "the chosen value sits at the top of the searched grid rather "
             "than at a found optimum — `yacht`/`mcd` is such a case.",
    ),
    Table(
        name="ensemble_epochs_sweep", source="ensemble_epochs_sweep.csv",
        title="Deep ensembles — epochs against member disagreement, mean ± sem over seeds",
        note="The measurement behind O8: how much of the ensemble's spread is "
             "a function of how long its members train.",
        group_by=("dataset", "epochs"),
        metrics=("train_time_s", "rmse_in_range", "ll_in_range",
                 "picp95_in_range", "mpiw95_in_range",
                 "median_std_epi_in_range", "rmse_extrapolation",
                 "ll_extrapolation"),
    ),
    Table(
        name="bbb_elbo_samples_cost", source="bbb_elbo_samples_cost.csv",
        title="BBB — cost of the ELBO sample count (D14d/D14e)",
        note="Why `elbo_samples=32` is used at the synthetic scale and 1 "
             "elsewhere: `projected_hours_20_splits` is the reason the "
             "setting does not transfer to the UCI table.",
        drop=("split_index", "init_seed"),
    ),
    Table(
        name="bbb_posterior_diagnostic", source="bbb_posterior_diagnostic.csv",
        title="BBB — how much of the posterior stayed at the prior",
        note="Overpruning check: the fraction of weights whose posterior "
             "variance never moved away from the prior. Background for D14b "
             "and for reading BBB's epistemic term.",
        group_by=("dataset", "protocol", "epochs", "elbo_samples"),
        metrics=("frac_var_below_prior_weights", "frac_var_below_prior_all"),
    ),
    Table(
        name="gp_duplicate_collapse", source="gp_duplicate_collapse.csv",
        title="GP — the same fit with and without repeated rows, mean ± sem over 20 splits",
        note="The second half of the `wine_quality_red` story (part E.1). "
             "`collapsed` is the **design factor**, not a measurement: each "
             "dataset is fitted twice per split, once exactly as E2 does "
             "(`collapsed=False`) and once with repeated feature rows "
             "collapsed to their first occurrence (`collapsed=True`). The "
             "diagnostic fit is never a reported result — it exists to show "
             "what the repeated rows do to the fitted noise level. "
             "`n_repeats_removed` is what collapsing removed on that split "
             "(0 throughout on `energy`, up to 208 on `wine_quality_red`). "
             "Read together with `duplicate_ll_diagnostic`, which splits the "
             "log-likelihood by duplicate/unique test rows.",
        group_by=("dataset", "collapsed"),
        metrics=("n_train", "n_repeats_removed", "noise_level", "length_scale",
                 "constant_value", "log_marginal_likelihood", "rmse", "ll",
                 "picp95", "mpiw95", "n_warnings", "lbfgs_abnormal",
                 "bound_warnings"),
        carry=("n_restarts_optimizer", "n_train_full", "n_test"),
    ),
    Table(
        name="gp_convergence_diagnostic", source="gp_convergence_diagnostic.csv",
        title="GP — optimiser warnings and the fitted noise floor, mean ± sem over splits",
        note="Whether sklearn's convergence warnings coincide with the "
             "collapsed noise level. See also `gp_restarts_check.csv` "
             "(2 rows, not reproduced here) for the restart-count check.",
        group_by=("dataset", "n_restarts_optimizer"),
        metrics=("n_warnings", "lbfgs_abnormal", "bound_warnings",
                 "constant_value", "length_scale", "noise_level"),
    ),
    Table(
        name="logvar_clamp_contact_check", source="logvar_clamp_contact_check.csv",
        title="Would the old log-variance bound have been binding? (D29)",
        note="`would_hit_old_bound` per dataset and method — the check that "
             "the current clamp `[-12, 6]` is inactive on everything used.",
        drop=("epochs",),
    ),
    Table(
        name="e5_depth", source="e5_depth.csv",
        title="E5 — depth ablation, mean ± sem over 3 seeds",
        note="`depth=0` is the GP, which has no hidden layers. Three seeds "
             "only, so `sem` is a weak estimate — read it as an order of "
             "magnitude, not a confidence interval. Three rows are missing "
             "(`sin_gap`, seed 2); `n` shows where.",
        group_by=("dataset", "method", "depth"),
        metrics=("n_parameters", "rmse_in_range", "ll_in_range",
                 "picp95_in_range", "mpiw95_in_range",
                 "median_std_epi_in_range", "rmse_extrapolation",
                 "ll_extrapolation", "picp95_extrapolation",
                 "mpiw95_extrapolation", "median_std_epi_extrapolation",
                 "ratio_at_m2", "ratio_at_8", "gap_ratio", "train_time_s"),
    ),
    Table(
        name="e6a_mc_samples", source="e6a_mc_samples.csv",
        title="E6a — number of MC samples T, mean ± sem over repeats",
        note="Justifies `T = 100` (Figure 3.7): the point at which the "
             "variance estimator stops moving. `mcd` and `bbb` only — the "
             "other four methods have no sampling axis.",
        group_by=("dataset", "method", "T"),
        metrics=("mpiw95", "picp95", "ll", "rmse", "mean_std_epistemic",
                 "mean_var_epistemic"),
    ),
    Table(
        name="e6c_laplace_structure", source="e6c_laplace_structure.csv",
        title="E6c — Laplace covariance structure x prior mode, mean ± sem over 3 seeds",
        note="**Only `status == \"ok\"` rows are averaged.** `n` counts those, "
             "`n_failed` the cells whose Hessian would not factorise — for "
             "some (structure, prior) pairs the failure *is* the result "
             "(P5, P6). The failed rows are also malformed in the source CSV "
             "(9 fields against a 21-field header); their `status` and "
             "`failure` text are correct, everything after is shifted.",
        group_by=("dataset", "hessian_structure", "prior_precision_mode"),
        metrics=("prior_precision", "mpiw95", "mpiw95_in_range",
                 "mpiw95_extrapolation", "picp95_in_range",
                 "picp95_extrapolation", "rmse_in_range", "ll_in_range",
                 "mean_std_epi_in_range", "mean_std_epi_extrapolation",
                 "epi_extrap_ratio"),
        row_filter=_ok_rows, extra=_e6c_failure_counts,
    ),
    Table(
        name="e6d_activation", source="e6d_activation.csv",
        title="E6d — ReLU vs TanH, mean ± sem over 3 seeds",
        note="Closes P4. The motivation is in part E of the notes: a ReLU "
             "network's Jacobian is a step function, so the linearised "
             "Laplace variance jumps at the activation kinks; TanH has a "
             "continuous Jacobian and the mechanism disappears.",
        group_by=("dataset", "method", "activation"),
        metrics=("n_parameters", "mean_std_epi_in_range",
                 "mean_std_epi_extrapolation", "epi_extrap_ratio",
                 "std_epi_at_edge", "mpiw95_in_range", "mpiw95_extrapolation",
                 "picp95_in_range", "picp95_extrapolation", "rmse_in_range",
                 "ll_in_range"),
    ),
    Table(
        name="p13_gal_protocol", source="p13_gal_protocol.csv",
        title="P13 — gal2016's protocol reproduced, mean ± sem over 20 folds",
        note="The literature-validation run: 4000 epochs, ReLU, input "
             "dropout, per-fold grid over `p` and `tau`. `rmse_difference` "
             "and `ll_difference` are paired against the published value on "
             "the same fold, which is why they are the number to quote and "
             "not the difference of the two means.",
        group_by=("dataset",),
        metrics=("chosen_dropout_p", "chosen_tau", "validation_rmse",
                 "validation_ll", "rmse", "ll", "rmse_difference",
                 "ll_difference"),
        carry=("epochs", "hidden", "n_train", "n_validation", "n_test",
               "published_rmse", "published_ll", "t_samples"),
    ),
    Table(
        name="literature_comparison", source="literature_comparison.csv",
        title="Our numbers against the published ones, mean ± sem over splits",
        note="`difference` is paired per split. Present for `mcd` only — the "
             "BBB and deep-ensemble reference rows still have to be read off "
             "the papers by hand (notes, O6).",
        group_by=("dataset", "method", "metric"),
        metrics=("own_value", "difference"),
        carry=("published_value",),
    ),
    Table(
        name="depth_exploration", source="depth_exploration_summary.csv",
        title="Depth x width exploration, seed 0",
        note="The sweep behind the decision to keep `1x50` (D-topologia-E5): "
             "six methods x six (depth, width) configurations x two datasets. "
             "**One seed only** (`seed=0`) — this is an exploration, not the "
             "ablation; E5 is the one with three seeds and a `sem`. Copied "
             "through unchanged. The per-grid-point curves it was computed "
             "from (`results/depth_exploration/`, 72 574 rows) are not needed "
             "to quote these numbers.",
    ),
    Table(
        name="p13_dropout_diagnostic", source="p13_dropout_diagnostic.csv",
        title="P13 — dropout rate sweep, mean ± sem over splits",
        note="Why our MC dropout differed from the published numbers before "
             "the protocol was equalised.",
        group_by=("dataset", "dropout_p"),
        metrics=("rmse", "ll"),
    ),
    Table(
        name="duplicate_ll_diagnostic", source="duplicate_ll_diagnostic.csv",
        title="Repeated rows in wine_quality_red, mean ± sem over splits",
        note="The GP's log-likelihood on `wine` split by whether the test row "
             "also appears in training. The reporting decision (two numbers "
             "plus a paragraph) is in part E.1 of the notes.",
        group_by=("dataset", "method"),
        metrics=("ll_all", "dup_ll", "uniq_ll", "ll_gap_dup_minus_uniq",
                 "ll_shift_from_dups", "dup_n", "uniq_n", "dup_rmse",
                 "uniq_rmse"),
    ),
    Table(
        name="e0_gp_scaling", source="e0_gp_scaling_norestart.csv",
        title="E0 — exact GP fit time against N",
        note="P14's slope, measured in an exclusive run. The memory columns "
             "quoted in section 4.1 of the notes come from an earlier run and "
             "have no CSV — they need recomputing before they can be plotted.",
        drop=("repeat",),
    ),
    Table(
        name="expectations_check", source="expectations_check.csv",
        title="P1-P14 — predictions against what the runs produced",
        note="Copied through unchanged. Every verdict is computed from a CSV "
             "by `experiments/expectations_check.py`; none is typed in. The "
             "four refuted predictions are chapter-5 material, not defects.",
    ),
    Table(
        name="epistemic_growth", source="epistemic_growth.csv",
        title="Growth of epistemic uncertainty away from the data (L1/L3)",
        note="Read off E1's saved predictions by `scripts/epistemic_growth.py`.",
    ),
    Table(
        name="mcd_dropout_sweep", source="mcd_dropout_sweep.csv",
        title="MC dropout — dropout rate sweep on the synthetic data",
        note="The digest of `mcd_dropout_profiles.csv` (12 000 per-grid-point "
             "rows), which is not needed to quote these numbers.",
    ),
    Table(
        name="logvar_clamp", source="logvar_clamp_diagnostic_summary.csv",
        title="Was the log-variance clamp binding? (D29)",
        note="The digest of `logvar_clamp_diagnostic.csv` (72 000 per-step "
             "rows). See also `logvar_clamp_contact_check.csv`.",
    ),
    Table(
        name="dataset_duplicates", source="dataset_duplicates.csv",
        title="Repeated rows per UCI dataset",
        note="Counted directly from the fetched data files.",
    ),
    Table(
        name="thread_determinism_check", source="thread_determinism_check.csv",
        title="How much a run depends on the thread count",
        note="Why `set_seed` pins `torch.set_num_threads(1)`. Same seed, "
             "different thread count, different network.",
    ),
    Table(
        name="e2_gp_skipped", source="e2_gp_skipped.csv",
        title="E2 — the GP cells that were not run, and why",
        note="Recorded rather than left blank, so an empty cell in the main "
             "table has a stated reason.",
    ),
]


def _sem(s: pd.Series) -> float:
    return s.std(ddof=1) / (len(s) ** 0.5) if s.notna().sum() > 1 else float("nan")


def _order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by the repo's method and dataset order where those columns exist."""
    by = []
    for col, order in (("dataset", DATASET_ORDER), ("method", METHOD_ORDER)):
        if col in df.columns:
            known = [v for v in order if v in set(df[col])]
            rest = sorted(set(df[col]) - set(known))
            df[col] = pd.Categorical(df[col], categories=known + rest, ordered=True)
            by.append(col)
    other = [c for c in df.columns if c not in by and c.endswith(("_mean", "_sem")) is False]
    return df.sort_values(by + [c for c in other if c in ("depth", "T", "hidden")]) if by else df


def build(table: Table) -> tuple[pd.DataFrame, str]:
    src = RESULTS_DIR / table.source
    raw = pd.read_csv(src)
    df = table.row_filter(raw) if table.row_filter else raw.copy()

    if not table.group_by:
        out = df.drop(columns=[c for c in (*BOOKKEEPING, *table.drop)
                               if c in df.columns])
        out = out.dropna(axis=1, how="all")
        return _order(out).reset_index(drop=True), f"pass-through, {len(out)} rows"

    metrics = [m for m in table.metrics if m in df.columns]

    # A metric column can arrive as text when the source CSV has a malformed
    # row (E6c's failure path — see the module docstring). Coerce explicitly
    # and say what was lost, rather than dropping the column or averaging
    # whatever parsed.
    for m in metrics:
        if not pd.api.types.is_numeric_dtype(df[m]):
            coerced = pd.to_numeric(df[m], errors="coerce")
            lost = int(df[m].notna().sum() - coerced.notna().sum())
            if lost:
                print(f"    ! {table.name}.{m}: {lost} value(s) are not numeric "
                      f"and are excluded from the mean")
            df[m] = coerced

    grouped = df.groupby(list(table.group_by), observed=True, dropna=False)

    out = grouped.size().rename("n").reset_index()
    for m in metrics:
        agg = grouped[m].agg(mean="mean", sem=_sem)
        out = out.merge(agg.rename(columns={"mean": f"{m}_mean", "sem": f"{m}_sem"}),
                        on=list(table.group_by), how="left")
    for c in table.carry:
        if c not in df.columns:
            continue
        vals = grouped[c].agg(lambda s: s.iloc[0] if s.nunique(dropna=False) == 1 else "varies")
        out = out.merge(vals.rename(c), on=list(table.group_by), how="left")

    if table.extra:
        out = table.extra(raw, out)
    out = out.dropna(axis=1, how="all")
    return _order(out).reset_index(drop=True), f"{len(df)} rows -> {len(out)}"


def _fmt(v) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, str):
        return v
    if float(v).is_integer() and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.4g}"


def to_markdown(df: pd.DataFrame) -> str:
    """Render with `mean ± sem` collapsed into one cell, which is how the
    number is written in the thesis."""
    cols, merged = [], {}
    for c in df.columns:
        if c.endswith("_mean") and f"{c[:-5]}_sem" in df.columns:
            base = c[:-5]
            cols.append(base)
            merged[base] = [
                _fmt(m) if pd.isna(s) else f"{_fmt(m)} ± {_fmt(s)}"
                for m, s in zip(df[c], df[f"{base}_sem"])
            ]
        elif c.endswith("_sem") and f"{c[:-4]}_mean" in df.columns:
            continue
        else:
            cols.append(c)
            merged[c] = [_fmt(v) for v in df[c]]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for i in range(len(df)):
        lines.append("| " + " | ".join(merged[c][i] for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = [
        "# Result tables for chapters 4 and 5",
        "",
        "Generated by `scripts/make_summary_tables.py` from the CSVs in",
        "`results/`. Do not edit by hand — re-run the script instead.",
        "",
        "`x ± y` is mean ± standard error of the mean over the repeat axis",
        "named in each section; `n` is how many fits went into the cell. A",
        "cell with no `±` had `n = 1`.",
        "",
    ]
    total = 0
    for table in TABLES:
        if not (RESULTS_DIR / table.source).exists():
            print(f"  SKIP {table.name}: {table.source} not found")
            continue
        out, how = build(table)
        path = OUT_DIR / f"{table.name}.csv"
        out.to_csv(path, index=False)
        total += path.stat().st_size
        print(f"  {table.name:26s} {how:22s} -> {path.relative_to(RESULTS_DIR.parent)}")
        md += [f"## {table.title}", "",
               f"Source: `results/{table.source}` ({how}).", "",
               table.note, "", to_markdown(out), ""]

    md_path = OUT_DIR / "TABLES.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    total += md_path.stat().st_size
    print(f"\n  {len(TABLES)} tables, {total / 1024:.1f} KB in "
          f"{OUT_DIR.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
