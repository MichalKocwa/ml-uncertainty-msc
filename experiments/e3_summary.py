"""E3's two tables: real holes, and the dimensions where there is no hole to find.

`results/e3_gap_ratio.csv` holds one row per fit, over all `d` dimensions of
each dataset. Averaging `epi_gap_ratio` over those `d` mixes two different
things, and the mixture is not a compromise between them — it is meaningless:

  * **Real gaps** (`gap_leak_fraction < 0.05`): removing the middle third by
    rank leaves no training row with a comparable feature value, so a test
    point in the band really is unsupported. `epi_gap_ratio > 1` here is the
    result E3 exists to report.
  * **Degenerate dimensions** (`gap_leak_fraction >= 0.05`): the feature has
    so few distinct values that rows identical in it stay in training. There
    is no hole, so `epi_gap_ratio ~ 1` is the CORRECT answer, for every
    method — which makes these dimensions a negative control. A method
    showing growth here is showing a defect, not a capability (author's
    instruction, 2026-08-31).

The 0.05 threshold is a reporting split, not a filter: every dimension is
computed, stored and printed. Nothing is dropped.

**Duplicate axes are counted once.** `energy`'s features 0 and 1 (relative
compactness, surface area) are a bijection on this dataset — 12 distinct
pairs for 12 distinct values, correlation -0.992 — so their middle thirds are
the SAME set of rows and their `epi_gap_ratio`s are identical by
construction. Averaging both would weight that single hole twice. Concrete's
three clean dimensions overlap at 0.28-0.37, i.e. chance level for thirds, so
they are genuinely different holes.

Usage:
  python experiments/e3_summary.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data import load_uci_raw
from src.results import RESULTS_DIR

RATIO_PATH = RESULTS_DIR / "e3_gap_ratio.csv"
OUT_PATH = RESULTS_DIR / "e3_gap_summary.csv"
LEAK_THRESHOLD = 0.05
METHOD_ORDER = ("map", "mcd", "laplace", "ensemble", "bbb", "gp")

# `energy`'s feature 1 is a deterministic function of feature 0 on this
# dataset, so the two produce the same middle third; keep the first.
DUPLICATE_AXES = {("energy", 1)}


def _duplicate_check() -> None:
    """Re-derive the duplicate rather than trusting the constant above."""
    from experiments.e3_gap_split import middle_third_membership

    X, _ = load_uci_raw("energy")
    same = bool((middle_third_membership(X, 0) == middle_third_membership(X, 1)).all())
    print(f"energy dims 0 and 1 share a middle third: {same} "
          f"(correlation {np.corrcoef(X[:, 0], X[:, 1])[0, 1]:+.4f}) — "
          f"{'dim 1 excluded from the main result' if same else 'BOTH KEPT: the duplicate is gone, update DUPLICATE_AXES'}")


def main():
    df = pd.read_csv(RATIO_PATH)
    _duplicate_check()

    df["axis"] = list(zip(df.dataset, df.dimension))
    df["duplicate_axis"] = df.axis.isin(DUPLICATE_AXES)
    df["real_gap"] = df.gap_leak_fraction < LEAK_THRESHOLD

    gap = df[(df.variant == "gap") & ~df.duplicate_axis]
    control = df[df.variant == "control"]

    for label, sub in (("REAL GAPS (gap_leak_fraction < 0.05)", gap[gap.real_gap]),
                       ("NEGATIVE CONTROL (leak >= 0.05: no hole to find)", gap[~gap.real_gap])):
        if sub.empty:
            continue
        print(f"\n{label}")
        dims = sub.groupby("dataset").dimension.unique().to_dict()
        print("  dimensions: " + "; ".join(f"{k} {sorted(v)}" for k, v in dims.items()))
        table = sub.pivot_table(index="method", columns="dataset", values="epi_gap_ratio",
                                aggfunc=["mean", "sem"])
        print(table.reindex([m for m in METHOD_ORDER if m in set(sub.method)]).round(3).to_string())

    if not control.empty:
        print("\nRANDOM-REMOVAL CONTROL (same number of rows deleted at random)")
        table = control.pivot_table(index="method", columns="dataset", values="epi_gap_ratio",
                                    aggfunc=["mean", "sem"])
        print(table.reindex([m for m in METHOD_ORDER if m in set(control.method)]).round(3).to_string())

    synthetic = df[(df.variant == "gap") & (df.dataset == "sin_gap")]
    if not synthetic.empty:
        print("\nsin_gap (hole by construction, 3 seeds)")
        print(synthetic.groupby("method").epi_gap_ratio.agg(["mean", "sem"])
              .reindex([m for m in METHOD_ORDER if m in set(synthetic.method)]).round(3).to_string())

    summary = (
        df.assign(group=np.where(df.variant == "control", "control",
                                 np.where(df.duplicate_axis, "duplicate_axis",
                                          np.where(df.real_gap, "real_gap", "negative_control"))))
        .groupby(["dataset", "method", "group"])
        .agg(epi_gap_ratio_mean=("epi_gap_ratio", "mean"),
             epi_gap_ratio_sem=("epi_gap_ratio", "sem"),
             rmse_gap_in=("rmse_gap_in", "mean"), rmse_gap_out=("rmse_gap_out", "mean"),
             ll_gap_in=("ll_gap_in", "mean"), ll_gap_out=("ll_gap_out", "mean"),
             picp95_gap_in=("picp95_gap_in", "mean"), picp95_gap_out=("picp95_gap_out", "mean"),
             n_fits=("epi_gap_ratio", "size"))
        .reset_index()
    )
    summary.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
