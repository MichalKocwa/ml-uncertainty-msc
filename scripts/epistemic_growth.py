"""Epistemic-growth table from `results/predictions_1d/` — no retraining.

Recomputes, for every (dataset, method) already saved, how much the
epistemic std grows from inside the training support out to the
extrapolation edges (and, for `sin_gap`, into the held-out gap).

**Why this script exists: the single-point reference it replaces was
wrong.** D14d/D14g quoted `std_epi(x=8) / std_epi(x=3)`. On `sin_homo` at
seed=0 MC dropout's profile is flat at 0.13-0.15 across the whole grid
apart from one narrow dip to 0.040 at x~3.2 — and x=3 landed inside it,
turning a real ratio of ~1.2x into a reported 3.9x and supporting a
"MCD's uncertainty grows ~4x in extrapolation" reading that the profile
does not. `src.metrics.epistemic_growth` anchors on the MEDIAN over the
training support instead, which no single narrow feature can move.

Reads only `results/predictions_1d/`, same contract as `src/plotting.py`:
these are diagnostics computed on saved predictions, never a retrain.

Usage:
  python scripts/epistemic_growth.py [--datasets sin_homo,sin_gap] [--csv]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data import SYNTHETIC_DATASETS, range_masks
from src.metrics import epistemic_growth, median_by_mask
from src.results import RESULTS_DIR
from src.style import METHOD_ORDER

PREDICTIONS_DIR = RESULTS_DIR / "predictions_1d"
PROBES = (-2.0, 8.0)


def rows_for(dataset: str):
    out = []
    for method in METHOD_ORDER:
        path = PREDICTIONS_DIR / f"{dataset}_{method}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        masks = range_masks(dataset, df.x.values)

        if float(np.max(df.std_epi.values)) == 0.0:
            # map: var_epistemic is identically zero by construction
            # (src/methods/base.py). Reported as such rather than as a
            # ratio of 0/0 — "no epistemic term" is the measurement.
            out.append(dict(dataset=dataset, method=method, median_in_range=0.0,
                            ratio_at_m2=np.nan, ratio_at_8=np.nan, gap_ratio=np.nan,
                            std_epi_at_m2=0.0, std_epi_at_8=0.0, note="zero by construction"))
            continue

        g = epistemic_growth(df.std_epi.values, df.x.values, masks["in_range"], probes=PROBES)
        row = dict(
            dataset=dataset, method=method,
            median_in_range=g["median_in_range"],
            std_epi_at_m2=g["std_epi_at_m2"], std_epi_at_8=g["std_epi_at_8"],
            ratio_at_m2=g["ratio_at_m2"], ratio_at_8=g["ratio_at_8"],
            note="",
        )
        if "in_gap" in masks:
            row["gap_ratio"] = median_by_mask(df.std_epi.values, masks["in_gap"]) / g["median_in_range"]
        else:
            row["gap_ratio"] = np.nan
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=",".join(SYNTHETIC_DATASETS.keys()))
    parser.add_argument("--csv", action="store_true", help="also write results/epistemic_growth.csv")
    args = parser.parse_args()

    all_rows = []
    for dataset in args.datasets.split(","):
        rows = rows_for(dataset)
        all_rows.extend(rows)
        print(f"\n{dataset}  (reference = median std_epi over the training support)")
        print(f"  {'method':10s} {'median_in':>10s} {'@x=-2':>9s} {'@x=8':>9s} "
              f"{'ratio(-2)':>10s} {'ratio(8)':>9s} {'gap_ratio':>10s}")
        for r in rows:
            print(f"  {r['method']:10s} {r['median_in_range']:10.4f} {r['std_epi_at_m2']:9.4f} "
                  f"{r['std_epi_at_8']:9.4f} {r['ratio_at_m2']:10.2f} {r['ratio_at_8']:9.2f} "
                  f"{r['gap_ratio']:10.2f}" + (f"   ({r['note']})" if r["note"] else ""))

    if args.csv:
        out = RESULTS_DIR / "epistemic_growth.csv"
        pd.DataFrame(all_rows).to_csv(out, index=False)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
