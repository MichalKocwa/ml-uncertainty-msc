"""Shared results-CSV writer (brief section 8).

One growing file per experiment, fixed schema, appended never overwritten
("Dopisywanie do CSV, nie nadpisywanie"). A single place for this so every
`experiments/e*.py` script writes rows the same way and the schema cannot
drift between them.
"""
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Section 8's fixed schema, verbatim column order.
RESULTS_SCHEMA = [
    "experiment_id", "dataset", "method", "config_id", "split_index", "init_seed",
    "n_train", "n_test", "split_type",
    "rmse", "mae", "ll", "nll", "crps", "picp95", "mpiw95", "interval_score95", "ece_reg", "qice",
    "mean_var_aleatoric", "mean_var_epistemic", "epi_ratio",
    "train_time_s", "predict_time_ms_per_1k", "n_parameters",
    # `torch_threads`: the intra-op thread count the row was computed under.
    # Added 2026-08-31 after `experiments/thread_determinism_check.py` measured
    # the same seed giving different networks at 1 and 8 threads (up to 6.75e-4
    # RMSE, 3.1e-3 LL). The column exists so that a mixed-configuration table
    # cannot happen again unnoticed: it is part of the configuration, exactly
    # like `init_seed`. `"n/a"` for methods that do not use torch at all (`gp`
    # is sklearn — its BLAS thread count is a different knob, and it was
    # measured insensitive to it anyway).
    "torch_threads",
    "timestamp", "git_commit",
]


def git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except Exception:
        return "unknown"


def append_result_row(path: Path, row: Dict[str, Any]) -> None:
    """Append one row to a results CSV. `row` may omit schema columns not
    yet computed (e.g. `qice`, unimplemented pending source verification —
    see `docs/chapter4_notes.md` O2): missing columns are written empty,
    never invented. Writes the header only if the file does not exist yet.
    """
    unknown = set(row) - set(RESULTS_SCHEMA)
    if unknown:
        raise ValueError(f"row has columns outside the section 8 schema: {sorted(unknown)}")

    # Appending a row whose column list differs from the file's own header
    # would silently shift every field after the first mismatch. That is
    # exactly what a schema change (e.g. adding `torch_threads`) does to files
    # written before it, so refuse rather than corrupt: an old file must be
    # renamed and regenerated, not appended to.
    if path.exists():
        existing_header = path.read_text(encoding="utf-8").split("\n", 1)[0].strip()
        if existing_header and existing_header.split(",") != RESULTS_SCHEMA:
            raise ValueError(
                f"{path.name} was written with a different schema than the current one "
                f"(its header has {len(existing_header.split(','))} columns, the schema has "
                f"{len(RESULTS_SCHEMA)}). Rename the old file and regenerate it rather than "
                f"appending — mixing schemas in one CSV shifts every later column."
            )

    full_row = {**{k: None for k in RESULTS_SCHEMA}, **row}
    df = pd.DataFrame([full_row], columns=RESULTS_SCHEMA)

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def upsert_csv(path: Path, df: "pd.DataFrame", key_columns) -> "pd.DataFrame":
    """Write `df` into `path`, replacing rows with matching keys and KEEPING the rest.

    For sweeps that write their whole table at once but can be run over a
    subset (`--datasets`, `--seeds`, `--methods`). A plain `to_csv` there is a
    silent data-loss bug: the narrow run's frame becomes the entire file.

    This is not hypothetical and it is not a one-off. `experiments/e5_depth.py`
    lost its completed `sin_homo` half to exactly this once, grew a local
    `write_merged()` in response — and then lost 63 of 66 rows AGAIN on
    2026-08-31, because `main()` still called `to_csv` directly. Hence one
    implementation, here, used by every sweep that has the shape.

    Returns the merged frame that was written, so a caller can report on the
    whole file rather than on its own slice.
    """
    df = pd.DataFrame(df)
    missing = [c for c in key_columns if c not in df.columns]
    if missing:
        raise ValueError(f"key columns absent from the new rows: {missing}")

    if path.exists():
        existing = pd.read_csv(path)
        if all(c in existing.columns for c in key_columns):
            incoming = set(map(tuple, df[list(key_columns)].astype(str).values))
            keep = [tuple(v) not in incoming
                    for v in existing[list(key_columns)].astype(str).values]
            df = pd.concat([existing[keep], df], ignore_index=True)
        else:
            raise ValueError(
                f"{path.name} has a different schema than the rows being written "
                f"(missing key columns {[c for c in key_columns if c not in existing.columns]}); "
                f"rename the old file rather than overwriting it"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(list(key_columns)).to_csv(path, index=False)
    return df


def append_predictions_1d(path: Path, x, mean, std_alea, std_epi, y_true) -> None:
    """`results/predictions_1d/{dataset}_{method}.csv` (section 8):
    `(x, mean, std_alea, std_epi, y_true)`, so figures can be redrawn
    without repeating training. Overwrites (not appends): one deterministic
    run per (dataset, method) in E1, re-running is meant to replace it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "x": x, "mean": mean, "std_alea": std_alea, "std_epi": std_epi, "y_true": y_true,
    }).to_csv(path, index=False)


def save_train_points(path: Path, x, y) -> None:
    """`results/predictions_1d/{dataset}_train.csv`: the training scatter
    plotted alongside every posterior figure. Not part of section 8's
    `predictions_1d` spec verbatim (x, mean, std_alea, std_epi, y_true), but
    without it `src/plotting.py` would need to regenerate the dataset via
    `src.data` to draw the training points — harmless for
    `experiments/e1_figures.py` (which already imports the library), but it
    would break `figures/redraw.py`'s "reads only results/predictions_1d/,
    imports no method or data code" contract. One file per dataset (shared
    across all six methods' panels), overwritten not appended, same
    reasoning as `append_predictions_1d`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": x, "y": y}).to_csv(path, index=False)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_generic_csv(path: Path, row: Dict[str, Any]) -> None:
    """For the "Osobno" files listed in section 8 that intentionally have a
    different schema from `RESULTS_SCHEMA` (e.g. E1's L0 sigma-calibration
    table) — no schema check, the caller owns column consistency for that
    one file. Still append-only, still writes the header only once.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)
