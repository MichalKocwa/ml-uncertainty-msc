"""E0 — GP scaling (brief section 9): measured cost, not an asserted `O(N^3)`.

Exact GP fit time and peak memory for growing `N`, so the eventual claim
"GP excluded above N~X" in chapter 5 is backed by a number from this
repository's own hardware, not a textbook complexity bound (brief, section
4.1's GP row, lines ~196-206).

Memory measurement (user decision, this session — `psutil` is not in the
brief's section 1.1 approved-dependency table, and adding it would need an
explicit exception like `numpyro`'s for E4): `tracemalloc`, stdlib only,
reports two numbers per run:
  - `kernel_matrix_mb` — theoretical, `8*N**2/1e6` (one NxN float64 Gram
    matrix), the dominant term algebraically.
  - `peak_memory_mb` — `tracemalloc.get_traced_memory()`'s peak, measured
    around `GPMethod.fit()`. `tracemalloc` traces allocations through
    Python's own allocator; large NumPy/LAPACK buffers routed through
    `mmap`/direct `malloc` can fall outside that trace, so this may
    *underestimate* true process RSS. Reporting both, plus their ratio, was
    the user's explicit instruction: if the ratio is roughly constant
    across `N`, it is itself the answer (how many same-sized matrices
    scikit-learn keeps live at once); if `tracemalloc` diverges from the
    theoretical curve, that gap is worth reporting, not hidden behind a
    single trusted number.

P14 (brief): fit-time scaling should be close to `N**3` (Cholesky
factorisation of the Gram matrix dominates) — checked via the slope of
`log(fit_time_s)` regressed on `log(N)`, printed at the end of a full run.

**`--norestart` mode (user decision, added after the first full run showed
2x spread between repeats at N=4000 — 530s/602s/295s under identical
config).** That spread is `GaussianProcessRegressor`'s `n_restarts_optimizer`
doing its job: each restart is an independent L-BFGS run from a random
hyperparameter init, and the *number of L-BFGS steps to converge* varies
run to run — so the measured `fit_time_s` conflates two different costs:
the algebraic cost of one Cholesky-based optimisation trajectory (what P14
is actually about) and the unpredictable multiplier from how many restarts
happen to need how many steps. `--norestart` fits with
`n_restarts_optimizer=0` (a single optimisation attempt, still iterative,
but without the restart-count multiplier) to isolate the first cost —
written to a *separate* file, `results/e0_gp_scaling_norestart.csv`, not
mixed into `e0_gp_scaling.csv`'s `fit_time_s` column, since the two measure
different things and mixing them would misrepresent both. Report:
`fit_time_norestart_s`'s own log-log slope (the real P14 check) and, per
`N`, `fit_time_s / fit_time_norestart_s` (how much the restart search
multiplies the base cost, and whether that multiplier itself is stable
across `N` or grows with it).

Data: `sin_homo`'s own generative process (`f(x)=sin(x)`, `sigma=0.1`,
`x ~ linspace(0, 6, N)`) at the requested `N`, not the fixed `SYNTHETIC_N`
dataset — E0 is about `N` as a free variable, section 5.1's fixed-`N`
datasets are a different experiment. `GPMethod`'s own defaults
(`n_restarts_optimizer=5`) are kept unchanged from how GP is actually used
in E1/E2: reducing them for this measurement would understate the real
per-fit cost this experiment exists to document.

Usage:
  python experiments/e0_gp_scaling.py [--quick] [--n-values 250,500,...] [--repeats 3]
"""
import argparse
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.methods.gp import GPMethod
from src.results import RESULTS_DIR, append_generic_csv, git_commit_short, now_iso

EXPERIMENT_ID = "e0_gp_scaling"
DEFAULT_N_VALUES = [250, 500, 1000, 2000, 4000, 8000]
DEFAULT_REPEATS = 3
QUICK_N_VALUES = [250, 500]
QUICK_REPEATS = 1
NORESTART_EXPERIMENT_ID = "e0_gp_scaling_norestart"


def make_data(n: int, seed: int):
    rng = np.random.RandomState(seed)
    X = np.linspace(0, 6, n).reshape(-1, 1)
    y = np.sin(X.ravel()) + 0.1 * rng.randn(n)
    return X.astype(np.float64), y.astype(np.float64)


def run_one(n: int, repeat: int, seed: int) -> dict:
    X, y = make_data(n, seed)
    method = GPMethod()

    tracemalloc.start()
    t0 = time.perf_counter()
    method.fit(X, y, seed=seed, use_cache=False)  # E0 measures fit_time_s itself -- a cache hit would corrupt the measurement (src/methods/cache.py)
    fit_time_s = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    kernel_matrix_mb = 8 * n ** 2 / 1e6
    peak_memory_mb = peak_bytes / 1e6
    row = dict(
        experiment_id=EXPERIMENT_ID,
        n=n,
        repeat=repeat,
        seed=seed,
        fit_time_s=fit_time_s,
        kernel_matrix_mb=kernel_matrix_mb,
        peak_memory_mb=peak_memory_mb,
        ratio_peak_over_kernel=peak_memory_mb / kernel_matrix_mb,
        n_parameters=method.n_parameters,
        timestamp=now_iso(),
        git_commit=git_commit_short(),
    )
    print(f"  N={n:6d} repeat={repeat} fit_time_s={fit_time_s:8.3f} "
          f"kernel_mb={kernel_matrix_mb:9.2f} peak_mb={peak_memory_mb:9.2f} "
          f"ratio={row['ratio_peak_over_kernel']:.3f}")
    return row


def run_one_norestart(n: int, repeat: int, seed: int) -> dict:
    X, y = make_data(n, seed)
    method = GPMethod(n_restarts_optimizer=0)

    t0 = time.perf_counter()
    method.fit(X, y, seed=seed, use_cache=False)  # E0 measures fit_time_s itself -- a cache hit would corrupt the measurement (src/methods/cache.py)
    fit_time_norestart_s = time.perf_counter() - t0

    row = dict(
        experiment_id=NORESTART_EXPERIMENT_ID,
        n=n,
        repeat=repeat,
        seed=seed,
        fit_time_norestart_s=fit_time_norestart_s,
        timestamp=now_iso(),
        git_commit=git_commit_short(),
    )
    print(f"  N={n:6d} repeat={repeat} fit_time_norestart_s={fit_time_norestart_s:8.3f}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="smoke test: small N, 1 repeat")
    parser.add_argument("--n-values", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument(
        "--norestart", action="store_true",
        help="isolate algebraic fit cost from optimizer-restart-count noise (n_restarts_optimizer=0); "
             "writes to results/e0_gp_scaling_norestart.csv instead of e0_gp_scaling.csv",
    )
    args = parser.parse_args()

    if args.n_values is not None:
        n_values = [int(v) for v in args.n_values.split(",")]
    else:
        n_values = QUICK_N_VALUES if args.quick else DEFAULT_N_VALUES
    repeats = args.repeats if args.repeats is not None else (QUICK_REPEATS if args.quick else DEFAULT_REPEATS)

    if args.norestart:
        out_path = RESULTS_DIR / f"{NORESTART_EXPERIMENT_ID}.csv"
        rows = []
        for n in n_values:
            print(f"N={n}:")
            for r in range(repeats):
                row = run_one_norestart(n, repeat=r, seed=r)
                append_generic_csv(out_path, row)
                rows.append(row)
        if len(n_values) >= 2:
            log_n = np.log(np.array([row["n"] for row in rows], dtype=np.float64))
            log_t = np.log(np.array([row["fit_time_norestart_s"] for row in rows], dtype=np.float64))
            slope, intercept = np.polyfit(log_n, log_t, 1)
            print(f"\nP14 (norestart): slope of log(fit_time_norestart_s) vs log(N) = {slope:.3f} (expected close to 3)")
        return

    out_path = RESULTS_DIR / f"{EXPERIMENT_ID}.csv"
    rows = []
    for n in n_values:
        print(f"N={n}:")
        for r in range(repeats):
            row = run_one(n, repeat=r, seed=r)
            append_generic_csv(out_path, row)
            rows.append(row)

    if len(n_values) >= 2:
        log_n = np.log(np.array([row["n"] for row in rows], dtype=np.float64))
        log_t = np.log(np.array([row["fit_time_s"] for row in rows], dtype=np.float64))
        slope, intercept = np.polyfit(log_n, log_t, 1)
        print(f"\nP14: slope of log(fit_time_s) vs log(N) = {slope:.3f} (expected close to 3)")


if __name__ == "__main__":
    main()
