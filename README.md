# ml_uncertainty_msc

Experimental layer for an MSc thesis comparing five uncertainty-estimation
methods in regression (Gaussian processes, Bayes by Backprop, MC dropout,
Laplace approximation, deep ensembles) against a deterministic MAP baseline.
Full specification: `docs/experiment_brief.md`.

## Setup

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

### Windows: `numpy` build failure during install

`requirements.txt` pins `curvlinops-for-pytorch==2.0.1` (needed for
`laplace-torch`'s `kron` backend — see `docs/chapter4_notes.md`, D17). That
version caps `numpy<2.0`, and no prebuilt `numpy<2.0` wheel exists for
Python 3.13 on any platform (the 1.x branch predates 3.13's release), so
pip falls back to building `numpy` from source. On Windows this needs a
working MSVC toolchain; a plain terminal usually fails with:

```
ERROR: Found GNU link.exe instead of MSVC link.exe in ...\Git\usr\bin\link.EXE
```

(Git's own `link.exe` shadows MSVC's on `PATH`.) Fix: install the
"Desktop development with C++" workload from the Visual Studio Build Tools,
then run the install from an environment with the MSVC toolchain on `PATH`
— either the "x64 Native Tools Command Prompt for VS 2022" shortcut, or by
sourcing `vcvars64.bat` first:

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
pip install -r requirements.txt -r requirements-dev.txt
```

Verified end-to-end in a fresh venv (`numpy` builds successfully, `pip show
curvlinops-for-pytorch` reports `2.0.1`, `pytest tests/ -q` passes) — this
is not a hypothetical workaround.

## Data

UCI benchmark data and the literature's fixed train/test splits (see
`docs/datasets.md` for provenance and licensing) are not committed to this
repository — fetch them with:

```bash
python scripts/fetch_data.py
```

This downloads from a pinned upstream commit and verifies every file against
`data/uci_splits.checksums.json`, failing loudly on any mismatch.

## Reproducing the experiments

Every result in `results/` comes from one command below. Run them in this
order — later scripts read earlier scripts' CSVs.

**Two rules that the numbers depend on:**

1. **`set_seed` pins `torch.set_num_threads(1)`** (`src/seeding.py`). This is
   not a performance setting: the same seed at 8 threads produces a
   *different network*, because the thread count changes the order of
   float64 reductions and an optimiser amplifies it (measured: max
   |parameter delta| 0.124 on `map`/`kin8nm`; up to 6.75e-4 RMSE and 3.1e-3
   LL across the E2 table). Every row records the count it ran under in the
   `torch_threads` column. Do not override `ML_UNCERTAINTY_TORCH_THREADS`
   outside `experiments/thread_determinism_check.py`.
2. **`--workers N` is safe for values, never for timings.** Running 8
   processes changes no result, but it inflates `train_time_s` by a median
   of 1.60x. The two scripts marked **EXCLUSIVE** below measure time, so
   they must run alone — nothing else on the machine, no other experiment.

| # | command | produces | notes |
|---|---|---|---|
| 0 | `python scripts/fetch_data.py` | `data/uci_splits/` | verifies 306 files against committed checksums |
| 1 | `python experiments/e1_synthetic.py` | `results/e1_synthetic.csv`, `e1_sigma_calibration.csv`, `predictions_1d/` | six methods x three synthetic datasets, `seed=0` |
| 2 | `python scripts/epistemic_growth.py` | `results/epistemic_growth.csv` | growth/gap ratios read off E1's saved predictions |
| 3 | `python experiments/e0_gp_scaling.py --n-values 250,500,1000,2000,4000 --repeats 1 --norestart` | `results/e0_gp_scaling_norestart.csv` | **EXCLUSIVE** — this is a cost measurement (P14's slope). ~5 min. Add `N=8000` only if you have hours: one earlier run took 42 min there, another did not finish in two |
| 4 | `python experiments/e2_uci.py --workers 8` | `results/e2_uci.csv`, `calibration_curves.csv`, `literature_comparison.csv`, `e2_gp_skipped.csv` | the main table: 6 methods x 6 datasets x 20 splits. Resumes from the CSV — a kill costs at most one fit per worker. ~1.5 h |
| 5 | `python experiments/e2_uci.py --timing-pass` | `results/e2_cost.csv` | **EXCLUSIVE** — split 0, sequential, one thread. This is the cost column for chapter 5; `train_time_s` in `e2_uci.csv` is a by-product, not a measurement (D23a) |
| 6 | `python experiments/p13_dropout_diagnostic.py` | `results/p13_dropout_diagnostic.csv` | why our MC dropout differs from the published numbers: `dropout_p` sweep on `yacht`/`energy` |
| 7 | `python experiments/p13_gal_protocol.py --workers 8` | `results/p13_gal_protocol.csv`, `p13_gal_protocol_grid.csv` | the literature-validation run: gal2016's protocol reproduced on our implementation (4000 epochs, relu, input dropout, per-fold grid over `p` and `tau`). ~1.5 h |
| 8 | `python experiments/e3_gap_split.py --workers 8` | `results/e3_gap_split.csv`, `e3_gap_ratio.csv` | gap splits, all `d` dimensions plus a random-removal control. Resumes from `e3_gap_ratio.csv`. ~1.5 h |
| 9 | `python experiments/e3_summary.py` | `results/e3_gap_summary.csv` | splits E3 into real gaps / negative control / random control — the mixture of all dimensions is not reportable (E.1b) |
| 10 | `python experiments/e5_depth.py` | `results/e5_depth.csv` | depth ablation, 2 datasets x 3 seeds |
| 11 | `python experiments/e6a_mc_samples.py` | `results/e6a_mc_samples.csv` | justifies `T=100` (Figure 3.7) |
| 12 | `python experiments/e6c_laplace_structure.py --workers 6` | `results/e6c_laplace_structure.csv` | Laplace structures x prior modes (Figure 3.9, P5, P6). Cells that fail to factorise are recorded with `status="failed"`, not skipped |
| 13 | `python experiments/e6d_activation.py --workers 6` | `results/e6d_activation.csv` | ReLU vs TanH (P4) |
| 14 | `python experiments/expectations_check.py` | `results/expectations_check.csv` | P1-P14 against what the runs produced; every verdict computed from a CSV, none typed in |

### Figures

```bash
python experiments/e1_figures.py          # figures/rodzial{2,3}_rys/*.png from results/predictions_1d/ (no retraining)
python experiments/e6_figures.py          # figures/rodzial3_rys/img3_7.png, img3_9.png
python scripts/fig_gp_prior_samples.py    # figures/rodzial3_rys/img3_1.png
python scripts/fig_heteroscedastic_aleatoric.py   # figures/rodzial2_rys/img2_1.png
```

All chapter-3 figures share forced axes from `src/style.py` (`X_RANGE`,
`Y_RANGE`) and `seed=0`; that is a requirement of the comparison, not a
convention — see `docs/experiment_brief.md` section 10.

### Diagnostics (not part of the main pipeline)

Each answers one question that came up and left a number behind; none is
needed to reproduce the tables.

```bash
python experiments/thread_determinism_check.py --workers 8   # how much a run depends on the thread count
python experiments/duplicate_ll_diagnostic.py --workers 8    # wine's repeated rows and what the GP does with them
python experiments/gp_convergence_diagnostic.py --collapse-check --skip-warnings   # sklearn's GP warnings; the noise floor's cause
python experiments/uci_epochs_sweep.py                       # how the per-dataset epoch counts (D30) were chosen
python experiments/logvar_clamp_diagnostic.py                # whether the log-variance clamp was binding (D29)
```

### Smoke tests

Most scripts take `--quick` (2 splits, fewer epochs) — enough to check the
plumbing, never a result:

```bash
python experiments/e2_uci.py --quick
python experiments/e3_gap_split.py --quick
```

## Tests

```bash
pytest tests/ -q
```
