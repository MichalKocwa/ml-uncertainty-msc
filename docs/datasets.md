# UCI benchmark data — provenance

The six UCI benchmark datasets used in the protocol from
`docs/brief_claude_code_eksperymenty.md` section 5.2 are **not** loaded from
the raw UCI repository with our own train/test split. Instead we use the
exact data files and 20-way split indices from Hernandez-Lobato & Adams
(2015), as reused by Gal & Ghahramani (2016) and Lakshminarayanan et al.
(2017). Splitting the data ourselves would produce numbers that are not
comparable with those papers — the whole point of the protocol in section
5.2.

## Source

- Repository: <https://github.com/yaringal/DropoutUncertaintyExps>
- Pinned commit: `6eb4497628d12b0f300f4b4f6bdc386bebad565c` (2018-08-09)
- Path used: `UCI_Datasets/{dataset}/data/`

## License

The repository's `LICENSE` file is ambiguous: it states that *"All
contributions by Yarin Gal are licensed under a Creative Commons
Attribution-NonCommercial 4.0 International License"*, followed by a
separate permissive (BSD-2-clause-style) redistribution clause whose scope
is not clearly delimited against the CC BY-NC clause.

We treat the data and split-index files as **CC BY-NC 4.0**, which is
compatible with the non-commercial, academic use in this thesis but:

- **requires attribution** (given below),
- **prohibits commercial redistribution**.

Because of this ambiguity, and to avoid mixing a CC BY-NC-licensed dataset
into a repository whose own code has a different (permissive) license, the
data is **not vendored**: `data/uci_splits/` is gitignored and populated on
demand by `scripts/fetch_data.py`, which re-fetches from the pinned commit
and verifies every file's SHA-256 against `data/uci_splits.checksums.json`
(committed, so the check is meaningful on a fresh clone rather than "trust
whatever is at the URL today").

## Attribution

> Data splits and preprocessing from Yarin Gal, *DropoutUncertaintyExps*
> (<https://github.com/yaringal/DropoutUncertaintyExps>), commit `6eb4497`,
> based on the protocol from Hernández-Lobato, J. M. and Adams, R. P. (2015),
> "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural
> Networks". Used under CC BY-NC 4.0 for non-commercial academic research.

## Files fetched per dataset

| File | Contents |
|---|---|
| `data.txt` | full data matrix, whitespace-separated, last column(s) selected via `index_target.txt` |
| `index_features.txt` | 0-based column indices of the input features |
| `index_target.txt` | 0-based column index of the target |
| `index_train_{0..n_splits-1}.txt`, `index_test_{0..n_splits-1}.txt` | 0-based row indices for each of the literature's fixed train/test splits |
| `n_splits.txt` | number of splits (20 for all six datasets used here) |
| `n_hidden.txt` | hidden units per layer used in the reference protocol (50) |
| `n_epochs.txt` | base epoch count in the reference protocol (40 — see caveat below) |
| `results/test_rmse_100_xepochs_1_hidden_layers.txt` | published per-split (20 lines) test RMSE, single deterministic forward pass (dropout off) — does **not** reproduce the README table, see below |
| `results/test_MC_rmse_100_xepochs_1_hidden_layers.txt` | published per-split (20 lines) test RMSE, MC-dropout-averaged — this is the one the README's RMSE column reports |
| `results/test_ll_100_xepochs_1_hidden_layers.txt` | published per-split (20 lines) test log-likelihood — already MC-based (see below), no separate deterministic-vs-MC file exists for LL |

## Dataset-specific notes (resolved by using these files instead of our own loaders)

- **`energy`**: the upstream `data.txt` already contains only 9 columns (8
  features + 1 target) — the second target (Y2, cooling load) has been
  dropped, keeping Y1 (heating load), consistent with the literature's
  reported numbers.
- **`wine_quality_red`**: the upstream `data.txt` contains only the 1599 red
  wine rows (1599, 12 columns) — no red/white mixing to resolve.
- **`yacht`**: fetched as a local file from this repo instead of a UCI URL
  that (as of this writing) `ucimlrepo` refuses to import programmatically.
- **`kin8nm`**: `index_train_0.txt` has exactly 7373 rows, matching the
  brief's protocol table exactly (the splits were generated with
  `round(n * 0.9)`, not the `ceil`-based rounding `sklearn.train_test_split`
  uses, which gives 7372 for this N and caused a one-row mismatch when we
  split ourselves).

## Resolved — epoch count is 4000, not "10x" / 400

The repo's own `readme.md` prose says the 2018 update uses "10x training
epochs"; the `results/` filenames say `100_xepochs`. These are checked
against `experiment.py` (the script that produced these files), which
settles it unambiguously: `str(epochs_multiplier)` — the literal
`--epochx` CLI argument — is substituted directly into the result filename,
and training uses `n_epochs = int(n_epochs.txt * epochs_multiplier)`. The
files in this repo are named `..._100_xepochs_...`, i.e. they were produced
with `--epochx 100`, so the actual training length behind the published
numbers is `40 * 100 = 4000` epochs. The README's "10x" is prose
describing the multiplier relative to some other baseline, not the value
used for these particular files — the filename is the authoritative
record, the prose is not. This affects how "40 epochs" should be stated in
the brief/thesis text — left to the author to resolve, not changed here.

## Verified — published RMSE is `test_MC_rmse`, not `test_rmse`

Recomputing mean ± standard error over the 20 per-split values and
comparing against the repo's own README table (2026-08-24):

| dataset | metric | our mean ± SE | README |
|---|---|---|---|
| concrete | MC RMSE | 4.83 ± 0.16 | 4.82 ± 0.16 |
| concrete | LL | -2.94 ± 0.02 | -2.93 ± 0.02 |
| energy | MC RMSE | 0.54 ± 0.01 | 0.54 ± 0.06 |
| energy | LL | -1.21 ± 0.01 | -1.21 ± 0.01 |
| kin8nm | MC RMSE | 0.08 ± 0.00 | 0.08 ± 0.00 |
| kin8nm | LL | 1.14 ± 0.01 | 1.14 ± 0.01 |
| power_plant | MC RMSE | 4.01 ± 0.04 | 4.01 ± 0.04 |
| power_plant | LL | -2.81 ± 0.01 | -2.80 ± 0.01 |
| wine_quality_red | MC RMSE | 0.62 ± 0.01 | 0.62 ± 0.01 |
| wine_quality_red | LL | -0.93 ± 0.01 | -0.93 ± 0.01 |
| yacht | MC RMSE | 0.67 ± 0.05 | 0.67 ± 0.05 |
| yacht | LL | -1.25 ± 0.02 | -1.25 ± 0.01 |

`test_rmse` (the plain, non-MC-averaged file) does **not** reproduce this
table at all — e.g. concrete comes out at 5.45 ± 0.19, energy at
0.97 ± 0.06. `test_MC_rmse` is the correct file to compare against.

`test_ll` needed no such resolution: `net/net.py`'s `predict()` (in the
pinned commit) computes it via `logsumexp` over `T` stochastic
(dropout-on) forward passes — the standard Gal & Ghahramani MC predictive
log-likelihood estimator — so it is already the MC-evaluated quantity,
on the same footing as `test_MC_rmse`. There is no separate
deterministic-vs-MC file for LL because there is no non-MC version defined
in the reference code; fetching `test_ll` alongside `test_MC_rmse` already
gives RMSE and LL from the same evaluation mode.

**One unresolved mismatch, reported rather than papered over per
instruction:** `energy`'s MC RMSE standard error comes out as 0.01, not the
README's 0.06. The mean (0.54) matches exactly, and the *population
standard deviation* of the same 20 values is 0.06 — i.e. the README's
`energy` **RMSE** cell looks like it reports standard deviation, not
standard error, while every other cell in the table — including `energy`'s
own LL row — is consistent with standard error. This could be a
transcription slip in the upstream README for that one cell, or a mismatch
between which run produced that specific number and the `results/` files
currently in the pinned commit. Not resolved here.

**Consequence for how this README is used:** because of this one
inconsistent cell, the README table is treated as a **sanity check on
means only** (all six datasets' means matched their `test_MC_rmse`/`test_ll`
counterparts above) — it is not used as a source of dispersion values.
Every number that goes into `results/literature_comparison.csv` for P13
(mean, standard error, or a paired per-split comparison) is computed
directly from the 20-line `results/test_MC_rmse_*` / `results/test_ll_*`
files fetched by `scripts/fetch_data.py`, never transcribed from the
README prose.

## Fetch date

`data/uci_splits.checksums.json` was generated on 2026-08-24.
