"""Dataset loading and preprocessing (brief sections 5 and 6).

Standardisation convention (section 5.3): both `X` and `y` are standardised
to mean 0, variance 1, fitted on the training split only. Metrics must be
computed after inverting the `y` transform — `sigma_original =
sigma_standardised * y_scaler.scale_[0]` — otherwise RMSE is correct but NLL
and MPIW are not comparable across datasets or with the literature.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------- #
# Synthetic 1D datasets (section 5.1) — the only place with full ground truth
# --------------------------------------------------------------------- #
@dataclass
class SyntheticDataset:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray            # CLEAN f(x) at X_eval — for plotting the true-function line only, never for metrics (see y_eval_noisy)
    y_eval_noisy: np.ndarray      # f(x) + fresh noise from the same sigma_fn — what RMSE/LL/PICP/MPIW/interval_score/ECE/QICE must be computed against
    sigma_true_train: np.ndarray  # sigma(x) at X_train, for reference
    sigma_true_eval: np.ndarray   # sigma(x) at X_eval, ground truth for L0 (RMSE(sigma_hat, sigma_true))


def _sigma_homo(x: np.ndarray) -> np.ndarray:
    return np.full_like(x, 0.1)


def _sigma_hetero(x: np.ndarray) -> np.ndarray:
    return 0.05 + 0.15 * x / 6.0


EVAL_GRID_N = 1000  # E1 figures need >=1000 points (brief section 10, this session's decision)


def _make_sin_dataset(name, X_train, sigma_fn, seed) -> SyntheticDataset:
    rng = np.random.RandomState(seed)
    x_train_flat = X_train.ravel()
    sigma_train = sigma_fn(x_train_flat)
    y_train = np.sin(x_train_flat) + sigma_train * rng.randn(len(x_train_flat))

    # Symmetric [-2, 8] (this session's decision, correcting an earlier brief
    # asymmetry: [-2, 10] was 2 units left of training but 4 right — GP came
    # out looking better on the right purely because it was closer to data,
    # and every method's right-side extrapolation covered 64% of sin's period
    # rather than a comparable distance to the left).
    X_eval = np.linspace(-2, 8, EVAL_GRID_N).reshape(-1, 1)
    x_eval_flat = X_eval.ravel()
    y_eval = np.sin(x_eval_flat)  # clean f(x) -- plotting only, see dataclass docstring
    sigma_eval = sigma_fn(x_eval_flat)
    # Fresh noise draw, same rng (continues the training-noise stream, still
    # deterministic from `seed` alone), same sigma_fn as training. Bug fixed
    # this session: an evaluation grid with no noise makes PICP/MPIW/interval_score/
    # ECE/QICE measure coverage of the smooth function, not of an observation —
    # trivially ~1.0 whenever the mean is close, regardless of band width, for
    # any method. RMSE/LL against noise-free targets are not the coverage bug
    # but were also computed against the wrong target for consistency with E2
    # (real data has no noise-free targets to fall back on).
    y_eval_noisy = np.sin(x_eval_flat) + sigma_eval * rng.randn(len(x_eval_flat))

    return SyntheticDataset(
        name=name,
        X_train=X_train.astype(np.float64),
        y_train=y_train.astype(np.float64),
        X_eval=X_eval.astype(np.float64),
        y_eval=y_eval.astype(np.float64),
        y_eval_noisy=y_eval_noisy.astype(np.float64),
        sigma_true_train=sigma_train.astype(np.float64),
        sigma_true_eval=sigma_eval.astype(np.float64),
    )


# N=250, not the brief's original N=50 (this session's decision — see
# docs/chapter4_notes.md and src/methods/laplace.py's module docstring).
# Diagnosed cause of the `full`/`kron` Laplace `var_epistemic` needle
# spikes: N=50 < ~151 network parameters leaves the GGN Hessian rank
# deficient, and the resulting posterior precision is severely
# ill-conditioned regardless of float precision. N=250 > n_parameters
# gives the GGN full rank; verified on a 6-seed sweep before adopting this
# (see chat report). N=50 was arbitrary in the brief, not a requirement.
SYNTHETIC_N = 250


def load_sin_homo(seed: int = 42) -> SyntheticDataset:
    """f(x) = sin(x), sigma = 0.1 constant, train on [0, 6], N=250. Extrapolation test."""
    X_train = np.linspace(0, 6, SYNTHETIC_N).reshape(-1, 1)
    return _make_sin_dataset("sin_homo", X_train, _sigma_homo, seed)


def load_sin_hetero(seed: int = 42) -> SyntheticDataset:
    """f(x) = sin(x), sigma(x) = 0.05 + 0.15*x/6, train on [0, 6], N=250. Heteroscedasticity test."""
    X_train = np.linspace(0, 6, SYNTHETIC_N).reshape(-1, 1)
    return _make_sin_dataset("sin_hetero", X_train, _sigma_hetero, seed)


def load_sin_gap(seed: int = 42) -> SyntheticDataset:
    """f(x) = sin(x), sigma = 0.1 constant, train on [0,2] u [4,6], N=250. In-between uncertainty test."""
    half = SYNTHETIC_N // 2
    X_train = np.concatenate([
        np.linspace(0, 2, half),
        np.linspace(4, 6, half),
    ]).reshape(-1, 1)
    return _make_sin_dataset("sin_gap", X_train, _sigma_homo, seed)


SYNTHETIC_DATASETS: Dict[str, Callable[..., SyntheticDataset]] = {
    "sin_homo": load_sin_homo,
    "sin_hetero": load_sin_hetero,
    "sin_gap": load_sin_gap,
}


# Training-support geometry per synthetic variant, in one place. Both
# `experiments/e1_synthetic.py` (which splits its metric rows by these) and
# every diagnostic that needs "where the training data actually is"
# (`scripts/epistemic_growth.py`, the sweep scripts) read it from here, so
# the definition of `in_range` cannot drift between the results table and
# the analyses computed on top of it. `sin_gap`'s `in_range` deliberately
# EXCLUDES its held-out (2, 4) interval — that interval is its own
# `in_gap` split, and folding it into `in_range` would put the very region
# the dataset exists to probe into the baseline it is compared against.
def range_masks(dataset_name: str, x: np.ndarray) -> Dict[str, np.ndarray]:
    """`{split_type: boolean mask over x}` for a synthetic variant.

    `in_range`: where training data actually is (`sin_gap`: its two
    training intervals). `extrapolation`: outside [0, 6] for every variant.
    `in_gap` (`sin_gap` only): the held-out (2, 4) interval — open,
    matching how `sin_gap`'s training halves are built up to but not
    including 2 and from 4 (`load_sin_gap` above).
    """
    if dataset_name not in SYNTHETIC_DATASETS:
        raise ValueError(f"unknown dataset '{dataset_name}', expected one of {sorted(SYNTHETIC_DATASETS)}")
    x = np.asarray(x)
    if dataset_name == "sin_gap":
        return {
            "in_range": ((x >= 0) & (x <= 2)) | ((x >= 4) & (x <= 6)),
            "in_gap": (x > 2) & (x < 4),
            "extrapolation": (x < 0) | (x > 6),
        }
    return {
        "in_range": (x >= 0) & (x <= 6),
        "extrapolation": (x < 0) | (x > 6),
    }



# --------------------------------------------------------------------- #
# UCI benchmarks (section 5.2) — protocol comparable with the literature
# --------------------------------------------------------------------- #
@dataclass
class UCIDataset:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_scaler: StandardScaler
    split: int


# Expected (n_total, n_train, d) per the protocol table (section 5.2). Now
# that we read the literature's own split files (see docs/datasets.md),
# these are exact, not approximate, and the accompanying test checks them
# without tolerance.
UCI_SPEC: Dict[str, Dict[str, int]] = {
    "yacht":            dict(n=308,  n_train=277,  d=6),
    "energy":           dict(n=768,  n_train=691,  d=8),
    "concrete":         dict(n=1030, n_train=927,  d=8),
    "wine_quality_red": dict(n=1599, n_train=1439, d=11),
    "kin8nm":           dict(n=8192, n_train=7373, d=8),
    "power_plant":      dict(n=9568, n_train=8611, d=4),
}

# data/uci_splits/{name}/ is populated by `python scripts/fetch_data.py`,
# not committed to the repo — see docs/datasets.md for why.
_UCI_SPLITS_ROOT = Path(__file__).resolve().parent.parent / "data" / "uci_splits"


def _uci_dataset_dir(name: str) -> Path:
    if name not in UCI_SPEC:
        raise ValueError(f"unknown UCI dataset '{name}', expected one of {sorted(UCI_SPEC)}")
    dataset_dir = _UCI_SPLITS_ROOT / name
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"'{dataset_dir}' not found. Run `python scripts/fetch_data.py` "
            "first (see docs/datasets.md)."
        )
    return dataset_dir


def load_uci_raw(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Full (X, y) for a UCI benchmark, before any splitting or scaling.

    Public: this is the entry point for `gap_split` (section 5.4) on UCI
    data, e.g. `gap_split(*load_uci_raw("concrete"), name="concrete")` — gap
    split needs the full dataset, not one of the 20 fixed literature splits
    from `load_uci`.
    """
    dataset_dir = _uci_dataset_dir(name)
    data = np.loadtxt(dataset_dir / "data.txt")
    feature_idx = np.loadtxt(dataset_dir / "index_features.txt", dtype=int).reshape(-1)
    target_idx = np.loadtxt(dataset_dir / "index_target.txt", dtype=int).reshape(-1)
    X = data[:, feature_idx]
    y = data[:, target_idx].ravel()
    return X, y


def n_uci_splits(name: str) -> int:
    """Number of literature splits available for a UCI dataset (20 for all six used here)."""
    return int(np.loadtxt(_uci_dataset_dir(name) / "n_splits.txt", dtype=int))


def uci_split_indices(name: str, split: int) -> Tuple[np.ndarray, np.ndarray]:
    """`(train_idx, test_idx)` row indices for literature split `split` (section 5.2).

    Public because the split files are the protocol: anything that needs the
    training fold in a form `load_uci` does not return (e.g. the epoch-count
    measurement in `experiments/uci_epochs_sweep.py`, which needs the RAW
    training rows so it can fit its own scaler on an inner sub-fold) must
    still read exactly these indices, not re-split the data itself.
    """
    dataset_dir = _uci_dataset_dir(name)
    n_splits = n_uci_splits(name)
    if not (0 <= split < n_splits):
        raise ValueError(f"'{name}' has {n_splits} splits (0..{n_splits - 1}), got split={split}")
    train_idx = np.loadtxt(dataset_dir / f"index_train_{split}.txt", dtype=int)
    test_idx = np.loadtxt(dataset_dir / f"index_test_{split}.txt", dtype=int)
    return train_idx, test_idx


def load_uci(name: str, split: int, max_train: int = None) -> UCIDataset:
    """Load a UCI benchmark using literature split number `split` (section 5.2).

    Train/test row indices come from `data/uci_splits/{name}/index_{train,test}_{split}.txt`
    — the exact splits used by Hernandez-Lobato & Adams (2015) and reused by
    Gal & Ghahramani (2016) and Lakshminarayanan et al. (2017). No random
    splitting happens here: that is the entire point of using these files
    instead of our own `train_test_split` (see docs/datasets.md — a
    self-generated split is not comparable with the published numbers).

    `X` and `y` are standardised on the training fold only (section 5.3).

    `max_train` subsamples the training fold after loading the fixed split;
    it exists only for the `gp_subsampled` diagnostic row / `--quick` runs,
    never the default path.
    """
    X, y = load_uci_raw(name)
    train_idx, test_idx = uci_split_indices(name, split)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    if max_train is not None and len(X_train) > max_train:
        rng = np.random.RandomState(split)
        idx = rng.choice(len(X_train), max_train, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

    X_train_s = x_scaler.transform(X_train).astype(np.float64)
    X_test_s = x_scaler.transform(X_test).astype(np.float64)
    y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel().astype(np.float64)
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel().astype(np.float64)

    return UCIDataset(name, X_train_s, y_train_s, X_test_s, y_test_s, y_scaler, split)


# --------------------------------------------------------------------- #
# Gap split (section 5.4) — no ground truth, but a testable ordering
# --------------------------------------------------------------------- #
@dataclass
class GapSplitDataset:
    name: str
    feature_index: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_in_range: np.ndarray
    y_in_range: np.ndarray
    X_in_gap: np.ndarray
    y_in_gap: np.ndarray
    y_scaler: StandardScaler


def gap_split(X: np.ndarray, y: np.ndarray, name: str = "", seed: int = 42, test_size: float = 0.1) -> GapSplitDataset:
    """Hold out the middle third of one feature as an "in-between" test region.

    Design decision, documented per CLAUDE.md: the brief (section 5.4) says
    to pick "the feature with the largest variance", but that is ill-defined
    after standardisation (every standardised feature has unit variance).
    We instead pick the feature with the highest |corr(x_j, y)| on the raw,
    unstandardised data — the feature most predictive of the target is the
    one where an epistemic gap is most likely to be visible in y-space.

    Rows whose selected feature falls in [q33, q66] (quantiles over the full
    dataset) are held out entirely as the `in_gap` evaluation set. The
    remaining rows are split further (`test_size`) into a training fold and
    an `in_range` evaluation set drawn from the same (dense) region as
    training. Standardisation is fit on the training fold only, after gap
    removal, to avoid leaking gap statistics into the scaler.
    """
    correlations = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range(X.shape[1])])
    feature_idx = int(np.argmax(correlations))
    feature = X[:, feature_idx]
    q33, q66 = np.quantile(feature, [1 / 3, 2 / 3])

    in_gap_mask = (feature >= q33) & (feature <= q66)
    outside_idx = np.where(~in_gap_mask)[0]
    gap_idx = np.where(in_gap_mask)[0]

    train_idx, in_range_idx = train_test_split(outside_idx, test_size=test_size, random_state=seed)

    x_scaler = StandardScaler().fit(X[train_idx])
    y_scaler = StandardScaler().fit(y[train_idx].reshape(-1, 1))

    def _prep(idx):
        X_s = x_scaler.transform(X[idx]).astype(np.float64)
        y_s = y_scaler.transform(y[idx].reshape(-1, 1)).ravel().astype(np.float64)
        return X_s, y_s

    X_train, y_train = _prep(train_idx)
    X_in_range, y_in_range = _prep(in_range_idx)
    X_in_gap, y_in_gap = _prep(gap_idx)

    return GapSplitDataset(
        name=name,
        feature_index=feature_idx,
        X_train=X_train, y_train=y_train,
        X_in_range=X_in_range, y_in_range=y_in_range,
        X_in_gap=X_in_gap, y_in_gap=y_in_gap,
        y_scaler=y_scaler,
    )
