import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_sin(seed=42):
    rng = np.random.RandomState(seed)
    n_train = 50
    noise = 0.1
    X_train = np.linspace(0, 6, n_train).reshape(-1, 1)
    y_train = np.sin(X_train).ravel() + noise * rng.randn(n_train)
    X_eval = np.linspace(-2, 10, 300).reshape(-1, 1)
    y_eval = np.sin(X_eval).ravel()
    return (X_train.astype(np.float32), y_train.astype(np.float32),
            X_eval.astype(np.float32), y_eval.astype(np.float32), True)


def load_concrete(seed=42, max_train=2000):
    return _load_uci(uci_id=165, seed=seed, max_train=max_train)


def load_power_plant(seed=42, max_train=2000):
    return _load_uci(uci_id=294, seed=seed, max_train=max_train)


def load_california_housing(seed=42, max_train=2000):
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed)
    data = fetch_california_housing()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    if max_train is not None and len(X_train) > max_train:
        idx = rng.choice(len(X_train), max_train, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, y_train, X_test, y_test, False


def _load_uci(uci_id, seed, max_train):
    rng = np.random.RandomState(seed)
    data = fetch_ucirepo(id=uci_id)
    X = data.data.features.values.astype(np.float32)
    y = data.data.targets.values.astype(np.float32).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    if max_train is not None and len(X_train) > max_train:
        idx = rng.choice(len(X_train), max_train, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    scaler = StandardScaler().fit(X_train) # used to normalise to mean = 1 and std = 1
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, y_train, X_test, y_test, False
