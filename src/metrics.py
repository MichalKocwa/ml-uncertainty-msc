import numpy as np


# Root Mean Squared Error
def rmse(y_true, mean):
    return float(np.sqrt(np.mean((y_true - mean) ** 2)))


# Mean Absolute Error
def mae(y_true, mean):
    return float(np.mean(np.abs(y_true - mean)))


# Negative Log-Likelihood (Gaussian) — minimalizować; ocenia cały rozkład predykcyjny
def nll(y_true, mean, std):
    var = std ** 2
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + (y_true - mean) ** 2 / (2 * var)))


# Prediction Interval Coverage Probability — ideał: 0.95 (95% prawdziwych wartości w przedziale ±1.96σ)
def picp(y_true, mean, std, z=1.96):
    lower = mean - z * std
    upper = mean + z * std
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


# Mean Prediction Interval Width — minimalizowac
def mpiw(std, z=1.96):
    return float(np.mean(2 * z * std))


def summary(y_true, mean, std):
    return {
        "RMSE": rmse(y_true, mean),
        "MAE":  mae(y_true, mean),
        "NLL":  nll(y_true, mean, std),
        "PICP": picp(y_true, mean, std),
        "MPIW": mpiw(std),
    }
