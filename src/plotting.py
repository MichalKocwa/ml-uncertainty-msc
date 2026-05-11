import numpy as np
import matplotlib.pyplot as plt


def plot_sorted_predictions(y_test, mean, std, title="", ylabel="Target"):
    sort_idx = np.argsort(y_test.squeeze())
    y_sorted = y_test[sort_idx].squeeze()
    mean_sorted = mean[sort_idx]
    std_sorted = std[sort_idx]

    plt.figure(figsize=(10, 5))
    plt.fill_between(
        range(len(mean_sorted)),
        mean_sorted - 1.96 * std_sorted,
        mean_sorted + 1.96 * std_sorted,
        color="blue", alpha=0.2, label="95% CI", zorder=1,
    )
    plt.plot(mean_sorted, "b-", label="Predictive mean", zorder=2)
    plt.plot(y_sorted, "ro", markersize=2, alpha=0.5, label="True values", zorder=3)
    plt.xlabel("Sample index (sorted by true value)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_predicted_vs_true(y_test, means, stds, titles, ylabel="Target"):
    n = len(means)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, mean, std, title in zip(axes, means, stds, titles):
        lim = [min(y_test.min(), mean.min()), max(y_test.max(), mean.max())]
        ax.plot(lim, lim, "k--", alpha=0.4, zorder=1)
        ax.errorbar(y_test, mean, yerr=1.96 * std, fmt="o", markersize=2,
                    alpha=0.3, elinewidth=0.5, ecolor="C3", zorder=2)
        ax.set_xlabel("True values")
        ax.set_title(title)
    axes[0].set_ylabel(ylabel)
    fig.suptitle("Predicted vs True ", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_uncertainty_for_1d_data(X_train, y_train, X_eval, y_eval, y_pred, y_std, title=""):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(X_train, y_train, s=15, alpha=0.6, label="train", color="C0")
    if y_eval is not None:
        ax.plot(X_eval, y_eval, "k--", alpha=0.4, label="true function")
    ax.plot(X_eval, y_pred, "r-", linewidth=2, label="prediction")
    ax.fill_between(X_eval.ravel(),
                    y_pred - 2 * y_std,
                    y_pred + 2 * y_std,
                    alpha=0.25, color="red", label="±2σ")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.show()
