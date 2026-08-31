"""figures/rodzial2_rys/img2_1.png — input-dependent aleatoric noise, two-head network.

Chapter 2.1's illustration of what a heteroscedastic likelihood buys:
`sin_hetero`'s noise grows linearly across the input range
(`sigma(x) = 0.05 + 0.15 x / 6`), and a network with a second head for
`log sigma^2(x)` recovers that shape, while one global `sigma` can only
average it.

**Outside the main comparison, deliberately** (author's instruction, and
D1 in docs/chapter4_notes.md). Every method in chapters 3-5 shares ONE
homoscedastic backbone so that differences between them are differences in
the posterior approximation and not in the noise model; the two-head
network below exists only to draw this figure and is defined here rather
than in `src/methods/` so it cannot be mistaken for one of the six.

The comparison drawn is therefore against the same architecture with a
single global `sigma`, not against another method: the point is the noise
model, not the inference.

Usage:
  python scripts/fig_heteroscedastic_aleatoric.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.data import load_sin_hetero
from src.methods.backbone import DTYPE
from src.seeding import set_seed
from src.style import METHOD_COLORS, SEED, X_RANGE

HIDDEN = 50
EPOCHS = 3000
LR = 1e-2
OUT_PATH = Path(__file__).resolve().parent.parent / "figures" / "rodzial2_rys" / "img2_1.png"

HETERO_COLOUR = "#7B3294"   # deliberately outside METHOD_COLORS: this is not one of the six
HOMO_COLOUR = METHOD_COLORS["map"]


class TwoHeadMLP(nn.Module):
    """`x -> tanh(Linear(h)) -> (mean, log sigma^2)`, both heads on the same
    hidden layer. `log_var` is the second OUTPUT, i.e. a function of `x` —
    that is the whole difference from `HomoscedasticMLP`, whose `log_sigma2`
    is one global parameter."""

    def __init__(self, hidden: int = HIDDEN, heteroscedastic: bool = True):
        super().__init__()
        self.body = nn.Linear(1, hidden, dtype=DTYPE)
        self.mean_head = nn.Linear(hidden, 1, dtype=DTYPE)
        self.heteroscedastic = heteroscedastic
        if heteroscedastic:
            self.log_var_head = nn.Linear(hidden, 1, dtype=DTYPE)
        else:
            self.log_sigma2 = nn.Parameter(torch.zeros((), dtype=DTYPE))

    def forward(self, x):
        h = torch.tanh(self.body(x))
        mu = self.mean_head(h)
        log_var = self.log_var_head(h) if self.heteroscedastic else self.log_sigma2.expand_as(mu)
        return mu, log_var.clamp(-12.0, 6.0)


def _fit(X, y, heteroscedastic: bool) -> TwoHeadMLP:
    set_seed(SEED)
    model = TwoHeadMLP(heteroscedastic=heteroscedastic)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    X_t = torch.as_tensor(X, dtype=DTYPE)
    y_t = torch.as_tensor(y, dtype=DTYPE).reshape(-1, 1)
    for _ in range(EPOCHS):
        opt.zero_grad()
        mu, log_var = model(X_t)
        loss = torch.mean(0.5 * log_var + 0.5 * (y_t - mu) ** 2 / torch.exp(log_var))
        loss.backward()
        opt.step()
    return model


def _predict(model, x_grid):
    with torch.no_grad():
        mu, log_var = model(torch.as_tensor(x_grid, dtype=DTYPE))
    return mu.numpy().ravel(), np.sqrt(np.exp(log_var.numpy().ravel()))


def main():
    ds = load_sin_hetero(seed=SEED)
    x_grid = np.linspace(*X_RANGE, 500).reshape(-1, 1)
    x = x_grid.ravel()

    hetero_mean, hetero_sigma = _predict(_fit(ds.X_train, ds.y_train, True), x_grid)
    homo_mean, homo_sigma = _predict(_fit(ds.X_train, ds.y_train, False), x_grid)
    sigma_true = 0.05 + 0.15 * np.clip(x, 0.0, None) / 6.0  # src/data.py's _sigma_hetero

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    ax.scatter(ds.X_train.ravel(), ds.y_train, s=6, color="black", alpha=0.3, zorder=5, label="train")
    ax.plot(x, np.sin(x), color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="true f(x)")
    ax.plot(x, hetero_mean, color=HETERO_COLOUR, linewidth=1.4, label="two-head mean")
    ax.fill_between(x, hetero_mean - 2 * hetero_sigma, hetero_mean + 2 * hetero_sigma,
                    color=HETERO_COLOUR, alpha=0.20, label="two-head $\\pm 2\\hat\\sigma(x)$")
    ax.fill_between(x, homo_mean - 2 * homo_sigma, homo_mean + 2 * homo_sigma,
                    facecolor="none", edgecolor=HOMO_COLOUR, linestyle=":", linewidth=1.2,
                    label="one global $\\sigma$, $\\pm 2\\hat\\sigma$")
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(-3.0, 3.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Predictive band")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.plot(x, sigma_true, color="black", linestyle="--", linewidth=1.2, label="true $\\sigma(x)$")
    ax.plot(x, hetero_sigma, color=HETERO_COLOUR, linewidth=1.4, label="two-head $\\hat\\sigma(x)$")
    ax.plot(x, homo_sigma, color=HOMO_COLOUR, linestyle=":", linewidth=1.4, label="one global $\\hat\\sigma$")
    ax.axvspan(0.0, 6.0, color="grey", alpha=0.10, label="training support")
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(0.0, max(0.6, float(hetero_sigma.max()) * 1.1))
    ax.set_xlabel("x")
    ax.set_ylabel("$\\sigma$")
    ax.set_title("Recovered noise scale")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Input-dependent aleatoric noise on sin_hetero: a second output head recovers "
        f"$\\sigma(x)$, a global $\\sigma$ averages it (seed = {SEED}; "
        "outside the six-method comparison)", fontsize=10,
    )
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    plt.close(fig)

    inside = (x >= 0) & (x <= 6)
    print(f"wrote {OUT_PATH}")
    print(f"  sigma_true over the training support: {sigma_true[inside].min():.3f} -> {sigma_true[inside].max():.3f}")
    print(f"  two-head sigma_hat:                   {hetero_sigma[inside].min():.3f} -> {hetero_sigma[inside].max():.3f}")
    print(f"  global sigma_hat:                     {homo_sigma[inside].mean():.3f} (constant by construction)")


if __name__ == "__main__":
    main()
