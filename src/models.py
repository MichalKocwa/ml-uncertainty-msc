"""Model definitions for the uncertainty comparison project.

Three model families:
- MLP — simple feed-forward neural network in PyTorch
- BayesianRBFRegression — parametric Bayesian regression with RBF features
- make_gp — factory for sklearn's GaussianProcessRegressor with sensible defaults
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import BayesianRidge


# --------------------------------------------------------------------- #
# Neural network
# --------------------------------------------------------------------- #
class MLP(nn.Module):
    """Plain feed-forward network with optional dropout."""
    def __init__(self, in_dim=1, hidden=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def make_mlp(in_dim, dropout=0.0):
    """Factory used by load_nn — must reproduce the architecture used at training time."""
    return MLP(in_dim=in_dim, dropout=dropout)


def train_mlp(X, y, dropout=0.0, epochs=1000, lr=0.01, seed=42):
    """Train a deterministic MLP with full-batch gradient descent."""
    torch.manual_seed(seed)
    model = MLP(in_dim=X.shape[1], dropout=dropout)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        opt.step()
    return model


def mlp_point_predict(model, X):
    """Deterministic point prediction (dropout disabled)."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return model(X_t).numpy().ravel()


# --------------------------------------------------------------------- #
# Variational Inference BNN (Bayes by Backprop, Blundell et al. 2015)
# --------------------------------------------------------------------- #
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
import torch.nn.functional as F

@variational_estimator
class BayesianMLP(nn.Module):
    def __init__(self, in_dim=1, hidden=64):
        super().__init__()
        self.l1 = BayesianLinear(in_dim, hidden)
        self.l2 = BayesianLinear(hidden, hidden)
        self.l3 = BayesianLinear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


# --------------------------------------------------------------------- #
# Gaussian Process
# --------------------------------------------------------------------- #
def make_gp(random_state=42):
    """Factory for a GP regressor with a reasonable default kernel."""
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        random_state=random_state,
    )
