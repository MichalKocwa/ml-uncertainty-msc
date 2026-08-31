"""P13's two structural claims about matching gal2016's training objective.

P13 asks whether our MC dropout reproduces the published UCI numbers once
the protocol is equalised. Two pieces of that equalisation are conversions
rather than settings, and a conversion that is wrong by a constant would
produce a plausible-looking but meaningless answer:

1. **The penalty coefficient.** gal2016's Keras loss is
   `mean((y - yhat)^2) + reg * sum(W^2)` over WEIGHTS only, with
   `reg = lengthscale^2 (1 - p) / (2 N tau)`. Ours is a Gaussian NLL at
   fixed `sigma^2` plus `sum_theta c_theta * theta^2`, i.e. the same
   objective scaled by `1/(2 sigma^2)`. The scripts convert with
   `c_weight = reg / (2 sigma^2)`, `c_bias = 0`, which makes our loss
   exactly `1/(2 sigma^2)` times his. `test_penalty_conversion_matches_gal_gradient`
   checks that on real gradients rather than on the algebra, and
   `test_adam_step_matches_under_the_conversion` checks the thing that
   actually matters — that Adam takes the same step, since Adam is
   invariant to a constant rescaling of the loss.

2. **The input-dropout flag.** `MLP(input_dropout=True)` must add
   gal2016's `Dropout(p)(inputs)` and nothing else; with the flag off the
   backbone must be bit-identical to what produced every existing result.
"""
import math

import numpy as np
import pytest
import torch

from src.methods.backbone import DTYPE, HomoscedasticMLP, gaussian_nll
from src.seeding import set_seed

LENGTHSCALE = 1e-2  # gal2016 net.py


def _gal_reg(dropout_p: float, n: int, tau: float) -> float:
    """`reg` exactly as gal2016's `net.py` computes it."""
    return LENGTHSCALE ** 2 * (1 - dropout_p) / (2.0 * n * tau)


def _batch(n=64, d=5, seed=0):
    rng = np.random.RandomState(seed)
    X = torch.as_tensor(rng.randn(n, d), dtype=DTYPE)
    y = torch.as_tensor(rng.randn(n, 1), dtype=DTYPE)
    return X, y


def _model(sigma2, dropout_p=0.0, seed=0, in_dim=5):
    set_seed(seed)
    return HomoscedasticMLP(
        in_dim=in_dim, hidden=8, dropout_p=dropout_p, always_on=True,
        activation="relu", fixed_sigma2=sigma2,
    )


def _weight_only_penalty(sigma2, reg):
    """The conversion under test: gal2016 regularises weights, not biases."""
    return {
        "linear1.weight": reg / (2.0 * sigma2), "linear1.bias": 0.0,
        "mean_head.weight": reg / (2.0 * sigma2), "mean_head.bias": 0.0,
    }


def _gal_loss(model, X, y, reg):
    mu, _ = model(X)
    mse = torch.mean((y - mu) ** 2)
    l2 = sum(p.pow(2).sum() for name, p in model.mlp.named_parameters() if name.endswith("weight"))
    return mse + reg * l2


@pytest.mark.parametrize("tau,dropout_p", [(0.25, 0.005), (0.75, 0.1), (0.05, 0.05)])
def test_penalty_conversion_matches_gal_gradient(tau, dropout_p):
    """Our gradient must equal gal2016's, scaled by exactly `1/(2 sigma^2)`.

    Same weights, same batch, dropout switched off so the two losses see
    the same forward pass rather than two different dropout masks — the
    claim under test is about the objective, not about the noise.
    """
    n, sigma2 = 64, 1.0 / tau
    X, y = _batch(n=n)
    reg = _gal_reg(dropout_p, n, tau)
    scale = 1.0 / (2.0 * sigma2)

    ours = _model(sigma2)
    theirs = _model(sigma2)
    theirs.load_state_dict(ours.state_dict())

    mu, log_var = ours(X)
    loss_ours = gaussian_nll(mu, log_var, y) + ours.prior_penalty(_weight_only_penalty(sigma2, reg))
    loss_ours.backward()

    loss_gal = _gal_loss(theirs, X, y, reg)
    loss_gal.backward()

    for (name, p_ours), (_, p_gal) in zip(
        ours.mlp.named_parameters(), theirs.mlp.named_parameters()
    ):
        np.testing.assert_allclose(
            p_ours.grad.numpy(), scale * p_gal.grad.numpy(), rtol=1e-12, atol=1e-14,
            err_msg=f"gradient mismatch on {name} at tau={tau}, p={dropout_p}",
        )


def test_adam_step_matches_under_the_conversion():
    """Same first Adam step under both losses.

    This is the claim the run depends on: Adam's update is invariant to a
    constant rescaling of the loss (the scale cancels in `m_hat/sqrt(v_hat)`),
    so a correctly converted penalty reproduces the trajectory, not merely a
    proportional gradient. The agreement is ~1e-6 relative rather than exact
    because `eps=1e-8` sits in the denominator un-scaled; that residual is
    the only difference the conversion leaves behind, and the unconverted
    coefficient below shows what a real mismatch looks like by comparison.
    """
    tau, dropout_p, n = 0.25, 0.05, 64
    sigma2 = 1.0 / tau
    X, y = _batch(n=n)
    reg = _gal_reg(dropout_p, n, tau)

    def _run(penalty):
        model = _model(sigma2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
        for _ in range(3):
            opt.zero_grad()
            mu, log_var = model(X)
            (gaussian_nll(mu, log_var, y) + model.prior_penalty(penalty)).backward()
            opt.step()
        return {name: p.detach().numpy().copy() for name, p in model.mlp.named_parameters()}

    def _gal_run(reg_value):
        model = _model(sigma2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
        for _ in range(3):
            opt.zero_grad()
            _gal_loss(model, X, y, reg_value).backward()
            opt.step()
        return {name: p.detach().numpy().copy() for name, p in model.mlp.named_parameters()}

    def _worst_relative_gap(a, b):
        return max(
            float(np.max(np.abs(a[name] - b[name]) / (np.abs(b[name]) + 1e-12))) for name in a
        )

    assert _worst_relative_gap(_run(_weight_only_penalty(sigma2, reg)), _gal_run(reg)) < 1e-5

    # Same check where the penalty actually moves the trajectory. At
    # gal2016's own `reg` the L2 term is minute — `lengthscale=1e-2` puts it
    # at 3e-6 here and 7e-7 on `yacht`'s real N and tau — so small that even
    # an eightfold error in the conversion would not show up in three Adam
    # steps (measured: 1.2e-6 relative, i.e. the eps floor). That is worth
    # knowing about the protocol, but it means the trajectory check needs a
    # penalty with weight behind it to have any power; the algebra under
    # test does not depend on the magnitude.
    big = 0.1
    assert _worst_relative_gap(_run(_weight_only_penalty(sigma2, big)), _gal_run(big)) < 1e-5
    unconverted = {"linear1.weight": big, "linear1.bias": 0.0,
                   "mean_head.weight": big, "mean_head.bias": 0.0}
    assert _worst_relative_gap(_run(unconverted), _gal_run(big)) > 1e-3, (
        "the negative control did not diverge — this test would not detect a "
        "missing 1/(2 sigma^2) factor"
    )


def test_fixed_sigma2_from_tau_is_the_noise_gal_reports_with():
    """`sigma^2_standardised = 1 / (tau * scale^2)` puts our predictive noise
    where gal2016's `tau` puts his: `1/tau` in the ORIGINAL target units."""
    tau, scale = 0.25, 3.7
    sigma2_std = 1.0 / (tau * scale ** 2)
    model = _model(sigma2_std)
    _, log_var = model(_batch()[0])
    sigma2_original = math.exp(float(log_var)) * scale ** 2
    assert sigma2_original == pytest.approx(1.0 / tau, rel=1e-12)


def test_input_dropout_off_is_bit_identical():
    """The default must not perturb anything: same weights, same output."""
    X, _ = _batch()
    set_seed(0)
    without = HomoscedasticMLP(in_dim=5, hidden=8, dropout_p=0.1, always_on=True, activation="relu")
    set_seed(0)
    explicit = HomoscedasticMLP(in_dim=5, hidden=8, dropout_p=0.1, always_on=True,
                                activation="relu", input_dropout=False)
    for (_, a), (_, b) in zip(without.named_parameters(), explicit.named_parameters()):
        np.testing.assert_array_equal(a.detach().numpy(), b.detach().numpy())

    set_seed(1)
    out_without = without(X)[0].detach().numpy()
    set_seed(1)
    out_explicit = explicit(X)[0].detach().numpy()
    np.testing.assert_array_equal(out_without, out_explicit)


def test_input_dropout_on_zeroes_input_features():
    """With it on, whole input columns must vanish in some forward passes —
    gal2016 applies dropout to `inputs` itself, not only to the hidden layer.
    """
    set_seed(0)
    model = HomoscedasticMLP(in_dim=5, hidden=8, dropout_p=0.5, always_on=True,
                             activation="relu", input_dropout=True)
    X = torch.ones((1, 5), dtype=DTYPE)
    set_seed(3)
    masks = np.array([model.mlp.drop_input(X).numpy().ravel() for _ in range(200)])
    assert (masks == 0.0).any(), "input dropout never dropped a feature"
    # Inverted dropout: surviving entries are scaled by 1/(1-p), so the mask
    # mean is still 1 in expectation — the check that it is dropout and not
    # a plain zeroing.
    assert masks.mean() == pytest.approx(1.0, abs=0.05)
