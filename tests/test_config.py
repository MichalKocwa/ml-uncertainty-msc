import numpy as np
import pytest
import torch

from src.config import prior_parametrisations


def _log_density(w, gamma):
    """log N(w; 0, gamma^2 I), up to additive constants that cancel in comparisons."""
    d = w.size
    return -0.5 * np.sum(w ** 2) / gamma ** 2 - 0.5 * d * np.log(2 * np.pi * gamma ** 2)


@pytest.mark.parametrize("gamma,n", [(1.0, 50), (0.1, 277), (3.5, 8611)])
def test_prior_parametrisations_describe_the_same_distribution(gamma, n):
    params = prior_parametrisations(gamma, n)

    rng = np.random.RandomState(0)
    w = rng.randn(37)

    gamma_from_prior_sigma = params.prior_sigma
    gamma_from_precision = 1.0 / np.sqrt(params.prior_precision)
    # Inverted independently from src/config.py's own formulas, so this test
    # cannot pass merely by inverting whatever config.py happens to compute.
    gamma_from_penalty_coefficient = 1.0 / np.sqrt(2.0 * params.prior_penalty_coefficient * n)
    gamma_from_weight_decay = 1.0 / np.sqrt(params.weight_decay * n)
    gamma_from_numpyro = params.numpyro_scale

    # Pin every parametrisation to the actual input, not just to each other —
    # a consistent shift across all four would otherwise pass silently.
    for recovered in [gamma_from_prior_sigma, gamma_from_precision,
                      gamma_from_penalty_coefficient, gamma_from_weight_decay,
                      gamma_from_numpyro]:
        assert recovered == pytest.approx(gamma, rel=1e-9)

    densities = [
        _log_density(w, gamma_from_prior_sigma),
        _log_density(w, gamma_from_precision),
        _log_density(w, gamma_from_penalty_coefficient),
        _log_density(w, gamma_from_weight_decay),
        _log_density(w, gamma_from_numpyro),
    ]

    for d in densities[1:]:
        assert d == pytest.approx(densities[0], rel=1e-9)


def test_weight_decay_matches_explicit_prior_penalty():
    """Documents the relationship between the two ways `prior_parametrisations`
    exposes the same regularisation strength — `prior_penalty_coefficient`
    (recommended: an explicit loss term) and `weight_decay` (fallback, SGD
    only) — rather than guarding against the sign/factor bug it originally
    caught (see git history / src/config.py docstring for that story).

    `torch.optim`'s `weight_decay=wd` adds `wd*theta` directly to the
    gradient (SGD, no momentum: `p -= lr*(grad + wd*p)`). An explicit loss
    term `prior_penalty_coefficient * sum(theta**2)` contributes gradient
    `2*prior_penalty_coefficient*theta` (differentiating the square).
    Matching the two requires `weight_decay = 2*prior_penalty_coefficient`,
    which is exactly what `prior_parametrisations` returns. Verified
    empirically: two identical models, zero data-loss gradient, one step of
    plain SGD (no momentum, where the weight_decay convention is
    unambiguous) must land on the same parameters whether the penalty is
    applied via `weight_decay` or written out explicitly in the loss.
    """
    gamma, n = 0.7, 37
    params = prior_parametrisations(gamma, n)
    assert params.weight_decay == pytest.approx(2.0 * params.prior_penalty_coefficient, rel=1e-12)

    lr = 0.1
    torch.manual_seed(0)
    model_a = torch.nn.Linear(5, 3)
    model_b = torch.nn.Linear(5, 3)
    model_b.load_state_dict(model_a.state_dict())

    opt_a = torch.optim.SGD(model_a.parameters(), lr=lr, weight_decay=params.weight_decay, momentum=0.0)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=lr, weight_decay=0.0, momentum=0.0)

    # Model A: zero-valued "data loss" that still has a real (zero) gradient,
    # so weight_decay is the only thing that moves the parameters.
    loss_a = sum((0.0 * p.pow(2).sum()) for p in model_a.parameters())
    opt_a.zero_grad()
    loss_a.backward()
    opt_a.step()

    # Model B: no weight_decay; the prior penalty is written explicitly —
    # this is the recommended path for MAP/MC dropout (see config.py).
    loss_b = sum((params.prior_penalty_coefficient * p.pow(2).sum()) for p in model_b.parameters())
    opt_b.zero_grad()
    loss_b.backward()
    opt_b.step()

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(p_a, p_b, rtol=1e-6, atol=1e-8)


def test_prior_parametrisations_rejects_invalid_input():
    with pytest.raises(ValueError):
        prior_parametrisations(gamma=0.0, n=50)
    with pytest.raises(ValueError):
        prior_parametrisations(gamma=1.0, n=0)
