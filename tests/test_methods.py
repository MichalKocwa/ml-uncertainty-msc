"""Etap 2 method tests (brief section 12): correct shapes, non-negative
variances, `map` has a zero epistemic term, and results are stable given
the same seed. All six methods (brief section 3/4) are covered.
"""
import numpy as np
import pytest

from src.data import load_sin_homo
from src.methods.backbone import HomoscedasticMLP
from src.methods.bbb import BBBMethod
from src.methods.ensemble import DeepEnsembleMethod
from src.methods.gp import GPMethod
from src.methods.laplace import LaplaceMethod
from src.methods.map import MAPMethod
from src.methods.mcd import MCDropoutMethod

# Small epoch/member counts — these tests check the protocol (shapes,
# signs, determinism), not fit quality, so they are tuned for runtime.
_FAST_KWARGS = {
    "map": dict(epochs=150),
    "mcd": dict(epochs=150, T=20),
    "ensemble": dict(epochs=150, M=3),
    "laplace": dict(epochs=150),  # default prior_precision_mode="fixed" — no marglik steps to configure
    "bbb": dict(epochs=150, T=20),
}

METHOD_FACTORIES = {
    "map": lambda: MAPMethod(**_FAST_KWARGS["map"]),
    "mcd": lambda: MCDropoutMethod(**_FAST_KWARGS["mcd"]),
    "ensemble": lambda: DeepEnsembleMethod(**_FAST_KWARGS["ensemble"]),
    "gp": lambda: GPMethod(),
    "laplace": lambda: LaplaceMethod(**_FAST_KWARGS["laplace"]),
    "bbb": lambda: BBBMethod(**_FAST_KWARGS["bbb"]),
}


@pytest.fixture(scope="module")
def toy_data():
    ds = load_sin_homo(seed=0)
    return ds.X_train, ds.y_train, ds.X_eval


@pytest.mark.parametrize("method_name", list(METHOD_FACTORIES))
def test_predict_shapes_and_nonnegative_variances(method_name, toy_data):
    X_train, y_train, X_eval = toy_data
    method = METHOD_FACTORIES[method_name]().fit(X_train, y_train, seed=0)
    pred = method.predict(X_eval)

    n = X_eval.shape[0]
    assert pred.mean.shape == (n,)
    assert pred.var_aleatoric.shape == (n,)
    assert pred.var_epistemic.shape == (n,)
    assert np.all(np.isfinite(pred.mean))
    assert np.all(pred.var_aleatoric >= 0)
    assert np.all(pred.var_epistemic >= 0)
    assert np.all(pred.var_total >= 0)
    assert method.n_parameters > 0


@pytest.mark.parametrize("method_name", list(METHOD_FACTORIES))
def test_stable_given_same_seed(method_name, toy_data):
    X_train, y_train, X_eval = toy_data
    pred_a = METHOD_FACTORIES[method_name]().fit(X_train, y_train, seed=7).predict(X_eval)
    pred_b = METHOD_FACTORIES[method_name]().fit(X_train, y_train, seed=7).predict(X_eval)

    np.testing.assert_allclose(pred_a.mean, pred_b.mean, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(pred_a.var_aleatoric, pred_b.var_aleatoric, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(pred_a.var_epistemic, pred_b.var_epistemic, rtol=1e-5, atol=1e-6)


def test_map_has_zero_epistemic_term(toy_data):
    X_train, y_train, X_eval = toy_data
    method = MAPMethod(**_FAST_KWARGS["map"]).fit(X_train, y_train, seed=0)
    pred = method.predict(X_eval)
    np.testing.assert_array_equal(pred.var_epistemic, np.zeros_like(pred.var_epistemic))


def test_map_aleatoric_variance_is_constant_across_points(toy_data):
    """Homoscedastic backbone: one global log_sigma2, so var_aleatoric must
    not vary across x (this is the point of the Etap 2 decision — see
    chat report)."""
    X_train, y_train, X_eval = toy_data
    method = MAPMethod(**_FAST_KWARGS["map"]).fit(X_train, y_train, seed=0)
    pred = method.predict(X_eval)
    assert np.allclose(pred.var_aleatoric, pred.var_aleatoric[0])


def test_fixed_sigma2_is_variance_not_std(toy_data):
    """`fixed_sigma2` must be a VARIANCE (sigma^2), not a standard deviation —
    `HomoscedasticMLP` stores it directly as `log(fixed_sigma2)`. Regression
    test for a real bug (docs/chapter4_notes.md, D-sigma-E1): E1's first
    recompute passed sin_homo's true `sigma=0.1` where `fixed_sigma2` needed
    `sigma**2=0.01`, giving `var_aleatoric=0.1` (10x too large) — passed all
    60 tests at the time because nothing checked the actual numeric value
    against a known ground truth, only shapes/signs/determinism.
    """
    X_train, y_train, X_eval = toy_data
    method = MAPMethod(**_FAST_KWARGS["map"], fixed_sigma2=0.01).fit(X_train, y_train, seed=0)
    pred = method.predict(X_eval)
    assert np.allclose(pred.var_aleatoric, 0.01)


def test_e1_known_homoscedastic_sigma_is_variance_not_std():
    """`experiments/e1_synthetic.py::KNOWN_HOMOSCEDASTIC_SIGMA` must hold
    sigma_true**2 (variance), not sigma_true (std) — same bug class as
    `test_fixed_sigma2_is_variance_not_std` above, but at the call site that
    actually broke (the dict's *value*, not the mechanism that consumes it).
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from experiments.e1_synthetic import KNOWN_HOMOSCEDASTIC_SIGMA
    from src.data import _sigma_homo

    true_sigma = float(_sigma_homo(np.array([0.0]))[0])
    assert KNOWN_HOMOSCEDASTIC_SIGMA["sin_homo"] == pytest.approx(true_sigma ** 2)
    assert KNOWN_HOMOSCEDASTIC_SIGMA["sin_gap"] == pytest.approx(true_sigma ** 2)
    assert "sin_hetero" not in KNOWN_HOMOSCEDASTIC_SIGMA  # no single true value to fix to — see D-sigma-E1


def test_ensemble_epistemic_variance_uses_sample_members_not_zero(toy_data):
    X_train, y_train, X_eval = toy_data
    method = DeepEnsembleMethod(**_FAST_KWARGS["ensemble"]).fit(X_train, y_train, seed=0)
    pred = method.predict(X_eval)
    assert pred.samples.shape == (_FAST_KWARGS["ensemble"]["M"], X_eval.shape[0])
    # far outside the training range [0, 6], members should disagree at least somewhat
    # (eval grid is [-2, 8] — see src/data.py; 7.0 stays comfortably inside it)
    far_idx = X_eval.ravel() > 7.0
    assert np.any(pred.var_epistemic[far_idx] > 1e-8)


# --------------------------------------------------------------------- #
# GP-specific
# --------------------------------------------------------------------- #
def test_gp_var_aleatoric_equals_fitted_white_kernel_noise(toy_data):
    X_train, y_train, X_eval = toy_data
    method = GPMethod().fit(X_train, y_train, seed=0)
    pred = method.predict(X_eval)
    np.testing.assert_allclose(pred.var_aleatoric, method.noise_level_)


def test_prior_penalty_excludes_noise_parameter(toy_data):
    """Documents the invariant `HomoscedasticMLP.prior_penalty` now enforces
    structurally (see its docstring): the penalty is hard-coded onto
    `self.mlp.parameters()`, so there is no call site left that could pass
    `model.parameters()` (including `log_sigma2`) by mistake — this test is
    a record of the bug that motivated that design, not a live guard
    against a call-site error that can no longer happen.

    Comparing *fitted* `log_sigma2` across different `gamma` is not a valid
    way to test this: a stronger weight prior legitimately worsens the fit,
    which legitimately inflates the optimal noise estimate through the
    residuals — that happens even with a fully correct implementation, so
    it would not distinguish "correct" from "leaking". Testing the
    gradient directly does: if `log_sigma2` is absent from the penalty's
    computation graph (as it must be), backpropagating the penalty alone
    leaves `log_sigma2.grad` as `None`, not zero — the parameter was never
    touched.
    """
    X_train, _, _ = toy_data
    model = HomoscedasticMLP(in_dim=X_train.shape[1], hidden=50, dropout_p=0.0)

    penalty = model.prior_penalty(coefficient=1.0)
    penalty.backward()

    assert model.log_sigma2.grad is None
    assert model.mlp.linear1.weight.grad is not None


# --------------------------------------------------------------------- #
# BBB-specific — same trap as prior_penalty, different guise: the KL term
# must not reach log_sigma2 either (brief section 4, this session's request 4).
# --------------------------------------------------------------------- #
def test_bbb_kl_loss_excludes_noise_parameter(toy_data):
    """`dnn_to_bnn` converts `nn.Linear` submodules in place; `log_sigma2`
    is a plain `nn.Parameter` on `HomoscedasticMLP`, never a `nn.Module`, so
    neither `dnn_to_bnn`'s module-tree walk nor `get_kl_loss`'s `m.modules()`
    walk can reach it structurally. Verified directly, the same way as
    `test_prior_penalty_excludes_noise_parameter`: backpropagating the KL
    term alone must leave `log_sigma2.grad` as `None`.
    """
    from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

    X_train, _, _ = toy_data
    model = HomoscedasticMLP(in_dim=X_train.shape[1], hidden=50, dropout_p=0.0)
    dnn_to_bnn(model.mlp, dict(
        prior_mu=0.0, prior_sigma=1.0, posterior_mu_init=0.0,
        posterior_rho_init=-3.0, type="Reparameterization", moped_enable=False,
    ))

    kl = get_kl_loss(model.mlp)
    kl.backward()

    assert model.log_sigma2.grad is None
    assert model.mlp.linear1.mu_weight.grad is not None


def test_bbb_epistemic_variance_is_nonzero(toy_data):
    """Sanity check that BBB's variational posterior actually produces
    disagreement across samples — not a shape/sign check duplicate of the
    generic parametrized tests, but a check that sampling is really
    happening (each `LinearReparameterization.forward` call draws fresh
    weights, so this would fail loudly if e.g. `T=1`)."""
    X_train, y_train, X_eval = toy_data
    method = BBBMethod(**_FAST_KWARGS["bbb"]).fit(X_train, y_train, seed=0)
    pred = method.predict(X_eval)
    assert pred.samples.shape == (_FAST_KWARGS["bbb"]["T"], X_eval.shape[0])
    assert np.any(pred.var_epistemic > 1e-10)


# --------------------------------------------------------------------- #
# Laplace-specific — P8: mean must be bit-identical to MAP
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("hessian_structure", ["full", "kron", "diag"])
def test_laplace_mean_and_aleatoric_variance_bit_identical_to_map(toy_data, hessian_structure):
    """P8 (brief section 11): Laplace's linearised predictive mean must
    equal the deterministic baseline's mean exactly, not approximately — a
    difference of even one bit means an extra forward pass or a different
    numerical path crept in. `torch.equal`/`np.array_equal`, no tolerance.
    Runs with the default `prior_precision_mode="fixed"` — deliberately not
    `"marglik"`: marglik-tuning `prior_precision` breaks the shared-gamma
    assumption (section 4.6 / D11), see the E1 diagnosis in the chat report
    and the module docstring.

    `var_aleatoric` is checked the same way: by construction (see
    `src/methods/laplace.py` docstring) `sigma_noise` is fixed from `map`'s
    training and never optimised — true regardless of `prior_precision_mode`
    — so it must also match `map` exactly.
    """
    X_train, y_train, X_eval = toy_data
    shared_kwargs = dict(hidden=50, gamma=1.0, epochs=150)

    map_pred = MAPMethod(**shared_kwargs).fit(X_train, y_train, seed=0).predict(X_eval)
    laplace_pred = LaplaceMethod(
        **shared_kwargs, hessian_structure=hessian_structure,
    ).fit(X_train, y_train, seed=0).predict(X_eval)

    assert np.array_equal(laplace_pred.mean, map_pred.mean)
    assert np.array_equal(laplace_pred.var_aleatoric, map_pred.var_aleatoric)


def test_laplace_prior_precision_matches_gamma_by_default(toy_data):
    """The bug this guards against (section 4.6 / D11): `prior_precision_mode`
    defaulting to `"marglik"` would let Laplace approximate a posterior
    under a *different* prior than every other method, silently. `"fixed"`
    must give exactly `1/gamma**2`, not merely something close to it."""
    X_train, y_train, _ = toy_data
    gamma = 1.0
    method = LaplaceMethod(epochs=150, gamma=gamma).fit(X_train, y_train, seed=0)
    assert method.la.prior_precision.item() == pytest.approx(1.0 / gamma ** 2, rel=1e-6)


def test_laplace_unregularised_variant_runs(toy_data):
    """E6c ablation (brief section 9): smoke test only — this configuration
    is diagnostic (over-estimated uncertainty is the expected/interesting
    result per P5), not something to assert a specific value for."""
    X_train, y_train, X_eval = toy_data
    method = LaplaceMethod(
        epochs=150, prior_precision_mode="unregularised",
    ).fit(X_train, y_train, seed=0)
    pred = method.predict(X_eval)
    assert np.all(np.isfinite(pred.var_epistemic))


def test_laplace_marglik_variant_still_available_for_e6c(toy_data):
    """`"marglik"` is no longer the default (see
    test_laplace_prior_precision_matches_gamma_by_default) but must stay
    available and functional — E6c ablates it explicitly against `"fixed"`
    and `"unregularised"`."""
    X_train, y_train, X_eval = toy_data
    method = LaplaceMethod(
        epochs=150, prior_precision_mode="marglik", marglik_steps=20,
    ).fit(X_train, y_train, seed=0)
    pred = method.predict(X_eval)
    assert np.all(np.isfinite(pred.var_epistemic))
    assert np.all(pred.var_epistemic >= 0)
    assert np.all(pred.var_epistemic >= 0)


def test_gp_epistemic_variance_saturates_far_from_training_data(toy_data):
    """Far from any training point, the RBF kernel correlation with the
    training set vanishes, so `Var[f(x*)] -> ConstantKernel` (the GP prior
    variance) and `var_total -> ConstantKernel + noise_level`."""
    X_train, y_train, _ = toy_data
    method = GPMethod().fit(X_train, y_train, seed=0)

    constant_value = float(method.gp.kernel_.k1.k1.constant_value)
    X_far = np.array([[1000.0]], dtype=np.float32)
    pred = method.predict(X_far)

    assert pred.var_epistemic[0] == pytest.approx(constant_value, rel=0.05)
    assert pred.var_total[0] == pytest.approx(constant_value + method.noise_level_, rel=0.05)


def test_depth_one_is_bit_identical_to_pre_depth_backbone():
    """`depth` must be a pure addition: at DEFAULT_DEPTH=1 the network is
    the same operations on the same parameters in the same order as before
    the parameter existed, so every already-published E1 number reproduces.

    Checked structurally (no extra layers, same module names) and
    numerically (a forward pass equals the explicit pre-depth expression
    bit-for-bit, not approximately).
    """
    import torch
    from src.methods.backbone import DEFAULT_DEPTH, MLP, count_parameters
    from src.seeding import set_seed

    assert DEFAULT_DEPTH == 1, "the E5 sweep must not have changed the shared default"

    set_seed(0)
    net = MLP(1, hidden=50)
    assert len(net.extra_hidden) == 0
    assert count_parameters(net) == 151  # 1*50+50 + 50*1+1

    x = torch.linspace(-2.0, 8.0, 97, dtype=torch.float64).reshape(-1, 1)
    net.eval()
    with torch.no_grad():
        got = net(x)
        # the pre-depth forward, written out
        expected = net.mean_head(net.drop_hidden(torch.tanh(net.linear1(x))))
    assert torch.equal(got, expected)


def test_depth_two_adds_one_hidden_layer_and_keeps_dropout_on_each():
    import torch
    from src.methods.backbone import MLP, count_parameters
    from src.seeding import set_seed

    set_seed(0)
    net = MLP(1, hidden=50, depth=2, dropout_p=0.5, always_on=True)
    assert len(net.extra_hidden) == 1
    assert count_parameters(net) == 2701  # 151 + (50*50 + 50)

    # dropout is live at BOTH hidden positions: with p=0.5 and always_on,
    # two forward passes must differ, and the variance must exceed what a
    # single stochastic layer of the same width would give at depth=1
    x = torch.zeros((8, 1), dtype=torch.float64) + 1.5
    with torch.no_grad():
        a, b = net(x), net(x)
    assert not torch.equal(a, b)


def test_frac_posterior_var_below_prior_is_relative_to_each_layer_prior():
    """`frac_posterior_var_below_prior` (D14d's diagnostic) must compare the
    posterior variance against that layer's OWN prior variance — the whole
    point of the metric is "collapsed relative to the prior", so a
    hard-coded threshold, or one read off a single global gamma, would
    silently mismeasure under D14c's layer-scaled prior.
    """
    import math

    import numpy as np
    import torch

    from src.methods.bbb import BBBMethod, frac_posterior_var_below_prior
    from src.seeding import set_seed

    set_seed(0)
    X = np.linspace(-1.0, 1.0, 32).reshape(-1, 1)
    y = np.sin(3 * X).ravel()
    method = BBBMethod(hidden=4, epochs=1).fit(X, y, seed=0, use_cache=False)

    layers = [method.model.mlp.linear1, method.model.mlp.mean_head]

    # every posterior sigma driven far below its prior -> everything pruned
    with torch.no_grad():
        for layer in layers:
            for rho, prior_sigma in ((layer.rho_weight, layer.prior_weight_sigma),
                                     (layer.rho_bias, layer.prior_bias_sigma)):
                target = 0.001 * prior_sigma  # variance 1e-6 x prior's, far under the 1% threshold
                rho.copy_(torch.log(torch.expm1(target)))
    result = frac_posterior_var_below_prior(method.model)
    assert result["weights"] == 1.0
    assert result["all"] == 1.0
    assert result["n_weights"] == 4 + 4  # 1*4 weights in, 4*1 weights out
    assert result["n_all"] == 8 + 4 + 1  # + 4 hidden biases + 1 output bias

    # every posterior sigma equal to its own prior -> nothing pruned, even
    # though the prior differs per layer under a layer-scaled prior
    with torch.no_grad():
        for i, layer in enumerate(layers):
            layer.prior_weight_sigma.fill_(0.5 if i == 0 else 2.0)
            layer.prior_bias_sigma.fill_(0.5 if i == 0 else 2.0)
            for rho, prior_sigma in ((layer.rho_weight, layer.prior_weight_sigma),
                                     (layer.rho_bias, layer.prior_bias_sigma)):
                rho.copy_(torch.log(torch.expm1(prior_sigma)))
    result = frac_posterior_var_below_prior(method.model)
    assert result["weights"] == 0.0
    assert result["all"] == 0.0

    # exactly at the boundary: sigma_post^2 = 0.01 * sigma_prior^2 is NOT
    # pruned (strict inequality); just below it is
    with torch.no_grad():
        layer = layers[0]
        layer.rho_weight.copy_(torch.log(torch.expm1(math.sqrt(0.01) * layer.prior_weight_sigma)))
    assert frac_posterior_var_below_prior(method.model)["weights"] == 0.0
    with torch.no_grad():
        layer.rho_weight.copy_(torch.log(torch.expm1(0.99 * math.sqrt(0.01) * layer.prior_weight_sigma)))
    assert frac_posterior_var_below_prior(method.model)["weights"] == 4 / 8


def test_frac_posterior_var_below_prior_rejects_a_non_bayesian_model():
    """A plain `HomoscedasticMLP` has no variational parameters at all — a
    zero fraction there would read as a perfectly healthy posterior."""
    import pytest

    from src.methods.backbone import HomoscedasticMLP
    from src.methods.bbb import frac_posterior_var_below_prior
    from src.seeding import set_seed

    set_seed(0)
    with pytest.raises(TypeError):
        frac_posterior_var_below_prior(HomoscedasticMLP(in_dim=1, hidden=4))
