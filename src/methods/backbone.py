"""Shared backbone for every NN-based method (brief section 4, "Wspólny backbone").

Homoscedastic single-head MLP: 1 hidden layer x 50 ReLU units, plus one
global learned scalar `log_sigma2`. Decided in Etap 2 (see chat report):
`laplace-torch`'s `sigma_noise` only accepts a scalar ("Only homoscedastic
output noise supported", baselaplace.py) and its `kron` Hessian structure
does not exist for subnetwork Laplace at all — so the two-head,
per-point-variance design from the brief's original section 4 text is not
implementable with the mandated library. `log_sigma2` plays the same role
as `WhiteKernel` in the GP: one constant estimated from data, with
identical status across every network method (map, mcd, ensemble, bbb,
laplace).

Same architecture for every NN method, per section 4's hard requirement
("Ta sama architektura ... dla BBB, MCD, LA, DE i baseline'u").
"""
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import prior_parametrisations
from src.methods import cache
from src.seeding import set_seed

# Numerical guard on `log_sigma2`, and NOTHING ELSE — it must never be the
# thing that determines a fitted noise level. Widened from (-6, 6) to
# (-12, 6) (author's decision, 2026-08-27) after
# `experiments/logvar_clamp_diagnostic.py` measured the old lower bound
# BINDING on 2 of the 6 UCI datasets: `map` on `yacht` reached it at epoch
# 342 and on `energy` at epoch 295 (3/3 seeds), with the raw parameter
# pushed past it (-6.05 on yacht) while the residual variance said
# `sigma^2` should have been 0.00069 against the 0.00248 = exp(-6) the
# clamp forced — 3.6x too high. That is a floor on the aleatoric term
# applied to every method at once, i.e. a noise-model hyperparameter
# masquerading as an overflow guard, and it is the same structural defect
# as one declared `batch_size` (D18) or one declared epoch count (O4)
# meaning different things on different datasets. It also fabricated a
# convergence plateau: `yacht`'s validation-NLL curve is flat from 500
# epochs on ONLY because of the clamp; unclamped it peaks at ~350 epochs
# and degrades after. Cost of the change, stated rather than hidden:
# `yacht`'s best validation NLL worsens from -1.41 to -1.16 (~0.25 nats),
# which is the part of that number the floor was providing.
#
# exp(-12) = 6.1e-6, below every fitted `sigma^2` measured on all six
# datasets (smallest: 0.00053 on `yacht`), so the bound is now inactive
# everywhere it was measured. See docs/chapter4_notes.md D29.
LOGVAR_CLAMP = (-12.0, 6.0)

# float64 everywhere (this session's decision, see chat report): networks here
# have 151-550 parameters and the largest dataset (E2) has 8611 rows, so the
# cost is negligible, and it removes an entire class of numerical questions —
# concretely, the Laplace `full`/`kron` GGN posterior precision at N < n_params
# (e.g. the original brief's N=50, 151 params) is severely ill-conditioned
# (cond ~1e5-1e6) and float32's ~7 digits of precision are not enough to
# invert it reliably (diagnosed via a PSD violation of the GGN+prior_precision*I
# matrix far exceeding the float32 rounding floor, `largest_eigval * eps_f32`,
# but within the float64 rounding floor once N was also raised above
# n_params — see docs/chapter4_notes.md). `set_seed()` sets this globally so
# every model constructed after it (including `bayesian_torch`'s
# `LinearReparameterization`, which has no `dtype` constructor argument at
# all) picks it up automatically — see bbb.py's module docstring.
DTYPE = torch.float64

# Shared minibatch size across every NN method (this session's decision —
# see docs/chapter4_notes.md). Belongs to the "controlled" column of
# section 4.1's table, not the "varies per method" one: batch size sets
# the number of optimizer steps per epoch (`ceil(N/batch_size)`), and a
# method-specific value would make that a second axis of variation
# alongside the posterior approximation — at N=8611 (E2's largest set),
# `batch_size=32` vs full-batch is a 270x difference in steps/epoch, an
# entirely different optimization regime, not a cost detail. `128`, not
# full-batch: deep ensembles (lakshminarayanan2017) draw part of their
# diversity from minibatch order, not only initialisation, so full-batch
# would be a deviation from the method as published, not just a slower one.
DEFAULT_BATCH_SIZE = 128

# Shared prior std across every NN method (D11, section 4.6) — must be
# identical everywhere by construction, so it lives here rather than as an
# independently-typed literal in five constructors.
#
# Reverted to 1.0 (this session) after a brief detour to 0.3. 0.3 did cut
# the Laplace `var_epistemic` jump at ReLU kink boundaries (~4.75x, no
# in-range fit degradation, PICP@95 stayed >=0.90 — see
# docs/chapter4_notes.md D11b for the full sweep over {1.0, 0.5, 0.3, 0.1}),
# but it moves the prior itself: the thesis's claim is that all six methods
# approximate the *same* posterior and differ only in how they approximate
# it, and gamma is part of that posterior's definition (D11) — picking it
# to make one method's diagnostic plot look better is choosing a different
# Bayesian model per method's convenience, not a neutral fix. 0.3 also
# nets out worse across methods, not better: it visibly degraded MCD and
# Laplace's fit while only helping ensemble. 1.0 is the standard unit
# Gaussian prior, chosen independently of how the results look.
DEFAULT_GAMMA = 1.0

# Shared hidden-layer activation across every NN method (D7/E6d, this
# session's decision — see docs/chapter4_notes.md). Was "relu" (Hernandez-
# Lobato & Adams protocol, the literature default this thesis otherwise
# follows); switched to "tanh" after an explicit sweep comparing the two at
# gamma=1.0, 6 methods x 3 seeds x 3 datasets:
#   - Laplace's `var_epistemic` jump at ReLU activation-boundary kinks
#     (J(x) discontinuous under the linearised predictive f_var = J^T Sigma J)
#     is a real property of ReLU, not of Laplace — it disappears at the
#     source under tanh's continuous Jacobian (~99-500x reduction in
#     max grid-adjacent |delta var|, not merely smoothed).
#   - BBB's left/right extrapolation-band asymmetry (~10-12x under relu)
#     collapses to ~1.0x under tanh.
#   - relu/gamma=1.0 pushes laplace's and bbb's predictive bands outside
#     Y_RANGE=[-5,5] (laplace reaches [-6.69, 4.76] on sin_hetero); tanh's
#     bands stay inside it for every method/dataset tested.
#   - laplace/ensemble/map extrapolation RMSE and LL improve under tanh.
# Trade-off, not a free win: BBB and MCD's extrapolation PICP@95 drops
# sharply under tanh (e.g. BBB sin_homo: 0.64 -> 0.35) — judged as these two
# methods' known failure modes (BBB underestimates variance; MCD is
# overconfident) surfacing honestly, rather than relu's piecewise-linear
# extrapolation accidentally widening their bands and masking it. Full
# comparison table: docs/chapter4_notes.md, D7b. Reverting to relu remains
# available (E6d ablation, and P13's MC-dropout literature-validation run,
# which must match Gal's own relu-based protocol — see laplace.py/mcd.py).
DEFAULT_ACTIVATION = "tanh"

# Shared hidden-layer count across every NN method. **1, unchanged** — this
# constant exists so that E5 (brief section 9: "depth ablation, depth in
# {1,2,4} x all methods x 2 datasets") can vary depth explicitly per method
# instance, NOT because the default moved. `depth=1` is bit-identical to the
# backbone as it stood before this parameter existed (see `MLP`'s docstring).
#
# Why the ablation is worth running despite D14j's caveats: foong2020's
# universality result for >=2 hidden layers is an EXISTENCE proof, so depth
# cannot be promised as a fix — but that is a reason not to promise it, not
# a reason not to measure it (author, 2026-08-26). Two hidden layers are
# also ordinary in this line of work: foong2020 reports MFVI at 2 hidden
# layers alongside 1 on the same UCI sets. The N/p objection (2x50 gives
# ~2701 parameters against N=250, N/p ~ 0.09) is weakened by this project's
# own width ablation (D-width-E5): at h=200, N/p = 0.42 — the same N < p
# regime that produced the original Laplace needle spikes at N=50 — and the
# needle did NOT return. N/p alone was already shown not to be the
# controlling quantity there.
DEFAULT_DEPTH = 1


class AlwaysOnDropout(nn.Module):
    """Dropout that samples regardless of `model.train()`/`model.eval()`.

    `torch.nn.Dropout` is a no-op under `model.eval()`, which is exactly the
    mode predictions are normally made in — with plain `nn.Dropout`, MC
    dropout sampling silently does not happen at predict time unless the
    caller remembers to leave the model in `.train()` mode. This module
    makes that impossible to forget: it always applies dropout when `p > 0`,
    independent of the module's training flag.
    """

    def __init__(self, p: float):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return x
        return F.dropout(x, p=self.p, training=True)


class MLP(nn.Module):
    """`input -> [Linear(hidden) -> activation -> Dropout] x depth -> Linear(1)`,
    activation tanh by default (see `DEFAULT_ACTIVATION`).

    Single mean head. `dropout_p=0.0` (the map/ensemble/laplace default)
    makes every dropout position identity, so this one module is the
    shared architecture across all five NN-based methods.

    **`depth` (E5, brief section 9's depth ablation).** `DEFAULT_DEPTH = 1`
    keeps the historical `input -> Linear(h) -> act -> Dropout -> Linear(1)`
    network EXACTLY — at `depth=1` the `extra_hidden` list is empty and the
    forward pass is the same sequence of operations on the same parameters
    in the same order, so every result produced before `depth` existed
    reproduces bit-identically. Verified by
    `tests/test_methods.py::test_depth_one_is_bit_identical_to_pre_depth_backbone`.

    **Dropout placement at `depth > 1`: after EVERY hidden activation, not
    only the last.** This is the standard multi-layer MC dropout and the
    configuration Foong et al.'s theorem (docs/chapter4_notes.md D14j)
    speaks about; putting dropout only before the output layer would leave
    a `depth=2` network with the same single stochastic layer as `depth=1`,
    which would test nothing. Note this does mean MC dropout's total
    injected noise grows with depth — that is a property of the method
    being tested, not a confound introduced here, and it is exactly why the
    E5 sweep reports in-range fit quality alongside the uncertainty shape.
    Input->hidden dropout stays removed at every depth, for the reason
    below.

    Deviation from gal2016's literal protocol (dropout before *every*
    weighted layer, including input->hidden), and from the brief's section
    4.3 as originally written — deliberate, not an oversight. At `d=1`
    (the synthetic 1D datasets), input dropout zeroes the network's *only*
    feature in `dropout_p` of forward passes, at both train and predict
    time (`AlwaysOnDropout` samples regardless of mode). During training
    this corrupts a fraction of batches to `x=0` outright; the network
    compensates by inflating `log_sigma2` to cover the resulting loss
    spikes, not because the data-generating noise is that large — measured
    on `sin_homo` (true `sigma^2=0.01`): `mean_var_aleatoric` was `0.122`,
    13x too large, before removing input dropout; `0.038` after. Re-measured
    2026-08-31 through the `input_dropout` flag (tanh, 2000 epochs, seed 0,
    one thread): 0.106 with input dropout against 0.038 without, which
    confirms the post-fix figure and corrects the `0.0096` this docstring
    carried until then — see `docs/chapter4_notes.md` D15. Removed
    for all datasets, not only `d=1`, rather than special-casing by
    dimensionality: the failure mode (dropped features absorbed into the
    noise estimate) is the same mechanism at any `d`, just proportionally
    smaller when more than one feature survives a dropped mask.

    **`input_dropout=True` puts it back, for P13 only** (author's decision,
    2026-08-31). P13 asks whether *our* MC dropout reproduces gal2016's
    published numbers once the protocol is equalised, and his `net.py`
    applies `Dropout(p)(inputs, training=True)` before the first Dense — so
    omitting it would be an unforced deviation, and an alternative
    explanation left standing if the gap failed to close. The `d=1`
    argument above does not transfer to P13's datasets (`yacht` has 6
    features, `energy` 8, `concrete` 8), where a dropped feature leaves the
    rest of the input intact. The default stays `False`, and every result
    produced before this flag existed is unchanged: `nn.Identity` consumes
    no RNG draws and holds no parameters, so `input_dropout=False` is the
    same forward pass on the same weights in the same order as before.
    """

    def __init__(
        self, in_dim: int, hidden: int = 50, dropout_p: float = 0.0, always_on: bool = False,
        activation: str = DEFAULT_ACTIVATION, depth: int = DEFAULT_DEPTH,
        input_dropout: bool = False,
    ):
        super().__init__()
        if activation not in ("relu", "tanh"):
            raise ValueError(f"activation must be 'relu' or 'tanh', got {activation!r}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        drop_cls = AlwaysOnDropout if always_on else nn.Dropout
        self.depth = depth
        self.drop_input = drop_cls(dropout_p) if input_dropout else nn.Identity()
        self.linear1 = nn.Linear(in_dim, hidden, dtype=DTYPE)
        # `depth - 1` further hidden->hidden layers; empty at DEFAULT_DEPTH=1, which is
        # what makes depth=1 bit-identical to the pre-depth version of this class
        self.extra_hidden = nn.ModuleList(
            [nn.Linear(hidden, hidden, dtype=DTYPE) for _ in range(depth - 1)]
        )
        self.drop_hidden = drop_cls(dropout_p)
        self.mean_head = nn.Linear(hidden, 1, dtype=DTYPE)
        self.activation = activation
        self._activation_fn = torch.tanh if activation == "tanh" else torch.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop_hidden(self._activation_fn(self.linear1(self.drop_input(x))))
        for layer in self.extra_hidden:
            h = self.drop_hidden(self._activation_fn(layer(h)))
        return self.mean_head(h)


class HomoscedasticMLP(nn.Module):
    """`MLP` plus one global `log_sigma2`, clamped to [-6, 6] (post-standardisation).

    `log_sigma2` is learned by default (`fixed_sigma2=None`). Passing
    `fixed_sigma2` registers it as a plain buffer instead of an
    `nn.Parameter` — no gradient, never touched by the optimizer — fixed to
    `log(fixed_sigma2)` for the model's lifetime. Only meaningful where the
    true noise level is known by construction (synthetic data, D14c's Foong-
    setup variant comparison, `docs/chapter4_notes.md`): fixing it on real
    (UCI) data would not be a fair comparison, since the true noise there is
    exactly what every method has to estimate.

    `logvar_clamp` defaults to the shared `LOGVAR_CLAMP` and exists so that
    a diagnostic can ask whether that bound is BINDING (i.e. whether the
    flat tail of a loss curve is the model converging or `sigma^2` sitting
    on the clamp floor) without changing the constant every method shares
    — see `experiments/logvar_clamp_diagnostic.py`. Passing anything other
    than the default changes the model, so it must never be silently
    different between two methods being compared.
    """

    def __init__(
        self, in_dim: int, hidden: int = 50, dropout_p: float = 0.0, always_on: bool = False,
        activation: str = DEFAULT_ACTIVATION, fixed_sigma2: float = None, depth: int = DEFAULT_DEPTH,
        logvar_clamp: Tuple[float, float] = LOGVAR_CLAMP, input_dropout: bool = False,
    ):
        super().__init__()
        self.logvar_clamp = tuple(logvar_clamp)
        self.mlp = MLP(in_dim, hidden=hidden, dropout_p=dropout_p, always_on=always_on,
                       activation=activation, depth=depth, input_dropout=input_dropout)
        if fixed_sigma2 is None:
            self.log_sigma2 = nn.Parameter(torch.zeros((), dtype=DTYPE))
        else:
            if fixed_sigma2 <= 0:
                raise ValueError(f"fixed_sigma2 must be positive, got {fixed_sigma2}")
            self.register_buffer("log_sigma2", torch.tensor(math.log(fixed_sigma2), dtype=DTYPE))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mlp(x)
        log_var = self.log_sigma2.clamp(*self.logvar_clamp)
        return mu, log_var

    def prior_penalty(self, coefficient) -> torch.Tensor:
        """`coefficient * sum(theta**2)` over `self.mlp.parameters()` (brief
        section 4.6) — weights and biases alike, since the brief's prior is
        stated generically over theta with no weight/bias distinction.

        `coefficient` is a scalar (D11's flat `N(0, gamma^2 I)`, the
        default) or a `{parameter_name: coefficient}` dict keyed by
        `self.mlp.named_parameters()`'s names (D14c's layer-scaled Foong
        prior — `config.layerwise_prior_sigma` computes the per-name
        values). Both paths still only ever touch `self.mlp.parameters()`.

        Deliberately *not* a free function taking a parameter list: that
        shape lets a call site pass `model.parameters()` by mistake and
        silently regularise `log_sigma2` too (caught by inspection once
        already — see `tests/test_methods.py::test_prior_penalty_excludes_noise_parameter`).
        Hard-coding `self.mlp.parameters()`/`self.mlp.named_parameters()`
        here removes the call site that could get this wrong: there is only
        one way to compute this model's prior penalty, and it is this method.
        """
        if isinstance(coefficient, dict):
            return sum(coefficient[name] * p.pow(2).sum() for name, p in self.mlp.named_parameters())
        return coefficient * sum(p.pow(2).sum() for p in self.mlp.parameters())


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean-reduced Gaussian NLL, up to the constant 0.5*log(2*pi) term.

    `log_var` may be a scalar (homoscedastic) or per-point tensor; broadcasts
    either way.
    """
    var = torch.exp(log_var)
    return torch.mean(0.5 * log_var + 0.5 * (y - mu) ** 2 / var)


def train_homoscedastic_mlp(
    model_factory,
    X: "np.ndarray",
    y: "np.ndarray",
    seed: int,
    gamma: float,
    epochs: int,
    lr: float = 1e-2,
    batch_size: int = None,
    penalty_coefficient_override=None,
    use_cache: bool = True,
    cache_key_extra: dict = None,
    epoch_callback=None,
) -> HomoscedasticMLP:
    """Shared training loop for map / mcd / ensemble members (brief section 4.6).

    `model_factory` is a zero-arg callable that constructs a fresh
    `HomoscedasticMLP` — it must be called *after* `set_seed`, not before,
    since `nn.Linear` draws its initial weights at construction time. Taking
    a factory instead of an already-built model makes that ordering
    impossible to get backwards by accident.

    Adam, `weight_decay=0`, explicit `prior_penalty_coefficient * sum(theta**2)`
    term in the loss (recommended path — see `src/config.py`). `batch_size=None`
    trains full-batch; otherwise minibatches are reshuffled every epoch from
    the per-call `seed`, which is how deep-ensemble members get both a
    different initialisation and a different batch order (section 4.5) from
    a single base seed.

    `penalty_coefficient_override` (scalar or `{name: coefficient}` dict,
    see `HomoscedasticMLP.prior_penalty`): when given, replaces the
    `gamma`-derived coefficient entirely — `gamma` is still a required
    argument (keeps the normal D11 call sites unchanged) but is then unused.
    Only exists for D14c's layer-scaled Foong-prior variant comparison
    (`docs/chapter4_notes.md`), not part of the default training path.

    `epoch_callback(epoch, model)`, if given, is called after every epoch,
    purely to OBSERVE (it must not touch the model or the RNG). It exists
    for `experiments/logvar_clamp_diagnostic.py`, which needs the
    `log_sigma2` trajectory that a final model does not preserve. Because
    the training trajectory is deterministic given the seed, one
    instrumented run of `E` epochs sees exactly the models that `E`
    separate runs of `1..E` epochs would produce. Passing a callback
    DISABLES the cache: a cache hit skips training and would silently
    produce no epochs at all.

    `use_cache`/`cache_key_extra` (see `src/methods/cache.py`): `model_factory`
    is an opaque closure, so whatever it bakes in beyond this function's own
    arguments (`hidden`, `activation`, `dropout_p`, `always_on`,
    `fixed_sigma2`) must be passed explicitly via `cache_key_extra` by the
    caller, or two different architectures trained with otherwise-identical
    arguments would collide in the cache. The cache stores only
    `model.state_dict()` — on a hit, `model_factory()` still runs (rebuilds
    the right architecture, including any deterministic prior buffers a
    caller like `BBBMethod` sets up) and the trained weights are loaded into
    it; the model's random initial values are irrelevant on a hit, since
    `load_state_dict` overwrites them.
    """
    n = X.shape[0]
    cache_path = None
    if epoch_callback is not None:
        use_cache = False
    if use_cache:
        config = dict(
            gamma=gamma, epochs=epochs, lr=lr, batch_size=batch_size,
            penalty_coefficient_override=penalty_coefficient_override,
            **(cache_key_extra or {}),
        )
        cache_path = cache.cache_path("homoscedastic_mlp", config, X, y, seed)
        cached = cache.load(cache_path)
        if cached is not None:
            model = model_factory()
            model.load_state_dict(cached["state_dict"])
            return model

    set_seed(seed)
    model = model_factory()
    if penalty_coefficient_override is not None:
        penalty_coef = penalty_coefficient_override
    else:
        penalty_coef = prior_parametrisations(gamma, n).prior_penalty_coefficient

    X_t = torch.as_tensor(X, dtype=DTYPE)
    y_t = torch.as_tensor(y, dtype=DTYPE).reshape(-1, 1)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    bs = n if batch_size is None else min(batch_size, n)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            opt.zero_grad()
            mu, log_var = model(X_t[idx])
            loss = gaussian_nll(mu, log_var, y_t[idx]) + model.prior_penalty(penalty_coef)
            loss.backward()
            opt.step()
        if epoch_callback is not None:
            epoch_callback(epoch + 1, model)

    if use_cache:
        cache.save(cache_path, {"state_dict": model.state_dict()})
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
