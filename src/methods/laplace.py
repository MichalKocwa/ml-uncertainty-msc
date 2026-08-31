"""Laplace approximation (brief section 4.4, `laplace-torch` / Daxberger et al. 2021).

Full-network Laplace (`subset_of_weights="all"`) around the same MAP model
trained by `MAPMethod` — subnetwork Laplace was ruled out in Etap 2 (see
chat report): `sigma_noise` only accepts a scalar, and `kron` does not
exist at all for subnetwork Laplace. Both restrictions disappear in
full-network mode with the homoscedastic backbone, which is why that
backbone was adopted.

Predictive mean is `la()`'s own linearised `f_mu`, not a separately
computed forward pass — verified empirically (`torch.equal`, all three
Hessian structures) to equal a plain forward through the same weights,
which is the property P8 asserts. `var_aleatoric` is taken from the exact
`log_sigma2` learned during MAP training (not round-tripped through
`sigma_noise`'s sqrt/square), so it is bit-identical to `MAPMethod`'s
`var_aleatoric` too — see the "does marglik tune sigma_noise" note below.

**Environment note:** `laplace-torch==0.2.3`'s `kron` backend calls
`KFACLinearOperator._compute_kfac()`, a method renamed to
`compute_kronecker_factors()` in `curvlinops-for-pytorch>=3.0` (no upper
bound is pinned by `laplace-torch`, so a plain `pip install` pulls the
incompatible 3.0.0). Fixed by pinning `curvlinops-for-pytorch==2.0.1` in
`requirements.txt`.

**Does marglik tune `sigma_noise` too?** No — verified empirically.
`Laplace.optimize_prior_precision(method="marglik")` only makes
`log_prior_precision` differentiable and optimises it; `sigma_noise` is
left untouched (confirmed: identical to 6 decimal places before/after, for
all three Hessian structures). The module-level `laplace.marglik_training`
helper *does* tune both jointly, but it retrains a network from scratch —
not used here, since it would not share weights with `MAPMethod` and
would break P8. Consequence: `var_aleatoric` from `laplace` is exactly the
`var_aleatoric` from `map`, not merely close to it — the two methods
differ only in `var_epistemic`.

**`prior_precision_mode="fixed"` is the default, not `"marglik"` — reversed
from the first cut of this class, per E1 diagnosis (see chat report).**
Marglik-tuning `prior_precision` picks whatever value maximises the
marginal likelihood, which is *not* `1/gamma**2` for the shared `gamma` —
on `sin_homo`/seed 0 it drifted to `prior_precision=0.519`, i.e. an
implied `gamma=1.388` instead of the shared `gamma=1.0`. That breaks
section 4.6 (D11): Laplace would then be approximating a posterior under a
*different* prior than BBB/MAP/ensemble, so the comparison stops being
"different posterior approximations of the same model" and starts being
"different models". `"fixed"` sets `prior_precision = prior_parametrisations(gamma,
n).prior_precision` and never calls `optimize_prior_precision` — the same
gamma as everywhere else, by construction, not by coincidence.

`"marglik"` and `"unregularised"` are kept as explicit, opt-in
configurations for the E6c ablation (brief section 9), which now compares
three `prior_precision` variants, not two. Both are also implicated in the
E1 diagnosis: `"unregularised"` (`prior_precision=1e-4`) makes the
posterior precision matrix numerically indefinite outright (Cholesky
fails, "leading minor ... is not positive-definite") — the
`prior_precision`-vs-conditioning relationship in `"marglik"`'s pathology
(see below) is not a one-off, it is the same fragility at a different
point on the same curve.

**The `var_epistemic` spike — root cause and fix (full history in
docs/chapter4_notes.md, this was a multi-session diagnosis).** With
`sin_homo`'s original brief-specified N=50, `full`/`kron` `var_epistemic`
showed narrow (~0.02-wide, grid-density-independent, so real not a
sampling artifact), reproducible needle spikes far inside the training
range, in most seeds tried, in *both* float32 and float64. Traced through
several false leads (`curvlinops`'s confirmed-but-minor float32-downcast
internal to its GGN computation, ≤0.04% relative error — real, but not the
cause) to the actual mechanism: N=50 < ~151 network parameters leaves the
GGN Hessian rank-deficient (rank ≤ N), so in `full`/`kron` mode most of
parameter space is purely prior-determined; the resulting posterior
precision has condition number ~1e5-1e6 regardless of float precision,
and a `full`/`kron` linearised-variance evaluation at a specific x can hit
that ill-conditioning hard. Root-caused, not float-precision noise: raising
N above n_parameters (this session's decision, `src/data.py`, N=50->250)
gives the GGN full rank and eliminates the null space that produces the
spikes — verified (see chat report) on a 6-seed sweep before this was
adopted as the default. `float64` (project-wide, `src/methods/backbone.py`'s
`DTYPE` via `set_seed`) is kept regardless: negligible cost at these
network/dataset sizes, and it removes float32 as a variable across every
method, not just Laplace.

**A second, distinct `var_epistemic` discontinuity — not this one, and not
fixed by N.** Separately from the rank-deficiency spikes above, the
linearised predictive `f_var(x) = J(x)^T Sigma J(x)` has a genuine
discontinuity at every ReLU activation-boundary kink (`J(x)` is a step
function of which hidden units are active) — confirmed by an exact
coincidence between jump locations and computed kink locations to 6
decimal places (`docs/chapter4_notes.md`, part E). This is why
`DEFAULT_ACTIVATION` in `backbone.py` is `"tanh"`, not `"relu"`: tanh's
smooth Jacobian removes this discontinuity at the source (~99-500x
reduction in max grid-adjacent |delta var_epistemic|, measured across all
three E1 datasets — see `docs/chapter4_notes.md` D7b). Passing
`activation="relu"` reproduces it, by design — that is the E6d ablation.
"""
import numpy as np
import torch
from laplace import Laplace
from torch.utils.data import DataLoader, TensorDataset

from src.config import layerwise_penalty_coefficients, layerwise_prior_precisions, prior_parametrisations
from src.methods.backbone import (
    DEFAULT_ACTIVATION, DEFAULT_BATCH_SIZE, DEFAULT_DEPTH, DEFAULT_GAMMA, DTYPE, LOGVAR_CLAMP, HomoscedasticMLP,
    count_parameters, train_homoscedastic_mlp,
)
from src.methods.base import Prediction

# E6c ablation (brief section 9): "unregularised" full Laplace, prior_precision
# near zero rather than tuned by marglik. Not exactly 0 — an exactly-zero
# prior precision leaves the posterior precision singular. 1e-4 (tried
# first) still failed Cholesky even at N=250 (full-rank GGN): the smallest
# genuine GGN eigenvalue is apparently smaller than that. 1e-3 is
# numerically safe (checked directly); still gives deliberately large
# var_epistemic (E6c's point — P5 predicts overestimation), not a
# borderline value chosen to "just barely work". Arbitrary otherwise,
# flagged for verification alongside T/batch_size (E6a/Etap 4 territory).
UNREGULARISED_PRIOR_PRECISION = 1e-3


class LaplaceMethod:
    name = "laplace"

    def __init__(
        self,
        hidden: int = 50,
        gamma: float = DEFAULT_GAMMA,
        epochs: int = 2000,
        lr: float = 1e-2,
        hessian_structure: str = "full",  # "full" | "kron" | "diag" — all three needed for Figure 3.9 (E6c)
        prior_precision_mode: str = "fixed",  # "fixed" (default: prior_precision = 1/gamma**2, matches every other method) | "marglik" | "unregularised" (both E6c-only ablations)
        marglik_steps: int = 100,
        batch_size: int = DEFAULT_BATCH_SIZE,  # for the MAP backbone's training only — see fit(); la.fit()'s own loader stays full-batch, it accumulates curvature, not gradient steps
        activation: str = DEFAULT_ACTIVATION,  # "tanh" (default, D7b) | "relu" — E6d ablation, see backbone.py's DEFAULT_ACTIVATION. tanh's continuous Jacobian is why this method's `var_epistemic` jump (see module docstring) disappears at the source under the default.
        fixed_sigma2: float = None,  # D14c Foong-setup variant only, synthetic data only — see HomoscedasticMLP
        layerwise_prior_omega: float = None,  # D14c Foong-setup variant only — overrides gamma with the layer-scaled prior for BOTH the MAP backbone's training penalty and `Laplace(prior_precision=...)`, see config.layerwise_prior_sigma
        depth: int = DEFAULT_DEPTH,  # E5 depth ablation (brief section 9) — 1 is the shared default and is bit-identical to the pre-depth backbone; see backbone.py's DEFAULT_DEPTH
    ):
        if hessian_structure not in ("full", "kron", "diag"):
            raise ValueError(f"hessian_structure must be 'full', 'kron' or 'diag', got {hessian_structure!r}")
        if prior_precision_mode not in ("fixed", "marglik", "unregularised"):
            raise ValueError(
                f"prior_precision_mode must be 'fixed', 'marglik' or 'unregularised', got {prior_precision_mode!r}"
            )
        self.hidden = hidden
        self.gamma = gamma
        self.epochs = epochs
        self.lr = lr
        self.hessian_structure = hessian_structure
        self.prior_precision_mode = prior_precision_mode
        self.marglik_steps = marglik_steps
        self.batch_size = batch_size
        self.activation = activation
        self.fixed_sigma2 = fixed_sigma2
        self.layerwise_prior_omega = layerwise_prior_omega
        self.depth = depth

        self.la: Laplace = None
        self._noise_var: torch.Tensor = None  # exp(log_sigma2), reused verbatim for var_aleatoric
        self.n_parameters: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int, use_cache: bool = True) -> "LaplaceMethod":
        """`use_cache` covers only the MAP backbone (the expensive, 2000-epoch
        part) via `train_homoscedastic_mlp`'s own cache — the subsequent
        `la.fit()` below is cheap (curvature only, no gradient descent) and
        is not itself cached, to avoid depending on `laplace-torch`'s
        `Laplace` object being safely serialisable (`src/methods/cache.py`).
        """
        factory = lambda: HomoscedasticMLP(
            in_dim=X.shape[1], hidden=self.hidden, dropout_p=0.0, activation=self.activation,
            fixed_sigma2=self.fixed_sigma2, depth=self.depth,
        )
        penalty_override = None
        if self.layerwise_prior_omega is not None:
            probe = HomoscedasticMLP(in_dim=X.shape[1], hidden=self.hidden, activation=self.activation, depth=self.depth)
            penalty_override = layerwise_penalty_coefficients(
                probe.mlp.named_parameters(), self.layerwise_prior_omega, X.shape[0],
            )
        cache_key_extra = dict(
            depth=self.depth, hidden=self.hidden, activation=self.activation, dropout_p=0.0, always_on=False,
            fixed_sigma2=self.fixed_sigma2, layerwise_prior_omega=self.layerwise_prior_omega,
        )
        map_model = train_homoscedastic_mlp(
            factory, X, y, seed=seed, gamma=self.gamma, epochs=self.epochs, lr=self.lr,
            batch_size=self.batch_size, penalty_coefficient_override=penalty_override,
            use_cache=use_cache, cache_key_extra=cache_key_extra,
        )
        map_model.eval()
        mean_net = map_model.mlp
        log_var = map_model.log_sigma2.clamp(*LOGVAR_CLAMP).detach()
        self._noise_var = torch.exp(log_var)
        sigma_noise = self._noise_var.sqrt()

        n = X.shape[0]
        if self.layerwise_prior_omega is not None:
            # order must match mean_net.parameters()'s own iteration order — Laplace's `prior_precision`
            # vector is positional (per parameter GROUP), not name-keyed
            precisions = layerwise_prior_precisions(mean_net.named_parameters(), self.layerwise_prior_omega)
            matched_prior_precision = torch.tensor(
                [precisions[name] for name, _ in mean_net.named_parameters()], dtype=DTYPE,
            )
        else:
            matched_prior_precision = prior_parametrisations(self.gamma, n).prior_precision  # = 1/gamma**2, same gamma as every other method
        if self.prior_precision_mode == "unregularised":
            prior_precision_init = UNREGULARISED_PRIOR_PRECISION
        else:  # "fixed" or "marglik" (marglik's value is the optimizer's starting point)
            prior_precision_init = matched_prior_precision

        self.la = Laplace(
            mean_net,
            "regression",
            subset_of_weights="all",
            hessian_structure=self.hessian_structure,
            sigma_noise=sigma_noise,
            prior_precision=prior_precision_init,
        )

        X_t = torch.as_tensor(X, dtype=DTYPE)
        y_t = torch.as_tensor(y, dtype=DTYPE).reshape(-1, 1)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=n)
        self.la.fit(loader)

        if self.prior_precision_mode == "marglik":
            self.la.optimize_prior_precision(pred_type="glm", method="marglik", n_steps=self.marglik_steps)
        # "fixed" and "unregularised": prior_precision is never optimised away from what was set above

        self.n_parameters = count_parameters(map_model)
        return self

    def predict(self, X: np.ndarray) -> Prediction:
        X_t = torch.as_tensor(X, dtype=DTYPE)
        f_mu, f_var = self.la(X_t, pred_type="glm", diagonal_output=True)

        mean = f_mu.detach().numpy().ravel()
        var_epistemic = np.clip(f_var.detach().numpy().ravel(), 0.0, None)
        var_aleatoric = np.full_like(mean, float(self._noise_var))
        return Prediction(mean=mean, var_aleatoric=var_aleatoric, var_epistemic=var_epistemic, samples=None)
