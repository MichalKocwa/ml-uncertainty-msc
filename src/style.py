"""Shared plotting configuration (brief section 10).

One palette, one colour per method, everywhere. Axis limits are forced from
here into a single drawing function (`src/plotting.py`) — never set per
figure — so that "identical axes and identical scale" (brief section 10's
"wymóg krytyczny" for the GP/BBB/MCD/Laplace posterior figures) is a
structural property of the code, not a convention someone has to remember
to follow figure by figure.
"""

# Okabe-Ito colour-blind-safe qualitative palette. `map` gets black: it is
# the deterministic reference point (zero epistemic term by construction),
# not a Bayesian approximation alongside the other five, and black reads as
# "baseline" rather than competing with the five colours for attention.
METHOD_COLORS = {
    "map": "#000000",
    "gp": "#0072B2",
    "bbb": "#E69F00",
    "mcd": "#009E73",
    "laplace": "#D55E00",
    "ensemble": "#CC79A7",
}

METHOD_LABELS = {
    "map": "MAP",
    "gp": "Gaussian process",
    "bbb": "Bayes by Backprop",
    "mcd": "MC dropout",
    "laplace": "Laplace",
    "ensemble": "Deep ensemble",
}

# Display order for the 2x3 comparison grid (brief section 10, "poza pracą").
METHOD_ORDER = ["map", "mcd", "ensemble", "gp", "laplace", "bbb"]

# E1's synthetic datasets evaluate on [-2, 8] — symmetric 2 units either
# side of the [0,6] training range (this session's decision, correcting
# brief section 5.1's [-2, 10]: that was 2 units left but 4 right of
# training, an asymmetry with no stated reason and real consequences —
# see docs/chapter4_notes.md).
X_RANGE = (-2.0, 8.0)

# DECIDED 2026-08-25 (this session, after the N=250/float64 fix — see
# docs/chapter4_notes.md and src/methods/laplace.py): [-5, 5] comfortably
# contains every method's +/-2 sigma band on sin_homo/sin_hetero/sin_gap at
# seed=0, N=250, float64, hessian_structure="full" — computed range was
# approximately [-4.3, 3.3] with plenty of margin. This is tuned to the
# *current* results, not a law of nature: if a future rerun (different
# seed, different hyperparameters, a method whose bands widen) produces a
# +/-2 sigma band that touches this range, the figures will silently clip
# it — check against the saved results/predictions_1d/*.csv before trusting
# a figure, and revise this constant if any method's band has grown.
Y_RANGE = (-5.0, 5.0)

# E1 figures use only this seed (this session's decision) — chosen because
# it showed no residual numerical artifacts in the N=250/float64 Laplace
# validation sweep (see chat report; seed=5 had a small one, seed=0 did not).
SEED = 0

# Brief section 10's ">=1000 points" evaluation-grid requirement is met at
# the source: `src.data.EVAL_GRID_N` (1000) sets the resolution of every
# synthetic dataset's `X_eval`, which is what `experiments/e1_synthetic.py`
# both evaluates metrics on and saves to `results/predictions_1d/` — not
# duplicated here to avoid the two drifting apart.


# Y-axis for the epistemic-only panels (`img3_*_epistemic.png`), separate
# from `Y_RANGE` above. Corrects an earlier instruction that these figures
# must share `Y_RANGE` with their `+/-2*sigma_total` counterparts: brief
# section 10's "identical axes and scale" requirement is about
# comparability BETWEEN METHODS, so that one method's band cannot be made
# to look wider than another's by drawing it on a different scale. It says
# nothing about comparability between FIGURE TYPES — and the two types
# plot different quantities, so a shared scale buys nothing there while
# costing the epistemic panels most of their resolution. This constant is
# still forced identically across all five methods by
# `src/plotting.py::_draw_posterior_epistemic`, so the requirement that
# actually matters is untouched.
#
# Value: the tightest round range containing every method's
# `mean +/- 2*std_epi` on both datasets that get epistemic panels, plus
# their training scatter — computed from results/predictions_1d/, seed=0:
# sin_homo [-3.073, 2.787], sin_gap [-2.791, 3.154], train y within
# [-1.198, 1.187]. (-3.5, 3.5) clears the widest of these with margin.
#
# The gain is real but modest — 10 units of axis height down to 7, ~1.4x —
# and the binding constraint is laplace, whose epistemic band alone reaches
# +/-1.8 at the extrapolation edges while bbb/mcd/ensemble stay under
# +/-0.31. No shared LINEAR scale can make a 0.009-to-0.13 range and a
# 0.017-to-0.89 range legible at once; that is what
# `make_epistemic_profile_figure` (log scale, all methods on one axis) is
# for. This constant fixes the wasted-whitespace half of the problem, not
# the dynamic-range half.
#
# Same caveat as Y_RANGE: tuned to the CURRENT saved results. A rerun whose
# epistemic band grows past this will be silently clipped — check
# results/predictions_1d/*.csv before trusting a figure.
Y_RANGE_EPISTEMIC = (-3.5, 3.5)
