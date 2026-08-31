"""E1 figures (brief section 10). Reads results/predictions_1d/ (written by
e1_synthetic.py) — no retraining except img2_2 (GP function samples, see
src/plotting.py). Run e1_synthetic.py first.

Directory name "rodzial2_rys"/"rodzial3_rys" (not "rozdzial...") matches an
existing typo in the .tex sources — preserved deliberately, not a mistake
here; flagged to the thesis author separately.

Filenames in rodzial3_rys are `img3_{method}.png` (this session's decision),
not the brief's original numbered `img3_2/5/6/8.png` — flagged: if the .tex
sources reference the numbered names, they will need updating to match, the
same way the directory typo was flagged rather than silently fixed.

Usage:
  python experiments/e1_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plotting import (
    FIGURES_DIR, make_epistemic_profile_figure, make_grid_figure, make_img2_2_function_samples,
    make_img2_3_nested_bands, make_single_method_figure, make_single_method_figure_epistemic,
)
from src.style import METHOD_ORDER

# brief section 10: identical axes and scale across these — enforced by
# src/plotting.py's shared _draw_posterior, not by this list's order.
THESIS_POSTERIOR_METHODS = ["gp", "bbb", "mcd", "laplace", "ensemble"]

# Both geometries get a full set of panels (this session's decision). `sin_homo`
# is one data island, so it probes EXTRAPOLATION on either side; `sin_gap` is two
# islands with a hole, so it probes IN-BETWEEN uncertainty — and the two separate
# the methods very differently. On sin_gap, mcd's epistemic std is LOWER inside
# the gap than in the data (0.065 vs 0.127 at seed=0), the opposite sign, while
# gp and laplace bulge as they should. That is the figure that argues the
# chapter's point without a table; sin_homo's version only argues it quantitatively.
# Which pair goes into chapter 3 is the author's call, so both are generated and
# neither is overwritten.
#
# Naming: `sin_homo` keeps the unsuffixed `img3_{method}.png` it has always had
# (the .tex sources may already reference those names — see the module docstring's
# note on the numbered-vs-named filenames still being an open question with the
# author), and `sin_gap` gets an explicit `img3_{method}_sin_gap.png`. The
# asymmetry is deliberate: renaming the existing files to match would be a silent
# change to whatever already references them.
POSTERIOR_DATASETS = {
    "sin_homo": "img3_{method}.png",
    "sin_gap": "img3_{method}_sin_gap.png",
}


def main():
    for dataset, pattern in POSTERIOR_DATASETS.items():
        for method in THESIS_POSTERIOR_METHODS:
            out_path = FIGURES_DIR / "rodzial3_rys" / pattern.format(method=method)
            make_single_method_figure(dataset, method, out_path)
            print(f"wrote {out_path}")

            # D14f (docs/chapter4_notes.md): epistemic-only counterpart. Same
            # X_RANGE and the same y scale across all five methods, but that y
            # scale is now Y_RANGE_EPISTEMIC, not Y_RANGE — see src/style.py.
            epi_pattern = pattern.replace(".png", "_epistemic.png")
            out_path_epi = FIGURES_DIR / "rodzial3_rys" / epi_pattern.format(method=method)
            make_single_method_figure_epistemic(dataset, method, out_path_epi)
            print(f"wrote {out_path_epi}")

        # The figure the panels above cannot be: two orders of magnitude of
        # epistemic std on one log axis (see make_epistemic_profile_figure).
        profile_path = FIGURES_DIR / "rodzial3_rys" / f"img3_epistemic_profile_{dataset}.png"
        make_epistemic_profile_figure(dataset, profile_path)
        print(f"wrote {profile_path}")

    # MAP: no rozdzial 3 section for the deterministic baseline (brief section 10) — outside figures/rodzial*_rys.
    map_out = FIGURES_DIR / "e1_map.png"
    make_single_method_figure("sin_homo", "map", map_out)
    print(f"wrote {map_out}")

    make_img2_2_function_samples(FIGURES_DIR / "rodzial2_rys" / "img2_2.png")
    print(f"wrote {FIGURES_DIR / 'rodzial2_rys' / 'img2_2.png'}")

    make_img2_3_nested_bands(FIGURES_DIR / "rodzial2_rys" / "img2_3.png")
    print(f"wrote {FIGURES_DIR / 'rodzial2_rys' / 'img2_3.png'}")

    # outside the thesis, visual QA only — one grid per synthetic variant, all six methods
    for dataset in ["sin_homo", "sin_hetero", "sin_gap"]:
        out_path = FIGURES_DIR / f"e1_comparison_grid_{dataset}.png"
        make_grid_figure(dataset, METHOD_ORDER, out_path,
                          title=f"E1 comparison grid — {dataset}, seed=0 (visual QA only, not a thesis figure)")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
