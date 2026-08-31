"""figures/rodzial3_rys/img3_1.png — GP PRIOR samples at three length-scales.

Chapter 3.1's illustration of what the length-scale is: draws from
`GP(0, k)` before any data, for `l` an order of magnitude either side of 1,
with the amplitude and the random draw held fixed so the only thing that
changes between panels is the length-scale.

Deliberately the PRIOR, not the posterior: `img2_2.png` already shows
posterior draws on `sin_homo`, and there the length-scale's effect is
confounded with the data pinning the functions down. The kernel is the same
`ConstantKernel * RBF` the project's `GPMethod` fits, so the panel with
`l = 1.0` is the prior that E1/E2's GP starts from.

No experiment behind this: it is a picture of the kernel, computed from the
kernel, and reproducible from `SEED` alone.

Usage:
  python scripts/fig_gp_prior_samples.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from src.style import METHOD_COLORS, SEED

LENGTH_SCALES = (0.3, 1.0, 3.0)
N_SAMPLES = 5
X_RANGE = (-3.0, 3.0)
N_GRID = 400
OUT_PATH = Path(__file__).resolve().parent.parent / "figures" / "rodzial3_rys" / "img3_1.png"


def main():
    x = np.linspace(*X_RANGE, N_GRID).reshape(-1, 1)
    fig, axes = plt.subplots(1, len(LENGTH_SCALES), figsize=(11.0, 3.4), sharey=True)

    for ax, length_scale in zip(axes, LENGTH_SCALES):
        kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale)
        # `optimizer=None` and no `fit`: sampling from an unfitted regressor
        # draws from the prior, which is the whole point of the figure.
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)
        samples = gp.sample_y(x, n_samples=N_SAMPLES, random_state=SEED)
        for i in range(N_SAMPLES):
            ax.plot(x.ravel(), samples[:, i], color=METHOD_COLORS["gp"],
                    alpha=0.75, linewidth=1.1)
        # +/-2 prior standard deviations: constant at 2, since the amplitude is
        # fixed at 1 and the prior is stationary. Drawn to make the point that
        # the length-scale changes the WIGGLINESS, not the marginal spread.
        ax.axhspan(-2.0, 2.0, color=METHOD_COLORS["gp"], alpha=0.06)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4, linestyle="--")
        ax.set_title(f"$\\ell$ = {length_scale}")
        ax.set_xlabel("x")
        ax.set_xlim(*X_RANGE)

    axes[0].set_ylabel("f(x)")
    axes[0].set_ylim(-3.0, 3.0)
    fig.suptitle(
        "Samples from the GP prior, $k(x, x') = \\sigma^2 \\exp(-\\|x - x'\\|^2 / 2\\ell^2)$, "
        f"$\\sigma^2$ = 1, seed = {SEED}", fontsize=10,
    )
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
