"""Is `LOGVAR_CLAMP`'s lower bound binding? (raised by the author at O4 step 2, point 3d)

`experiments/uci_epochs_sweep.py` found `mean_var_aleatoric` sitting at
0.0025 = `exp(-6)` — exactly `LOGVAR_CLAMP[0]` — on `yacht` and `energy`
from 500 epochs on, and on `concrete`/`wine_quality_red` found validation
NLL rising while validation RMSE stood still, i.e. the mean was fine and
`sigma^2` alone was collapsing. Both observations point at the same
question: on the flat tail of those loss curves, is the model converging,
or is `sigma^2` resting on the clamp floor? A binding clamp is a
configuration error, not a property of the model, and it would enter E2
whatever epoch count is chosen — so it has to be settled before the epoch
count is.

Three things measured, `map`, split 0, seeds 0-2, all six datasets:
  1. the epoch at which the raw `log_sigma2` parameter first reaches the
     lower bound (the parameter keeps drifting below it; the FORWARD pass
     is what gets clamped, so both are recorded)
  2. the residual variance `mean((y_fit - mu)^2)` at that epoch — what
     `sigma^2` "should" be, since for a homoscedastic Gaussian NLL with a
     fixed mean that residual variance IS the maximiser
  3. the same run with the bound widened to `[-12, 6]`, to see whether the
     validation NLL curve changes shape

**One instrumented run per (dataset, seed, clamp) gives the whole curve.**
The training trajectory is deterministic given the seed, so the model seen
after epoch `e` of a 2000-epoch run is the same model a separate
`epochs=e` run would produce — the same property that made the grid in
`uci_epochs_sweep.py` valid. This measures every epoch instead of six, and
costs one run instead of six.

Uses the same inner 80/20 validation split as `uci_epochs_sweep.py`, for
the same reason (see `_inner_split` there) — imported, not re-derived, so
the two measurements cannot drift apart.

`LOGVAR_CLAMP` itself is NOT changed here: it is shared by every method, so
changing it is the author's decision, and this script only measures what
changing it would do. The widened bound is passed per-model through
`HomoscedasticMLP(logvar_clamp=...)`.

Writes results/logvar_clamp_diagnostic.csv (one row per epoch),
results/logvar_clamp_diagnostic_summary.csv, and
figures/logvar_clamp_diagnostic.png.

Usage:
  python experiments/logvar_clamp_diagnostic.py
  python experiments/logvar_clamp_diagnostic.py --datasets yacht,energy --seeds 0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src import metrics
from src.data import UCI_SPEC, load_uci
from src.methods.backbone import (
    DEFAULT_ACTIVATION, DEFAULT_BATCH_SIZE, DEFAULT_GAMMA, DTYPE, LOGVAR_CLAMP, HomoscedasticMLP,
    train_homoscedastic_mlp,
)
from src.plotting import FIGURES_DIR
from src.results import RESULTS_DIR, git_commit_short, now_iso, upsert_csv
from src.seeding import set_seed

from uci_epochs_sweep import DEFAULT_DATASETS, DEFAULT_SEEDS, _inner_split  # noqa: E402

from src.methods import METHODS  # noqa: E402

EPOCHS = 2000  # the longest value on uci_epochs_sweep.py's grid; shorter runs are prefixes of it
WIDE_CLAMP = (-12.0, 6.0)
CLAMPS = {"default": LOGVAR_CLAMP, "wide": WIDE_CLAMP}

# Grid points reported in the comparison table — the same ones
# uci_epochs_sweep.py used, so the two are directly comparable.
REPORT_EPOCHS = (50, 100, 200, 500, 1000, 2000)

# `log_sigma2` never lands exactly on the bound in floating point once the
# raw parameter has drifted past it, so "touching" is tested with a small
# tolerance rather than equality.
CLAMP_TOL = 1e-9

# The bound as it stood before 2026-08-27, kept so `contact_check` can ask
# "would the old bound have been binding here?" while training under the
# current, widened one.
OLD_CLAMP_LO = -6.0

# BBB's ELBO-sample count for the contact check — 8, matching what
# uci_epochs_sweep.py runs, not the class default of 1.
CONTACT_BBB_ELBO_SAMPLES = 8


def contact_check(datasets, method_names, seed, epochs):
    """Does `log_sigma2` reach the OLD lower bound (-6) for methods other
    than `map`? (author's follow-up, 2026-08-27.)

    `map` was the only method the trajectory measurement above covered, but
    every NN method shares this backbone and this clamp, so "the clamp was
    binding" needs checking on the regularised methods too. Run under the
    CURRENT (widened) clamp and read where the raw parameter ends up: below
    -6 means the old bound would have been binding for that method.

    One split, one seed, full E2 training fold — a contact test, not a
    convergence measurement, so it does not need the seed replication the
    epoch sweep has.
    """
    rows = []
    for name in datasets:
        ds = load_uci(name, split=0)
        for method_name in method_names:
            kwargs = dict(epochs=epochs)
            if method_name == "bbb":
                kwargs["elbo_samples"] = CONTACT_BBB_ELBO_SAMPLES
            set_seed(seed)
            method = METHODS[method_name](**kwargs)
            method.fit(ds.X_train, ds.y_train, seed=seed, use_cache=False)
            raw = float(method.model.log_sigma2.detach())
            rows.append(dict(
                dataset=name, method=method_name, seed=seed, epochs=epochs,
                log_sigma2_raw=raw,
                var_aleatoric=float(np.exp(min(raw, LOGVAR_CLAMP[1]))),
                would_hit_old_bound=bool(raw <= OLD_CLAMP_LO + CLAMP_TOL),
                timestamp=now_iso(), git_commit=git_commit_short(),
            ))
            print(f"  {name:8s} {method_name:4s} log_sigma2={raw:8.4f}  "
                  f"would have hit -6: {rows[-1]['would_hit_old_bound']}")
    return pd.DataFrame(rows)


def _run_one(name, seed, clamp_name, clamp, X_fit, y_fit, X_val, y_val):
    """One instrumented `map` training run; one row per epoch.

    Duplicates `MAPMethod`'s three-line predict rather than calling it: the
    method builds its own model internally and has no `logvar_clamp`
    argument, and adding one would put a knob on every method's constructor
    for the sake of a diagnostic.
    """
    X_fit_t = torch.as_tensor(X_fit, dtype=DTYPE)
    y_fit_t = torch.as_tensor(y_fit, dtype=DTYPE).reshape(-1, 1)
    X_val_t = torch.as_tensor(X_val, dtype=DTYPE)

    records = []

    def callback(epoch, model):
        model.eval()
        with torch.no_grad():
            mu_fit, log_var = model(X_fit_t)
            mu_val, _ = model(X_val_t)
        model.train()
        std = float(torch.exp(0.5 * log_var))
        mean_val = mu_val.numpy().ravel()
        records.append(dict(
            epoch=epoch,
            log_sigma2_raw=float(model.log_sigma2.detach()),
            log_sigma2_effective=float(log_var),
            var_aleatoric=float(torch.exp(log_var)),
            residual_var_fit=float(torch.mean((y_fit_t - mu_fit) ** 2)),
            val_nll=metrics.nll(y_val, mean_val, np.full_like(mean_val, std)),
            val_rmse=metrics.rmse(y_val, mean_val),
        ))

    set_seed(seed)
    factory = lambda: HomoscedasticMLP(
        in_dim=X_fit.shape[1], hidden=50, dropout_p=0.0,
        activation=DEFAULT_ACTIVATION, logvar_clamp=clamp,
    )
    train_homoscedastic_mlp(
        factory, X_fit, y_fit, seed=seed, gamma=DEFAULT_GAMMA, epochs=EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE, use_cache=False, epoch_callback=callback,
    )

    df = pd.DataFrame(records)
    df.insert(0, "dataset", name)
    df.insert(1, "seed", seed)
    df.insert(2, "clamp", clamp_name)
    df.insert(3, "clamp_lo", clamp[0])
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, clamp): when the bound is first reached, what the
    residual variance says `sigma^2` should have been at that moment, and
    where things end up at 2000 epochs. Averaged over seeds; `n_seeds_hit`
    says how many seeds reached the bound at all."""
    out = []
    for (name, clamp_name), sub in df.groupby(["dataset", "clamp"], sort=False):
        clamp_lo = float(sub["clamp_lo"].iloc[0])
        per_seed = []
        for seed, s in sub.groupby("seed"):
            hit = s[s["log_sigma2_raw"] <= clamp_lo + CLAMP_TOL]
            if hit.empty:
                per_seed.append(dict(seed=seed, hit_epoch=np.nan, residual_var_at_hit=np.nan,
                                     var_alea_at_hit=np.nan))
            else:
                first = hit.iloc[0]
                per_seed.append(dict(seed=seed, hit_epoch=first["epoch"],
                                     residual_var_at_hit=first["residual_var_fit"],
                                     var_alea_at_hit=first["var_aleatoric"]))
        per_seed = pd.DataFrame(per_seed)
        final = sub[sub["epoch"] == EPOCHS]
        out.append(dict(
            dataset=name, clamp=clamp_name, clamp_lo=clamp_lo,
            n_seeds_hit=int(per_seed["hit_epoch"].notna().sum()),
            n_seeds=int(per_seed.shape[0]),
            mean_hit_epoch=float(per_seed["hit_epoch"].mean()),
            mean_residual_var_at_hit=float(per_seed["residual_var_at_hit"].mean()),
            mean_var_alea_at_hit=float(per_seed["var_alea_at_hit"].mean()),
            final_log_sigma2_raw=float(final["log_sigma2_raw"].mean()),
            final_var_aleatoric=float(final["var_aleatoric"].mean()),
            final_residual_var_fit=float(final["residual_var_fit"].mean()),
            final_val_nll=float(final["val_nll"].mean()),
            final_val_rmse=float(final["val_rmse"].mean()),
        ))
    return pd.DataFrame(out)


def make_figure(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = list(dict.fromkeys(df["dataset"]))
    fig, axes = plt.subplots(2, len(datasets), figsize=(3.6 * len(datasets), 6.8), squeeze=False)
    styles = {"default": ("#000000", "-"), "wide": ("#D55E00", "--")}

    for col, name in enumerate(datasets):
        sub_ds = df[df.dataset == name]
        ax_sigma, ax_nll = axes[0][col], axes[1][col]
        for clamp_name, (colour, ls) in styles.items():
            sub = sub_ds[sub_ds.clamp == clamp_name]
            if sub.empty:
                continue
            curve = sub.groupby("epoch")[["log_sigma2_raw", "residual_var_fit", "val_nll"]].mean()
            ax_sigma.plot(curve.index, curve["log_sigma2_raw"], color=colour, ls=ls,
                          label=f"log sigma^2 ({clamp_name})")
            ax_sigma.plot(curve.index, np.log(curve["residual_var_fit"]), color=colour, ls=":",
                          alpha=0.7, label=f"log residual var ({clamp_name})")
            ax_nll.plot(curve.index, curve["val_nll"], color=colour, ls=ls, label=clamp_name)
        for bound, colour in ((LOGVAR_CLAMP[0], "#000000"), (WIDE_CLAMP[0], "#D55E00")):
            ax_sigma.axhline(bound, color=colour, lw=0.8, alpha=0.5)
        ax_sigma.set_title(f"{name} (n_train={UCI_SPEC[name]['n_train']})")
        ax_sigma.set_xscale("log")
        ax_sigma.set_ylabel("log sigma^2 (raw parameter)")
        ax_nll.set_xscale("log")
        ax_nll.set_xlabel("epochs")
        ax_nll.set_ylabel("validation NLL [nats]")
        for ax in (ax_sigma, ax_nll):
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6)

    fig.suptitle("Is LOGVAR_CLAMP's lower bound binding? (map, split 0, seeds 0-2; "
                 "horizontal lines = the two bounds)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--clamps", type=str, default=",".join(CLAMPS))
    parser.add_argument("--contact-check", type=str, default="",
                        help="instead of the trajectory sweep, run the cross-method contact test "
                             "on the given comma-separated methods, e.g. 'map,mcd,bbb'")
    parser.add_argument("--contact-epochs", type=int, default=2000)
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    seeds = [int(v) for v in args.seeds.split(",")]
    clamp_names = args.clamps.split(",")

    if args.contact_check:
        df = contact_check(datasets, args.contact_check.split(","), seeds[0], args.contact_epochs)
        out = RESULTS_DIR / "logvar_clamp_contact_check.csv"
        # Overwrite by design: `--contact-check` is a single-shot probe whose
        # frame IS the whole file, and its row key is not stable enough to
        # upsert on. Nothing else writes here.
        df.to_csv(out, index=False)
        print(f"\nwrote {out}")
        return

    frames = []
    for name in datasets:
        X_fit, y_fit, X_val, y_val = _inner_split(name)
        for clamp_name in clamp_names:
            for seed in seeds:
                frame = _run_one(name, seed, clamp_name, CLAMPS[clamp_name],
                                 X_fit, y_fit, X_val, y_val)
                frames.append(frame)
                last = frame.iloc[-1]
                hit = frame[frame["log_sigma2_raw"] <= CLAMPS[clamp_name][0] + CLAMP_TOL]
                hit_epoch = int(hit.iloc[0]["epoch"]) if not hit.empty else None
                print(f"  {name:17s} clamp={clamp_name:7s} seed={seed}  "
                      f"hit_epoch={str(hit_epoch):>5s}  "
                      f"final log_sigma2={last['log_sigma2_raw']:8.3f} "
                      f"var_alea={last['var_aleatoric']:.5f} "
                      f"resid_var={last['residual_var_fit']:.5f} "
                      f"val_nll={last['val_nll']:.4f}")

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = now_iso()
    df["git_commit"] = git_commit_short()
    out = RESULTS_DIR / "logvar_clamp_diagnostic.csv"
    upsert_csv(out, df, ["dataset", "method", "seed", "epochs"])
    print(f"\nwrote {out}")

    summary = summarise(df)
    upsert_csv(RESULTS_DIR / "logvar_clamp_diagnostic_summary.csv", summary,
               ["dataset", "method", "seed"])
    make_figure(df, FIGURES_DIR / "logvar_clamp_diagnostic.png")

    print("\nclamp contact and what sigma^2 should have been (mean over seeds):")
    print(f"  {'dataset':17s} {'clamp':7s} {'hit':>6s} {'seeds':>6s} {'var_alea@hit':>12s} "
          f"{'resid_var@hit':>13s} {'final var_alea':>14s} {'final resid_var':>15s}")
    for _, r in summary.iterrows():
        hit = "-" if np.isnan(r["mean_hit_epoch"]) else f"{r['mean_hit_epoch']:.0f}"
        print(f"  {r['dataset']:17s} {r['clamp']:7s} {hit:>6s} "
              f"{r['n_seeds_hit']:d}/{r['n_seeds']:d}".ljust(2) +
              f"   {r['mean_var_alea_at_hit']:12.6f} {r['mean_residual_var_at_hit']:13.6f} "
              f"{r['final_var_aleatoric']:14.6f} {r['final_residual_var_fit']:15.6f}")

    print("\nvalidation NLL by epoch, default vs wide clamp (mean over seeds):")
    for name in datasets:
        sub = df[df.dataset == name]
        print(f"\n{name}")
        print(f"  {'epochs':>6s} {'default':>9s} {'wide':>9s} {'delta':>8s}")
        for e in REPORT_EPOCHS:
            row = sub[sub.epoch == e].groupby("clamp")["val_nll"].mean()
            if len(row) < 2:
                continue
            print(f"  {e:6d} {row['default']:9.4f} {row['wide']:9.4f} "
                  f"{row['wide'] - row['default']:8.4f}")


if __name__ == "__main__":
    main()
