"""Regenerate the toy predictive-posterior and reliability figures.

Trains every method on the same task benchmark_toy.py uses (same generator,
configs, and seed), then produces two figures consistent with the results
table:
  1. predictive_posterior_toy.png - P(class 1) over a plane extended past the
     data, so off-manifold over/under-confidence is visible (the OOD story).
  2. calibration_curves_toy.png   - reliability diagram overlaying all methods.

Usage:
    python scripts/plot_toy_posteriors.py
"""

import argparse
import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.benchmark_toy import (  # noqa: E402
    CONFIGS, HIDDEN, generate_toy_dataset,
)
from uqbench.evaluation import calibration  # noqa: E402
from uqbench.models.method_registry import METHODS  # noqa: E402

LABELS = {
    "FNN": "FNN", "TemperatureScaledFNN": "TempScaled", "DeepEnsemble": "DeepEnsemble",
    "DropoutFNN": "Dropout (MC)", "BayesianFNN": "Bayesian (BBB)",
    "LaplaceFNN": "Laplace", "MCMCFNN": "MCMC (NUTS)",
}


def analytic_posterior(grid: np.ndarray) -> np.ndarray:
    """Bayes-optimal P(class 1) for the QDA generator (equal priors)."""
    def gauss(x, mean, var):
        d = x - mean
        return np.exp(-0.5 * np.sum(d * d, axis=1) / var) / (2 * np.pi * var)
    p0 = gauss(grid, np.array([0.0, 0.0]), 3.0)   # diffuse sea
    p1 = gauss(grid, np.array([1.0, 1.0]), 0.35)  # tight island
    return p1 / (p0 + p1 + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()

    X, y, y_oh = generate_toy_dataset(seed=args.seed)
    X_tr, X_te, _, _, y_tr_oh, y_te_oh = train_test_split(
        X, y, y_oh, test_size=0.3, random_state=args.seed, stratify=y
    )
    data = {"X_train": X_tr, "y_train_onehot": y_tr_oh}

    # Grid extended well past the training support (data lives within ~r=4).
    lo, hi, n = -7.0, 7.0, 200
    gx, gy = np.meshgrid(np.linspace(lo, hi, n), np.linspace(lo, hi, n))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    # Train each method once and cache grid + test predictions.
    posteriors, curves = {}, {}
    for name, spec in METHODS.items():
        cfg = {"hidden_dims": HIDDEN, "num_classes": 2, "seed": args.seed}
        cfg.update(CONFIGS[name])
        rng = jax.random.PRNGKey(args.seed)
        rng, tr, pg, pt = jax.random.split(rng, 4)
        art = spec["train"](cfg, data, tr)
        posteriors[name] = np.asarray(spec["predict_proba"](art, grid, cfg, pg))[:, 1].reshape(n, n)
        probs_te = spec["predict_proba"](art, X_te, cfg, pt)
        curves[name] = calibration.calibration_curve(probs_te, np.asarray(y_te_oh), num_bins=10)
        print(f"  trained {name}", flush=True)

    # --- Figure 1: predictive posteriors (8 panels: 7 methods + ground truth) ---
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    panels = list(METHODS) + ["__truth__"]
    for ax, name in zip(axes.ravel(), panels):
        field = analytic_posterior(grid).reshape(n, n) if name == "__truth__" else posteriors[name]
        cf = ax.contourf(gx, gy, field, levels=np.linspace(0, 1, 21), cmap="RdBu_r", vmin=0, vmax=1)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=3, c="#08306b", alpha=0.35)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=3, c="#7f2704", alpha=0.35)
        ax.set_title("Ground truth (Bayes-optimal)" if name == "__truth__" else LABELS[name],
                     fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Predictive posterior P(class 1), extended past the data — QDA toy, seed 0", fontsize=13)
    fig.colorbar(cf, ax=axes, fraction=0.025, pad=0.02, label="P(class 1)")
    out1 = args.outdir / "predictive_posterior_toy.png"
    fig.savefig(out1, dpi=120, bbox_inches="tight")

    # --- Figure 2: reliability diagram ---
    fig2, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", alpha=0.6, label="Perfect calibration")
    for name, spec in METHODS.items():
        frac, conf = curves[name]
        ax.plot(conf, frac, marker=spec["marker"], color=spec["color"], label=LABELS[name], alpha=0.85)
    ax.set_xlabel("Mean predicted confidence (top label)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram (toy, seed 0)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = args.outdir / "calibration_curves_toy.png"
    fig2.savefig(out2, dpi=120)

    print(f"Saved -> {out1}\nSaved -> {out2}")


if __name__ == "__main__":
    main()
