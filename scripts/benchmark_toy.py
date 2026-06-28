"""Run the full method matrix on the 2D QDA toy dataset across seeds.

Trains every method in the registry on a controlled 2D QDA classification
task (diffuse class-0 Gaussian + tight class-1 Gaussian), repeated over
several seeds, and reports mean +/- std for accuracy, proper scoring rules
(NLL, Brier), calibration (ECE, ACE), selective prediction (AURC), and OOD
detection (AUROC against a far-field OOD set). Results are written to JSON
and printed as a Markdown table.

Usage:
    python scripts/benchmark_toy.py --seeds 0,1,2,3,4
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from uqbench.evaluation import calibration
from uqbench.models.method_registry import METHODS


def generate_toy_dataset(n_samples: int = 2000, seed: int = 42):
    """QDA toy: diffuse class-0 Gaussian + tight class-1 Gaussian (equal priors).

    Unequal covariances produce a curved, closed Bayes-optimal boundary —
    class 1 occupies a compact island inside a diffuse class-0 sea.
    """
    rng = np.random.default_rng(seed)
    n = n_samples // 2
    X0 = rng.multivariate_normal([0.0, 0.0], 3.0 * np.eye(2), n)   # diffuse sea
    X1 = rng.multivariate_normal([1.0, 1.0], 0.35 * np.eye(2), n)  # tight island
    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.hstack([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    y_onehot = np.zeros((len(y), 2), dtype=np.float32)
    y_onehot[np.arange(len(y)), y] = 1.0
    return X, y, y_onehot


def far_field_ood(n: int, seed: int = 7) -> np.ndarray:
    """Points on an annulus far from the training support (radius 5-7)."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(5.0, 7.0, n)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1).astype(np.float32)


def nll(probs: jnp.ndarray, labels: jnp.ndarray) -> float:
    p = jnp.clip(probs, 1e-12, 1.0)
    return float(-jnp.mean(jnp.sum(labels * jnp.log(p), axis=1)))


def aurc(probs: jnp.ndarray, labels: jnp.ndarray) -> float:
    """Area under the risk-coverage curve (lower = better selective prediction)."""
    conf = np.asarray(jnp.max(probs, axis=1))
    correct = np.asarray(jnp.argmax(probs, axis=1) == jnp.argmax(labels, axis=1))
    order = np.argsort(-conf)
    errors = (~correct[order]).astype(np.float64)
    cum_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(np.mean(cum_risk))


def ood_auroc(probs_id: jnp.ndarray, probs_ood: jnp.ndarray) -> float:
    """AUROC separating ID vs OOD by predictive entropy (higher = better)."""
    def entropy(p):
        p = np.clip(np.asarray(p), 1e-12, 1.0)
        return -np.sum(p * np.log(p), axis=1)

    scores = np.concatenate([entropy(probs_id), entropy(probs_ood)])
    labels = np.concatenate([np.zeros(len(probs_id)), np.ones(len(probs_ood))])
    return float(roc_auc_score(labels, scores))


# Per-method configs. BayesianFNN uses the tuned values from experiments/hyperopt.
HIDDEN = (64, 32, 32, 32, 32)
CONFIGS = {
    "FNN": {"epochs": 200, "lr": 0.01},
    "DropoutFNN": {"epochs": 200, "lr": 0.01, "dropout_rate": 0.2, "n_samples": 100},
    "BayesianFNN": {
        "epochs": 400, "lr": 0.000667, "beta": 0.000227,
        "posterior_std_init": 0.0115, "warm_up_epochs": 50, "n_samples": 100,
    },
    "LaplaceFNN": {"epochs": 200, "lr": 0.01, "prior_precision": 10.0,
                   "subset_size": 1000, "n_samples": 100},
    "MCMCFNN": {"prior_std": 0.1, "temperature": 0.05, "num_warmup": 150,
                "num_samples": 150, "sampler": "nuts", "n_samples": 150},
    "TemperatureScaledFNN": {"epochs": 200, "lr": 0.01, "max_iter": 1000},
    "DeepEnsemble": {"epochs": 200, "lr": 0.01, "n_members": 5},
}

METRICS = ["accuracy", "nll", "brier", "ece", "ace", "aurc", "ood_auroc"]


def run_seed(seed: int) -> dict[str, dict[str, float]]:
    """Train and evaluate every method for one seed."""
    X, y, y_onehot = generate_toy_dataset(seed=seed)
    X_train, X_test, _, _, y_train_oh, y_test_oh = train_test_split(
        X, y, y_onehot, test_size=0.3, random_state=seed, stratify=y
    )
    data = {"X_train": X_train, "y_train_onehot": y_train_oh}
    X_ood = far_field_ood(len(X_test), seed=seed + 1)
    y_test = jnp.array(y_test_oh)

    out: dict[str, dict[str, float]] = {}
    for name, spec in METHODS.items():
        cfg = {"hidden_dims": HIDDEN, "num_classes": 2, "seed": seed}
        cfg.update(CONFIGS[name])
        rng = jax.random.PRNGKey(seed)
        rng, train_rng, pred_rng, ood_rng = jax.random.split(rng, 4)
        artifact = spec["train"](cfg, data, train_rng)
        probs = spec["predict_proba"](artifact, X_test, cfg, pred_rng)
        probs_ood = spec["predict_proba"](artifact, X_ood, cfg, ood_rng)
        out[name] = {
            "accuracy": float((jnp.argmax(probs, axis=1) == jnp.argmax(y_test, axis=1)).mean()),
            "nll": nll(probs, y_test),
            "brier": calibration.brier_score(probs, y_test),
            "ece": calibration.expected_calibration_error(probs, y_test),
            "ace": calibration.adaptive_calibration_error(probs, y_test),
            "aurc": aurc(probs, y_test),
            "ood_auroc": ood_auroc(probs, probs_ood),
            "fwd_passes": float(spec["inference_cost"](cfg)["forward_passes_per_example"]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/results/toy_benchmark.json"))
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    raw: dict[str, dict[str, list[float]]] = {
        name: {m: [] for m in METRICS + ["fwd_passes"]} for name in METHODS
    }
    for seed in seeds:
        print(f"=== seed {seed} ===", flush=True)
        for name, metrics in run_seed(seed).items():
            for m, v in metrics.items():
                raw[name][m].append(v)
            print(f"  {name}: " + ", ".join(f"{m}={metrics[m]:.4f}" for m in METRICS), flush=True)

    summary = {
        name: {m: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
               for m, vals in mdict.items()}
        for name, mdict in raw.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"dataset": "toy_2gaussian", "seeds": seeds,
                   "summary": summary, "raw": raw}, f, indent=2)

    # Markdown table: mean +/- std
    header = "| Method | " + " | ".join(m.upper() for m in METRICS) + " | Fwd |"
    sep = "|" + "---|" * (len(METRICS) + 2)
    print("\n" + header + "\n" + sep)
    for name in METHODS:
        cells = [f"{summary[name][m]['mean']:.3f} ± {summary[name][m]['std']:.3f}" for m in METRICS]
        cells.append(str(int(summary[name]["fwd_passes"]["mean"])))
        print(f"| {name} | " + " | ".join(cells) + " |")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
