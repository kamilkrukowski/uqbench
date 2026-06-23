"""Visualize the toy task exactly as benchmark_toy.py generates it.

Plots the in-distribution two-Gaussian training data (colored by class)
together with the far-field annulus used as the OOD set for OOD-AUROC, so
the figure grounds both the classification task and the OOD evaluation.

Usage:
    python scripts/plot_toy_dataset.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.benchmark_toy import far_field_ood, generate_toy_dataset  # noqa: E402

C0, C1 = "#2E86AB", "#F18F01"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/results/toy_dataset.png"))
    args = parser.parse_args()

    X, y, _ = generate_toy_dataset(seed=args.seed)
    X_ood = far_field_ood(600, seed=args.seed + 1)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(X_ood[:, 0], X_ood[:, 1], s=12, c="0.6", marker="x",
               alpha=0.6, zorder=1)
    # Single scatter in the generator's already-shuffled order so neither class
    # systematically occludes the other in the overlap region.
    colors = np.where(y == 0, C0, C1)
    ax.scatter(X[:, 0], X[:, 1], s=14, c=colors, alpha=0.7, zorder=2)

    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Toy task: two overlapping Gaussians (ID) + far-field OOD set")
    handles = [
        Line2D([0], [0], marker="x", color="0.6", linestyle="None", label="Far-field OOD set"),
        Line2D([0], [0], marker="o", color=C0, linestyle="None", label="Class 0 (ID)"),
        Line2D([0], [0], marker="o", color=C1, linestyle="None", label="Class 1 (ID)"),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
