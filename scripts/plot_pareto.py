"""Plot the inference-cost vs probabilistic-quality Pareto frontier.

Reads the multi-seed toy benchmark JSON and plots forward passes (log x)
against NLL (y, with std error bars), colored by ECE. Draws the true
non-dominated frontier (minimizing both cost and NLL) and labels which
methods are dominated.

Usage:
    python scripts/plot_pareto.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def pareto_front(points: list[tuple[str, float, float]]) -> set[str]:
    """Names of non-dominated points minimizing (cost, nll)."""
    front = set()
    for name, cost, nll in points:
        dominated = any(
            (c <= cost and n <= nll) and (c < cost or n < nll)
            for other, c, n in points if other != name
        )
        if not dominated:
            front.add(name)
    return front


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("experiments/results/toy_benchmark.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/results/pareto_frontier_toy.png"))
    args = parser.parse_args()

    summary = json.load(open(args.input))["summary"]
    names = list(summary)
    cost = np.array([summary[n]["fwd_passes"]["mean"] for n in names])
    nll = np.array([summary[n]["nll"]["mean"] for n in names])
    nll_std = np.array([summary[n]["nll"]["std"] for n in names])
    ece = np.array([summary[n]["ece"]["mean"] for n in names])

    front = pareto_front(list(zip(names, cost, nll)))

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(cost, nll, c=ece, s=260, cmap="viridis_r",
                    edgecolors="black", linewidths=1.2, zorder=3)
    ax.errorbar(cost, nll, yerr=nll_std, fmt="none", ecolor="gray",
                elinewidth=1, capsize=3, zorder=2)

    # Frontier line through non-dominated points, sorted by cost.
    fr = sorted([(cost[i], nll[i]) for i, n in enumerate(names) if n in front])
    fx, fy = zip(*fr)
    ax.plot(fx, fy, "--", color="crimson", linewidth=2, zorder=1,
            label="Pareto frontier (non-dominated)")

    # Per-label offsets to avoid overlap among the crowded high-cost points.
    offsets = {
        "DropoutFNN": (10, 8), "MCMCFNN": (10, -16),
        "BayesianFNN": (10, 0), "LaplaceFNN": (10, 0),
        "TemperatureScaledFNN": (-6, 12), "FNN": (10, -14),
    }
    for i, n in enumerate(names):
        dominated = n not in front
        ax.annotate(
            n + ("  (dominated)" if dominated else ""),
            (cost[i], nll[i]), xytext=offsets.get(n, (8, 6)),
            textcoords="offset points",
            fontsize=9, fontstyle="italic" if dominated else "normal",
            color="gray" if dominated else "black",
        )

    ax.set_xscale("log")
    ax.set_xlim(right=float(cost.max()) * 3.0)  # room for right-edge labels
    ax.set_xlabel("Forward passes per prediction (log scale)  —  lower is cheaper")
    ax.set_ylabel("Test NLL  —  lower is better")
    ax.set_title("Inference cost vs probabilistic quality (toy, 5 seeds)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("ECE (Expected Calibration Error)")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print(f"Frontier methods: {sorted(front)}")
    print(f"Dominated: {sorted(set(names) - front)}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
