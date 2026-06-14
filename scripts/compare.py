"""Comparison script for Bayesian vs Feedforward models."""

import argparse
from pathlib import Path

from bayescal.config import settings
from bayescal.evaluation import calibration
from bayescal.utils import visualization


def main() -> None:
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description="Compare Bayesian and Feedforward models"
    )
    parser.add_argument(
        "--bayesian-model",
        type=Path,
        required=True,
        help="Path to Bayesian model",
    )
    parser.add_argument(
        "--feedforward-model",
        type=Path,
        required=True,
        help="Path to Feedforward model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.results_dir,
        help="Output directory for comparison results",
    )

    args = parser.parse_args()

    # TODO: Load models and evaluate
    # TODO: Calculate ECE, Brier scores
    # TODO: Generate calibration curves
    # TODO: Compare OOD detection performance

    print("Comparison complete")


if __name__ == "__main__":
    main()

