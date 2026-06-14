"""Visualization utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_calibration_curve(
    fraction_of_positives: np.ndarray,
    mean_predicted_value: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    Plot calibration curve.

    Args:
        fraction_of_positives: Fraction of positive predictions per bin
        mean_predicted_value: Mean predicted probability per bin
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(8, 8))
    plt.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label="Model",
    )
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confidence_distribution(
    confidences: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    Plot distribution of prediction confidences.

    Args:
        confidences: Array of confidence scores
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(8, 6))
    plt.hist(confidences, bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Confidences")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
