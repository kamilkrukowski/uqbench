"""Visualization utilities for toy dataset experiments."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from uqbench.utils.model_result import ModelResult


def plot_toy_dataset(
    X: np.ndarray,
    y: np.ndarray,
    n_plot_points: int = 2000,
    seed: int = 42,
    figures_dir: Path | None = None,
) -> None:
    """
    Visualize the toy dataset as a scatter plot.

    Args:
        X: Feature array (n_samples, 2)
        y: Labels array (n_samples,)
        n_plot_points: Number of points to plot (subsampled if larger)
        seed: Random seed for subsampling
        figures_dir: Directory to save figures (optional)
    """
    # Subsample if large
    if len(X) > n_plot_points:
        _, X_plot, _, y_plot = train_test_split(
            X, y, test_size=n_plot_points / len(X), random_state=seed, stratify=y
        )
    else:
        X_plot, y_plot = X, y

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    scatter = ax.scatter(
        X_plot[:, 0],
        X_plot[:, 1],
        c=y_plot,
        cmap="coolwarm",
        alpha=0.5,
        s=15,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)
    ax.set_title(
        "Toy Dataset: 2-Class Overlapping Gaussians", fontsize=14, fontweight="bold"
    )
    ax.legend(*scatter.legend_elements(), title="Class", loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if figures_dir:
        plt.savefig(figures_dir / "toy_dataset.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_calibration_curves_comparison(
    model_results: list[dict[str, Any]],
    figures_dir: Path | None = None,
    metric: str = "ece",
    num_bins: int = 15,
) -> None:
    """
    Plot calibration curves for multiple models on the same figure.

    The binning scheme changes based on the metric:
    - 'ece' / 'tce': Fixed-width bins (standard calibration curve)
    - 'ace': Adaptive bins with equal samples per bin (quantile-based)
    - 'all': Uses fixed-width bins but shows all metrics in legend

    Args:
        model_results: List of dictionaries, each containing:
            - 'predictions': Predicted probabilities array (n_samples, n_classes)
            - 'labels': True labels (one-hot) array (n_samples, n_classes)
            - 'ece': Expected calibration error (float)
            - 'ace': Adaptive calibration error (float, optional)
            - 'tce': Top-label calibration error (float, optional)
            - 'label': Model label string
            - 'color': Color for the plot (optional)
            - 'marker': Marker style (optional, default 'o')

            Legacy support (if predictions/labels not provided):
            - 'mpv': Mean predicted value array (pre-computed)
            - 'fop': Fraction of positives array (pre-computed)
        figures_dir: Directory to save figures (optional)
        metric: Which metric to display and binning to use. Options: 'ece', 'ace', 'tce', 'all'
        num_bins: Number of bins for calibration curve (default 15)
    """
    from uqbench.evaluation import calibration

    # Default colors and markers if not provided
    default_colors = ["#F18F01", "#A23B72", "#2E86AB", "#C73E1D", "#6A994E"]
    default_markers = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for i, result in enumerate(model_results):
        color = result.get("color", default_colors[i % len(default_colors)])
        marker = result.get("marker", default_markers[i % len(default_markers)])
        label = result.get("label", f"Model {i+1}")

        # Compute calibration curve based on metric
        if "predictions" in result and "labels" in result:
            predictions = result["predictions"]
            labels = result["labels"]

            # For binary classification, use P(class=1) directly to get full [0, 1] range
            # For multiclass, use top-label confidence
            n_classes = predictions.shape[1] if len(predictions.shape) > 1 else 2
            strategy = "binary_class1" if n_classes == 2 else "top_label"

            # Use adaptive binning for ACE, fixed-width for ECE/TCE
            if metric == "ace":
                fop, mpv = calibration.adaptive_calibration_curve(
                    predictions, labels, num_bins=num_bins, strategy=strategy
                )
            else:
                # ECE, TCE, and 'all' use fixed-width bins
                fop, mpv = calibration.calibration_curve(
                    predictions,
                    labels,
                    num_bins=num_bins,
                    prune_small_bins=False,
                    strategy=strategy,
                )
        else:
            # Legacy support: use pre-computed mpv/fop
            mpv = result.get("mpv", np.array([]))
            fop = result.get("fop", np.array([]))

        # Build metric string based on selected metric
        if metric == "all":
            ece = result.get("ece", 0.0)
            ace = result.get("ace", 0.0)
            tce = result.get("tce", 0.0)
            metric_str = f"ECE={ece:.3f}, ACE={ace:.3f}, TCE={tce:.3f}"
        elif metric == "ace":
            ace = result.get("ace", 0.0)
            metric_str = f"ACE={ace:.3f}"
        elif metric == "tce":
            tce = result.get("tce", 0.0)
            metric_str = f"TCE={tce:.3f}"
        else:  # default to ece
            ece = result.get("ece", 0.0)
            metric_str = f"ECE={ece:.3f}"

        # Filter out NaN values for plotting
        valid_mask = ~np.isnan(fop) & ~np.isnan(mpv)
        mpv_valid = mpv[valid_mask]
        fop_valid = fop[valid_mask]

        ax.plot(
            mpv_valid,
            fop_valid,
            f"{marker}-",
            label=f"{label} ({metric_str})",
            color=color,
            linewidth=2,
            markersize=8,
            alpha=0.8,
        )

    # Perfect calibration line
    ax.plot(
        [0.0, 1], [0.0, 1], "k--", label="Perfect calibration", linewidth=2, alpha=0.7
    )

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)

    # Title based on metric
    metric_names = {
        "ece": "ECE (Fixed Bins)",
        "ace": "ACE (Adaptive Bins)",
        "tce": "TCE (Fixed Bins)",
        "all": "All Metrics (Fixed Bins)",
    }
    title_metric = metric_names.get(metric, "ECE")
    ax.set_title(
        f"Calibration Curves Comparison - {title_metric}",
        fontsize=14,
        fontweight="bold",
    )

    ax.legend(loc="lower right", fontsize=10 if metric == "all" else 11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    plt.tight_layout()
    if figures_dir:
        suffix = f"_{metric}" if metric != "ece" else ""
        plt.savefig(
            figures_dir / f"calibration_curves_comparison{suffix}.png",
            dpi=150,
            bbox_inches="tight",
        )
    plt.show()


def plot_single_sample_predictions(
    model: Any,
    params: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    n_samples_to_show: int = 5,
    n_plot_points: int = 2000,
    seed: int = 42,
    figures_dir: Path | None = None,
) -> None:
    """
    Plot decision boundaries from individual MC samples (not averaged).

    Args:
        model: Model instance
        params: Model parameters
        X_test: Test features
        y_test: Test labels (not one-hot)
        model_name: Name of the model
        n_samples_to_show: Number of individual samples to plot
        n_plot_points: Number of test points to plot (subsampled if larger)
        seed: Random seed
        figures_dir: Directory to save figures (optional)
    """
    # Subsample test set for visualization
    if len(X_test) > n_plot_points:
        _, X_test_plot, _, y_test_plot = train_test_split(
            X_test,
            y_test,
            test_size=n_plot_points / len(X_test),
            random_state=seed,
            stratify=y_test,
        )
    else:
        X_test_plot, y_test_plot = X_test, y_test

    # Create a grid
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_jax = jnp.array(grid_points)

    rng = jax.random.PRNGKey(seed)

    # Get individual sample predictions (not averaged)
    fig, axes = plt.subplots(1, n_samples_to_show, figsize=(5 * n_samples_to_show, 5))
    if n_samples_to_show == 1:
        axes = [axes]

    for i in range(n_samples_to_show):
        rng, eval_rng = jax.random.split(rng)
        # Single sample, no averaging
        probs = model.apply(
            params, inputs=grid_jax, rng=eval_rng, training=False, n_samples=1
        )
        probs_class1 = np.array(probs[:, 1]).reshape(xx.shape)

        contour = axes[i].contourf(
            xx, yy, probs_class1, levels=20, cmap="coolwarm", alpha=0.8
        )
        axes[i].contour(
            xx, yy, probs_class1, levels=[0.5], colors="black", linewidths=2
        )
        axes[i].scatter(
            X_test_plot[:, 0],
            X_test_plot[:, 1],
            c=y_test_plot,
            cmap="coolwarm",
            edgecolors="black",
            linewidths=0.3,
            s=15,
            alpha=0.5,
        )
        axes[i].set_title(f"{model_name} - Sample {i+1}", fontsize=10)
        axes[i].set_xlabel("Feature 1", fontsize=9)
        axes[i].set_ylabel("Feature 2", fontsize=9)

    plt.suptitle(
        f"{model_name}: Individual MC Samples (n_samples=1 each)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    if figures_dir:
        model_name_safe = model_name.lower().replace(" ", "_")
        plt.savefig(
            figures_dir / f"{model_name_safe}_individual_samples.png",
            dpi=150,
            bbox_inches="tight",
        )
    plt.show()

    # Also plot the averaged version for comparison
    print(f"\n{model_name} - Averaged over 100 samples:")
    rng_avg = jax.random.PRNGKey(seed)
    rng_avg, eval_rng_avg = jax.random.split(rng_avg)
    probs_avg = model.apply(
        params, inputs=grid_jax, rng=eval_rng_avg, training=False, n_samples=100
    )
    probs_class1_avg = np.array(probs_avg[:, 1]).reshape(xx.shape)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    contour = ax.contourf(
        xx, yy, probs_class1_avg, levels=20, cmap="coolwarm", alpha=0.8
    )
    ax.contour(xx, yy, probs_class1_avg, levels=[0.5], colors="black", linewidths=2)
    ax.scatter(
        X_test_plot[:, 0],
        X_test_plot[:, 1],
        c=y_test_plot,
        cmap="coolwarm",
        edgecolors="black",
        linewidths=0.3,
        s=15,
        alpha=0.5,
    )
    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)
    ax.set_title(
        f"{model_name}: Averaged Decision Boundary (MC=100)",
        fontsize=12,
        fontweight="bold",
    )
    plt.colorbar(contour, ax=ax)
    plt.tight_layout()
    if figures_dir:
        model_name_safe = model_name.lower().replace(" ", "_")
        plt.savefig(
            figures_dir / f"{model_name_safe}_averaged_boundary_mc100.png",
            dpi=150,
            bbox_inches="tight",
        )
    plt.show()


def plot_predictive_posterior(
    model: Any,
    params: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    n_samples: int = 1,
    n_plot_points: int = 2000,
    seed: int = 42,
    figures_dir: Path | None = None,
    ax: Any | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> Any:
    """
    Plot predictive posterior over the input space.

    Args:
        model: Model instance
        params: Model parameters
        X_test: Test features
        y_test: Test labels (not one-hot)
        model_name: Name of the model
        n_samples: Number of MC samples for prediction
        n_plot_points: Number of test points to plot (subsampled if larger)
        seed: Random seed
        figures_dir: Directory to save figures (optional)
        ax: Matplotlib axes to plot on. If None, creates a new figure.
        vmin: Minimum value for color scale (default 0.0 for probability)
        vmax: Maximum value for color scale (default 1.0 for probability)

    Returns:
        The axes object used for plotting (for subplot compatibility)
    """
    # Subsample test set for visualization
    if len(X_test) > n_plot_points:
        _, X_test_plot, _, y_test_plot = train_test_split(
            X_test,
            y_test,
            test_size=n_plot_points / len(X_test),
            random_state=seed,
            stratify=y_test,
        )
    else:
        X_test_plot, y_test_plot = X_test, y_test

    # Create a grid matching the data extent (no extra margins)
    x_min, x_max = X_test[:, 0].min(), X_test[:, 0].max()
    y_min, y_max = X_test[:, 1].min(), X_test[:, 1].max()
    grid_resolution = 60  # Reduced from 100 to save memory
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Get predictions on grid
    rng = jax.random.PRNGKey(seed)
    rng, eval_rng = jax.random.split(rng)
    grid_jax = jnp.array(grid_points)

    probs = model.apply(
        params, inputs=grid_jax, rng=eval_rng, training=False, n_samples=n_samples
    )
    probs_class1 = np.array(probs[:, 1]).reshape(xx.shape)

    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        created_figure = True
    else:
        created_figure = False
        fig = ax.figure

    # Probability contour with fixed color scale for comparison across plots
    levels = np.linspace(vmin, vmax, 21)
    contour = ax.contourf(
        xx,
        yy,
        probs_class1,
        levels=levels,
        cmap="coolwarm",
        alpha=0.8,
        vmin=vmin,
        vmax=vmax,
    )

    # Overlay decision boundary (where P(Class=1) = 0.5)
    boundary_contour = ax.contour(
        xx,
        yy,
        probs_class1,
        levels=[0.5],
        colors="black",
        linewidths=2.5,
        linestyles="-",
    )

    # Scatter test points
    scatter = ax.scatter(
        X_test_plot[:, 0],
        X_test_plot[:, 1],
        c=y_test_plot,
        cmap="coolwarm",
        edgecolors="black",
        linewidths=0.3,
        s=15,
        alpha=0.5,
        zorder=10,
    )

    # Labels and title
    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)
    ax.set_title(
        f"{model_name}: Predictive Probability P(Class=1)",
        fontsize=12,
        fontweight="bold",
    )

    # Colorbar for probability - use fig.colorbar when ax is provided
    if created_figure:
        plt.colorbar(contour, ax=ax, label="P(Class=1)")
    else:
        fig.colorbar(contour, ax=ax, label="P(Class=1)")

    # Legend for decision boundary
    boundary_line = Line2D(
        [0],
        [0],
        color="black",
        linewidth=2.5,
        linestyle="-",
        label="Decision Boundary (P=0.5)",
    )
    ax.legend(handles=[boundary_line], loc="upper right", fontsize=11)

    # Only save/show if we created the figure
    if created_figure:
        plt.tight_layout()
        if figures_dir:
            model_name_safe = model_name.lower().replace(" ", "_")
            plt.savefig(
                figures_dir
                / f"{model_name_safe}_predictive_posterior_mc{n_samples}.png",
                dpi=150,
                bbox_inches="tight",
            )
        plt.show()

    return ax


def plot_uncertainty(
    model: Any,
    params: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    n_samples: int = 100,
    n_plot_points: int = 2000,
    seed: int = 42,
    figures_dir: Path | None = None,
    ax: Any | None = None,
    vmin: float | None = 0.0,
    vmax: float | None = None,
    log_scale: bool = False,
) -> Any:
    """
    Plot uncertainty (entropy of predictions) over the input space.

    Args:
        model: Model instance
        params: Model parameters
        X_test: Test features
        y_test: Test labels (not one-hot)
        model_name: Name of the model
        n_samples: Number of MC samples for uncertainty estimation
        n_plot_points: Number of test points to plot (subsampled if larger)
        seed: Random seed
        figures_dir: Directory to save figures (optional)
        ax: Matplotlib axes to plot on. If None, creates a new figure.
        vmin: Minimum value for color scale (default 0.0)
        vmax: Maximum value for color scale (default log(2) ≈ 0.693 for binary entropy)
        log_scale: If True, apply log transform to entropy values for visualization

    Returns:
        The axes object used for plotting (for subplot compatibility)
    """
    # Default vmax to max binary entropy: log(2) ≈ 0.693
    if vmax is None:
        vmax = np.log(2)

    # Subsample test set for visualization
    if len(X_test) > n_plot_points:
        _, X_test_plot, _, y_test_plot = train_test_split(
            X_test,
            y_test,
            test_size=n_plot_points / len(X_test),
            random_state=seed,
            stratify=y_test,
        )
    else:
        X_test_plot, y_test_plot = X_test, y_test

    # Create a grid matching the data extent (no extra margins)
    x_min, x_max = X_test[:, 0].min(), X_test[:, 0].max()
    y_min, y_max = X_test[:, 1].min(), X_test[:, 1].max()
    grid_resolution = 60  # Reduced from 100 to save memory
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Get multiple predictions on grid
    rng = jax.random.PRNGKey(seed)
    grid_jax = jnp.array(grid_points)

    all_probs = []
    for i in range(n_samples):
        rng, eval_rng = jax.random.split(rng)
        probs = model.apply(
            params, inputs=grid_jax, rng=eval_rng, training=False, n_samples=1
        )
        all_probs.append(np.array(probs))

    all_probs = np.array(all_probs)  # (n_samples, n_points, n_classes)
    mean_probs = np.mean(all_probs, axis=0)
    std_probs = np.std(all_probs, axis=0)

    # Use entropy as uncertainty measure: -sum(p * log(p))
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)
    entropy = entropy.reshape(xx.shape)

    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        created_figure = True
    else:
        created_figure = False
        fig = ax.figure

    # Plot with fixed color scale for comparison across plots
    if log_scale:
        # Apply log transform: log(1 + entropy) to handle zeros
        # This emphasizes differences in low-entropy regions
        entropy_plot = np.log1p(
            entropy * 10
        )  # Scale by 10 before log for better visibility
        plot_vmin = np.log1p(vmin * 10) if vmin is not None else 0
        plot_vmax = (
            np.log1p(vmax * 10) if vmax is not None else np.log1p(np.log(2) * 10)
        )
        levels = np.linspace(plot_vmin, plot_vmax, 21)
        contour = ax.contourf(
            xx, yy, entropy_plot, levels=levels, cmap="viridis", alpha=0.8
        )
    else:
        levels = np.linspace(vmin, vmax, 21)
        contour = ax.contourf(
            xx,
            yy,
            entropy,
            levels=levels,
            cmap="viridis",
            alpha=0.8,
            vmin=vmin,
            vmax=vmax,
        )
    ax.scatter(
        X_test_plot[:, 0],
        X_test_plot[:, 1],
        c=y_test_plot,
        cmap="coolwarm",
        edgecolors="black",
        linewidths=0.3,
        s=15,
        alpha=0.5,
        zorder=10,
    )
    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)
    title_suffix = " (Log Scale)" if log_scale else ""
    ax.set_title(
        f"{model_name}: Predictive Uncertainty (Entropy){title_suffix}",
        fontsize=12,
        fontweight="bold",
    )

    # Colorbar - use fig.colorbar when ax is provided
    colorbar_label = "log(1 + 10×Entropy)" if log_scale else "Entropy (bits)"
    if created_figure:
        plt.colorbar(contour, ax=ax, label=colorbar_label)
    else:
        fig.colorbar(contour, ax=ax, label=colorbar_label)

    # Only save/show if we created the figure
    if created_figure:
        plt.tight_layout()
        if figures_dir:
            model_name_safe = model_name.lower().replace(" ", "_")
            plt.savefig(
                figures_dir / f"{model_name_safe}_uncertainty_mc{n_samples}.png",
                dpi=150,
                bbox_inches="tight",
            )
        plt.show()

    return ax


# =============================================================================
# High-level wrappers for ModelResult objects
# =============================================================================


def plot_calibration_comparison(
    results: list[ModelResult],
    figures_dir: Path | None = None,
    metric: str = "tce",
    num_bins: int = 15,
) -> None:
    """
    Plot calibration curves for multiple ModelResult objects.

    This is a convenience wrapper around plot_calibration_curves_comparison
    that automatically extracts data from ModelResult objects.

    Args:
        results: List of evaluated ModelResult objects
        figures_dir: Directory to save figures (optional)
        metric: Which metric to display ('ece', 'ace', 'tce', 'all')
        num_bins: Number of bins for calibration curve
    """
    model_dicts = [r.to_calibration_dict() for r in results]
    plot_calibration_curves_comparison(model_dicts, figures_dir, metric, num_bins)


def plot_posterior_grid(
    results: list[ModelResult],
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    figures_dir: Path | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """
    Plot predictive posteriors for all models in a single row of subplots.

    Args:
        results: List of evaluated ModelResult objects
        X_test: Test features
        y_test: Test labels (not one-hot)
        seed: Random seed
        figures_dir: Directory to save figures (optional)
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))

    # Handle single model case
    if n_models == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        plot_predictive_posterior(
            r.model,
            r.params,
            X_test,
            y_test,
            r.name,
            n_samples=r.n_samples,
            seed=seed,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )

    plt.tight_layout()
    if figures_dir:
        plt.savefig(
            figures_dir / "predictive_posterior_comparison.png",
            dpi=150,
            bbox_inches="tight",
        )
    plt.show()


def plot_uncertainty_grid(
    results: list[ModelResult],
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    figures_dir: Path | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
    log_scale: bool = False,
) -> None:
    """
    Plot uncertainty (entropy) for all models in a single row of subplots.

    Args:
        results: List of evaluated ModelResult objects
        X_test: Test features
        y_test: Test labels (not one-hot)
        seed: Random seed
        figures_dir: Directory to save figures (optional)
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale (default: log(2) for binary entropy)
        log_scale: If True, apply log transform to entropy values
    """
    # Default vmax to max binary entropy
    if vmax is None:
        vmax = np.log(2)

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))

    # Handle single model case
    if n_models == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        plot_uncertainty(
            r.model,
            r.params,
            X_test,
            y_test,
            r.name,
            n_samples=r.n_samples,
            seed=seed,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            log_scale=log_scale,
        )

    plt.tight_layout()
    if figures_dir:
        plt.savefig(
            figures_dir / "uncertainty_entropy_comparison.png",
            dpi=150,
            bbox_inches="tight",
        )
    plt.show()
