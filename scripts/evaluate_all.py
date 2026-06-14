"""Evaluate all models and create overlapping calibration curves."""

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import serialization
from tqdm import tqdm

from uqbench.config import settings
from uqbench.data import loaders
from uqbench.evaluation import calibration
from uqbench.models import BayesianCNN, CNN, DropoutCNN


def load_model(
    model_path: Path, 
    metadata_path: Path, 
    input_shape: tuple[int, ...] = (3072,)
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load model and parameters from saved files."""
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Reconstruct model
    model_type = metadata["model_type"]
    num_classes = metadata["num_classes"]
    
    # Support new conv_layers_config format and legacy formats
    if "conv_layers_config" in metadata:
        conv_layers_config = tuple(tuple(layer) for layer in metadata["conv_layers_config"])
    elif "num_filters" in metadata:
        # Legacy: convert num_filters to conv_layers_config
        num_filters = tuple(metadata["num_filters"])
        conv_stride = tuple(metadata.get("conv_stride", [1, 1]))
        # Default to 5x5 kernels
        conv_layers_config = tuple((5, 5, f, conv_stride[0]) for f in num_filters)
    elif "hidden_dims" in metadata:
        # Very old legacy: convert hidden_dims to conv_layers_config
        hidden_dims = tuple(metadata["hidden_dims"])
        # Default CNN architecture with 5x5 kernels
        conv_layers_config = ((5, 5, 32, 1), (5, 5, 64, 1))
    else:
        # Default CNN architecture
        conv_layers_config = ((5, 5, 32, 1), (5, 5, 64, 1))
    
    # Get num_groups and dropout_rate from metadata or use defaults
    num_groups = metadata.get("num_groups", 8)
    dropout_rate = metadata.get("dropout_rate", 0.2)
    
    if model_type == "bayesian":
        model = BayesianCNN(
            conv_layers_config=conv_layers_config,
            num_groups=num_groups,
            num_classes=num_classes,
        )
    elif model_type == "dropout_ffn":
        model = DropoutCNN(
            conv_layers_config=conv_layers_config,
            num_groups=num_groups,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
        )
    else:  # ffn
        model = CNN(
            conv_layers_config=conv_layers_config,
            num_groups=num_groups,
            num_classes=num_classes,
        )
    
    # Initialize model to get parameter structure, then load saved params
    rng = jax.random.PRNGKey(settings.seed)
    rng, init_rng = jax.random.split(rng)
    dummy_params = model.init_params(init_rng, input_shape=input_shape)
    
    # Load saved parameters
    model_bytes = model_path.read_bytes()
    params = serialization.from_bytes(dummy_params, model_bytes)
    
    return model, params, metadata


def evaluate_model(
    model: Any,
    params: dict[str, Any],
    test_loader: list,
    model_type: str,
    n_samples: int,
    rng: Any,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, float]]:
    """Evaluate a model and return predictions, labels, and metrics."""
    all_predictions = []
    all_labels = []
    
    for batch in tqdm(test_loader, desc=f"Evaluating {model_type}", leave=False):
        inputs, labels = batch
        rng, batch_rng = jax.random.split(rng)
        
        # Get predictions
        probs = model.apply(
            params,
            inputs=inputs,
            rng=batch_rng,
            training=False,
            n_samples=n_samples,
        )
        
        all_predictions.append(probs)
        all_labels.append(labels)
    
    # Concatenate all predictions and labels
    predictions = jnp.concatenate(all_predictions, axis=0)
    labels = jnp.concatenate(all_labels, axis=0)
    
    # Compute metrics
    predicted_classes = jnp.argmax(predictions, axis=-1)
    true_classes = jnp.argmax(labels, axis=-1)
    accuracy = (predicted_classes == true_classes).mean()
    
    ece = calibration.expected_calibration_error(predictions, labels)
    mce = calibration.maximum_calibration_error(predictions, labels)
    brier = calibration.brier_score(predictions, labels)
    
    metrics = {
        "accuracy": float(accuracy),
        "ece": ece,
        "mce": mce,
        "brier": brier,
    }
    
    return predictions, labels, metrics


def plot_overlapping_calibration_curves(
    results_pruned: dict[str, tuple[np.ndarray, np.ndarray, dict[str, float]]],
    results_unpruned: dict[str, tuple[np.ndarray, np.ndarray, dict[str, float]]],
    save_path: Path | None,
    dataset: str,
) -> None:
    """Plot overlapping calibration curves for all models (both pruned and unpruned)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Color and marker mapping for each model type
    model_styles = {
        "bayesian": {"color": "#2E86AB", "marker": "o", "label": "Bayesian CNN"},
        "dropout_ffn": {"color": "#A23B72", "marker": "s", "label": "Dropout CNN"},
        "ffn": {"color": "#F18F01", "marker": "^", "label": "Traditional CNN"},
    }
    
    # Plot pruned curves (left subplot)
    for model_type, (fraction_of_positives, mean_predicted_value, metrics) in results_pruned.items():
        style = model_styles.get(model_type, {"color": "gray", "marker": "x", "label": model_type})
        ece = metrics["ece"]
        label = f"{style['label']} (ECE: {ece:.3f})"
        
        ax1.plot(
            mean_predicted_value,
            fraction_of_positives,
            marker=style["marker"],
            color=style["color"],
            label=label,
            linewidth=2,
            markersize=8,
            alpha=0.8,
        )
    
    # Plot unpruned curves (right subplot)
    for model_type, (fraction_of_positives, mean_predicted_value, metrics) in results_unpruned.items():
        style = model_styles.get(model_type, {"color": "gray", "marker": "x", "label": model_type})
        ece = metrics["ece"]
        label = f"{style['label']} (ECE: {ece:.3f})"
        
        ax2.plot(
            mean_predicted_value,
            fraction_of_positives,
            marker=style["marker"],
            color=style["color"],
            label=label,
            linewidth=2,
            markersize=8,
            alpha=0.8,
        )
    
    # Plot perfect calibration line on both subplots
    for ax in [ax1, ax2]:
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    ax1.set_title(f"Pruned (bins with ≥1% samples) - {dataset.upper()}", fontsize=14, fontweight="bold")
    ax2.set_title(f"Unpruned (all bins) - {dataset.upper()}", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nOverlapping calibration curves saved to {save_path}")
    # If save_path is None, don't close - let caller handle display
    # Note: Caller should call plt.show() or plt.close() after this function


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate all models and create comparison plots")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=settings.results_dir,
        help="Directory containing saved models",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100"],
        default="cifar10",
        help="Dataset to evaluate on (cifar10 for in-distribution, cifar100 for OOD)",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["cifar10", "cifar10c"],
        default="cifar10",
        help="Test dataset: 'cifar10' for clean test set, 'cifar10c' for CIFAR-10-C corrupted",
    )
    parser.add_argument(
        "--corruption-type",
        type=str,
        default="gaussian_noise",
        help="Corruption type for CIFAR-10-C (only used if --test=cifar10c). "
             "Options: gaussian_noise, shot_noise, impulse_noise, defocus_blur, "
             "glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, "
             "contrast, elastic_transform, pixelate, jpeg_compression, etc.",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Severity level for CIFAR-10-C (1-5, only used if --test=cifar10c)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.batch_size,
        help="Batch size",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=512,
        help="Number of Monte Carlo samples for Bayesian/Dropout models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots (defaults to results-dir)",
    )

    args = parser.parse_args()

    # Set random seed
    rng = jax.random.PRNGKey(settings.seed)

    # Load test data based on --test parameter
    if args.test == "cifar10c":
        # Load CIFAR-10-C corrupted dataset
        print(f"Loading CIFAR-10-C with corruption: {args.corruption_type}, severity: {args.severity}")
        test_loader = loaders.load_cifar10_corrupted(
            corruption_type=args.corruption_type,
            severity=args.severity,
            shuffle=False,
            batch_size=args.batch_size,
        )
        input_shape = (32, 32, 3)  # CIFAR-10-C: height=32, width=32, channels=3
        dataset_name = f"cifar10c_{args.corruption_type}_s{args.severity}"
    elif args.dataset == "cifar10":
        # Load clean CIFAR-10 test set
        test_loader = loaders.load_cifar10(
            split="test",
            shuffle=False,
            batch_size=args.batch_size,
        )
        input_shape = (32, 32, 3)  # CIFAR-10: height=32, width=32, channels=3
        dataset_name = "cifar10"
    else:  # cifar100 for OOD evaluation
        test_loader = loaders.load_cifar100(
            split="test",
            shuffle=False,
            batch_size=args.batch_size,
        )
        input_shape = (32, 32, 3)  # CIFAR-100: height=32, width=32, channels=3
        dataset_name = "cifar100"

    # Model types to evaluate
    model_types = ["bayesian", "ffn"]#, "dropout_ffn"]
    
    # Evaluate all models
    all_results = {}
    calibration_curves_pruned = {}
    calibration_curves_unpruned = {}
    
    print("=" * 60)
    print("Evaluating all models")
    print("=" * 60)
    
    for model_type in model_types:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {model_type} model...")
        print(f"{'=' * 60}")
        
        # Find model files (models are always trained on CIFAR-10)
        model_path = args.results_dir / f"{model_type}_cifar10_params.flax"
        metadata_path = args.results_dir / f"{model_type}_cifar10_metadata.json"
        
        if not model_path.exists() or not metadata_path.exists():
            print(f"⚠️  Skipping {model_type}: model files not found")
            continue
        
        # Load model
        model, params, metadata = load_model(model_path, metadata_path, input_shape=input_shape)
        print(f"Loaded {model_type} model from {model_path}")
        
        # Evaluate
        rng, eval_rng = jax.random.split(rng)
        predictions, labels, metrics = evaluate_model(
            model, params, test_loader, model_type, args.n_samples, eval_rng
        )
        
        # Compute calibration curves (both pruned and unpruned)
        fraction_of_positives_pruned, mean_predicted_value_pruned = calibration.calibration_curve(
            predictions, labels, prune_small_bins=True
        )
        fraction_of_positives_unpruned, mean_predicted_value_unpruned = calibration.calibration_curve(
            predictions, labels, prune_small_bins=False
        )
        
        all_results[model_type] = {
            "predictions": predictions,
            "labels": labels,
            "metrics": metrics,
        }
        calibration_curves_pruned[model_type] = (
            np.array(fraction_of_positives_pruned),
            np.array(mean_predicted_value_pruned),
            metrics,
        )
        calibration_curves_unpruned[model_type] = (
            np.array(fraction_of_positives_unpruned),
            np.array(mean_predicted_value_unpruned),
            metrics,
        )
        
        # Print metrics
        print(f"\n{model_type.upper()} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  ECE: {metrics['ece']:.4f}")
        print(f"  MCE: {metrics['mce']:.4f}")
        print(f"  Brier Score: {metrics['brier']:.4f}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 80)
    print(f"{'Model':<20} {'Accuracy':<12} {'ECE':<12} {'MCE':<12} {'Brier':<12}")
    print("-" * 80)
    for model_type, results in all_results.items():
        m = results["metrics"]
        print(
            f"{model_type:<20} {m['accuracy']:<12.4f} {m['ece']:<12.4f} "
            f"{m['mce']:<12.4f} {m['brier']:<12.4f}"
        )
    print("=" * 80)
    print("\nNote: ECE is weighted by bin size, so low ECE can hide miscalibration")
    print("      in bins with few samples. MCE shows the worst-case error.")
    print("=" * 80)
    
    # Create overlapping calibration curve plot
    if calibration_curves_pruned and calibration_curves_unpruned:
        output_dir = args.output_dir or args.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = output_dir / f"calibration_curves_comparison_{dataset_name}_mc{args.n_samples}.png"
        plot_overlapping_calibration_curves(
            calibration_curves_pruned, 
            calibration_curves_unpruned, 
            plot_path, 
            dataset_name
        )
        
        # Also try to display interactively
        try:
            # Create a new figure for interactive display
            plot_overlapping_calibration_curves(
                calibration_curves_pruned,
                calibration_curves_unpruned,
                None,
                dataset_name
            )
            plt.show()
        except Exception:
            pass  # Silent fail if display not available


if __name__ == "__main__":
    main()

