"""Evaluation script."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import serialization
from tqdm import tqdm

from bayescal.config import settings
from bayescal.data import loaders, preprocessing
from bayescal.evaluation import calibration
from bayescal.models import BayesianCNN, CNN, DropoutCNN
from bayescal.utils import visualization


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to saved model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100"],
        default="cifar10",
        help="Dataset to evaluate on (cifar10 for in-distribution, cifar100 for OOD)",
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
        default=32,
        help="Number of Monte Carlo samples for Bayesian models (use >1 for uncertainty estimation)",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save calibration curve plots to file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots (defaults to model directory)",
    )

    args = parser.parse_args()

    # Set random seed
    rng = jax.random.PRNGKey(settings.seed)

    # Determine paths
    model_path = args.model_path
    
    # Handle different path formats
    if model_path.suffix == ".flax":
        # If params file provided, find corresponding metadata
        metadata_path = model_path.parent / model_path.name.replace("_params.flax", "_metadata.json")
    elif model_path.is_dir():
        # If directory provided, look for .flax file
        flax_files = list(model_path.glob("*_params.flax"))
        if not flax_files:
            raise FileNotFoundError(f"No .flax parameter files found in {model_path}")
        model_path = flax_files[0]
        metadata_path = model_path.parent / model_path.name.replace("_params.flax", "_metadata.json")
    else:
        # Assume base name, try to find .flax file
        if not model_path.suffix:
            # No extension, try adding _params.flax
            potential_path = model_path.parent / f"{model_path.name}_params.flax"
            if potential_path.exists():
                model_path = potential_path
            else:
                # Try as-is with .flax extension
                model_path = model_path.with_suffix(".flax")
        
        # Find corresponding metadata
        if model_path.suffix == ".flax":
            metadata_path = model_path.parent / model_path.name.replace("_params.flax", "_metadata.json")
        else:
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            model_path = model_path.parent / f"{model_path.stem}_params.flax"

    # Load metadata
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load test data
    if args.dataset == "cifar10":
        test_loader = loaders.load_cifar10(
            split="test",
            shuffle=False,
            batch_size=args.batch_size,
        )
        input_shape = (32, 32, 3)  # CIFAR-10: height=32, width=32, channels=3
    else:  # cifar100 for OOD evaluation
        test_loader = loaders.load_cifar100(
            split="test",
            shuffle=False,
            batch_size=args.batch_size,
        )
        input_shape = (32, 32, 3)  # CIFAR-100: height=32, width=32, channels=3

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
    else:  # feedforward
        model = CNN(
            conv_layers_config=conv_layers_config,
            num_groups=num_groups,
            num_classes=num_classes,
        )

    # Initialize model to get parameter structure, then load saved params
    rng, init_rng = jax.random.split(rng)
    dummy_params = model.init_params(init_rng, input_shape=input_shape)
    
    # Load saved parameters
    if not model_path.exists():
        raise FileNotFoundError(f"Model parameters file not found: {model_path}")
    
    model_bytes = model_path.read_bytes()
    params = serialization.from_bytes(dummy_params, model_bytes)

    print(f"Loaded {model_type} model from {model_path}")
    print(f"Model config: conv_layers={conv_layers_config}, num_classes={num_classes}")

    # Evaluate model
    all_predictions = []
    all_labels = []
    
    rng, eval_rng = jax.random.split(rng)
    for batch in tqdm(test_loader, desc="Evaluating"):
        inputs, labels = batch
        rng, batch_rng = jax.random.split(rng)
        
        # Get predictions
        # Both models now support n_samples for Monte Carlo sampling
        probs = model.apply(
            params,
            inputs=inputs,
            rng=batch_rng,
            training=False,
            n_samples=args.n_samples,
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
    brier = calibration.brier_score(predictions, labels)
    
    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration.calibration_curve(
        predictions, labels
    )

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Accuracy: {float(accuracy):.4f}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Brier Score: {brier:.4f}")
    if args.n_samples > 1:
        if model_type == "bayesian":
            mc_type = "Bayesian"
        elif model_type == "dropout_ffn":
            mc_type = "MC Dropout"
        else:
            mc_type = "Samples"  # Traditional CNN doesn't use MC, but n_samples is accepted
        print(f"{mc_type} Samples: {args.n_samples}")
    print("=" * 50)
    
    # Plot calibration curve
    output_dir = args.output_dir or model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = args.dataset
    plot_name = f"{model_type}_{dataset_name}_calibration_curve.png"
    if args.n_samples > 1:
        plot_name = f"{model_type}_{dataset_name}_mc{args.n_samples}_calibration_curve.png"
    
    plot_path = output_dir / plot_name
    visualization.plot_calibration_curve(
        fraction_of_positives,
        mean_predicted_value,
        save_path=plot_path,
    )
    print(f"\nCalibration curve saved to {plot_path}")
    
    # Also try to display interactively if possible
    try:
        import matplotlib.pyplot as plt
        # Recreate plot for display (since plot_calibration_curve closes it)
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
        plt.show()
    except Exception:
        pass  # Silent fail if display not available


if __name__ == "__main__":
    main()

