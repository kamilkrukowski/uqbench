"""Training script."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import serialization

from bayescal.config import settings
from bayescal.data import loaders, preprocessing
from bayescal.models import BayesianCNN, CNN, DropoutCNN
from bayescal.training import optimizers, trainer


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["bayesian", "ffn", "dropout_ffn"],
        default="bayesian",
        help="Type of model to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=settings.num_epochs,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.batch_size,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=settings.learning_rate,
        help="Learning rate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.results_dir,
        help="Output directory for saved models",
    )

    args = parser.parse_args()

    # Set random seed
    rng = jax.random.PRNGKey(settings.seed)

    # Load data - always use CIFAR-10 for training
    if settings.downsample_training_factor < 1.0:
        print(f"Downsampling training data by factor {settings.downsample_training_factor:.2f} (using {settings.downsample_training_factor*100:.1f}% of data)")
    train_loader = loaders.load_cifar10(
        split="train",
        batch_size=args.batch_size,
        downsample_factor=settings.downsample_training_factor,
        seed=settings.seed,
    )
    input_shape = (32, 32, 3)  # CIFAR-10: height=32, width=32, channels=3

    # Initialize model
    if args.model_type == "bayesian":
        model = BayesianCNN(
            conv_layers_config=settings.conv_layers,
            num_groups=settings.group_norm_groups,
            num_classes=10,
            prior_std=settings.prior_std,
            posterior_std_init=settings.posterior_std_init,
            beta=settings.beta,
        )
    elif args.model_type == "dropout_ffn":
        model = DropoutCNN(
            conv_layers_config=settings.conv_layers,
            num_groups=settings.group_norm_groups,
            dropout_rate=settings.dropout_rate,
            num_classes=10,
        )
    else:  # feedforward
        model = CNN(
            conv_layers_config=settings.conv_layers,
            num_groups=settings.group_norm_groups,
            num_classes=10,
        )

    # Initialize parameters
    rng, init_rng = jax.random.split(rng)
    params = model.init_params(init_rng, input_shape=input_shape)

    # Initialize optimizer with gradient clipping for stability
    optimizer = optimizers.get_optimizer(
        learning_rate=args.learning_rate,
        optimizer_type="adam",
        max_grad_norm=settings.max_grad_norm,
    )
    opt_state = optimizer.init(params)

    # Training loop
    final_metrics = {}
    # Get beta parameter for Bayesian models (from config or model default)
    if args.model_type == "bayesian":
        beta = settings.beta if hasattr(settings, "beta") else getattr(model, "beta", 0.001)
    else:
        beta = 1.0
    # Get n_vi_samples for Bayesian models
    n_vi_samples = settings.n_vi_samples if args.model_type == "bayesian" else 1
    
    completed_epochs = 0
    interrupted = False
    try:
        for epoch in range(args.epochs):
            rng, epoch_rng = jax.random.split(rng)
            params, opt_state, metrics = trainer.train_epoch(
                model,
                params,
                opt_state,
                train_loader,
                epoch_rng,
                optimizer,
                beta=beta,
                n_vi_samples=n_vi_samples,
            )
            final_metrics = metrics  # Keep track of final epoch metrics
            completed_epochs = epoch + 1
            _metrics = {k: f"{v:.3f}" for k, v in metrics.items()}
            print(f"Epoch {epoch + 1}/{args.epochs}: {_metrics}")
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n⚠️  Training interrupted by user after {completed_epochs}/{args.epochs} epochs")
        print("Saving model before exiting...")

    # Save model (whether completed or interrupted)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model parameters
    model_path = args.output_dir / f"{args.model_type}_cifar10_params.flax"
    model_bytes = serialization.to_bytes(params)
    model_path.write_bytes(model_bytes)
    print(f"Model parameters saved to {model_path}")
    
    # Save model metadata
    metadata = {
        "model_type": args.model_type,
        "dataset": "cifar10",
        "epochs": args.epochs,
        "completed_epochs": completed_epochs,
        "interrupted": interrupted,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "conv_layers_config": [list(layer) for layer in settings.conv_layers],  # CNN architecture: (kx, ky, filters, stride)
        "num_groups": settings.group_norm_groups,  # GroupNorm groups
        "dropout_rate": settings.dropout_rate if args.model_type == "dropout_ffn" else None,  # Dropout rate
        "downsample_training_factor": settings.downsample_training_factor,  # Fraction of training data used
        "num_classes": 10,
        "final_metrics": {k: float(v) for k, v in final_metrics.items()} if final_metrics else {},
    }
    metadata_path = args.output_dir / f"{args.model_type}_cifar10_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()

