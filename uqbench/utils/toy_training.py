"""Training and evaluation utilities for toy dataset experiments."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from uqbench.evaluation import calibration
from uqbench.training import optimizers, trainer


def train_model(
    model: Any,
    model_name: str,
    X_train: np.ndarray,
    y_train_onehot: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 0.01,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
    warm_up_epochs: int = 0,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """
    Train a model on the toy dataset.

    Args:
        model: Model instance to train
        model_name: Name of the model (for logging)
        X_train: Training features
        y_train_onehot: Training labels (one-hot encoded)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        max_grad_norm: Maximum gradient norm for clipping
        seed: Random seed
        verbose: Whether to print training progress
        warm_up_epochs: Number of epochs for KL warmup (only for Bayesian models).
                       Beta gradually increases from 0 to model.beta over warm_up_epochs.
                       Helps model learn data before KL penalty kicks in.

    Returns:
        Tuple of (trained_params, training_history)
    """
    rng = jax.random.PRNGKey(seed)

    # Initialize model
    input_shape = (X_train.shape[1],)
    rng, init_rng = jax.random.split(rng)
    params = model.init_params(init_rng, input_shape=input_shape)

    # Create optimizer with gradient clipping for stability
    optimizer = optimizers.get_optimizer(
        learning_rate=lr, optimizer_type="adam", max_grad_norm=max_grad_norm
    )
    opt_state = optimizer.init(params)

    # Create batches
    n_batches = (len(X_train) + batch_size - 1) // batch_size
    batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_train))
        batch_X = jnp.array(X_train[start_idx:end_idx])
        batch_y = jnp.array(y_train_onehot[start_idx:end_idx])
        batches.append((batch_X, batch_y))

    # Training loop
    history = {"loss": [], "accuracy": []}
    rng, train_rng = jax.random.split(rng)

    # Determine if model is Bayesian (outside loop since it doesn't change)
    is_bayesian = hasattr(model, "compute_kl_divergence") and hasattr(model, "beta")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_kl = 0.0
        epoch_likelihood = 0.0
        num_batches = 0

        for batch_X, batch_y in batches:
            rng, batch_rng = jax.random.split(rng)

            # Determine n_vi_samples for Bayesian models
            # Use 1-2 samples for sharper gradients (better for learning complex patterns)
            # More samples (5-10) reduce variance but can smooth gradients too much
            # For complex patterns like circular boundaries, fewer samples work better
            n_vi_samples = 1 if is_bayesian else 1

            # KL warmup for Bayesian models: gradually increase beta from 0 to final value
            if is_bayesian and warm_up_epochs > 0:
                base_beta = getattr(model, "beta", 1.0)
                if epoch < warm_up_epochs:
                    # Linear warmup: beta = (epoch / warm_up_epochs) * base_beta
                    beta = (epoch / warm_up_epochs) * base_beta
                else:
                    beta = base_beta
            else:
                beta = getattr(model, "beta", 1.0) if is_bayesian else 1.0

            # Get number of training samples for KL normalization (only for Bayesian models)
            n_train = len(X_train) if is_bayesian else None

            params, opt_state, metrics = trainer.train_step(
                model,
                params,
                opt_state,
                (batch_X, batch_y),
                batch_rng,
                optimizer,
                beta=beta,
                n_vi_samples=n_vi_samples,
                n_train=n_train,
            )

            epoch_loss += metrics["loss"]
            epoch_acc += metrics["accuracy"]
            if is_bayesian:
                epoch_kl += metrics.get("kl_loss", 0.0)
                epoch_likelihood += metrics.get("likelihood_loss", 0.0)
            num_batches += 1

        history["loss"].append(epoch_loss / num_batches)
        history["accuracy"].append(epoch_acc / num_batches)

        if verbose and (epoch + 1) % 50 == 0:
            # For Bayesian models, also print KL and likelihood loss
            if is_bayesian:
                avg_kl = epoch_kl / num_batches
                avg_likelihood = epoch_likelihood / num_batches
                # Get the actual beta used (may be warmup beta)
                base_beta = getattr(model, "beta", 1.0)
                if warm_up_epochs > 0 and epoch < warm_up_epochs:
                    current_beta = (epoch / warm_up_epochs) * base_beta
                    warmup_str = f" (warmup β={current_beta:.6f})"
                else:
                    current_beta = base_beta
                    warmup_str = ""
                n_train = len(X_train)
                # Compute normalized KL for display
                avg_kl_normalized = avg_kl / n_train if n_train > 0 else avg_kl
                beta_kl_normalized = current_beta * avg_kl_normalized

                # Extract weight variance statistics for diagnostics
                if hasattr(model, "get_weight_variance_stats"):
                    sigma_stats = model.get_weight_variance_stats(params)
                    sigma_str = f", σ_mean={sigma_stats['mean']:.4f}, σ_median={sigma_stats['median']:.4f}, σ_max={sigma_stats['max']:.4f}"
                else:
                    sigma_str = ""

                print(
                    f"{model_name} - Epoch {epoch+1}/{epochs}: "
                    f"Loss={history['loss'][-1]:.4f}, Acc={history['accuracy'][-1]:.4f}, "
                    f"Likelihood={avg_likelihood:.4f}, KL={avg_kl:.0f}, KL_norm={avg_kl_normalized:.4f}, β*KL_norm={beta_kl_normalized:.4f}{warmup_str}{sigma_str}"
                )
            else:
                print(
                    f"{model_name} - Epoch {epoch+1}/{epochs}: "
                    f"Loss={history['loss'][-1]:.4f}, Acc={history['accuracy'][-1]:.4f}"
                )

    return params, history


def evaluate_model(
    model: Any,
    params: dict[str, Any],
    X_test: np.ndarray,
    y_test_onehot: np.ndarray,
    n_samples: int = 1,
    model_name: str = "",
    seed: int = 42,
    num_bins: int = 10,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate model and return predictions, labels, and metrics.

    Args:
        model: Model instance
        params: Model parameters
        X_test: Test features
        y_test_onehot: Test labels (one-hot encoded)
        n_samples: Number of MC samples for evaluation
        model_name: Name of the model (for logging, optional)
        seed: Random seed
        num_bins: Number of bins for calibration curve
    Returns:
        Tuple of (predictions, labels, metrics_dict, fraction_of_positives, mean_predicted_value)
    """
    rng = jax.random.PRNGKey(seed)
    rng, eval_rng = jax.random.split(rng)

    X_test_jax = jnp.array(X_test)
    y_test_jax = jnp.array(y_test_onehot)

    # Get predictions
    probs = model.apply(
        params, inputs=X_test_jax, rng=eval_rng, training=False, n_samples=n_samples
    )

    # Compute metrics
    predicted_classes = jnp.argmax(probs, axis=-1)
    true_classes = jnp.argmax(y_test_jax, axis=-1)
    accuracy = (predicted_classes == true_classes).mean()

    ece = calibration.expected_calibration_error(probs, y_test_jax, num_bins=num_bins)
    mce = calibration.maximum_calibration_error(probs, y_test_jax, num_bins=num_bins)
    tce = calibration.top_label_calibration_error(probs, y_test_jax, num_bins=num_bins)
    ace = calibration.adaptive_calibration_error(probs, y_test_jax, num_bins=num_bins)
    brier = calibration.brier_score(probs, y_test_jax)

    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration.calibration_curve(
        probs, y_test_jax, prune_small_bins=False, num_bins=num_bins
    )

    metrics = {
        "accuracy": float(accuracy),
        "ece": ece,
        "mce": mce,
        "tce": tce,
        "ace": ace,
        "brier": brier,
    }

    return probs, y_test_jax, metrics, fraction_of_positives, mean_predicted_value
