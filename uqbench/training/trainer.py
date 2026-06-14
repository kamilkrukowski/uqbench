"""Training loop implementation."""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from tqdm import tqdm

from uqbench.evaluation import metrics as eval_metrics

# Create a cached JIT-compiled function for gradient computation
# This will be compiled once per unique function signature
_grad_fn_cache = {}


def clear_grad_cache() -> None:
    """Clear the JIT compilation cache for gradient functions.

    Call this if you change model architecture and get shape errors.
    """
    global _grad_fn_cache
    _grad_fn_cache = {}


def train_step(
    model: nn.Module,
    params: dict[str, Any],
    opt_state: optax.OptState,
    batch: tuple[jnp.ndarray, jnp.ndarray],
    rng: Any,
    optimizer: optax.GradientTransformation,
    beta: float = 1.0,
    n_vi_samples: int = 1,
    n_train: int | None = None,
) -> tuple[dict[str, Any], optax.OptState, dict[str, float]]:
    """
    Single training step.

    Args:
        model: Model instance
        params: Model parameters
        opt_state: Optimizer state
        batch: Training batch (inputs, labels)
        rng: Random number generator
        optimizer: Optimizer
        beta: Beta parameter for beta-VI (only used for Bayesian models)
        n_vi_samples: Number of samples for variational inference during training.
                     Used for Bayesian models to get more stable gradient estimates.

    Returns:
        Updated parameters, optimizer state, and metrics
    """
    inputs, labels = batch
    rng, step_rng = jax.random.split(rng)

    # Determine n_vi_samples: use provided value for Bayesian models, 1 otherwise
    is_bayesian = hasattr(model, "compute_kl_divergence") and hasattr(model, "beta")
    n_samples = n_vi_samples if is_bayesian else 1

    # Create a loss function that captures the model's get_loss method
    # We'll JIT-compile the gradient computation
    def loss_fn(p: dict[str, Any]) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        return model.get_loss(
            p,
            inputs=inputs,
            labels=labels,
            rng=step_rng,
            n_vi_samples=n_samples,
            n_train=n_train,
        )

    # JIT-compile the value_and_grad computation
    # Use a cache key based on the model type and architecture to avoid recompiling unnecessarily
    # Include model architecture (hidden_dims, num_classes) to prevent shape mismatches
    # when architecture changes between runs
    model_config = None
    if hasattr(model, "hidden_dims") and hasattr(model, "num_classes"):
        model_config = (model.hidden_dims, model.num_classes)
    elif hasattr(model, "conv_layers_config") and hasattr(model, "num_classes"):
        model_config = (
            model.conv_layers_config,
            model.num_classes,
        )

    # Note: n_train is not included in cache key as it's a static argument
    cache_key = (type(model).__name__, is_bayesian, n_samples, model_config)
    if cache_key not in _grad_fn_cache:
        # Create a template function that will be JIT-compiled
        # The actual model.get_loss will be called inside, but JAX will trace through it
        # Mark n_samples_val as static so the Python if statement in get_loss can be evaluated
        def grad_fn_template(p, inputs, labels, rng_key, n_samples_val, n_train_val):
            def inner_loss(pp):
                # This will call model.get_loss, which JAX can trace through
                return model.get_loss(
                    pp,
                    inputs=inputs,
                    labels=labels,
                    rng=rng_key,
                    n_vi_samples=n_samples_val,
                    n_train=n_train_val,
                )

            return jax.value_and_grad(inner_loss, has_aux=True)(p)

        # Mark n_samples_val and n_train_val (argument indices 4, 5) as static so Python conditionals work
        _grad_fn_cache[cache_key] = jax.jit(grad_fn_template, static_argnums=(4, 5))

    grad_fn = _grad_fn_cache[cache_key]
    (loss, metrics), grads = grad_fn(
        params, inputs, labels, step_rng, n_samples, n_train
    )

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    # Clamp log_std parameters for Bayesian models to prevent sigma from exceeding max_std
    # This must be done after parameter updates to prevent unbounded growth
    if is_bayesian and hasattr(model, "max_std"):
        from flax import traverse_util

        max_std = model.max_std
        max_log_std = jnp.log(max_std + 1e-8)

        # Flatten params, clamp log_std, then unflatten
        flat_params = traverse_util.flatten_dict(params, sep="/")
        clamped_params = {}
        for key, value in flat_params.items():
            if "log_std" in key:
                # Clamp log_std to prevent sigma from exceeding max_std
                clamped_params[key] = jnp.clip(value, -10.0, max_log_std)
            else:
                clamped_params[key] = value
        params = traverse_util.unflatten_dict(clamped_params, sep="/")

    # Convert JAX arrays to Python floats after gradient computation
    metrics_float = {k: float(v) for k, v in metrics.items()}
    metrics_float["loss"] = float(loss)
    return params, opt_state, metrics_float


def train_epoch(
    model: nn.Module,
    params: dict[str, Any],
    opt_state: optax.OptState,
    train_loader: Any,
    rng: Any,
    optimizer: optax.GradientTransformation,
    beta: float = 1.0,
    n_vi_samples: int = 1,
) -> tuple[dict[str, Any], optax.OptState, dict[str, float]]:
    """
    Train for one epoch.

    Args:
        model: Model instance
        params: Model parameters
        opt_state: Optimizer state
        train_loader: Training data loader
        rng: Random number generator
        optimizer: Optimizer
        beta: Beta parameter for beta-VI (only used for Bayesian models)
        n_vi_samples: Number of samples for variational inference during training.
                     Used for Bayesian models to get more stable gradient estimates.

    Returns:
        Updated parameters, optimizer state, and epoch metrics
    """
    epoch_metrics = {}
    num_batches = 0

    # Collect predictions and labels for macro metrics
    all_predictions = []
    all_labels = []

    for batch in tqdm(train_loader, desc="Training"):
        inputs, labels = batch
        params, opt_state, batch_metrics = train_step(
            model,
            params,
            opt_state,
            batch,
            rng,
            optimizer,
            beta=beta,
            n_vi_samples=n_vi_samples,
        )
        # Accumulate all metrics from batch_metrics
        for key, value in batch_metrics.items():
            if key not in epoch_metrics:
                epoch_metrics[key] = 0.0
            epoch_metrics[key] += value
        num_batches += 1
        rng, batch_rng = jax.random.split(rng)

        # Get predictions for macro metrics (using current params)
        probs = model.apply(
            params, inputs=inputs, rng=batch_rng, training=False, n_samples=1
        )
        all_predictions.append(probs)
        all_labels.append(labels)

    # Average batch metrics (skip macro metrics which are computed separately)
    macro_keys = {"macro_auroc", "macro_f1"}
    for key in epoch_metrics:
        if key not in macro_keys:
            epoch_metrics[key] /= num_batches

    # Compute macro metrics across entire epoch
    if all_predictions:
        predictions = jnp.concatenate(all_predictions, axis=0)
        labels = jnp.concatenate(all_labels, axis=0)

        epoch_metrics["macro_auroc"] = eval_metrics.macro_auroc(predictions, labels)
        epoch_metrics["macro_f1"] = eval_metrics.macro_f1(predictions, labels)
    else:
        epoch_metrics["macro_auroc"] = 0.0
        epoch_metrics["macro_f1"] = 0.0

    return params, opt_state, epoch_metrics
