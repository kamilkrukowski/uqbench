"""Out-of-distribution detection utilities."""

from typing import Any

import jax
import jax.numpy as jnp


def predict_with_uncertainty(
    model: Any,
    params: dict[str, Any],
    inputs: jnp.ndarray,
    rng: Any,
    num_samples: int = 100,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Predict with uncertainty estimation using Monte Carlo sampling.

    Draws individual samples to compute both mean and std of predictions.
    For BayesianMLP, each call samples from the posterior weight distribution.

    Args:
        model: Model instance
        params: Model parameters
        inputs: Input data
        rng: Random number generator
        num_samples: Number of MC samples

    Returns:
        Tuple of (mean_predictions, std_predictions)
    """
    predictions = []
    for _ in range(num_samples):
        rng, sample_rng = jax.random.split(rng)
        # Use n_samples=1 to get individual samples (needed for std calculation)
        # For models without n_samples, this will be ignored
        try:
            probs = model.apply(
                params, inputs=inputs, rng=sample_rng, training=False, n_samples=1
            )
        except TypeError:
            # Model doesn't support n_samples parameter (legacy models)
            probs = model.apply(params, inputs=inputs, rng=sample_rng, training=False)
        predictions.append(probs)

    predictions = jnp.stack(predictions)
    mean_pred = jnp.mean(predictions, axis=0)
    std_pred = jnp.std(predictions, axis=0)

    return mean_pred, std_pred


def detect_ood(
    in_distribution_scores: jnp.ndarray,
    ood_scores: jnp.ndarray,
    threshold_percentile: float = 95.0,
) -> tuple[float, jnp.ndarray]:
    """
    Detect out-of-distribution samples using uncertainty scores.

    Args:
        in_distribution_scores: Uncertainty scores for in-distribution data
        ood_scores: Uncertainty scores for out-of-distribution data
        threshold_percentile: Percentile to use as threshold

    Returns:
        Tuple of (threshold, predictions) where predictions are boolean array
    """
    threshold = jnp.percentile(in_distribution_scores, threshold_percentile)
    predictions = ood_scores > threshold
    return float(threshold), predictions
