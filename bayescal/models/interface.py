"""Shared interface for uncertainty quantification methods."""

from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np


class UncertaintyMethod(Protocol):
    """Protocol defining the interface for uncertainty quantification methods."""

    def train(
        self,
        cfg: dict[str, Any],
        data: dict[str, Any],
        rng: jax.random.PRNGKey,
    ) -> dict[str, Any]:
        """
        Train the method and return an artifact (model, params, etc.).

        Args:
            cfg: Method-specific configuration dictionary
            data: Dictionary with 'X_train' and 'y_train_onehot'
            rng: Random number generator key

        Returns:
            Dictionary containing trained model artifacts (e.g., {'model': ..., 'params': ...})
        """
        ...

    def predict_proba(
        self,
        artifact: dict[str, Any],
        X: np.ndarray,
        cfg: dict[str, Any],
        rng: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """
        Compute predictive probabilities.

        Args:
            artifact: Trained model artifact from train()
            X: Input features of shape (n_samples, n_features)
            cfg: Method-specific configuration dictionary
            rng: Random number generator key

        Returns:
            Predicted probabilities of shape (n_samples, n_classes)
        """
        ...

    def inference_cost(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """
        Return inference cost metrics.

        Args:
            cfg: Method-specific configuration dictionary

        Returns:
            Dictionary with cost metrics (e.g., {'forward_passes_per_example': int})
        """
        ...
