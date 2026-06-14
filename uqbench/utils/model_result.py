"""Model result container for unified evaluation and visualization."""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from uqbench.evaluation import calibration


@dataclass
class ModelResult:
    """
    Container for a trained model and all its evaluation artifacts.

    This dataclass provides a unified interface for collecting models,
    evaluating them, and passing results to visualization functions.

    Example:
        >>> results = [
        ...     ModelResult("FNN", fnn_model, fnn_params, n_samples=1, color="#F18F01"),
        ...     ModelResult("BayesianFNN", bayesian_model, bayesian_params, color="#2E86AB"),
        ...     ModelResult("LaplaceFNN", laplace_model, {}, color="#3D5A80"),  # params ignored
        ... ]
        >>> for r in results:
        ...     r.evaluate(X_test, y_test_onehot, num_bins=50)
        >>> plot_calibration_comparison(results, figures_dir)

    Attributes:
        name: Display name for the model (used in plots and tables)
        model: The model instance (any model with .apply() method)
        params: Model parameters (can be empty dict for fitted models like Laplace/MCMC)
        n_samples: Number of MC samples for evaluation (default 100)
        color: Color for plotting (hex string)
        marker: Marker style for plotting
        probs: Predicted probabilities after evaluate() is called
        labels: True labels (one-hot) after evaluate() is called
        metrics: Dict of calibration metrics after evaluate() is called
    """

    name: str
    model: Any
    params: dict[str, Any]

    # Inference settings
    n_samples: int = 100

    # Plotting style
    color: str = "#2E86AB"
    marker: str = "o"

    # Evaluation results (populated after evaluate())
    probs: jnp.ndarray | None = field(default=None, repr=False)
    labels: jnp.ndarray | None = field(default=None, repr=False)
    metrics: dict[str, float] = field(default_factory=dict)

    def evaluate(
        self,
        X_test: np.ndarray | jnp.ndarray,
        y_test_onehot: np.ndarray | jnp.ndarray,
        seed: int = 42,
        num_bins: int = 50,
    ) -> "ModelResult":
        """
        Evaluate the model and populate probs, labels, and metrics.

        Uses the unified .apply() interface that works for all model types:
        - FNN, DropoutFNN, BayesianFNN: uses params
        - LaplaceFNN, MCMCFNN: ignores params (uses internal state)

        Args:
            X_test: Test features array
            y_test_onehot: Test labels (one-hot encoded)
            seed: Random seed for reproducibility
            num_bins: Number of bins for calibration metrics

        Returns:
            self (for method chaining)
        """
        rng = jax.random.PRNGKey(seed)
        rng, eval_rng = jax.random.split(rng)

        X_test_jax = jnp.array(X_test)
        y_test_jax = jnp.array(y_test_onehot)

        # Unified apply interface
        probs = self.model.apply(
            self.params,
            inputs=X_test_jax,
            rng=eval_rng,
            training=False,
            n_samples=self.n_samples,
        )

        # Store results
        self.probs = probs
        self.labels = y_test_jax

        # Compute metrics
        predicted_classes = jnp.argmax(probs, axis=-1)
        true_classes = jnp.argmax(y_test_jax, axis=-1)
        accuracy = float((predicted_classes == true_classes).mean())

        self.metrics = {
            "accuracy": accuracy,
            "ece": calibration.expected_calibration_error(
                probs, y_test_jax, num_bins=num_bins
            ),
            "mce": calibration.maximum_calibration_error(
                probs, y_test_jax, num_bins=num_bins
            ),
            "tce": calibration.top_label_calibration_error(
                probs, y_test_jax, num_bins=num_bins
            ),
            "ace": calibration.adaptive_calibration_error(
                probs, y_test_jax, num_bins=num_bins
            ),
            "brier": calibration.brier_score(probs, y_test_jax),
        }

        return self

    def to_calibration_dict(self) -> dict[str, Any]:
        """
        Convert to format expected by plot_calibration_curves_comparison.

        Returns:
            Dictionary with predictions, labels, metrics, and styling info.
        """
        if self.probs is None:
            raise ValueError(
                f"Model '{self.name}' not evaluated yet. Call evaluate() first."
            )

        return {
            "predictions": self.probs,
            "labels": self.labels,
            "ece": self.metrics.get("ece", 0.0),
            "ace": self.metrics.get("ace", 0.0),
            "tce": self.metrics.get("tce", 0.0),
            "label": self.name,
            "color": self.color,
            "marker": self.marker,
        }

    def to_metrics_row(self) -> dict[str, Any]:
        """
        Convert to a row for a metrics DataFrame.

        Returns:
            Dictionary with model name and all metrics.
        """
        return {"Model": self.name, **self.metrics}

    def __repr__(self) -> str:
        evaluated = "✓" if self.probs is not None else "✗"
        return (
            f"ModelResult(name='{self.name}', n_samples={self.n_samples}, "
            f"evaluated={evaluated})"
        )

