"""Temperature Scaling (Platt Scaling) for Feedforward Neural Networks.

Implements post-hoc temperature scaling following:
- "On Calibration of Modern Neural Networks" (Guo et al., 2017)

Temperature scaling learns a single temperature parameter T that scales the logits:
    logits_scaled = logits / T

For binary classification, this is equivalent to Platt scaling.
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from bayescal.models.fnn import FNN


class TemperatureScaledFNN:
    """
    Temperature Scaling for Feedforward Neural Networks.

    This is a post-hoc calibration method that learns a single temperature parameter
    to scale the logits of a pre-trained model. The temperature is learned by minimizing
    the negative log-likelihood on a validation set.

    For binary classification, temperature scaling is equivalent to Platt scaling.

    Workflow:
    1. Train a base FNN to get logits
    2. Learn temperature T by minimizing NLL on validation set
    3. At inference: scale logits by T and apply softmax
    """

    def __init__(
        self,
        base_model: FNN,
        params: dict[str, Any],
        temperature: float,
    ):
        """
        Initialize TemperatureScaledFNN with fitted temperature.

        Args:
            base_model: The trained base FNN model
            params: Trained model parameters
            temperature: Learned temperature parameter
        """
        self.base_model = base_model
        self.params = params
        self.temperature = temperature
        self.num_classes = base_model.num_classes

    @classmethod
    def fit(
        cls,
        base_model: FNN,
        params: dict[str, Any],
        X_val: jnp.ndarray,
        y_val: jnp.ndarray,
        lr: float = 0.01,
        max_iter: int = 1000,
        seed: int = 42,
    ) -> "TemperatureScaledFNN":
        """
        Fit temperature scaling to a trained FNN.

        Learns temperature parameter T by minimizing negative log-likelihood
        on validation data.

        Args:
            base_model: Trained FNN model instance
            params: Trained model parameters
            X_val: Validation features
            y_val: Validation labels (one-hot encoded)
            lr: Learning rate for temperature optimization
            max_iter: Maximum number of optimization iterations
            seed: Random seed

        Returns:
            Fitted TemperatureScaledFNN instance
        """
        X_val = jnp.array(X_val)
        y_val = jnp.array(y_val)

        # Get logits from base model
        # We'll use a workaround: get probabilities and compute logits
        # For temperature scaling, we can work with log-probabilities directly
        # which is mathematically equivalent: scaled_log_probs = log_probs / T
        
        # Get probabilities from base model
        probs_val = base_model.apply(params, X_val, rng=jax.random.PRNGKey(0), training=False)
        
        # Convert to log-probabilities (this is equivalent to logits up to a constant)
        # For temperature scaling: logits_scaled = logits / T
        # Since softmax(logits / T) = softmax(log_probs / T), we can use log_probs
        log_probs_val = jnp.log(probs_val + 1e-10)
        
        # We'll optimize temperature using log-probabilities
        # The scaling is: scaled_log_probs = log_probs / T
        # Then softmax: probs = exp(scaled_log_probs) / sum(exp(scaled_log_probs))

        # Initialize temperature parameter (start at 1.0)
        temperature = jnp.array(1.0)

        # Optimize temperature using gradient descent
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(temperature)

        def loss_fn(temp: jnp.ndarray) -> jnp.ndarray:
            """Negative log-likelihood loss for temperature scaling."""
            # Scale log-probabilities by temperature
            scaled_log_probs = log_probs_val / (temp + 1e-8)
            
            # Renormalize to get probabilities (softmax of scaled log-probs)
            # Subtract max for numerical stability
            scaled_log_probs_max = jnp.max(scaled_log_probs, axis=-1, keepdims=True)
            scaled_log_probs_stable = scaled_log_probs - scaled_log_probs_max
            exp_scaled = jnp.exp(scaled_log_probs_stable)
            probs = exp_scaled / jnp.sum(exp_scaled, axis=-1, keepdims=True)
            
            # Compute NLL
            log_probs = jnp.log(probs + 1e-10)
            nll = -jnp.mean(jnp.sum(y_val * log_probs, axis=1))
            return nll

        # Gradient function
        grad_fn = jax.value_and_grad(loss_fn)

        # Optimization loop
        rng = jax.random.PRNGKey(seed)
        for i in range(max_iter):
            loss, grad = grad_fn(temperature)
            updates, opt_state = optimizer.update(grad, opt_state, temperature)
            temperature = optax.apply_updates(temperature, updates)

            # Clip temperature to reasonable range [0.01, 100]
            temperature = jnp.clip(temperature, 0.01, 100.0)

            # Early stopping if gradient is very small
            if jnp.abs(grad) < 1e-6:
                break

        return cls(
            base_model=base_model,
            params=params,
            temperature=float(temperature),
        )

    def __call__(
        self,
        inputs: jnp.ndarray,
        rng: Any = None,
        training: bool = False,
        n_samples: int = 1,
    ) -> jnp.ndarray:
        """
        Forward pass with temperature scaling.

        Args:
            inputs: Input data of shape (batch_size, input_dim)
            rng: Random number generator (unused, kept for API compatibility)
            training: Whether in training mode (unused, kept for API compatibility)
            n_samples: Number of samples (unused, kept for API compatibility)

        Returns:
            Calibrated class probabilities of shape (batch_size, num_classes)
        """
        # Get probabilities from base model
        probs_base = self.base_model.apply(
            self.params, inputs, rng=rng, training=False
        )
        
        # Convert to log-probabilities and scale by temperature
        log_probs = jnp.log(probs_base + 1e-10)
        scaled_log_probs = log_probs / self.temperature
        
        # Renormalize to get calibrated probabilities
        # Subtract max for numerical stability
        scaled_log_probs_max = jnp.max(scaled_log_probs, axis=-1, keepdims=True)
        scaled_log_probs_stable = scaled_log_probs - scaled_log_probs_max
        exp_scaled = jnp.exp(scaled_log_probs_stable)
        probs = exp_scaled / jnp.sum(exp_scaled, axis=-1, keepdims=True)
        
        return probs

    def apply(
        self,
        params: dict[str, Any],  # Ignored, uses internal params
        inputs: jnp.ndarray,
        rng: Any = None,
        training: bool = False,
        n_samples: int = 1,
    ) -> jnp.ndarray:
        """
        Apply method for API compatibility with other models.

        Args:
            params: Ignored (uses internal params)
            inputs: Input data
            rng: Random number generator
            training: Ignored
            n_samples: Ignored

        Returns:
            Calibrated predicted probabilities
        """
        return self(inputs, rng=rng, training=training, n_samples=n_samples)
