"""Bayesian Dense layer implementation for JAX."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn


class BayesianDense(nn.Module):
    """
    Bayesian dense layer using Bayes by Backprop with Local Reparameterization Trick.

    This layer maintains a distribution over weights rather than point estimates.
    Uses the Local Reparameterization Trick (LRT) for efficient training:
    instead of sampling weights, we directly sample output activations.
    Uses Flax's compact pattern to infer input dimension from first call.
    """

    features: int
    prior_std: float = 1.0
    posterior_std_init: float = 0.1
    max_std: float = (
        0.1  # Maximum allowed standard deviation (caps sigma to prevent excessive noise)
    )

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        rng: Any,
        training: bool = True,
        sample: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass with Local Reparameterization Trick (LRT).

        Args:
            inputs: Input tensor of shape (batch, input_dim)
            rng: Random number generator
            training: Whether in training mode
            sample: Whether to sample from the weight distribution (True) or use mean weights (False)

        Returns:
            Output tensor of shape (batch, features)
        """
        input_dim = inputs.shape[-1]
        weight_shape = (input_dim, self.features)
        bias_shape = (self.features,)

        # Define parameters (Flax will initialize on first call)
        # Weight mean and log_std
        mean = self.param(
            "mean",
            nn.initializers.normal(stddev=0.1),
            weight_shape,
        )
        log_std = self.param(
            "log_std",
            lambda rng, shape: jnp.full(shape, jnp.log(self.posterior_std_init)),
            weight_shape,
        )
        # Bias (deterministic, like standard Dense layers)
        bias = self.param(
            "bias",
            nn.initializers.zeros,
            bias_shape,
        )

        # Use jax.lax.cond for JIT-compatible conditional execution
        def training_forward(inputs, rng, mean, log_std, bias):
            """
            Training mode: Local Reparameterization Trick (LRT) for dense layers.

            Instead of sampling weights and then multiplying, we directly sample
            the output activations. This is more memory efficient and faster.

            For dense(x, W) where W ~ N(μ, σ²) with independent weights:
            - Mean output: x @ μ + bias
            - Variance output: x² @ σ² (element-wise square of inputs, element-wise square of std)
            - Sample: mean + sqrt(variance) * ε
            """
            # Cap log_std to prevent sigma from exceeding max_std
            # This ensures sigma = exp(log_std) <= max_std
            # By clamping log_std, we prevent the parameter from growing unbounded
            max_log_std = jnp.log(self.max_std + 1e-8)  # log(max_std) to cap log_std
            log_std = jnp.clip(
                log_std, -10.0, max_log_std
            )  # Also cap lower bound for stability
            std = jnp.exp(log_std)
            std_sq = std**2  # Variance of weights

            # Compute mean output: inputs @ mean_weights + bias
            mean_output = inputs @ mean + bias

            # Compute variance output: (inputs²) @ (std²)
            # For dense layer: Var[y] = sum(x² * σ²) where x² is element-wise square
            inputs_sq = inputs**2
            var_output = inputs_sq @ std_sq

            # Sample output activations: mean + sqrt(variance) * epsilon
            std_output = jnp.sqrt(
                var_output + 1e-8
            )  # Add small epsilon for numerical stability
            output_shape = mean_output.shape
            eps = jax.random.normal(rng, output_shape)
            sampled_output = mean_output + std_output * eps

            return sampled_output

        def mean_forward(inputs, mean, bias):
            """Use mean weights (deterministic)."""
            return inputs @ mean + bias

        # Use jax.lax.cond for JIT-compatible conditional execution
        return jax.lax.cond(
            sample,
            lambda: training_forward(inputs, rng, mean, log_std, bias),
            lambda: mean_forward(inputs, mean, bias),
        )
