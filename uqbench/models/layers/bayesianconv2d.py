"""Bayesian 2D Convolutional layer implementation for JAX."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn


class BayesianConv2D(nn.Module):
    """
    Bayesian 2D Convolutional layer using Bayes by Backprop with Local Reparameterization Trick.

    This layer maintains a distribution over weights rather than point estimates.
    Uses the Local Reparameterization Trick (LRT) for efficient training:
    instead of sampling weights, we directly sample output activations.
    Uses Flax's compact pattern to infer input shape from first call.
    """

    features: int
    kernel_size: tuple[int, int] = (3, 3)
    strides: tuple[int, int] = (1, 1)
    padding: str = "SAME"
    prior_std: float = 1.0
    posterior_std_init: float = 0.1

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
            inputs: Input tensor of shape (batch, height, width, channels)
            rng: Random number generator
            training: Whether in training mode
            sample: Whether to sample from the weight distribution (True) or use mean weights (False)

        Returns:
            Output tensor of shape (batch, height', width', features)
        """
        input_channels = inputs.shape[-1]
        kernel_shape = (
            self.kernel_size[0],
            self.kernel_size[1],
            input_channels,
            self.features,
        )

        # Define parameters (Flax will initialize on first call)
        mean = self.param(
            "mean",
            nn.initializers.normal(stddev=0.1),
            kernel_shape,
        )
        log_std = self.param(
            "log_std",
            lambda rng, shape: jnp.full(shape, jnp.log(self.posterior_std_init)),
            kernel_shape,
        )

        # Define bias parameter
        bias = self.param(
            "bias",
            nn.initializers.zeros,
            (self.features,),
        )

        dimension_numbers = ("NHWC", "HWIO", "NHWC")
        padding = self.padding

        def training_forward(inputs, rng, mean, log_std):
            """
            Training mode: Local Reparameterization Trick (LRT) for CNNs.

            Instead of sampling weights and then convolving, we directly sample
            the output activations. This is more memory efficient and faster.

            For conv(x, W) where W ~ N(μ, σ²) with independent weights:
            - Each output element y[i,j,k] = Σ_{patch} x[patch] * W[patch, k]
            - Mean: E[y[i,j,k]] = Σ_{patch} x[patch] * μ[patch, k] = conv(x, μ)
            - Variance: Var[y[i,j,k]] = Σ_{patch} x[patch]² * σ²[patch, k] = conv(x², σ²)
              (variance of sum of independent RVs = sum of variances)
            - Sample: mean + sqrt(variance) * ε

            This works because convolution applies the same operation over patches,
            and the variance computation correctly accounts for the patch-wise summation.
            """
            std = jnp.exp(log_std)
            std_sq = std**2  # Variance of weights

            # Compute mean output: conv(inputs, mean_weights)
            # This computes E[y] = Σ_{patch} x[patch] * μ[patch]
            mean_output = jax.lax.conv_general_dilated(
                inputs,
                mean,
                window_strides=self.strides,
                padding=padding,
                dimension_numbers=dimension_numbers,
            )

            # Compute variance output: conv(inputs², std²)
            # This computes Var[y] = Σ_{patch} x[patch]² * σ²[patch]
            # The element-wise square of inputs is crucial: variance scales with x²
            inputs_sq = inputs**2
            var_output = jax.lax.conv_general_dilated(
                inputs_sq,
                std_sq,
                window_strides=self.strides,
                padding=padding,
                dimension_numbers=dimension_numbers,
            )

            # Sample output activations: mean + sqrt(variance) * epsilon
            std_output = jnp.sqrt(
                var_output + 1e-8
            )  # Add small epsilon for numerical stability
            output_shape = mean_output.shape
            eps = jax.random.normal(rng, output_shape)
            sampled_output = mean_output + std_output * eps

            return sampled_output

        def mean_forward(inputs, mean):
            """Use mean weights (deterministic)."""
            kernel = mean
            return jax.lax.conv_general_dilated(
                inputs,
                kernel,
                window_strides=self.strides,
                padding=padding,
                dimension_numbers=dimension_numbers,
            )

        # Use jax.lax.cond for JIT-compatible conditional execution
        out = jax.lax.cond(
            sample,
            lambda: training_forward(inputs, rng, mean, log_std),
            lambda: mean_forward(inputs, mean),
        )

        # Add bias
        out = out + bias

        return out
