"""Bayesian Convolutional Neural Network using Bayes by Backprop."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import traverse_util

from uqbench.models.layers.bayesianconv2d import BayesianConv2D


class BayesianCNN(nn.Module):
    """
    Bayesian Convolutional Neural Network using Bayes by Backprop.

    This model maintains distributions over weights and uses variational inference.
    """

    # Each tuple is (kernel_x, kernel_y, num_filters, stride)
    conv_layers_config: tuple[tuple[int, int, int, int], ...] = (
        (5, 5, 32, 1),
        (5, 5, 64, 1),
    )
    num_classes: int = 10
    num_groups: int = 8  # Number of groups for GroupNorm
    prior_std: float = 1.0
    posterior_std_init: float = 0.1
    beta: float = 0.001  # Beta for beta-VI (KL penalty weight)

    def setup(self) -> None:
        """Initialize model layers."""
        self.conv_layers = [
            BayesianConv2D(
                features=num_filters,
                kernel_size=(kernel_x, kernel_y),
                strides=(stride, stride),
                padding="SAME",
                prior_std=self.prior_std,
                posterior_std_init=self.posterior_std_init,
            )
            for kernel_x, kernel_y, num_filters, stride in self.conv_layers_config
        ]
        # GroupNorm layers after each conv
        self.norm_layers = [
            nn.GroupNorm(num_groups=min(self.num_groups, num_filters), name=f"norm_{i}")
            for i, (_, _, num_filters, _) in enumerate(self.conv_layers_config)
        ]
        # Global average pooling + final dense layer
        self.output_layer = nn.Dense(self.num_classes)

    def _forward_single(
        self,
        inputs: jnp.ndarray,
        rng: Any,
        training: bool = True,
        sample: bool = True,
    ) -> jnp.ndarray:
        """
        Single forward pass through the Bayesian CNN.

        Args:
            inputs: Input data of shape (batch_size, height, width, channels)
            rng: Random number generator for Bayesian layer sampling
            training: Whether in training mode
            sample: Whether to sample from weight distributions (True) or use mean weights (False)

        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        x = inputs
        rngs = jax.random.split(rng, len(self.conv_layers))

        for i, (conv_layer, norm_layer, layer_rng) in enumerate(
            zip(self.conv_layers, self.norm_layers, rngs, strict=False)
        ):
            x = conv_layer(x, layer_rng, training=training, sample=sample)
            x = norm_layer(x)
            x = nn.relu(x)
            # Max pooling after each conv layer, but only if spatial dims are > 1
            # Skip pooling on last layer to avoid dimension collapse
            if i < len(self.conv_layers) - 1:
                h, w = x.shape[1], x.shape[2]
                if h > 1 and w > 1:
                    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Global average pooling: (batch, h, w, c) -> (batch, c)
        # Ensure we have valid spatial dimensions
        if x.shape[1] > 0 and x.shape[2] > 0:
            x = jnp.mean(x, axis=(1, 2))
        else:
            # If dimensions collapsed, just take the first spatial element
            x = x[:, 0, 0, :]

        logits = self.output_layer(x)
        probs = nn.softmax(logits)
        return probs

    def __call__(
        self,
        inputs: jnp.ndarray,
        rng: Any,
        training: bool = True,
        n_samples: int = 1,
        sample: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass through the Bayesian CNN with Monte Carlo sampling.

        Args:
            inputs: Input data of shape (batch_size, height, width, channels)
            rng: Random number generator for Bayesian layer sampling
            training: Whether in training mode
            n_samples: Number of Monte Carlo samples to draw. For training, use 1.
                       For inference, use >1 to get better uncertainty estimates.
            sample: Whether to sample from weight distributions (True) or use mean weights (False).
                   Default True for proper Bayesian uncertainty estimation.

        Returns:
            Class probabilities of shape (batch_size, num_classes).
            If n_samples > 1, returns mean probabilities across samples.
        """
        if n_samples == 1:
            return self._forward_single(inputs, rng, training=training, sample=sample)

        # Multiple samples (for inference/uncertainty estimation)
        # Use vmap for efficient parallel sampling
        sample_rngs = jax.random.split(rng, n_samples)

        # Vectorize over samples using vmap
        def single_sample(sample_rng):
            return self._forward_single(
                inputs, sample_rng, training=training, sample=sample
            )

        # vmap over the rng dimension
        all_probs = jax.vmap(single_sample)(sample_rngs)

        # Average: (n_samples, batch_size, num_classes) -> (batch_size, num_classes)
        mean_probs = jnp.mean(all_probs, axis=0)
        return mean_probs

    def init_params(
        self,
        rng: Any,
        input_shape: tuple[int, ...],
    ) -> dict[str, Any]:
        """
        Initialize model parameters.

        Args:
            rng: Random number generator
            input_shape: Shape of input data (without batch dimension).
                        For images: (height, width, channels)

        Returns:
            Initialized parameters
        """
        # Create dummy input for initialization: (batch_size=1, height, width, channels)
        if len(input_shape) == 3:
            dummy_input = jnp.zeros((1, *input_shape), dtype=jnp.float32)
        elif len(input_shape) == 1:
            # Legacy support: if flattened, reshape to (1, 32, 32, 3) for CIFAR-10
            # Assume CIFAR-10: 3072 = 32*32*3
            if input_shape[0] == 3072:
                dummy_input = jnp.zeros((1, 32, 32, 3), dtype=jnp.float32)
            else:
                raise ValueError(f"Unknown flattened input shape: {input_shape}")
        else:
            dummy_input = jnp.zeros((1, *input_shape), dtype=jnp.float32)

        rng1, rng2 = jax.random.split(rng)
        return self.init(rng1, dummy_input, rng2, training=True)

    def compute_kl_divergence(self, params: dict[str, Any]) -> jnp.ndarray:
        """
        Compute KL divergence between posterior and prior for all Bayesian layers.

        For Gaussian posterior q(w|μ,σ) and prior p(w) = N(0, σ_prior^2):
        KL(q||p) = 0.5 * sum(σ^2/σ_prior^2 + μ^2/σ_prior^2 - 1 - 2*log(σ/σ_prior))

        Args:
            params: Model parameters (Flax nested structure)

        Returns:
            Total KL divergence (scalar)
        """
        total_kl = jnp.array(0.0)
        prior_var = self.prior_std**2

        # Flatten the parameter structure to access nested layers
        flat_params = traverse_util.flatten_dict(params, sep="/")

        # Group parameters by layer (params/conv_layers_0/mean, params/conv_layers_0/log_std, etc.)
        # Note: Flax adds a "params/" prefix when flattening
        layer_params = {}
        for key, value in flat_params.items():
            # Look for keys like "params/conv_layers_0/mean" or "conv_layers_0/mean"
            parts = key.split("/")
            # Handle both "params/conv_layers_0/mean" and "conv_layers_0/mean" formats
            if len(parts) >= 2:
                # Check if first part is "params" and second part starts with "conv_layers_"
                if (
                    parts[0] == "params"
                    and len(parts) >= 3
                    and parts[1].startswith("conv_layers_")
                ):
                    layer_name = parts[1]
                    param_name = "/".join(parts[2:])
                # Or check if first part starts with "conv_layers_" directly
                elif parts[0].startswith("conv_layers_"):
                    layer_name = parts[0]
                    param_name = "/".join(parts[1:])
                else:
                    continue

                if layer_name not in layer_params:
                    layer_params[layer_name] = {}
                layer_params[layer_name][param_name] = value

        # Debug: Check if we found any layers (this will help diagnose the issue)
        # Note: This print won't work inside JAX tracing, but we can check the structure

        # Compute KL for each Bayesian conv layer
        for layer_name, layer_dict in layer_params.items():
            if "mean" in layer_dict and "log_std" in layer_dict:
                mean = layer_dict["mean"]
                log_std = layer_dict["log_std"]
                std = jnp.exp(log_std)
                std_sq = std**2

                # KL divergence for Gaussian: 0.5 * sum(σ^2/σ_prior^2 + μ^2/σ_prior^2 - 1 - 2*log(σ/σ_prior))
                kl = 0.5 * jnp.sum(
                    std_sq / prior_var
                    + (mean**2) / prior_var
                    - 1.0
                    - 2.0 * log_std
                    + 2.0 * jnp.log(self.prior_std)
                )
                total_kl = total_kl + kl

        return total_kl

    def get_loss(
        self,
        params: dict[str, Any],
        inputs: jnp.ndarray,
        labels: jnp.ndarray,
        rng: Any,
        n_vi_samples: int = 1,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """
        Compute loss for Bayesian CNN with variational inference.

        For n_vi_samples > 1, computes loss per sample and averages (proper VI).
        Loss = likelihood_loss + beta * kl_loss

        Args:
            params: Model parameters
            inputs: Input data of shape (batch_size, height, width, channels)
            labels: One-hot encoded labels of shape (batch_size, num_classes)
            rng: Random number generator
            n_vi_samples: Number of samples for variational inference during training.
                         Use >1 for more stable gradient estimates.

        Returns:
            Tuple of (total_loss, metrics_dict) where metrics includes:
            - accuracy: Classification accuracy
            - kl_loss: KL divergence term
            - likelihood_loss: Cross-entropy loss term
        """
        # For n_vi_samples > 1, compute loss per sample and average (proper VI)
        if n_vi_samples > 1:
            sample_rngs = jax.random.split(rng, n_vi_samples)

            def single_sample_loss(sample_rng):
                probs = self.apply(
                    params, inputs=inputs, rng=sample_rng, training=True, n_samples=1
                )
                log_probs = jnp.log(probs + 1e-8)
                return -jnp.sum(labels * log_probs, axis=-1)

            # Compute loss for each sample and average
            sample_losses = jax.vmap(single_sample_loss)(sample_rngs)
            likelihood_loss = jnp.mean(sample_losses)
            # Use averaged probabilities for accuracy
            probs = self.apply(
                params, inputs=inputs, rng=rng, training=True, n_samples=n_vi_samples
            )
        else:
            probs = self.apply(
                params, inputs=inputs, rng=rng, training=True, n_samples=1
            )
            # Cross-entropy loss: -sum(y * log(p))
            log_probs = jnp.log(probs + 1e-8)
            likelihood_loss = -jnp.sum(labels * log_probs, axis=-1).mean()

        # Compute KL divergence
        kl_loss = self.compute_kl_divergence(params)

        # Total loss with beta-VI
        total_loss = likelihood_loss + self.beta * kl_loss

        # Compute accuracy
        accuracy = (probs.argmax(axis=-1) == labels.argmax(axis=-1)).mean()

        metrics = {
            "accuracy": accuracy,
            "kl_loss": kl_loss,
            "likelihood_loss": likelihood_loss,
        }

        return total_loss, metrics
