"""Bayesian Fully Connected Feedforward Neural Network using Bayes by Backprop."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import traverse_util

from uqbench.models.layers.bayesiandense import BayesianDense


class BayesianFNN(nn.Module):
    """
    Bayesian Fully Connected Feedforward Neural Network using Bayes by Backprop.

    This model maintains distributions over weights and uses variational inference.
    """

    hidden_dims: tuple[int, ...] = (512, 256)
    num_classes: int = 10
    prior_std: float = 1.0
    posterior_std_init: float = 0.1
    beta: float = 0.001  # Beta for beta-VI (KL penalty weight)
    max_std: float = 0.1  # Maximum allowed weight standard deviation (caps sigma)

    def setup(self) -> None:
        """Initialize model layers."""
        self.dense_layers = [
            BayesianDense(
                features=dim,
                prior_std=self.prior_std,
                posterior_std_init=self.posterior_std_init,
                max_std=self.max_std,
            )
            for dim in self.hidden_dims
        ]
        # Output layer should also be Bayesian for proper uncertainty quantification
        self.output_layer = BayesianDense(
            features=self.num_classes,
            prior_std=self.prior_std,
            posterior_std_init=self.posterior_std_init,
            max_std=self.max_std,
        )

    def _forward_single(
        self,
        inputs: jnp.ndarray,
        rng: Any,
        training: bool = True,
        sample: bool = True,
    ) -> jnp.ndarray:
        """
        Single forward pass through the Bayesian FNN.

        Args:
            inputs: Input data of shape (batch_size, input_dim) - flattened
            rng: Random number generator for Bayesian layer sampling
            training: Whether in training mode
            sample: Whether to sample from weight distributions (True) or use mean weights (False)

        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        x = inputs
        # Split RNG for all layers (hidden + output)
        rngs = jax.random.split(rng, len(self.dense_layers) + 1)

        for dense_layer, layer_rng in zip(self.dense_layers, rngs[:-1], strict=False):
            x = dense_layer(x, layer_rng, training=training, sample=sample)
            x = nn.relu(x)

        # Output layer is also Bayesian - apply softmax to get probabilities
        logits = self.output_layer(x, rngs[-1], training=training, sample=sample)
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
        Forward pass through the Bayesian FNN with Monte Carlo sampling.

        Args:
            inputs: Input data of shape (batch_size, input_dim) - flattened
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
                        For flattened images: (3072,) for CIFAR-10

        Returns:
            Initialized parameters
        """
        # Create dummy input for initialization: (batch_size=1, input_dim)
        if len(input_shape) == 1:
            dummy_input = jnp.zeros((1, input_shape[0]), dtype=jnp.float32)
        elif len(input_shape) == 3:
            # If image shape provided, flatten it
            dummy_input = jnp.zeros(
                (1, input_shape[0] * input_shape[1] * input_shape[2]), dtype=jnp.float32
            )
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

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

        # Group parameters by layer (params/dense_layers_0/mean, params/output_layer/mean, etc.)
        layer_params = {}
        for key, value in flat_params.items():
            parts = key.split("/")
            if len(parts) >= 2:
                # Check if first part is "params" and second part is a Bayesian layer
                if parts[0] == "params" and len(parts) >= 3:
                    if (
                        parts[1].startswith("dense_layers_")
                        or parts[1] == "output_layer"
                    ):
                        layer_name = parts[1]
                        param_name = "/".join(parts[2:])
                    else:
                        continue
                elif parts[0].startswith("dense_layers_") or parts[0] == "output_layer":
                    layer_name = parts[0]
                    param_name = "/".join(parts[1:])
                else:
                    continue

                if layer_name not in layer_params:
                    layer_params[layer_name] = {}
                layer_params[layer_name][param_name] = value

        # Compute KL for each Bayesian dense layer
        for layer_name, layer_dict in layer_params.items():
            if "mean" in layer_dict and "log_std" in layer_dict:
                mean = layer_dict["mean"]
                log_std = layer_dict["log_std"]
                std = jnp.exp(log_std)
                std_sq = std**2

                # KL divergence for Gaussian
                kl = 0.5 * jnp.sum(
                    std_sq / prior_var
                    + (mean**2) / prior_var
                    - 1.0
                    - 2.0 * log_std
                    + 2.0 * jnp.log(self.prior_std)
                )
                total_kl = total_kl + kl

        return total_kl

    def get_weight_variance_stats(self, params: dict[str, Any]) -> dict[str, float]:
        """
        Extract weight variance (sigma) statistics for diagnostic purposes.

        Returns statistics about the posterior standard deviations (sigma = exp(log_std))
        across all Bayesian layers. Useful for diagnosing if variances are too large.

        Args:
            params: Model parameters (Flax nested structure)

        Returns:
            Dictionary with sigma statistics: mean, median, max, min, std
        """
        import jax.numpy as jnp
        from flax import traverse_util

        all_sigmas = []
        flat_params = traverse_util.flatten_dict(params, sep="/")

        # Extract all log_std parameters and convert to sigma (with clamp applied)
        max_log_std = jnp.log(self.max_std + 1e-8)  # Same clamp as in forward pass
        for key, value in flat_params.items():
            if "log_std" in key:
                # Apply the same clamp as in forward pass, then convert to sigma
                log_std_clamped = jnp.clip(value, -10.0, max_log_std)
                sigma = jnp.exp(log_std_clamped)
                all_sigmas.append(sigma.flatten())

        if len(all_sigmas) == 0:
            return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}

        all_sigmas = jnp.concatenate(all_sigmas)

        return {
            "mean": float(jnp.mean(all_sigmas)),
            "median": float(jnp.median(all_sigmas)),
            "max": float(jnp.max(all_sigmas)),
            "min": float(jnp.min(all_sigmas)),
            "std": float(jnp.std(all_sigmas)),
        }

    def get_loss(
        self,
        params: dict[str, Any],
        inputs: jnp.ndarray,
        labels: jnp.ndarray,
        rng: Any,
        n_vi_samples: int = 1,
        n_train: int | None = None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """
        Compute loss for Bayesian FNN with variational inference.

        For n_vi_samples > 1, computes loss per sample and averages (proper VI).
        Loss = likelihood_loss + beta * (kl_loss / n_train)

        The KL divergence is normalized by the number of training samples to ensure
        proper scaling regardless of dataset or model size. This follows the standard
        ELBO formulation: ELBO = E[log p(y|x,θ)] - (1/N) * KL(q(θ)||p(θ))

        Args:
            params: Model parameters
            inputs: Input data of shape (batch_size, input_dim) - flattened
            labels: One-hot encoded labels of shape (batch_size, num_classes)
            rng: Random number generator
            n_vi_samples: Number of samples for variational inference during training.
                         Use >1 for more stable gradient estimates.
            n_train: Number of training samples. If None, KL is not normalized.
                    Should be set to len(X_train) for proper scaling.

        Returns:
            Tuple of (total_loss, metrics_dict) where metrics includes:
            - accuracy: Classification accuracy
            - kl_loss: KL divergence term (unnormalized, for logging)
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

        # Normalize KL by number of training samples (standard ELBO formulation)
        # This ensures KL doesn't dominate for large models/datasets
        if n_train is not None and n_train > 0:
            kl_loss_normalized = kl_loss / n_train
        else:
            kl_loss_normalized = kl_loss

        # Total loss with beta-VI
        total_loss = likelihood_loss + self.beta * kl_loss_normalized

        # Compute accuracy
        accuracy = (probs.argmax(axis=-1) == labels.argmax(axis=-1)).mean()

        metrics = {
            "accuracy": accuracy,
            "kl_loss": kl_loss,  # Unnormalized for logging
            "kl_loss_normalized": kl_loss_normalized,  # Normalized for actual loss
            "likelihood_loss": likelihood_loss,
        }

        return total_loss, metrics
