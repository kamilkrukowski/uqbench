"""Fully Connected Feedforward Neural Network implementations."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn


class FNN(nn.Module):
    """
    Traditional Fully Connected Feedforward Neural Network.

    This model uses point estimates for weights with no regularization.
    """

    hidden_dims: tuple[int, ...] = (512, 256)
    num_classes: int = 10
    dropout_rate: float = 0.0  # No dropout by default

    def setup(self) -> None:
        """Initialize model layers."""
        self.dense_layers = [nn.Dense(features=dim) for dim in self.hidden_dims]
        self.dropout_layers = [
            nn.Dropout(rate=self.dropout_rate) for _ in self.hidden_dims
        ]
        self.output_layer = nn.Dense(self.num_classes)

    def __call__(
        self,
        inputs: jnp.ndarray,
        rng: Any,
        training: bool = True,
        n_samples: int = 1,
    ) -> jnp.ndarray:
        """
        Forward pass through the FNN.

        Args:
            inputs: Input data of shape (batch_size, input_dim) - flattened
            rng: Random number generator (unused if dropout_rate=0)
            training: Whether in training mode
            n_samples: Number of samples (unused, kept for API consistency)

        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        x = inputs

        for dense, dropout in zip(self.dense_layers, self.dropout_layers, strict=False):
            x = dense(x)
            x = nn.relu(x)
            if self.dropout_rate > 0:
                x = dropout(x, rng=rng, deterministic=not training)

        logits = self.output_layer(x)
        probs = nn.softmax(logits)
        return probs

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

        return self.init(rng, dummy_input, rng, training=True)

    def get_loss(
        self,
        params: dict[str, Any],
        inputs: jnp.ndarray,
        labels: jnp.ndarray,
        rng: Any,
        n_vi_samples: int = 1,  # Unused, kept for API consistency
        n_train: int | None = None,  # Unused, kept for API consistency
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """
        Compute cross-entropy loss for FNN.

        Args:
            params: Model parameters
            inputs: Input data of shape (batch_size, input_dim) - flattened
            labels: One-hot encoded labels of shape (batch_size, num_classes)
            rng: Random number generator
            n_vi_samples: Unused, kept for API consistency
            n_train: Unused, kept for API consistency with Bayesian models

        Returns:
            Tuple of (loss, metrics_dict) where metrics includes:
            - accuracy: Classification accuracy
        """
        probs = self.apply(params, inputs=inputs, rng=rng, training=True, n_samples=1)

        # Cross-entropy loss: -sum(y * log(p))
        log_probs = jnp.log(probs + 1e-8)
        loss = -jnp.sum(labels * log_probs, axis=-1).mean()

        # Compute accuracy
        accuracy = (probs.argmax(axis=-1) == labels.argmax(axis=-1)).mean()

        metrics = {"accuracy": accuracy}

        return loss, metrics


class DropoutFNN(nn.Module):
    """
    Fully Connected Feedforward Neural Network with Dropout.

    This model uses point estimates for weights and dropout for regularization.
    Supports Monte Carlo Dropout for uncertainty estimation.
    """

    hidden_dims: tuple[int, ...] = (512, 256)
    num_classes: int = 10
    dropout_rate: float = 0.2

    def setup(self) -> None:
        """Initialize model layers."""
        self.dense_layers = [nn.Dense(features=dim) for dim in self.hidden_dims]
        self.dropout_layers = [
            nn.Dropout(rate=self.dropout_rate) for _ in self.hidden_dims
        ]
        self.output_layer = nn.Dense(self.num_classes)

    def _forward_single(
        self,
        inputs: jnp.ndarray,
        rng: Any,
        use_dropout: bool = True,
    ) -> jnp.ndarray:
        """
        Single forward pass through the FNN.

        Args:
            inputs: Input data of shape (batch_size, input_dim) - flattened
            rng: Random number generator
            use_dropout: Whether to apply dropout (for Monte Carlo Dropout at inference)

        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        x = inputs
        # Only split RNG when dropout is actually used to avoid issues with jax2onnx tracing
        if use_dropout:
            rngs = jax.random.split(rng, len(self.dense_layers))
        else:
            # Create dummy rngs when dropout is disabled (won't be used)
            rngs = [rng] * len(self.dense_layers)

        for dense, dropout, rng_key in zip(
            self.dense_layers, self.dropout_layers, rngs, strict=False
        ):
            x = dense(x)
            x = nn.relu(x)
            # deterministic=False means apply dropout
            # deterministic=True means no dropout
            x = dropout(x, rng=rng_key, deterministic=not use_dropout)

        logits = self.output_layer(x)
        probs = nn.softmax(logits)
        return probs

    def __call__(
        self,
        inputs: jnp.ndarray,
        rng: Any,
        training: bool = True,
        n_samples: int = 1,
    ) -> jnp.ndarray:
        """
        Forward pass through the FNN with Monte Carlo Dropout.

        Args:
            inputs: Input data of shape (batch_size, input_dim) - flattened
            rng: Random number generator
            training: Whether in training mode
            n_samples: Number of Monte Carlo samples to draw. For training, use 1.
                       For inference, use >1 to get predictive posterior via MC Dropout.

        Returns:
            Class probabilities of shape (batch_size, num_classes).
            If n_samples > 1, returns mean probabilities across samples (predictive posterior).
        """
        if n_samples == 1:
            # Single sample: use dropout only during training
            use_dropout = training
            return self._forward_single(inputs, rng, use_dropout=use_dropout)

        # Multiple samples: Monte Carlo Dropout
        # Always use dropout to get predictive posterior, even at inference
        # Use vmap for efficient parallel sampling
        sample_rngs = jax.random.split(rng, n_samples)

        # Vectorize over samples using vmap
        def single_sample(sample_rng):
            return self._forward_single(inputs, sample_rng, use_dropout=True)

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

    def get_loss(
        self,
        params: dict[str, Any],
        inputs: jnp.ndarray,
        labels: jnp.ndarray,
        rng: Any,
        n_vi_samples: int = 1,  # Unused, kept for API consistency
        n_train: int | None = None,  # Unused, kept for API consistency
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """
        Compute cross-entropy loss for DropoutFNN.

        Args:
            params: Model parameters
            inputs: Input data of shape (batch_size, input_dim) - flattened
            labels: One-hot encoded labels of shape (batch_size, num_classes)
            rng: Random number generator
            n_vi_samples: Unused, kept for API consistency
            n_train: Unused, kept for API consistency with Bayesian models

        Returns:
            Tuple of (loss, metrics_dict) where metrics includes:
            - accuracy: Classification accuracy
        """
        probs = self.apply(params, inputs=inputs, rng=rng, training=True, n_samples=1)

        # Cross-entropy loss: -sum(y * log(p))
        log_probs = jnp.log(probs + 1e-8)
        loss = -jnp.sum(labels * log_probs, axis=-1).mean()

        # Compute accuracy
        accuracy = (probs.argmax(axis=-1) == labels.argmax(axis=-1)).mean()

        metrics = {"accuracy": accuracy}

        return loss, metrics
