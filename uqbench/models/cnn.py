"""Convolutional Neural Network implementations."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn


class CNN(nn.Module):
    """
    Traditional Convolutional Neural Network without Dropout.

    This model uses point estimates for weights with no regularization.
    Demonstrates poor calibration without uncertainty estimation.
    """

    # Each tuple is (kernel_x, kernel_y, num_filters, stride)
    conv_layers_config: tuple[tuple[int, int, int, int], ...] = (
        (5, 5, 32, 1),
        (5, 5, 64, 1),
    )
    num_classes: int = 10
    num_groups: int = 8  # Number of groups for GroupNorm

    def setup(self) -> None:
        """Initialize model layers."""
        self.conv_layers = [
            nn.Conv(
                features=num_filters,
                kernel_size=(kernel_x, kernel_y),
                strides=(stride, stride),
                padding="SAME",
            )
            for kernel_x, kernel_y, num_filters, stride in self.conv_layers_config
        ]
        # GroupNorm layers after each conv
        self.norm_layers = [
            nn.GroupNorm(num_groups=min(self.num_groups, num_filters), name=f"norm_{i}")
            for i, (_, _, num_filters, _) in enumerate(self.conv_layers_config)
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
        Forward pass through the CNN.

        Args:
            inputs: Input data of shape (batch_size, height, width, channels)
            rng: Random number generator (unused, kept for API consistency)
            training: Whether in training mode (unused, kept for API consistency)
            n_samples: Number of samples (unused, kept for API consistency).
                       Always returns deterministic predictions.

        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        x = inputs

        for i, (conv, norm) in enumerate(
            zip(self.conv_layers, self.norm_layers, strict=False)
        ):
            x = conv(x)
            x = norm(x)
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
            if input_shape[0] == 3072:
                dummy_input = jnp.zeros((1, 32, 32, 3), dtype=jnp.float32)
            else:
                raise ValueError(f"Unknown flattened input shape: {input_shape}")
        else:
            dummy_input = jnp.zeros((1, *input_shape), dtype=jnp.float32)

        return self.init(rng, dummy_input, rng, training=True)

    def get_loss(
        self,
        params: dict[str, Any],
        inputs: jnp.ndarray,
        labels: jnp.ndarray,
        rng: Any,
        n_vi_samples: int = 1,  # Unused, kept for API consistency
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """
        Compute cross-entropy loss for CNN.

        Args:
            params: Model parameters
            inputs: Input data of shape (batch_size, height, width, channels)
            labels: One-hot encoded labels of shape (batch_size, num_classes)
            rng: Random number generator (unused, kept for API consistency)
            n_vi_samples: Unused, kept for API consistency

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


class DropoutCNN(nn.Module):
    """
    Convolutional Neural Network with Dropout.

    This model uses point estimates for weights and dropout for regularization.
    Supports Monte Carlo Dropout for uncertainty estimation.
    """

    # Each tuple is (kernel_x, kernel_y, num_filters, stride)
    conv_layers_config: tuple[tuple[int, int, int, int], ...] = (
        (5, 5, 32, 1),
        (5, 5, 64, 1),
    )
    num_classes: int = 10
    num_groups: int = 8  # Number of groups for GroupNorm
    dropout_rate: float = 0.2

    def setup(self) -> None:
        """Initialize model layers."""
        self.conv_layers = [
            nn.Conv(
                features=num_filters,
                kernel_size=(kernel_x, kernel_y),
                strides=(stride, stride),
                padding="SAME",
            )
            for kernel_x, kernel_y, num_filters, stride in self.conv_layers_config
        ]
        # GroupNorm layers after each conv
        self.norm_layers = [
            nn.GroupNorm(num_groups=min(self.num_groups, num_filters), name=f"norm_{i}")
            for i, (_, _, num_filters, _) in enumerate(self.conv_layers_config)
        ]
        self.dropout_layers = [
            nn.Dropout(rate=self.dropout_rate) for _ in self.conv_layers_config
        ]
        self.output_layer = nn.Dense(self.num_classes)

    def _forward_single(
        self,
        inputs: jnp.ndarray,
        rng: Any,
        use_dropout: bool = True,
    ) -> jnp.ndarray:
        """
        Single forward pass through the CNN.

        Args:
            inputs: Input data of shape (batch_size, height, width, channels)
            rng: Random number generator
            use_dropout: Whether to apply dropout (for Monte Carlo Dropout at inference)

        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        x = inputs
        rngs = jax.random.split(rng, len(self.conv_layers))

        for i, (conv, norm, dropout, rng_key) in enumerate(
            zip(
                self.conv_layers,
                self.norm_layers,
                self.dropout_layers,
                rngs,
                strict=False,
            )
        ):
            x = conv(x)
            x = norm(x)
            x = nn.relu(x)
            # Max pooling after each conv layer, but only if spatial dims are > 1
            # Skip pooling on last layer to avoid dimension collapse
            if i < len(self.conv_layers) - 1:
                h, w = x.shape[1], x.shape[2]
                if h > 1 and w > 1:
                    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
            # deterministic=False means apply dropout
            # deterministic=True means no dropout
            x = dropout(x, rng=rng_key, deterministic=not use_dropout)

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
    ) -> jnp.ndarray:
        """
        Forward pass through the CNN with Monte Carlo Dropout.

        Args:
            inputs: Input data of shape (batch_size, height, width, channels)
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
                        For images: (height, width, channels)

        Returns:
            Initialized parameters
        """
        # Create dummy input for initialization: (batch_size=1, height, width, channels)
        if len(input_shape) == 3:
            dummy_input = jnp.zeros((1, *input_shape), dtype=jnp.float32)
        elif len(input_shape) == 1:
            # Legacy support: if flattened, reshape to (1, 32, 32, 3) for CIFAR-10
            if input_shape[0] == 3072:
                dummy_input = jnp.zeros((1, 32, 32, 3), dtype=jnp.float32)
            else:
                raise ValueError(f"Unknown flattened input shape: {input_shape}")
        else:
            dummy_input = jnp.zeros((1, *input_shape), dtype=jnp.float32)

        rng1, rng2 = jax.random.split(rng)
        return self.init(rng1, dummy_input, rng2, training=True)

    def get_loss(
        self,
        params: dict[str, Any],
        inputs: jnp.ndarray,
        labels: jnp.ndarray,
        rng: Any,
        n_vi_samples: int = 1,  # Unused, kept for API consistency
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """
        Compute cross-entropy loss for DropoutCNN.

        Args:
            params: Model parameters
            inputs: Input data of shape (batch_size, height, width, channels)
            labels: One-hot encoded labels of shape (batch_size, num_classes)
            rng: Random number generator
            n_vi_samples: Unused, kept for API consistency

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
