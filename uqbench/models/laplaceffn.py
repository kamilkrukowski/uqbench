"""Laplace Approximation for Feedforward Neural Networks.

Implements post-hoc Laplace approximation following:
- "A Practical Framework for Laplace Approximation" (Daxberger et al., 2021)
- "Laplace Redux" paper

The Laplace approximation fits a Gaussian posterior centered at the MAP estimate
with covariance given by the inverse Hessian of the negative log-posterior.
"""

from typing import Any

import jax
import jax.numpy as jnp

from uqbench.models.fnn import FNN


class LaplaceFNN:
    """
    Laplace Approximation for Feedforward Neural Networks.

    This is a post-hoc method that fits a Gaussian posterior to a pre-trained
    deterministic network. The posterior is centered at the MAP estimate (trained
    weights) with covariance given by the inverse of the Generalized Gauss-Newton
    (GGN) approximation to the Hessian.

    For classification, we use the last-layer Laplace approximation for efficiency,
    which only computes uncertainty over the last layer weights.

    Workflow:
    1. Train a base FNN to get MAP estimate θ*
    2. Compute GGN Hessian H ≈ J^T diag(p(1-p)) J for last layer
    3. Posterior ≈ N(θ*, (H + λI)^{-1})
    4. At inference: sample weights from posterior and average predictions
    """

    def __init__(
        self,
        base_model: FNN,
        map_params: dict[str, Any],
        posterior_mean: jnp.ndarray,
        posterior_cov: jnp.ndarray,
        prior_precision: float = 1.0,
    ):
        """
        Initialize LaplaceFNN with fitted posterior.

        Args:
            base_model: The trained base FNN model
            map_params: MAP parameters from trained FNN
            posterior_mean: Mean of the Gaussian posterior (last layer weights flattened)
            posterior_cov: Covariance matrix of the posterior
            prior_precision: Precision (1/variance) of the prior
        """
        self.base_model = base_model
        self.map_params = map_params
        self.posterior_mean = posterior_mean
        self.posterior_cov = posterior_cov
        self.prior_precision = prior_precision
        self.num_classes = base_model.num_classes

    @classmethod
    def fit(
        cls,
        base_model: FNN,
        params: dict[str, Any],
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        prior_precision: float = 1.0,
        subset_size: int | None = None,
    ) -> "LaplaceFNN":
        """
        Fit Laplace approximation to a trained FNN.

        Uses last-layer Laplace approximation for efficiency:
        - Only fits posterior over the output layer weights
        - Uses GGN approximation to the Hessian

        Args:
            base_model: Trained FNN model instance
            params: Trained model parameters (MAP estimate)
            X_train: Training features
            y_train: Training labels (one-hot encoded)
            prior_precision: Prior precision (regularization strength)
            subset_size: Optional subset of training data to use for Hessian

        Returns:
            Fitted LaplaceFNN instance
        """
        X_train = jnp.array(X_train)
        y_train = jnp.array(y_train)

        # Optionally subsample for efficiency
        if subset_size is not None and subset_size < len(X_train):
            indices = jnp.arange(len(X_train))[:subset_size]
            X_train = X_train[indices]
            y_train = y_train[indices]

        # Extract last layer weights (output layer)
        # Structure: params['params']['output_layer']['kernel'] and ['bias']
        last_layer_kernel = params["params"]["output_layer"]["kernel"]
        last_layer_bias = params["params"]["output_layer"]["bias"]

        # Flatten last layer weights
        kernel_flat = last_layer_kernel.flatten()
        bias_flat = last_layer_bias.flatten()
        posterior_mean = jnp.concatenate([kernel_flat, bias_flat])

        # Compute features (activations before last layer)
        features = cls._get_features(base_model, params, X_train)

        # Get predictions for GGN computation
        probs = base_model.apply(
            params, inputs=X_train, rng=jax.random.PRNGKey(0), training=False
        )

        # Compute GGN approximation to Hessian for last layer
        # H = J^T D J where D = diag(p_i * (1 - p_i)) for each class
        # For multi-class: use block structure

        n_samples = features.shape[0]
        n_features = features.shape[1]
        n_classes = probs.shape[1]
        n_params = n_features * n_classes + n_classes  # kernel + bias

        # Build Jacobian of logits w.r.t. last layer params
        # For linear layer: logits = features @ kernel + bias
        # Jacobian is structured as [features, 1] for each class

        # Compute GGN: sum over samples of J^T @ diag(p*(1-p)) @ J
        # This is equivalent to: sum_i (features_i^T features_i) * p_i * (1-p_i)

        # For softmax, the Hessian of cross-entropy w.r.t. logits is:
        # H_ij = p_i * (δ_ij - p_j) which is diag(p) - p @ p^T

        ggn = jnp.zeros((n_params, n_params))

        # Efficient computation using batched operations
        # For each sample, compute contribution to GGN
        for i in range(n_samples):
            feat = features[i]  # (n_features,)
            p = probs[i]  # (n_classes,)

            # Hessian of softmax cross-entropy: diag(p) - outer(p, p)
            H_softmax = jnp.diag(p) - jnp.outer(p, p)

            # Jacobian: for kernel params, it's outer(feat, e_c) for class c
            # For bias, it's e_c
            # Full Jacobian is block diagonal with blocks [feat, 1] for each class

            # Build full Jacobian for this sample
            # Shape: (n_classes, n_params)
            J = jnp.zeros((n_classes, n_params))
            for c in range(n_classes):
                # Kernel params for class c: indices c*n_features to (c+1)*n_features
                start_k = c * n_features
                end_k = (c + 1) * n_features
                J = J.at[c, start_k:end_k].set(feat)
                # Bias for class c: index n_features*n_classes + c
                J = J.at[c, n_features * n_classes + c].set(1.0)

            # GGN contribution: J^T @ H_softmax @ J
            ggn += J.T @ H_softmax @ J

        # Add prior precision (L2 regularization)
        ggn += prior_precision * jnp.eye(n_params)

        # Posterior covariance is inverse of GGN
        # Add small jitter for numerical stability
        ggn += 1e-6 * jnp.eye(n_params)
        posterior_cov = jnp.linalg.inv(ggn)

        return cls(
            base_model=base_model,
            map_params=params,
            posterior_mean=posterior_mean,
            posterior_cov=posterior_cov,
            prior_precision=prior_precision,
        )

    @staticmethod
    def _get_features(
        model: FNN, params: dict[str, Any], inputs: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Get activations before the output layer (features for last-layer Laplace).

        Args:
            model: FNN model
            params: Model parameters
            inputs: Input data

        Returns:
            Features of shape (n_samples, n_features)
        """
        # Forward pass through hidden layers only
        # Flax names list-based layers as dense_layers_0, dense_layers_1, etc.
        x = inputs
        for i in range(len(model.hidden_dims)):
            layer_key = f"dense_layers_{i}"
            kernel = params["params"][layer_key]["kernel"]
            bias = params["params"][layer_key]["bias"]
            x = x @ kernel + bias
            x = jax.nn.relu(x)
        return x

    def __call__(
        self,
        inputs: jnp.ndarray,
        rng: Any,
        training: bool = False,
        n_samples: int = 100,
    ) -> jnp.ndarray:
        """
        Forward pass with Monte Carlo sampling from Laplace posterior.

        Args:
            inputs: Input data of shape (batch_size, input_dim)
            rng: Random number generator
            training: Ignored (for API compatibility)
            n_samples: Number of MC samples from posterior

        Returns:
            Mean predicted probabilities of shape (batch_size, num_classes)
        """
        inputs = jnp.array(inputs)

        # Get features (activations before last layer)
        features = self._get_features(self.base_model, self.map_params, inputs)

        n_features = features.shape[1]
        n_classes = self.num_classes

        # Sample from posterior
        samples = jax.random.multivariate_normal(
            rng, self.posterior_mean, self.posterior_cov, shape=(n_samples,)
        )

        # Compute predictions for each sample
        all_probs = []
        for sample in samples:
            # Reshape sample back to kernel and bias
            kernel = sample[: n_features * n_classes].reshape(n_features, n_classes)
            bias = sample[n_features * n_classes :]

            # Forward through last layer
            logits = features @ kernel + bias
            probs = jax.nn.softmax(logits, axis=-1)
            all_probs.append(probs)

        # Average predictions
        all_probs = jnp.stack(all_probs, axis=0)
        mean_probs = jnp.mean(all_probs, axis=0)

        return mean_probs

    def apply(
        self,
        params: dict[str, Any],  # Ignored, uses internal params
        inputs: jnp.ndarray,
        rng: Any,
        training: bool = False,
        n_samples: int = 100,
    ) -> jnp.ndarray:
        """
        Apply method for API compatibility with other models.

        Args:
            params: Ignored (uses internal MAP params)
            inputs: Input data
            rng: Random number generator
            training: Ignored
            n_samples: Number of MC samples

        Returns:
            Predicted probabilities
        """
        return self(inputs, rng, training=training, n_samples=n_samples)
