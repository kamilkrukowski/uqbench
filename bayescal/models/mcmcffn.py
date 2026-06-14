"""MCMC-based Bayesian Feedforward Neural Network using BlackJAX.

Implements Hamiltonian Monte Carlo (HMC) and No-U-Turn Sampler (NUTS)
for approximating the posterior distribution of neural network weights.

When temperature=1.0, approximates the true Bayesian posterior.
When temperature≠1.0, approximates a tempered posterior (cold when T<1, hot when T>1).

MCMC yields approximate posterior expectations with Monte Carlo error.
This provides a gold-standard approximation but is computationally expensive for large networks.
"""

from typing import Any, Literal

import blackjax
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import traverse_util


class MCMCFNN:
    """
    MCMC-based Bayesian Feedforward Neural Network.

    Uses Hamiltonian Monte Carlo (HMC) or NUTS to approximate the posterior
    distribution of network weights. When temperature=1.0, approximates the true
    Bayesian posterior. When temperature≠1.0, approximates a tempered posterior.
    
    MCMC yields approximate posterior expectations with Monte Carlo error.
    This is computationally expensive but provides a gold-standard approximation
    of uncertainty.

    Best suited for:
    - Small networks (few hundred parameters)
    - Small datasets
    - When you need high-quality uncertainty estimates

    Workflow:
    1. Define network architecture and prior
    2. Run MCMC sampling (HMC/NUTS) to approximate the (possibly tempered) posterior
    3. Store posterior samples
    4. At inference: average predictions over posterior samples
    """

    hidden_dims: tuple[int, ...]
    num_classes: int
    prior_std: float

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (32, 32),
        num_classes: int = 2,
        prior_std: float = 1.0,
        posterior_samples: jnp.ndarray | None = None,
        param_structure: dict[str, Any] | None = None,
    ):
        """
        Initialize MCMCFNN.

        Args:
            hidden_dims: Tuple of hidden layer dimensions
            num_classes: Number of output classes
            prior_std: Standard deviation of Gaussian prior on weights
            posterior_samples: Pre-computed posterior samples (flat array)
            param_structure: Structure template for unflattening parameters
        """
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.prior_std = prior_std
        self.posterior_samples = posterior_samples
        self.param_structure = param_structure

    def _create_network(self) -> nn.Module:
        """Create the underlying Flax network module."""

        class _Network(nn.Module):
            hidden_dims: tuple[int, ...]
            num_classes: int

            @nn.compact
            def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
                for dim in self.hidden_dims:
                    x = nn.Dense(dim)(x)
                    x = nn.relu(x)
                logits = nn.Dense(self.num_classes)(x)
                return nn.softmax(logits, axis=-1)

        return _Network(hidden_dims=self.hidden_dims, num_classes=self.num_classes)

    def _flatten_params(self, params: dict[str, Any]) -> jnp.ndarray:
        """Flatten nested parameter dict to 1D array."""
        flat = traverse_util.flatten_dict(params, sep="/")
        arrays = [v.flatten() for v in flat.values()]
        return jnp.concatenate(arrays)

    def _unflatten_params(
        self, flat_params: jnp.ndarray, template: dict[str, Any]
    ) -> dict[str, Any]:
        """Unflatten 1D array back to nested parameter dict."""
        flat_template = traverse_util.flatten_dict(template, sep="/")
        result = {}
        offset = 0
        for key, value in flat_template.items():
            size = value.size
            result[key] = flat_params[offset : offset + size].reshape(value.shape)
            offset += size
        return traverse_util.unflatten_dict(result, sep="/")

    def _log_prior(self, flat_params: jnp.ndarray) -> jnp.ndarray:
        """Compute log prior probability (Gaussian prior)."""
        return -0.5 * jnp.sum(flat_params**2) / (self.prior_std**2)

    def _log_likelihood(
        self,
        flat_params: jnp.ndarray,
        network: nn.Module,
        template: dict[str, Any],
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute log likelihood of data given parameters."""
        params = self._unflatten_params(flat_params, template)
        probs = network.apply({"params": params}, X)
        # Cross-entropy: sum of y * log(p)
        log_probs = jnp.log(probs + 1e-8)
        return jnp.sum(y * log_probs)

    def _log_posterior(
        self,
        flat_params: jnp.ndarray,
        network: nn.Module,
        template: dict[str, Any],
        X: jnp.ndarray,
        y: jnp.ndarray,
        temperature: float = 1.0,
    ) -> jnp.ndarray:
        """Compute log posterior (unnormalized) with optional tempering.

        The tempered posterior is:
            log p_T(θ|D) ∝ temperature * log p(θ) + log p(D|θ)

        With temperature < 1 (cold posterior), the prior has less influence,
        making the posterior more concentrated around high-likelihood modes.
        This matches the effect of beta < 1 in variational inference.

        Args:
            flat_params: Flattened parameter vector
            network: Flax network module
            template: Parameter structure template
            X: Input data
            y: Labels (one-hot)
            temperature: Prior temperature (default 1.0 approximates true posterior).
                        When temperature=1.0, approximates the true Bayesian posterior.
                        When temperature<1.0, approximates a cold posterior (tempered).
                        When temperature>1.0, approximates a hot posterior (tempered).
                        Use temperature < 1 for cold posterior (matches beta in VI).

        Returns:
            Log posterior (unnormalized)
        """
        log_prior = self._log_prior(flat_params)
        log_lik = self._log_likelihood(flat_params, network, template, X, y)
        return temperature * log_prior + log_lik

    @classmethod
    def fit(
        cls,
        hidden_dims: tuple[int, ...],
        num_classes: int,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        prior_std: float = 1.0,
        temperature: float = 1.0,
        num_warmup: int = 500,
        num_samples: int = 500,
        sampler: Literal["nuts", "hmc"] = "nuts",
        step_size: float = 0.001,
        seed: int = 42,
    ) -> "MCMCFNN":
        """
        Fit MCMC posterior to training data.

        Args:
            hidden_dims: Hidden layer dimensions
            num_classes: Number of output classes
            X_train: Training features
            y_train: Training labels (one-hot)
            prior_std: Prior standard deviation
            temperature: Prior temperature for cold/tempered posterior (default 1.0).
                        Use temperature < 1 for cold posterior that downweights the prior.
                        This matches the effect of beta < 1 in variational inference
                        (e.g., temperature=0.05 matches beta=0.05 in BayesianFNN).
            num_warmup: Number of warmup/burnin samples
            num_samples: Number of posterior samples to collect
            sampler: MCMC sampler to use ("nuts" or "hmc")
            step_size: Initial step size for HMC/NUTS
            seed: Random seed

        Returns:
            Fitted MCMCFNN with posterior samples
        """
        X_train = jnp.array(X_train)
        y_train = jnp.array(y_train)

        # Create instance for helper methods
        instance = cls(
            hidden_dims=hidden_dims, num_classes=num_classes, prior_std=prior_std
        )

        # Create network and initialize
        network = instance._create_network()
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        dummy_input = jnp.zeros((1, X_train.shape[1]))
        init_params = network.init(init_rng, dummy_input)["params"]

        # Flatten parameters
        flat_init = instance._flatten_params(init_params)
        n_params = len(flat_init)

        print(f"MCMCFNN: {n_params} parameters to sample")
        if temperature != 1.0:
            print(f"Using cold posterior with temperature={temperature}")

        # Define log probability function
        def log_prob_fn(flat_params: jnp.ndarray) -> jnp.ndarray:
            return instance._log_posterior(
                flat_params,
                network,
                init_params,
                X_train,
                y_train,
                temperature=temperature,
            )

        # Initialize sampler
        rng, warmup_rng = jax.random.split(rng)

        if sampler == "nuts":
            # Use window adaptation for NUTS
            warmup = blackjax.window_adaptation(blackjax.nuts, log_prob_fn)
            (state, parameters), _ = warmup.run(
                warmup_rng, flat_init, num_steps=num_warmup
            )
            kernel = blackjax.nuts(log_prob_fn, **parameters).step
        else:
            # HMC with fixed step size
            kernel = blackjax.hmc(
                log_prob_fn,
                step_size=step_size,
                inverse_mass_matrix=jnp.ones(n_params),
                num_integration_steps=10,
            ).step
            state = blackjax.hmc.init(flat_init, log_prob_fn)

        # Sampling loop
        print(f"Running {sampler.upper()} sampling...")
        samples = []
        rng, sample_rng = jax.random.split(rng)

        for i in range(num_samples):
            sample_rng, step_rng = jax.random.split(sample_rng)
            state, _ = kernel(step_rng, state)
            samples.append(state.position)

            if (i + 1) % 100 == 0:
                print(f"  Collected {i + 1}/{num_samples} samples")

        posterior_samples = jnp.stack(samples, axis=0)
        print(f"Sampling complete. Shape: {posterior_samples.shape}")

        return cls(
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            prior_std=prior_std,
            posterior_samples=posterior_samples,
            param_structure=init_params,
        )

    def __call__(
        self,
        inputs: jnp.ndarray,
        rng: Any = None,
        training: bool = False,
        n_samples: int | None = None,
    ) -> jnp.ndarray:
        """
        Forward pass averaging over posterior samples.

        Args:
            inputs: Input data of shape (batch_size, input_dim)
            rng: Random key (optional, for subsampling posterior)
            training: Ignored
            n_samples: Number of posterior samples to use (None = all)

        Returns:
            Mean predicted probabilities
        """
        if self.posterior_samples is None:
            raise ValueError("Model not fitted. Call fit() first.")

        inputs = jnp.array(inputs)
        network = self._create_network()

        # Optionally subsample posterior
        samples = self.posterior_samples
        if n_samples is not None and n_samples < len(samples):
            if rng is not None:
                indices = jax.random.choice(
                    rng, len(samples), shape=(n_samples,), replace=False
                )
                samples = samples[indices]
            else:
                samples = samples[:n_samples]

        # Average predictions over posterior samples
        all_probs = []
        for flat_params in samples:
            params = self._unflatten_params(flat_params, self.param_structure)
            probs = network.apply({"params": params}, inputs)
            all_probs.append(probs)

        all_probs = jnp.stack(all_probs, axis=0)
        mean_probs = jnp.mean(all_probs, axis=0)

        return mean_probs

    def apply(
        self,
        params: dict[str, Any],  # Ignored
        inputs: jnp.ndarray,
        rng: Any = None,
        training: bool = False,
        n_samples: int | None = None,
    ) -> jnp.ndarray:
        """Apply method for API compatibility."""
        return self(inputs, rng=rng, training=training, n_samples=n_samples)

    def init_params(self, rng: Any, input_shape: tuple[int, ...]) -> dict[str, Any]:
        """
        Initialize parameters (for API compatibility).

        Note: For MCMCFNN, parameters are sampled during fit(), not initialized.
        This returns a dummy structure.
        """
        network = self._create_network()
        if len(input_shape) == 1:
            dummy_input = jnp.zeros((1, input_shape[0]))
        else:
            dummy_input = jnp.zeros(
                (1, input_shape[0] * input_shape[1] * input_shape[2])
            )
        return network.init(rng, dummy_input)
