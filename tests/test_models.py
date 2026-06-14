"""Tests for model implementations."""

import jax
import jax.numpy as jnp
import pytest

from bayescal.models import BayesianCNN, CNN, DropoutCNN


def test_bayesian_cnn_initialization() -> None:
    """Test Bayesian CNN initialization."""
    rng = jax.random.PRNGKey(42)
    model = BayesianCNN(
        conv_layers_config=((5, 5, 32, 1), (5, 5, 64, 1)),
        num_classes=10,
    )
    params = model.init_params(rng, input_shape=(32, 32, 3))
    assert params is not None


def test_cnn_initialization() -> None:
    """Test CNN initialization."""
    rng = jax.random.PRNGKey(42)
    model = CNN(
        conv_layers_config=((5, 5, 32, 1), (5, 5, 64, 1)),
        num_classes=10,
    )
    params = model.init_params(rng, input_shape=(32, 32, 3))
    assert params is not None


def test_dropout_cnn_initialization() -> None:
    """Test DropoutCNN initialization."""
    rng = jax.random.PRNGKey(42)
    model = DropoutCNN(
        conv_layers_config=((5, 5, 32, 1), (5, 5, 64, 1)),
        num_classes=10,
    )
    params = model.init_params(rng, input_shape=(32, 32, 3))
    assert params is not None


def test_bayesian_cnn_forward() -> None:
    """Test Bayesian CNN forward pass."""
    rng = jax.random.PRNGKey(42)
    model = BayesianCNN(
        conv_layers_config=((5, 5, 32, 1), (5, 5, 64, 1)),
        num_classes=10,
    )
    params = model.init_params(rng, input_shape=(32, 32, 3))
    inputs = jnp.ones((32, 32, 32, 3))
    rng, forward_rng = jax.random.split(rng)
    # BayesianCNN now requires rng parameter for BayesianConv2D layers
    # Test single sample (training mode)
    outputs = model.apply(params, inputs=inputs, rng=forward_rng, training=True, n_samples=1)
    assert outputs.shape == (32, 10)
    
    # Test multiple samples (inference mode)
    rng, forward_rng = jax.random.split(rng)
    outputs_mc = model.apply(params, inputs=inputs, rng=forward_rng, training=False, n_samples=10)
    assert outputs_mc.shape == (32, 10)


def test_cnn_forward() -> None:
    """Test CNN forward pass."""
    rng = jax.random.PRNGKey(42)
    model = CNN(
        conv_layers_config=((5, 5, 32, 1), (5, 5, 64, 1)),
        num_classes=10,
    )
    params = model.init_params(rng, input_shape=(32, 32, 3))
    inputs = jnp.ones((32, 32, 32, 3))
    rng, forward_rng = jax.random.split(rng)
    outputs = model.apply(params, inputs=inputs, rng=forward_rng, training=True)
    assert outputs.shape == (32, 10)


def test_dropout_cnn_forward() -> None:
    """Test DropoutCNN forward pass."""
    rng = jax.random.PRNGKey(42)
    model = DropoutCNN(
        conv_layers_config=((5, 5, 32, 1), (5, 5, 64, 1)),
        num_classes=10,
    )
    params = model.init_params(rng, input_shape=(32, 32, 3))
    inputs = jnp.ones((32, 32, 32, 3))
    rng, forward_rng = jax.random.split(rng)
    # Test single sample
    outputs = model.apply(params, inputs=inputs, rng=forward_rng, training=True, n_samples=1)
    assert outputs.shape == (32, 10)
    # Test multiple samples (MC Dropout)
    rng, forward_rng = jax.random.split(rng)
    outputs_mc = model.apply(params, inputs=inputs, rng=forward_rng, training=False, n_samples=10)
    assert outputs_mc.shape == (32, 10)

