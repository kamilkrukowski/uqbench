"""Tests for data loading and preprocessing."""

import jax.numpy as jnp
import pytest

from bayescal.data import loaders, preprocessing


def test_normalize_images() -> None:
    """Test image normalization."""
    images = jnp.array([[[[255.0]]]], dtype=jnp.float32)
    normalized = preprocessing.normalize_images(images)
    assert jnp.allclose(normalized, 1.0)


def test_flatten_images() -> None:
    """Test image flattening."""
    images = jnp.ones((2, 28, 28, 1))
    flattened = preprocessing.flatten_images(images)
    assert flattened.shape == (2, 784)


def test_one_hot_encode() -> None:
    """Test one-hot encoding."""
    labels = jnp.array([0, 1, 2])
    encoded = preprocessing.one_hot_encode(labels, num_classes=3)
    assert encoded.shape == (3, 3)
    assert jnp.allclose(encoded[0], jnp.array([1.0, 0.0, 0.0]))

