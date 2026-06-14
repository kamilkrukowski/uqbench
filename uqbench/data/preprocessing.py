"""Data preprocessing utilities."""

import jax.numpy as jnp


def normalize_images(images: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize images to [0, 1] range.

    Args:
        images: Image array with values in [0, 255]

    Returns:
        Normalized image array with values in [0, 1]
    """
    return images.astype(jnp.float32) / 255.0


def flatten_images(images: jnp.ndarray) -> jnp.ndarray:
    """
    Flatten image arrays from (batch, height, width, channels) to (batch, features).

    Args:
        images: Image array of shape (batch, height, width, channels)

    Returns:
        Flattened array of shape (batch, height * width * channels)
    """
    return images.reshape(images.shape[0], -1)


def one_hot_encode(labels: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    """
    One-hot encode labels.

    Args:
        labels: Label array of shape (batch,)
        num_classes: Number of classes

    Returns:
        One-hot encoded array of shape (batch, num_classes)
    """
    return jnp.eye(num_classes)[labels]
