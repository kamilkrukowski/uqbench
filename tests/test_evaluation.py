"""Tests for evaluation metrics."""

import jax.numpy as jnp
import pytest

from bayescal.evaluation import calibration


def test_expected_calibration_error() -> None:
    """Test ECE calculation."""
    # Perfect calibration
    predictions = jnp.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    labels = jnp.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    ece = calibration.expected_calibration_error(predictions, labels)
    assert ece >= 0.0


def test_brier_score() -> None:
    """Test Brier score calculation."""
    predictions = jnp.array([[0.9, 0.1], [0.8, 0.2]])
    labels = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    brier = calibration.brier_score(predictions, labels)
    assert brier >= 0.0

