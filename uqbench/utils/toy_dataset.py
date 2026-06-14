"""Utilities for toy dataset generation, analysis and visualization."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# Default cache directory (relative to working directory)
_DEFAULT_CACHE_DIR = Path(".cache/datasets")


def _get_cache_key(
    n_samples: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    min_prob: float,
    max_prob: float,
    frequency: float,
    amplitude: float,
    seed: int,
) -> str:
    """Generate a unique cache key from dataset parameters."""
    params = {
        "n_samples": n_samples,
        "x_range": list(x_range),
        "y_range": list(y_range),
        "min_prob": min_prob,
        "max_prob": max_prob,
        "frequency": frequency,
        "amplitude": amplitude,
        "seed": seed,
    }
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()[:12]


def _make_prob_func(
    min_prob: float, max_prob: float, frequency: float, amplitude: float
) -> Callable[[float, float], float]:
    """Create the probability function (can't be cached, recreated on load)."""

    def prob_class1(x: float, y: float) -> float:
        """Compute P(class=1 | x, y) = σ(f(x, y)) bounded to [min_prob, max_prob]."""
        r = np.sqrt(x**2 + y**2)
        logit = amplitude * np.sin(frequency * r)
        prob_sigmoid = 1.0 / (1.0 + np.exp(-logit))
        prob = min_prob + (max_prob - min_prob) * prob_sigmoid
        return np.clip(prob, min_prob, max_prob)

    return prob_class1


def generate_concentric_rings_dataset(
    n_samples: int = 3000,
    x_range: tuple[float, float] = (-5.0, 5.0),
    y_range: tuple[float, float] = (-5.0, 5.0),
    min_prob: float = 0.05,
    max_prob: float = 0.95,
    frequency: float = 1.8,
    amplitude: float = 3.0,
    seed: int = 42,
    cache_dir: Path | str | None = None,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Callable[[float, float], float]]:
    """
    Generate a 2D dataset with concentric ring probability patterns.

    Uses a sinusoidal logit field based on radial distance to create
    clean, regular concentric rings with alternating class probabilities.

    The probability field is: p(x,y) = σ(amplitude * sin(frequency * r))
    where r = sqrt(x² + y²) and σ is the sigmoid function.

    **Caching**: Results are cached to disk based on parameters. Subsequent calls
    with the same parameters will load from cache instead of regenerating.

    Args:
        n_samples: Number of samples to generate
        x_range: (x_min, x_max) range for x coordinates
        y_range: (y_min, y_max) range for y coordinates
        min_prob: Minimum probability (clipped)
        max_prob: Maximum probability (clipped)
        frequency: Controls ring spacing (higher = thinner rings)
        amplitude: Controls sharpness of transitions (higher = sharper)
        seed: Random seed for reproducibility
        cache_dir: Directory for caching datasets. Defaults to .cache/datasets/
        use_cache: Whether to use caching (default True)

    Returns:
        X: Features of shape (n_samples, 2)
        y: Labels (0 or 1) of shape (n_samples,)
        y_onehot: One-hot encoded labels of shape (n_samples, 2)
        prob_func: Function that returns P(class=1 | x, y) for any point

    Example:
        >>> X, y, y_onehot, prob_func = generate_concentric_rings_dataset(n_samples=1000)
        >>> print(X.shape, y.shape)
        (1000, 2) (1000,)
    """
    # Set up cache directory
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)

    # Generate cache key from parameters
    cache_key = _get_cache_key(
        n_samples, x_range, y_range, min_prob, max_prob, frequency, amplitude, seed
    )
    cache_file = cache_dir / f"concentric_rings_{cache_key}.npz"

    # Create the probability function (always needed, can't be cached)
    prob_func = _make_prob_func(min_prob, max_prob, frequency, amplitude)

    # Try to load from cache
    if use_cache and cache_file.exists():
        data = np.load(cache_file)
        X = data["X"]
        y = data["y"]
        y_onehot = data["y_onehot"]
        print(f"Loaded dataset from cache: {cache_file}")
        return X, y, y_onehot, prob_func

    # Generate fresh dataset
    np.random.seed(seed)
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Generate points uniformly in the box
    margin_x = 0.05 * (x_max - x_min)
    margin_y = 0.05 * (y_max - y_min)

    X = []
    y_labels = []

    for _ in range(n_samples):
        x = np.random.uniform(x_min + margin_x, x_max - margin_x)
        y = np.random.uniform(y_min + margin_y, y_max - margin_y)
        X.append([x, y])

        # Sample label from Bernoulli distribution
        prob = prob_func(x, y)
        label = np.random.binomial(1, prob)
        y_labels.append(label)

    X = np.array(X)
    y = np.array(y_labels)

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # One-hot encode
    y_onehot = np.zeros((len(y), 2))
    y_onehot[np.arange(len(y)), y] = 1

    # Save to cache
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, X=X, y=y, y_onehot=y_onehot)
        print(f"Saved dataset to cache: {cache_file}")

    return X, y, y_onehot, prob_func


def compute_bayes_optimal_boundary(
    overlap: float = 0.8,
    x_range: tuple[float, float] = (-4, 4),
    y_range: tuple[float, float] = (-4, 4),
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Bayes optimal decision boundary for two overlapping Gaussians.

    For equal priors P(y=0) = P(y=1) = 0.5, the optimal boundary is where:
    log P(x|y=0) = log P(x|y=1)

    Which simplifies to:
    (x - μ₀)ᵀ Σ₀⁻¹ (x - μ₀) - log|Σ₀| = (x - μ₁)ᵀ Σ₁⁻¹ (x - μ₁) - log|Σ₁|

    Args:
        overlap: Overlap parameter used in dataset generation (must match generate_toy_dataset)
        x_range: (x_min, x_max) for grid
        y_range: (y_min, y_max) for grid
        n_points: Number of points in each dimension for the grid

    Returns:
        boundary_points: Array of shape (n_boundary_points, 2) with points on the boundary
        grid_x, grid_y: Meshgrid for plotting
        bayes_probs: Array of shape (n_points, n_points) with P(y=1|x) for each grid point
    """
    # Define the distributions (matching generate_toy_dataset)
    mean_0 = np.array([-1.0, -1.0])
    cov_0 = np.array([[1.0, 0.0], [0.0, 1.0]])

    mean_1 = np.array([1.0, 1.0])
    cov_1 = np.array([[1.0 + overlap, 0.0], [0.0, 1.0 + overlap]])

    # Create grid
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    # Compute log-likelihoods for each class
    log_likelihood_0 = multivariate_normal.logpdf(grid_points, mean=mean_0, cov=cov_0)
    log_likelihood_1 = multivariate_normal.logpdf(grid_points, mean=mean_1, cov=cov_1)

    # For equal priors, posterior is proportional to likelihood
    # P(y=1|x) = P(x|y=1) / (P(x|y=0) + P(x|y=1))
    # Using log-space for numerical stability
    log_likelihood_0_2d = log_likelihood_0.reshape(grid_x.shape)
    log_likelihood_1_2d = log_likelihood_1.reshape(grid_x.shape)

    # Compute posterior probabilities using log-sum-exp trick
    max_log = np.maximum(log_likelihood_0_2d, log_likelihood_1_2d)
    log_sum = max_log + np.log(
        np.exp(log_likelihood_0_2d - max_log) + np.exp(log_likelihood_1_2d - max_log)
    )
    log_posterior_1 = log_likelihood_1_2d - log_sum
    bayes_probs = np.exp(log_posterior_1)

    # Find boundary points (where P(y=1|x) = 0.5, or equivalently log_likelihood_0 = log_likelihood_1)
    log_diff = log_likelihood_1_2d - log_likelihood_0_2d

    # Use matplotlib's contour to find the boundary
    fig = plt.figure(figsize=(1, 1))
    cs = plt.contour(grid_x, grid_y, log_diff, levels=[0], colors="red")
    plt.close(fig)

    # Extract boundary points from contour
    boundary_points = []
    if len(cs.allsegs) > 0 and len(cs.allsegs[0]) > 0:
        for seg in cs.allsegs[0]:
            boundary_points.append(seg)
        if boundary_points:
            boundary_points = np.vstack(boundary_points)
        else:
            boundary_points = np.array([]).reshape(0, 2)
    else:
        boundary_points = np.array([]).reshape(0, 2)

    return boundary_points, grid_x, grid_y, bayes_probs
