"""Calibration metrics: ECE, Brier score, and calibration curves."""

import jax.numpy as jnp
import numpy as np


def expected_calibration_error(
    predictions: jnp.ndarray,
    labels: jnp.ndarray,
    num_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE) for multiclass classification.

    Uses the top-label (confidence-based) approach for multiclass:
    - Confidence = max predicted probability (top-1 prediction confidence)
    - Accuracy = whether top prediction matches true label
    - Bins predictions by confidence level

    This is the standard multiclass ECE definition from:
    "On Calibration of Modern Neural Networks" (Guo et al., 2017)

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels (one-hot encoded) of shape (n_samples, n_classes)
        num_bins: Number of bins for calibration

    Returns:
        ECE score: Weighted average of |confidence - accuracy| across bins
    """
    # For multiclass: use top-1 confidence and check if top prediction is correct
    confidences = jnp.max(predictions, axis=1)  # Top-1 confidence
    predicted_classes = jnp.argmax(predictions, axis=1)
    true_classes = jnp.argmax(labels, axis=1)
    accuracies = predicted_classes == true_classes  # Top-1 accuracy

    bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        # Only include bins with >= 1% samples
        if prop_in_bin >= 0.01:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += jnp.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def maximum_calibration_error(
    predictions: jnp.ndarray,
    labels: jnp.ndarray,
    num_bins: int = 10,
) -> float:
    """
    Calculate Maximum Calibration Error (MCE).

    Unlike ECE, MCE is not weighted by bin size, so it captures
    the worst-case calibration error across all bins.

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels (one-hot encoded) of shape (n_samples, n_classes)
        num_bins: Number of bins for calibration

    Returns:
        MCE score
    """
    confidences = jnp.max(predictions, axis=1)
    predicted_classes = jnp.argmax(predictions, axis=1)
    true_classes = jnp.argmax(labels, axis=1)
    accuracies = predicted_classes == true_classes

    bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    mce = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        # Only consider bins with >= 1% samples
        if prop_in_bin >= 0.01:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            bin_error = jnp.abs(avg_confidence_in_bin - accuracy_in_bin)
            mce = jnp.maximum(mce, bin_error)

    return float(mce)


def calibration_bin_statistics(
    predictions: jnp.ndarray,
    labels: jnp.ndarray,
    num_bins: int = 10,
) -> dict[str, np.ndarray]:
    """
    Get detailed statistics for each calibration bin.

    Useful for diagnosing why ECE might be low despite visible miscalibration.

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels (one-hot encoded) of shape (n_samples, n_classes)
        num_bins: Number of bins for calibration

    Returns:
        Dictionary with arrays for each bin:
        - 'bin_centers': Center of each bin
        - 'proportions': Proportion of samples in each bin
        - 'accuracies': Accuracy in each bin
        - 'confidences': Average confidence in each bin
        - 'errors': Calibration error (|confidence - accuracy|) in each bin
    """
    confidences = jnp.max(predictions, axis=1)
    predicted_classes = jnp.argmax(predictions, axis=1)
    true_classes = jnp.argmax(labels, axis=1)
    accuracies = predicted_classes == true_classes

    bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    proportions = []
    bin_accuracies = []
    bin_confidences = []
    bin_errors = []
    bin_centers_filtered = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        # Only include bins with >= 1% samples
        if prop_in_bin >= 0.01:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            bin_error = jnp.abs(avg_confidence_in_bin - accuracy_in_bin)

            bin_centers_filtered.append((bin_lower + bin_upper) / 2)
            proportions.append(float(prop_in_bin))
            bin_accuracies.append(float(accuracy_in_bin))
            bin_confidences.append(float(avg_confidence_in_bin))
            bin_errors.append(float(bin_error))

    return {
        "bin_centers": np.array(bin_centers_filtered),
        "proportions": np.array(proportions),
        "accuracies": np.array(bin_accuracies),
        "confidences": np.array(bin_confidences),
        "errors": np.array(bin_errors),
    }


def brier_score(
    predictions: jnp.ndarray,
    labels: jnp.ndarray,
) -> float:
    """
    Calculate Brier score.

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels (one-hot encoded) of shape (n_samples, n_classes)

    Returns:
        Brier score
    """
    return float(jnp.mean(jnp.sum((predictions - labels) ** 2, axis=1)))


def calibration_curve(
    predictions: jnp.ndarray,
    labels: jnp.ndarray,
    num_bins: int = 10,
    prune_small_bins: bool = True,
    strategy: str = "top_label",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate calibration curve.

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels (one-hot encoded) of shape (n_samples, n_classes)
        num_bins: Number of bins for calibration
        prune_small_bins: If True, only include bins with >= 1% samples. If False, include all bins.
        strategy: Calibration strategy:
            - "top_label": Use max probability as confidence (standard, but [0.5, 1.0] for binary)
            - "binary_class1": Use P(class=1) as confidence (full [0.0, 1.0] range for binary)

    Returns:
        Tuple of (fraction_of_positives, mean_predicted_value) arrays
    """
    if strategy == "binary_class1" and predictions.shape[1] == 2:
        # For binary classification: use P(class=1) directly
        # This gives the full [0.0, 1.0] range
        confidences = predictions[:, 1]
        true_classes = jnp.argmax(labels, axis=1)
        # "Accuracy" here means: is the true class = 1?
        accuracies = (true_classes == 1).astype(jnp.float32)
    else:
        # Standard top-label approach
        confidences = jnp.max(predictions, axis=1)
        predicted_classes = jnp.argmax(predictions, axis=1)
        true_classes = jnp.argmax(labels, axis=1)
        accuracies = (predicted_classes == true_classes).astype(jnp.float32)

    bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    fraction_of_positives = []
    mean_predicted_value = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()  # Use mean for proportion

        # Include bin based on pruning setting
        if not prune_small_bins or prop_in_bin >= 0.01:
            # For empty bins, use NaN or skip
            if prop_in_bin > 0:
                fraction_of_positives.append(float(accuracies[in_bin].mean()))
                mean_predicted_value.append(float(confidences[in_bin].mean()))
            elif not prune_small_bins:
                # Include empty bins as NaN for unpruned version
                fraction_of_positives.append(float("nan"))
                mean_predicted_value.append(float((bin_lower + bin_upper) / 2))

    return (
        np.array(fraction_of_positives),
        np.array(mean_predicted_value),
    )


def top_label_calibration_error(
    predictions: jnp.ndarray,
    labels: jnp.ndarray,
    num_bins: int = 10,
) -> float:
    """
    Calculate Top-label Calibration Error (TCE).

    TCE is similar to ECE but explicitly focuses on the top-label (most confident)
    prediction. For binary classification, this is equivalent to ECE.
    For multiclass, it uses the confidence of the top prediction.

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels (one-hot encoded) of shape (n_samples, n_classes)
        num_bins: Number of bins for calibration

    Returns:
        TCE score
    """
    # Use top-1 confidence and check if top prediction is correct
    confidences = jnp.max(predictions, axis=1)  # Top-1 confidence
    predicted_classes = jnp.argmax(predictions, axis=1)
    true_classes = jnp.argmax(labels, axis=1)
    accuracies = predicted_classes == true_classes  # Top-1 accuracy

    bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    tce = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        # Only include bins with >= 1% samples
        if prop_in_bin >= 0.01:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            tce += jnp.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(tce)


def adaptive_calibration_error(
    predictions: jnp.ndarray,
    labels: jnp.ndarray,
    num_bins: int = 10,
) -> float:
    """
    Calculate Adaptive Calibration Error (ACE).

    ACE uses adaptive binning where bins are chosen to have equal numbers
    of samples (quantile-based) rather than equal width intervals.
    This is more robust to imbalanced confidence distributions.

    Reference: "Verified Uncertainty Calibration" (Kumar et al., 2019)

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels (one-hot encoded) of shape (n_samples, n_classes)
        num_bins: Number of bins for calibration

    Returns:
        ACE score
    """
    # Use top-1 confidence and check if top prediction is correct
    confidences = jnp.max(predictions, axis=1)  # Top-1 confidence
    predicted_classes = jnp.argmax(predictions, axis=1)
    true_classes = jnp.argmax(labels, axis=1)
    accuracies = predicted_classes == true_classes  # Top-1 accuracy

    # Sort by confidence to create adaptive bins
    sorted_indices = jnp.argsort(confidences)
    sorted_confidences = confidences[sorted_indices]
    sorted_accuracies = accuracies[sorted_indices]

    # Create bins with equal number of samples
    n_samples = len(confidences)
    samples_per_bin = n_samples // num_bins

    ace = 0.0
    for i in range(num_bins):
        start_idx = i * samples_per_bin
        # Last bin gets remaining samples
        if i == num_bins - 1:
            end_idx = n_samples
        else:
            end_idx = (i + 1) * samples_per_bin

        bin_confidences = sorted_confidences[start_idx:end_idx]
        bin_accuracies = sorted_accuracies[start_idx:end_idx]

        if len(bin_confidences) > 0:
            accuracy_in_bin = bin_accuracies.mean()
            avg_confidence_in_bin = bin_confidences.mean()
            prop_in_bin = len(bin_confidences) / n_samples
            ace += jnp.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ace)


def adaptive_calibration_curve(
    predictions: jnp.ndarray,
    labels: jnp.ndarray,
    num_bins: int = 10,
    strategy: str = "top_label",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate calibration curve using adaptive (quantile-based) binning.

    Unlike the standard calibration_curve which uses fixed-width bins,
    this uses bins with equal numbers of samples, which is more robust
    to imbalanced confidence distributions.

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels (one-hot encoded) of shape (n_samples, n_classes)
        num_bins: Number of bins for calibration
        strategy: Calibration strategy:
            - "top_label": Use max probability as confidence (standard, but [0.5, 1.0] for binary)
            - "binary_class1": Use P(class=1) as confidence (full [0.0, 1.0] range for binary)

    Returns:
        Tuple of (fraction_of_positives, mean_predicted_value) arrays
    """
    if strategy == "binary_class1" and predictions.shape[1] == 2:
        # For binary classification: use P(class=1) directly
        confidences = predictions[:, 1]
        true_classes = jnp.argmax(labels, axis=1)
        accuracies = (true_classes == 1).astype(jnp.float32)
    else:
        # Standard top-label approach
        confidences = jnp.max(predictions, axis=1)
        predicted_classes = jnp.argmax(predictions, axis=1)
        true_classes = jnp.argmax(labels, axis=1)
        accuracies = (predicted_classes == true_classes).astype(jnp.float32)

    # Sort by confidence to create adaptive bins
    sorted_indices = jnp.argsort(confidences)
    sorted_confidences = confidences[sorted_indices]
    sorted_accuracies = accuracies[sorted_indices]

    # Create bins with equal number of samples
    n_samples = len(confidences)
    samples_per_bin = n_samples // num_bins

    fraction_of_positives = []
    mean_predicted_value = []

    for i in range(num_bins):
        start_idx = i * samples_per_bin
        # Last bin gets remaining samples
        if i == num_bins - 1:
            end_idx = n_samples
        else:
            end_idx = (i + 1) * samples_per_bin

        bin_confidences = sorted_confidences[start_idx:end_idx]
        bin_accuracies = sorted_accuracies[start_idx:end_idx]

        if len(bin_confidences) > 0:
            fraction_of_positives.append(float(bin_accuracies.mean()))
            mean_predicted_value.append(float(bin_confidences.mean()))

    return (
        np.array(fraction_of_positives),
        np.array(mean_predicted_value),
    )
