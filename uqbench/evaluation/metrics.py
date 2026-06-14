"""Additional evaluation metrics: AUROC and F1 score."""

import jax.numpy as jnp
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def macro_auroc(
    predictions: jnp.ndarray,
    labels: jnp.ndarray,
) -> float:
    """
    Calculate Macro AUROC for multiclass classification.

    Computes AUROC for each class (one-vs-rest) and averages across classes.

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels (one-hot encoded) of shape (n_samples, n_classes)

    Returns:
        Macro AUROC score (average across classes)
    """
    # Convert to numpy for sklearn
    pred_np = np.array(predictions)
    label_np = np.array(labels)

    # Get true class indices
    true_classes = np.argmax(label_np, axis=1)

    # Compute AUROC for each class (one-vs-rest)
    n_classes = pred_np.shape[1]
    auroc_scores = []

    for class_idx in range(n_classes):
        # Binary labels for this class (1 if this class, 0 otherwise)
        y_true_binary = (true_classes == class_idx).astype(int)

        # Skip if class has no positive or negative samples
        if len(np.unique(y_true_binary)) < 2:
            continue

        # Get probabilities for this class
        y_score = pred_np[:, class_idx]

        try:
            auroc = roc_auc_score(y_true_binary, y_score)
            auroc_scores.append(auroc)
        except ValueError:
            # Skip if AUROC cannot be computed (e.g., all predictions same)
            continue

    if len(auroc_scores) == 0:
        return 0.0

    return float(np.mean(auroc_scores))


def macro_f1(
    predictions: jnp.ndarray,
    labels: jnp.ndarray,
) -> float:
    """
    Calculate Macro F1 score for multiclass classification.

    Computes F1 score for each class and averages across classes.

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels (one-hot encoded) of shape (n_samples, n_classes)

    Returns:
        Macro F1 score (average across classes)
    """
    # Convert to numpy for sklearn
    pred_np = np.array(predictions)
    label_np = np.array(labels)

    # Get predicted and true class indices
    predicted_classes = np.argmax(pred_np, axis=1)
    true_classes = np.argmax(label_np, axis=1)

    # Compute macro F1 score
    f1 = f1_score(true_classes, predicted_classes, average="macro", zero_division=0.0)

    return float(f1)
