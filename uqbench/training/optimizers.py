"""Optimizer configurations."""

import optax


def get_optimizer(
    learning_rate: float = 1e-3,
    optimizer_type: str = "adam",
    max_grad_norm: float | None = None,
) -> optax.GradientTransformation:
    """
    Get optimizer based on type.

    Args:
        learning_rate: Learning rate
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
        max_grad_norm: Maximum gradient norm for clipping. If None, no clipping is applied.
                      Recommended: 1.0 for stable training, especially for Bayesian models.

    Returns:
        Optimizer instance (with gradient clipping if max_grad_norm is provided)
    """
    if optimizer_type.lower() == "adam":
        base_optimizer = optax.adam(learning_rate)
    elif optimizer_type.lower() == "sgd":
        base_optimizer = optax.sgd(learning_rate, momentum=0.9)
    elif optimizer_type.lower() == "adamw":
        base_optimizer = optax.adamw(learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Add gradient clipping if specified
    if max_grad_norm is not None:
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            base_optimizer,
        )
    else:
        return base_optimizer
