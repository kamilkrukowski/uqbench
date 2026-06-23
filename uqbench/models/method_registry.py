"""Method registry implementing the UncertaintyMethod interface for each model type."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split

from uqbench.models.bayesffn import BayesianFNN
from uqbench.models.fnn import DropoutFNN, FNN
from uqbench.models.laplaceffn import LaplaceFNN
from uqbench.models.mcmcffn import MCMCFNN
from uqbench.models.tempscaledffn import TemperatureScaledFNN
from uqbench.training import optimizers, trainer


def _shared_train_loop(
    model: Any,
    model_name: str,
    X_train: np.ndarray,
    y_train_onehot: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 0.01,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
    warm_up_epochs: int = 0,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """Shared training loop for trainable models."""
    rng = jax.random.PRNGKey(seed)

    # Initialize model
    input_shape = (X_train.shape[1],)
    rng, init_rng = jax.random.split(rng)
    params = model.init_params(init_rng, input_shape=input_shape)

    # Create optimizer with gradient clipping
    optimizer = optimizers.get_optimizer(
        learning_rate=lr, optimizer_type="adam", max_grad_norm=max_grad_norm
    )
    opt_state = optimizer.init(params)

    # Create batches
    n_batches = (len(X_train) + batch_size - 1) // batch_size
    batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_train))
        batch_X = jnp.array(X_train[start_idx:end_idx])
        batch_y = jnp.array(y_train_onehot[start_idx:end_idx])
        batches.append((batch_X, batch_y))

    # Training loop
    history = {"loss": [], "accuracy": []}
    rng, train_rng = jax.random.split(rng)

    # Determine if model is Bayesian
    is_bayesian = hasattr(model, "compute_kl_divergence") and hasattr(model, "beta")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_kl = 0.0
        epoch_likelihood = 0.0
        num_batches = 0

        for batch_X, batch_y in batches:
            rng, batch_rng = jax.random.split(rng)

            n_vi_samples = 1 if is_bayesian else 1

            if is_bayesian and warm_up_epochs > 0:
                base_beta = getattr(model, "beta", 1.0)
                if epoch < warm_up_epochs:
                    beta = (epoch / warm_up_epochs) * base_beta
                else:
                    beta = base_beta
            else:
                beta = getattr(model, "beta", 1.0) if is_bayesian else 1.0

            n_train = len(X_train) if is_bayesian else None

            params, opt_state, metrics = trainer.train_step(
                model,
                params,
                opt_state,
                (batch_X, batch_y),
                batch_rng,
                optimizer,
                beta=beta,
                n_vi_samples=n_vi_samples,
                n_train=n_train,
            )

            epoch_loss += metrics["loss"]
            epoch_acc += metrics["accuracy"]
            if is_bayesian:
                epoch_kl += metrics.get("kl_loss", 0.0)
                epoch_likelihood += metrics.get("likelihood_loss", 0.0)
            num_batches += 1

        history["loss"].append(epoch_loss / num_batches)
        history["accuracy"].append(epoch_acc / num_batches)

        if verbose and (epoch + 1) % 50 == 0:
            if is_bayesian:
                avg_kl = epoch_kl / num_batches
                avg_likelihood = epoch_likelihood / num_batches
                base_beta = getattr(model, "beta", 1.0)
                if warm_up_epochs > 0 and epoch < warm_up_epochs:
                    current_beta = (epoch / warm_up_epochs) * base_beta
                    warmup_str = f" (warmup β={current_beta:.6f})"
                else:
                    current_beta = base_beta
                    warmup_str = ""
                n_train = len(X_train)
                avg_kl_normalized = avg_kl / n_train if n_train > 0 else avg_kl
                beta_kl_normalized = current_beta * avg_kl_normalized

                if hasattr(model, "get_weight_variance_stats"):
                    sigma_stats = model.get_weight_variance_stats(params)
                    sigma_str = f", σ_mean={sigma_stats['mean']:.4f}, σ_median={sigma_stats['median']:.4f}, σ_max={sigma_stats['max']:.4f}"
                else:
                    sigma_str = ""

                print(
                    f"{model_name} - Epoch {epoch+1}/{epochs}: "
                    f"Loss={history['loss'][-1]:.4f}, Acc={history['accuracy'][-1]:.4f}, "
                    f"Likelihood={avg_likelihood:.4f}, KL={avg_kl:.0f}, KL_norm={avg_kl_normalized:.4f}, β*KL_norm={beta_kl_normalized:.4f}{warmup_str}{sigma_str}"
                )
            else:
                print(
                    f"{model_name} - Epoch {epoch+1}/{epochs}: "
                    f"Loss={history['loss'][-1]:.4f}, Acc={history['accuracy'][-1]:.4f}"
                )

    return params, history


# FNN
def train_fnn(cfg: dict[str, Any], data: dict[str, Any], rng: jax.random.PRNGKey) -> dict[str, Any]:
    """Train deterministic FNN."""
    model = FNN(hidden_dims=cfg["hidden_dims"], num_classes=cfg["num_classes"], dropout_rate=0.0)
    params, history = _shared_train_loop(
        model, "FNN", data["X_train"], data["y_train_onehot"],
        epochs=cfg.get("epochs", 200), lr=cfg.get("lr", 0.01), seed=cfg.get("seed", 42)
    )
    return {"model": model, "params": params, "history": history}


def predict_proba_fnn(artifact: dict[str, Any], X: np.ndarray, cfg: dict[str, Any], rng: jax.random.PRNGKey) -> jnp.ndarray:
    """Predict with deterministic FNN (single forward pass)."""
    probs = artifact["model"].apply(
        artifact["params"], inputs=jnp.array(X), rng=rng, training=False, n_samples=1
    )
    return probs


def inference_cost_fnn(cfg: dict[str, Any]) -> dict[str, Any]:
    """Inference cost for FNN."""
    return {"forward_passes_per_example": 1}


# DropoutFNN
def train_dropout_fnn(cfg: dict[str, Any], data: dict[str, Any], rng: jax.random.PRNGKey) -> dict[str, Any]:
    """Train FNN with dropout."""
    model = DropoutFNN(
        hidden_dims=cfg["hidden_dims"],
        num_classes=cfg["num_classes"],
        dropout_rate=cfg.get("dropout_rate", 0.2)
    )
    params, history = _shared_train_loop(
        model, "DropoutFNN", data["X_train"], data["y_train_onehot"],
        epochs=cfg.get("epochs", 200), lr=cfg.get("lr", 0.01), seed=cfg.get("seed", 42)
    )
    return {"model": model, "params": params, "history": history}


def predict_proba_dropout_fnn(artifact: dict[str, Any], X: np.ndarray, cfg: dict[str, Any], rng: jax.random.PRNGKey) -> jnp.ndarray:
    """Predict with MC Dropout (N stochastic forward passes)."""
    n_samples = cfg.get("n_samples", 100)
    probs = artifact["model"].apply(
        artifact["params"], inputs=jnp.array(X), rng=rng, training=True, n_samples=n_samples
    )
    return probs


def inference_cost_dropout_fnn(cfg: dict[str, Any]) -> dict[str, Any]:
    """Inference cost for MC Dropout."""
    return {"forward_passes_per_example": cfg.get("n_samples", 100)}


# BayesianFNN
def train_bayesian_fnn(cfg: dict[str, Any], data: dict[str, Any], rng: jax.random.PRNGKey) -> dict[str, Any]:
    """Train Bayesian FNN with Bayes by Backprop."""
    model = BayesianFNN(
        hidden_dims=cfg["hidden_dims"],
        num_classes=cfg["num_classes"],
        beta=cfg.get("beta", 0.05),
        posterior_std_init=cfg.get("posterior_std_init", 0.1),
        max_std=cfg.get("max_std", 0.1),
    )
    params, history = _shared_train_loop(
        model, "BayesianFNN", data["X_train"], data["y_train_onehot"],
        epochs=cfg.get("epochs", 400), lr=cfg.get("lr", 0.0015),
        seed=cfg.get("seed", 42), warm_up_epochs=cfg.get("warm_up_epochs", 50)
    )
    return {"model": model, "params": params, "history": history}


def predict_proba_bayesian_fnn(artifact: dict[str, Any], X: np.ndarray, cfg: dict[str, Any], rng: jax.random.PRNGKey) -> jnp.ndarray:
    """Predict with Bayesian FNN (N weight samples or local reparam)."""
    n_samples = cfg.get("n_samples", 100)
    probs = artifact["model"].apply(
        artifact["params"], inputs=jnp.array(X), rng=rng, training=False, n_samples=n_samples
    )
    return probs


def inference_cost_bayesian_fnn(cfg: dict[str, Any]) -> dict[str, Any]:
    """Inference cost for Bayesian FNN."""
    return {"forward_passes_per_example": cfg.get("n_samples", 100)}


# LaplaceFNN
def train_laplace_fnn(cfg: dict[str, Any], data: dict[str, Any], rng: jax.random.PRNGKey) -> dict[str, Any]:
    """Fit Laplace approximation (post-hoc on trained FNN)."""
    from uqbench.models.fnn import FNN
    # First train base FNN
    base_model = FNN(hidden_dims=cfg["hidden_dims"], num_classes=cfg["num_classes"], dropout_rate=0.0)
    base_params, _ = _shared_train_loop(
        base_model, "FNN", data["X_train"], data["y_train_onehot"],
        epochs=cfg.get("epochs", 200), lr=cfg.get("lr", 0.01), seed=cfg.get("seed", 42), verbose=False
    )

    # Then fit Laplace
    model = LaplaceFNN.fit(
        base_model=base_model,
        params=base_params,
        X_train=data["X_train"],
        y_train=data["y_train_onehot"],
        prior_precision=cfg.get("prior_precision", 10.0),
        subset_size=cfg.get("subset_size", 1000),
    )
    return {"model": model, "params": {}, "history": {}}


def predict_proba_laplace_fnn(artifact: dict[str, Any], X: np.ndarray, cfg: dict[str, Any], rng: jax.random.PRNGKey) -> jnp.ndarray:
    """Predict with Laplace approximation (N samples from Gaussian approx)."""
    n_samples = cfg.get("n_samples", 100)
    probs = artifact["model"].apply(
        artifact["params"], inputs=jnp.array(X), rng=rng, training=False, n_samples=n_samples
    )
    return probs


def inference_cost_laplace_fnn(cfg: dict[str, Any]) -> dict[str, Any]:
    """Inference cost for Laplace approximation."""
    return {"forward_passes_per_example": cfg.get("n_samples", 100)}


# MCMCFNN
def train_mcmc_fnn(cfg: dict[str, Any], data: dict[str, Any], rng: jax.random.PRNGKey) -> dict[str, Any]:
    """Fit MCMC (NUTS sampling of tempered posterior)."""
    model = MCMCFNN.fit(
        hidden_dims=cfg["hidden_dims"],
        num_classes=cfg["num_classes"],
        X_train=data["X_train"],
        y_train=data["y_train_onehot"],
        prior_std=cfg.get("prior_std", 0.1),
        temperature=cfg.get("temperature", 0.05),
        num_warmup=cfg.get("num_warmup", 50),
        num_samples=cfg.get("num_samples", 50),
        sampler=cfg.get("sampler", "nuts"),
        seed=cfg.get("seed", 42),
    )
    return {"model": model, "params": {}, "history": {}}


def predict_proba_mcmc_fnn(artifact: dict[str, Any], X: np.ndarray, cfg: dict[str, Any], rng: jax.random.PRNGKey) -> jnp.ndarray:
    """Predict with MCMC (posterior samples, offline)."""
    probs = artifact["model"].apply(
        artifact["params"], inputs=jnp.array(X), rng=rng, training=False, n_samples=cfg.get("n_samples", 50)
    )
    return probs


def inference_cost_mcmc_fnn(cfg: dict[str, Any]) -> dict[str, Any]:
    """Inference cost for MCMC."""
    return {"forward_passes_per_example": cfg.get("n_samples", 50)}


# TemperatureScaledFNN (Post-hoc Temperature Scaling)
def train_tempscaled_fnn(cfg: dict[str, Any], data: dict[str, Any], rng: jax.random.PRNGKey) -> dict[str, Any]:
    """Fit temperature scaling (post-hoc on trained FNN)."""
    # First train base FNN
    base_model = FNN(hidden_dims=cfg["hidden_dims"], num_classes=cfg["num_classes"], dropout_rate=0.0)
    base_params, _ = _shared_train_loop(
        base_model, "FNN", data["X_train"], data["y_train_onehot"],
        epochs=cfg.get("epochs", 200), lr=cfg.get("lr", 0.01), seed=cfg.get("seed", 42), verbose=False
    )

    # Split training data for temperature fitting (use a portion as validation)
    # Use 20% of training data for temperature calibration
    X_train_base, X_val_temp, y_train_base, y_val_temp = train_test_split(
        data["X_train"], data["y_train_onehot"], test_size=0.2, random_state=cfg.get("seed", 42)
    )

    # Fit temperature scaling
    model = TemperatureScaledFNN.fit(
        base_model=base_model,
        params=base_params,
        X_val=jnp.array(X_val_temp),
        y_val=jnp.array(y_val_temp),
        lr=cfg.get("temp_lr", 0.01),
        max_iter=cfg.get("max_iter", 1000),
        seed=cfg.get("seed", 42),
    )
    return {"model": model, "params": {}, "history": {}}


def predict_proba_tempscaled_fnn(artifact: dict[str, Any], X: np.ndarray, cfg: dict[str, Any], rng: jax.random.PRNGKey) -> jnp.ndarray:
    """Predict with temperature scaling (single forward pass with calibrated temperature)."""
    probs = artifact["model"].apply(
        artifact["params"], inputs=jnp.array(X), rng=rng, training=False, n_samples=1
    )
    return probs


def inference_cost_tempscaled_fnn(cfg: dict[str, Any]) -> dict[str, Any]:
    """Inference cost for temperature scaling."""
    return {"forward_passes_per_example": 1}


# DeepEnsemble
def train_deep_ensemble(cfg: dict[str, Any], data: dict[str, Any], rng: jax.random.PRNGKey) -> dict[str, Any]:
    """Train an ensemble of independently-seeded deterministic FNNs."""
    n_members = cfg.get("n_members", 5)
    base_seed = cfg.get("seed", 42)
    model = FNN(hidden_dims=cfg["hidden_dims"], num_classes=cfg["num_classes"], dropout_rate=0.0)
    members = []
    for i in range(n_members):
        params, _ = _shared_train_loop(
            model, f"DeepEnsemble[{i}]", data["X_train"], data["y_train_onehot"],
            epochs=cfg.get("epochs", 200), lr=cfg.get("lr", 0.01),
            seed=base_seed + 1000 * (i + 1), verbose=False,
        )
        members.append(params)
    return {"model": model, "members": members, "params": {}, "history": {}}


def predict_proba_deep_ensemble(artifact: dict[str, Any], X: np.ndarray, cfg: dict[str, Any], rng: jax.random.PRNGKey) -> jnp.ndarray:
    """Predict by averaging member softmax outputs (one forward pass per member)."""
    member_probs = [
        artifact["model"].apply(params, inputs=jnp.array(X), rng=rng, training=False, n_samples=1)
        for params in artifact["members"]
    ]
    return jnp.mean(jnp.stack(member_probs), axis=0)


def inference_cost_deep_ensemble(cfg: dict[str, Any]) -> dict[str, Any]:
    """Inference cost for deep ensemble."""
    return {"forward_passes_per_example": cfg.get("n_members", 5)}


# Method registry
METHODS = {
    "FNN": {
        "train": train_fnn,
        "predict_proba": predict_proba_fnn,
        "inference_cost": inference_cost_fnn,
        "color": "#F18F01",
        "marker": "o",
    },
    "DropoutFNN": {
        "train": train_dropout_fnn,
        "predict_proba": predict_proba_dropout_fnn,
        "inference_cost": inference_cost_dropout_fnn,
        "color": "#A23B72",
        "marker": "s",
    },
    "BayesianFNN": {
        "train": train_bayesian_fnn,
        "predict_proba": predict_proba_bayesian_fnn,
        "inference_cost": inference_cost_bayesian_fnn,
        "color": "#2E86AB",
        "marker": "^",
    },
    "LaplaceFNN": {
        "train": train_laplace_fnn,
        "predict_proba": predict_proba_laplace_fnn,
        "inference_cost": inference_cost_laplace_fnn,
        "color": "#3D5A80",
        "marker": "D",
    },
    "MCMCFNN": {
        "train": train_mcmc_fnn,
        "predict_proba": predict_proba_mcmc_fnn,
        "inference_cost": inference_cost_mcmc_fnn,
        "color": "#6B9080",
        "marker": "v",
    },
    "TemperatureScaledFNN": {
        "train": train_tempscaled_fnn,
        "predict_proba": predict_proba_tempscaled_fnn,
        "inference_cost": inference_cost_tempscaled_fnn,
        "color": "#F77F00",
        "marker": "p",
    },
    "DeepEnsemble": {
        "train": train_deep_ensemble,
        "predict_proba": predict_proba_deep_ensemble,
        "inference_cost": inference_cost_deep_ensemble,
        "color": "#E63946",
        "marker": "*",
    },
}
