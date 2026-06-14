"""Model export utilities for saving Flax models using Orbax checkpointing."""

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
)
from orbax.checkpoint import args as ocp_args


def export_model(
    model: Any,
    params: dict[str, Any],
    input_shape: tuple[int, ...],
    output_path: Path | str,
    model_name: str = "model",
    seed: int = 42,
) -> None:
    """
    Export a Flax/JAX model using Orbax checkpointing.

    Args:
        model: Flax model instance
        params: Model parameters dictionary
        input_shape: Input shape without batch dimension (e.g., (2,) for 2D input)
        output_path: Path where the model will be saved
        model_name: Name of the model (used for directory naming)
        seed: Random seed for inference (used to test model)
    """
    output_path = Path(output_path)

    # Create model-specific directory structure: {model_name}/
    if output_path.suffix == "":
        # output_path is already a directory
        model_dir = output_path / model_name.lower()
    else:
        # output_path is a file, create directory based on model name in same location
        model_dir = output_path.parent / model_name.lower()

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (model_dir / "checkpoints").resolve()
    metadata_file = model_dir / "model.json"

    # Ensure params are in the correct format for apply()
    if "params" not in params:
        variables = {"params": params}
    else:
        variables = params

    # Test the model to get output shape
    rng_key = jax.random.PRNGKey(seed)
    batch_size = 1
    if len(input_shape) == 1:
        dummy_input = jnp.zeros((batch_size, input_shape[0]), dtype=jnp.float32)
    else:
        raise ValueError(f"Unsupported input shape: {input_shape}")

    # Test that the model works
    test_output = model.apply(
        variables,
        dummy_input,
        rng=rng_key,
        training=False,
        n_samples=1,
        method=model.__call__,
    )
    output_array = np.array(test_output)
    output_shape = output_array.shape[1:]  # Remove batch dimension

    # Save model parameters using Orbax checkpointing
    options = CheckpointManagerOptions(create=True)
    with CheckpointManager(checkpoint_dir, options=options) as checkpoint_manager:
        # Save checkpoint with step 0 using StandardSave
        checkpoint_manager.save(
            0,
            args=ocp_args.StandardSave({"params": params}),
        )

    # Save model metadata
    metadata = {
        "model_name": model_name,
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "framework": "jax/flax",
        "checkpoint_format": "orbax",
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


def load_model(
    checkpoint_dir: Path | str,
    metadata_path: Path | str | None = None,
    step: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load a Flax model from Orbax checkpoint.

    Args:
        checkpoint_dir: Path to the checkpoint directory
        metadata_path: Optional path to metadata file. If None, looks for model.json
                      in the parent directory of checkpoint_dir
        step: Optional checkpoint step to load. If None, loads the latest checkpoint.

    Returns:
        Tuple of (parameters dictionary, metadata dictionary)
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()

    # Load checkpoint using Orbax
    options = CheckpointManagerOptions(create=False)
    with CheckpointManager(checkpoint_dir, options=options) as checkpoint_manager:
        # Load the latest checkpoint if step not specified
        if step is None:
            step = checkpoint_manager.latest_step()
            if step is None:
                raise ValueError(f"No checkpoint found in {checkpoint_dir}")

        # Restore checkpoint - CheckpointManager knows the object is saved using
        # standard pytree logic, so we can restore directly
        restored = checkpoint_manager.restore(step)
        params = restored["params"]

    print(f"Loaded checkpoint from step {step} in {checkpoint_dir}")

    # Load metadata
    if metadata_path is None:
        # Look for model.json in parent directory
        metadata_path = checkpoint_dir.parent / "model.json"
    else:
        metadata_path = Path(metadata_path)

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return params, metadata


def export_laplace(
    laplace_model: Any,
    output_path: Path | str,
    model_name: str = "LaplaceFNN",
) -> None:
    """
    Export a fitted LaplaceFNN model.

    Saves the posterior mean, covariance, MAP parameters, and model config.

    Args:
        laplace_model: Fitted LaplaceFNN instance
        output_path: Directory where the model will be saved
        model_name: Name for the model directory
    """
    output_path = Path(output_path)
    model_dir = output_path / model_name.lower()
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save posterior arrays
    np.savez(
        model_dir / "posterior.npz",
        posterior_mean=np.array(laplace_model.posterior_mean),
        posterior_cov=np.array(laplace_model.posterior_cov),
    )

    # Save MAP parameters as flattened arrays with structure info
    flat_params = traverse_util.flatten_dict(laplace_model.map_params, sep="/")
    arrays_to_save = {k.replace("/", "__"): np.array(v) for k, v in flat_params.items()}
    np.savez(model_dir / "map_params.npz", **arrays_to_save)

    # Save metadata
    metadata = {
        "model_name": model_name,
        "model_type": "LaplaceFNN",
        "prior_precision": float(laplace_model.prior_precision),
        "num_classes": laplace_model.num_classes,
        "hidden_dims": list(laplace_model.base_model.hidden_dims),
        "param_keys": list(flat_params.keys()),
    }
    with open(model_dir / "model.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported LaplaceFNN to {model_dir}")


def load_laplace(
    model_dir: Path | str,
) -> tuple[Any, dict[str, Any]]:
    """
    Load a saved LaplaceFNN model.

    Args:
        model_dir: Directory containing the saved model

    Returns:
        Tuple of (LaplaceFNN instance, metadata dict)
    """
    from uqbench.models import FNN, LaplaceFNN

    model_dir = Path(model_dir)

    # Load metadata
    with open(model_dir / "model.json") as f:
        metadata = json.load(f)

    # Load posterior
    posterior_data = np.load(model_dir / "posterior.npz")
    posterior_mean = jnp.array(posterior_data["posterior_mean"])
    posterior_cov = jnp.array(posterior_data["posterior_cov"])

    # Load and reconstruct MAP parameters
    map_data = np.load(model_dir / "map_params.npz")
    flat_params = {
        k.replace("__", "/"): jnp.array(map_data[k])
        for k in map_data.files
    }
    map_params = traverse_util.unflatten_dict(flat_params, sep="/")

    # Reconstruct base model
    base_model = FNN(
        hidden_dims=tuple(metadata["hidden_dims"]),
        num_classes=metadata["num_classes"],
        dropout_rate=0.0,
    )

    # Create LaplaceFNN instance
    laplace_model = LaplaceFNN(
        base_model=base_model,
        map_params=map_params,
        posterior_mean=posterior_mean,
        posterior_cov=posterior_cov,
        prior_precision=metadata["prior_precision"],
    )

    print(f"Loaded LaplaceFNN from {model_dir}")
    return laplace_model, metadata


def export_mcmc(
    mcmc_model: Any,
    output_path: Path | str,
    model_name: str = "MCMCFNN",
) -> None:
    """
    Export a fitted MCMCFNN model.

    Saves the posterior samples and parameter structure.

    Args:
        mcmc_model: Fitted MCMCFNN instance
        output_path: Directory where the model will be saved
        model_name: Name for the model directory
    """
    output_path = Path(output_path)
    model_dir = output_path / model_name.lower()
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save posterior samples
    np.savez(
        model_dir / "posterior_samples.npz",
        samples=np.array(mcmc_model.posterior_samples),
    )

    # Save parameter structure template
    flat_structure = traverse_util.flatten_dict(mcmc_model.param_structure, sep="/")
    structure_info = {
        k.replace("/", "__"): {"shape": list(v.shape), "dtype": str(v.dtype)}
        for k, v in flat_structure.items()
    }

    # Save metadata
    metadata = {
        "model_name": model_name,
        "model_type": "MCMCFNN",
        "hidden_dims": list(mcmc_model.hidden_dims),
        "num_classes": mcmc_model.num_classes,
        "prior_std": float(mcmc_model.prior_std),
        "num_samples": len(mcmc_model.posterior_samples),
        "param_structure": structure_info,
    }
    with open(model_dir / "model.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported MCMCFNN to {model_dir}")


def load_mcmc(
    model_dir: Path | str,
) -> tuple[Any, dict[str, Any]]:
    """
    Load a saved MCMCFNN model.

    Args:
        model_dir: Directory containing the saved model

    Returns:
        Tuple of (MCMCFNN instance, metadata dict)
    """
    from uqbench.models import MCMCFNN

    model_dir = Path(model_dir)

    # Load metadata
    with open(model_dir / "model.json") as f:
        metadata = json.load(f)

    # Load posterior samples
    samples_data = np.load(model_dir / "posterior_samples.npz")
    posterior_samples = jnp.array(samples_data["samples"])

    # Reconstruct parameter structure template
    structure_info = metadata["param_structure"]
    flat_structure = {}
    for k, info in structure_info.items():
        key = k.replace("__", "/")
        flat_structure[key] = jnp.zeros(info["shape"], dtype=jnp.float32)
    param_structure = traverse_util.unflatten_dict(flat_structure, sep="/")

    # Create MCMCFNN instance
    mcmc_model = MCMCFNN(
        hidden_dims=tuple(metadata["hidden_dims"]),
        num_classes=metadata["num_classes"],
        prior_std=metadata["prior_std"],
        posterior_samples=posterior_samples,
        param_structure=param_structure,
    )

    print(f"Loaded MCMCFNN from {model_dir}")
    return mcmc_model, metadata
