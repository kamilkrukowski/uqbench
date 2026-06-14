"""Configuration management for the uqbench project."""

from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Project paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    results_dir: Path = project_root / "experiments" / "results"
    configs_dir: Path = project_root / "experiments" / "configs"

    # Training settings
    seed: int = 42
    batch_size: int = 4096
    num_epochs: int = 2000
    learning_rate: float = 1.0e-2
    downsample_training_factor: float = (
        0.1  # Fraction of training data to use (1.0 = use all, 0.5 = use half)
    )

    # CNN settings
    # Each tuple is (kernel_x, kernel_y, num_filters, stride)
    conv_layers: tuple[tuple[int, int, int, int], ...] = (
        (4, 4, 16, 8),  # First conv: 6x6 kernel, 32 filters, stride 4
    )
    group_norm_groups: int = 1  # Number of groups for GroupNorm
    dropout_rate: float = 0.2  # Dropout rate for DropoutCNN models

    # Bayesian settings
    prior_std: float = 1.0
    posterior_std_init: float = 0.1
    n_vi_samples: int = 1  # Number of samples for variational inference during training
    beta: float = (
        0.00005  # Beta for beta-VI (KL penalty weight). Lower values allow more learning.
    )
    max_grad_norm: float | None = (
        1.0  # Maximum gradient norm for clipping. None to disable. Recommended: 1.0 for Bayesian models.
    )

    # Evaluation settings
    num_samples: int = 32  # For MC sampling in Bayesian models
    num_bins: int = 10  # For calibration metrics

    def model_post_init(self, __context: Any) -> None:
        """Create directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
