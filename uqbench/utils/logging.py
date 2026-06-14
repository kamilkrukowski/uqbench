"""Logging configuration."""

import logging
from pathlib import Path


def setup_logging(
    log_dir: Path | None = None,
    log_level: int = logging.INFO,
) -> None:
    """
    Set up logging configuration.

    Args:
        log_dir: Directory to save log files (optional)
        log_level: Logging level
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_dir / "training.log")] if log_dir else []),
        ],
    )
