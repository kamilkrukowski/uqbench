"""Train all model types."""

import subprocess
import sys
from pathlib import Path

from uqbench.config import settings


def main() -> None:
    """Train all model types."""
    model_types = ["bayesian", "ffn", "dropout_ffn"]
    
    print("=" * 60)
    print("Training all model types")
    print("=" * 60)
    
    for model_type in model_types:
        print(f"\n{'=' * 60}")
        print(f"Training {model_type} model...")
        print(f"{'=' * 60}\n")
        
        cmd = [
            sys.executable,
            "scripts/train.py",
            "--model-type",
            model_type,
            "--epochs",
            str(settings.num_epochs),
            "--batch-size",
            str(settings.batch_size),
            "--learning-rate",
            str(settings.learning_rate),
        ]
        
        result = subprocess.run(cmd, check=False)
        
        if result.returncode != 0:
            print(f"\n❌ Failed to train {model_type} model")
            sys.exit(1)
        else:
            print(f"\n✅ Successfully trained {model_type} model")
    
    print("\n" + "=" * 60)
    print("All models trained successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

