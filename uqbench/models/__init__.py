"""Neural network models."""

from uqbench.models.bayescnn import BayesianCNN
from uqbench.models.bayesffn import BayesianFNN
from uqbench.models.cnn import CNN, DropoutCNN
from uqbench.models.fnn import FNN, DropoutFNN
from uqbench.models.laplaceffn import LaplaceFNN
from uqbench.models.mcmcffn import MCMCFNN
from uqbench.models.method_registry import METHODS
from uqbench.models.tempscaledffn import TemperatureScaledFNN

__all__ = [
    "BayesianCNN",
    "BayesianFNN",
    "CNN",
    "DropoutCNN",
    "FNN",
    "DropoutFNN",
    "LaplaceFNN",
    "MCMCFNN",
    "METHODS",
    "TemperatureScaledFNN",
]
