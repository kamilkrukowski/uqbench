"""Custom JAX layers."""

from uqbench.models.layers.bayesianconv2d import BayesianConv2D
from uqbench.models.layers.bayesiandense import BayesianDense

__all__ = ["BayesianDense", "BayesianConv2D"]
