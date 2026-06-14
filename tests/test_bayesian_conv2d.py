"""Test that BayesianConv2D matches nn.Conv when using mean weights."""

import jax
import jax.numpy as jnp
from flax import linen as nn

from uqbench.models.layers.bayesianconv2d import BayesianConv2D


def test_bayesian_conv2d_matches_regular_conv():
    """Test that BayesianConv2D with mean weights matches nn.Conv."""
    rng = jax.random.PRNGKey(42)
    batch_size = 2
    height, width, in_channels = 16, 16, 3
    out_channels = 8
    kernel_size = (5, 5)
    strides = (2, 2)
    
    # Create test input
    inputs = jax.random.normal(rng, (batch_size, height, width, in_channels))
    
    # Initialize regular Conv2D
    regular_conv = nn.Conv(
        features=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding="SAME",
    )
    
    # Initialize BayesianConv2D
    bayesian_conv = BayesianConv2D(
        features=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding="SAME",
        prior_std=1.0,
        posterior_std_init=0.1,
    )
    
    # Initialize both
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    regular_params = regular_conv.init(rng1, inputs)
    bayesian_params = bayesian_conv.init(rng2, inputs, rng3, training=True)
    
    # Copy weights from regular Conv to Bayesian Conv mean
    # Regular Conv has: params['kernel'] and params['bias']
    # Bayesian Conv has: params['params']['mean'], params['params']['log_std'], params['params']['bias']
    regular_kernel = regular_params['params']['kernel']
    regular_bias = regular_params['params']['bias']
    
    # Set Bayesian mean to match regular kernel
    bayesian_params['params']['mean'] = regular_kernel
    bayesian_params['params']['bias'] = regular_bias
    # Set log_std to very negative value so std ≈ 0 (deterministic)
    bayesian_params['params']['log_std'] = jnp.full_like(regular_kernel, -20.0)
    
    # Forward pass: regular Conv
    regular_out = regular_conv.apply(regular_params, inputs)
    
    # Forward pass: Bayesian Conv with mean weights (training=False uses mean)
    rng_forward = jax.random.PRNGKey(999)
    bayesian_out = bayesian_conv.apply(bayesian_params, inputs, rng_forward, training=False)
    
    # They should match exactly (or very close due to numerical precision)
    max_diff = jnp.max(jnp.abs(regular_out - bayesian_out))
    mean_diff = jnp.mean(jnp.abs(regular_out - bayesian_out))
    
    print(f"Regular Conv output shape: {regular_out.shape}")
    print(f"Bayesian Conv output shape: {bayesian_out.shape}")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    
    # Check if they match
    assert regular_out.shape == bayesian_out.shape, f"Shape mismatch: {regular_out.shape} vs {bayesian_out.shape}"
    assert max_diff < 1e-5, f"Outputs don't match! Max diff: {max_diff:.2e}"
    print("✅ Test passed: BayesianConv2D matches nn.Conv when using mean weights!")
    
    return True


def test_bayesian_conv2d_with_stride():
    """Test that BayesianConv2D correctly handles different strides."""
    rng = jax.random.PRNGKey(42)
    batch_size = 1
    height, width, in_channels = 32, 32, 3
    out_channels = 16
    
    inputs = jax.random.normal(rng, (batch_size, height, width, in_channels))
    
    for stride in [1, 2, 3, 4]:
        bayesian_conv = BayesianConv2D(
            features=out_channels,
            kernel_size=(3, 3),
            strides=(stride, stride),
            padding="SAME",
        )
        
        rng1, rng2 = jax.random.split(rng)
        params = bayesian_conv.init(rng1, inputs, rng2, training=True)
        
        rng_forward = jax.random.PRNGKey(123)
        out = bayesian_conv.apply(params, inputs, rng_forward, training=False)
        
        # Expected output size with SAME padding and stride
        expected_h = (height + stride - 1) // stride
        expected_w = (width + stride - 1) // stride
        
        print(f"Stride {stride}: Input {inputs.shape[1:3]} -> Output {out.shape[1:3]} (expected ~{expected_h}x{expected_w})")
        assert out.shape[1] == expected_h, f"Height mismatch for stride {stride}"
        assert out.shape[2] == expected_w, f"Width mismatch for stride {stride}"
        assert out.shape[3] == out_channels, f"Channel mismatch for stride {stride}"
    
    print("✅ Test passed: BayesianConv2D handles strides correctly!")
    return True


if __name__ == "__main__":
    print("Testing BayesianConv2D implementation...")
    print("=" * 60)
    test_bayesian_conv2d_matches_regular_conv()
    print()
    test_bayesian_conv2d_with_stride()
    print()
    print("=" * 60)
    print("All tests passed! ✅")

