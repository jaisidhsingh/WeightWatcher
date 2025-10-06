#!/usr/bin/env python3
"""Test script for WeightWatcher with Flax.nnx models"""

import jax
import jax.numpy as jnp
from flax import nnx
import sys

# Import WeightWatcher from the local package
from weightwatcher import WeightWatcher


# Define a simple Flax.nnx model
class SimpleFlaxModel(nnx.Module):
    def __init__(self, rngs):
        self.conv1 = nnx.Conv(in_features=3, out_features=32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(in_features=32, out_features=64, kernel_size=(3, 3), rngs=rngs)
        self.linear1 = nnx.Linear(in_features=64 * 6 * 6, out_features=128, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=128, out_features=10, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nnx.relu(self.conv2(x))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def test_weightwatcher_flax():
    """Test WeightWatcher.analyze() on a Flax.nnx model"""

    print("Creating Flax.nnx model...")
    # Create model with random initialization
    rngs = nnx.Rngs(0)
    model = SimpleFlaxModel(rngs)

    print("Model created successfully!")
    print(f"Model type: {type(model)}")

    # Test forward pass
    print("\nTesting forward pass...")
    x = jnp.ones((1, 28, 28, 3))
    y = model(x)
    print(f"Output shape: {y.shape}")

    # Initialize WeightWatcher
    print("\nInitializing WeightWatcher...")
    watcher = WeightWatcher(model)

    # Run analysis
    print("\nRunning WeightWatcher analysis...")
    try:
        details = watcher.analyze()

        print("\n" + "="*50)
        print("SUCCESS! WeightWatcher analysis completed")
        print("="*50)

        # Print summary statistics
        print(f"\nNumber of layers analyzed: {len(details)}")
        print("\nLayer details:")
        print(details[['layer_id', 'name', 'layer_type', 'N', 'M', 'alpha']].to_string())

        # Check that we got reasonable results
        assert len(details) > 0, "No layers were analyzed"
        assert 'alpha' in details.columns, "Alpha column missing"
        assert details['alpha'].notna().any(), "No alpha values computed"

        print("\n" + "="*50)
        print("All checks passed!")
        print("="*50)

        return True

    except Exception as e:
        print("\n" + "="*50)
        print("FAILED! Error during analysis:")
        print("="*50)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_weightwatcher_flax()
    sys.exit(0 if success else 1)
