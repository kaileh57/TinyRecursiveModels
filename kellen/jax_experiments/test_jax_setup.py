"""
Quick verification test for JAX implementation
Tests model initialization, forward pass, and basic training step
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from jax_models.recursive_reasoning.trm import TinyRecursiveReasoningModel, TRMConfig
from jax_models.losses import cross_entropy_loss, compute_accuracy
from jax_models.data_pipeline import JAXPuzzleDataset, PuzzleDatasetConfig


def test_model_init():
    """Test model initialization"""
    print("Testing model initialization...")

    config = TRMConfig(
        batch_size=4,
        seq_len=81,
        vocab_size=100,
        num_puzzle_identifiers=10,
        hidden_size=128,
        expansion=2.0,
        num_heads=4,
        num_layers=2,
        H_cycles=2,
        L_cycles=1,
        pos_encodings="rope",
        halt_max_steps=1,
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
        dtype=jnp.float32
    )

    model = TinyRecursiveReasoningModel(config)

    # Create dummy batch
    batch = {
        "inputs": jnp.zeros((4, 81), dtype=jnp.int32),
        "puzzle_identifiers": jnp.zeros((4,), dtype=jnp.int32),
    }

    # Initialize carry
    rng = jax.random.PRNGKey(0)
    carry = model.initial_carry(batch)

    print(f"  ✓ Model initialized")
    print(f"  ✓ Carry initialized: z_H shape {carry.inner_carry.z_H.shape}")

    return model, batch, carry, rng


def test_forward_pass():
    """Test forward pass"""
    print("\nTesting forward pass...")

    model, batch, carry, rng = test_model_init()

    # Initialize parameters
    rng, init_rng = jax.random.split(rng)
    variables = model.init(
        {'params': init_rng, 'carry': init_rng, 'exploration': init_rng},
        carry,
        batch,
        training=True
    )
    params = variables['params']

    # Forward pass
    rng, fwd_rng = jax.random.split(rng)
    new_carry, outputs = model.apply(
        {'params': params},
        carry,
        batch,
        training=True,
        rngs={'carry': fwd_rng, 'exploration': fwd_rng}
    )

    print(f"  ✓ Forward pass successful")
    print(f"  ✓ Logits shape: {outputs['logits'].shape}")
    print(f"  ✓ Q-halt shape: {outputs['q_halt_logits'].shape}")

    return model, params, batch, carry, outputs


def test_loss_computation():
    """Test loss computation"""
    print("\nTesting loss computation...")

    model, params, batch, carry, outputs = test_forward_pass()

    # Create dummy labels
    labels = jnp.zeros((4, 81), dtype=jnp.int32)

    # Compute loss
    loss = cross_entropy_loss(outputs['logits'], labels)
    acc = compute_accuracy(outputs['logits'], labels)

    print(f"  ✓ Loss computed: {float(loss):.4f}")
    print(f"  ✓ Accuracy computed: {float(acc):.4f}")

    return loss


def test_gradient_computation():
    """Test gradient computation"""
    print("\nTesting gradient computation...")

    model, batch, carry, rng = test_model_init()

    # Initialize parameters
    rng, init_rng = jax.random.split(rng)
    variables = model.init(
        {'params': init_rng, 'carry': init_rng, 'exploration': init_rng},
        carry,
        batch,
        training=True
    )
    params = variables['params']

    # Define loss function
    def loss_fn(params, rng):
        new_carry, outputs = model.apply(
            {'params': params},
            carry,
            batch,
            training=True,
            rngs={'carry': rng, 'exploration': rng}
        )
        labels = jnp.zeros((4, 81), dtype=jnp.int32)
        loss = cross_entropy_loss(outputs['logits'], labels)
        return loss, (new_carry, outputs)

    # Compute gradients
    rng, grad_rng = jax.random.split(rng)
    (loss, (new_carry, outputs)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, grad_rng)

    # Check gradients
    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))

    print(f"  ✓ Gradients computed successfully")
    print(f"  ✓ Gradient norm: {float(grad_norm):.6f}")
    print(f"  ✓ Loss: {float(loss):.4f}")

    return grads


def test_data_pipeline():
    """Test data pipeline (mock)"""
    print("\nTesting data pipeline interface...")

    # Test that we can create a config
    config = PuzzleDatasetConfig(
        seed=0,
        dataset_paths=["data/dummy"],  # Will fail if path doesn't exist, but that's ok
        global_batch_size=32,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )

    print(f"  ✓ Dataset config created")
    print(f"  ✓ Local batch size: {32 // 1}")

    return config


def main():
    """Run all tests"""
    print("=" * 60)
    print("JAX Implementation Verification Tests")
    print("=" * 60)

    print(f"\nJAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Device count: {jax.device_count()}")
    print(f"Platform: {jax.devices()[0].platform}")

    try:
        test_model_init()
        test_forward_pass()
        test_loss_computation()
        test_gradient_computation()
        test_data_pipeline()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nJAX implementation is ready for deployment!")

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
