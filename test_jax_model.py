#!/usr/bin/env python3
"""
Test script for JAX model forward pass.
Verifies that the TRM model works correctly with JAX.
"""

import jax
import jax.numpy as jnp
from jax import random

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


def test_model_forward():
    """Test a simple forward pass through the TRM model."""
    print("="*80)
    print("Testing JAX TRM Model Forward Pass")
    print("="*80)

    # Check JAX setup
    print(f"\nJAX devices: {jax.devices()}")
    print(f"Process count: {jax.process_count()}")
    print(f"Device count: {jax.device_count()}")

    # Create a minimal config
    config_dict = {
        'batch_size': 4,
        'seq_len': 81,  # Sudoku grid
        'puzzle_emb_ndim': 0,  # No puzzle embeddings for simple test
        'num_puzzle_identifiers': 1000,
        'vocab_size': 20,
        'H_cycles': 2,
        'L_cycles': 2,
        'H_layers': 1,
        'L_layers': 2,
        'hidden_size': 128,
        'expansion': 2.0,
        'num_heads': 4,
        'pos_encodings': 'rope',
        'rms_norm_eps': 1e-5,
        'rope_theta': 10000.0,
        'halt_max_steps': 10,
        'halt_exploration_prob': 0.1,
        'forward_dtype': 'bfloat16',
        'mlp_t': False,
        'puzzle_emb_len': 0,
        'no_ACT_continue': True,
    }

    print(f"\nModel config: {config_dict['hidden_size']}d, {config_dict['H_cycles']}H, {config_dict['L_cycles']}L")

    # Create model
    model = TinyRecursiveReasoningModel_ACTV1(config_dict=config_dict)

    # Create dummy batch
    rng = random.PRNGKey(0)
    batch = {
        'inputs': jnp.ones((config_dict['batch_size'], config_dict['seq_len']), dtype=jnp.int32),
        'labels': jnp.ones((config_dict['batch_size'], config_dict['seq_len']), dtype=jnp.int32),
        'puzzle_identifiers': jnp.zeros((config_dict['batch_size'],), dtype=jnp.int32),
    }

    print(f"\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")

    # Initialize model parameters
    print("\nInitializing model...")
    rng, init_rng = random.split(rng)
    carry = model.initial_carry(batch)
    params = model.init(init_rng, carry=carry, batch=batch, training=False, rng=init_rng)

    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {num_params:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    rng, fwd_rng = random.split(rng)
    carry, outputs = model.apply(params, carry=carry, batch=batch, training=False, rng=fwd_rng)

    print(f"\nOutputs:")
    for k, v in outputs.items():
        if isinstance(v, jnp.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")

    print(f"\nCarry state:")
    print(f"  halted: {carry.halted}")
    print(f"  steps: {carry.steps}")

    # Test ACT loop
    print("\n" + "="*80)
    print("Testing ACT Loop")
    print("="*80)

    carry = model.initial_carry(batch)
    for step in range(5):
        rng, step_rng = random.split(rng)
        carry, outputs = model.apply(params, carry=carry, batch=batch, training=True, rng=step_rng)
        print(f"Step {step}: halted={jnp.sum(carry.halted)}/{config_dict['batch_size']}, "
              f"q_halt={outputs['q_halt_logits'][:2]}")

        if jnp.all(carry.halted):
            print(f"All sequences halted at step {step}")
            break

    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)


if __name__ == "__main__":
    test_model_forward()
