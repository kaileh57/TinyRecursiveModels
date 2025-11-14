"""
JAX/Flax implementation of Tiny Recursive Reasoning Model (TRM)
Converted from PyTorch models/recursive_reasoning/trm.py
"""

from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
import chex

from jax_models.layers import (
    Dense, Embedding, RotaryEmbedding, Attention,
    SwiGLU, rms_norm, CosSin
)

IGNORE_LABEL_ID = -100


@struct.dataclass
class InnerCarry:
    """Inner carry state for recursive reasoning"""
    z_H: chex.Array  # [B, L, D]
    z_L: chex.Array  # [B, L, D]


@struct.dataclass
class OuterCarry:
    """Outer carry state with ACT halting"""
    inner_carry: InnerCarry
    steps: chex.Array  # [B]
    halted: chex.Array  # [B] bool
    current_data: Dict[str, chex.Array]


@struct.dataclass
class TRMConfig:
    """Configuration for TRM model"""
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int

    # Architecture
    hidden_size: int
    expansion: float
    num_heads: int
    num_layers: int  # L_layers

    # Recursion
    H_cycles: int
    L_cycles: int

    # Position encodings
    pos_encodings: str = "rope"  # "rope", "learned", or "none"
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    # ACT halting
    halt_max_steps: int = 1
    halt_exploration_prob: float = 0.0
    no_ACT_continue: bool = True

    # Puzzle embeddings
    puzzle_emb_ndim: int = 0
    puzzle_emb_len: int = 16
    mlp_t: bool = False

    # Dtype
    dtype: Any = jnp.bfloat16


class TRMBlock(nn.Module):
    """Single TRM transformer block"""
    config: TRMConfig

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray, cos_sin: Optional[CosSin] = None) -> jnp.ndarray:
        # B, L, D = hidden_states.shape

        if self.config.mlp_t:
            # MLP over sequence dimension (transpose trick)
            x = jnp.transpose(hidden_states, (0, 2, 1))  # [B, D, L]
            x = SwiGLU(
                hidden_size=self.config.seq_len + self.config.puzzle_emb_len,
                expansion=self.config.expansion,
                name='mlp_t'
            )(x)
            x = jnp.transpose(x, (0, 2, 1))  # [B, L, D]
            hidden_states = rms_norm(hidden_states + x, self.config.rms_norm_eps)
        else:
            # Self-attention
            attn_out = Attention(
                hidden_size=self.config.hidden_size,
                head_dim=self.config.hidden_size // self.config.num_heads,
                num_heads=self.config.num_heads,
                num_key_value_heads=self.config.num_heads,
                causal=False,
                name='self_attn'
            )(hidden_states, cos_sin)
            hidden_states = rms_norm(hidden_states + attn_out, self.config.rms_norm_eps)

        # MLP
        mlp_out = SwiGLU(
            hidden_size=self.config.hidden_size,
            expansion=self.config.expansion,
            name='mlp'
        )(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_out, self.config.rms_norm_eps)

        return hidden_states


class ReasoningModule(nn.Module):
    """Stack of TRM blocks for reasoning"""
    config: TRMConfig

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray, input_injection: jnp.ndarray, cos_sin: Optional[CosSin] = None) -> jnp.ndarray:
        # Add input injection
        hidden_states = hidden_states + input_injection

        # Apply layers
        for i in range(self.config.num_layers):
            hidden_states = TRMBlock(self.config, name=f'layer_{i}')(hidden_states, cos_sin)

        return hidden_states


class TRMInner(nn.Module):
    """Inner TRM model without ACT wrapper"""
    config: TRMConfig

    def setup(self):
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Calculate puzzle embedding length
        if self.config.puzzle_emb_len == 0:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # Ceil div
        else:
            self.puzzle_emb_len = self.config.puzzle_emb_len

    @nn.compact
    def __call__(self, carry: InnerCarry, batch: Dict[str, jnp.ndarray]) -> Tuple[InnerCarry, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        # Input embeddings
        embed_init_std = 1.0 / self.embed_scale

        # Token embedding
        embedding = Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.hidden_size,
            init_std=embed_init_std,
            name='embed_tokens'
        )(batch["inputs"].astype(jnp.int32))

        # Puzzle embeddings (if enabled)
        if self.config.puzzle_emb_ndim > 0:
            # Initialize puzzle embeddings
            puzzle_emb_init = nn.initializers.truncated_normal(stddev=0.0)  # Zero init
            puzzle_embedding = self.param(
                'puzzle_emb',
                puzzle_emb_init,
                (self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim)
            )

            # Gather puzzle embeddings
            puzzle_ids = batch["puzzle_identifiers"]
            puzzle_emb_out = puzzle_embedding[puzzle_ids]

            # Pad if needed
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_emb_out.shape[-1]
            if pad_count > 0:
                puzzle_emb_out = jnp.pad(puzzle_emb_out, ((0, 0), (0, pad_count)))

            # Reshape and concatenate
            puzzle_emb_out = puzzle_emb_out.reshape(-1, self.puzzle_emb_len, self.config.hidden_size)
            embedding = jnp.concatenate([puzzle_emb_out, embedding], axis=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            pos_emb = Embedding(
                num_embeddings=self.config.seq_len + self.puzzle_emb_len,
                embedding_dim=self.config.hidden_size,
                init_std=embed_init_std,
                name='embed_pos'
            )(jnp.arange(self.config.seq_len + self.puzzle_emb_len))
            # Scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + pos_emb)

        # Scale embeddings
        input_embeddings = self.embed_scale * embedding

        # Get position encodings
        if self.config.pos_encodings == "rope":
            rope = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
                name='rotary_emb'
            )
            cos_sin = rope()
        else:
            cos_sin = None

        # Initialize L-level reasoning module
        L_level = ReasoningModule(self.config, name='L_level')

        # Initialize carry states
        z_H, z_L = carry.z_H, carry.z_L

        # Recursive reasoning: H_cycles with L_cycles each
        # First H_cycles-1 without gradient (use jax.lax.stop_gradient)
        for h_step in range(self.config.H_cycles - 1):
            for l_step in range(self.config.L_cycles):
                z_L_input = jax.lax.stop_gradient(z_L)
                z_H_input = jax.lax.stop_gradient(z_H)
                z_L = L_level(z_L_input, z_H_input + input_embeddings, cos_sin)
                z_L = jax.lax.stop_gradient(z_L)
            z_H = L_level(jax.lax.stop_gradient(z_H), jax.lax.stop_gradient(z_L), cos_sin)
            z_H = jax.lax.stop_gradient(z_H)

        # Last H_cycle with gradient
        for l_step in range(self.config.L_cycles):
            z_L = L_level(z_L, z_H + input_embeddings, cos_sin)
        z_H = L_level(z_H, z_L, cos_sin)

        # Output heads
        lm_head = Dense(features=self.config.vocab_size, use_bias=False, name='lm_head')
        q_head = Dense(features=2, use_bias=True, name='q_head')

        # LM output (skip puzzle embeddings)
        output = lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q-head output (use first position)
        q_logits = q_head(z_H[:, 0])
        q_halt_logits = q_logits[..., 0]
        q_continue_logits = q_logits[..., 1]

        # New carry (detached)
        new_carry = InnerCarry(
            z_H=jax.lax.stop_gradient(z_H),
            z_L=jax.lax.stop_gradient(z_L)
        )

        return new_carry, output, (q_halt_logits, q_continue_logits)


class TinyRecursiveReasoningModel(nn.Module):
    """TRM with ACT halting wrapper"""
    config: TRMConfig

    def empty_carry(self, batch_size: int) -> InnerCarry:
        """Create empty inner carry"""
        seq_len = self.config.seq_len + self.config.puzzle_emb_len
        return InnerCarry(
            z_H=jnp.zeros((batch_size, seq_len, self.config.hidden_size), dtype=self.config.dtype),
            z_L=jnp.zeros((batch_size, seq_len, self.config.hidden_size), dtype=self.config.dtype)
        )

    def reset_carry(self, reset_flag: jnp.ndarray, carry: InnerCarry) -> InnerCarry:
        """Reset carry for halted sequences"""
        # Initialize with truncated normal
        batch_size = reset_flag.shape[0]
        seq_len = self.config.seq_len + self.config.puzzle_emb_len

        # Use random initialization for reset
        H_init = jax.random.truncated_normal(
            self.make_rng('carry'),
            lower=-2.0, upper=2.0,
            shape=(batch_size, seq_len, self.config.hidden_size),
            dtype=self.config.dtype
        )
        L_init = jax.random.truncated_normal(
            self.make_rng('carry'),
            lower=-2.0, upper=2.0,
            shape=(batch_size, seq_len, self.config.hidden_size),
            dtype=self.config.dtype
        )

        reset_flag_expanded = reset_flag[:, None, None]
        return InnerCarry(
            z_H=jnp.where(reset_flag_expanded, H_init, carry.z_H),
            z_L=jnp.where(reset_flag_expanded, L_init, carry.z_L)
        )

    def initial_carry(self, batch: Dict[str, jnp.ndarray]) -> OuterCarry:
        """Create initial carry state"""
        batch_size = batch["inputs"].shape[0]

        return OuterCarry(
            inner_carry=self.empty_carry(batch_size),
            steps=jnp.zeros((batch_size,), dtype=jnp.int32),
            halted=jnp.ones((batch_size,), dtype=bool),
            current_data={k: jnp.zeros_like(v) for k, v in batch.items()}
        )

    @nn.compact
    def __call__(
        self,
        carry: OuterCarry,
        batch: Dict[str, jnp.ndarray],
        training: bool = False
    ) -> Tuple[OuterCarry, Dict[str, jnp.ndarray]]:
        """Forward pass with ACT halting"""

        # Reset carry for halted sequences
        new_inner_carry = self.reset_carry(carry.halted, carry.inner_carry)

        # Reset steps for halted sequences
        new_steps = jnp.where(carry.halted, 0, carry.steps)

        # Update current data for halted sequences
        new_current_data = {}
        for k, v in carry.current_data.items():
            halted_expanded = carry.halted.reshape((-1,) + (1,) * (batch[k].ndim - 1))
            new_current_data[k] = jnp.where(halted_expanded, batch[k], v)

        # Forward through inner model
        inner = TRMInner(self.config, name='inner')
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = inner(new_inner_carry, new_current_data)

        # Outputs
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        # Update steps and compute halting
        new_steps = new_steps + 1
        is_last_step = new_steps >= self.config.halt_max_steps
        halted = is_last_step

        # ACT halting (only during training)
        if training and self.config.halt_max_steps > 1:
            if self.config.no_ACT_continue:
                halted = halted | (q_halt_logits > 0)
            else:
                halted = halted | (q_halt_logits > q_continue_logits)

            # Exploration
            if self.config.halt_exploration_prob > 0:
                explore_rng = self.make_rng('exploration')
                rand_vals = jax.random.uniform(explore_rng, q_halt_logits.shape)
                min_halt_steps = jnp.where(
                    rand_vals < self.config.halt_exploration_prob,
                    jax.random.randint(explore_rng, new_steps.shape, 2, self.config.halt_max_steps + 1),
                    0
                )
                halted = halted & (new_steps >= min_halt_steps)

        # Create new carry
        new_carry = OuterCarry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=new_current_data
        )

        return new_carry, outputs
