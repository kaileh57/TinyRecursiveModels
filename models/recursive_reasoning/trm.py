from typing import Tuple, List, Dict, Optional, NamedTuple
import math
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax import struct
from pydantic import BaseModel

from models.common import trunc_normal_init
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


@struct.dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    """Inner carry state for TRM model."""
    z_H: jnp.ndarray
    z_L: jnp.ndarray


@struct.dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    """Full carry state for TRM model with ACT."""
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    steps: jnp.ndarray
    halted: jnp.ndarray
    current_data: Dict[str, jnp.ndarray]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    """Configuration for TRM model."""
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int  # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False  # use mlp on L instead of transformer
    puzzle_emb_len: int = 16  # if non-zero, its specified to this value
    no_ACT_continue: bool = True  # No continue ACT loss, only use the sigmoid of the halt


class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    """Single transformer block for TRM."""
    config: TinyRecursiveReasoningModel_ACTV1Config
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        if self.config.mlp_t:
            self.puzzle_emb_len = (
                -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
                if self.config.puzzle_emb_len == 0
                else self.config.puzzle_emb_len
            )
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,  # L
                expansion=self.config.expansion,
                dtype=self.dtype
            )
        else:
            self.self_attn = Attention(
                hidden_size=self.config.hidden_size,
                head_dim=self.config.hidden_size // self.config.num_heads,
                num_heads=self.config.num_heads,
                num_key_value_heads=self.config.num_heads,
                causal=False,
                dtype=self.dtype
            )

        self.mlp = SwiGLU(
            hidden_size=self.config.hidden_size,
            expansion=self.config.expansion,
            dtype=self.dtype
        )

    def __call__(self, hidden_states: jnp.ndarray, cos_sin: Optional[CosSin] = None) -> jnp.ndarray:
        # Post Norm
        if self.config.mlp_t:
            hidden_states_t = jnp.transpose(hidden_states, (0, 2, 1))
            out = self.mlp_t(hidden_states_t)
            hidden_states_t = rms_norm(hidden_states_t + out, variance_epsilon=self.config.rms_norm_eps)
            hidden_states = jnp.transpose(hidden_states_t, (0, 2, 1))
        else:
            # Self Attention
            hidden_states = rms_norm(
                hidden_states + self.self_attn(hidden_states, cos_sin=cos_sin),
                variance_epsilon=self.config.rms_norm_eps
            )

        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.config.rms_norm_eps)

        return hidden_states


class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    """Reasoning module with multiple transformer blocks."""
    config: TinyRecursiveReasoningModel_ACTV1Config
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.layers = [
            TinyRecursiveReasoningModel_ACTV1Block(self.config, dtype=self.dtype)
            for _ in range(self.num_layers)
        ]

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        input_injection: jnp.ndarray,
        cos_sin: Optional[CosSin] = None
    ) -> jnp.ndarray:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin=cos_sin)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    """Inner TRM model."""
    config: TinyRecursiveReasoningModel_ACTV1Config

    def setup(self):
        self.forward_dtype = getattr(jnp, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            dtype=self.forward_dtype
        )
        self.lm_head = CastedLinear(
            self.config.hidden_size,
            self.config.vocab_size,
            use_bias=False,
            dtype=self.forward_dtype
        )
        self.q_head = CastedLinear(
            self.config.hidden_size,
            2,
            use_bias=True,
            dtype=self.forward_dtype
        )

        self.puzzle_emb_len = (
            -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
            if self.config.puzzle_emb_len == 0
            else self.config.puzzle_emb_len
        )

        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                init_std=0,
                dtype=self.forward_dtype
            )

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                dtype=self.forward_dtype
            )

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            self.config,
            num_layers=self.config.L_layers,
            dtype=self.forward_dtype
        )

    def _input_embeddings(self, input: jnp.ndarray, puzzle_identifiers: jnp.ndarray, training: bool = True):
        """Compute input embeddings with puzzle and position embeddings."""
        # Token embedding
        embedding = self.embed_tokens(input.astype(jnp.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers, training=training)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = jnp.pad(puzzle_embedding, ((0, 0), (0, pad_count)))

            puzzle_embedding = puzzle_embedding.reshape(-1, self.puzzle_emb_len, self.config.hidden_size)
            embedding = jnp.concatenate((puzzle_embedding, embedding), axis=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.astype(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        """Create empty carry state."""
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=jnp.zeros(
                (batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size),
                dtype=self.forward_dtype
            ),
            z_L=jnp.zeros(
                (batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size),
                dtype=self.forward_dtype
            ),
        )

    def reset_carry(
        self,
        reset_flag: jnp.ndarray,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        H_init: jnp.ndarray,
        L_init: jnp.ndarray
    ):
        """Reset carry state for halted sequences."""
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=jnp.where(reset_flag[:, None, None], H_init, carry.z_H),
            z_L=jnp.where(reset_flag[:, None, None], L_init, carry.z_L),
        )

    def __call__(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, jnp.ndarray],
        H_init: jnp.ndarray,
        L_init: jnp.ndarray,
        training: bool = True
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Forward pass through inner model."""
        # Get position encodings
        cos_sin = None
        if hasattr(self, "rotary_emb"):
            cos_sin = self.rotary_emb()

        # Input encoding
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch["puzzle_identifiers"],
            training=training
        )

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L

        # H_cycles-1 without grad (for faster training)
        # Note: In JAX, we explicitly stop gradients rather than using no_grad context
        for _H_step in range(self.config.H_cycles - 1):
            for _L_step in range(self.config.L_cycles):
                z_L = jax.lax.stop_gradient(
                    self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
                )
            z_H = jax.lax.stop_gradient(self.L_level(z_H, z_L, cos_sin=cos_sin))

        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
        z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)

        # LM Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=jax.lax.stop_gradient(z_H),
            z_L=jax.lax.stop_gradient(z_L)
        )
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).astype(jnp.float32)  # Q-head; uses the first puzzle_emb position

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper for TRM model."""
    config_dict: dict

    def setup(self):
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**self.config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

        # Initial states
        forward_dtype = getattr(jnp, self.config.forward_dtype)

        # These will be initialized as parameters
        def h_init_fn(rng, shape, dtype):
            return trunc_normal_init(rng, shape, dtype=dtype, std=1.0)

        self.H_init = self.param('H_init', h_init_fn, (self.config.hidden_size,), forward_dtype)
        self.L_init = self.param('L_init', h_init_fn, (self.config.hidden_size,), forward_dtype)

    def initial_carry(self, batch: Dict[str, jnp.ndarray]):
        """Create initial carry state."""
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=jnp.zeros((batch_size,), dtype=jnp.int32),
            halted=jnp.ones((batch_size,), dtype=jnp.bool_),  # Default to halted
            current_data={k: jnp.zeros_like(v) for k, v in batch.items()}
        )

    def __call__(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1Carry,
        batch: Dict[str, jnp.ndarray],
        training: bool = True,
        rng: Optional[random.PRNGKey] = None
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, jnp.ndarray]]:
        """Forward pass through TRM model with ACT."""
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(
            carry.halted,
            carry.inner_carry,
            self.H_init,
            self.L_init
        )

        new_steps = jnp.where(carry.halted, 0, carry.steps)

        # Update current data for halted sequences
        new_current_data = {}
        for k, v in carry.current_data.items():
            expand_dims = tuple(1 for _ in range(batch[k].ndim - 1))
            mask = carry.halted.reshape((-1,) + expand_dims)
            new_current_data[k] = jnp.where(mask, batch[k], v)

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry,
            new_current_data,
            self.H_init,
            self.L_init,
            training=training
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        # Update steps and compute halting
        new_steps = new_steps + 1
        is_last_step = new_steps >= self.config.halt_max_steps

        halted = is_last_step

        # if training, and ACT is enabled
        if training and (self.config.halt_max_steps > 1):
            # Halt signal
            if self.config.no_ACT_continue:
                halted = halted | (q_halt_logits > 0)
            else:
                halted = halted | (q_halt_logits > q_continue_logits)

            # Exploration (only during training)
            if rng is not None:
                rng1, rng2 = random.split(rng)
                explore_mask = random.uniform(rng1, q_halt_logits.shape) < self.config.halt_exploration_prob
                min_halt_steps = explore_mask * random.randint(
                    rng2, new_steps.shape, minval=2, maxval=self.config.halt_max_steps + 1
                )
                halted = halted & (new_steps >= min_halt_steps)

            if not self.config.no_ACT_continue:
                # Compute target Q (for Q-learning)
                _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(
                    new_inner_carry,
                    new_current_data,
                    self.H_init,
                    self.L_init,
                    training=training
                )
                target_q = jnp.where(
                    is_last_step,
                    next_q_halt_logits,
                    jnp.maximum(next_q_halt_logits, next_q_continue_logits)
                )
                outputs["target_q_continue"] = jax.nn.sigmoid(target_q)

        new_carry = TinyRecursiveReasoningModel_ACTV1Carry(
            new_inner_carry,
            new_steps,
            halted,
            new_current_data
        )

        return new_carry, outputs
