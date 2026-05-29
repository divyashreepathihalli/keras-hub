"""Paged attention sublayer for vLLM/tpu-inference.

Wraps KerasHub's q/k/v/o projection weights with per-token RoPE
(using input_positions from attention_metadata) and delegates KV
write + attend to tpu-inference's attention() interface.
"""

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from tpu_inference.layers.common.attention import attention as paged_attention
except ImportError:
    nnx = None
    paged_attention = None


def _apply_rope_per_token(x, positions, head_dim):
    """Apply rotary position embeddings per-token.

    Args:
        x: [num_tokens, num_heads, head_dim]
        positions: [num_tokens] int32 positions
        head_dim: int

    Returns:
        x with RoPE applied, same shape.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    # positions: [num_tokens] → [num_tokens, half]
    freqs = jnp.outer(positions.astype(jnp.float32), inv_freq)
    cos = jnp.cos(freqs)  # [num_tokens, half]
    sin = jnp.sin(freqs)

    # x: [num_tokens, num_heads, head_dim]
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos[:, None, :]  # broadcast over heads
    sin = sin[:, None, :]
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return x * jnp.concatenate([cos, cos], axis=-1) + rotated * jnp.concatenate(
        [sin, sin], axis=-1
    )


class PagedGemmaAttention(nnx.Module if nnx is not None else object):
    """Single Gemma attention layer adapted for paged KV caches.

    Uses KerasHub weight shapes for q/k/v/o projections but applies
    per-token positional encoding and paged KV via tpu-inference.
    """

    def __init__(self, dims: dict, mesh, rngs):
        if nnx is None:
            raise ImportError(
                "flax.nnx and tpu-inference are required. "
                "Install keras-hub[tpu]."
            )
        num_q = dims["num_query_heads"]
        num_kv = dims["num_key_value_heads"]
        head_dim = dims["head_dim"]
        hidden_dim = dims["hidden_dim"]

        self.num_query_heads = num_q
        self.num_key_value_heads = num_kv
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.logit_soft_cap = dims.get("attention_logit_soft_cap")
        self.query_head_dim_normalize = dims.get("query_head_dim_normalize", True)

        # q/k/v/o kernel shapes matching KerasHub EinsumDense layouts.
        # query: "btd,ndh->btnh"  kernel: [num_q, hidden_dim, head_dim]
        # key:   "bsd,kdh->bskh"  kernel: [num_kv, hidden_dim, head_dim]
        self.q_kernel = nnx.Param(
            jnp.zeros((num_q, hidden_dim, head_dim), dtype=jnp.bfloat16)
        )
        self.k_kernel = nnx.Param(
            jnp.zeros((num_kv, hidden_dim, head_dim), dtype=jnp.bfloat16)
        )
        self.v_kernel = nnx.Param(
            jnp.zeros((num_kv, hidden_dim, head_dim), dtype=jnp.bfloat16)
        )
        # output: "btnh,nhd->btd"  kernel: [num_q, head_dim, hidden_dim]
        self.o_kernel = nnx.Param(
            jnp.zeros((num_q, head_dim, hidden_dim), dtype=jnp.bfloat16)
        )

    def __call__(self, kv_cache, x, attention_metadata):
        """
        Args:
            kv_cache: paged KV buffer for this layer (tpu-inference managed)
            x: [num_tokens, hidden_dim] flattened batch
            attention_metadata: tpu-inference AttentionMetadata

        Returns:
            (updated_kv_cache, output [num_tokens, hidden_dim])
        """
        positions = attention_metadata.input_positions  # [num_tokens]

        # Project — shapes: [num_tokens, num_heads, head_dim]
        q = jnp.einsum("td,ndh->tnh", x, self.q_kernel.value)
        k = jnp.einsum("td,ndh->tnh", x, self.k_kernel.value)
        v = jnp.einsum("td,ndh->tnh", x, self.v_kernel.value)

        # Per-token RoPE with Gemma's interleaved layout.
        q = _apply_rope_per_token(q, positions, self.head_dim)
        k = _apply_rope_per_token(k, positions, self.head_dim)
        # Gemma interleave: stack halves then reshape.
        q = self._gemma_interleave(q)
        k = self._gemma_interleave(k)

        # Paged KV write + attention via tpu-inference.
        if self.query_head_dim_normalize:
            scale = 1.0 / np.sqrt(self.head_dim)
        else:
            scale = 1.0 / np.sqrt(
                self.hidden_dim // self.num_query_heads
            )

        kv_cache, out = paged_attention(
            kv_cache, q, k, v, attention_metadata, scale=scale
        )

        # out: [num_tokens, num_query_heads, head_dim] → [num_tokens, hidden]
        out = jnp.einsum("tnh,nhd->td", out, self.o_kernel.value)
        return kv_cache, out

    def _gemma_interleave(self, x):
        """Match KerasHub's Gemma RoPE interleave convention."""
        half = self.head_dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        interleaved = jnp.stack([x1, x2], axis=-1)
        return interleaved.reshape(x.shape)
