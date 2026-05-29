"""KerasHub-backed causal LM models for vLLM/tpu-inference.

Each class implements the tpu-inference model contract:
  - __init__(vllm_config, rng_key, mesh)
  - __call__(kv_caches, input_ids, attention_metadata) → (kv_caches, hidden, aux)
  - compute_logits(hidden_states) → logits
  - load_weights(rng)

The only non-trivial sublayer is PagedGemmaAttention (paged_attention.py);
all other math (embedding, norms, MLP) is expressed inline with JAX/NNX
weight shapes that match KerasHub's conventions exactly so weight_loader.py
can copy without reshaping.
"""

try:
    import jax.numpy as jnp
    from flax import nnx
except ImportError:
    nnx = None

from keras_hub.src.utils.vllm.paged_attention import PagedGemmaAttention
from keras_hub.src.utils.vllm.weight_loader import load_preset_weights


def _dims_from_vllm_config(vllm_config):
    hf = vllm_config.model_config.hf_config
    return {
        "vocabulary_size": hf.vocab_size,
        "num_layers": hf.num_hidden_layers,
        "num_query_heads": hf.num_attention_heads,
        "num_key_value_heads": hf.num_key_value_heads,
        "hidden_dim": hf.hidden_size,
        "intermediate_dim": hf.intermediate_size,
        "head_dim": hf.head_dim,
        "layer_norm_epsilon": getattr(hf, "rms_norm_eps", 1e-6),
        "query_head_dim_normalize": getattr(hf, "query_head_dim_normalize", True),
        "use_post_ffw_norm": getattr(hf, "use_post_ffw_norm", False),
        "use_post_attention_norm": getattr(hf, "use_post_attention_norm", False),
        "attention_logit_soft_cap": getattr(hf, "attn_logit_softcapping", None),
        "final_logit_soft_cap": getattr(hf, "final_logit_softcapping", None),
    }


class _PagedDecoderLayer(nnx.Module if nnx is not None else object):
    """Single Gemma decoder layer with paged KV cache."""

    def __init__(self, dims: dict, mesh, rngs):
        if nnx is None:
            raise ImportError("flax.nnx required; install keras-hub[tpu]")
        hidden_dim = dims["hidden_dim"]
        intermediate_dim = dims["intermediate_dim"]

        self.attention = PagedGemmaAttention(dims, mesh, rngs)

        # RMS norms stored as raw scale params (no bias).
        self.pre_attention_norm = nnx.Param(
            jnp.ones((hidden_dim,), dtype=jnp.bfloat16)
        )
        self.pre_ffw_norm = nnx.Param(
            jnp.ones((hidden_dim,), dtype=jnp.bfloat16)
        )
        if dims.get("use_post_attention_norm", False):
            self.post_attention_norm = nnx.Param(
                jnp.ones((hidden_dim,), dtype=jnp.bfloat16)
            )
        if dims.get("use_post_ffw_norm", False):
            self.post_ffw_norm = nnx.Param(
                jnp.ones((hidden_dim,), dtype=jnp.bfloat16)
            )

        # MLP kernels — Gemma SwiGLU: gating_ffw/gating_ffw_2 project to
        # intermediate_dim//2, then ffw_linear projects back to hidden_dim.
        half_inter = intermediate_dim // 2
        self.gate_kernel = nnx.Param(
            jnp.zeros((hidden_dim, half_inter), dtype=jnp.bfloat16)
        )
        self.ffw1_kernel = nnx.Param(
            jnp.zeros((hidden_dim, half_inter), dtype=jnp.bfloat16)
        )
        self.ffw2_kernel = nnx.Param(
            jnp.zeros((half_inter, hidden_dim), dtype=jnp.bfloat16)
        )

        self.layer_norm_epsilon = dims.get("layer_norm_epsilon", 1e-6)
        self.dims = dims

    def _rms_norm(self, x, scale):
        x = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.layer_norm_epsilon)
        return (x / rms * (1.0 + scale.astype(jnp.float32))).astype(jnp.bfloat16)

    def _mlp(self, x):
        # Gemma SwiGLU: gelu(gate) * up, then project down.
        gate = jax_gelu(jnp.einsum("td,df->tf", x, self.gate_kernel.value))
        up = jnp.einsum("td,df->tf", x, self.ffw1_kernel.value)
        return jnp.einsum("tf,fd->td", gate * up, self.ffw2_kernel.value)

    def __call__(self, kv_cache, x, attention_metadata):
        # Pre-attention norm + residual.
        normed = self._rms_norm(x, self.pre_attention_norm.value)
        kv_cache, attn_out = self.attention(kv_cache, normed, attention_metadata)
        if hasattr(self, "post_attention_norm"):
            attn_out = self._rms_norm(attn_out, self.post_attention_norm.value)
        x = x + attn_out

        # Pre-FFW norm + residual.
        normed = self._rms_norm(x, self.pre_ffw_norm.value)
        ffw_out = self._mlp(normed)
        if hasattr(self, "post_ffw_norm"):
            ffw_out = self._rms_norm(ffw_out, self.post_ffw_norm.value)
        x = x + ffw_out
        return kv_cache, x


def jax_gelu(x):
    """GELU activation matching Gemma's approximate implementation."""
    import jax.nn as jnn
    return jnn.gelu(x, approximate=True)


class KerasHubGemmaForCausalLM(nnx.Module if nnx is not None else object):
    """Gemma CausalLM for vLLM/tpu-inference, weights from a KerasHub preset.

    Implements the tpu-inference model contract so that vLLM's scheduler,
    PagedAttention engine, and continuous batching work unchanged.
    """

    def __init__(self, vllm_config, rng_key, mesh):
        if nnx is None:
            raise ImportError("flax.nnx required; install keras-hub[tpu]")
        import jax

        self.vllm_config = vllm_config
        self.mesh = mesh
        rngs = nnx.Rngs(rng_key)
        dims = _dims_from_vllm_config(vllm_config)
        self._dims = dims

        vocab_size = dims["vocabulary_size"]
        hidden_dim = dims["hidden_dim"]
        num_layers = dims["num_layers"]

        # Token embedding table — [vocab_size, hidden_dim]
        self.token_embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=hidden_dim,
            dtype=jnp.bfloat16,
            rngs=rngs,
        )

        self.layers = [
            _PagedDecoderLayer(dims, mesh, rngs) for _ in range(num_layers)
        ]

        self.final_norm_scale = nnx.Param(
            jnp.ones((hidden_dim,), dtype=jnp.bfloat16)
        )
        self._layer_norm_epsilon = dims.get("layer_norm_epsilon", 1e-6)
        self._final_logit_soft_cap = dims.get("final_logit_soft_cap")

    def _rms_norm_final(self, x):
        x = x.astype(jnp.float32)
        rms = jnp.sqrt(
            jnp.mean(x ** 2, axis=-1, keepdims=True) + self._layer_norm_epsilon
        )
        scale = 1.0 + self.final_norm_scale.value.astype(jnp.float32)
        return (x / rms * scale).astype(jnp.bfloat16)

    def __call__(self, kv_caches, input_ids, attention_metadata):
        """Forward pass — matches tpu-inference model contract.

        Args:
            kv_caches: list[jax.Array] — one paged buffer per layer.
            input_ids: [num_tokens] int32 — flattened, ragged batch.
            attention_metadata: tpu-inference AttentionMetadata.

        Returns:
            (kv_caches, hidden_states [num_tokens, hidden_dim], [])
        """
        x = self.token_embedding(input_ids)
        # Gemma scales embeddings by sqrt(hidden_dim).
        x = x * jnp.sqrt(float(self._dims["hidden_dim"])).astype(x.dtype)

        for i, layer in enumerate(self.layers):
            kv_caches[i], x = layer(kv_caches[i], x, attention_metadata)

        x = self._rms_norm_final(x)
        return kv_caches, x, []

    def compute_logits(self, hidden_states):
        """Project hidden states to vocabulary logits.

        Uses the tied embedding matrix (reverse embedding), matching
        KerasHub's ReversibleEmbedding.call(reverse=True).
        """
        logits = jnp.einsum(
            "td,vd->tv",
            hidden_states.astype(jnp.float32),
            self.token_embedding.embedding.value.astype(jnp.float32),
        )
        if self._final_logit_soft_cap is not None:
            cap = float(self._final_logit_soft_cap)
            logits = jnp.tanh(logits / cap) * cap
        return logits

    def load_weights(self, rng):
        """Load weights directly from the KerasHub preset.

        Called by tpu-inference after nnx.eval_shape construction.
        The preset name is taken from vllm_config.model_config.model.
        """
        preset_name = self.vllm_config.model_config.model
        load_preset_weights(self, preset_name, self.mesh)
