"""Gemma3 text-only model for the tpu-inference JAX/vLLM serving stack.

Provides a native JAX/Flax nnx implementation of Gemma3 that plugs into
tpu-inference's JAX paged-attention serving infrastructure. Any Gemma3
text-only checkpoint loadable from HuggingFace (e.g. gemma3_1b,
gemma3_270m) is supported via the StandardWeightLoader.

Registration in tpu-inference (add once at startup):

    from keras_hub.src.utils.vllm.gemma3_jax import Gemma3TextForCausalLM
    from tpu_inference.models.common.model_loader import register_model
    register_model("Gemma3TextForCausalLM", Gemma3TextForCausalLM)

The HuggingFace architecture string "Gemma3TextForCausalLM" (set in the
model's config.json) is used to select this class at serving time.
"""

from itertools import islice
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

# ---------------------------------------------------------------------------
# Soft-import tpu_inference.  The module is importable without tpu_inference
# installed; inference requires it.
# ---------------------------------------------------------------------------
try:
    from tpu_inference import utils as tpu_utils
    from tpu_inference.distributed.jax_parallel_state import get_pp_group
    from tpu_inference.layers.common.attention_interface import attention
    from tpu_inference.layers.common.attention_metadata import AttentionMetadata
    from tpu_inference.layers.common.sharding import ShardingAxisName
    from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
    from tpu_inference.layers.jax.rope_interface import apply_rope
    from tpu_inference.models.jax.jax_intermediate_tensor import (
        JaxIntermediateTensors,
    )
    from tpu_inference.models.jax.utils.weight_utils import StandardWeightLoader

    _HAS_TPU_INFERENCE = True
except ImportError:
    _HAS_TPU_INFERENCE = False

# vLLM config is required for the constructor signature
from vllm.config import VllmConfig

_init_fn = nnx.initializers.uniform()

# ---------------------------------------------------------------------------
# Helpers for reading per-layer RoPE configuration from the HF config.
# Gemma3 uses 10 000 for local (sliding-window) layers and 1 000 000 for
# global (full-attention) layers.  The exact values live in either the
# `rope_parameters` dict (transformers ≥ 5.x) or the flat `rope_theta` /
# `rope_local_base_freq` fields.
# ---------------------------------------------------------------------------


def _local_rope_theta(hf_config) -> float:
    rope_params = getattr(hf_config, "rope_parameters", None) or {}
    if isinstance(rope_params, dict) and "sliding_attention" in rope_params:
        return rope_params["sliding_attention"].get("rope_theta", 10_000.0)
    return float(getattr(hf_config, "rope_local_base_freq", 10_000.0))


def _global_rope_theta(hf_config) -> float:
    rope_params = getattr(hf_config, "rope_parameters", None) or {}
    if isinstance(rope_params, dict) and "full_attention" in rope_params:
        return rope_params["full_attention"].get("rope_theta", 1_000_000.0)
    return float(getattr(hf_config, "rope_theta", 1_000_000.0))


def _is_sliding_layer(hf_config, layer_idx: int) -> bool:
    """True when layer_idx uses sliding-window (local) attention."""
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is not None and layer_idx < len(layer_types):
        return layer_types[layer_idx] == "sliding_attention"
    # Fallback: Gemma3 pattern – 5 local then 1 global, repeating.
    return layer_idx % 6 < 5


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class Gemma3MLP(nnx.Module):
    """Gated MLP (SwiGLU) for Gemma3."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: jnp.dtype,
        rng: nnx.Rngs,
    ):
        self.gate_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                _init_fn, (None, ShardingAxisName.MLP_TENSOR)
            ),
            rngs=rng,
        )
        self.up_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                _init_fn, (None, ShardingAxisName.MLP_TENSOR)
            ),
            rngs=rng,
        )
        self.down_proj = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                _init_fn, (ShardingAxisName.MLP_TENSOR, None)
            ),
            rngs=rng,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = jax.nn.gelu(self.gate_proj(x), approximate=True)
        return self.down_proj(gate * self.up_proj(x))


class Gemma3Attention(nnx.Module):
    """Multi-head grouped-query attention for Gemma3.

    Differences from Llama-style attention:
    - Per-head RMS norm on queries and keys (use_query_key_norm).
    - Optional sliding-window (local) attention via attention_chunk_size.
    - Different RoPE theta per layer type.
    """

    def __init__(
        self,
        hf_config,
        layer_idx: int,
        dtype: jnp.dtype,
        rng: nnx.Rngs,
        mesh: Mesh,
        kv_cache_dtype: str,
    ):
        hidden_size = hf_config.hidden_size
        self.num_heads = hf_config.num_attention_heads
        self.num_kv_heads = hf_config.num_key_value_heads
        self.head_dim_original = hf_config.head_dim
        self.head_dim = tpu_utils.get_padded_head_dim(self.head_dim_original)
        self.mesh = mesh

        # Pad head/kv-head counts to be divisible by the TP size.
        sharding_size = tpu_utils.get_mesh_shape_product(
            mesh, ShardingAxisName.MLP_TENSOR
        )
        self.num_heads = tpu_utils.get_padded_num_heads(
            self.num_heads, sharding_size
        )
        self.num_kv_heads = tpu_utils.get_padded_num_heads(
            self.num_kv_heads, sharding_size
        )

        # Sliding window config
        self.is_sliding = _is_sliding_layer(hf_config, layer_idx)
        self.sliding_window = (
            hf_config.sliding_window if self.is_sliding else None
        )

        # RoPE theta depends on layer type
        self.rope_theta = (
            _local_rope_theta(hf_config)
            if self.is_sliding
            else _global_rope_theta(hf_config)
        )
        self.rope_scaling = getattr(hf_config, "rope_scaling", None)

        # KV cache quantization
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = (
                tpu_utils.get_jax_dtype_from_str_dtype(kv_cache_dtype)
            )

        # Projections using Einsum so weight shapes match get_default_maps
        # expectations: q/k/v → (D, N, H), o → (N, H, D).
        self.q_proj = nnx.Einsum(
            "TD,DNH->TNH",
            (hidden_size, self.num_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                _init_fn, (None, ShardingAxisName.ATTN_HEAD, None)
            ),
            rngs=rng,
        )
        self.k_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                _init_fn, (None, ShardingAxisName.ATTN_HEAD, None)
            ),
            rngs=rng,
        )
        self.v_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                _init_fn, (None, ShardingAxisName.ATTN_HEAD, None)
            ),
            rngs=rng,
        )
        self.o_proj = nnx.Einsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                _init_fn, (ShardingAxisName.ATTN_HEAD, None, None)
            ),
            rngs=rng,
        )

        # Per-head RMS norms on queries and keys (Gemma3-specific).
        self.q_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=hf_config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(_init_fn, (None,)),
            rngs=rng,
        )
        self.k_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=hf_config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(_init_fn, (None,)),
            rngs=rng,
        )

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: "AttentionMetadata",
    ) -> Tuple[Optional[jax.Array], jax.Array]:
        md = attention_metadata

        # Project and normalize queries and keys.
        q = self.q_proj(x)  # (T, N, H)
        q = self.q_norm(q)
        q = apply_rope(
            q, md.input_positions, self.head_dim_original, self.rope_theta,
            self.rope_scaling
        )

        k = self.k_proj(x)  # (T, K, H)
        k = self.k_norm(k)
        k = apply_rope(
            k, md.input_positions, self.head_dim_original, self.rope_theta,
            self.rope_scaling
        )

        v = self.v_proj(x)  # (T, K, H)

        new_kv_cache, output = attention(
            kv_cache,
            q,
            k,
            v,
            md,
            self.mesh,
            head_dim_original=self.head_dim_original,
            attention_chunk_size=self.sliding_window,
        )

        return new_kv_cache, self.o_proj(output)


class Gemma3DecoderLayer(nnx.Module):
    """Single Gemma3 transformer decoder layer."""

    def __init__(
        self,
        hf_config,
        layer_idx: int,
        dtype: jnp.dtype,
        rng: nnx.Rngs,
        mesh: Mesh,
        kv_cache_dtype: str,
    ):
        hidden_size = hf_config.hidden_size
        eps = hf_config.rms_norm_eps
        norm_init = nnx.with_partitioning(_init_fn, (None,))

        self.pre_attention_norm = nnx.RMSNorm(
            hidden_size,
            epsilon=eps,
            param_dtype=dtype,
            scale_init=norm_init,
            rngs=rng,
        )
        self.post_attention_norm = nnx.RMSNorm(
            hidden_size,
            epsilon=eps,
            param_dtype=dtype,
            scale_init=norm_init,
            rngs=rng,
        )
        self.pre_ffw_norm = nnx.RMSNorm(
            hidden_size,
            epsilon=eps,
            param_dtype=dtype,
            scale_init=norm_init,
            rngs=rng,
        )
        self.post_ffw_norm = nnx.RMSNorm(
            hidden_size,
            epsilon=eps,
            param_dtype=dtype,
            scale_init=norm_init,
            rngs=rng,
        )

        self.self_attn = Gemma3Attention(
            hf_config=hf_config,
            layer_idx=layer_idx,
            dtype=dtype,
            rng=rng,
            mesh=mesh,
            kv_cache_dtype=kv_cache_dtype,
        )
        self.mlp = Gemma3MLP(
            hidden_size=hidden_size,
            intermediate_size=hf_config.intermediate_size,
            dtype=dtype,
            rng=rng,
        )

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: "AttentionMetadata",
    ) -> Tuple[Optional[jax.Array], jax.Array]:
        # Self-attention block
        residual = x
        x = self.pre_attention_norm(x)
        kv_cache, attn_out = self.self_attn(kv_cache, x, attention_metadata)
        attn_out = self.post_attention_norm(attn_out)
        x = residual + attn_out

        # Feed-forward block
        residual = x
        x = self.pre_ffw_norm(x)
        x = self.mlp(x)
        x = self.post_ffw_norm(x)
        x = residual + x

        return kv_cache, x


class Gemma3Model(nnx.Module):
    """Gemma3 text transformer (embedding + N decoder layers + final norm)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        rng: nnx.Rngs,
        mesh: Mesh,
    ):
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        hidden_size = hf_config.hidden_size
        self.hidden_size = hidden_size
        tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", True)

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        if self.is_first_rank or (tie_word_embeddings and self.is_last_rank):
            self.embed = nnx.Embed(
                num_embeddings=vocab_size,
                features=hidden_size,
                param_dtype=dtype,
                embedding_init=nnx.with_partitioning(
                    _init_fn, (ShardingAxisName.VOCAB, None)
                ),
                rngs=rng,
            )
        else:
            self.embed = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            hf_config.num_hidden_layers,
            lambda i: Gemma3DecoderLayer(
                hf_config=hf_config,
                layer_idx=i,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
            ),
        )

        if self.is_last_rank:
            self.norm = nnx.RMSNorm(
                hidden_size,
                epsilon=hf_config.rms_norm_eps,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(_init_fn, (None,)),
                rngs=rng,
            )
            if tie_word_embeddings:
                self.lm_head = self.embed.embedding
            else:
                self.lm_head = nnx.Param(
                    _init_fn(rng.params(), (hidden_size, vocab_size), dtype),
                    sharding=(None, ShardingAxisName.VOCAB),
                )
        else:
            self.norm = PPMissingLayer()
            self.lm_head = PPMissingLayer()

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: "AttentionMetadata",
        intermediate_tensors: Optional["JaxIntermediateTensors"] = None,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        if self.is_first_rank:
            x = self.embed(input_ids)
            # Gemma3 scales embeddings by sqrt(hidden_size).
            x = x * jnp.sqrt(float(self.hidden_size)).astype(x.dtype)
        else:
            assert intermediate_tensors is not None
            x = intermediate_tensors["hidden_states"]

        for i, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            kv_cache = kv_caches[i]
            kv_cache, x = layer(kv_cache, x, attention_metadata)
            kv_caches[i] = kv_cache

        if not self.is_last_rank:
            return kv_caches, JaxIntermediateTensors({"hidden_states": x}), []

        return kv_caches, self.norm(x), []


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class Gemma3TextForCausalLM(nnx.Module):
    """Gemma3 text-only causal LM for tpu-inference JAX serving.

    Implements the interface expected by
    ``tpu_inference.models.common.model_loader.get_flax_model``:

    * ``__init__(vllm_config, rng_key, mesh)``
    * ``__call__(kv_caches, input_ids, attention_metadata, ...)``
    * ``compute_logits(hidden_states)``
    * ``load_weights(rng_key)``

    Register with tpu-inference once at startup:

        from keras_hub.src.utils.vllm.gemma3_jax import Gemma3TextForCausalLM
        from tpu_inference.models.common.model_loader import register_model
        register_model("Gemma3TextForCausalLM", Gemma3TextForCausalLM)
    """

    WeightLoader = StandardWeightLoader

    def __init__(
        self,
        vllm_config: VllmConfig,
        rng_key: jax.Array,
        mesh: Mesh,
    ):
        if not _HAS_TPU_INFERENCE:
            raise ImportError(
                "tpu_inference is required to use Gemma3TextForCausalLM. "
                "Install it from the tpu-inference repository."
            )
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = Gemma3Model(
            vllm_config=vllm_config,
            rng=self.rng,
            mesh=mesh,
        )

        # Collect PP-missing layer paths for weight-loading skip logic.
        self.pp_missing_layers = []
        for path, module in nnx.iter_graph(self.model):
            if isinstance(module, PPMissingLayer):
                self.pp_missing_layers.append(
                    ".".join(str(s) for s in path)
                )

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: "AttentionMetadata",
        _input_embeds=None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: Optional["JaxIntermediateTensors"] = None,
        _is_first_rank: Optional[bool] = None,
        _is_last_rank: Optional[bool] = None,
        *args,
    ) -> Tuple[
        List[jax.Array],
        jax.Array,
        List[jax.Array],
        Optional[jax.Array],
    ]:
        kv_caches, hidden_states, aux = self.model(
            kv_caches, input_ids, attention_metadata, intermediate_tensors
        )
        return kv_caches, hidden_states, aux, None

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        hf_config = self.vllm_config.model_config.hf_config
        if getattr(hf_config, "tie_word_embeddings", True):
            return jnp.dot(hidden_states, self.model.lm_head.value.T)
        return jnp.dot(hidden_states, self.model.lm_head.value)

    def load_weights(self, rng_key: jax.Array):
        self.rng = nnx.Rngs(rng_key)

        # HF key (without ".weight" suffix) → nnx state path
        mappings = {
            # Token embedding
            "model.embed_tokens": "model.embed.embedding",
            # Per-layer norms
            "model.layers.*.input_layernorm":
                "model.layers.*.pre_attention_norm.scale",
            "model.layers.*.post_attention_layernorm":
                "model.layers.*.post_attention_norm.scale",
            "model.layers.*.pre_feedforward_layernorm":
                "model.layers.*.pre_ffw_norm.scale",
            "model.layers.*.post_feedforward_layernorm":
                "model.layers.*.post_ffw_norm.scale",
            # Attention projections
            "model.layers.*.self_attn.q_proj":
                "model.layers.*.self_attn.q_proj.kernel",
            "model.layers.*.self_attn.k_proj":
                "model.layers.*.self_attn.k_proj.kernel",
            "model.layers.*.self_attn.v_proj":
                "model.layers.*.self_attn.v_proj.kernel",
            "model.layers.*.self_attn.o_proj":
                "model.layers.*.self_attn.o_proj.kernel",
            # Query / key per-head norms
            "model.layers.*.self_attn.q_norm":
                "model.layers.*.self_attn.q_norm.scale",
            "model.layers.*.self_attn.k_norm":
                "model.layers.*.self_attn.k_norm.scale",
            # MLP
            "model.layers.*.mlp.gate_proj":
                "model.layers.*.mlp.gate_proj.kernel",
            "model.layers.*.mlp.up_proj":
                "model.layers.*.mlp.up_proj.kernel",
            "model.layers.*.mlp.down_proj":
                "model.layers.*.mlp.down_proj.kernel",
            # Final normalization
            "model.norm": "model.norm.scale",
        }

        hf_config = self.vllm_config.model_config.hf_config
        if not getattr(hf_config, "tie_word_embeddings", True):
            mappings["lm_head"] = "model.lm_head"

        loader = self.WeightLoader(self.vllm_config, self.mesh)
        loader.load_weights(self, mappings)
