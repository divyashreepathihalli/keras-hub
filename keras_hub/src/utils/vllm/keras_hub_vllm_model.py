"""Generic vLLM model wrapper for KerasHub CausalLM models.

This module provides ``KerasHubVLLMModel``, a wrapper that adapts any
KerasHub ``CausalLM`` model to the interface expected by vLLM's JAX/TPU
model runner.  The wrapper:

* Converts between vLLM's per-layer KV cache list and KerasHub's stacked
  cache tensor.
* Manages JAX ``StatelessScope`` for pure-functional execution inside
  ``jax.jit``.
* Dispatches to the correct code path for standard transformer models,
  multimodal models (Gemma3, PaliGemma) and RWKV7.

Usage::

    import keras_hub

    model = keras_hub.models.LlamaCausalLM.from_preset("llama3_8b_en")
    vllm_model = KerasHubVLLMModel(model)

    # vLLM model runner calls:
    updated_caches, hidden_states, _ = vllm_model(
        kv_caches, input_ids, attention_metadata
    )
    logits = vllm_model.compute_logits(hidden_states)
"""

import keras
from keras import ops

from keras_hub.src.utils.vllm.cache_adapter import kerashub_to_vllm_cache
from keras_hub.src.utils.vllm.cache_adapter import rwkv7_to_vllm_cache
from keras_hub.src.utils.vllm.cache_adapter import vllm_to_kerashub_cache
from keras_hub.src.utils.vllm.cache_adapter import vllm_to_rwkv7_cache
from keras_hub.src.utils.vllm.config_adapter import get_vllm_config

# Model type constants.
_STANDARD = "standard"
_MULTIMODAL = "multimodal"
_RWKV = "rwkv"

# Classes whose ``call_with_cache`` accepts extra vision arguments.
_MULTIMODAL_CLASS_NAMES = frozenset(
    {
        "Gemma3CausalLM",
        "PaliGemmaCausalLM",
    }
)

_RWKV_CLASS_NAMES = frozenset(
    {
        "RWKV7CausalLM",
    }
)


class KerasHubVLLMModel:
    """Wraps a KerasHub ``CausalLM`` for serving with vLLM's JAX model runner.

    This class adapts the KerasHub ``call_with_cache`` / ``_build_cache``
    interface to the forward-pass signature that vLLM's JAX/TPU worker
    expects.  It supports three categories of models:

    * **Standard** – All transformer-based ``CausalLM`` models whose
      ``call_with_cache(token_ids, cache, cache_update_index)`` returns
      ``(logits, hidden_states, cache)``.
    * **Multimodal** – ``Gemma3CausalLM`` and ``PaliGemmaCausalLM`` which
      accept additional vision inputs (image embeddings, vision masks).
    * **RWKV** – ``RWKV7CausalLM`` which uses an RNN-style state cache
      instead of standard KV attention caches.

    Args:
        keras_model: A ``keras_hub.models.CausalLM`` instance.  The model
            must have a ``backbone`` attribute and implement
            ``call_with_cache`` and ``_build_cache``.
    """

    def __init__(self, keras_model):
        self.model = keras_model
        self.backbone = keras_model.backbone
        self.config = get_vllm_config(self.backbone)
        self._model_type = self._detect_model_type()

    # ------------------------------------------------------------------
    # Model type detection
    # ------------------------------------------------------------------

    def _detect_model_type(self):
        cls_name = type(self.model).__name__
        if cls_name in _RWKV_CLASS_NAMES:
            return _RWKV
        if cls_name in _MULTIMODAL_CLASS_NAMES:
            return _MULTIMODAL
        return _STANDARD

    @property
    def is_multimodal(self):
        return self._model_type == _MULTIMODAL

    @property
    def is_rwkv(self):
        return self._model_type == _RWKV

    # ------------------------------------------------------------------
    # vLLM model runner interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        kv_caches,
        input_ids,
        attention_metadata,
    ):
        """Run the forward pass expected by vLLM's JAX model runner.

        Args:
            kv_caches: Per-layer KV caches.
                * For standard/multimodal models: a list of arrays each of
                  shape ``[batch, 2, max_len, num_kv_heads, head_dim]``.
                * For RWKV: a tuple ``(state_list, last_token_list)`` of
                  per-layer lists.
            input_ids: Integer token ids, shape ``[batch, seq_len]``.
            attention_metadata: An object carrying scheduling metadata from
                vLLM.  At minimum it should expose:
                * ``cache_update_index`` (int or int tensor) – position in
                  the sequence to write the new KV entry.
                * ``is_prefill`` (bool) – whether this is a prefill pass.
                Multimodal models additionally read:
                * ``img_embeddings`` – pre-computed image embeddings.
                * ``vision_mask``, ``vision_indices``, ``padding_mask``.

        Returns:
            A tuple ``(updated_kv_caches, hidden_states, aux)`` where
            ``aux`` is an empty list (reserved for future use).
        """
        if self.is_rwkv:
            return self._forward_rwkv(kv_caches, input_ids, attention_metadata)
        if self.is_multimodal:
            return self._forward_multimodal(
                kv_caches, input_ids, attention_metadata
            )
        return self._forward_standard(
            kv_caches, input_ids, attention_metadata
        )

    def compute_logits(self, hidden_states):
        """Compute token logits from the final hidden states.

        Args:
            hidden_states: Float tensor of shape
                ``[batch, seq_len, hidden_dim]``.

        Returns:
            Logits tensor of shape ``[batch, seq_len, vocab_size]``.
        """
        return self.backbone.token_embedding(hidden_states, reverse=True)

    # ------------------------------------------------------------------
    # Cache initialisation helpers
    # ------------------------------------------------------------------

    def build_cache(self, token_ids, **kwargs):
        """Build and seed a KV cache for the given token ids.

        This delegates to the underlying model's ``_build_cache`` and
        returns caches in vLLM's per-layer list format.

        Args:
            token_ids: Integer token ids, shape ``[batch, seq_len]``.
            **kwargs: Extra arguments forwarded to ``_build_cache``
                (e.g. ``padding_mask`` for RWKV, vision inputs for
                multimodal models).

        Returns:
            A tuple ``(hidden_states, kv_caches)`` where ``kv_caches`` is
            in vLLM format (per-layer list).
        """
        if self.is_rwkv:
            padding_mask = kwargs.get("padding_mask")
            hidden_states, cache = self.model._build_cache(
                token_ids, padding_mask
            )
            state_list, last_token_list = rwkv7_to_vllm_cache(cache)
            return hidden_states, (state_list, last_token_list)

        if self.is_multimodal:
            hidden_states, cache = self.model._build_cache(
                token_ids, **kwargs
            )
            return hidden_states, kerashub_to_vllm_cache(cache)

        hidden_states, cache = self.model._build_cache(token_ids)
        return hidden_states, kerashub_to_vllm_cache(cache)

    def build_empty_cache(self, batch_size, max_length):
        """Create a zero-initialised KV cache in vLLM format.

        Useful when vLLM allocates cache memory upfront.

        Args:
            batch_size: int. Batch size.
            max_length: int. Maximum sequence length.

        Returns:
            A list of zero arrays (one per layer), each of shape
            ``[batch, 2, max_length, num_kv_heads, head_dim]``.
        """
        cfg = self.config
        num_layers = cfg["num_layers"]
        num_kv_heads = cfg["num_kv_heads"]
        head_dim = cfg["head_dim"]
        dtype = cfg["dtype"]

        layer_cache_shape = [batch_size, 2, max_length, num_kv_heads, head_dim]
        return [
            ops.zeros(layer_cache_shape, dtype=dtype)
            for _ in range(num_layers)
        ]

    # ------------------------------------------------------------------
    # Forward-pass implementations
    # ------------------------------------------------------------------

    def _forward_standard(self, kv_caches, input_ids, attention_metadata):
        """Forward pass for standard transformer CausalLM models."""
        cache = vllm_to_kerashub_cache(kv_caches)
        cache_update_index = attention_metadata.cache_update_index

        logits, hidden_states, updated_cache = self.model.call_with_cache(
            input_ids, cache, cache_update_index
        )
        return kerashub_to_vllm_cache(updated_cache), hidden_states, []

    def _forward_multimodal(self, kv_caches, input_ids, attention_metadata):
        """Forward pass for multimodal CausalLM models (Gemma3, PaliGemma)."""
        cache = vllm_to_kerashub_cache(kv_caches)
        cache_update_index = attention_metadata.cache_update_index

        extra_kwargs = {}
        for attr in (
            "img_embeddings",
            "vision_mask",
            "padding_mask",
            "vision_indices",
            "cache_update_mask",
        ):
            val = getattr(attention_metadata, attr, None)
            if val is not None:
                extra_kwargs[attr] = val

        logits, hidden_states, updated_cache = self.model.call_with_cache(
            token_ids=input_ids,
            cache=cache,
            cache_update_index=cache_update_index,
            **extra_kwargs,
        )
        return kerashub_to_vllm_cache(updated_cache), hidden_states, []

    def _forward_rwkv(self, kv_caches, input_ids, attention_metadata):
        """Forward pass for RWKV7 RNN-based CausalLM."""
        state_list, last_token_list = kv_caches
        cache = vllm_to_rwkv7_cache(state_list, last_token_list)

        is_prefill = getattr(attention_metadata, "is_prefill", False)

        logits, hidden_states, updated_cache = self.model.call_with_cache(
            token_ids=input_ids,
            cache=cache,
            compute_head=True,
            padding_mask=getattr(attention_metadata, "padding_mask", None),
            rnn_mode=not is_prefill,
        )
        updated_state, updated_tokens = rwkv7_to_vllm_cache(updated_cache)
        return (updated_state, updated_tokens), hidden_states, []

    # ------------------------------------------------------------------
    # Weight access (for vLLM weight loading interface)
    # ------------------------------------------------------------------

    @property
    def trainable_variables(self):
        """Expose model trainable variables for vLLM's weight loader."""
        return self.model.trainable_variables

    @property
    def non_trainable_variables(self):
        """Expose model non-trainable variables."""
        return self.model.non_trainable_variables

    @property
    def variables(self):
        """All model variables."""
        return self.model.variables

    # ------------------------------------------------------------------
    # JAX JIT compilation helper
    # ------------------------------------------------------------------

    def make_jit_forward(self):
        """Return a ``jax.jit``-compiled version of :meth:`__call__`.

        Uses ``keras.StatelessScope`` to make the forward pass
        pure-functional, matching KerasHub's own JAX generation pattern.

        Returns:
            A function ``(kv_caches, input_ids, attention_metadata, state)``
            that returns ``(updated_kv_caches, hidden_states, aux)``.
            ``state`` is a tuple of variable value lists.
        """
        if keras.config.backend() != "jax":
            raise RuntimeError(
                "make_jit_forward() requires the JAX backend. "
                f"Current backend: {keras.config.backend()}"
            )

        import itertools
        from functools import partial

        import jax

        @partial(jax.jit, static_argnames=["attention_metadata"])
        def jit_forward(kv_caches, input_ids, attention_metadata, state):
            trainable_variables, non_trainable_variables = state
            mapping = itertools.chain(
                zip(self.model.trainable_variables, trainable_variables),
                zip(
                    self.model.non_trainable_variables,
                    non_trainable_variables,
                ),
            )
            with keras.StatelessScope(state_mapping=mapping):
                return self(kv_caches, input_ids, attention_metadata)

        def wrapped(kv_caches, input_ids, attention_metadata):
            state = (
                [v.value for v in self.model.trainable_variables],
                [v.value for v in self.model.non_trainable_variables],
            )
            return jit_forward(kv_caches, input_ids, attention_metadata, state)

        return wrapped
