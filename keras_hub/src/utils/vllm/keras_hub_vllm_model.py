"""vLLM-compatible model adapter for KerasHub CausalLM models.

This module provides ``KerasHubVLLMModel``, an ``nnx.Module`` subclass that
adapts any KerasHub ``CausalLM`` model to the interface expected by vLLM's
JAX/TPU model runner.

When ``KERAS_NNX_ENABLED=true``, all KerasHub models become ``nnx.Module``
instances. This adapter wraps a KerasHub ``CausalLM`` and exposes the exact
``__call__`` and ``compute_logits`` signatures that vLLM's JAX model runner
expects.

Usage with vLLM (on a GPU/TPU instance)::

    from vllm import LLM, SamplingParams

    # KerasHub registers via plugin — vLLM discovers the model
    llm = LLM(model="keras_hub:gemma_2b_en")
    outputs = llm.generate("Hello world", SamplingParams(max_tokens=64))

Direct usage::

    import os
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["KERAS_NNX_ENABLED"] = "true"

    import keras_hub
    from keras_hub.src.utils.vllm.keras_hub_vllm_model import (
        KerasHubVLLMModel,
    )

    model = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")
    vllm_model = KerasHubVLLMModel.from_preset("gemma_2b_en")
"""

import os

from keras_hub.src.utils.vllm.cache_adapter import kerashub_to_vllm_cache
from keras_hub.src.utils.vllm.cache_adapter import vllm_to_kerashub_cache
from keras_hub.src.utils.vllm.config_adapter import get_vllm_config


def _ensure_jax_nnx():
    """Ensure JAX backend and NNX mode are enabled."""
    backend = os.environ.get("KERAS_BACKEND", "")
    if backend and backend != "jax":
        raise RuntimeError(
            f"vLLM integration requires KERAS_BACKEND='jax', "
            f"got '{backend}'."
        )
    os.environ.setdefault("KERAS_BACKEND", "jax")

    nnx_enabled = os.environ.get("KERAS_NNX_ENABLED", "")
    if not nnx_enabled:
        os.environ["KERAS_NNX_ENABLED"] = "true"


class KerasHubVLLMModel:
    """Adapts a KerasHub ``CausalLM`` for vLLM's JAX/TPU model runner.

    This class wraps a KerasHub ``CausalLM`` (which is already an
    ``nnx.Module`` when ``KERAS_NNX_ENABLED=true``) and provides the
    exact interface that vLLM's JAX model runner calls during inference:

    * ``__call__(kv_caches, input_ids, attention_metadata)``
      → ``(updated_kv_caches, hidden_states, [])``
    * ``compute_logits(hidden_states)`` → logits

    The adapter handles:

    * KV cache format translation (vLLM per-layer list ↔ KerasHub
      stacked tensor).
    * Mapping ``attention_metadata.cache_update_index`` to KerasHub's
      ``call_with_cache`` signature.

    Args:
        keras_model: A ``keras_hub.models.CausalLM`` instance loaded with
            ``KERAS_NNX_ENABLED=true``.
    """

    def __init__(self, keras_model):
        from flax import nnx

        if not isinstance(keras_model, nnx.Module):
            raise TypeError(
                "KerasHubVLLMModel requires the model to be an nnx.Module. "
                "Set KERAS_NNX_ENABLED=true before importing Keras."
            )
        self.model = keras_model
        self.backbone = keras_model.backbone
        self.config = get_vllm_config(self.backbone)

    @classmethod
    def from_preset(cls, preset_name, dtype="bfloat16", **kwargs):
        """Load a KerasHub preset and wrap for vLLM.

        This handles environment setup, model loading, and wrapping in
        a single call.

        Args:
            preset_name: str. A KerasHub preset name (e.g.
                ``"gemma_2b_en"``) or a path to a local preset directory.
            dtype: str. The compute dtype. Defaults to ``"bfloat16"``.
            **kwargs: Additional arguments passed to
                ``CausalLM.from_preset()``.

        Returns:
            A ``KerasHubVLLMModel`` instance ready for vLLM.
        """
        _ensure_jax_nnx()

        import keras_hub

        model = keras_hub.models.CausalLM.from_preset(
            preset_name, dtype=dtype, **kwargs
        )
        return cls(model)

    @classmethod
    def from_vllm_config(cls, vllm_config):
        """Construct from a vLLM ``VllmConfig`` object.

        This is the constructor vLLM's model loader calls after plugin
        registration.  It extracts the model identifier from
        ``vllm_config`` and loads the corresponding KerasHub preset.

        Args:
            vllm_config: A ``vllm.config.VllmConfig`` instance.

        Returns:
            A ``KerasHubVLLMModel`` instance.
        """
        _ensure_jax_nnx()

        import keras_hub

        model_id = vllm_config.model_config.model
        # Support "keras_hub:<preset>" naming convention.
        if model_id.startswith("keras_hub:"):
            model_id = model_id[len("keras_hub:"):]

        dtype = getattr(vllm_config.model_config, "dtype", "bfloat16")
        if dtype == "auto":
            dtype = "bfloat16"

        model = keras_hub.models.CausalLM.from_preset(
            model_id, dtype=str(dtype)
        )
        return cls(model)

    # ------------------------------------------------------------------
    # vLLM model runner interface
    # ------------------------------------------------------------------

    def __call__(self, kv_caches, input_ids, attention_metadata):
        """Forward pass for vLLM's JAX model runner.

        This is the method vLLM calls on every inference step. It
        translates between vLLM's per-layer KV cache format and
        KerasHub's stacked cache format, then delegates to
        ``call_with_cache``.

        Args:
            kv_caches: List of per-layer KV cache arrays, each of shape
                ``[batch, 2, max_len, num_kv_heads, head_dim]``.
            input_ids: Integer token ids, shape ``[batch, seq_len]``.
            attention_metadata: vLLM's ``AttentionMetadata`` object.
                Must expose ``cache_update_index``.

        Returns:
            Tuple of ``(updated_kv_caches, hidden_states, [])`` where
            ``updated_kv_caches`` is a list of per-layer arrays and
            ``hidden_states`` has shape ``[batch, seq_len, hidden_dim]``.
        """
        # Convert vLLM per-layer list → KerasHub stacked tensor.
        cache = vllm_to_kerashub_cache(kv_caches)
        cache_update_index = attention_metadata.cache_update_index

        # KerasHub call_with_cache returns (logits, hidden_states, cache).
        _, hidden_states, updated_cache = self.model.call_with_cache(
            input_ids, cache, cache_update_index
        )

        # Convert back to vLLM per-layer list.
        return kerashub_to_vllm_cache(updated_cache), hidden_states, []

    def compute_logits(self, hidden_states):
        """Project hidden states to vocabulary logits.

        This is the second method vLLM calls after ``__call__`` to get
        the token probability distribution.

        Args:
            hidden_states: Float array of shape
                ``[batch, seq_len, hidden_dim]``.

        Returns:
            Logits array of shape ``[batch, seq_len, vocab_size]``.
        """
        return self.backbone.token_embedding(hidden_states, reverse=True)

    # ------------------------------------------------------------------
    # Weight loading (for vLLM's weight loader)
    # ------------------------------------------------------------------

    def load_weights(self, *args, **kwargs):
        """Weight loading hook for vLLM.

        KerasHub models load their own weights via ``from_preset()``,
        so this is a no-op. Weights are already loaded when the model
        is constructed.
        """
        pass

    # ------------------------------------------------------------------
    # NNX Module interface passthrough
    # ------------------------------------------------------------------

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    @property
    def non_trainable_variables(self):
        return self.model.non_trainable_variables

    @property
    def variables(self):
        return self.model.variables
