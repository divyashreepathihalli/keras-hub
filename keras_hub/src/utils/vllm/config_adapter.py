"""Adapter to extract vLLM-compatible configuration from KerasHub backbones.

vLLM requires model configuration metadata (vocab size, number of layers,
head dimensions, etc.) to set up its scheduler, memory manager and attention
backend.  This module extracts that information from a KerasHub backbone
instance so it can be passed to vLLM without manual specification.
"""

from keras_hub.src.models.rwkv7.rwkv7_backbone import RWKV7Backbone

# Attributes to probe on the backbone, in priority order.  Different model
# families use slightly different attribute names (e.g. ``num_query_heads``
# vs ``num_heads``).
_HEAD_DIM_ATTRS = ("head_dim",)
_NUM_HEADS_ATTRS = ("num_query_heads", "num_heads", "num_attention_heads")
_NUM_KV_HEADS_ATTRS = ("num_key_value_heads",)
_HIDDEN_DIM_ATTRS = ("hidden_dim", "hidden_size")
_INTERMEDIATE_DIM_ATTRS = ("intermediate_dim", "intermediate_size")
_NUM_LAYERS_ATTRS = ("num_layers",)
_VOCAB_SIZE_ATTRS = ("vocabulary_size", "vocab_size")


def _get_attr(backbone, names, default=None):
    """Return the first matching attribute from *names*."""
    for name in names:
        val = getattr(backbone, name, None)
        if val is not None:
            return val
    return default


def get_vllm_config(backbone):
    """Extract a vLLM-friendly configuration dict from a KerasHub backbone.

    The returned dictionary contains the model architecture metadata that
    vLLM needs to configure memory allocation, KV cache sizing and the
    attention backend.

    Args:
        backbone: A ``keras_hub.models.Backbone`` instance.

    Returns:
        A dict with keys:
        - ``vocab_size``
        - ``num_layers``
        - ``num_heads``
        - ``num_kv_heads``
        - ``head_dim``
        - ``hidden_dim``
        - ``intermediate_dim``
        - ``max_sequence_length``
        - ``dtype``
        - ``model_type``: one of ``"transformer"``, ``"rwkv"``
    """
    is_rwkv = isinstance(backbone, RWKV7Backbone)

    vocab_size = _get_attr(backbone, _VOCAB_SIZE_ATTRS)
    num_layers = _get_attr(backbone, _NUM_LAYERS_ATTRS)
    hidden_dim = _get_attr(backbone, _HIDDEN_DIM_ATTRS)
    intermediate_dim = _get_attr(backbone, _INTERMEDIATE_DIM_ATTRS)

    if is_rwkv:
        head_dim = getattr(backbone, "head_size", None)
        num_heads = (
            hidden_dim // head_dim if hidden_dim and head_dim else None
        )
        num_kv_heads = num_heads
    else:
        num_heads = _get_attr(backbone, _NUM_HEADS_ATTRS)
        num_kv_heads = _get_attr(backbone, _NUM_KV_HEADS_ATTRS, num_heads)
        head_dim = _get_attr(backbone, _HEAD_DIM_ATTRS)
        if head_dim is None and hidden_dim and num_heads:
            head_dim = hidden_dim // num_heads

    max_seq_len = getattr(backbone, "max_sequence_length", None)

    dtype = str(backbone.dtype_policy.compute_dtype)

    return {
        "vocab_size": vocab_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "max_sequence_length": max_seq_len,
        "dtype": dtype,
        "model_type": "rwkv" if is_rwkv else "transformer",
    }
