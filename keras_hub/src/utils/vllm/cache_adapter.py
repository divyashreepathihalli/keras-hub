"""Cache format conversion between vLLM and KerasHub.

vLLM JAX/TPU model runner passes KV caches as a List[jax.Array], one per
layer. KerasHub CausalLM models use a single stacked tensor of shape
[batch, num_layers, 2, max_len, num_kv_heads, head_dim].

This module provides bidirectional conversion between these formats, as well
as adapters for non-standard cache structures (RWKV7 RNN state).
"""

from keras import ops

# ---------------------------------------------------------------------------
# Standard transformer KV cache adapters
# ---------------------------------------------------------------------------


def vllm_to_kerashub_cache(kv_caches):
    """Convert vLLM per-layer KV cache list to KerasHub stacked cache.

    Args:
        kv_caches: List of arrays, one per layer. Each array has shape
            ``[batch, 2, max_len, num_kv_heads, head_dim]``.

    Returns:
        A single array of shape
        ``[batch, num_layers, 2, max_len, num_kv_heads, head_dim]``.
    """
    return ops.stack(kv_caches, axis=1)


def kerashub_to_vllm_cache(cache):
    """Convert KerasHub stacked cache to vLLM per-layer KV cache list.

    Args:
        cache: Array of shape
            ``[batch, num_layers, 2, max_len, num_kv_heads, head_dim]``.

    Returns:
        List of arrays (one per layer), each of shape
        ``[batch, 2, max_len, num_kv_heads, head_dim]``.
    """
    num_layers = cache.shape[1]
    return [cache[:, i, ...] for i in range(num_layers)]


# ---------------------------------------------------------------------------
# RWKV7 state cache adapters
# ---------------------------------------------------------------------------


def vllm_to_rwkv7_cache(state_list, last_token_list):
    """Convert vLLM per-layer RWKV7 state to KerasHub format.

    Args:
        state_list: List of arrays, one per layer, each of shape
            ``[batch, num_heads, head_dim, head_dim]``.
        last_token_list: List of arrays, one per layer, each of shape
            ``[batch, 2, 1, hidden_size]``.

    Returns:
        A tuple ``(state_cache, last_token_cache)`` where:
        - state_cache: ``[batch, num_layers, num_heads, head_dim, head_dim]``
        - last_token_cache: ``[batch, num_layers, 2, 1, hidden_size]``
    """
    state_cache = ops.stack(state_list, axis=1)
    last_token_cache = ops.stack(last_token_list, axis=1)
    return [state_cache, last_token_cache]


def rwkv7_to_vllm_cache(cache):
    """Convert KerasHub RWKV7 cache to vLLM per-layer lists.

    Args:
        cache: A list ``[state_cache, last_token_cache]`` where:
        - state_cache: ``[batch, num_layers, num_heads, head_dim, head_dim]``
        - last_token_cache: ``[batch, num_layers, 2, 1, hidden_size]``

    Returns:
        A tuple ``(state_list, last_token_list)`` of per-layer lists.
    """
    state_cache, last_token_cache = cache
    num_layers = state_cache.shape[1]
    state_list = [state_cache[:, i, ...] for i in range(num_layers)]
    last_token_list = [
        last_token_cache[:, i, ...] for i in range(num_layers)
    ]
    return state_list, last_token_list
