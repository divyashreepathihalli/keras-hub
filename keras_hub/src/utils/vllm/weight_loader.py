"""Load KerasHub preset weights directly into a tpu-inference nnx.Module."""

import numpy as np


def _get_keras_backbone(preset_name: str):
    """Load a Gemma backbone from a preset with weights."""
    from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone

    return GemmaBackbone.from_preset(preset_name, load_weights=True)


def _copy_array(src_np: np.ndarray, dst_param, mesh=None):
    """Copy a numpy array into an nnx.Param, optionally sharding on mesh."""
    try:
        import jax
        import jax.numpy as jnp
        from flax import nnx
    except ImportError as e:
        raise ImportError("flax and jax are required for weight loading") from e

    arr = jnp.array(src_np, dtype=dst_param.value.dtype)
    if mesh is not None:
        # Simple 1-D model-parallel sharding along axis 0 when possible.
        arr = jax.device_put(arr, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()))
    dst_param.value = arr


def load_preset_weights(model, preset_name: str, mesh=None):
    """Copy weights from a KerasHub preset into the vLLM nnx model.

    Variable name mapping (KerasHub → PagedGemmaAttention):
      backbone.token_embedding.embeddings → model.token_embedding.embedding
      backbone.transformer_layers[i].attention.query_dense.kernel → layers[i].attention.q_kernel
      backbone.transformer_layers[i].attention.key_dense.kernel   → layers[i].attention.k_kernel
      backbone.transformer_layers[i].attention.value_dense.kernel → layers[i].attention.v_kernel
      backbone.transformer_layers[i].attention.output_dense.kernel → layers[i].attention.o_kernel
      backbone.transformer_layers[i].pre_attention_norm.scale     → layers[i].pre_attention_norm
      backbone.transformer_layers[i].pre_ffw_norm.scale           → layers[i].pre_ffw_norm
      backbone.transformer_layers[i].{gate,ffw1,ffw2}_dense.kernel → layers[i].{gate,ffw1,ffw2}
      backbone.layer_norm.scale                                   → model.final_norm
    """
    backbone = _get_keras_backbone(preset_name)

    # Token embedding
    emb_weights = backbone.token_embedding.get_weights()
    _copy_array(emb_weights[0], model.token_embedding.embedding, mesh)

    for i, kb_layer in enumerate(backbone.transformer_layers):
        vllm_layer = model.layers[i]

        # Attention projections
        attn = vllm_layer.attention
        q_kernel = kb_layer.attention.query_dense.kernel.numpy()  # [nq, hid, hd]
        k_kernel = kb_layer.attention.key_dense.kernel.numpy()    # [nkv, hid, hd]
        v_kernel = kb_layer.attention.value_dense.kernel.numpy()  # [nkv, hid, hd]
        o_kernel = kb_layer.attention.output_dense.kernel.numpy() # [nq, hd, hid]
        _copy_array(q_kernel, attn.q_kernel, mesh)
        _copy_array(k_kernel, attn.k_kernel, mesh)
        _copy_array(v_kernel, attn.v_kernel, mesh)
        _copy_array(o_kernel, attn.o_kernel, mesh)

        # Layer norms
        _copy_array(
            kb_layer.pre_attention_norm.scale.numpy(),
            vllm_layer.pre_attention_norm,
            mesh,
        )
        _copy_array(
            kb_layer.pre_ffw_norm.scale.numpy(),
            vllm_layer.pre_ffw_norm,
            mesh,
        )
        if hasattr(kb_layer, "post_attention_norm") and hasattr(
            vllm_layer, "post_attention_norm"
        ):
            _copy_array(
                kb_layer.post_attention_norm.scale.numpy(),
                vllm_layer.post_attention_norm,
                mesh,
            )
        if hasattr(kb_layer, "post_ffw_norm") and hasattr(
            vllm_layer, "post_ffw_norm"
        ):
            _copy_array(
                kb_layer.post_ffw_norm.scale.numpy(),
                vllm_layer.post_ffw_norm,
                mesh,
            )

        # MLP (Gemma uses SwiGLU: gating_ffw, gating_ffw_2, ffw_linear)
        _copy_array(
            kb_layer.gating_ffw.kernel.numpy(), vllm_layer.gate_kernel, mesh
        )
        _copy_array(
            kb_layer.gating_ffw_2.kernel.numpy(), vllm_layer.ffw1_kernel, mesh
        )
        _copy_array(
            kb_layer.ffw_linear.kernel.numpy(), vllm_layer.ffw2_kernel, mesh
        )

    # Final norm
    _copy_array(backbone.layer_norm.scale.numpy(), model.final_norm_scale, mesh)
