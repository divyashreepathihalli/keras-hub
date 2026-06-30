"""vLLM/tpu-inference integration utilities for KerasHub models.

This package provides native JAX/Flax nnx model implementations that plug
into tpu-inference's vLLM-compatible JAX serving infrastructure. Models
implemented here can be registered with tpu-inference via:

    from tpu_inference.models.common.model_loader import register_model
    from keras_hub.src.utils.vllm.gemma3_jax import Gemma3TextForCausalLM
    register_model("Gemma3TextForCausalLM", Gemma3TextForCausalLM)

Each model in this package soft-imports tpu_inference — the module is
importable without it installed, but functional inference requires it.
"""
