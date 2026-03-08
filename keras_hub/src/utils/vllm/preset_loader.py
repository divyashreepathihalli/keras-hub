"""Load a KerasHub preset and wrap it for vLLM serving.

This module provides a convenience function to load a KerasHub CausalLM
preset with the JAX backend and return a ``KerasHubVLLMModel`` wrapper
ready for use with vLLM's model runner.

Example::

    from keras_hub.src.utils.vllm import load_from_preset

    vllm_model = load_from_preset("llama3_8b_en", dtype="bfloat16")
"""

import os


def load_from_preset(preset_name, dtype="bfloat16", **kwargs):
    """Load a KerasHub CausalLM preset and wrap for vLLM.

    This sets the Keras backend to JAX (required for vLLM TPU
    integration), loads the model from the given preset, and wraps it
    in a ``KerasHubVLLMModel``.

    Args:
        preset_name: str. A KerasHub preset name (e.g.
            ``"llama3_8b_en"``) or a path to a local preset directory.
        dtype: str. The dtype for model weights and computation.
            Defaults to ``"bfloat16"`` which is optimal for TPU.
        **kwargs: Additional keyword arguments passed to
            ``CausalLM.from_preset()``.

    Returns:
        A ``KerasHubVLLMModel`` instance.

    Raises:
        RuntimeError: If the Keras backend is not JAX.
    """
    # Ensure JAX backend is active.
    backend = os.environ.get("KERAS_BACKEND", "")
    if backend and backend != "jax":
        raise RuntimeError(
            f"vLLM integration requires the JAX backend, but "
            f"KERAS_BACKEND is set to '{backend}'. Please set "
            f"KERAS_BACKEND='jax' before importing Keras."
        )
    os.environ.setdefault("KERAS_BACKEND", "jax")

    import keras

    if keras.config.backend() != "jax":
        raise RuntimeError(
            f"vLLM integration requires the JAX backend. "
            f"Current backend: {keras.config.backend()}. "
            f"Set KERAS_BACKEND='jax' before importing Keras."
        )

    import keras_hub
    from keras_hub.src.utils.vllm.keras_hub_vllm_model import KerasHubVLLMModel

    model = keras_hub.models.CausalLM.from_preset(
        preset_name, dtype=dtype, **kwargs
    )
    return KerasHubVLLMModel(model)
