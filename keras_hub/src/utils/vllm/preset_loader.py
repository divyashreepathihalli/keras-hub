"""Load a KerasHub preset and wrap it for vLLM serving.

Example::

    from keras_hub.src.utils.vllm.preset_loader import load_from_preset

    vllm_model = load_from_preset("gemma_2b_en")
"""

from keras_hub.src.utils.vllm.keras_hub_vllm_model import KerasHubVLLMModel


def load_from_preset(preset_name, dtype="bfloat16", **kwargs):
    """Load a KerasHub CausalLM preset and wrap for vLLM.

    This is a convenience alias for
    ``KerasHubVLLMModel.from_preset()``.

    Args:
        preset_name: str. A KerasHub preset name or local path.
        dtype: str. Compute dtype. Defaults to ``"bfloat16"``.
        **kwargs: Additional arguments passed to
            ``CausalLM.from_preset()``.

    Returns:
        A ``KerasHubVLLMModel`` instance.
    """
    return KerasHubVLLMModel.from_preset(preset_name, dtype=dtype, **kwargs)
