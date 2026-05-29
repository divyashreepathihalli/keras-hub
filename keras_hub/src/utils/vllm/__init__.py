"""vLLM integration for KerasHub — TPU/JAX serving via tpu-inference."""

from keras_hub.src.utils.vllm.config_parser import KerasHubConfigParser
from keras_hub.src.utils.vllm.keras_hub_for_causal_lm import (
    KerasHubGemmaForCausalLM,
)


def register():
    """Register KerasHub model classes with tpu-inference.

    Called automatically as a vllm.general_plugins entry point when
    keras-hub[tpu] is installed.
    """
    try:
        from tpu_inference.models.common.model_loader import register_model
    except ImportError as e:
        raise ImportError(
            "tpu-inference is required for vLLM/TPU serving. "
            "Install it with: pip install keras-hub[tpu]"
        ) from e

    register_model("GemmaForCausalLM", KerasHubGemmaForCausalLM)
