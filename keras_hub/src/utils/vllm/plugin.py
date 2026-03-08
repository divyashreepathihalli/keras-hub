"""vLLM plugin entry point for KerasHub model registration.

This module is referenced as a vLLM plugin via the
``[project.entry-points."vllm.general_plugins"]`` section in
``pyproject.toml``.  When vLLM starts, it discovers and calls
:func:`register` to make KerasHub models available in vLLM's model
registry.

After registration, users can serve a KerasHub model with::

    vllm serve keras-hub:<preset_name>

For example::

    vllm serve keras-hub:llama3_8b_en
"""


def register():
    """Register KerasHub models with vLLM's model registry.

    This function is called automatically by vLLM's plugin system
    during startup.  It registers a ``KerasHubModel`` model class
    that wraps any KerasHub ``CausalLM`` for serving.
    """
    try:
        from vllm import ModelRegistry

        ModelRegistry.register_model(
            "KerasHubModel",
            "keras_hub.src.utils.vllm.keras_hub_vllm_model:"
            "KerasHubVLLMModel",
        )
    except ImportError:
        # vLLM is not installed; skip registration silently.
        pass
