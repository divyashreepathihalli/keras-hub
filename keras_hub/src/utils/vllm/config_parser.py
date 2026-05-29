"""KerasHub config parser — builds vLLM model config from a preset."""

import types


def _preset_to_dims(preset_name: str) -> dict:
    """Load a KerasHub preset and extract model dimensions."""
    from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone

    backbone = GemmaBackbone.from_preset(preset_name, load_weights=False)
    cfg = backbone.get_config()
    return {
        "vocabulary_size": cfg["vocabulary_size"],
        "num_layers": cfg["num_layers"],
        "num_query_heads": cfg["num_query_heads"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "hidden_dim": cfg["hidden_dim"],
        "intermediate_dim": cfg["intermediate_dim"],
        "head_dim": cfg["head_dim"],
        "layer_norm_epsilon": cfg.get("layer_norm_epsilon", 1e-6),
        "query_head_dim_normalize": cfg.get("query_head_dim_normalize", True),
        "use_post_ffw_norm": cfg.get("use_post_ffw_norm", False),
        "use_post_attention_norm": cfg.get("use_post_attention_norm", False),
        "attention_logit_soft_cap": cfg.get("attention_logit_soft_cap"),
        "final_logit_soft_cap": cfg.get("final_logit_soft_cap"),
        "use_sliding_window_attention": cfg.get(
            "use_sliding_window_attention", False
        ),
        "sliding_window_size": cfg.get("sliding_window_size", 4096),
    }


class KerasHubConfigParser:
    """Parse a KerasHub preset name into a vLLM-compatible model config.

    Builds an in-memory config object from preset metadata so vLLM core
    can size the paged KV cache before the model is instantiated.
    No config.json or HF Hub fetch is required.
    """

    @classmethod
    def from_preset(cls, preset_name: str, dtype: str = "bfloat16"):
        dims = _preset_to_dims(preset_name)
        hf_config = types.SimpleNamespace(
            architectures=["GemmaForCausalLM"],
            vocab_size=dims["vocabulary_size"],
            num_hidden_layers=dims["num_layers"],
            num_attention_heads=dims["num_query_heads"],
            num_key_value_heads=dims["num_key_value_heads"],
            hidden_size=dims["hidden_dim"],
            intermediate_size=dims["intermediate_dim"],
            head_dim=dims["head_dim"],
            rms_norm_eps=dims["layer_norm_epsilon"],
            max_position_embeddings=8192,
        )
        return hf_config, dims
