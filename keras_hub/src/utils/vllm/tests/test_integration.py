"""Integration tests for the vLLM wrapper.

These tests verify that the wrapper works end-to-end with multiple
model architectures.  They use small randomly initialised models
(not pretrained weights) to keep tests fast.
"""

import numpy as np
from keras import ops

from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_hub.src.models.gpt2.gpt2_causal_lm import GPT2CausalLM
from keras_hub.src.models.llama.llama_backbone import LlamaBackbone
from keras_hub.src.models.llama.llama_causal_lm import LlamaCausalLM
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.vllm.config_adapter import get_vllm_config
from keras_hub.src.utils.vllm.keras_hub_vllm_model import KerasHubVLLMModel


class _FakeAttentionMetadata:
    def __init__(self, cache_update_index, **kwargs):
        self.cache_update_index = cache_update_index
        for k, v in kwargs.items():
            setattr(self, k, v)


class LlamaVLLMIntegrationTest(TestCase):
    def test_llama_forward_pass(self):
        backbone = LlamaBackbone(
            vocabulary_size=128,
            num_layers=2,
            num_query_heads=4,
            hidden_dim=32,
            intermediate_dim=64,
            num_key_value_heads=2,
        )
        model = LlamaCausalLM(backbone=backbone, preprocessor=None)
        wrapper = KerasHubVLLMModel(model)

        # Verify config.
        cfg = wrapper.config
        self.assertEqual(cfg["vocab_size"], 128)
        self.assertEqual(cfg["num_layers"], 2)
        self.assertEqual(cfg["num_heads"], 4)
        self.assertEqual(cfg["num_kv_heads"], 2)
        self.assertEqual(cfg["head_dim"], 8)  # 32 / 4
        self.assertEqual(cfg["model_type"], "transformer")

        # Build cache and do a decode step.
        token_ids = ops.convert_to_tensor(
            np.array([[1, 2, 3, 0, 0, 0, 0, 0]], dtype="int32")
        )
        hidden_states, kv_caches = wrapper.build_cache(token_ids)

        new_token = ops.convert_to_tensor(np.array([[4]], dtype="int32"))
        metadata = _FakeAttentionMetadata(cache_update_index=2)
        updated_caches, hidden_states, aux = wrapper(
            kv_caches, new_token, metadata
        )

        self.assertEqual(len(updated_caches), 2)
        logits = wrapper.compute_logits(hidden_states)
        self.assertEqual(logits.shape[-1], 128)


class GPT2VLLMIntegrationTest(TestCase):
    def test_gpt2_forward_pass(self):
        backbone = GPT2Backbone(
            vocabulary_size=100,
            num_layers=2,
            num_heads=2,
            hidden_dim=32,
            intermediate_dim=64,
        )
        model = GPT2CausalLM(backbone=backbone, preprocessor=None)
        wrapper = KerasHubVLLMModel(model)

        token_ids = ops.convert_to_tensor(
            np.array([[5, 10, 15, 0, 0, 0]], dtype="int32")
        )
        hidden_states, kv_caches = wrapper.build_cache(token_ids)

        new_token = ops.convert_to_tensor(np.array([[20]], dtype="int32"))
        metadata = _FakeAttentionMetadata(cache_update_index=2)
        updated_caches, hidden_states, aux = wrapper(
            kv_caches, new_token, metadata
        )

        self.assertEqual(len(updated_caches), 2)
        logits = wrapper.compute_logits(hidden_states)
        self.assertEqual(logits.shape[-1], 100)


class ConfigAdapterTest(TestCase):
    def test_llama_config(self):
        backbone = LlamaBackbone(
            vocabulary_size=256,
            num_layers=4,
            num_query_heads=8,
            hidden_dim=64,
            intermediate_dim=128,
            num_key_value_heads=4,
        )
        cfg = get_vllm_config(backbone)
        self.assertEqual(cfg["vocab_size"], 256)
        self.assertEqual(cfg["num_layers"], 4)
        self.assertEqual(cfg["num_heads"], 8)
        self.assertEqual(cfg["num_kv_heads"], 4)
        self.assertEqual(cfg["head_dim"], 8)
        self.assertEqual(cfg["hidden_dim"], 64)
        self.assertEqual(cfg["intermediate_dim"], 128)
        self.assertEqual(cfg["model_type"], "transformer")

    def test_gpt2_config(self):
        backbone = GPT2Backbone(
            vocabulary_size=100,
            num_layers=3,
            num_heads=4,
            hidden_dim=64,
            intermediate_dim=256,
        )
        cfg = get_vllm_config(backbone)
        self.assertEqual(cfg["vocab_size"], 100)
        self.assertEqual(cfg["num_layers"], 3)
        self.assertEqual(cfg["num_heads"], 4)
        self.assertEqual(cfg["hidden_dim"], 64)
