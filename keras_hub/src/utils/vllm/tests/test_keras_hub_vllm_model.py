import numpy as np
from keras import ops

from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_hub.src.models.gpt2.gpt2_causal_lm import GPT2CausalLM
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.vllm.keras_hub_vllm_model import KerasHubVLLMModel


class _FakeAttentionMetadata:
    """Minimal attention metadata stub for testing."""

    def __init__(self, cache_update_index, is_prefill=False, **kwargs):
        self.cache_update_index = cache_update_index
        self.is_prefill = is_prefill
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_small_gpt2():
    """Create a tiny GPT2 model for testing."""
    backbone = GPT2Backbone(
        vocabulary_size=100,
        num_layers=2,
        num_heads=2,
        hidden_dim=32,
        intermediate_dim=64,
    )
    return GPT2CausalLM(backbone=backbone, preprocessor=None)


class KerasHubVLLMModelTest(TestCase):
    def test_init_standard_model(self):
        model = _make_small_gpt2()
        wrapper = KerasHubVLLMModel(model)

        self.assertFalse(wrapper.is_multimodal)
        self.assertFalse(wrapper.is_rwkv)
        self.assertEqual(wrapper.config["model_type"], "transformer")
        self.assertEqual(wrapper.config["vocab_size"], 100)
        self.assertEqual(wrapper.config["num_layers"], 2)
        self.assertEqual(wrapper.config["num_heads"], 2)
        self.assertEqual(wrapper.config["head_dim"], 16)  # 32 / 2

    def test_build_empty_cache(self):
        model = _make_small_gpt2()
        wrapper = KerasHubVLLMModel(model)

        batch_size, max_length = 2, 8
        caches = wrapper.build_empty_cache(batch_size, max_length)

        self.assertEqual(len(caches), 2)  # 2 layers
        for layer_cache in caches:
            self.assertEqual(
                layer_cache.shape,
                (batch_size, 2, max_length, 2, 16),  # kv, seq, heads, dim
            )

    def test_forward_returns_correct_shapes(self):
        model = _make_small_gpt2()
        wrapper = KerasHubVLLMModel(model)

        # Build cache first via the model's own _build_cache.
        token_ids = ops.convert_to_tensor(
            np.array([[1, 2, 3, 4, 0, 0, 0, 0]], dtype="int32")
        )
        hidden_states, kv_caches = wrapper.build_cache(token_ids)

        # Now do a single-token decode step.
        new_token = ops.convert_to_tensor(
            np.array([[5]], dtype="int32")
        )
        metadata = _FakeAttentionMetadata(cache_update_index=3)

        updated_caches, hidden_states, aux = wrapper(
            kv_caches, new_token, metadata
        )

        self.assertEqual(len(updated_caches), 2)  # 2 layers
        self.assertEqual(len(aux), 0)

        # hidden_states: [batch, seq_len, hidden_dim]
        self.assertEqual(hidden_states.shape[-1], 32)

    def test_compute_logits(self):
        model = _make_small_gpt2()
        wrapper = KerasHubVLLMModel(model)

        batch_size = 1
        hidden = ops.convert_to_tensor(
            np.random.randn(batch_size, 1, 32).astype("float32")
        )
        logits = wrapper.compute_logits(hidden)

        self.assertEqual(logits.shape, (batch_size, 1, 100))  # vocab_size=100

    def test_logit_equivalence(self):
        """Logits from wrapper must match direct call_with_cache."""
        model = _make_small_gpt2()
        wrapper = KerasHubVLLMModel(model)

        token_ids = ops.convert_to_tensor(
            np.array([[10, 20, 30, 0, 0, 0, 0, 0]], dtype="int32")
        )
        # Direct call via model.
        _, direct_cache = model._build_cache(token_ids)
        single_token = ops.convert_to_tensor(np.array([[40]], dtype="int32"))
        direct_logits, _, _ = model.call_with_cache(
            single_token, direct_cache, 2
        )

        # Via wrapper.
        _, wrapper_caches = wrapper.build_cache(token_ids)
        metadata = _FakeAttentionMetadata(cache_update_index=2)
        updated_caches, hidden_states, _ = wrapper(
            wrapper_caches, single_token, metadata
        )
        wrapper_logits = wrapper.compute_logits(hidden_states)

        self.assertAllClose(
            ops.convert_to_numpy(ops.squeeze(direct_logits, axis=1)),
            ops.convert_to_numpy(ops.squeeze(wrapper_logits, axis=1)),
            atol=1e-5,
        )

    def test_to_vllm_method(self):
        """CausalLM.to_vllm() returns a KerasHubVLLMModel."""
        model = _make_small_gpt2()
        wrapper = model.to_vllm()
        self.assertIsInstance(wrapper, KerasHubVLLMModel)
        self.assertIs(wrapper.model, model)

    def test_variables_exposed(self):
        model = _make_small_gpt2()
        wrapper = KerasHubVLLMModel(model)

        self.assertEqual(
            len(wrapper.trainable_variables),
            len(model.trainable_variables),
        )
        self.assertEqual(
            len(wrapper.non_trainable_variables),
            len(model.non_trainable_variables),
        )
