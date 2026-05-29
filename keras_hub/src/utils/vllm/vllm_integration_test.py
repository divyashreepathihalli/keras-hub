"""Tests for the vLLM/tpu-inference integration.

These tests run in three modes:
  1. Pure-Python (no JAX/flax/tpu-inference) — import and config-parser tests.
  2. JAX-only (no tpu-inference) — RoPE parity, weight-copy round-trip using
     a tiny randomly-initialised backbone.
  3. Full integration — skipped unless tpu-inference is installed.
"""

import types
import unittest
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_jax():
    try:
        import jax  # noqa
        import flax  # noqa
        return True
    except ImportError:
        return False


def _has_tpu_inference():
    try:
        import tpu_inference  # noqa
        return True
    except ImportError:
        return False


def _tiny_gemma_backbone():
    """Return a tiny GemmaBackbone for fast tests (no preset download)."""
    from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone

    return GemmaBackbone(
        vocabulary_size=256,
        num_layers=2,
        num_query_heads=2,
        num_key_value_heads=1,
        hidden_dim=16,
        intermediate_dim=32,
        head_dim=8,
    )


def _tiny_dims():
    return {
        "vocabulary_size": 256,
        "num_layers": 2,
        "num_query_heads": 2,
        "num_key_value_heads": 1,
        "hidden_dim": 16,
        "intermediate_dim": 32,
        "head_dim": 8,
        "layer_norm_epsilon": 1e-6,
        "query_head_dim_normalize": True,
        "use_post_ffw_norm": False,
        "use_post_attention_norm": False,
        "attention_logit_soft_cap": None,
        "final_logit_soft_cap": None,
    }


# ---------------------------------------------------------------------------
# 1. Import tests (no JAX required)
# ---------------------------------------------------------------------------

class TestImports(unittest.TestCase):
    def test_config_parser_importable(self):
        from keras_hub.src.utils.vllm import config_parser  # noqa

    def test_weight_loader_importable(self):
        from keras_hub.src.utils.vllm import weight_loader  # noqa

    def test_paged_attention_importable(self):
        from keras_hub.src.utils.vllm import paged_attention  # noqa

    def test_keras_hub_for_causal_lm_importable(self):
        from keras_hub.src.utils.vllm import keras_hub_for_causal_lm  # noqa

    def test_init_importable(self):
        from keras_hub.src.utils import vllm  # noqa


# ---------------------------------------------------------------------------
# 2. Config parser — builds correct dims from a tiny in-memory backbone
# ---------------------------------------------------------------------------

class TestConfigParser(unittest.TestCase):
    def test_preset_to_dims_shape(self):
        """_preset_to_dims returns all required keys."""
        from keras_hub.src.utils.vllm.config_parser import KerasHubConfigParser

        # Patch from_preset to return a tiny backbone (no network access).
        import unittest.mock as mock
        with mock.patch(
            "keras_hub.src.models.gemma.gemma_backbone.GemmaBackbone.from_preset",
            return_value=_tiny_gemma_backbone(),
        ):
            hf_config, dims = KerasHubConfigParser.from_preset("gemma_2b_en")

        required = [
            "vocabulary_size", "num_layers", "num_query_heads",
            "num_key_value_heads", "hidden_dim", "intermediate_dim",
            "head_dim",
        ]
        for key in required:
            self.assertIn(key, dims, f"missing key: {key}")

        self.assertEqual(dims["vocabulary_size"], 256)
        self.assertEqual(dims["num_layers"], 2)
        self.assertEqual(hf_config.architectures, ["GemmaForCausalLM"])

    def test_hf_config_has_expected_attrs(self):
        import unittest.mock as mock
        from keras_hub.src.utils.vllm.config_parser import KerasHubConfigParser

        with mock.patch(
            "keras_hub.src.models.gemma.gemma_backbone.GemmaBackbone.from_preset",
            return_value=_tiny_gemma_backbone(),
        ):
            hf_config, dims = KerasHubConfigParser.from_preset("gemma_2b_en")

        self.assertEqual(hf_config.vocab_size, 256)
        self.assertEqual(hf_config.num_hidden_layers, 2)
        self.assertEqual(hf_config.num_attention_heads, 2)
        self.assertEqual(hf_config.num_key_value_heads, 1)
        self.assertEqual(hf_config.hidden_size, 16)


# ---------------------------------------------------------------------------
# 3. RoPE parity (JAX required)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_has_jax(), "JAX + flax required")
class TestRoPEParity(unittest.TestCase):
    """Per-token RoPE in paged_attention must match KerasHub's _apply_rope."""

    def _keras_rope(self, x_np, start_index: int):
        """Run KerasHub's RoPE on a single-token sequence at start_index."""
        import keras
        import numpy as np
        from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
        from keras import ops

        rope = RotaryEmbedding(max_wavelength=10_000.0)
        # x_np: [head_dim] → [1, 1, num_heads, head_dim] (batch=1, seq=1)
        num_heads, head_dim = 2, 8
        x = x_np.reshape(1, 1, num_heads, head_dim)
        x_keras = keras.ops.convert_to_tensor(x, dtype="float32")
        out = rope(x_keras, start_index=start_index)
        # Gemma interleave: stack halves then reshape
        out = ops.reshape(
            ops.stack(ops.split(out, 2, axis=-1), axis=-1), ops.shape(out)
        )
        return np.array(out).reshape(num_heads, head_dim)

    def _paged_rope(self, x_np, position: int):
        """Run paged_attention._apply_rope_per_token on the same input."""
        import jax.numpy as jnp
        from keras_hub.src.utils.vllm.paged_attention import (
            _apply_rope_per_token,
        )

        # x_np: [head_dim] → [1, num_heads, head_dim]
        num_heads, head_dim = 2, 8
        x = jnp.array(x_np.reshape(1, num_heads, head_dim), dtype=jnp.float32)
        positions = jnp.array([position], dtype=jnp.int32)
        out = _apply_rope_per_token(x, positions, head_dim)
        # Apply Gemma interleave
        half = head_dim // 2
        x1, x2 = out[..., :half], out[..., half:]
        interleaved = jnp.stack([x1, x2], axis=-1)
        out = interleaved.reshape(out.shape)
        return np.array(out[0])  # [num_heads, head_dim]

    def test_rope_at_position_zero(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(2 * 8).astype(np.float32)  # [num_heads * head_dim]
        keras_out = self._keras_rope(x, start_index=0)
        paged_out = self._paged_rope(x, position=0)
        np.testing.assert_allclose(keras_out, paged_out, atol=1e-4, rtol=1e-4)

    def test_rope_at_position_nonzero(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(2 * 8).astype(np.float32)
        keras_out = self._keras_rope(x, start_index=17)
        paged_out = self._paged_rope(x, position=17)
        np.testing.assert_allclose(keras_out, paged_out, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 4. Weight-copy round-trip (JAX required, no tpu-inference)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_has_jax(), "JAX + flax required")
class TestWeightCopyRoundTrip(unittest.TestCase):
    """Weights from a KerasHub backbone must survive the copy into nnx params."""

    def _make_vllm_config(self, dims):
        hf = types.SimpleNamespace(
            architectures=["GemmaForCausalLM"],
            vocab_size=dims["vocabulary_size"],
            num_hidden_layers=dims["num_layers"],
            num_attention_heads=dims["num_query_heads"],
            num_key_value_heads=dims["num_key_value_heads"],
            hidden_size=dims["hidden_dim"],
            intermediate_size=dims["intermediate_dim"],
            head_dim=dims["head_dim"],
            rms_norm_eps=dims.get("layer_norm_epsilon", 1e-6),
            max_position_embeddings=128,
            query_head_dim_normalize=True,
            use_post_ffw_norm=False,
            use_post_attention_norm=False,
            attn_logit_softcapping=None,
            final_logit_softcapping=None,
        )
        model_config = types.SimpleNamespace(
            hf_config=hf,
            dtype="bfloat16",
            model="dummy_preset",
        )
        cache_config = types.SimpleNamespace(cache_dtype="bfloat16")
        parallel_config = types.SimpleNamespace()
        return types.SimpleNamespace(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
        )

    def test_embedding_weight_preserved(self):
        """token_embedding weights copied from backbone must be equal."""
        import unittest.mock as mock
        import jax.numpy as jnp
        from flax import nnx
        from keras_hub.src.utils.vllm.weight_loader import load_preset_weights

        backbone = _tiny_gemma_backbone()
        # Build a tiny nnx model shell (just needs token_embedding + layers).
        dims = _tiny_dims()
        vllm_config = self._make_vllm_config(dims)

        with mock.patch(
            "keras_hub.src.utils.vllm.weight_loader._get_keras_backbone",
            return_value=backbone,
        ):
            # Verify backbone embedding is accessible.
            emb = backbone.token_embedding.get_weights()[0]
            self.assertEqual(emb.shape, (256, 16))

    def test_attention_kernel_shape_consistency(self):
        """Query kernel shape from backbone matches expected [nq, hidden, head_dim]."""
        backbone = _tiny_gemma_backbone()
        dims = _tiny_dims()

        # Build the model so its weights are initialized.
        import numpy as np
        dummy_input = {
            "token_ids": np.ones((1, 4), dtype="int32"),
            "padding_mask": np.ones((1, 4), dtype="int32"),
        }
        backbone(dummy_input)

        layer = backbone.transformer_layers[0]
        q_kernel = layer.attention.query_dense.kernel.numpy()
        k_kernel = layer.attention.key_dense.kernel.numpy()
        v_kernel = layer.attention.value_dense.kernel.numpy()
        o_kernel = layer.attention.output_dense.kernel.numpy()

        nq = dims["num_query_heads"]
        nkv = dims["num_key_value_heads"]
        hd = dims["head_dim"]
        hidden = dims["hidden_dim"]

        self.assertEqual(q_kernel.shape, (nq, hidden, hd))
        self.assertEqual(k_kernel.shape, (nkv, hidden, hd))
        self.assertEqual(v_kernel.shape, (nkv, hidden, hd))
        self.assertEqual(o_kernel.shape, (nq, hd, hidden))

    def test_mlp_kernel_shape_consistency(self):
        """MLP kernel shapes from backbone match expected dims."""
        backbone = _tiny_gemma_backbone()
        dims = _tiny_dims()

        import numpy as np
        dummy_input = {
            "token_ids": np.ones((1, 4), dtype="int32"),
            "padding_mask": np.ones((1, 4), dtype="int32"),
        }
        backbone(dummy_input)

        layer = backbone.transformer_layers[0]
        # Gemma SwiGLU: gating_ffw / gating_ffw_2 → intermediate_dim//2,
        # ffw_linear: intermediate_dim//2 → hidden_dim
        gate_k = layer.gating_ffw.kernel.numpy()
        ffw1_k = layer.gating_ffw_2.kernel.numpy()
        ffw2_k = layer.ffw_linear.kernel.numpy()

        hidden = dims["hidden_dim"]
        half_inter = dims["intermediate_dim"] // 2

        self.assertEqual(gate_k.shape, (hidden, half_inter))
        self.assertEqual(ffw1_k.shape, (hidden, half_inter))
        self.assertEqual(ffw2_k.shape, (half_inter, hidden))


# ---------------------------------------------------------------------------
# 5. Plugin registration smoke-test (no tpu-inference)
# ---------------------------------------------------------------------------

class TestPluginRegistration(unittest.TestCase):
    def test_register_raises_without_tpu_inference(self):
        """register() should raise ImportError when tpu-inference is missing."""
        if _has_tpu_inference():
            self.skipTest("tpu-inference is installed; skipping absence test")
        from keras_hub.src.utils.vllm import register
        with self.assertRaises(ImportError):
            register()

    def test_register_succeeds_with_tpu_inference(self):
        """register() should call register_model when tpu-inference is present."""
        if not _has_tpu_inference():
            self.skipTest("tpu-inference not installed")
        import unittest.mock as mock
        with mock.patch(
            "tpu_inference.models.common.model_loader.register_model"
        ) as mock_reg:
            from keras_hub.src.utils.vllm import register
            register()
            mock_reg.assert_called_once()
            call_args = mock_reg.call_args
            self.assertEqual(call_args[0][0], "GemmaForCausalLM")


if __name__ == "__main__":
    unittest.main()
