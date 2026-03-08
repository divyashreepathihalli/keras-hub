import numpy as np
from keras import ops

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.vllm.cache_adapter import kerashub_to_vllm_cache
from keras_hub.src.utils.vllm.cache_adapter import rwkv7_to_vllm_cache
from keras_hub.src.utils.vllm.cache_adapter import vllm_to_kerashub_cache
from keras_hub.src.utils.vllm.cache_adapter import vllm_to_rwkv7_cache


class VllmToKerasHubCacheTest(TestCase):
    def test_round_trip(self):
        """Converting vllm -> kerashub -> vllm preserves values."""
        batch, num_layers, max_len, num_kv_heads, head_dim = 2, 4, 16, 8, 64
        kv_caches = [
            ops.convert_to_tensor(
                np.random.randn(batch, 2, max_len, num_kv_heads, head_dim)
                .astype("float32")
            )
            for _ in range(num_layers)
        ]

        stacked = vllm_to_kerashub_cache(kv_caches)
        self.assertEqual(
            stacked.shape,
            (batch, num_layers, 2, max_len, num_kv_heads, head_dim),
        )

        restored = kerashub_to_vllm_cache(stacked)
        self.assertEqual(len(restored), num_layers)
        for i in range(num_layers):
            self.assertAllClose(
                ops.convert_to_numpy(restored[i]),
                ops.convert_to_numpy(kv_caches[i]),
            )

    def test_kerashub_to_vllm_shapes(self):
        """Each per-layer cache has the expected shape."""
        batch, num_layers, max_len, num_kv_heads, head_dim = 1, 3, 8, 4, 32
        stacked = ops.zeros(
            (batch, num_layers, 2, max_len, num_kv_heads, head_dim)
        )
        per_layer = kerashub_to_vllm_cache(stacked)
        self.assertEqual(len(per_layer), num_layers)
        for layer_cache in per_layer:
            self.assertEqual(
                layer_cache.shape,
                (batch, 2, max_len, num_kv_heads, head_dim),
            )


class RWKV7CacheAdapterTest(TestCase):
    def test_round_trip(self):
        """Converting vllm -> rwkv7 -> vllm preserves values."""
        batch, num_layers, num_heads, head_dim, hidden_size = 2, 4, 8, 64, 512

        state_list = [
            ops.convert_to_tensor(
                np.random.randn(batch, num_heads, head_dim, head_dim)
                .astype("float32")
            )
            for _ in range(num_layers)
        ]
        last_token_list = [
            ops.convert_to_tensor(
                np.random.randn(batch, 2, 1, hidden_size).astype("float32")
            )
            for _ in range(num_layers)
        ]

        cache = vllm_to_rwkv7_cache(state_list, last_token_list)
        self.assertIsInstance(cache, list)
        self.assertEqual(len(cache), 2)

        restored_state, restored_tokens = rwkv7_to_vllm_cache(cache)
        self.assertEqual(len(restored_state), num_layers)
        self.assertEqual(len(restored_tokens), num_layers)

        for i in range(num_layers):
            self.assertAllClose(
                ops.convert_to_numpy(restored_state[i]),
                ops.convert_to_numpy(state_list[i]),
            )
            self.assertAllClose(
                ops.convert_to_numpy(restored_tokens[i]),
                ops.convert_to_numpy(last_token_list[i]),
            )
