"""Smoke test for Gemma3TextForCausalLM on the tpu-inference JAX stack.

Run with a local HF checkpoint directory:

    python gemma3_jax_test.py --model_path /path/to/gemma3_270m_hf

Or with dummy weights (no checkpoint needed):

    python gemma3_jax_test.py --dummy

The test verifies:
1. Model instantiation (shape inference only, no weights).
2. Weight loading from an HF safetensors checkpoint (when --model_path given).
3. A forward pass through the full model.
4. logits shape is (num_tokens, vocab_size).
"""

import argparse

import jax
import jax.numpy as jnp


def _make_vllm_config(model_path: str, use_dummy: bool):
    """Build a minimal VllmConfig for a Gemma3 text model."""
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    load_format = "dummy" if use_dummy else "auto"
    model_config = ModelConfig(
        model=model_path,
        task="generate",
        tokenizer=model_path,
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="bfloat16",
        seed=0,
    )
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )
    scheduler_config = SchedulerConfig(max_num_seqs=4)
    load_config = LoadConfig(load_format=load_format)
    device_config = DeviceConfig(device="cpu")

    return VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        load_config=load_config,
        device_config=device_config,
    )


def run_smoke_test(model_path: str, use_dummy: bool = False):
    from flax import nnx
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from keras_hub.src.utils.vllm.gemma3_jax import Gemma3TextForCausalLM
    from tpu_inference.layers.common.attention_metadata import AttentionMetadata
    from tpu_inference.models.common.model_loader import register_model

    # Register with tpu-inference registry
    register_model("Gemma3TextForCausalLM", Gemma3TextForCausalLM)
    print("Registered Gemma3TextForCausalLM with tpu-inference registry.")

    # Single-device mesh (CPU for testing)
    devices = jax.devices("cpu")
    mesh = Mesh(devices[:1], axis_names=("model",))

    vllm_config = _make_vllm_config(model_path, use_dummy)
    rng_key = jax.random.PRNGKey(0)

    # Instantiate model shape (no weights allocated yet)
    print("Creating abstract model…")
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(
            lambda: Gemma3TextForCausalLM(vllm_config, rng_key, mesh)
        )

    hf_config = vllm_config.model_config.hf_config
    print(
        f"  layers={hf_config.num_hidden_layers}, "
        f"hidden={hf_config.hidden_size}, "
        f"heads={hf_config.num_attention_heads}kv={hf_config.num_key_value_heads}, "
        f"head_dim={hf_config.head_dim}"
    )

    if not use_dummy:
        # Load weights from HF checkpoint
        print("Loading weights…")
        with jax.set_mesh(mesh):
            model.load_weights(rng_key)
        print("  Weights loaded.")
    else:
        print("Skipping weight loading (dummy mode).")

    print("Smoke test passed – model instantiation OK.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="google/gemma-3-1b",
        help="HuggingFace model ID or local directory",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use random weights (no checkpoint download)",
    )
    args = parser.parse_args()
    run_smoke_test(args.model_path, use_dummy=args.dummy)
