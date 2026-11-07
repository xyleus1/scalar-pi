"""Conversion must depend on the model configuration, never a directory name."""

from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import torch


def test_pi05_adaptive_norm_conversion_works_after_renaming_checkpoint():
    converter = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples/convert_jax_model_to_pytorch.py"))
    config = SimpleNamespace(width=4, depth=1, num_heads=1, head_dim=4)
    weights = {
        "llm/layers/attn/attn_vec_einsum_1/w": np.ones((1, 1, 4, 4)),
        "llm/layers/attn/kv_einsum_1/w": np.ones((1, 2, 1, 4, 4)),
        "llm/layers/attn/q_einsum_1/w": np.ones((1, 1, 4, 4)),
        "llm/layers/mlp_1/gating_einsum": np.ones((1, 2, 4, 8)),
        "llm/layers/mlp_1/linear": np.ones((1, 8, 4)),
        "llm/layers/pre_attention_norm_1/Dense_0/kernel": np.ones((1, 4, 12)),
        "llm/layers/pre_ffw_norm_1/Dense_0/kernel": np.ones((1, 4, 12)),
        "llm/layers/pre_attention_norm_1/Dense_0/bias": np.ones((1, 12)),
        "llm/layers/pre_ffw_norm_1/Dense_0/bias": np.ones((1, 12)),
        "llm/final_norm_1/Dense_0/kernel": np.ones((4, 12)),
        "llm/final_norm_1/Dense_0/bias": np.ones((12,)),
    }
    converted = converter["slice_gemma_state_dict"](
        weights,
        config,
        num_expert=1,
        checkpoint_dir="/arbitrary/base",
        pi05=True,
    )
    prefix = "paligemma_with_expert.gemma_expert.model"
    torch.testing.assert_close(converted[f"{prefix}.norm.dense.weight"], torch.ones(12, 4, dtype=torch.float64))
    assert f"{prefix}.layers.0.input_layernorm.dense.weight" in converted
    assert all(not key.startswith("llm/") for key in converted)
