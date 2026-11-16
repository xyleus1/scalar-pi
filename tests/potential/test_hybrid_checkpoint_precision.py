"""Header-only hybrid precision checks; no Torch/JAX/model allocation required."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from openpi.shared import potential_checkpoint
from openpi.shared.potential_precision import FP32_BRIDGE_PRECISION
from openpi.shared.potential_precision import RESEARCH_PRECISION
from openpi.shared.potential_precision import VLM_FP32_SELECTORS
from openpi.shared.potential_precision import checkpoint_float_dtype
from openpi.shared.potential_precision import validate_research_tensor_dtypes

VLM = "paligemma_with_expert.paligemma."
EXPERT = "paligemma_with_expert.gemma_expert.model."
TENSORS = {
    VLM + "language_model.embed_tokens.weight": "BF16",
    VLM + "language_model.layers.0.self_attn.q_proj.weight": "BF16",
    VLM + "language_model.layers.0.mlp.up_proj.weight": "BF16",
    VLM + "multi_modal_projector.linear.weight": "BF16",
    VLM + "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight": "BF16",
    VLM + "vision_tower.vision_model.embeddings.patch_embedding.weight": "F32",
    VLM + "vision_tower.vision_model.embeddings.patch_embedding.bias": "F32",
    VLM + "vision_tower.vision_model.embeddings.position_embedding.weight": "F32",
    VLM + "language_model.layers.0.input_layernorm.weight": "F32",
    VLM + "language_model.layers.0.post_attention_layernorm.weight": "F32",
    VLM + "language_model.norm.weight": "F32",
    EXPERT + "layers.0.self_attn.q_proj.weight": "F32",
    EXPERT + "layers.0.input_layernorm.dense.weight": "F32",
    "paligemma_with_expert.gemma_expert.lm_head.weight": "F32",
    "action_in_proj.weight": "F32",
    "action_out_proj.weight": "F32",
    "time_mlp_in.weight": "F32",
    "time_mlp_out.weight": "F32",
}


def write_weights(path, tensors, metadata):
    header = {"__metadata__": {key: str(value) for key, value in metadata.items()}}
    offset = 0
    sizes = {"BF16": 2, "F32": 4, "F16": 2, "F64": 8, "F8_E4M3": 1, "I64": 8, "BOOL": 1}
    for name, dtype in tensors.items():
        size = sizes.get(dtype, 1)
        header[name] = {"dtype": dtype, "shape": [1], "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(offset))


def checkpoint(path, *, mode="potential_dot", tensors=None, protocol=None):
    metadata = {
        "schema_version": 1,
        "trainer": "potential_pi",
        "velocity_mode": mode,
        "action_horizon": 10,
        "action_dim": 32,
        **(RESEARCH_PRECISION if protocol is None else protocol),
    }
    (path / potential_checkpoint.METADATA_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")
    for name in ("model.safetensors", "raw_model.safetensors"):
        write_weights(path / name, TENSORS if tensors is None else tensors, metadata)
    return metadata


def validate(path, *, mode="potential_dot"):
    return potential_checkpoint.validate_checkpoint_metadata(path, velocity_mode=mode, action_horizon=10, action_dim=32)


@pytest.mark.parametrize(("name", "dtype"), TENSORS.items())
def test_actual_parameter_domains_and_upstream_exceptions(name, dtype):
    assert checkpoint_float_dtype(name) == dtype


def test_exact_vlm_prefix_cannot_enable_bf16_for_other_modules():
    for name in (
        "fake." + VLM + "language_model.embed_tokens.weight",
        "paligemma_with_expert.paligemma_fake.language_model.embed_tokens.weight",
        "paligemma_with_expert.gemma_expert.model.layers.0.mlp.up_proj.weight",
    ):
        assert checkpoint_float_dtype(name) == "F32"


@pytest.mark.parametrize("mode", ["direct", "potential_dot"])
def test_real_raw_and_ema_headers_accept_exact_hybrid_policy(tmp_path, mode):
    expected = checkpoint(tmp_path, mode=mode)
    assert validate(tmp_path, mode=mode) == expected
    assert potential_checkpoint.RESEARCH_PRECISION is RESEARCH_PRECISION


@pytest.mark.parametrize("filename", ["model.safetensors", "raw_model.safetensors"])
@pytest.mark.parametrize(
    ("name", "wrong"),
    [
        (VLM + "language_model.layers.0.self_attn.q_proj.weight", "F32"),
        (VLM + "language_model.layers.0.self_attn.q_proj.weight", "F16"),
        (VLM + "language_model.layers.0.self_attn.q_proj.weight", "F64"),
        (VLM + "language_model.layers.0.self_attn.q_proj.weight", "F8_E4M3"),
        (VLM + "language_model.norm.weight", "BF16"),
        (VLM + "vision_tower.vision_model.embeddings.patch_embedding.weight", "BF16"),
        (EXPERT + "layers.0.self_attn.q_proj.weight", "BF16"),
        ("action_out_proj.weight", "BF16"),
        ("time_mlp_in.weight", "F16"),
        ("time_mlp_out.weight", "UNKNOWN"),
    ],
)
def test_actual_tensor_dtype_must_match_in_each_checkpoint(tmp_path, filename, name, wrong):
    metadata = checkpoint(tmp_path)
    write_weights(tmp_path / filename, {**TENSORS, name: wrong}, metadata)
    with pytest.raises(ValueError, match="Hybrid research weights require"):
        validate(tmp_path)


def test_old_fp32_research_checkpoint_cannot_be_relabelled_hybrid(tmp_path):
    all_fp32 = dict.fromkeys(TENSORS, "F32")
    checkpoint(tmp_path, tensors=all_fp32)
    with pytest.raises(ValueError, match="require BF16"):
        validate(tmp_path)
    checkpoint(tmp_path, protocol=FP32_BRIDGE_PRECISION)
    with pytest.raises(ValueError, match="precision_protocol"):
        validate(tmp_path)


@pytest.mark.parametrize("filename", ["sidecar", "model.safetensors", "raw_model.safetensors"])
def test_new_dtype_policy_is_bound_in_sidecar_and_both_headers(tmp_path, filename):
    metadata = checkpoint(tmp_path)
    del metadata["checkpoint_dtype_policy"]
    if filename == "sidecar":
        (tmp_path / potential_checkpoint.METADATA_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")
    else:
        write_weights(tmp_path / filename, TENSORS, metadata)
    with pytest.raises(ValueError, match="checkpoint_dtype_policy"):
        validate(tmp_path)


@pytest.mark.parametrize("filename", ["model.safetensors", "raw_model.safetensors"])
def test_another_precision_policy_in_header_is_rejected(tmp_path, filename):
    metadata = checkpoint(tmp_path)
    metadata["checkpoint_dtype_policy"] = "other_policy"
    write_weights(tmp_path / filename, TENSORS, metadata)
    with pytest.raises(ValueError, match="checkpoint_dtype_policy"):
        validate(tmp_path)


def test_unmarked_native_direct_checkpoint_behavior_is_unchanged(tmp_path):
    write_weights(tmp_path / "model.safetensors", {"legacy.weight": "BF16"}, {})
    assert validate(tmp_path, mode="direct") is None
    with pytest.raises(ValueError, match="requires"):
        validate(tmp_path)


def test_integer_and_boolean_buffers_are_not_floating_precision_casts():
    validate_research_tensor_dtypes(
        {
            "__metadata__": {},
            "position_ids": {"dtype": "I64"},
            "mask": {"dtype": "BOOL"},
        }
    )


@pytest.mark.parametrize("descriptor", [None, [], {}, {"dtype": 1}])
def test_malformed_tensor_descriptors_fail_with_validation_error(descriptor):
    with pytest.raises(ValueError, match="Invalid safetensors tensor dtype"):
        validate_research_tensor_dtypes({"weight": descriptor})


def test_bridge_metadata_and_exceptions_are_explicit():
    assert FP32_BRIDGE_PRECISION == {
        "precision_protocol": "potential_pi_fp32_v1",
        "vlm_precision": "float32",
        "expert_precision": "float32",
        "matmul_precision": "highest",
    }
    assert RESEARCH_PRECISION["precision_protocol"] == "potential_pi_bf16_vlm_fp32_expert_v1"
    assert RESEARCH_PRECISION["checkpoint_dtype_policy"] == "pi05_frozen_vlm_bf16_except_embeddings_norm_fp32_v1"
    assert VLM_FP32_SELECTORS == (
        "embeddings.patch_embedding",
        "embeddings.position_embedding",
        "input_layernorm",
        "post_attention_layernorm",
        "model.norm",
    )


def test_metadata_imports_remain_cpu_stdlib_only():
    source = Path(__file__).resolve().parents[2] / "src"
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import openpi.shared.potential_checkpoint; "
        "assert not any(n.split('.')[0] in {'torch','jax','transformers'} for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-c", code, str(source)], check=True, timeout=15)
