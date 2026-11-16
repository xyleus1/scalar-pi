"""The matched hybrid study's parameter and checkpoint precision contract.

The original FP32 conversion remains the initialization source. Only the frozen
VLM is cast to BF16, retaining the upstream embedding/norm exceptions. The
trainable expert and all other floating tensors stay FP32. These pure helpers
let model construction and header-only checkpoint validation use one policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

FP32_BRIDGE_PRECISION = {
    "precision_protocol": "potential_pi_fp32_v1",
    "vlm_precision": "float32",
    "expert_precision": "float32",
    "matmul_precision": "highest",
}

RESEARCH_PRECISION = {
    "precision_protocol": "potential_pi_bf16_vlm_fp32_expert_v1",
    "vlm_precision": "bfloat16",
    "expert_precision": "float32",
    "matmul_precision": "highest",
    "checkpoint_dtype_policy": "pi05_frozen_vlm_bf16_except_embeddings_norm_fp32_v1",
}

# Match configure_expert_only's original upstream exceptions. Names passed to
# checkpoint_float_dtype are full model.state_dict() keys, not expert-only keys.
VLM_FP32_SELECTORS = (
    "embeddings.patch_embedding",
    "embeddings.position_embedding",
    "input_layernorm",
    "post_attention_layernorm",
    "model.norm",
)
VLM_PREFIX = "paligemma_with_expert.paligemma."
_NONFLOAT_DTYPES = frozenset({"BOOL", "I8", "I16", "I32", "I64", "U8", "U16", "U32", "U64"})


def checkpoint_float_dtype(name: str) -> str:
    """Return the required safetensors dtype for a floating state-dict tensor."""
    if not isinstance(name, str) or not name:
        raise ValueError("Checkpoint tensor name must be a nonempty string")
    if name.startswith(VLM_PREFIX):
        relative = name[len(VLM_PREFIX) :]
        if not any(selector in relative for selector in VLM_FP32_SELECTORS):
            return "BF16"
    return "F32"


def validate_research_tensor_dtypes(header: Mapping[str, Any]) -> None:
    """Check actual marked-checkpoint header dtypes without importing a model.

    This checks dtype placement, not tensor completeness, shapes or payload
    hashes. The strict model loader and existing checkpoint verifier retain
    those responsibilities. Integer/bool buffers are not precision casts.
    """
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, Mapping) or not isinstance(tensor.get("dtype"), str):
            raise ValueError(f"Invalid safetensors tensor dtype: {name}")
        dtype = tensor["dtype"]
        if dtype in _NONFLOAT_DTYPES:
            continue
        expected = checkpoint_float_dtype(name)
        if dtype != expected:
            raise ValueError(f"Hybrid research weights require {expected} precision for {name}, got {dtype}")
