"""Validate the hybrid oracle gate independently of Torch or model loading."""

import json
from pathlib import Path
import re

from openpi.research.policy_checks import file_digest
from openpi.shared.potential_precision import FP32_BRIDGE_PRECISION
from openpi.shared.potential_precision import RESEARCH_PRECISION

PURPOSE = "matched_hybrid_cache_and_uncached_reference"
ATOL = RTOL = 1e-4
FIELDS = ("velocity_t05", "trajectory_k1", "trajectory_k10", "sample_k1", "sample_k10")


def validate_hybrid_report(
    report,
    *,
    identity,
    corpus_sha256,
    base_normalization_sha256,
    reference_checkpoint_sha256,
    reference_normalization_sha256,
    reference_conversion_sha256,
):
    """Require both checkpoints, all observations, and the separate FP32 oracle gate."""
    if (
        report.get("purpose") != PURPOSE
        or report.get("passed") is not True
        or report.get("measured_gpu") is not True
        or report.get("identity") != identity
        or report.get("precision") != RESEARCH_PRECISION
        or report.get("fp32_bridge_precision") != FP32_BRIDGE_PRECISION
        or report.get("corpus_sha256") != corpus_sha256
        or report.get("corpus_observations") != 40
        or report.get("stage_order") != ["base_fp32", "reference_fp32", "base_hybrid", "reference_hybrid"]
    ):
        raise ValueError("Hybrid oracle qualification differs from current source, inputs, precision, or gate order")
    checkpoints = report.get("checkpoints", {})
    if set(checkpoints) != {"base", "reference"}:
        raise ValueError("Hybrid oracle qualification requires base and released reference checkpoints")
    if (
        checkpoints["base"].get("checkpoint_sha256") != identity["base_checkpoint_sha256"]
        or checkpoints["base"].get("normalization_sha256") != base_normalization_sha256
        or checkpoints["reference"].get("checkpoint_sha256") != reference_checkpoint_sha256
        or checkpoints["reference"].get("normalization_sha256") != reference_normalization_sha256
        or checkpoints["reference"].get("conversion_receipt_sha256") != reference_conversion_sha256
    ):
        raise ValueError("Hybrid base or released reference identity differs")
    for checkpoint in checkpoints.values():
        hashes = [
            checkpoint.get("checkpoint_sha256"),
            checkpoint.get("normalization_sha256"),
            *checkpoint.get("input_sha256", []),
        ]
        if len(checkpoint.get("input_sha256", [])) != 40 or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes
        ):
            raise ValueError("Hybrid oracle must identify all forty transformed observations")
        for mode in ("fp32", "hybrid"):
            stage = checkpoint.get(mode, {})
            rows = stage.get("observations", [])
            expected = (
                {*FIELDS, "native_velocity", "native_trajectory_k1", "native_trajectory_k10"}
                if mode == "fp32"
                else set(FIELDS)
            )
            if stage.get("passed") is not True or [row.get("index") for row in rows] != list(range(40)):
                raise ValueError("Hybrid oracle requires forty passing ordered observations in every stage")
            for row in rows:
                checks = row.get("checks", {})
                if row.get("cache_unchanged") is not True or set(checks) != expected:
                    raise ValueError("Hybrid oracle is missing cache or trajectory evidence")
                if any(
                    check.get("passed") is not True or check.get("atol") != ATOL or check.get("rtol") != RTOL
                    for check in checks.values()
                ):
                    raise ValueError("Hybrid oracle requires unchanged atol=rtol=1e-4 on every comparison")
    return report


def verify_hybrid_report(path, **expected):
    path = Path(path)
    report = validate_hybrid_report(json.loads(path.read_text(encoding="utf-8")), **expected)
    return {"passed": True, "report_sha256": file_digest(path), "purpose": report["purpose"]}
