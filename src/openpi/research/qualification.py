"""Bind paid-work prerequisites to the exact inputs and execution environment."""

import json
import math
import os
from pathlib import Path
import re
import time

from openpi.research.evaluation import file_digest
from openpi.research.evaluation import source_identity


def execution_identity(*, checkpoint=None, base_hash=None, conversion_hash=None, repository=None):
    repository = Path(repository) if repository else Path(__file__).resolve().parents[3]
    source = source_identity(repository)
    if source["tracked_changes"]:
        raise ValueError("Qualification requires the same clean committed source")
    image = os.environ.get("SCALAR_PI_IMAGE_ID", "")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
        raise ValueError("Qualification requires the actual SCALAR_PI_IMAGE_ID")
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        base_hash = file_digest(checkpoint / "model.safetensors")
        conversion_hash = file_digest(checkpoint / "config.json")
    if any(re.fullmatch(r"[0-9a-f]{64}", value or "") is None for value in (base_hash, conversion_hash)):
        raise ValueError("Qualification requires actual base and conversion receipt hashes")
    return {
        "base_checkpoint_sha256": base_hash,
        "base_conversion_sha256": conversion_hash,
        "code_commit": source["commit"],
        "source_sha256": source["source_sha256"],
        "container_image_id": image,
    }


def verify_qualification(path, *, checkpoint=None, base_hash=None, conversion_hash=None, require_serving=True):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    identity = execution_identity(checkpoint=checkpoint, base_hash=base_hash, conversion_hash=conversion_hash)
    if report.get("identity") != identity:
        raise ValueError("Numerical qualification differs from current base/source/container")
    if report.get("correctness_checks_passed") is not True or report.get("measured_gpu") is not True:
        raise ValueError("A passing full-size CUDA qualification is required")
    if report.get("precision", {}).get("matmul") != "highest":
        raise ValueError("Qualification must use highest matmul precision")
    if require_serving and (
        report.get("math_only") is not False
        or report.get("actual_policy_serving", {}).get("passed") is not True
        or report.get("conversion_parity", {}).get("passed") is not True
    ):
        raise ValueError("Full corpus serving and conversion qualification are required")
    return report


def verify_recovery(path, *, identity, microbatch_size, gradient_checkpointing, data_manifest, normalization_hash):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "identity": identity,
        "microbatch_size": microbatch_size,
        "gradient_checkpointing": gradient_checkpointing,
        "data_manifest_sha256": file_digest(data_manifest),
        "normalization_sha256": normalization_hash,
    }
    if report.get("passed") is not True or report.get("purpose") != "full_size_cuda_recovery":
        raise ValueError("Passing selected-batch GPU recovery qualification is required")
    if (
        report.get("measured_gpu") is not True
        or report.get("actual_policy_serving", {}).get("passed") is not True
        or report.get("exact_resume") != {"raw_weights": True, "ema_weights": True, "optimizer_ema_rng_cursor": True}
    ):
        raise ValueError("Recovery must include exact training state and actual P serving")
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Recovery qualification differs from current inputs or batching")
    for key in ("update_seconds", "checkpoint_seconds"):
        value = report.get(key)
        if not isinstance(value, float | int) or not 0 < value < float("inf"):
            raise ValueError(f"Recovery qualification lacks measured {key}")
    return report


def can_start_update(deadline, *, update_seconds, checkpoint_seconds, reserve_seconds=600, now=None):
    """Leave measured update/save time and at least ten minutes for shutdown."""
    if reserve_seconds < 600:
        raise ValueError("Shutdown reserve must be at least 600 seconds")
    values = (update_seconds, checkpoint_seconds, reserve_seconds)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Deadline estimates must be finite and nonnegative")
    if deadline is None:
        return True
    if not math.isfinite(deadline):
        raise ValueError("Deadline must be finite")
    now = time.time() if now is None else now
    return deadline - now > sum(values)
