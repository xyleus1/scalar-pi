"""Receipt and stage-order tests; no Torch, model assets or GPU required."""

import copy

import numpy as np
import pytest

from openpi.research.hybrid_qualification import FIELDS
from openpi.research.hybrid_qualification import PURPOSE
from openpi.research.hybrid_qualification import validate_hybrid_report
from openpi.shared.potential_precision import FP32_BRIDGE_PRECISION
from openpi.shared.potential_precision import RESEARCH_PRECISION
from scripts import check_hybrid_cache as check


def receipt():
    report = {
        "purpose": PURPOSE,
        "passed": True,
        "measured_gpu": True,
        "identity": {"base_checkpoint_sha256": "a" * 64},
        "precision": dict(RESEARCH_PRECISION),
        "fp32_bridge_precision": dict(FP32_BRIDGE_PRECISION),
        "corpus_sha256": "b" * 64,
        "corpus_observations": 40,
        "stage_order": ["base_fp32", "reference_fp32", "base_hybrid", "reference_hybrid"],
        "checkpoints": {},
    }
    for label in ("base", "reference"):
        value = {
            "checkpoint_sha256": "a" * 64,
            "normalization_sha256": "c" * 64,
            "conversion_receipt_sha256": "e" * 64,
            "input_sha256": [f"{index:064x}" for index in range(40)],
        }
        for mode in ("fp32", "hybrid"):
            fields = (
                [*FIELDS, "native_velocity", "native_trajectory_k1", "native_trajectory_k10"]
                if mode == "fp32"
                else FIELDS
            )
            value[mode] = {
                "passed": True,
                "observations": [
                    {
                        "index": index,
                        "cache_unchanged": True,
                        "checks": {field: {"passed": True, "atol": 1e-4, "rtol": 1e-4} for field in fields},
                    }
                    for index in range(40)
                ],
            }
        report["checkpoints"][label] = value
    return report


def validate(report):
    original = receipt()
    return validate_hybrid_report(
        report,
        identity=original["identity"],
        corpus_sha256=original["corpus_sha256"],
        base_normalization_sha256="c" * 64,
        reference_checkpoint_sha256="a" * 64,
        reference_normalization_sha256="c" * 64,
        reference_conversion_sha256="e" * 64,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "precision",
        "bridge",
        "order",
        "reference",
        "count",
        "native",
        "trajectory",
        "cache",
        "tolerance",
        "failed",
        "base",
        "reference_weights",
        "reference_norms",
        "input_hash",
    ],
)
def test_incomplete_or_cross_precision_receipts_never_qualify(mutation):
    report = receipt()
    if mutation == "source":
        report["identity"]["code_commit"] = "other"
    elif mutation in ("precision", "bridge"):
        report["precision" if mutation == "precision" else "fp32_bridge_precision"]["vlm_precision"] = "other"
    elif mutation == "order":
        report["stage_order"][1:3] = report["stage_order"][1:3][::-1]
    elif mutation == "reference":
        del report["checkpoints"]["reference"]
    elif mutation == "base":
        report["checkpoints"]["base"]["checkpoint_sha256"] = "d" * 64
    elif mutation in ("reference_weights", "reference_norms"):
        del report["checkpoints"]["reference"][
            "checkpoint_sha256" if mutation == "reference_weights" else "normalization_sha256"
        ]
    elif mutation == "input_hash":
        report["checkpoints"]["reference"]["input_sha256"][39] = "not-a-sha256"
    else:
        stage = report["checkpoints"]["reference"]["fp32" if mutation == "native" else "hybrid"]
        if mutation == "count":
            stage["observations"].pop()
        elif mutation == "failed":
            stage["passed"] = False
        else:
            row = stage["observations"][39]
            if mutation == "cache":
                row["cache_unchanged"] = False
            elif mutation == "tolerance":
                row["checks"]["velocity_t05"]["atol"] = 0.02
            else:
                del row["checks"]["native_velocity" if mutation == "native" else "trajectory_k10"]
    with pytest.raises(ValueError, match="Hybrid"):
        validate(report)


def test_cross_precision_drift_is_descriptive_and_does_not_replace_a_matched_gate():
    report = receipt()
    report["checkpoints"]["base"]["cross_precision_drift"] = {"max_absolute_error": 100}
    assert validate(report) is report
    report["checkpoints"]["base"]["hybrid"]["observations"][0]["checks"]["velocity_t05"]["passed"] = False
    with pytest.raises(ValueError, match="Hybrid"):
        validate(report)


@pytest.mark.parametrize("failed_stage", [0, 1, 2, 3, None])
def test_actual_orchestration_stops_before_later_precision_after_failure(monkeypatch, failed_stage):
    report = {"stage_order": [], "checkpoints": {"base": {}, "reference": {}}}
    calls = []

    def stage(checkpoint, inputs, corpus, mode):
        del inputs, corpus
        calls.append((checkpoint, mode))
        outputs = [
            {
                field: np.array([1 if mode == "fp32" else 2])
                for field in ("velocity_t05", "trajectory_k1", "trajectory_k10")
            }
        ]
        return {"passed": len(calls) - 1 != failed_stage}, outputs

    monkeypatch.setattr(check, "check_stage", stage)
    if failed_stage is not None:
        with pytest.raises(ValueError, match="no later precision"):
            check.run_precision_stages(
                report, {"base": "base", "reference": "reference"}, {"base": [], "reference": []}, {}
            )
        assert len(calls) == failed_stage + 1
    else:
        check.run_precision_stages(
            report, {"base": "base", "reference": "reference"}, {"base": [], "reference": []}, {}
        )
        assert calls == [("base", "fp32"), ("reference", "fp32"), ("base", "hybrid"), ("reference", "hybrid")]
        assert report["checkpoints"]["base"]["cross_precision_drift"]["velocity_t05"] == {"max_absolute_error": 1.0}


def test_hybrid_and_fp32_metadata_are_independent():
    before = copy.deepcopy(FP32_BRIDGE_PRECISION)
    assert RESEARCH_PRECISION != FP32_BRIDGE_PRECISION
    assert before == FP32_BRIDGE_PRECISION
    assert FP32_BRIDGE_PRECISION["vlm_precision"] == "float32"
    assert RESEARCH_PRECISION["vlm_precision"] == "bfloat16"
