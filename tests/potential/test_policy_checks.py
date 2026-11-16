import json

import numpy as np
import pytest

from openpi.research.policy_checks import array_error
from openpi.research.policy_checks import input_digest
from openpi.research.policy_checks import load_corpus
from openpi.research.policy_checks import require_staged_tokenizer
from openpi.research.policy_checks import timing_summary
from openpi.shared.potential_precision import FP32_BRIDGE_PRECISION
from scripts.benchmark_potential import measure_calls
from scripts.check_potential_parity import compare_exports


def _corpus():
    return {
        "images": np.zeros((2, 4, 4, 3), np.uint8),
        "wrist_images": np.zeros((2, 4, 4, 3), np.uint8),
        "states": np.zeros((2, 8), np.float32),
        "prompts": np.asarray(["first task", "second task"]),
        "noise": np.arange(640, dtype=np.float32).reshape(2, 10, 32),
    }


def test_corpus_requires_finite_consistent_arrays_without_pickle(tmp_path):
    path = tmp_path / "corpus.npz"
    corpus = _corpus()
    np.savez(path, **corpus)
    assert len(load_corpus(path)["states"]) == 2
    corpus["noise"][0, 0, 0] = np.nan
    np.savez(path, **corpus)
    with pytest.raises(ValueError, match="noise must be finite"):
        load_corpus(path)


def test_input_digest_binds_values_dtype_shape_and_ignores_dictionary_order():
    array = np.arange(4, dtype=np.float32)
    assert input_digest({"a": array, "b": {"c": True}}) == input_digest({"b": {"c": True}, "a": array})
    assert input_digest({"a": array}) != input_digest({"a": array.reshape(2, 2)})
    assert input_digest({"a": array}) != input_digest({"a": array.astype(np.float64)})


def test_timing_excludes_warmup_reuses_noise_and_separates_executed_throughput():
    class FakePolicy:
        def __init__(self):
            self.calls = []

        def infer(self, obs, *, noise):
            self.calls.append((obs["prompt"], noise.copy()))
            # Warmup samples have much larger timings and must be excluded.
            milliseconds = 1000.0 if len(self.calls) <= 3 else 10.0
            return {
                "actions": np.zeros((10, 7)),
                "policy_timing": {"infer_ms": milliseconds / 2, "total_ms": milliseconds},
            }

    events = []
    policy = FakePolicy()
    corpus = _corpus()
    samples = measure_calls(
        policy,
        corpus,
        warmup=3,
        calls=4,
        synchronize=lambda: events.append("sync"),
        reset_peak=lambda: events.append("reset"),
    )
    assert len(policy.calls) == 7
    assert len(samples) == 4
    assert events.count("reset") == 1
    assert [item["corpus_index"] for item in samples] == [0, 1, 0, 1]
    np.testing.assert_array_equal(policy.calls[3][1], corpus["noise"][0])
    summary = timing_summary(samples)
    assert summary["total_ms"]["median"] == 10
    assert summary["generated_actions_per_second"] == 1000
    assert summary["executed_actions_per_second"] == 500


def test_parity_comparison_rejects_changed_inputs_and_detects_output_error(tmp_path):
    def write(backend, *, digest="same", delta=0, cache_checks=True):
        metadata = {
            **FP32_BRIDGE_PRECISION,
            "schema_version": 1,
            "backend": backend,
            "precision": "float32",
            "corpus_sha256": "corpus",
            "normalization_sha256": "norms",
            "flow_time": 0.5,
            "corpus_observations": 40,
            "task_indices": list(range(40)),
            "cached_joint_fp32": [
                {"passed": True, "atol": 1e-4, "rtol": 1e-4, "max_abs": 0.0, "rms": 0.0} for _ in range(40)
            ]
            if cache_checks
            else [],
        }
        path = tmp_path / f"{backend}.npz"
        np.savez(
            path,
            metadata_json=np.asarray(json.dumps(metadata)),
            input_sha256=np.asarray([f"{digest}-{index}" for index in range(40)]),
            velocity_t05=np.ones((40, 10, 32)) + delta,
            actions_k1=np.ones((40, 10, 32)),
            actions_k10=np.ones((40, 10, 32)),
        )
        return path

    original = write("jax")
    assert compare_exports(original, write("torch"))["passed"]
    assert not compare_exports(original, write("torch", delta=0.1))["passed"]
    with pytest.raises(ValueError, match="identical transformed"):
        compare_exports(original, write("torch", digest="different"))
    with pytest.raises(ValueError, match="Every exported observation"):
        compare_exports(original, write("torch", cache_checks=False))


@pytest.mark.parametrize("change", ["bf16", "missing_precision", "short_corpus", "duplicate_task", "loose_cache"])
def test_parity_cannot_admit_stale_precision_or_incomplete_corpus(tmp_path, change):
    def export(backend):
        metadata = {
            **FP32_BRIDGE_PRECISION,
            "schema_version": 1,
            "backend": backend,
            "precision": "float32",
            "corpus_sha256": "corpus",
            "normalization_sha256": "norms",
            "flow_time": 0.5,
            "corpus_observations": 40,
            "task_indices": list(range(40)),
            "cached_joint_fp32": [
                {"passed": True, "atol": 1e-4, "rtol": 1e-4, "max_abs": 0.0, "rms": 0.0} for _ in range(40)
            ],
        }
        count = 40
        if backend == "torch":
            if change == "bf16":
                metadata["vlm_precision"] = "bfloat16"
            elif change == "missing_precision":
                metadata.pop("precision_protocol")
            elif change == "short_corpus":
                count = metadata["corpus_observations"] = 39
                metadata["task_indices"] = list(range(count))
            elif change == "duplicate_task":
                metadata["task_indices"][-1] = 38
            elif change == "loose_cache":
                metadata["cached_joint_fp32"][-1]["atol"] = 2e-2
        path = tmp_path / f"{backend}.npz"
        np.savez(
            path,
            metadata_json=np.asarray(json.dumps(metadata)),
            input_sha256=np.asarray([str(index) for index in range(count)]),
            **{name: np.zeros((count, 10, 32)) for name in ("velocity_t05", "actions_k1", "actions_k10")},
        )
        return path

    with pytest.raises(ValueError, match="precision|forty|unchanged"):
        compare_exports(export("jax"), export("torch"))


def test_parity_rejects_nonfinite_values():
    assert not array_error([np.nan], [0.0], atol=1e-4, rtol=1e-4)["passed"]


def test_missing_tokenizer_fails_before_an_implicit_gcs_download(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Stage the PaliGemma tokenizer"):
        require_staged_tokenizer()
    tokenizer = tmp_path / "big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir()
    tokenizer.write_bytes(b"fixture")
    assert require_staged_tokenizer() == tokenizer


def test_frozen_parameter_digest_handles_bfloat16_and_detects_changes():
    import torch

    from scripts.qualify_potential import _parameter_digest

    module = torch.nn.Linear(3, 2).bfloat16()
    original = _parameter_digest(module)
    assert original == _parameter_digest(module)
    with torch.no_grad():
        module.weight.add_(1)
    assert original != _parameter_digest(module)
