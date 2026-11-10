"""Batch-selection orchestration tests mock the trainer; no CUDA or model is loaded."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest


def load_script():
    path = Path(__file__).resolve().parents[2] / "scripts/select_potential_batch.py"
    spec = importlib.util.spec_from_file_location("select_batch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"fixture weights")
    (weights / "config.json").write_text("{}")
    data = tmp_path / "manifest.json"
    data.write_text("{}")
    return argparse.Namespace(
        weights=weights,
        data_manifest=data,
        output_root=tmp_path / "trials",
        output=tmp_path / "selection.json",
        hourly_rate=3.29,
        seed=42,
        num_workers=2,
    )


def trainer_mock(module, args, monkeypatch, outcomes):
    called = []
    monkeypatch.setenv("SCALAR_PI_IMAGE_ID", "sha256:fixture")

    def run(argv, *, stdout, stderr, check):
        assert stderr == subprocess.STDOUT
        assert check is False
        batch = int(argv[argv.index("--microbatch-size") + 1])
        checkpointed = "--gradient-checkpointing" in argv
        called.append((batch, checkpointed))
        outcome = outcomes[(batch, checkpointed)]
        if isinstance(outcome, str):
            stdout.write(outcome.encode())
            return types.SimpleNamespace(returncode=1)
        name = argv[argv.index("--exp-name") + 1]
        path = args.output_root / module.CONFIG / name
        path.mkdir(parents=True)
        profile = {
            "measured_gpu": True,
            "mode": "profile",
            "completed_updates": 2,
            "timed_optimizer_updates": 1,
            "effective_batch_size": 256,
            "checkpoint_seconds": [1.0],
            "total_memory_bytes": 100,
            "peak_reserved_bytes": outcome,
        }
        (path / "profile.json").write_text(json.dumps(profile))
        manifest = {
            "container_image_id": "sha256:fixture",
            "code": {"commit": "a" * 40},
            "source_weights_sha256": module.digest(args.weights / "model.safetensors"),
            "base_conversion_receipt_sha256": module.digest(args.weights / "config.json"),
            "data_manifest_sha256": module.digest(args.data_manifest),
            "normalization_hash": "b" * 64,
        }
        (path / "run_manifest.json").write_text(json.dumps(manifest))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    return called


def test_select_largest_fit_before_memory_reserve_limit(tmp_path, monkeypatch):
    module, args = load_script(), arguments(tmp_path)
    called = trainer_mock(module, args, monkeypatch, {(1, False): 40, (2, False): 60, (4, False): 90})
    selected = module.select(args)
    assert called == [(1, False), (2, False), (4, False)]
    assert selected["microbatch_size"] == 2
    assert selected["gradient_checkpointing"] is False
    assert selected["verified"]
    assert selected["valid_training_cost_profile"] is False
    assert selected["attempts"][-1]["status"] == "headroom_limit"


def test_checkpointing_only_when_ordinary_batch_one_cannot_fit(tmp_path, monkeypatch):
    module, args = load_script(), arguments(tmp_path)
    called = trainer_mock(
        module,
        args,
        monkeypatch,
        {
            (1, False): "torch.OutOfMemoryError: CUDA out of memory.",
            (1, True): 50,
            (2, True): 65,
            (4, True): "torch.cuda.OutOfMemoryError: CUDA out of memory.",
        },
    )
    selected = module.select(args)
    assert called == [(1, False), (1, True), (2, True), (4, True)]
    assert selected["microbatch_size"] == 2
    assert selected["gradient_checkpointing"] is True


def test_unknown_failure_aborts_instead_of_trying_another_configuration(tmp_path, monkeypatch):
    module, args = load_script(), arguments(tmp_path)
    called = trainer_mock(module, args, monkeypatch, {(1, False): "FloatingPointError: invalid loss"})
    with pytest.raises(RuntimeError, match="Candidate failed"):
        module.select(args)
    assert called == [(1, False)]
    assert not args.output.exists()


def test_no_fit_produces_no_verified_selection(tmp_path, monkeypatch):
    module, args = load_script(), arguments(tmp_path)
    trainer_mock(module, args, monkeypatch, {(1, False): 90, (1, True): 95})
    with pytest.raises(RuntimeError, match="No potential-model microbatch"):
        module.select(args)
    assert not args.output.exists()


def test_preview_does_not_import_model_or_launch_process(tmp_path, monkeypatch, capsys):
    module = load_script()
    monkeypatch.delenv("SCALAR_PI_IMAGE_ID", raising=False)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: pytest.fail("Preview must not launch"))
    module.main(
        [
            "--weights",
            str(tmp_path / "unstaged"),
            "--data-manifest",
            str(tmp_path / "missing.json"),
            "--output-root",
            str(tmp_path / "trials"),
            "--output",
            str(tmp_path / "selection.json"),
            "--hourly-rate",
            "3.29",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"]
    assert report["batch_candidates"] == list(module.BATCHES)
    assert not list(tmp_path.iterdir())
    assert "torch" not in module.__dict__


def test_help_works_in_a_clean_python_process():
    script = Path(__file__).resolve().parents[2] / "scripts/select_potential_batch.py"
    result = subprocess.run([sys.executable, "-S", str(script), "--help"], capture_output=True, check=True, text=True)
    assert "--execute" in result.stdout


@pytest.mark.parametrize("change", [None, "weights", "manifest", "source", "commit", "dirty", "image", "norms"])
def test_verify_existing_selection_checks_actual_inputs_without_cuda(tmp_path, monkeypatch, change):
    module, args = load_script(), arguments(tmp_path)
    args.verify_report = args.output
    source = tmp_path / "source.py"
    source.write_text("# frozen source\n")
    sources = {"source.py": module.digest(source)}
    for config in ("pi05_libero_b0", "pi05_libero_b1", module.CONFIG):
        path = tmp_path / "assets" / config / "physical-intelligence/libero/norm_stats.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    report = {
        "schema_version": 1,
        "kind": "batch_selection",
        "verified": True,
        "effective_batch_size": 256,
        "microbatch_size": 2,
        "gradient_checkpointing": False,
        "matmul_precision": "highest",
        "container_image_id": "fixture-image",
        "inputs_sha256": module.input_hashes(args),
        "normalization_sha256": module.digest(path),
        "code": {
            "commit": "a" * 40,
            "source_files_sha256": sources,
            "source_snapshot_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        },
    }
    args.output.write_text(json.dumps(report))
    monkeypatch.setenv("SCALAR_PI_IMAGE_ID", "changed-image" if change == "image" else "fixture-image")
    changed_paths = {
        "weights": args.weights / "model.safetensors",
        "manifest": args.data_manifest,
        "source": source,
        "norms": path,
    }
    if change in changed_paths:
        changed_paths[change].write_text("changed")

    def git(argv, **kwargs):
        assert argv[0] == "git", "Verification must never launch training or CUDA work"
        if argv[1] == "rev-parse":
            return types.SimpleNamespace(stdout="b" * 40 if change == "commit" else "a" * 40)
        return types.SimpleNamespace(stdout=" M source.py" if change == "dirty" else "")

    monkeypatch.setattr(module.subprocess, "run", git)
    if change is None:
        assert module.verify_report(args, repository=tmp_path) == report
    else:
        with pytest.raises(ValueError, match="differ|same clean"):
            module.verify_report(args, repository=tmp_path)
