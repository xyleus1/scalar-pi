"""Qualification gates use fixture receipts only; no GPU or pretrained assets."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import safetensors.torch
import torch

from openpi.research import qualification
from scripts import qualify_potential
from scripts import qualify_potential_recovery as recovery


def passing_receipt():
    return {
        "identity": {"base_checkpoint_sha256": "a" * 64},
        "correctness_checks_passed": True,
        "measured_gpu": True,
        "precision": {"matmul": "highest"},
        "math_only": False,
        "actual_policy_serving": {"passed": True},
        "conversion_parity": {"passed": True},
    }


@pytest.mark.parametrize("change", [None, "identity", "failed", "cpu", "precision", "math_only", "serving", "parity"])
def test_full_qualification_rejects_stale_or_incomplete_receipts(tmp_path, monkeypatch, change):
    report = passing_receipt()
    monkeypatch.setattr(qualification, "execution_identity", lambda **kwargs: passing_receipt()["identity"])
    if change == "identity":
        report["identity"]["base_checkpoint_sha256"] = "b" * 64
    elif change in ("failed", "cpu"):
        report["correctness_checks_passed" if change == "failed" else "measured_gpu"] = False
    elif change == "precision":
        report["precision"]["matmul"] = "high"
    elif change == "math_only":
        report["math_only"] = True
    elif change in ("serving", "parity"):
        report["actual_policy_serving" if change == "serving" else "conversion_parity"]["passed"] = False
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(report))
    if change is None:
        assert qualification.verify_qualification(path) == report
    else:
        with pytest.raises(ValueError, match="qualification|Qualification"):
            qualification.verify_qualification(path)


def test_math_gate_can_be_reused_only_as_math_gate(tmp_path, monkeypatch):
    report = {**passing_receipt(), "math_only": True}
    report.pop("actual_policy_serving")
    report.pop("conversion_parity")
    path = tmp_path / "math.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(qualification, "execution_identity", lambda **kwargs: report["identity"])
    assert qualification.verify_qualification(path, require_serving=False) == report
    with pytest.raises(ValueError, match="Full corpus"):
        qualification.verify_qualification(path)


def test_identity_checks_actual_bytes_source_and_container(tmp_path, monkeypatch):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setenv("SCALAR_PI_IMAGE_ID", "sha256:" + "c" * 64)
    source = {"tracked_changes": [], "commit": "d" * 40, "source_sha256": "e" * 64}
    monkeypatch.setattr(qualification, "source_identity", lambda root: source)
    identity = qualification.execution_identity(checkpoint=tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    assert qualification.execution_identity(checkpoint=tmp_path) != identity
    source["tracked_changes"] = ["source.py"]
    with pytest.raises(ValueError, match="clean"):
        qualification.execution_identity(checkpoint=tmp_path)


def test_recovery_receipt_binds_batch_norm_and_actual_state_checks(tmp_path):
    data = tmp_path / "manifest.json"
    data.write_text("{}")
    arguments = {
        "identity": {"base": "a"},
        "microbatch_size": 4,
        "gradient_checkpointing": False,
        "data_manifest": data,
        "normalization_hash": "b" * 64,
    }
    report = {
        "purpose": "full_size_cuda_recovery",
        "passed": True,
        "measured_gpu": True,
        "identity": arguments["identity"],
        "microbatch_size": 4,
        "gradient_checkpointing": False,
        "normalization_sha256": arguments["normalization_hash"],
        "data_manifest_sha256": qualification.file_digest(data),
        "actual_policy_serving": {"passed": True},
        "exact_resume": {"raw_weights": True, "ema_weights": True, "optimizer_ema_rng_cursor": True},
        "update_seconds": 10.0,
        "checkpoint_seconds": 50.0,
    }
    path = tmp_path / "recovery.json"
    path.write_text(json.dumps(report))
    assert qualification.verify_recovery(path, **arguments) == report
    for changed in ({"microbatch_size": 8}, {"normalization_hash": "c" * 64}, {"gradient_checkpointing": True}):
        with pytest.raises(ValueError, match="differs"):
            qualification.verify_recovery(path, **{**arguments, **changed})
    report["exact_resume"]["optimizer_ema_rng_cursor"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="exact training state"):
        qualification.verify_recovery(path, **arguments)


def test_deadline_reserves_update_save_and_shutdown_including_initialization():
    assert qualification.can_start_update(1000, now=200, update_seconds=60, checkpoint_seconds=100)
    assert not qualification.can_start_update(1000, now=240, update_seconds=60, checkpoint_seconds=100)
    assert not qualification.can_start_update(1000, now=401, update_seconds=0, checkpoint_seconds=0)
    with pytest.raises(ValueError, match="at least"):
        qualification.can_start_update(1000, now=200, update_seconds=1, checkpoint_seconds=1, reserve_seconds=599)


def test_math_only_cli_does_not_touch_corpus_or_policy_loader(tmp_path, monkeypatch):
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    destination = tmp_path / "math.json"
    seen = []
    monkeypatch.setattr(qualify_potential, "_qualify", lambda args, report: seen.append(args.math_only))
    monkeypatch.setattr(
        qualify_potential, "_qualify_policy", lambda *args: pytest.fail("Math gate must not load serving/data")
    )
    assert qualify_potential.main(["--checkpoint", str(tmp_path), "--math-only", "--report", str(destination)]) == 0
    assert seen == [True]
    assert json.loads(destination.read_text())["math_only"]


def test_recovery_exact_comparison_catches_optimizer_rng_and_weights(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    state = {
        "optimizer_updates": 2,
        "training_seconds": 1.0,
        "optimizer": {"moment": torch.tensor([1.0])},
        "ema": {"weight": torch.tensor([2.0])},
        "rng": {"numpy": ("MT19937", np.array([1, 2], dtype=np.uint32)), "torch": torch.tensor([3], dtype=torch.uint8)},
        "consumed_microbatches": 128,
    }
    for directory in (left, right):
        directory.mkdir()
        for filename in ("raw_model.safetensors", "model.safetensors"):
            safetensors.torch.save_file({"weight": torch.tensor([1.0])}, directory / filename)
        torch.save(state, directory / "training_state.pt")
    other = copy.deepcopy(state)
    other["training_seconds"] = 99
    torch.save(other, right / "training_state.pt")
    assert all(recovery.compare_checkpoints(left, right).values())
    other["optimizer"]["moment"].add_(1)
    torch.save(other, right / "training_state.pt")
    with pytest.raises(ValueError, match="optimizer.moment"):
        recovery.compare_checkpoints(left, right)
    torch.save(state, right / "training_state.pt")
    safetensors.torch.save_file({"weight": torch.tensor([2.0])}, right / "model.safetensors")
    with pytest.raises(ValueError, match="model.safetensors.weight"):
        recovery.compare_checkpoints(left, right)


def test_recovery_stage_stops_before_launch_when_only_shutdown_time_remains(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery.time, "time", lambda: 100)
    monkeypatch.setattr(recovery.subprocess, "run", lambda *a, **kw: pytest.fail("Deadline must prevent launch"))
    with pytest.raises(TimeoutError):
        recovery.run_stage(["python"], tmp_path / "log", 700)


def test_recovery_dry_run_needs_no_assets_or_torch_import(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/qualify_potential_recovery.py"
    command = [sys.executable, "-S", str(script)]
    for name in ("checkpoint", "data-manifest", "selection", "qualification-report", "corpus", "output-root", "report"):
        command.extend((f"--{name}", str(tmp_path / name)))
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["dry_run"]
    assert not list(tmp_path.iterdir())


def test_recovery_commands_keep_diagnostics_separate_and_resume_raw(tmp_path):
    args = argparse.Namespace(
        output_root=tmp_path,
        data_manifest=tmp_path / "data",
        qualification_report=tmp_path / "qual",
        checkpoint=tmp_path / "base",
        num_workers=2,
        deadline_unix=1234.0,
    )
    command = recovery.trainer_command(
        args, {"microbatch_size": 4, "gradient_checkpointing": True}, name="resumed", updates=2, resume=tmp_path / "1"
    )
    assert "--recovery-probe" in command
    assert "--resume" in command
    assert "--weights" not in command
    assert "--gradient-checkpointing" in command
    assert command[command.index("--deadline-unix") + 1] == "1234.0"


@pytest.mark.parametrize("comparison_fails", [False, True])
def test_recovery_requires_all_three_updates_then_actual_serving(tmp_path, monkeypatch, comparison_fails):
    args = argparse.Namespace(
        output_root=tmp_path / "runs",
        report=tmp_path / "receipt.json",
        checkpoint=tmp_path / "base",
        qualification_report=tmp_path / "qualification.json",
        selection=tmp_path / "selection.json",
        corpus=tmp_path / "corpus.npz",
        data_manifest=tmp_path / "data.json",
        num_workers=0,
        deadline_unix=None,
    )
    for path in (args.qualification_report, args.selection, args.corpus, args.data_manifest):
        path.write_text("fixture")
    monkeypatch.setattr(
        recovery,
        "verify_inputs",
        lambda args: (
            {"identity": {"base": "a"}, "normalization_sha256": "b" * 64},
            {"microbatch_size": 2, "gradient_checkpointing": False},
        ),
    )
    commands = []

    def stage(command, log, deadline):
        commands.append(command)
        if "--serving-only" in command:
            path = Path(command[command.index("--report") + 1])
            path.write_text(json.dumps({"correctness_checks_passed": True, "actual_policy_serving": {"passed": True}}))
            return
        name = command[command.index("--exp-name") + 1]
        updates = int(command[command.index("--updates") + 1])
        directory = args.output_root / recovery.CONFIG / name
        checkpoint = directory / str(updates)
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(b"weights")
        with (directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"seconds_per_update": float(updates)}) + "\n")
        with (directory / "checkpoint_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"seconds": float(updates * 10), "checkpoint": str(checkpoint)}) + "\n")
        (directory / "profile.json").write_text(
            json.dumps({"peak_reserved_bytes": 100, "startup": {"total_seconds": 2}})
        )

    def compare(*args):
        if comparison_fails:
            raise ValueError("Recovery differs: optimizer")
        return {"raw_weights": True, "ema_weights": True, "optimizer_ema_rng_cursor": True}

    monkeypatch.setattr(recovery, "run_stage", stage)
    monkeypatch.setattr(recovery, "compare_checkpoints", compare)
    report = recovery.qualify(args)
    assert report["passed"] is not comparison_fails
    assert args.report.is_file()
    assert len(commands) == (3 if comparison_fails else 4)
    if not comparison_fails:
        assert report["update_seconds"] == 2.0
        assert report["checkpoint_seconds"] == 20.0
        assert report["actual_policy_serving"]["passed"]
