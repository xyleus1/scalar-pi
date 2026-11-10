"""Terminal reuse checks read small fixture bytes; no model or checkpoint loading."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest


@pytest.fixture
def verifier():
    path = Path(__file__).resolve().parents[2] / "scripts/lambda/verify_training.py"
    spec = importlib.util.spec_from_file_location("verify_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artifact(tmp_path, config="pi05_libero_potential"):
    directory = tmp_path / "30000"
    directory.mkdir()
    normalization = b"{}"
    selection = {
        "code": {"commit": "a" * 40, "source_snapshot_sha256": "b" * 64},
        "container_image_id": "sha256:" + "c" * 64,
        "inputs_sha256": {"weights": "d" * 64, "conversion_receipt": "e" * 64, "data_manifest": "f" * 64},
        "normalization_sha256": hashlib.sha256(normalization).hexdigest(),
        "microbatch_size": 4,
        "gradient_checkpointing": False,
    }
    metadata = {
        "schema_version": 1,
        "trainer": "potential_pi",
        "run_mode": "training",
        "training_seed": 42,
        "optimizer_updates": 30000,
        "inference_weights": "ema",
        "velocity_mode": "potential_dot" if config == "pi05_libero_potential" else "direct",
        "initialization": "pretrained_output" if config == "pi05_libero_b0" else "zero_output",
        "action_horizon": 10,
        "action_dim": 32,
        "freeze_vlm": True,
        "expert_precision": "float32",
        "vlm_precision": "bfloat16",
        "matmul_precision": "highest",
        "code_commit": selection["code"]["commit"],
        "source_snapshot_sha256": selection["code"]["source_snapshot_sha256"],
        "container_image_id": selection["container_image_id"],
        "base_checkpoint_hash": selection["inputs_sha256"]["weights"],
        "base_conversion_receipt_sha256": selection["inputs_sha256"]["conversion_receipt"],
        "data_manifest_sha256": selection["inputs_sha256"]["data_manifest"],
        "normalization_hash": selection["normalization_sha256"],
    }
    metadata["resume_signature"] = {
        "config": config,
        "seed": 42,
        "microbatch_size": 4,
        "effective_batch_size": 256,
        "gradient_checkpointing": False,
        "deterministic": True,
        **{
            key: metadata[key]
            for key in (
                "initialization",
                "matmul_precision",
                "run_mode",
                "code_commit",
                "source_snapshot_sha256",
                "container_image_id",
            )
        },
    }
    header = json.dumps(
        {
            "__metadata__": {
                key: str(metadata[key])
                for key in ("schema_version", "trainer", "velocity_mode", "action_horizon", "action_dim")
            }
        }
    ).encode()
    for filename, field in (("model.safetensors", "model_sha256"), ("raw_model.safetensors", "raw_model_sha256")):
        payload = len(header).to_bytes(8, "little") + header + b"fixture tensor bytes"
        (directory / filename).write_bytes(payload)
        metadata[field] = hashlib.sha256(payload).hexdigest()
    (directory / "training_state.pt").write_bytes(b"not deserialized")
    norms = directory / "assets/physical-intelligence/libero/norm_stats.json"
    norms.parent.mkdir(parents=True)
    norms.write_bytes(normalization)
    (directory / "potential_metadata.json").write_text(json.dumps(metadata))
    return directory, selection


@pytest.mark.parametrize("config", ["pi05_libero_b0", "pi05_libero_b1", "pi05_libero_potential"])
def test_exact_terminal_run_can_be_reused(verifier, tmp_path, config):
    directory, selection = artifact(tmp_path, config)
    result = verifier.verify_terminal(directory, config, 42, selection)
    assert result["verified"]
    assert result["optimizer_updates"] == 30000
    assert result["metadata_sha256"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("optimizer_updates", 29999),
        ("run_mode", "profile"),
        ("training_seed", 17),
        ("code_commit", "1" * 40),
        ("source_snapshot_sha256", "1" * 64),
        ("container_image_id", "sha256:" + "1" * 64),
        ("base_checkpoint_hash", "1" * 64),
        ("base_conversion_receipt_sha256", "1" * 64),
        ("data_manifest_sha256", "1" * 64),
        ("normalization_hash", "1" * 64),
        ("matmul_precision", "high"),
    ],
)
def test_stale_or_nonterminal_checkpoint_is_rejected(verifier, tmp_path, field, value):
    directory, selection = artifact(tmp_path)
    path = directory / "potential_metadata.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=field):
        verifier.verify_terminal(directory, "pi05_libero_potential", 42, selection)


@pytest.mark.parametrize("field", ["microbatch_size", "gradient_checkpointing", "config"])
def test_changed_training_protocol_is_rejected(verifier, tmp_path, field):
    directory, selection = artifact(tmp_path)
    path = directory / "potential_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["resume_signature"][field] = {
        "microbatch_size": 8,
        "gradient_checkpointing": True,
        "config": "pi05_libero_b1",
    }[field]
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=field):
        verifier.verify_terminal(directory, "pi05_libero_potential", 42, selection)


@pytest.mark.parametrize(
    "filename", ["model.safetensors", "raw_model.safetensors", "assets/physical-intelligence/libero/norm_stats.json"]
)
def test_actual_checkpoint_bytes_are_checked(verifier, tmp_path, filename):
    directory, selection = artifact(tmp_path)
    with (directory / filename).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch|normalization asset"):
        verifier.verify_terminal(directory, "pi05_libero_potential", 42, selection)


def test_help_works_with_standard_library_only(verifier):
    result = subprocess.run(
        [sys.executable, "-S", verifier.__file__, "--help"], capture_output=True, text=True, check=True
    )
    assert "--selection" in result.stdout


def test_cli_verifies_base_inputs_separately_from_terminal_weights(verifier, tmp_path, monkeypatch, capsys):
    directory, selection = artifact(tmp_path)
    selection_script = Path(verifier.__file__).parents[1] / "select_potential_batch.py"
    selection_verifier = verifier.runpy.run_path(str(selection_script))["verify_report"]
    source = tmp_path / "source.py"
    source.write_text("# pinned experiment\n")
    sources = {"source.py": verifier.digest(source)}
    selection["code"].update(
        source_files_sha256=sources,
        source_snapshot_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
    )
    weights = tmp_path / "base"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"base, distinct from terminal")
    (weights / "config.json").write_text("{}")
    manifest = tmp_path / "data.json"
    manifest.write_text("{}")
    qualification = tmp_path / "qualification.json"
    qualification.write_text("{}")
    selection["inputs_sha256"] = {
        "weights": verifier.digest(weights / "model.safetensors"),
        "conversion_receipt": verifier.digest(weights / "config.json"),
        "data_manifest": verifier.digest(manifest),
        "qualification": verifier.digest(qualification),
    }
    selection.update(
        schema_version=1, kind="batch_selection", verified=True, effective_batch_size=256, matmul_precision="highest"
    )
    for config in verifier.CONFIGS:
        norm = tmp_path / "assets" / config / "physical-intelligence/libero/norm_stats.json"
        norm.parent.mkdir(parents=True)
        norm.write_text("{}")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    path = directory / "potential_metadata.json"
    metadata = json.loads(path.read_text())
    metadata.update(
        source_snapshot_sha256=selection["code"]["source_snapshot_sha256"],
        base_checkpoint_hash=selection["inputs_sha256"]["weights"],
        base_conversion_receipt_sha256=selection["inputs_sha256"]["conversion_receipt"],
        data_manifest_sha256=selection["inputs_sha256"]["data_manifest"],
    )
    metadata["resume_signature"]["source_snapshot_sha256"] = selection["code"]["source_snapshot_sha256"]
    path.write_text(json.dumps(metadata))
    monkeypatch.setenv("SCALAR_PI_IMAGE_ID", selection["container_image_id"])
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: types.SimpleNamespace(
            stdout=selection["code"]["commit"] if command[1] == "rev-parse" else ""
        ),
    )
    # Only redirect the repository root; run the real selection and terminal checks.
    monkeypatch.setattr(
        verifier.runpy,
        "run_path",
        lambda path: {"verify_report": lambda args: selection_verifier(args, repository=tmp_path)},
    )
    verifier.main(
        [
            "--checkpoint",
            str(directory),
            "--config",
            "pi05_libero_potential",
            "--seed",
            "42",
            "--selection",
            str(selection_path),
            "--weights",
            str(weights),
            "--data-manifest",
            str(manifest),
            "--qualification-report",
            str(qualification),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["verified"]
    assert report["checkpoint"] == str(directory)
