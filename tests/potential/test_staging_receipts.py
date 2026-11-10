"""CPU-only staging receipt and interrupted-producer tests."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def staging(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/lambda/staging.py"
    spec = importlib.util.spec_from_file_location("staging", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module, "producer", lambda: {"code_commit": "a" * 40, "container_image_id": "sha256:" + "b" * 64}
    )
    return module


def arguments(tmp_path):
    source = tmp_path / "source"
    source.write_text("source")
    return [
        "run",
        "--stage",
        "fixture",
        "--receipt",
        str(tmp_path / "receipts/stage.json"),
        "--input",
        f"source={source}",
        "--output",
        f"result={tmp_path / 'result'}",
        "--parameter",
        "mode=fp32",
        "--",
        "producer",
    ]


def test_reuses_identical_artifacts_without_rerunning(staging, tmp_path, monkeypatch):
    args = arguments(tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        (tmp_path / "result").write_text("result")

    monkeypatch.setattr(staging.subprocess, "run", run)
    staging.main(args)
    staging.main(args)
    assert calls == [["producer"]]
    assert json.loads((tmp_path / "receipts/stage.json").read_text())["producer_seconds"] >= 0


@pytest.mark.parametrize("changed", ["source", "result", "producer"])
def test_changed_evidence_never_triggers_expensive_automatic_rerun(staging, tmp_path, monkeypatch, changed):
    args = arguments(tmp_path)
    monkeypatch.setattr(staging.subprocess, "run", lambda *a, **kw: (tmp_path / "result").write_text("result"))
    staging.main(args)
    if changed == "producer":
        monkeypatch.setattr(staging, "producer", lambda: {"code_commit": "c" * 40})
    else:
        (tmp_path / changed).write_text("changed")
    monkeypatch.setattr(staging.subprocess, "run", lambda *a, **kw: pytest.fail("must not rerun changed stage"))
    with pytest.raises(ValueError, match="changed"):
        staging.main(args)


def test_failed_owned_producer_can_retry_but_unowned_output_cannot(staging, tmp_path, monkeypatch):
    args = arguments(tmp_path)

    def fail(*args, **kwargs):
        (tmp_path / "result").write_text("partial")
        raise subprocess.CalledProcessError(1, "producer")

    monkeypatch.setattr(staging.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        staging.main(args)
    assert not (tmp_path / "receipts/stage.json").exists()
    monkeypatch.setattr(staging.subprocess, "run", lambda *a, **kw: (tmp_path / "result").write_text("complete"))
    staging.main(args)
    (tmp_path / "receipts/stage.json").unlink()
    with pytest.raises(FileExistsError, match="Unreceipted"):
        staging.main(args)


def test_directory_inventory_detects_added_or_deleted_files(staging, tmp_path):
    path = tmp_path / "tree"
    path.mkdir()
    (path / "a").write_text("a")
    first = staging.inventory(path)
    (path / "b").write_text("b")
    assert staging.inventory(path) != first
