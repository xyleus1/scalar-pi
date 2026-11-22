"""Portable command planning does not start containers or change the study matrix."""

import os
from pathlib import Path
import subprocess

import pytest

from openpi.research import evaluation
from scripts import potential_matrix

ROOT = Path(__file__).resolve().parents[2]


def test_rendered_full_matrix_preserves_original_cells(tmp_path):
    timing = tmp_path / "timing.json"
    manifest = potential_matrix.render_matrix(ROOT, tmp_path, "full", timing_report=timing)
    cells = evaluation.matrix_cells(manifest)
    assert len(cells) == 80
    assert manifest["training_seeds"] == [17, 42, 93]
    assert manifest["num_trials_per_task"] == 50
    assert manifest["num_tasks_per_suite"] == 10
    assert manifest["timing_report"] == str(timing)
    runner = str(ROOT / "scripts/docker/run_potential.sh")
    assert manifest["server_command_prefix"] == ["bash", runner, "server", "--execute"]
    assert manifest["client_command_prefix"] == ["bash", runner, "simulator", "--execute"]
    for variant in manifest["variants"]:
        assert Path(variant["checkpoint_dir"]).is_relative_to(tmp_path / "data")
        if not variant.get("reference"):
            assert Path(variant["checkpoint_dir"]).name == "30000"


def test_pilot_and_smoke_remain_explicit_separate_definitions(tmp_path):
    pilot = potential_matrix.render_matrix(ROOT, tmp_path, "pilot")
    assert pilot["study_kind"] == "pilot"
    assert pilot["training_seeds"] == [42]
    smoke = potential_matrix.render_matrix(ROOT, tmp_path, "full", diagnostic=True, attempt="smoke")
    assert smoke["diagnostic"]
    assert all(variant.get("reference") for variant in smoke["variants"])
    assert smoke["num_tasks_per_suite"] == smoke["num_trials_per_task"] == 1
    with pytest.raises(ValueError, match="Attempt"):
        potential_matrix.render_matrix(ROOT, tmp_path, "full", attempt="../other")


@pytest.mark.skipif(os.name != "posix", reason="The container runner targets Linux bash")
def test_container_preview_never_starts_docker(tmp_path):
    marker = tmp_path / "docker-called"
    docker = tmp_path / "docker"
    docker.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 99\n")
    docker.chmod(0o755)
    git = tmp_path / "git"
    git.write_text("#!/bin/sh\nprintf '%s\\n' '0000000000000000000000000000000000000000'\n")
    git.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/docker/run_potential.sh"), "server", "--port", "8001"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Preview only" in result.stdout
    assert not marker.exists()
