"""Remaining-work forecasts use real-shaped artifacts, never completion counters."""

import hashlib
import json
from pathlib import Path
import runpy

import pytest
from test_evaluation import _records
from test_evaluation import _write_rows

from openpi.research import evaluation

ROOT = Path(__file__).resolve().parents[2]
choose_study = runpy.run_path(str(ROOT / "scripts/potential_preflight.py"))["choose_study"]


def _profile():
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    return {
        "measured_gpu": True,
        "model_kind": "pi05_full",
        "hardware": "CPU TEST FIXTURE ONLY",
        "measurement_source": "fixture",
        "evaluation_measurement_source": "fixture",
        "warmup_optimizer_updates": 10,
        "timed_optimizer_updates": 100,
        "effective_batch_size": 256,
        "optimizer_updates": 30000,
        "seconds_per_optimizer_update": {"B0": 1, "B1": 2, "P": 3},
        "checkpoint_seconds_per_save": {"B0": 20, "B1": 30, "P": 40},
        "checkpoint_interval_updates": {"B0": 5000, "B1": 5000, "P": 5000},
        "fixed_cost_usd": 100,
        "evaluation_seconds_per_episode": {
            group: {steps: dict.fromkeys(suites, 10) for steps in ("1", "10")}
            for group in ("B0", "B1", "P", "reference")
        },
    }


def _checkpoint(tmp_path, group="B0", seed=42, updates=5000):
    directory = tmp_path / f"{group}-{seed}-{updates}"
    directory.mkdir()
    metadata = {
        "schema_version": 1,
        "trainer": "potential_pi",
        "run_mode": "training",
        "training_seed": seed,
        "optimizer_updates": updates,
        "inference_weights": "ema",
        "velocity_mode": "potential_dot" if group == "P" else "direct",
        "initialization": "pretrained_output" if group == "B0" else "zero_output",
        "freeze_vlm": True,
        "action_horizon": 10,
        "action_dim": 32,
        "code_commit": "a" * 40,
        "container_image_id": "sha256:" + "b" * 64,
    }
    for filename, field in (("model.safetensors", "model_sha256"), ("raw_model.safetensors", "raw_model_sha256")):
        payload = f"fixture:{filename}:{group}:{seed}:{updates}".encode()
        (directory / filename).write_bytes(payload)
        metadata[field] = hashlib.sha256(payload).hexdigest()
    (directory / "potential_metadata.json").write_text(json.dumps(metadata))
    (directory / "training_state.pt").write_bytes(b"fixture resumable state")
    return {"group": group, "training_seed": seed, "optimizer_updates": updates, "checkpoint_dir": str(directory)}


def _eval_artifact(tmp_path, *, group="B0", mutate=None):
    manifest = json.loads((ROOT / "configs/potential/libero_matrix.pilot.json").read_text())
    manifest["output_dir"] = str(tmp_path / "episodes")
    cell = next(cell for cell in evaluation.matrix_cells(manifest) if cell["group"] == group)
    subset = {
        **manifest,
        "diagnostic": True,
        "variants": [variant for variant in manifest["variants"] if variant["group"] == group],
        "num_steps": [cell["num_steps"]],
        "suites": [cell["suite"]],
    }
    rows = _records(subset)
    if mutate:
        mutate(rows)
    _write_rows(cell["episode_log_path"], rows)
    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(json.dumps(manifest))
    return {
        "group": group,
        "training_seed": cell["training_seed"],
        "num_steps": cell["num_steps"],
        "suite": cell["suite"],
        "manifest_path": str(manifest_path),
    }


def test_partial_run_reduces_remaining_work_and_preserves_planned_exposure(tmp_path):
    profile = _profile()
    before = choose_study(profile, 4, 2000)
    profile["completed_training"] = [_checkpoint(tmp_path, updates=6000)]
    after = choose_study(profile, 4, 2000)
    for study, planned in (("full", 270000), ("pilot", 90000)):
        forecast = after["studies"][study]
        assert forecast["planned_optimizer_updates"] == planned
        assert forecast["completed_optimizer_updates"] == 6000
        assert forecast["remaining_optimizer_updates"] == planned - 6000
        assert forecast["training_gpu_hours"] == pytest.approx(
            before["studies"][study]["training_gpu_hours"] - 6000 / 3600
        )
        assert forecast["remaining_checkpoint_saves"] == before["studies"][study]["remaining_checkpoint_saves"] - 1
        assert forecast["checkpoint_gpu_hours"] == pytest.approx(
            before["studies"][study]["checkpoint_gpu_hours"] - 20 / 3600
        )
        assert forecast["projected_total_usd"] == pytest.approx(
            forecast["remaining_compute_cost_with_headroom_usd"] + 100
        )
    assert after["validated_progress"][0]["metadata_sha256"]
    assert after["checkpoint_interval_updates"] == profile["checkpoint_interval_updates"]
    assert (
        after["remaining_compute_cost_with_headroom_usd"]
        == after["studies"]["full"]["remaining_compute_cost_with_headroom_usd"]
    )


def test_other_training_seeds_reduce_full_but_not_fixed_seed_pilot(tmp_path):
    profile = _profile()
    profile["completed_training"] = [_checkpoint(tmp_path, group="P", seed=17, updates=30000)]
    result = choose_study(profile, 4, 2000)
    assert result["studies"]["full"]["completed_optimizer_updates"] == 30000
    assert result["studies"]["pilot"]["completed_optimizer_updates"] == 0
    assert result["studies"]["full"]["remaining_checkpoint_saves"] == 48
    assert result["studies"]["pilot"]["remaining_checkpoint_saves"] == 18


def test_final_checkpoint_does_not_leave_an_extra_save(tmp_path):
    profile = _profile()
    profile["checkpoint_interval_updates"] = dict.fromkeys(("B0", "B1", "P"), 4001)
    profile["completed_training"] = [_checkpoint(tmp_path, group=group, updates=30000) for group in ("B0", "B1", "P")]
    pilot = choose_study(profile, 4, 2000)["studies"]["pilot"]
    assert pilot["remaining_optimizer_updates"] == 0
    assert pilot["remaining_checkpoint_saves"] == 0
    assert pilot["checkpoint_gpu_hours"] == 0
    assert pilot["remaining_evaluation_episodes"] == 16000
    assert pilot["remaining_compute_cost_with_headroom_usd"] > 0


@pytest.mark.parametrize(("updates", "expected_saves"), [(4000, 8), (4001, 7), (29999, 1)])
def test_checkpoint_cadence_includes_terminal_save(tmp_path, updates, expected_saves):
    profile = _profile()
    profile["checkpoint_interval_updates"]["B0"] = 4001
    profile["completed_training"] = [_checkpoint(tmp_path, updates=updates)]
    pilot = choose_study(profile, 4, 2000)["studies"]["pilot"]
    assert pilot["remaining_checkpoint_saves"] == expected_saves + 12


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("group", "other"),
        ("training_seed", 0),
        ("training_seed", True),
        ("optimizer_updates", True),
        ("optimizer_updates", 0),
        ("optimizer_updates", 30001),
    ],
)
def test_rejects_unsupported_or_untyped_training_declarations(field, value):
    record = {"group": "B0", "training_seed": 42, "optimizer_updates": 5000, field: value}
    profile = {**_profile(), "completed_training": [record]}
    with pytest.raises(ValueError, match="unsupported|optimizer_updates"):
        choose_study(profile, 4, 2000)


def test_requires_actual_checkpoint_and_rejects_duplicate_runs(tmp_path):
    profile = _profile()
    record = _checkpoint(tmp_path)
    profile["completed_training"] = [record, record]
    with pytest.raises(ValueError, match="Duplicate"):
        choose_study(profile, 4, 2000)
    profile["completed_training"] = [{key: value for key, value in record.items() if key != "checkpoint_dir"}]
    with pytest.raises(ValueError, match="checkpoint_dir"):
        choose_study(profile, 4, 2000)


@pytest.mark.parametrize("corruption", ["profile", "update_count", "changed_weight", "missing_state", "zero_state"])
def test_no_unverified_training_counter_can_reduce_cost(tmp_path, corruption):
    record = _checkpoint(tmp_path)
    directory = Path(record["checkpoint_dir"])
    metadata_path = directory / "potential_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if corruption == "profile":
        metadata["run_mode"] = "profile"
    elif corruption == "update_count":
        metadata["optimizer_updates"] = 100
    elif corruption == "changed_weight":
        (directory / "model.safetensors").write_bytes(b"changed")
    elif corruption == "missing_state":
        (directory / "training_state.pt").unlink()
    else:
        (directory / "training_state.pt").write_bytes(b"")
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="disagrees|hash mismatch|resumable state"):
        choose_study({**_profile(), "completed_training": [record]}, 4, 2000)


def test_optional_identity_binds_completed_artifacts(tmp_path):
    profile = {**_profile(), "completed_training": [_checkpoint(tmp_path)]}
    profile["experiment_identity"] = {"code_commit": "a" * 40, "container_image_id": "sha256:" + "b" * 64}
    assert choose_study(profile, 4, 2000)["experiment_identity"] == profile["experiment_identity"]
    profile["experiment_identity"]["code_commit"] = "c" * 40
    with pytest.raises(ValueError, match="different experiment"):
        choose_study(profile, 4, 2000)


@pytest.mark.parametrize("group", ["B0", "reference"])
def test_verified_suite_removes_exactly_500_episodes(tmp_path, group):
    profile = {**_profile(), "completed_evaluation": [_eval_artifact(tmp_path, group=group)]}
    result = choose_study(profile, 4, 2000)
    for study, planned in (("full", 40000), ("pilot", 16000)):
        forecast = result["studies"][study]
        assert forecast["planned_evaluation_episodes"] == planned
        assert forecast["completed_evaluation_episodes"] == 500
        assert forecast["remaining_evaluation_episodes"] == planned - 500
        assert forecast["evaluation_gpu_hours"] == pytest.approx((planned - 500) * 10 / 3600)
    assert result["validated_progress"][0]["episode_log_sha256"]


@pytest.mark.parametrize("failure", ["incomplete", "duplicate", "infrastructure", "diagnostic"])
def test_incomplete_or_invalid_suite_never_reduces_cost(tmp_path, failure):
    def mutate(rows):
        if failure == "incomplete":
            rows.pop()
        elif failure == "duplicate":
            rows.append(rows[0])
        elif failure == "infrastructure":
            rows[0]["failure_kind"] = "infrastructure"

    record = _eval_artifact(tmp_path, mutate=mutate)
    if failure == "diagnostic":
        path = Path(record["manifest_path"])
        manifest = json.loads(path.read_text())
        manifest["diagnostic"] = True
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="[Ii]ncomplete|[Dd]uplicate|[Ii]nfrastructure|[Dd]iagnostic|[Mm]issing"):
        choose_study({**_profile(), "completed_evaluation": [record]}, 4, 2000)


def test_duplicate_completion_records_do_not_count_twice(tmp_path):
    record = _eval_artifact(tmp_path)
    with pytest.raises(ValueError, match="Duplicate completed evaluation"):
        choose_study({**_profile(), "completed_evaluation": [record, record]}, 4, 2000)
