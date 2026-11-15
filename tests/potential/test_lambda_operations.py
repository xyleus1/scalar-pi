"""Operational phases consume measured evidence without launching paid work."""

import copy
import importlib
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/lambda"))
    budget = importlib.import_module("measured_budget")
    spec = importlib.import_module("calibration_spec")
    return budget, spec


@pytest.fixture
def measured(modules, tmp_path, monkeypatch):
    budget, _ = modules
    reports = tmp_path / "data/experiments/reports"
    (reports / "profiles").mkdir(parents=True)
    (reports / "evaluation_calibration.json").write_text("{}")
    evidence = tmp_path / "cost-evidence.json"
    evidence.write_text('{"fixture_only": true}')
    common = {"measurement_source": str(evidence), "evidence": {str(evidence): budget.digest(evidence)}}
    inputs = {
        "schema_version": 1,
        "session_hours": 4,
        "accounting": {
            **common,
            "as_of_unix": 200,
            "incurred_cost_usd": 21,
            "current_session_cost_ceiling_usd": 7,
            "future_nonstorage_reserve_usd": 2,
        },
        "platform": {**common, "startup_overhead_seconds": 70, "shutdown_reserve_seconds": 600},
        "storage": {
            **common,
            "gib_month_rate_usd": 0.2,
            "current_gib": 300,
            "additional_gib": 500,
            "retention_hours": 1000,
        },
    }
    identity = {"code_commit": "a" * 40, "container_image_id": "sha256:" + "b" * 64}
    profiles = {}
    for index, (group, config) in enumerate(budget.CONFIGS.items(), 1):
        profiles[group] = {
            "identity": identity.copy(),
            "hardware": "TEST FIXTURE",
            "seconds_per_optimizer_update": {config: index * 2},
            "checkpoint_seconds": [index * 10, index * 11],
            "startup": {"total_seconds": index * 30},
            "measurement_source": f"/data/experiments/profiles/{config}/attempt/metrics.jsonl",
            "completed_updates": 110,
        }
        (reports / "profiles" / f"{config}.json").write_text(json.dumps(profiles[group]))
    for path in reports.rglob("*.json"):
        os.utime(path, (100, 100))
    calls = []
    monkeypatch.setattr(
        budget.preflight, "calibration_evidence", lambda record, required: calls.append((record, required))
    )
    return profiles, inputs, reports, calls


def test_budget_uses_measured_timings_and_explicit_costs(modules, measured):
    budget, _ = modules
    profiles, inputs, reports, calls = measured
    result = budget.assemble(profiles, inputs, reports=reports, forecast_rate=3.79, now=300)
    assert result["seconds_per_optimizer_update"] == {"B0": 2, "B1": 4, "P": 6}
    assert result["checkpoint_seconds_per_save"] == {"B0": 11, "B1": 22, "P": 33}
    assert result["checkpoint_interval_updates"] == {"B0": 1800, "B1": 900, "P": 600}
    assert result["startup_seconds_per_session"] == 160
    assert result["fixed_cost_usd"] == 30
    assert result["completed_training"] == result["completed_evaluation"] == []
    assert calls == [(result, True)]
    assert len(result["profile_sha256"]) == 3


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("accounting", "incurred_cost_usd", None),
        ("accounting", "current_session_cost_ceiling_usd", 0),
        ("accounting", "future_nonstorage_reserve_usd", float("nan")),
        ("platform", "startup_overhead_seconds", None),
        ("platform", "shutdown_reserve_seconds", 599),
        ("storage", "gib_month_rate_usd", None),
        ("storage", "additional_gib", -1),
        ("storage", "evidence", {}),
    ],
)
def test_budget_rejects_missing_or_invalid_operating_inputs(modules, measured, section, field, value):
    budget, _ = modules
    profiles, inputs, reports, _ = measured
    inputs[section][field] = value
    with pytest.raises(ValueError, match="must|requires|Shutdown"):
        budget.assemble(profiles, inputs, reports=reports, forecast_rate=3.79, now=300)


def test_budget_rejects_changed_cost_evidence(modules, measured):
    budget, _ = modules
    profiles, inputs, reports, _ = measured
    Path(next(iter(inputs["accounting"]["evidence"]))).write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        budget.assemble(profiles, inputs, reports=reports, forecast_rate=3.79, now=300)


def test_budget_requires_accounting_after_prerequisites(modules, measured):
    budget, _ = modules
    profiles, inputs, reports, _ = measured
    inputs["accounting"]["as_of_unix"] = 99
    with pytest.raises(ValueError, match="refreshed after"):
        budget.assemble(profiles, inputs, reports=reports, forecast_rate=3.79, now=300)


def test_budget_rejects_profile_identity_mismatch(modules, measured):
    budget, _ = modules
    profiles, inputs, reports, _ = measured
    profiles["P"]["identity"]["code_commit"] = "c" * 40
    with pytest.raises(ValueError, match="identities differ"):
        budget.assemble(profiles, inputs, reports=reports, forecast_rate=3.79, now=300)


def test_budget_propagates_calibration_verification_failure(modules, measured, monkeypatch):
    budget, _ = modules
    profiles, inputs, reports, _ = measured

    def reject(*_args, **_kwargs):
        raise ValueError("calibration changed")

    monkeypatch.setattr(budget.preflight, "calibration_evidence", reject)
    with pytest.raises(ValueError, match="calibration changed"):
        budget.assemble(profiles, inputs, reports=reports, forecast_rate=3.79, now=300)


def test_all_profiles_and_selection_are_reverified(modules, tmp_path, monkeypatch):
    budget, _ = modules
    calls = []
    monkeypatch.setattr(budget.selection, "verify_report", lambda args: calls.append("selection"))

    def profile(args):
        calls.append(args.config)
        return {"config": args.config}

    monkeypatch.setattr(budget.selection, "verify_profile", profile)
    result = budget.verified_profiles(tmp_path)
    assert calls == ["selection", *budget.CONFIGS.values()]
    assert set(result) == {"B0", "B1", "P"}


def test_calibration_spec_reuses_exact_profile_boundaries(modules, measured, monkeypatch, tmp_path):
    _, spec = modules
    profiles, _, reports, _ = measured
    (reports / "evaluation_calibration.json").unlink()
    smoke = reports / "smoke/compatibility/matrix.json"
    smoke.parent.mkdir(parents=True)
    smoke.write_text("{}")
    monkeypatch.setattr(spec, "verified_profiles", lambda _data: profiles)
    monkeypatch.setattr(spec.calibration, "smoke_overhead", lambda _path: (0.01, []))
    path = spec.prepare(tmp_path, attempt="first")
    before = path.read_bytes()
    spec.prepare(tmp_path, attempt="second")
    assert path.read_bytes() == before
    result = json.loads(before)
    assert result["output_dir"].endswith("calibration/first") or result["output_dir"].endswith("calibration\\first")
    for policy in result["policies"]:
        if policy["group"] != "reference":
            assert Path(policy["checkpoint_dir"]).name == "110"
    profiles["P"]["measurement_source"] = "/data/experiments/profiles/changed/metrics.jsonl"
    with pytest.raises(ValueError, match="specification differs"):
        spec.prepare(tmp_path, attempt="third")


def test_calibration_requires_explicit_smoke_when_multiple_complete(modules, measured, monkeypatch, tmp_path):
    _, spec = modules
    profiles, _, reports, _ = measured
    (reports / "evaluation_calibration.json").unlink()
    for name in ("one", "two"):
        path = reports / "smoke" / name / "matrix.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    monkeypatch.setattr(spec, "verified_profiles", lambda _data: profiles)
    monkeypatch.setattr(spec.calibration, "smoke_overhead", lambda _path: (0.01, []))
    with pytest.raises(ValueError, match="Exactly one completed smoke"):
        spec.prepare(tmp_path, attempt="first")


def test_calibration_rechecks_existing_aggregate(modules, measured, monkeypatch, tmp_path):
    _, spec = modules
    profiles, _, reports, _ = measured
    smoke = reports / "smoke/one/matrix.json"
    smoke.parent.mkdir(parents=True)
    smoke.write_text("{}")
    monkeypatch.setattr(spec, "verified_profiles", lambda _data: profiles)
    monkeypatch.setattr(spec.calibration, "smoke_overhead", lambda _path: (0.01, []))
    calls = []
    monkeypatch.setattr(spec, "verify_aggregate", lambda value, path: calls.append((copy.deepcopy(value), path)))
    spec.prepare(tmp_path, attempt="first")
    assert len(calls) == 1
    assert calls[0][1] == reports / "evaluation_calibration.json"
