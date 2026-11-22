"""Calibration specifications preserve measured profiles and retained smoke evidence."""

import copy
import json
from pathlib import Path

import pytest

from scripts import prepare_potential_calibration as spec


@pytest.fixture
def measured(tmp_path):
    reports = tmp_path / "data/experiments/reports"
    reports.mkdir(parents=True)
    (reports / "evaluation_calibration.json").write_text("{}")
    profiles = {
        group: {
            "measurement_source": f"/data/experiments/profiles/{config}/attempt/metrics.jsonl",
            "completed_updates": 110,
        }
        for group, config in spec.CONFIGS.items()
    }
    return profiles, reports


def test_all_profiles_and_selection_are_reverified(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(spec.selection, "verify_report", lambda args: calls.append("selection"))

    def profile(args):
        calls.append(args.config)
        return {"config": args.config}

    monkeypatch.setattr(spec.selection, "verify_profile", profile)
    result = spec.verified_profiles(tmp_path)
    assert calls == ["selection", *spec.CONFIGS.values()]
    assert set(result) == {"B0", "B1", "P"}


def test_calibration_spec_reuses_exact_profile_boundaries(measured, monkeypatch, tmp_path):
    profiles, reports = measured
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


def test_calibration_requires_explicit_smoke_when_multiple_complete(measured, monkeypatch, tmp_path):
    profiles, reports = measured
    (reports / "evaluation_calibration.json").unlink()
    for name in ("one", "two"):
        path = reports / "smoke" / name / "matrix.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    monkeypatch.setattr(spec, "verified_profiles", lambda _data: profiles)
    monkeypatch.setattr(spec.calibration, "smoke_overhead", lambda _path: (0.01, []))
    with pytest.raises(ValueError, match="Exactly one completed smoke"):
        spec.prepare(tmp_path, attempt="first")


def test_calibration_rechecks_existing_aggregate(measured, monkeypatch, tmp_path):
    profiles, reports = measured
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
