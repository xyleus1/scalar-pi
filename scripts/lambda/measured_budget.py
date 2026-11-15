"""Assemble a study forecast from verified GPU receipts and explicit cost evidence.

Run in the model container. This does not invent prices, platform startup costs,
taxes, storage projections or completed work; the operator supplies those inputs.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import potential_preflight as preflight
import select_potential_batch as selection

CONFIGS = {"B0": "pi05_libero_b0", "B1": "pi05_libero_b1", "P": "pi05_libero_potential"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verified_profiles(data):
    data = Path(data)
    reports = data / "experiments/reports"
    args = SimpleNamespace(
        weights=data / "weights/pi05_base_torch",
        data_manifest=data / "libero_manifest.json",
        qualification_report=reports / "gpu_qualification.json",
        recovery_report=reports / "recovery.json",
        verify_report=data / "experiments/batch_selection.json",
    )
    selection.verify_report(args)
    result = {}
    for group, config in CONFIGS.items():
        args.config = config
        args.verify_profile = reports / "profiles" / f"{config}.json"
        result[group] = selection.verify_profile(args)
    return result


def number(value, name, *, zero=False):
    return preflight.positive_number(value, name, allow_zero=zero)


def evidence(record, name):
    source = record.get("measurement_source")
    if not isinstance(source, str) or not source.strip() or "REPLACE" in source.upper():
        raise ValueError(f"{name} requires a concrete measurement_source")
    files = record.get("evidence")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{name} requires retained evidence file hashes")
    for path, expected in files.items():
        if digest(path) != expected:
            raise ValueError(f"{name} evidence changed: {path}")


def assemble(profiles, inputs, *, reports, forecast_rate, now=None):
    """Pure assembly after profile verification; calibration evidence is checked here."""
    reports = Path(reports)
    now = time.time() if now is None else now
    if inputs.get("schema_version") != 1:
        raise ValueError("Budget inputs require schema_version=1")
    accounting, platform, storage = (inputs.get(key, {}) for key in ("accounting", "platform", "storage"))
    for name, record in (("accounting", accounting), ("platform", platform), ("storage", storage)):
        if not isinstance(record, dict):
            raise ValueError(f"{name} must be an object")
        evidence(record, name)
    as_of = number(accounting.get("as_of_unix"), "accounting.as_of_unix")
    if as_of > now:
        raise ValueError("Accounting snapshot cannot be in the future")
    incurred = number(accounting.get("incurred_cost_usd"), "accounting.incurred_cost_usd", zero=True)
    session_allowance = number(
        accounting.get("current_session_cost_ceiling_usd"), "accounting.current_session_cost_ceiling_usd"
    )
    nonstorage = number(
        accounting.get("future_nonstorage_reserve_usd"), "accounting.future_nonstorage_reserve_usd", zero=True
    )
    overhead = number(platform.get("startup_overhead_seconds"), "platform.startup_overhead_seconds")
    shutdown = number(platform.get("shutdown_reserve_seconds"), "platform.shutdown_reserve_seconds")
    if shutdown < 600:
        raise ValueError("Shutdown reserve must be at least 600 measured/allowed seconds")
    for key in ("gib_month_rate_usd", "retention_hours"):
        number(storage.get(key), f"storage.{key}")
    for key in ("current_gib", "additional_gib"):
        number(storage.get(key), f"storage.{key}", zero=True)
    hours = number(inputs.get("session_hours"), "session_hours")
    if hours > 4:
        raise ValueError("Research sessions cannot exceed four hours")
    if set(profiles) != set(CONFIGS):
        raise ValueError("All three verified profiles are required")
    identities = [record["identity"] for record in profiles.values()]
    if any(identity != identities[0] for identity in identities):
        raise ValueError("Profile scientific identities differ")
    hardware = {record["hardware"] for record in profiles.values()}
    if len(hardware) != 1:
        raise ValueError("Profiles were measured on different hardware")
    timings, saves, intervals, startup = {}, {}, {}, []
    for group, config in CONFIGS.items():
        profile = profiles[group]
        timings[group] = number(profile["seconds_per_optimizer_update"][config], f"{group} update time")
        publications = profile.get("checkpoint_seconds", [])
        if not publications:
            raise ValueError(f"{group} lacks measured checkpoint publication")
        saves[group] = max(number(value, f"{group} publication time") for value in publications)
        intervals[group] = max(1, min(5000, math.floor(3600 / timings[group])))
        startup.append(number(profile.get("startup", {}).get("total_seconds"), f"{group} startup"))
    identity = {key: identities[0][key] for key in ("code_commit", "container_image_id")}
    calibration = reports / "evaluation_calibration.json"
    profile_paths = [reports / "profiles" / f"{config}.json" for config in CONFIGS.values()]
    if as_of < max(path.stat().st_mtime for path in [calibration, *profile_paths]):
        raise ValueError("Accounting must be refreshed after profiling and calibration finish")
    result = {
        "measured_gpu": True,
        "model_kind": "pi05_full",
        "hardware": next(iter(hardware)),
        "measurement_source": "; ".join(str(path) for path in profile_paths),
        "profile_sha256": {str(path): digest(path) for path in profile_paths},
        "evaluation_measurement_source": str(calibration),
        "warmup_optimizer_updates": 10,
        "timed_optimizer_updates": 100,
        "effective_batch_size": 256,
        "optimizer_updates": 30000,
        "seconds_per_optimizer_update": timings,
        "checkpoint_seconds_per_save": saves,
        "checkpoint_interval_updates": intervals,
        "startup_seconds_per_session": overhead + max(startup),
        "shutdown_reserve_seconds": shutdown,
        "forecast_hourly_rate_usd": number(forecast_rate, "forecast_hourly_rate_usd"),
        "session_hours": hours,
        "fixed_cost_usd": incurred + session_allowance + nonstorage,
        "storage": storage,
        "experiment_identity": identity,
        "calibration_report": str(calibration),
        "completed_training": inputs.get("completed_training", []),
        "completed_evaluation": inputs.get("completed_evaluation", []),
        "accounting": accounting,
        "platform": platform,
    }
    preflight.calibration_evidence(result, required=True)
    return result


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/data"))
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forecast-rate", type=float, default=os.environ.get("SCALAR_PI_FORECAST_RATE_USD"))
    args = parser.parse_args(argv)
    deadline = number(float(os.environ["SCALAR_PI_DEADLINE_UNIX"]), "session deadline")
    if deadline - time.time() <= 600:
        raise TimeoutError("No budget-assembly time remains before the shutdown reserve")
    profiles = verified_profiles(args.data)
    result = assemble(
        profiles, read_json(args.inputs), reports=args.data / "experiments/reports", forecast_rate=args.forecast_rate
    )
    result["operator_inputs"] = {"path": str(args.inputs), "sha256": digest(args.inputs)}
    publish(args.output, result)
    print(f"Assembled measured budget: {args.output}")


if __name__ == "__main__":
    main()
