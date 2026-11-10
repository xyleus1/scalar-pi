"""Paired LIBERO evaluation bookkeeping, with no simulator or model dependencies.

The official runner supplies observations and success labels. These helpers only
define identifiers, audit a declared experiment, and summarize its results.
"""

# ruff: noqa: UP038 -- Imported by the Python 3.8 simulator; runtime isinstance unions need Python 3.10.
from __future__ import annotations

from collections import Counter
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
from typing import Any

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
PROTOCOL_ID = "openpi-libero-paired-v2"
POLICY_FAILURE_TAGS = {"timeout", "invalid_actions"}


def content_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    files = {str(path.relative_to(root)): file_digest(path) for path in sorted(root.rglob("*")) if path.is_file()}
    if not files:
        raise ValueError(f"No simulator assets found in {root}")
    return {"files": len(files), "sha256": content_digest(files)}


def source_identity(root: str | Path) -> dict[str, Any]:
    """Record actual tracked source bytes, including a mounted checkout's local changes."""
    root = Path(root)

    def git(*args):
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()

    files = {
        name: file_digest(root / name)
        for name in git("ls-files").splitlines()
        if name.endswith((".py", ".json", ".toml", ".lock")) and (root / name).is_file()
    }
    return {
        "commit": git("rev-parse", "HEAD"),
        "tracked_changes": git("status", "--porcelain", "--untracked-files=no"),
        "source_sha256": content_digest(files),
    }


def write_provenance(log_path: str | Path, provenance: dict[str, Any]) -> str:
    """Write one immutable sidecar per suite attempt, not one large payload per episode."""
    path = Path(log_path).with_suffix(".provenance.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return content_digest(provenance)


def suite_provenance(*, server_metadata: dict, settings: dict, tasks: list[dict], simulator_paths: dict) -> dict:
    """Bind an evaluation attempt to model bytes, simulator assets, and numerical settings."""
    root = Path(__file__).resolve().parents[3]
    simulator_root = root / "third_party/libero"
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "server_metadata": server_metadata,
        "source": source_identity(root),
        "simulator_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=simulator_root, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "simulator_assets": {name: tree_digest(path) for name, path in sorted(simulator_paths.items())},
        "container_image": os.environ.get("SCALAR_PI_SIMULATOR_IMAGE", ""),
        "settings": settings,
        "tasks": tasks,
    }


def paired_noise_seed(seed: int, task: str, initial_state: int, query_index: int) -> int:
    """Stable uint32; intentionally independent of model, K, and training seed."""
    if initial_state < 0 or query_index < 0:
        raise ValueError("Initial state and query index must be nonnegative")
    payload = json.dumps([int(seed), str(task), int(initial_state), int(query_index)], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def episode_id(suite: str, task_id: int, initial_state_id: int, rollout_seed: int) -> str:
    return f"{suite}/task-{task_id:02d}/init-{initial_state_id:03d}/env-{rollout_seed}"


def append_episode(path: str | Path, record: dict[str, Any]) -> None:
    """Append one completed episode; reject non-JSON values and NaN before writing."""
    line = json.dumps(record, sort_keys=True, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()


def read_episodes(paths: list[str | Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        sidecar = Path(path).with_suffix(".provenance.json")
        provenance = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else None
        with Path(path).open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("Episode must be an object")
                    if record.get("provenance_sha256"):
                        if provenance is None or content_digest(provenance) != record["provenance_sha256"]:
                            raise ValueError("Missing or changed provenance sidecar")
                        record["_provenance"] = provenance
                    records.append(record)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid episode at {path}:{line_no}: {exc}") from exc
    return records


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Check the declared study; a one-seed pilot retains the complete benchmark."""
    if manifest.get("schema_version") != 1:
        raise ValueError("Manifest schema_version must be 1")
    for field in ("rollout_seed", "paired_noise_seed"):
        if type(manifest.get(field, 7)) is not int:
            raise ValueError(f"{field} must be an integer")
    diagnostic = manifest.get("diagnostic", False)
    if type(diagnostic) is not bool:
        raise ValueError("diagnostic must be a JSON boolean")
    if type(manifest.get("task_shards", False)) is not bool:
        raise ValueError("task_shards must be a JSON boolean")
    if any(not isinstance(manifest.get(field, {}), dict) for field in ("suite_attempts", "task_attempts")):
        raise ValueError("Attempt selections must be JSON objects")
    require_provenance = manifest.get("require_provenance", not diagnostic)
    if type(require_provenance) is not bool or (not diagnostic and not require_provenance):
        raise ValueError("Benchmark studies require provenance; only diagnostic runs may disable it")
    suites = manifest.get("suites", list(SUITES))
    steps = manifest.get("num_steps", [1, 10])
    trials = manifest.get("num_trials_per_task", 50)
    tasks = manifest.get("num_tasks_per_suite", 10)
    if not suites or len(set(suites)) != len(suites) or not set(suites).issubset(SUITES):
        raise ValueError("Declare unique supported LIBERO suites")
    if not steps or len(set(steps)) != len(steps) or any(type(k) is not int or k < 1 for k in steps):
        raise ValueError("num_steps must contain unique positive integers")
    if type(trials) is not int or not 1 <= trials <= 50 or type(tasks) is not int or not 1 <= tasks <= 10:
        raise ValueError("Require 1..50 initial states and 1..10 tasks per suite")
    if not diagnostic and (set(suites) != set(SUITES) or 10 not in steps or trials != 50 or tasks != 10):
        raise ValueError("Primary matrix requires all four suites, 10 tasks, 50 trials, and primary K=10")
    variants = manifest.get("variants", [])
    if not variants:
        raise ValueError("Manifest requires variants")
    ids = set()
    groups = defaultdict(list)
    references = []
    for variant in variants:
        if type(variant.get("reference", False)) is not bool:
            raise ValueError("reference must be a JSON boolean")
        for field in ("group", "checkpoint_id", "config", "checkpoint_dir"):
            if not isinstance(variant.get(field), str) or not variant[field]:
                raise ValueError(f"Variant requires a nonempty {field}")
        checkpoint_id = variant["checkpoint_id"]
        if checkpoint_id in {".", ".."} or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in checkpoint_id
        ):
            raise ValueError("checkpoint_id must be a filesystem-safe identifier")
        if checkpoint_id in ids:
            raise ValueError(f"Duplicate checkpoint_id: {checkpoint_id}")
        ids.add(checkpoint_id)
        if variant.get("reference", False):
            if variant.get("training_seed") is not None:
                raise ValueError("Released references must have training_seed=null")
            references.append(variant["group"])
        else:
            if type(variant.get("training_seed")) is not int:
                raise ValueError("Trained variants require an integer training_seed")
            groups[variant["group"]].append(variant["training_seed"])
    if set(references) & set(groups):
        raise ValueError("Reference and independently trained groups must have different names")
    if len(set(references)) != len(references):
        raise ValueError("Each released reference needs a separate group name")
    for group, seeds in groups.items():
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate training seed in {group}")
    if not diagnostic:
        if manifest.get("study_kind", "replicated") not in {"replicated", "pilot"}:
            raise ValueError("study_kind must be replicated or pilot")
        if set(groups) != {"B0", "B1", "P"} or references != ["reference"]:
            raise ValueError("Primary matrix requires B0, B1, P, and one released reference group")
        expected_seeds = manifest.get("training_seeds", [17, 42, 93])
        if (
            not expected_seeds
            or len(set(expected_seeds)) != len(expected_seeds)
            or any(type(seed) is not int for seed in expected_seeds)
        ):
            raise ValueError("Declare unique integer training seeds")
        if len(expected_seeds) == 1 and manifest.get("study_kind") != "pilot":
            raise ValueError("A single training seed must be explicitly labeled study_kind=pilot")
        if manifest.get("study_kind") == "pilot" and len(expected_seeds) != 1:
            raise ValueError("The declared pilot uses one training seed")
        if any(set(seeds) != set(expected_seeds) for seeds in groups.values()):
            raise ValueError("Each trained group must contain exactly the declared training seeds")
    for attempt in (*manifest.get("suite_attempts", {}).values(), *manifest.get("task_attempts", {}).values()):
        if (
            not isinstance(attempt, str)
            or not attempt
            or attempt in {".", ".."}
            or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in attempt)
        ):
            raise ValueError("Suite attempt IDs must be filesystem-safe identifiers")


def matrix_cells(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    validate_manifest(manifest)
    output_dir = Path(manifest.get("output_dir", "results/potential"))
    cells = []
    for variant in manifest["variants"]:
        for steps in manifest.get("num_steps", [1, 10]):
            for suite in manifest.get("suites", list(SUITES)):
                key = f"{variant['checkpoint_id']}/k{steps}/{suite}"
                attempt = manifest.get("suite_attempts", {}).get(key, "initial")
                directory = output_dir / variant["checkpoint_id"] / f"k{steps}"
                if attempt != "initial":
                    directory /= attempt
                cells.append(
                    {
                        **variant,
                        "num_steps": steps,
                        "suite": suite,
                        "attempt_id": attempt,
                        "episode_log_path": str(directory / f"{suite}.jsonl"),
                    }
                )
    declared = set(manifest.get("suite_attempts", {}))
    if declared - {f"{c['checkpoint_id']}/k{c['num_steps']}/{c['suite']}" for c in cells}:
        raise ValueError("suite_attempts contains a cell outside the declared matrix")
    task_keys = {
        f"{c['checkpoint_id']}/k{c['num_steps']}/{c['suite']}/task-{task:02d}"
        for c in cells
        for task in range(manifest.get("num_tasks_per_suite", 10))
    }
    if set(manifest.get("task_attempts", {})) - task_keys:
        raise ValueError("task_attempts contains a task outside the declared matrix")
    return cells


def task_log_path(cell: dict, task_id: int, attempt: str = "initial") -> str:
    path = Path(cell["episode_log_path"])
    return str(path.with_suffix(".tasks") / f"task-{task_id:02d}" / f"{attempt}.jsonl")


def task_attempt(manifest: dict, cell: dict, task_id: int) -> str:
    key = f"{cell['checkpoint_id']}/k{cell['num_steps']}/{cell['suite']}/task-{task_id:02d}"
    return manifest.get("task_attempts", {}).get(key, "initial")


def suite_manifest(manifest: dict, cell: dict) -> dict:
    """Validation view retaining the parent study's provenance and attempt selection."""
    key = f"{cell['checkpoint_id']}/k{cell['num_steps']}/{cell['suite']}"
    return {
        **manifest,
        "diagnostic": True,
        "variants": [v for v in manifest["variants"] if v["checkpoint_id"] == cell["checkpoint_id"]],
        "num_steps": [cell["num_steps"]],
        "suites": [cell["suite"]],
        "require_provenance": manifest.get("require_provenance", not manifest.get("diagnostic", False)),
        "suite_attempts": {k: v for k, v in manifest.get("suite_attempts", {}).items() if k == key},
        "task_attempts": {k: v for k, v in manifest.get("task_attempts", {}).items() if k.startswith(key + "/task-")},
    }


def validate_task_log(manifest: dict, cell: dict, task_id: int, path: str | Path) -> list[dict]:
    records = read_episodes([path])
    validate_results(
        records,
        suite_manifest(manifest, cell),
        task_ids=[task_id],
        require_primary_checkpoint=not manifest.get("diagnostic", False),
    )
    return records


def aggregate_tasks(manifest: dict, cell: dict) -> dict:
    """Publish a canonical suite only after every selected task attempt passes validation."""
    destination = Path(cell["episode_log_path"])
    if destination.exists():
        raise FileExistsError(f"Canonical suite already exists: {destination}")
    records, sources = [], []
    for task in range(manifest.get("num_tasks_per_suite", 10)):
        attempt = task_attempt(manifest, cell, task)
        path = task_log_path(cell, task, attempt)
        records.extend(validate_task_log(manifest, cell, task, path))
        sources.append({"task_id": task, "task_attempt_id": attempt, "path": path, "sha256": file_digest(path)})
    validation = validate_results(
        records, suite_manifest(manifest, cell), require_primary_checkpoint=not manifest.get("diagnostic", False)
    )
    digests = {row.get("provenance_sha256") for row in records}
    if len(digests) != 1:
        raise ValueError("Task shards do not share identical suite provenance")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if records[0].get("_provenance"):
        sidecar = destination.with_suffix(".provenance.json")
        if sidecar.exists():
            if content_digest(json.loads(sidecar.read_text())) != records[0]["provenance_sha256"]:
                raise ValueError("Existing suite sidecar differs from task provenance")
        else:
            write_provenance(destination, records[0]["_provenance"])
    temporary = destination.with_suffix(".assembling")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in sorted(records, key=lambda row: (row["task_id"], row["initial_state_id"])):
            stream.write(
                json.dumps({k: v for k, v in row.items() if k != "_provenance"}, sort_keys=True, allow_nan=False) + "\n"
            )
    temporary.replace(destination)
    receipt = {"schema_version": 1, "episode_log_sha256": file_digest(destination), "tasks": sources}
    destination.with_suffix(".task_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return validation


def task_budget_seconds(manifest: dict, cell: dict) -> float:
    timing = manifest
    if manifest.get("timing_report"):
        timing = json.loads(Path(manifest["timing_report"]).read_text())
    value = (
        timing.get("task_budget_seconds", {}).get(cell["group"], {}).get(str(cell["num_steps"]), {}).get(cell["suite"])
    )
    if value is None and manifest.get("diagnostic", False):
        value = manifest.get("diagnostic_task_budget_seconds", 900)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("A finite measured task budget is required before deadline-bound evaluation")
    return float(value)


def verify_timing_report(path: str | Path) -> dict:
    report = json.loads(Path(path).read_text())
    if report.get("purpose") != "evaluation_cost_calibration" or report.get("passed") is not True:
        raise ValueError("Evaluation requires a completed component calibration")
    if report.get("source") != source_identity(Path(__file__).resolve().parents[3]):
        raise ValueError("Evaluation calibration source differs from current execution")
    if report.get("model_image") != os.environ.get("SCALAR_PI_MODEL_IMAGE") or report.get(
        "simulator_image"
    ) != os.environ.get("SCALAR_PI_SIMULATOR_IMAGE"):
        raise ValueError("Evaluation calibration containers differ from current execution")
    evidence = report.get("evidence")
    if (
        not isinstance(evidence, dict)
        or not evidence
        or any(file_digest(name) != expected for name, expected in evidence.items())
    ):
        raise ValueError("Evaluation calibration evidence is missing or changed")
    return report


def _record_key(record: dict[str, Any]) -> tuple:
    return tuple(record[key] for key in ("checkpoint_id", "num_steps", "suite", "task_id", "initial_state_id"))


def _validate_checkpoint_record(record: dict[str, Any], *, primary: bool) -> None:
    if record.get("reference", False):
        return  # Released reference metadata does not attest an independently trained seed.
    research = record["server_metadata"].get("research")
    if research is None and not primary:
        return
    if not isinstance(research, dict):
        raise ValueError(f"Missing or invalid actual checkpoint research metadata: {record['checkpoint_id']}")
    if primary and research.get("run_mode") == "profile":
        raise ValueError("Profiling checkpoints cannot enter confirmatory evaluation")
    expected = {"training_seed": record["training_seed"]}
    if primary:
        if record["group"] not in {"B0", "B1", "P"}:
            raise ValueError("Unknown primary checkpoint group")
        expected.update(
            {
                "trainer": "potential_pi",
                "optimizer_updates": 30000,
                "inference_weights": "ema",
                "velocity_mode": "potential_dot" if record["group"] == "P" else "direct",
                "initialization": "pretrained_output" if record["group"] == "B0" else "zero_output",
                "freeze_vlm": True,
                "action_horizon": 10,
                "action_dim": 32,
            }
        )
    for field, value in expected.items():
        actual = research.get(field)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"Actual checkpoint {field}={actual!r}, expected {value!r}: {record['checkpoint_id']}")


def validate_provenance(record: dict[str, Any]) -> dict:
    provenance = record.get("_provenance")
    if not isinstance(provenance, dict) or content_digest(provenance) != record.get("provenance_sha256"):
        raise ValueError("Missing or changed evaluation provenance")
    if provenance.get("schema_version") != 1 or provenance.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unknown evaluation provenance schema or protocol")
    metadata = record["server_metadata"]
    if provenance.get("server_metadata") != metadata:
        raise ValueError("Episode metadata differs from its provenance")
    for field in ("checkpoint_sha256", "normalization_sha256"):
        digest = metadata.get(field, "")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Missing actual {field} in server provenance")
    if not record.get("reference", False):
        research = metadata.get("research") or {}
        for actual, receipt in (("checkpoint_sha256", "model_sha256"), ("normalization_sha256", "normalization_hash")):
            if metadata[actual] != research.get(receipt):
                raise ValueError(f"Actual served {actual} differs from the trainer's {receipt} receipt")
    if not record.get("reference", False) and metadata.get("matmul_precision") != "highest":
        raise ValueError("Research serving must use the declared highest matmul precision")
    if not provenance.get("container_image") or not metadata.get("runtime", {}).get("container_image"):
        raise ValueError("Record both model and simulator container identities")
    source = provenance.get("source", {})
    if not source.get("commit") or not source.get("source_sha256") or source.get("tracked_changes") != "":
        raise ValueError("Evaluation requires a clean, identified source checkout")
    if not provenance.get("simulator_commit") or not provenance.get("simulator_assets"):
        raise ValueError("Missing simulator source or asset identity")
    settings = provenance.get("settings", {})
    for field in ("suite", "num_steps", "rollout_seed", "paired_noise_seed", "replan_steps", "attempt_id"):
        if settings.get(field) != record.get(field, "initial" if field == "attempt_id" else None):
            raise ValueError(f"Provenance settings mismatch: {field}")
    for field, value in {
        "render_size": 256,
        "resize_size": 224,
        "num_steps_wait": 10,
        "max_steps": MAX_STEPS[record["suite"]],
    }.items():
        if settings.get(field) != value:
            raise ValueError(f"Provenance native LIBERO setting mismatch: {field}")
    tasks = {task["task_id"]: task for task in provenance.get("tasks", [])}
    task = tasks.get(record["task_id"], {})
    states = task.get("initial_states_sha256", [])
    if not task.get("bddl_sha256") or not 0 <= record["initial_state_id"] < len(states):
        raise ValueError("Missing task definition or official initial-state identity")
    return provenance


def validate_results(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    require_primary_checkpoint: bool = False,
    task_ids: list[int] | None = None,
) -> dict[str, int]:
    """Fail closed for missing/duplicate episodes, stale settings, and execution errors."""
    cells = matrix_cells(manifest)
    variants = {v["checkpoint_id"]: v for v in manifest["variants"]}
    trials = manifest.get("num_trials_per_task", 50)
    tasks = manifest.get("num_tasks_per_suite", 10)
    selected_tasks = list(range(tasks)) if task_ids is None else task_ids
    if (
        not selected_tasks
        or len(set(selected_tasks)) != len(selected_tasks)
        or any(type(task) is not int or not 0 <= task < tasks for task in selected_tasks)
    ):
        raise ValueError("Task validation requires unique original task IDs within the suite")
    expected = {
        (cell["checkpoint_id"], cell["num_steps"], cell["suite"], task, state)
        for cell in cells
        for task in selected_tasks
        for state in range(trials)
    }
    seen = set()
    matched_environment = None
    matched_normalization = None
    matched_training = None
    matched_runtime = None
    matched_tasks = {}
    cell_attempts = {(c["checkpoint_id"], c["num_steps"], c["suite"]): c["attempt_id"] for c in cells}
    for record in records:
        try:
            key = _record_key(record)
        except KeyError as exc:
            raise ValueError(f"Missing record identity field: {exc}") from exc
        if any(type(record[field]) is not int for field in ("task_id", "initial_state_id")):
            raise ValueError("Task and initial-state identities must be integers")
        if key in seen:
            raise ValueError(f"Duplicate episode: {key}")
        if key not in expected:
            raise ValueError(f"Unexpected episode: {key}")
        seen.add(key)
        variant = variants[record["checkpoint_id"]]
        for field in ("group", "training_seed", "config", "checkpoint_dir"):
            if record.get(field) != variant.get(field):
                raise ValueError(f"Mismatched {field} for {key}")
        if record.get("reference", False) != variant.get("reference", False):
            raise ValueError(f"Mismatched reference status for {key}")
        for field, value in {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "rollout_seed": manifest.get("rollout_seed", 7),
            "paired_noise_seed": manifest.get("paired_noise_seed", 7),
            "replan_steps": 5,
            "action_horizon": 10,
            "num_steps_wait": 10,
            "max_steps": MAX_STEPS[record["suite"]],
        }.items():
            if record.get(field) != value:
                raise ValueError(f"Protocol mismatch {field} for {key}: {record.get(field)!r} != {value!r}")
        wanted_id = episode_id(record["suite"], record["task_id"], record["initial_state_id"], record["rollout_seed"])
        if record.get("episode_id") != wanted_id or type(record.get("success")) is not bool:
            raise ValueError(f"Invalid episode ID or success label for {key}")
        errors = set(record.get("failure_tags", [])) - POLICY_FAILURE_TAGS
        if errors:
            raise ValueError(f"Execution errors require investigation before statistics: {key}: {sorted(errors)}")
        if record.get("failure_kind") == "infrastructure":
            raise ValueError(f"Infrastructure attempt cannot enter policy statistics: {key}")
        if record.get("failure_tags") and record["success"]:
            raise ValueError(f"A policy failure cannot be labeled successful: {key}")
        metadata = record.get("server_metadata", {})
        for field, value in {
            "checkpoint_id": record["checkpoint_id"],
            "config_name": record["config"],
            "num_steps": record["num_steps"],
        }.items():
            if metadata.get(field) != value:
                raise ValueError(f"Missing or mismatched server metadata {field}: {key}")
        _validate_checkpoint_record(
            record,
            primary=require_primary_checkpoint or not manifest.get("diagnostic", False),
        )
        if record.get("attempt_id", "initial") != cell_attempts[key[:3]]:
            raise ValueError(f"Attempt differs from the explicitly selected suite attempt: {key}")
        expected_task_attempt = task_attempt(
            manifest, {"checkpoint_id": key[0], "num_steps": key[1], "suite": key[2]}, record["task_id"]
        )
        if record.get("task_attempt_id", "initial") != expected_task_attempt:
            raise ValueError(f"Task attempt differs from the selected attempt: {key}")
        if manifest.get("require_provenance", not manifest.get("diagnostic", False)):
            provenance = validate_provenance(record)
            environment = {
                field: provenance[field]
                for field in ("source", "simulator_commit", "simulator_assets", "container_image")
            }
            if matched_environment is not None and environment != matched_environment:
                raise ValueError("Evaluation source, simulator assets, or container changed within the matrix")
            matched_environment = environment
            suite = record["suite"]
            if suite in matched_tasks and provenance["tasks"] != matched_tasks[suite]:
                raise ValueError("Task definitions or initial-state order changed between checkpoints")
            matched_tasks[suite] = provenance["tasks"]
            if not record.get("reference", False):
                normalization = metadata["normalization_sha256"]
                if matched_normalization is not None and normalization != matched_normalization:
                    raise ValueError("Research variants used different normalization assets")
                matched_normalization = normalization
                research = metadata["research"]
                training = {
                    field: research.get(field)
                    for field in (
                        "base_checkpoint_hash",
                        "data_manifest_sha256",
                        "source_snapshot_sha256",
                        "container_image_id",
                    )
                }
                if any(not value for value in training.values()):
                    raise ValueError("Missing training data/base/source/container identity")
                signature = research.get("resume_signature", {})
                if not signature:
                    raise ValueError("Missing actual training protocol signature")
                training["protocol"] = {
                    name: value for name, value in signature.items() if name not in {"config", "seed", "initialization"}
                }
                if matched_training is not None and training != matched_training:
                    raise ValueError("Research variants used different training data, base, source, or protocol")
                matched_training = training
                if matched_runtime is not None and metadata.get("runtime") != matched_runtime:
                    raise ValueError("Research serving runtime changed within the matrix")
                matched_runtime = metadata.get("runtime")
        queries = record.get("policy_queries")
        if type(queries) is not int or queries < 1:
            raise ValueError(f"Invalid policy query count: {key}")
        expected_noise = [
            paired_noise_seed(
                record["paired_noise_seed"], f"{record['suite']}/{record['task_id']}", record["initial_state_id"], q
            )
            for q in range(queries)
        ]
        if record.get("query_noise_seeds") != expected_noise:
            raise ValueError(f"Logged noise schedule does not match paired protocol: {key}")
        if len(record.get("latencies", [])) != queries:
            raise ValueError(f"Missing policy latency records: {key}")
        for timing in record["latencies"]:
            required_timings = {"client_roundtrip_ms"}
            if "invalid_actions" not in record.get("failure_tags", []):
                required_timings.update(("infer_ms", "total_ms"))
            if not isinstance(timing, dict) or not required_timings.issubset(timing):
                raise ValueError(f"Missing required policy latency fields: {key}")
            if any(
                not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0
                for value in timing.values()
            ):
                raise ValueError(f"Invalid policy latency: {key}")
    missing = expected - seen
    if missing:
        raise ValueError(f"Incomplete matrix: {len(missing)} missing episodes; example {min(missing)}")
    return {"episodes": len(records), "cells": len(cells), "checkpoints": len(variants)}


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-checkpoint/seed/suite summaries; released references stay separate."""
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["checkpoint_id"], record["num_steps"], record["suite"])].append(record)
    result = {"trained": [], "reference": []}
    for (checkpoint_id, steps, suite), rows in sorted(grouped.items()):
        first = rows[0]
        timings = defaultdict(list)
        for row in rows:
            for timing in row.get("latencies", []):
                for name, value in timing.items():
                    if isinstance(value, (float, int)) and math.isfinite(value):
                        timings[name].append(float(value))
        item = {
            "checkpoint_id": checkpoint_id,
            "group": first["group"],
            "training_seed": first.get("training_seed"),
            "num_steps": steps,
            "suite": suite,
            "episodes": len(rows),
            "successes": sum(row["success"] for row in rows),
            "success_rate": statistics.mean(float(row["success"]) for row in rows),
            "failure_tags": dict(Counter(tag for row in rows for tag in row.get("failure_tags", []))),
            "artifact_errors": dict(Counter(error["kind"] for row in rows for error in row.get("artifact_errors", []))),
            "latency_ms": {
                name: {"count": len(values), "mean": statistics.mean(values), "p95": _percentile(values, 0.95)}
                for name, values in sorted(timings.items())
            },
        }
        result["reference" if first.get("reference", False) else "trained"].append(item)
    checkpoint_groups = defaultdict(list)
    for section in ("trained", "reference"):
        for item in result[section]:
            checkpoint_groups[(section, item["checkpoint_id"], item["num_steps"])].append(item)
    result["checkpoint_macro_averages"] = [
        {
            "reference": section == "reference",
            "checkpoint_id": checkpoint_id,
            "num_steps": steps,
            "group": items[0]["group"],
            "training_seed": items[0]["training_seed"],
            "suites": [item["suite"] for item in items],
            "success_rate": statistics.mean(item["success_rate"] for item in items),
        }
        for (section, checkpoint_id, steps), items in sorted(checkpoint_groups.items())
    ]
    task_groups = defaultdict(list)
    for row in records:
        task_groups[(row["checkpoint_id"], row["num_steps"], row["suite"], row["task_id"])].append(row)
    result["tasks"] = [
        {
            "checkpoint_id": checkpoint_id,
            "num_steps": steps,
            "suite": suite,
            "task_id": task,
            "episodes": len(rows),
            "success_rate": statistics.mean(float(row["success"]) for row in rows),
        }
        for (checkpoint_id, steps, suite, task), rows in sorted(task_groups.items())
    ]
    return result


def paired_bootstrap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    resamples: int = 10000,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Paired seed/state bootstrap for equally weighted, fixed benchmark tasks.

    Initial-state draws are shared across sampled training seeds and both methods.
    A one-seed interval measures state variation, not training-seed uncertainty.
    """
    if resamples < 100:
        raise ValueError("Require at least 100 resamples")

    def index(rows):
        indexed = {}
        for row in rows:
            key = (row.get("training_seed"), row["suite"], row["task_id"], row["initial_state_id"])
            if key in indexed:
                raise ValueError(f"Duplicate paired key: {key}; select one group and K per side")
            indexed[key] = row
        return indexed

    li, ri = index(left), index(right)
    if not li or li.keys() != ri.keys():
        raise ValueError("Paired comparison requires identical nonempty seed/task/initial-state keys")
    for key in li:
        for field in (
            "episode_id",
            "rollout_seed",
            "paired_noise_seed",
            "protocol_id",
            "replan_steps",
            "action_horizon",
        ):
            if li[key].get(field) != ri[key].get(field):
                raise ValueError(f"Pairing mismatch in {field}: {key}")
        if bool(li[key].get("reference", False)) != bool(ri[key].get("reference", False)):
            raise ValueError("Do not treat a released reference as an independently trained seed")
    seeds = sorted({key[0] for key in li}, key=str)
    grid = defaultdict(lambda: defaultdict(set))
    for _, suite, task, state in li:
        grid[suite][task].add(state)
    for seed in seeds:
        for suite, tasks in grid.items():
            for task, states in tasks.items():
                if any((seed, suite, task, state) not in li for state in states):
                    raise ValueError("Unbalanced training seed/task/episode grid")
    differences = {key: float(li[key]["success"]) - float(ri[key]["success"]) for key in li}
    rng = random.Random(random_seed)
    draws = []
    suite_names = sorted(grid)
    for _ in range(resamples):
        sampled_seeds = rng.choices(seeds, k=len(seeds))
        suite_means = []
        for suite in suite_names:
            tasks = sorted(grid[suite])
            task_means = []
            for task in tasks:
                states = sorted(grid[suite][task])
                sampled_states = rng.choices(states, k=len(states))
                task_means.append(
                    sum(differences[(seed, suite, task, state)] for seed in sampled_seeds for state in sampled_states)
                    / (len(sampled_seeds) * len(sampled_states))
                )
            suite_means.append(statistics.mean(task_means))
        draws.append(statistics.mean(suite_means))
    ci95 = [_percentile(draws, 0.025), _percentile(draws, 0.975)]
    point = statistics.mean(
        statistics.mean(
            statistics.mean(differences[(seed, suite, task, state)] for seed in seeds for state in states)
            for task, states in grid[suite].items()
        )
        for suite in suite_names
    )
    return {
        "difference_left_minus_right": point,
        "ci95": ci95,
        "resamples": resamples,
        "random_seed": random_seed,
        "training_seeds": seeds,
        "paired_episodes": len(li),
        "method": "paired training-seed/initial-state percentile bootstrap; fixed equally weighted benchmark tasks",
        "training_seed_variability_estimated": len(seeds) > 1 and seeds != [None],
        "scope": (
            "conditional on one checkpoint; training-seed variability is not estimated"
            if len(seeds) == 1
            else "training-seed and initial-state variation on the fixed benchmark tasks; few-seed estimate"
        ),
    }
