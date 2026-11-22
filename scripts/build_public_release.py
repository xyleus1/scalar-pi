"""Build a new curated release from an immutable, locally retained study export.

No network or provider access. The private export is read-only. Dataset contents
are explicitly projected rather than copying opaque control/lifecycle receipts.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

REPO = Path(__file__).resolve().parents[1]
INDEX_SHA = "486cceba32fb9a6ee756e330a4a5139c8ad65530e551f924dadde76025c83f1c"
SCALAR_FIELDS = (
    "action_horizon",
    "attempt_id",
    "checkpoint_id",
    "config",
    "episode_id",
    "episode_seconds",
    "failure_kind",
    "group",
    "initial_state_id",
    "max_steps",
    "num_steps",
    "num_steps_wait",
    "paired_noise_seed",
    "policy_queries",
    "protocol_id",
    "reference",
    "replan_steps",
    "rollout_seed",
    "schema_version",
    "steps",
    "success",
    "suite",
    "task_attempt_id",
    "task_description",
    "task_id",
    "training_seed",
)
METADATA = (
    "checkpoint_dtype_policy",
    "checkpoint_id",
    "checkpoint_sha256",
    "config_name",
    "expert_precision",
    "matmul_precision",
    "normalization_sha256",
    "num_steps",
    "precision_protocol",
    "velocity_mode",
    "vlm_precision",
)
RUNTIME = ("backend", "container_image", "cuda_runtime", "cudnn", "device", "gpu", "torch", "jax", "jaxlib")
RESEARCH = (
    "action_dim",
    "action_horizon",
    "base_checkpoint_hash",
    "base_conversion_receipt_sha256",
    "checkpoint_dtype_policy",
    "code_commit",
    "container_image_id",
    "data_manifest_sha256",
    "expert_precision",
    "freeze_vlm",
    "inference_weights",
    "initialization",
    "matmul_precision",
    "model_sha256",
    "normalization_hash",
    "optimizer_updates",
    "precision_protocol",
    "raw_model_sha256",
    "run_mode",
    "schema_version",
    "source_snapshot_sha256",
    "training_seed",
    "velocity_mode",
    "vlm_precision",
)
PRIVATE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|/(?:lambda|home|var/lib|Users)/|arn:aws:|"
    r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])|"
    r"-----BEGIN (?:OPENSSH|RSA|EC|PRIVATE)|gh[pousr]_[A-Za-z0-9]{20,}"
)
BLOCKED_KEYS = {
    "slot_id",
    "instance_id",
    "machine_id",
    "boot_id",
    "gpu_uuid",
    "ssh_host",
    "ssh_key",
    "known_hosts",
    "access_token",
    "api_key",
    "secret",
    "credentials",
    "workspace_id",
    "account_id",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")


def pick(value, keys):
    return {k: value[k] for k in keys if k in value}


def metadata(value):
    result = pick(value, METADATA)
    if "runtime" in value:
        result["runtime"] = pick(value["runtime"], RUNTIME)
        if "devices" in value["runtime"]:
            result["runtime"]["devices"] = [
                pick(device, ("device", "kind", "platform")) for device in value["runtime"]["devices"]
            ]
    if isinstance(value.get("research"), dict):
        result["research"] = pick(value["research"], RESEARCH)
    elif "research" in value:
        result["research"] = value["research"]
    return result


def public_row(value, original_line):
    row = pick(value, SCALAR_FIELDS)
    for k in ("failure_tags", "query_noise_seeds", "latencies"):
        if k in value:
            row[k] = value[k]
    # Error kinds remain part of the original outcome summary; arbitrary error
    # strings may contain paths, addresses or command lines and are digest-only.
    if "artifact_errors" in value:
        row["artifact_errors"] = [{"kind": e["kind"]} for e in value["artifact_errors"]]
    if value.get("error_message"):
        row["error_detail_sha256"] = hashlib.sha256(value["error_message"].encode()).hexdigest()
    row["server_metadata"] = metadata(value.get("server_metadata", {}))
    row["original_record_sha256"] = hashlib.sha256(original_line).hexdigest()
    return row


def public_provenance(value):
    return {
        "kind": "public_scientific_provenance_projection_v1",
        **pick(value, ("container_image", "protocol_id", "schema_version", "simulator_commit")),
        "source": pick(value.get("source", {}), ("commit", "source_sha256")),
        "server_metadata": metadata(value.get("server_metadata", {})),
        "settings": pick(
            value.get("settings", {}),
            (
                "attempt_id",
                "max_steps",
                "num_steps",
                "num_steps_wait",
                "paired_noise_seed",
                "render_size",
                "replan_steps",
                "resize_size",
                "rollout_seed",
                "suite",
            ),
        ),
    }


def evidence_digests(value):
    """Preserve digest linkage, never machine paths/identities or opaque receipts."""
    if isinstance(value, dict):
        if "sha256" in value:
            return {"sha256": value["sha256"]}
        return {
            k: evidence_digests(v)
            for k, v in value.items()
            if k not in BLOCKED_KEYS and k not in ("path", "local", "retained", "worker", "origin")
        }
    if isinstance(value, list):
        return [evidence_digests(v) for v in value]
    if isinstance(value, str) and PRIVATE.search(value):
        return {"omitted_value_sha256": hashlib.sha256(value.encode()).hexdigest()}
    return value


def clean(value):
    if isinstance(value, dict):
        require(not BLOCKED_KEYS.intersection(value), "Private metadata key in public output")
        require(not any(PRIVATE.search(k) for k in value), "Private path in public metadata key")
        for v in value.values():
            clean(v)
    elif isinstance(value, list):
        for v in value:
            clean(v)
    elif isinstance(value, str):
        require(not PRIVATE.search(value), "Private path, address, or credential pattern in public output")


def checked_source(root, reference):
    path = (root / reference["path"]).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), "Input escapes the retained export")
    require(digest(path) == reference["sha256"], "Retained source digest mismatch")
    return path


def file_ref(path, root):
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size}


def checksums(root):
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "checksums.sha256")
    (root / "checksums.sha256").write_text(
        "".join(f"{digest(p)}  {p.relative_to(root).as_posix()}\n" for p in files), encoding="utf8"
    )


def zipped(source, output, prefix):
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in sorted(source.rglob("*")):
            if p.is_file():
                archive.write(
                    p, f"{prefix}/{p.relative_to(source).as_posix()}" if prefix else p.relative_to(source).as_posix()
                )


def build(source, output, *, resume=False):
    require(
        not output.is_relative_to(source) and not source.is_relative_to(output),
        "Private input and public output directories must be disjoint",
    )
    index_path = source / "dataset-index.json"
    require(digest(index_path) == INDEX_SHA, "Only the exact delivered 40k export is supported")
    original = json.loads(index_path.read_bytes())
    require((original["episode_count"], original["task_count"]) == (40000, 800), "Wrong source scope")
    if resume:
        require(
            output.is_dir()
            and not (output / "curation.json").exists()
            and not (output / "data/dataset-index.json").exists(),
            "Only an unfinished local curation can resume",
        )
    output.mkdir(parents=True, exist_ok=resume)
    data = output / "data"
    (data / "task-logs").mkdir(parents=True, exist_ok=resume)
    (data / "provenance").mkdir(exist_ok=resume)
    (data / "excluded-infrastructure").mkdir(exist_ok=resume)
    tasks, exclusions, projections, identities = [], [], [], set()
    aggregate = data / "episodes.jsonl"
    csv_path = data / "episodes.csv"
    with (
        aggregate.open("w", encoding="utf8", newline="\n") as all_rows,
        csv_path.open("w", encoding="utf8", newline="") as csv_stream,
    ):
        writer = csv.DictWriter(
            csv_stream, fieldnames=(*SCALAR_FIELDS, "original_record_sha256"), extrasaction="ignore"
        )
        writer.writeheader()
        for i, task in enumerate(original["tasks"]):
            input_log = checked_source(source, task["task_log"])
            input_prov = checked_source(source, task["provenance"])
            log = data / f"task-logs/{i:03d}.jsonl"
            prov = data / f"provenance/{i:03d}.json"
            public_prov = public_provenance(json.loads(input_prov.read_bytes()))
            clean(public_prov)
            dump(prov, public_prov)
            n = 0
            with log.open("w", encoding="utf8", newline="\n") as local:
                for line in input_log.read_bytes().splitlines(keepends=True):
                    raw = json.loads(line)
                    require(raw.get("failure_kind") != "infrastructure", "Selected infrastructure outcome")
                    key = tuple(raw[k] for k in ("checkpoint_id", "num_steps", "suite", "task_id", "initial_state_id"))
                    require(key not in identities, "Duplicate episode key")
                    require(
                        key[:4] == (task["checkpoint_id"], task["num_steps"], task["suite"], task["task_id"]),
                        "Wrong task row",
                    )
                    identities.add(key)
                    row = public_row(raw, line)
                    clean(row)
                    text = json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
                    local.write(text)
                    all_rows.write(text)
                    writer.writerow(row)
                    n += 1
            require(n == 50, "Selected task is not 50 episodes")
            item = pick(task, ("task_key", "checkpoint_id", "num_steps", "suite", "task_id", "selected_attempt"))
            item.update(
                episodes=n,
                task_log=file_ref(log, data),
                provenance=file_ref(prov, data),
                original_task_log_sha256=task["task_log"]["sha256"],
                original_provenance_sha256=task["provenance"]["sha256"],
                effective_selection=evidence_digests(task["effective_selection"]),
                existing_original_row_validation=True,
                existing_closed_lifecycle=True,
                hardware_acceptance_claimed=False,
                validation_evidence=evidence_digests(task["validation_evidence"]),
                closure_evidence=evidence_digests(task["closure_evidence"]),
            )
            clean(item)
            tasks.append(item)
            projections.append(
                {
                    "task_key": task["task_key"],
                    "source_log_sha256": task["task_log"]["sha256"],
                    "public_log": item["task_log"],
                    "source_provenance_sha256": task["provenance"]["sha256"],
                    "public_provenance": item["provenance"],
                }
            )
    require(len(identities) == 40000 and len(tasks) == 800, "Incomplete public dataset")
    for i, old in enumerate(original["excluded_infrastructure_attempts"]):
        src = checked_source(source, old["task_log"])
        lines = src.read_bytes().splitlines(keepends=True)
        require(len(lines) == 1, "Expected one original infrastructure row")
        value = json.loads(lines[0])
        require(value["failure_kind"] == "infrastructure", "Invalid excluded row")
        row = public_row(value, lines[0])
        clean(row)
        path = data / f"excluded-infrastructure/{i}.jsonl"
        path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf8")
        exclusions.append(
            {
                "task_key": old["task_key"],
                "disposition": old["disposition"],
                "source_log_sha256": old["task_log"]["sha256"],
                "public_log": file_ref(path, data),
            }
        )
    result = {
        "kind": "public_curated_study_dataset_v1",
        "author": "Nima Kamali",
        "license": "CC-BY-4.0",
        "episode_count": 40000,
        "task_count": 800,
        "suite_cells": 80,
        "checkpoint_solver_cells": 20,
        "source_index_sha256": INDEX_SHA,
        "hardware_acceptance_claimed": False,
        "canonical_accepted_full_marker_created": False,
        "original_completed_phase_episodes": 16000,
        "scientific_validator_reruns": 0,
        "tasks": tasks,
        "excluded_infrastructure_attempts": exclusions,
    }
    dump(data / "dataset-index.json", result)
    dump(
        data / "projection-map.json",
        {
            "kind": "public_projection_map_v1",
            "source_index_sha256": INDEX_SHA,
            "policy": "Explicit scientific field allowlists; private paths, machine IDs, commands and opaque receipts excluded. Source digests refer to unchanged private originals; public files have new digests.",
            "record_hash_definition": "SHA256 of each original task-log line including its retained newline, before projection.",
            "omitted_fields": [
                "checkpoint_dir",
                "error_message",
                "private research resume_signature",
                "provider identity",
                "filesystem paths",
                "control receipts",
            ],
            "error_handling": "Outcome, failure_kind, failure_tags and artifact error kinds remain; free-form error detail is digest-only.",
            "tasks": projections,
        },
    )
    dump(
        data / "effective-selections.json",
        {
            "source_manifest_sha256": "9e2036f5f8a78258288f2bd3b6782fc5ea37d6e9fa16411f2151882a74071686",
            "selections": [
                {
                    "task_key": t["task_key"],
                    "selected_attempt": t["selected_attempt"],
                    "selection": t["effective_selection"],
                }
                for t in tasks
                if t["selected_attempt"].startswith("tokenizer-retry-")
            ],
            "canonical_manifest_changed": False,
        },
    )
    shutil.copyfile(REPO / "LICENSE-RESEARCH", data / "LICENSE-RESEARCH")
    shutil.copyfile(REPO / "CITATION.cff", data / "CITATION.cff")
    (data / "README.md").write_text(
        "# Scalar-Pi curated results, v0.1.0\n\nNima Kamali; original results CC BY 4.0.\n\n40,000 selected episodes / 800 tasks / 80 suite cells. Original row validation and closed-lifecycle evidence are digest-linked; this derivative does not claim hardware acceptance. A100 PCIe failed the cross-H100 gate; A100 SXM remains unqualified. All mixed-hardware estimates are descriptive. Four initial infrastructure failures are retained separately; their four successful retries are explicitly selected. No outcomes or statistical settings are changed.\n\nPrivate filesystem paths, cloud identities and control receipts are omitted by an explicit field projection. Public task logs/provenance are derived files with new hashes, not byte-identical originals. See projection-map.json and effective-selections.json. Original per-query latencies and noise seeds are retained; no per-step action/state trajectories or physical jerk metrics were recorded. Timing must remain stratified by actual hardware/runtime/concurrency.\n\nThe source package contains paper/reproduce_descriptive.py. Run it with --dataset pointing here, first --check-only, then --output reproduced.json. It checks all dataset checksums and reproduces all 80 suite cells and eight original contrasts. The unmodified private evidence remains retained outside this public package.\n",
        encoding="utf8",
    )
    checksums(data)
    source_package = output / "source"
    source_package.mkdir()
    shutil.copytree(
        REPO / "paper", source_package / "paper", ignore=shutil.ignore_patterns("build", "__pycache__", ".gitignore")
    )
    shutil.copytree(REPO / "results", source_package / "results")
    for name in ("LICENSE", "LICENSE_GEMMA.txt", "LICENSE-RESEARCH", "CITATION.cff"):
        shutil.copyfile(REPO / name, source_package / name)
    checksums(source_package)
    for tree in (data, source_package):
        for path in tree.rglob("*.json"):
            clean(json.loads(path.read_bytes()))
    zipped(data, output / "scalar-pi-data.zip", "data")
    zipped(source_package, output / "scalar-pi-source.zip", "")
    dump(
        output / "curation.json",
        {
            "kind": "public_study_curation_v1",
            "source_index_sha256": INDEX_SHA,
            "episodes": 40000,
            "tasks": 800,
            "excluded_infrastructure_rows": len(exclusions),
            "source_inputs_modified": False,
            "hardware_acceptance_claimed": False,
            "data": file_ref(output / "scalar-pi-data.zip", output),
            "source": file_ref(output / "scalar-pi-source.zip", output),
        },
    )
    print(json.dumps({"status": "curated", "episodes": 40000, "tasks": 800, "output": str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New directory outside the Git checkout")
    parser.add_argument("--resume", action="store_true", help="Rebuild only an unfinished, unpublished local curation")
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to(REPO), "Keep large release archives outside Git")
    build(args.private_dataset.resolve(), args.output.resolve(), resume=args.resume)


if __name__ == "__main__":
    main()
