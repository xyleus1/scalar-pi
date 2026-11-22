"""Episode identifiers, immutable provenance and JSONL I/O. No model dependencies."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

PROTOCOL_ID = "openpi-libero-paired-v2"


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
