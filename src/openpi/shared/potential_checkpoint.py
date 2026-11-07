"""Fail-closed architecture metadata for otherwise shape-compatible checkpoints.

This module deliberately has no JAX, Transformers, or Torch dependency so policy
loaders and CPU-only tools can inspect a checkpoint before allocating a model.
"""

import json
from pathlib import Path
from typing import Any

METADATA_FILENAME = "potential_metadata.json"
SCHEMA_VERSION = 1
VELOCITY_MODES = frozenset({"direct", "potential_dot"})


def _weight_metadata(directory: Path) -> dict[str, str]:
    """Read only the safetensors header, without allocating weights or importing Torch."""
    weights = directory / "model.safetensors"
    if not weights.is_file():
        return {}
    with weights.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Truncated safetensors header: {weights}")
        length = int.from_bytes(length_bytes, "little")
        if not 2 <= length <= 64 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header length: {weights}")
        try:
            header = json.loads(stream.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid safetensors header: {weights}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header object: {weights}")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid safetensors metadata: {weights}")
    return metadata


def validate_checkpoint_metadata(
    path: str | Path, *, velocity_mode: str, action_horizon: int, action_dim: int
) -> dict[str, Any] | None:
    """Validate a local checkpoint; unmarked legacy direct checkpoints are allowed.

    New potential checkpoints must always carry metadata, since their weight
    shapes are identical to direct-field models. Never infer mode from shapes.
    """
    if velocity_mode not in VELOCITY_MODES:
        raise ValueError(f"Unsupported velocity mode: {velocity_mode!r}")
    directory = Path(path)
    if directory.is_file():
        directory = directory.parent
    embedded = _weight_metadata(directory)
    metadata_path = directory / METADATA_FILENAME
    if not metadata_path.exists():
        if velocity_mode == "potential_dot" or embedded.get("trainer") == "potential_pi":
            raise ValueError(f"Potential checkpoint requires {metadata_path}")
        if embedded.get("velocity_mode") == "potential_dot":
            raise ValueError("Potential safetensors weights cannot be loaded as a direct field")
        # A missing sidecar must not silently downgrade another marked format.
        for filename in ("config.json", "metadata.json", "training_config.json", "run_manifest.json"):
            candidate = directory / filename
            if candidate.exists() and "potential_dot" in candidate.read_text(encoding="utf-8"):
                raise ValueError(f"Potential evidence in {candidate}, but architecture sidecar is missing")
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid checkpoint metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint metadata schema in {metadata_path}")
    if embedded.get("trainer") == "potential_pi" and metadata.get("trainer") != "potential_pi":
        raise ValueError("Research safetensors trainer marker disagrees with checkpoint sidecar")
    expected = {"velocity_mode": velocity_mode, "action_horizon": action_horizon, "action_dim": action_dim}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Checkpoint {key}={metadata.get(key)!r}, expected {value!r}")
    embedded_expected = {**expected, "schema_version": SCHEMA_VERSION}
    for key, value in embedded_expected.items():
        if key in embedded and embedded[key] != str(value):
            raise ValueError(f"Safetensors {key} disagrees with checkpoint sidecar")
        if metadata.get("trainer") == "potential_pi" and key not in embedded:
            raise ValueError(f"Research weights are missing embedded {key} metadata")
    config_path = directory / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if "velocity_mode" in config and config["velocity_mode"] != velocity_mode:
            raise ValueError("config.json velocity_mode disagrees with checkpoint sidecar")
    return metadata
