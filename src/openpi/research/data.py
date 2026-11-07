"""Verify local LeRobot bytes against an immutable Hugging Face snapshot."""

import hashlib
import json
from pathlib import Path
import re
from typing import Any


def build_data_manifest(root: Path, *, repo_id: str, revision: str, remote_files: list[Any]) -> dict:
    """Check remote LFS SHA256 / Git blob SHA1, then hash every loaded file.

    A revision label alone is not provenance: these checks bind the local bytes
    to the remote commit, including small metadata files not stored in LFS.
    """
    if re.fullmatch(r"[0-9a-f]{40,64}", revision) is None:
        raise ValueError("An immutable hexadecimal dataset revision is required")
    root = root.resolve()
    remote = {
        item.rfilename: item for item in remote_files if item.rfilename.split("/")[0] in {"meta", "data", "videos"}
    }
    local = {
        path.relative_to(root).as_posix(): path
        for directory in ("meta", "data", "videos")
        for path in (root / directory).rglob("*")
        if path.is_file()
    }
    if "meta/info.json" not in remote or not any(name.startswith("data/") for name in remote):
        raise ValueError("Remote snapshot is not a complete LeRobot dataset")
    if set(local) != set(remote):
        raise ValueError(
            f"Snapshot file mismatch: missing={sorted(remote.keys() - local.keys())}, "
            f"extra={sorted(local.keys() - remote.keys())}"
        )
    hashes = {}
    for relative, path in sorted(local.items()):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Dataset symlink escapes root: {relative}")
        item = remote[relative]
        size = path.stat().st_size
        if item.size is not None and item.size != size:
            raise ValueError(f"Remote size mismatch: {relative}")
        sha256 = hashlib.sha256()
        git_blob = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                sha256.update(block)
                git_blob.update(block)
        lfs = item.lfs
        expected = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        if expected is not None:
            matches = sha256.hexdigest() == expected
        elif item.blob_id:
            matches = git_blob.hexdigest() == item.blob_id
        else:
            raise ValueError(f"Remote API supplied no content digest: {relative}")
        if not matches:
            raise ValueError(f"Local bytes differ from requested revision: {relative}")
        hashes[relative] = sha256.hexdigest()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "repo_id": repo_id,
        "revision": revision,
        "dataset_root": str(root),
        "files": hashes,
        "counts": {key: info.get(key) for key in ("total_episodes", "total_frames", "total_tasks")},
        "verified_against": "huggingface_git_blob_or_lfs_digest",
    }
