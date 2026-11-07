import hashlib
import json
from types import SimpleNamespace

import pytest

from openpi.research.data import build_data_manifest


@pytest.fixture
def snapshot(tmp_path):
    values = {"meta/info.json": json.dumps({"total_frames": 10}).encode(), "data/chunk/file.parquet": b"data-fixture"}
    remote = []
    for name, value in values.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
        lfs = SimpleNamespace(sha256=hashlib.sha256(value).hexdigest()) if name.startswith("data/") else None
        remote.append(
            SimpleNamespace(
                rfilename=name,
                size=len(value),
                lfs=lfs,
                blob_id=hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest(),
            )
        )
    return tmp_path, remote


def test_manifest_verifies_remote_lfs_and_git_blobs(snapshot):
    root, remote = snapshot
    result = build_data_manifest(root, repo_id="test/repo", revision="a" * 40, remote_files=remote)
    assert result["counts"]["total_frames"] == 10
    assert len(result["files"]) == 2


@pytest.mark.parametrize("name", ["meta/info.json", "data/chunk/file.parquet"])
def test_manifest_rejects_changed_bytes_even_at_same_size(snapshot, name):
    root, remote = snapshot
    path = root / name
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ValueError, match="Local bytes differ"):
        build_data_manifest(root, repo_id="test/repo", revision="a" * 40, remote_files=remote)


def test_manifest_rejects_stale_extra_dataset_file(snapshot):
    root, remote = snapshot
    (root / "data/stale.parquet").write_bytes(b"old")
    with pytest.raises(ValueError, match="extra"):
        build_data_manifest(root, repo_id="test/repo", revision="a" * 40, remote_files=remote)
