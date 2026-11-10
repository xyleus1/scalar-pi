"""Image cache verification requires no Docker daemon or image downloads."""

import json
from pathlib import Path
import runpy

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/lambda/images.py"))


def test_image_cache_rejects_changed_archive_or_image(tmp_path):
    archive = tmp_path / "images.tar.gz"
    archive.write_bytes(b"fixture image bytes")
    expected = MODULE["identity"]("a" * 40, "sha256:" + "b" * 64, "sha256:" + "c" * 64)
    receipt = tmp_path / "images.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                **expected,
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": MODULE["digest"](archive),
            }
        )
    )
    MODULE["verify"](receipt, archive, expected, hash_archive=True)
    with pytest.raises(ValueError, match="source/image"):
        MODULE["verify"](receipt, archive, {**expected, "model_id": "sha256:" + "d" * 64})
    archive.write_bytes(b"changed image bytes")
    with pytest.raises(ValueError, match="corrupt"):
        MODULE["verify"](receipt, archive, expected, hash_archive=True)


def test_image_identity_requires_immutable_references():
    with pytest.raises(ValueError, match="commit"):
        MODULE["identity"]("main", "model:latest", "sim:latest")
    with pytest.raises(ValueError, match="image IDs"):
        MODULE["identity"]("a" * 40, "model:latest", "sim:latest")
