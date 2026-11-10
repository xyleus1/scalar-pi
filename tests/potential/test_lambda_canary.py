"""Retained storage verification uses actual fixture files without cloud calls."""

import hashlib
import json
from pathlib import Path
import runpy

import pytest

VERIFY = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/lambda/canary.py"))["verify_retained"]


def test_canary_requires_retained_bytes_on_a_different_boot(tmp_path):
    with pytest.raises(ValueError, match="before staging"):
        VERIFY(tmp_path, "new-boot")
    directory = tmp_path / "data/experiments/reports/canary"
    directory.mkdir(parents=True)
    marker = directory / "test.marker"
    marker.write_bytes(b"retained evidence")
    receipt = {
        "passed": True,
        "marker_path": str(marker),
        "marker_sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
        "boot_id": "old-boot",
    }
    (directory / "test.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="later instance"):
        VERIFY(tmp_path, "old-boot")
    assert VERIFY(tmp_path, "new-boot")["marker_sha256"] == receipt["marker_sha256"]
    marker.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="corrupt"):
        VERIFY(tmp_path, "new-boot")
