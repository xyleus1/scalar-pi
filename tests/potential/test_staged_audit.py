"""Small local byte fixtures verify staging resumes without LeRobot decoding."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.shared import normalize
from scripts import prepare_potential_data as staging


@pytest.fixture
def staged(tmp_path, monkeypatch):
    root = tmp_path / "lerobot" / "physical-intelligence/libero"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    counts = {"total_frames": 10, "total_tasks": 2, "total_episodes": 2}
    (root / "meta/info.json").write_text(json.dumps(counts))
    (root / "data/fixture.parquet").write_bytes(b"Only byte verification; never decoded")
    manifest = {
        "schema_version": 1,
        "repo_id": "physical-intelligence/libero",
        "revision": "a" * 40,
        "dataset_root": str(root),
        "counts": counts,
        "files": {str(path.relative_to(root)): staging.sha256_file(path) for path in root.rglob("*") if path.is_file()},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "lerobot"))
    assets = tmp_path / "assets"
    norms = {"actions": normalize.NormStats(mean=np.zeros(7), std=np.ones(7))}
    arms = ("pi05_libero_b0", "pi05_libero_b1", "pi05_libero_potential")
    for arm in arms:
        destination = assets / arm / manifest["repo_id"]
        destination.mkdir(parents=True)
        normalize.save(destination, norms)
    identity = {
        "data_manifest_sha256": staging.sha256_file(manifest_path),
        "revision": manifest["revision"],
        "normalization_hash": hashlib.sha256(normalize.serialize_json(norms).encode()).hexdigest(),
    }
    corpus = {
        "images": np.zeros((2, 256, 256, 3), np.uint8),
        "wrist_images": np.zeros((2, 256, 256, 3), np.uint8),
        "states": np.zeros((2, 8), np.float32),
        "noise": np.zeros((2, 10, 32), np.float32),
        "prompts": np.asarray(["first", "second"]),
        "task_indices": np.arange(2),
        "dataset_indices": np.asarray([2, 7]),
        "episode_indices": np.arange(2),
        "frame_indices": np.asarray([2, 2]),
        "metadata_json": np.asarray(json.dumps(identity)),
    }
    corpus_path = tmp_path / "corpus.npz"
    np.savez(corpus_path, **corpus)
    report = {
        **identity,
        "corpus_sha256": staging.sha256_file(corpus_path),
        "decoded_windows": 10,
        "normalization_windows": 0,
        "normalization_omitted_tail": 10,
        "task_window_counts": {"0": 5, "1": 5},
        "transform_variants": ["pi05_libero", *arms],
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(report))
    monkeypatch.setattr(staging, "audit_libero_dataset", lambda *a, **kw: pytest.fail("Do not decode again"))
    return SimpleNamespace(
        root=root,
        manifest=manifest_path,
        assets=assets,
        corpus_path=corpus_path,
        audit_path=audit_path,
        corpus=corpus,
        report=report,
    )


def verify(staged):
    return staging.verify_audit(staged.manifest, staged.audit_path, staged.corpus_path, assets_root=staged.assets)


def test_existing_audit_is_verified_without_dataset_decoding_or_writes(staged):
    paths = [staged.manifest, staged.audit_path, staged.corpus_path]
    before = {path: path.read_bytes() for path in paths}
    assert verify(staged) == staged.report
    assert {path: path.read_bytes() for path in paths} == before


@pytest.mark.parametrize(
    "field",
    [
        "data_manifest_sha256",
        "revision",
        "normalization_hash",
        "corpus_sha256",
        "decoded_windows",
        "normalization_windows",
        "normalization_omitted_tail",
        "task_window_counts",
        "transform_variants",
    ],
)
def test_changed_audit_claims_fail(staged, field):
    staged.report[field] = {} if field == "task_window_counts" else [] if field == "transform_variants" else 999
    staged.audit_path.write_text(json.dumps(staged.report))
    with pytest.raises(ValueError, match="Audit|Corpus"):
        verify(staged)


def test_dataset_changed_since_audit_fails_byte_verification(staged):
    (staged.root / "data/fixture.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="file hash mismatch"):
        verify(staged)


def test_research_arms_must_still_share_canonical_normalization(staged):
    directory = staged.assets / "pi05_libero_b1/physical-intelligence/libero"
    normalize.save(directory, {"actions": normalize.NormStats(mean=np.ones(7), std=np.ones(7))})
    with pytest.raises(ValueError, match="Normalization differs"):
        verify(staged)


@pytest.mark.parametrize("change", ["metadata", "pickle", "duplicate_task", "out_of_range", "nonfinite"])
def test_corpus_requires_safe_metadata_and_complete_task_identities(staged, change):
    if change == "metadata":
        staged.corpus["metadata_json"] = np.asarray("{}")
    elif change == "pickle":
        staged.corpus["metadata_json"] = np.asarray({"untrusted": "object"}, dtype=object)
    elif change == "duplicate_task":
        staged.corpus["task_indices"][:] = 0
    elif change == "out_of_range":
        staged.corpus["episode_indices"][0] = 2
    else:
        staged.corpus["noise"][0, 0, 0] = np.nan
    np.savez(staged.corpus_path, **staged.corpus)
    staged.report["corpus_sha256"] = staging.sha256_file(staged.corpus_path)
    staged.audit_path.write_text(json.dumps(staged.report))
    with pytest.raises(ValueError, match="Corpus|Object arrays|Invalid audited|noise must"):
        verify(staged)


def test_verify_audit_cli_requires_all_retained_inputs():
    with pytest.raises(SystemExit):
        staging.main(["--verify-audit", "--manifest", "manifest.json"])


def test_verify_audit_cli_routes_to_read_only_verification(monkeypatch, capsys):
    calls = []

    def existing(*args):
        calls.append(args)
        return {"decoded_windows": 273465}

    monkeypatch.setattr(staging, "verify_audit", existing)
    staging.main(
        [
            "--verify-audit",
            "--manifest",
            "manifest.json",
            "--audit-output",
            "audit.json",
            "--corpus-output",
            "corpus.npz",
        ]
    )
    assert calls == [(Path("manifest.json"), Path("audit.json"), Path("corpus.npz"))]
    assert "273465" in capsys.readouterr().out
