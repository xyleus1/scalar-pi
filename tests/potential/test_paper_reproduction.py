"""Offline public-artifact checks; no GPU, dataset download, or bootstrap run."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPLAY = load("paper_replay", ROOT / "paper/reproduce_descriptive.py")
BUILDER = load("public_release_builder", ROOT / "scripts/build_public_release.py")


class PublicPaperTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((ROOT / "results/study-v0.1.0.json").read_bytes())

    def test_original_statistical_bodies_and_complete_report(self):
        api, manifest, proof = REPLAY.statistics_api(self.report)
        assert "paired_bootstrap" in api
        assert proof["original_functions"] == self.report["source_proof"]["original_functions"]
        active = ast.parse((ROOT / "src/openpi/research/evaluation_statistics.py").read_bytes())
        for node in active.body:
            if isinstance(node, ast.FunctionDef) and node.name in proof["original_functions"]:
                sha = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                assert sha == proof["original_functions"][node.name]
        assert manifest["training_seeds"] == [17, 42, 93]
        assert manifest["num_steps"] == [1, 10]
        cells = self.report["summary"]["trained"] + self.report["summary"]["reference"]
        assert len(cells) == 80
        assert sum(c["episodes"] for c in cells) == 40000
        assert {c["episodes"] for c in cells} == {500}
        assert len(self.report["summary"]["tasks"]) == 800
        assert len(self.report["comparisons"]) == 8
        assert not self.report["hardware_acceptance_claimed"]
        assert not self.report["accepted_full_study_claimed"]
        BUILDER.clean(self.report)
        BUILDER.clean(manifest)

    def test_modified_original_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reproducibility").mkdir()
            pins = json.loads((ROOT / "paper/reproducibility/sources.json").read_bytes())
            (root / "reproducibility/sources.json").write_text(json.dumps(pins), encoding="utf8")
            (root / "reproducibility/statistics.py").write_text("# changed\n", encoding="utf8")
            with patch.object(REPLAY, "HERE", root), pytest.raises(ValueError, match="Pinned source changed"):
                REPLAY.statistics_api(self.report)

    def test_privacy_projection_preserves_scientific_outcome(self):
        original = {
            "checkpoint_id": "P-seed93",
            "group": "P",
            "num_steps": 10,
            "success": False,
            "failure_kind": "policy",
            "failure_tags": ["invalid_action"],
            "artifact_errors": [{"kind": "invalid_action", "debug": "/home/private/log"}],
            "checkpoint_dir": "/home/private/weights",
            "error_message": "error at /home/private/log",
            "server_metadata": {
                "runtime": {"gpu": "NVIDIA H100 PCIe", "machine_id": "private"},
                "research": {"optimizer_updates": 30000, "resume_signature": {"private": True}},
            },
        }
        public = BUILDER.public_row(original, b"original line\n")
        assert not public["success"]
        assert public["failure_tags"] == original["failure_tags"]
        assert public["artifact_errors"] == [{"kind": "invalid_action"}]
        assert public["server_metadata"]["runtime"]["gpu"] == "NVIDIA H100 PCIe"
        assert public["server_metadata"]["research"]["optimizer_updates"] == 30000
        assert "checkpoint_dir" not in public
        assert "error_message" not in public
        BUILDER.clean(public)
        released = BUILDER.metadata({"research": None, "runtime": {"backend": "jax"}})
        assert released == {"research": None, "runtime": {"backend": "jax"}}
        with pytest.raises(ValueError, match="Private metadata"):
            BUILDER.clean({"instance_id": "private"})
        with pytest.raises(ValueError, match="Private metadata"):
            BUILDER.clean({"slot_id": "private/gpu-0"})
        BUILDER.clean({"citation": "https://example.org/paper", "title": r"Policy:\\A study"})
        with pytest.raises(ValueError, match="Private path"):
            BUILDER.clean({"unexpected": "/home/private/log"})

    def test_private_input_cannot_be_used_as_output(self):
        source = Path(tempfile.gettempdir()) / "retained-data"
        with pytest.raises(ValueError, match="must be disjoint"):
            BUILDER.build(source, source)
        with pytest.raises(ValueError, match="must be disjoint"):
            BUILDER.build(source, source / "nested")


if __name__ == "__main__":
    unittest.main()
