"""Component calibration tests use fabricated timing receipts, never models or simulators."""

import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from scripts import calibrate_potential_evaluation as cal


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    source = {"commit": "a" * 40, "tracked_changes": "", "source_sha256": "b" * 64}
    spec = {
        "output_dir": str(tmp_path),
        "corpus": "corpus.npz",
        "smoke_manifest": "smoke.json",
        "policy_command_prefix": ["model", "benchmark"],
        "simulator_command_prefix": ["simulator"],
        "policies": [
            {"group": group, "config": config, "checkpoint_dir": f"weights/{group}"}
            for group, config in cal.GROUPS.items()
        ],
    }
    jobs = cal.commands(spec)
    for job in jobs:
        if job["kind"] == "policy":
            report = {
                "purpose": "isolated_warmed_policy_latency",
                "source": source,
                "corpus_sha256": "c" * 64,
                "num_steps": job["num_steps"],
                "measured_calls": 120,
                "warmup_calls": 10,
                "corpus_observations": 40,
                "batch_size": 1,
                "startup": {"model_load_seconds": 2.0, "first_query_seconds": 1.0},
                "calls": [{"corpus_index": index % 40, "total_ms": 10.0} for index in range(120)],
                "policy": {
                    "config_name": cal.GROUPS[job["group"]],
                    "num_steps": job["num_steps"],
                    "checkpoint_sha256": "d" * 64,
                    "runtime": {
                        "container_image": "model-image",
                        "backend": "jax" if job["group"] == "reference" else "pytorch",
                    },
                },
            }
        else:
            suite = job["suite"]
            report = {
                "purpose": "libero_scene_timing",
                "suite": suite,
                "render_size": 256,
                "resize_size": 224,
                "num_steps_wait": 10,
                "max_steps": cal.ev.MAX_STEPS[suite],
                "process_seconds": 20.0,
                "provenance_seconds": 1.0,
                "provenance": {"source": source, "container_image": "sim-image"},
                "tasks": [
                    {
                        "task_id": task,
                        "task_start_parity": True,
                        "measured_steps": cal.ev.MAX_STEPS[suite] + 10,
                        "create_seconds": 1.0,
                        "reset_seconds": 0.1,
                        "settling_seconds": 0.1,
                        "step_seconds_p95": 0.01,
                        "preprocess_seconds_p95": 0.001,
                    }
                    for task in range(10)
                ],
            }
        path = Path(job["path"])
        path.write_text(json.dumps(report))
        path.with_suffix(".execution.json").write_text(
            json.dumps({"argv": job["argv"], "seconds": 25.0, "report_sha256": cal.ev.file_digest(path)})
        )
    monkeypatch.setattr(cal, "smoke_overhead", lambda _: (0.002, []))
    return (
        spec,
        jobs,
        {
            "source": source,
            "model_image": "model-image",
            "simulator_image": "sim-image",
            "corpus_hash": "c" * 64,
            "checkpoint_hashes": dict.fromkeys(cal.GROUPS, "d" * 64),
        },
    )


def test_calibration_uses_all_native_horizons_and_eight_policy_settings(evidence):
    spec, jobs, kwargs = evidence
    result = cal.compose(spec, jobs, **kwargs)
    assert len(jobs) == 12
    assert result["passed"]
    assert not result["valid_benchmark_evaluation"]
    assert len(result["evidence"]) == 24
    for group in cal.GROUPS:
        for steps in ("1", "10"):
            durations = result["evaluation_seconds_per_episode"][group][steps]
            assert durations["libero_10"] > durations["libero_spatial"]
            for suite in cal.ev.SUITES:
                assert result["task_budget_seconds"][group][steps][suite] == pytest.approx(1.2 * 50 * durations[suite])


@pytest.mark.parametrize("change", ["checkpoint", "corpus", "source", "coverage", "scene", "parity", "receipt"])
def test_calibration_rejects_changed_or_incomplete_evidence(evidence, change):
    spec, jobs, kwargs = evidence
    job = jobs[-1] if change in {"scene", "parity"} else jobs[0]
    path = Path(job["path"])
    report = json.loads(path.read_text())
    if change == "checkpoint":
        report["policy"]["checkpoint_sha256"] = "changed"
    elif change == "corpus":
        report["corpus_sha256"] = "changed"
    elif change == "source":
        report["source"]["commit"] = "changed"
    elif change == "coverage":
        report["calls"][0]["corpus_index"] = 999
    elif change == "scene":
        report["tasks"][0]["measured_steps"] = 1
    elif change == "parity":
        report["tasks"][0]["task_start_parity"] = False
    else:
        report["num_steps"] = 99
    path.write_text(json.dumps(report))
    if change != "receipt":
        receipt_path = path.with_suffix(".execution.json")
        receipt = json.loads(receipt_path.read_text())
        receipt["report_sha256"] = cal.ev.file_digest(path)
        receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="calibration|timing|Scene|receipt|parity"):
        cal.compose(spec, jobs, **kwargs)


def test_calibration_preview_never_authenticates_launches_or_loads_models(tmp_path, monkeypatch, capsys):
    spec = {
        "output_dir": str(tmp_path / "future"),
        "corpus": "missing.npz",
        "smoke_manifest": "missing.json",
        "policy_command_prefix": ["model"],
        "simulator_command_prefix": ["simulator"],
        "policies": [
            {"group": group, "config": config, "checkpoint_dir": f"missing/{group}"}
            for group, config in cal.GROUPS.items()
        ],
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(cal.subprocess, "Popen", lambda *a, **kw: pytest.fail("Preview launched a process"))
    cal.main(["--spec", str(path), "--output", str(tmp_path / "report.json")])
    assert json.loads(capsys.readouterr().out)["dry_run"]
    assert list(tmp_path.iterdir()) == [path]


def test_scene_calibration_uses_native_components_and_checks_task_initialization(tmp_path, monkeypatch):
    created = []

    class Environment:
        def __init__(self):
            created.append(self)

        def seed(self, seed):
            np.random.seed(seed)

        def reset(self):
            self.obs = {
                "agentview_image": np.zeros((2, 2, 3), np.uint8),
                "robot0_eye_in_hand_image": np.zeros((2, 2, 3), np.uint8),
                "state": np.random.rand(3),
            }

        def set_init_state(self, state):
            return self.obs

        def step(self, action):
            assert len(action) == 7
            return self.obs, 0, False, {}

        def close(self):
            pass

    class Suite:
        n_tasks = 10

        def get_task(self, index):
            return types.SimpleNamespace(language=f"task{index}", problem_folder="x", bddl_file=f"{index}.bddl")

        def get_task_init_states(self, index):
            return [np.zeros(3)]

    modules = {
        "imageio": {},
        "libero": {},
        "libero.libero": {
            "benchmark": types.SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": Suite}),
            "get_libero_path": lambda key: str(tmp_path),
        },
        "libero.libero.envs": {"OffScreenRenderEnv": lambda **kwargs: Environment()},
        "openpi_client": {
            "image_tools": types.SimpleNamespace(
                convert_to_uint8=lambda value: value, resize_with_pad=lambda value, *args: value
            ),
            "websocket_client_policy": types.SimpleNamespace(
                WebsocketClientPolicy=lambda *a, **kw: pytest.fail("Scene timing must not load a policy")
            ),
        },
        "tqdm": {"tqdm": lambda value: value},
        "tyro": {},
    }
    for name, values in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "fake_scene_runner", Path(__file__).resolve().parents[2] / "examples/libero/main.py"
    )
    runner = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, runner)
    spec.loader.exec_module(runner)
    fake_research = types.SimpleNamespace(
        SUITES=cal.ev.SUITES,
        MAX_STEPS=cal.ev.MAX_STEPS,
        file_digest=lambda path: "b" * 64,
        suite_provenance=lambda **kwargs: {"tasks": kwargs["tasks"]},
    )
    output = tmp_path / "scene.json"
    runner.calibrate_scenes(runner.Args(calibration_output=str(output)), fake_research)
    report = json.loads(output.read_text())
    assert len(created) == 20  # Native traversal plus standalone task-boundary checks.
    assert len(report["tasks"]) == 10
    assert all(task["measured_steps"] == 230 and task["task_start_parity"] for task in report["tasks"])
    assert report["valid_benchmark_evaluation"] is False
    with pytest.raises(FileExistsError, match="already exists"):
        runner.calibrate_scenes(runner.Args(calibration_output=str(output)), fake_research)
