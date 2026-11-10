"""Evaluation tests run without LIBERO, CUDA, checkpoints, or model dependencies."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

from openpi.research import evaluation as ev


def _manifest():
    return {
        "schema_version": 1,
        "diagnostic": True,
        "suites": ["libero_spatial"],
        "num_steps": [1, 10],
        "num_trials_per_task": 3,
        "num_tasks_per_suite": 2,
        "training_seeds": [0, 1, 2],
        "variants": [
            {
                "group": group,
                "training_seed": seed,
                "checkpoint_id": f"{group}-{seed}",
                "config": f"pi05_{group}",
                "checkpoint_dir": f"checkpoints/{group}/{seed}",
            }
            for group in ["B1", "P"]
            for seed in [0, 1, 2]
        ],
    }


def _research_metadata(cell):
    if cell.get("reference", False):
        return None
    return {
        "trainer": "potential_pi",
        "training_seed": cell["training_seed"],
        "optimizer_updates": 30000,
        "inference_weights": "ema",
        "velocity_mode": "potential_dot" if cell["group"] == "P" else "direct",
        "initialization": "pretrained_output" if cell["group"] == "B0" else "zero_output",
        "freeze_vlm": True,
        "action_horizon": 10,
        "action_dim": 32,
        "base_checkpoint_hash": "b" * 64,
        "data_manifest_sha256": "d" * 64,
        "source_snapshot_sha256": "e" * 64,
        "container_image_id": "sha256:fixture-model",
        "model_sha256": ev.content_digest(cell["checkpoint_id"]),
        "normalization_hash": "a" * 64,
        "resume_signature": {"microbatch_size": 4, "effective_batch_size": 256},
    }


def _records(manifest):
    records = []
    for cell in ev.matrix_cells(manifest):
        for task in range(manifest.get("num_tasks_per_suite", 10)):
            for state in range(manifest.get("num_trials_per_task", 50)):
                records.append(  # noqa: PERF401 -- Keep the synthetic record fixture readable.
                    {
                        **{key: value for key, value in cell.items() if key != "episode_log_path"},
                        "schema_version": 1,
                        "protocol_id": ev.PROTOCOL_ID,
                        "episode_id": ev.episode_id(cell["suite"], task, state, 7),
                        "task_id": task,
                        "initial_state_id": state,
                        "rollout_seed": 7,
                        "paired_noise_seed": 7,
                        "replan_steps": 5,
                        "action_horizon": 10,
                        "num_steps_wait": 10,
                        "max_steps": ev.MAX_STEPS[cell["suite"]],
                        "success": (task + state + (cell.get("training_seed") or 0)) % 2 == 0,
                        "failure_tags": [],
                        "latencies": [
                            {"infer_ms": 12.0, "total_ms": 14.0, "client_roundtrip_ms": 15.0},
                            {"infer_ms": 20.0, "total_ms": 22.0, "client_roundtrip_ms": 23.0},
                        ],
                        "policy_queries": 2,
                        "query_noise_seeds": [
                            ev.paired_noise_seed(7, f"{cell['suite']}/{task}", state, q) for q in range(2)
                        ],
                        "server_metadata": {
                            "checkpoint_id": cell["checkpoint_id"],
                            "config_name": cell["config"],
                            "num_steps": cell["num_steps"],
                            "research": _research_metadata(cell),
                        },
                    }
                )
    if manifest.get("require_provenance", not manifest.get("diagnostic", False)):
        for row in records:
            metadata = row["server_metadata"]
            metadata.update(
                checkpoint_sha256=ev.content_digest(row["checkpoint_id"]),
                normalization_sha256="a" * 64,
                matmul_precision="highest",
                runtime={"container_image": "sha256:fixture-model", "gpu": "fixture"},
            )
            provenance = {
                "schema_version": 1,
                "protocol_id": ev.PROTOCOL_ID,
                "server_metadata": copy.deepcopy(metadata),
                "source": {"commit": "f" * 40, "source_sha256": "e" * 64, "tracked_changes": ""},
                "simulator_commit": "a" * 40,
                "simulator_assets": {"assets": {"files": 1, "sha256": "f" * 64}},
                "container_image": "sha256:fixture-simulator",
                "settings": {
                    key: row[key]
                    for key in ("suite", "num_steps", "rollout_seed", "paired_noise_seed", "replan_steps", "attempt_id")
                },
                "tasks": [
                    {
                        "task_id": task,
                        "bddl_sha256": "b" * 64,
                        "initial_states_sha256": ["c" * 64] * manifest.get("num_trials_per_task", 50),
                    }
                    for task in range(manifest.get("num_tasks_per_suite", 10))
                ],
            }
            provenance["settings"].update(
                render_size=256, resize_size=224, num_steps_wait=10, max_steps=ev.MAX_STEPS[row["suite"]]
            )
            row["_provenance"] = provenance
            row["provenance_sha256"] = ev.content_digest(provenance)
    return records


def _sides(records):
    return (
        [row for row in records if row["group"] == "P" and row["num_steps"] == 1],
        [row for row in records if row["group"] == "B1" and row["num_steps"] == 1],
    )


def test_noise_schedule_stable_across_processes_and_unambiguous():
    seed = ev.paired_noise_seed(7, "libero_10/3", 4, 5)
    code = "from openpi.research.evaluation import paired_noise_seed; print(paired_noise_seed(7,'libero_10/3',4,5))"
    output = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert int(output) == seed
    assert 0 <= seed < 2**32
    assert len({ev.paired_noise_seed(7, "libero_10/3", 4, q) for q in range(100)}) == 100
    assert ev.paired_noise_seed(7, "a1", 2, 3) != ev.paired_noise_seed(7, "a", 12, 3)
    with pytest.raises(ValueError, match="nonnegative"):
        ev.paired_noise_seed(7, "a", 0, -1)


def test_primary_matrix_counts_and_three_independent_seeds():
    manifest = _manifest()
    for key in ("diagnostic", "suites", "num_trials_per_task", "num_tasks_per_suite"):
        manifest.pop(key)
    manifest["variants"].append(
        {
            "group": "reference",
            "training_seed": None,
            "checkpoint_id": "released",
            "reference": True,
            "config": "pi05_libero",
            "checkpoint_dir": "gs://reference",
        }
    )
    manifest["variants"].extend(
        {
            "group": "B0",
            "training_seed": seed,
            "checkpoint_id": f"B0-{seed}",
            "config": "pi05_B0",
            "checkpoint_dir": f"checkpoints/B0/{seed}",
        }
        for seed in [0, 1, 2]
    )
    assert len(ev.matrix_cells(manifest)) * 500 == 40000
    manifest["variants"].pop(0)
    with pytest.raises(ValueError, match="declared training seeds"):
        ev.validate_manifest(manifest)


def test_primary_rejects_silently_reduced_benchmark():
    manifest = _manifest()
    manifest["diagnostic"] = False
    with pytest.raises(ValueError, match="Primary matrix"):
        ev.validate_manifest(manifest)


def test_complete_matrix_and_jsonl_roundtrip(tmp_path):
    manifest = _manifest()
    records = _records(manifest)
    path = tmp_path / "episodes.jsonl"
    for row in records:
        ev.append_episode(path, row)
    loaded = ev.read_episodes([path])
    assert loaded == records
    assert ev.validate_results(loaded, manifest) == {"episodes": 72, "cells": 12, "checkpoints": 6}


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "extra", "seed", "horizon", "failure", "success"])
def test_validation_rejects_incomplete_or_confounded_results(corruption):
    manifest = _manifest()
    records = _records(manifest)
    if corruption == "missing":
        records.pop()
    elif corruption == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif corruption == "extra":
        records[0]["initial_state_id"] = 99
    elif corruption == "seed":
        records[0]["paired_noise_seed"] = 99
    elif corruption == "horizon":
        records[0]["action_horizon"] = 50
    elif corruption == "failure":
        records[0]["failure_tags"] = ["policy_inference_error"]
    else:
        records[0]["success"] = "false"
    with pytest.raises(ValueError, match=r"Incomplete|Duplicate|Unexpected|mismatch|Execution errors|Invalid"):
        ev.validate_results(records, manifest)


def test_summary_keeps_released_reference_separate():
    records = _records(_manifest())
    reference = copy.deepcopy(records[0])
    reference.update(checkpoint_id="released", group="reference", training_seed=None, reference=True)
    result = ev.summarize([*records, reference])
    assert len(result["reference"]) == 1
    assert len(result["trained"]) == 12
    assert result["reference"][0]["training_seed"] is None
    assert result["trained"][0]["latency_ms"]["infer_ms"]["mean"] == 16.0


def test_bootstrap_preserves_pairs_and_known_difference():
    left, right = _sides(_records(_manifest()))
    equal = ev.paired_bootstrap(left, right, resamples=100)
    assert equal["ci95"] == [0.0, 0.0]
    assert "bootstrap_equivalent" not in equal
    for row in left:
        row["success"] = True
    for row in right:
        row["success"] = False
    result = ev.paired_bootstrap(left, right, resamples=100)
    assert result["difference_left_minus_right"] == 1.0
    assert result["ci95"] == [1.0, 1.0]
    assert "equivalence_margin" not in result


def test_bootstrap_reproducible_and_accounts_for_training_seeds():
    left, right = _sides(_records(_manifest()))
    for row in right:
        row["success"] = False
    for row in left:
        row["success"] = row["training_seed"] == 0
    one = ev.paired_bootstrap(left, right, resamples=300, random_seed=42)
    two = ev.paired_bootstrap(list(reversed(left)), list(reversed(right)), resamples=300, random_seed=42)
    assert one == two
    assert one["ci95"][1] > one["ci95"][0]
    assert one["training_seeds"] == [0, 1, 2]


def test_bootstrap_rejects_missing_or_wrong_pairs():
    left, right = _sides(_records(_manifest()))
    with pytest.raises(ValueError, match="identical"):
        ev.paired_bootstrap(left[:-1], right, resamples=100)
    right[0]["paired_noise_seed"] = 43
    with pytest.raises(ValueError, match="Pairing mismatch"):
        ev.paired_bootstrap(left, right, resamples=100)


def _load_script(name):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_dry_run_builds_safe_argv_without_launching(tmp_path):
    manifest = _manifest()
    manifest["output_dir"] = str(tmp_path / "results with spaces")
    manifest["server_command_prefix"] = ["missing-server", "server.py"]
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    script = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_potential.py"
    output = subprocess.check_output([sys.executable, str(script), str(path)], text=True)
    plan = json.loads(output)
    assert plan["dry_run"] is True
    assert plan["expected_episodes"] == 72
    assert len(plan["batches"]) == 12
    server = plan["batches"][0]["server"]
    assert server[server.index("--num-steps") + 1] == "1"
    assert server[server.index("--checkpoint-id") + 1] == "B1-0"
    assert not Path(manifest["output_dir"]).exists()


def test_analysis_end_to_end_and_missing_file_failure(tmp_path):
    manifest = _manifest()
    manifest["output_dir"] = str(tmp_path)
    records = _records(manifest)
    for cell in ev.matrix_cells(manifest):
        for row in records:
            if (row["checkpoint_id"], row["num_steps"], row["suite"]) == (
                cell["checkpoint_id"],
                cell["num_steps"],
                cell["suite"],
            ):
                ev.append_episode(cell["episode_log_path"], row)
    module = _load_script("analyze_potential")
    result = module.analyze(manifest, comparisons=["P:B1"], resamples=100, random_seed=0)
    assert result["validation"]["episodes"] == 72
    assert len(result["comparisons"]) == 4
    Path(ev.matrix_cells(manifest)[0]["episode_log_path"]).unlink()
    with pytest.raises(FileNotFoundError):
        module.analyze(manifest, comparisons=["P:B1"], resamples=100, random_seed=0)


@pytest.mark.parametrize("corruption", ["metadata", "noise", "latency"])
def test_validation_checks_server_attestation_and_actual_noise_schedule(corruption):
    manifest = _manifest()
    records = _records(manifest)
    if corruption == "metadata":
        records[0]["server_metadata"]["num_steps"] = 77
    elif corruption == "noise":
        records[0]["query_noise_seeds"][0] = 42
    else:
        records[0]["latencies"][0]["infer_ms"] = float("nan")
    with pytest.raises(ValueError, match=r"metadata|noise schedule|latency"):
        ev.validate_results(records, manifest)


@pytest.mark.parametrize("inference_failure", [False, True, "nan", "shape", "structured", "video"])
def test_official_runner_logs_and_sends_paired_noise_with_fake_simulator(tmp_path, monkeypatch, inference_failure):
    """Exercise the changed runner without importing any real simulator libraries."""
    import numpy as np

    received = []

    def observation():
        return {
            "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.zeros(2),
        }

    class FakeEnvironment:
        def seed(self, seed):
            assert seed == 7

        def reset(self):
            self.steps = 0

        def set_init_state(self, state):
            return observation()

        def step(self, action):
            self.steps += 1
            return observation(), 0.0, self.steps == 11, {}

        def close(self):
            pass

    class FakeSuite:
        n_tasks = 1

        def get_task(self, index):
            return types.SimpleNamespace(language="test task", problem_folder="x", bddl_file="task.bddl")

        def get_task_init_states(self, index):
            return [0, 1]

    class FakeClient:
        def __init__(self, host, port):
            pass

        def get_server_metadata(self):
            return {"checkpoint_id": "P-17", "config_name": "pi05_P", "num_steps": 1}

        def infer(self, observation):
            received.append(observation["noise_seed"])
            if inference_failure is True:
                raise RuntimeError("Test server failure")
            if inference_failure == "structured":
                return {
                    "policy_error": {"type": "invalid_actions", "message": "nonfinite"},
                    "policy_timing": {"infer_ms": 12.0, "total_ms": 15.0},
                }
            if inference_failure in {"nan", "shape"}:
                actions = np.full((10, 7), np.nan) if inference_failure == "nan" else np.zeros((9, 7))
                return {"actions": actions, "policy_timing": {"infer_ms": 12.0, "total_ms": 15.0}}
            return {"actions": np.zeros((10, 7)), "policy_timing": {"infer_ms": 12.0, "total_ms": 15.0}}

    def write_video(*args, **kwargs):
        if inference_failure == "video":
            raise OSError("Video destination unavailable")

    modules = {
        "imageio": {"mimwrite": write_video},
        "libero": {},
        "libero.libero": {
            "benchmark": types.SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": FakeSuite}),
            "get_libero_path": lambda key: str(tmp_path),
        },
        "libero.libero.envs": {"OffScreenRenderEnv": lambda **kwargs: FakeEnvironment()},
        "openpi_client": {
            "image_tools": types.SimpleNamespace(convert_to_uint8=lambda x: x, resize_with_pad=lambda x, *args: x),
            "websocket_client_policy": types.SimpleNamespace(WebsocketClientPolicy=FakeClient),
        },
        "tqdm": {"tqdm": lambda iterable: iterable},
        "tyro": {},
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[2] / "examples" / "libero" / "main.py"
    spec = importlib.util.spec_from_file_location("fake_libero_runner", path)
    runner = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, runner)
    spec.loader.exec_module(runner)
    log = tmp_path / "episodes.jsonl"
    args = runner.Args(
        episode_log_path=str(log),
        checkpoint_id="P-17",
        checkpoint_dir="checkpoint",
        config="pi05_P",
        group="P",
        training_seed=17,
        num_steps=1,
        paired_noise_seed=7,
        num_trials_per_task=2,
        save_videos=inference_failure == "video",
        video_out_path=str(tmp_path / "videos"),
    )
    if inference_failure is True:
        with pytest.raises(RuntimeError, match="logged execution error"):
            runner.eval_libero(args)
        failed = ev.read_episodes([log])
        assert len(failed) == 1
        assert failed[0]["failure_tags"] == ["policy_inference_error"]
        assert failed[0]["success"] is False
        return
    runner.eval_libero(args)
    rows = ev.read_episodes([log])
    assert len(rows) == 2
    assert received == [ev.paired_noise_seed(7, "libero_spatial/0", state, 0) for state in range(2)]
    if inference_failure in {"nan", "shape", "structured"}:
        assert all(not row["success"] and row["failure_kind"] == "policy" for row in rows)
        assert all(row["failure_tags"] == ["invalid_actions"] for row in rows)
    else:
        assert all(row["success"] and row["steps"] == 1 and row["action_horizon"] == 10 for row in rows)
    if inference_failure == "video":
        assert all(row["artifact_errors"][0]["kind"] == "video_error" and not row["failure_tags"] for row in rows)
    manifest = {
        "schema_version": 1,
        "diagnostic": True,
        "suites": ["libero_spatial"],
        "num_steps": [1],
        "num_trials_per_task": 2,
        "num_tasks_per_suite": 1,
        "variants": [
            {
                "checkpoint_id": "P-17",
                "checkpoint_dir": "checkpoint",
                "config": "pi05_P",
                "group": "P",
                "training_seed": 17,
            }
        ],
    }
    assert ev.validate_results(rows, manifest)["episodes"] == 2


@pytest.mark.parametrize("group", ["B0", "B1", "P", "reference"])
def test_primary_manifest_rejects_missing_entire_group(group):
    path = Path(__file__).resolve().parents[2] / "configs" / "potential" / "libero_matrix.example.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["variants"] = [v for v in manifest["variants"] if v["group"] != group]
    with pytest.raises(ValueError, match="requires B0, B1, P"):
        ev.validate_manifest(manifest)


@pytest.mark.parametrize("missing", ["all", "client_roundtrip_ms", "infer_ms", "total_ms"])
def test_validation_requires_each_latency_measurement(missing):
    manifest = _manifest()
    rows = _records(manifest)
    if missing == "all":
        rows[0]["latencies"][0] = {}
    else:
        del rows[0]["latencies"][0][missing]
    with pytest.raises(ValueError, match="Missing required policy latency fields"):
        ev.validate_results(rows, manifest)


def _cell_rows(records, cell):
    return [
        row for row in records if all(row[field] == cell[field] for field in ("checkpoint_id", "num_steps", "suite"))
    ]


def _write_rows(path, rows):
    if rows and rows[0].get("_provenance"):
        ev.write_provenance(path, rows[0]["_provenance"])
    for row in rows:
        ev.append_episode(path, {key: value for key, value in row.items() if key != "_provenance"})


def test_resume_validates_cell_without_relaxing_original_primary_manifest(tmp_path):
    path = Path(__file__).resolve().parents[2] / "configs" / "potential" / "libero_matrix.example.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["output_dir"] = str(tmp_path)
    original = copy.deepcopy(manifest)
    cell = ev.matrix_cells(manifest)[0]
    subset = {
        **manifest,
        "diagnostic": True,
        "variants": [manifest["variants"][0]],
        "num_steps": [cell["num_steps"]],
        "suites": [cell["suite"]],
    }
    _write_rows(cell["episode_log_path"], _records(subset))
    module = _load_script("evaluate_potential")
    with pytest.raises(FileExistsError, match="resume-completed-suites"):
        module.commands(manifest)
    plan = module.commands(manifest, resume_completed_suites=True)
    assert manifest == original
    assert manifest["diagnostic"] is False
    assert sum(len(batch["completed_suites"]) for batch in plan) == 1
    assert sum(len(batch["clients"]) for batch in plan) == 79
    assert plan[0]["completed_suites"][0]["episodes"] == 500
    assert "--no-sync" in plan[0]["server"]


@pytest.mark.parametrize("corruption", ["partial", "duplicate", "error", "mismatch", "empty", "malformed"])
def test_resume_rejects_unusable_suite_logs(tmp_path, corruption):
    manifest = _manifest()
    manifest["output_dir"] = str(tmp_path)
    cell = ev.matrix_cells(manifest)[0]
    rows = _cell_rows(_records(manifest), cell)
    if corruption == "partial":
        rows.pop()
    elif corruption == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif corruption == "error":
        rows[0]["failure_tags"] = ["policy_inference_error"]
    elif corruption == "mismatch":
        rows[0]["server_metadata"]["num_steps"] = 77
    path = Path(cell["episode_log_path"])
    path.parent.mkdir(parents=True)
    if corruption in {"empty", "malformed"}:
        path.write_text("" if corruption == "empty" else "{incomplete", encoding="utf-8")
    else:
        _write_rows(path, rows)
    module = _load_script("evaluate_potential")
    with pytest.raises(ValueError, match="needs investigation; cannot resume"):
        module.commands(manifest, resume_completed_suites=True)


def test_resume_all_completed_never_launches_server_and_validates_full_matrix(tmp_path, monkeypatch, capsys):
    manifest = _manifest()
    manifest["output_dir"] = str(tmp_path)
    rows = _records(manifest)
    for cell in ev.matrix_cells(manifest):
        _write_rows(cell["episode_log_path"], _cell_rows(rows, cell))
    module = _load_script("evaluate_potential")

    def unexpected(*args, **kwargs):
        pytest.fail("No process or network access is needed for a fully completed matrix")

    monkeypatch.setattr(module.subprocess, "Popen", unexpected)
    monkeypatch.setattr(module.socket, "create_connection", unexpected)
    module.execute(manifest, resume_completed_suites=True)
    assert json.loads(capsys.readouterr().out)["episodes"] == 72
    assert not list(tmp_path.rglob("server.log"))


def test_resume_runs_only_pending_suite_and_appends_server_log(tmp_path, monkeypatch, capsys):
    manifest = _manifest()
    manifest.update(output_dir=str(tmp_path), suites=["libero_spatial", "libero_goal"], num_steps=[1])
    manifest["variants"] = manifest["variants"][:1]
    cells = ev.matrix_cells(manifest)
    rows = _records(manifest)
    _write_rows(cells[0]["episode_log_path"], _cell_rows(rows, cells[0]))
    server_log = Path(cells[0]["episode_log_path"]).parent / "server.log"
    server_log.write_text("previous server attempt\n", encoding="utf-8")
    module = _load_script("evaluate_potential")
    launched = []
    executed = []

    class FakeProcess:
        def __init__(self, argv, **kwargs):
            launched.append(argv)
            kwargs["stdout"].write("new server attempt output\n")

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    def connection_refused(*args, **kwargs):
        raise ConnectionRefusedError

    def run_client(argv, **kwargs):
        executed.append(argv)
        assert argv[argv.index("--args.task-suite-name") + 1] == "libero_goal"
        _write_rows(cells[1]["episode_log_path"], _cell_rows(rows, cells[1]))

    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(module, "_run_client", run_client)
    monkeypatch.setattr(module.socket, "create_connection", connection_refused)
    monkeypatch.setattr(module, "_wait_for_server", lambda *args: None)
    module.execute(manifest, resume_completed_suites=True)
    assert len(launched) == len(executed) == 1
    assert json.loads(capsys.readouterr().out)["episodes"] == 12
    assert server_log.read_text(encoding="utf-8").startswith("previous server attempt\n")
    assert "new server attempt output" in server_log.read_text(encoding="utf-8")


@pytest.mark.parametrize("needs_kill", [False, True])
def test_client_timeout_gives_wrapper_a_cleanup_grace_period(tmp_path, monkeypatch, needs_kill):
    module = _load_script("evaluate_potential")
    events = []

    class Process:
        running = True

        def poll(self):
            return None if self.running else 0

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            if timeout == 1 or (timeout == 30 and needs_kill):
                raise module.subprocess.TimeoutExpired("fixture", timeout)
            self.running = False
            return 0

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")

    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: Process())
    with pytest.raises(module.subprocess.TimeoutExpired):
        module._run_client(["fixture"], cwd=tmp_path, env={}, timeout=1)  # noqa: SLF001
    expected = [("wait", 1), "terminate", ("wait", 30)]
    if needs_kill:
        expected.extend(["kill", ("wait", None)])
    assert events == expected


def test_completed_process_is_not_signalled(monkeypatch):
    module = _load_script("evaluate_potential")
    process = types.SimpleNamespace(
        poll=lambda: 0,
        terminate=lambda: pytest.fail("Completed processes must not be signalled"),
    )
    module._stop_process(process)  # noqa: SLF001


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trainer", "other_trainer"),
        ("training_seed", 999),
        ("optimizer_updates", 29999),
        ("inference_weights", "raw"),
        ("velocity_mode", "potential_dot"),
        ("initialization", "pretrained_output"),
        ("freeze_vlm", False),
        ("action_horizon", 50),
        ("action_dim", 7),
    ],
)
def test_primary_checkpoint_metadata_rejects_nonmatching_training_protocol(field, value):
    manifest = _manifest()
    rows = _records(manifest)
    assert rows[0]["group"] == "B1"
    rows[0]["server_metadata"]["research"][field] = value
    with pytest.raises(ValueError, match=f"Actual checkpoint {field}"):
        ev.validate_results(rows, manifest, require_primary_checkpoint=True)


def test_primary_checkpoint_requires_metadata_but_diagnostic_allows_missing_metadata():
    manifest = _manifest()
    rows = _records(manifest)
    rows[0]["server_metadata"]["research"] = None
    assert ev.validate_results(rows, manifest)["episodes"] == 72
    with pytest.raises(ValueError, match="Missing or invalid actual checkpoint"):
        ev.validate_results(rows, manifest, require_primary_checkpoint=True)


def test_diagnostic_checks_actual_training_seed_when_checkpoint_metadata_exists():
    manifest = _manifest()
    rows = _records(manifest)
    rows[0]["server_metadata"]["research"]["training_seed"] = 999
    with pytest.raises(ValueError, match="Actual checkpoint training_seed"):
        ev.validate_results(rows, manifest)


def test_primary_potential_requires_potential_field_and_reference_has_no_training_attestation():
    manifest = _manifest()
    rows = _records(manifest)
    next(row for row in rows if row["group"] == "P")["server_metadata"]["research"]["velocity_mode"] = "direct"
    with pytest.raises(ValueError, match="Actual checkpoint velocity_mode"):
        ev.validate_results(rows, manifest, require_primary_checkpoint=True)
    reference = {
        "group": "reference",
        "training_seed": None,
        "reference": True,
        "checkpoint_id": "released",
        "config": "pi05_libero",
        "checkpoint_dir": "gs://reference",
    }
    manifest["variants"] = [reference]
    assert ev.validate_results(_records(manifest), manifest, require_primary_checkpoint=True)["episodes"] == 12


def test_resume_primary_cell_cannot_hide_partial_training_behind_diagnostic_validation_view(tmp_path):
    path = Path(__file__).resolve().parents[2] / "configs" / "potential" / "libero_matrix.example.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["output_dir"] = str(tmp_path)
    cell = ev.matrix_cells(manifest)[0]
    subset = {
        **manifest,
        "diagnostic": True,
        "variants": [manifest["variants"][0]],
        "num_steps": [cell["num_steps"]],
        "suites": [cell["suite"]],
    }
    rows = _records(subset)
    rows[0]["server_metadata"]["research"]["optimizer_updates"] = 100
    _write_rows(cell["episode_log_path"], rows)
    module = _load_script("evaluate_potential")
    with pytest.raises(ValueError, match="Actual checkpoint optimizer_updates"):
        module.commands(manifest, resume_completed_suites=True)


def test_pilot_keeps_all_tasks_and_exposure_but_labels_single_seed():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "configs/potential/libero_matrix.pilot.json").read_text())
    assert len(ev.matrix_cells(manifest)) * 500 == 16000
    assert manifest["training_seeds"] == [42]
    manifest.pop("study_kind")
    with pytest.raises(ValueError, match="explicitly labeled"):
        ev.validate_manifest(manifest)


def test_bootstrap_holds_tasks_fixed_and_labels_one_seed_uncertainty():
    manifest = _manifest()
    manifest["variants"] = [v for v in manifest["variants"] if v["training_seed"] == 0]
    left, right = _sides(_records(manifest))
    for row in left:
        row["success"] = row["task_id"] == 0
    for row in right:
        row["success"] = row["task_id"] == 1
    result = ev.paired_bootstrap(left, right, resamples=100)
    assert result["difference_left_minus_right"] == 0
    assert result["ci95"] == [0, 0]  # Resampling tasks would introduce spurious variation here.
    assert result["training_seed_variability_estimated"] is False
    assert "not estimated" in result["scope"]


def test_fresh_suite_attempt_preserves_previous_log(tmp_path):
    manifest = _manifest()
    manifest["output_dir"] = str(tmp_path)
    original = ev.matrix_cells(manifest)[0]
    _write_rows(original["episode_log_path"], _cell_rows(_records(manifest), original)[:1])
    manifest["suite_attempts"] = {"B1-0/k1/libero_spatial": "retry-02"}
    replacement = ev.matrix_cells(manifest)[0]
    assert Path(original["episode_log_path"]).exists()
    assert Path(replacement["episode_log_path"]).parent.name == "retry-02"
    plan = _load_script("evaluate_potential").commands(manifest)
    client = plan[0]["clients"][0]["argv"]
    assert client[client.index("--args.attempt-id") + 1] == "retry-02"


def test_provenance_roundtrip_detects_changed_sidecar(tmp_path):
    manifest = _manifest()
    manifest.update(require_provenance=True, output_dir=str(tmp_path))
    manifest["variants"] = manifest["variants"][:1]
    manifest["num_steps"] = [1]
    rows = _records(manifest)
    path = ev.matrix_cells(manifest)[0]["episode_log_path"]
    _write_rows(path, rows)
    loaded = ev.read_episodes([path])
    assert ev.validate_results(loaded, manifest)["episodes"] == 6
    sidecar = Path(path).with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text())
    provenance["server_metadata"]["normalization_sha256"] = "9" * 64
    sidecar.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="changed provenance"):
        ev.read_episodes([path])


@pytest.mark.parametrize("field", ["normalization_sha256", "checkpoint_sha256", "matmul_precision"])
def test_provenance_binds_episode_server_metadata(field):
    manifest = _manifest()
    manifest["require_provenance"] = True
    rows = _records(manifest)
    rows[0]["server_metadata"][field] = "changed"
    with pytest.raises(ValueError, match="differs from its provenance"):
        ev.validate_results(rows, manifest)


def test_provenance_rejects_different_research_normalization():
    manifest = _manifest()
    manifest["require_provenance"] = True
    rows = _records(manifest)
    changed = next(row for row in rows if row["group"] == "P")
    changed["server_metadata"]["normalization_sha256"] = "9" * 64
    changed["server_metadata"]["research"]["normalization_hash"] = "9" * 64
    changed["_provenance"]["server_metadata"] = copy.deepcopy(changed["server_metadata"])
    changed["provenance_sha256"] = ev.content_digest(changed["_provenance"])
    with pytest.raises(ValueError, match="different normalization"):
        ev.validate_results(rows, manifest)


@pytest.mark.parametrize("field", ["checkpoint_sha256", "normalization_sha256"])
def test_provenance_rejects_served_bytes_that_differ_from_trainer_receipt(field):
    manifest = _manifest()
    manifest["require_provenance"] = True
    rows = _records(manifest)
    row = rows[0]
    row["server_metadata"][field] = "9" * 64
    row["_provenance"]["server_metadata"] = copy.deepcopy(row["server_metadata"])
    row["provenance_sha256"] = ev.content_digest(row["_provenance"])
    with pytest.raises(ValueError, match="differs from the trainer"):
        ev.validate_results(rows, manifest)


def test_profile_checkpoint_cannot_enter_confirmatory_evaluation():
    manifest = _manifest()
    rows = _records(manifest)
    rows[0]["server_metadata"]["research"]["run_mode"] = "profile"
    with pytest.raises(ValueError, match="Profiling checkpoints"):
        ev.validate_results(rows, manifest, require_primary_checkpoint=True)
