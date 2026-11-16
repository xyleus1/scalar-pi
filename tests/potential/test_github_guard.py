"""GitHub launch admission tests use a fake CLI; no dispatch or API call occurs."""

import datetime as dt
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/lambda"))
    spec = importlib.util.spec_from_file_location("github_guard_test", ROOT / "scripts/lambda/github_guard.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    monkeypatch.setenv("GH_TOKEN", "fixture-github-token")
    return result


@pytest.fixture
def setup(module, monkeypatch, tmp_path):
    now = [1800000000.0]
    gh_path = tmp_path / "gh.exe"
    gh_path.touch()
    guard = module.GitHubGuard(
        gh_path,
        tmp_path / "evidence",
        "a" * 40,
        now=lambda: now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )
    request = {
        "instance_name": "scalar-pi-0123456789ab",
        "workspace_id": "c" * 32,
        "worker_commit": "f53255a5965153172149546dc8eee95b2bd9b076",
        "region": "us-west-3",
        "instance_type": "gpu_1x_h100_pcie",
        "deadline_utc": dt.datetime.fromtimestamp(now[0] + 900, dt.UTC).isoformat(),
        "launch_window_seconds": 600,
    }
    calls = []
    run = {
        "id": 123,
        "run_attempt": 1,
        "workflow_id": 77,
        "event": "workflow_dispatch",
        "head_sha": "a" * 40,
        "head_branch": "main",
        "path": module.WORKFLOW_PATH,
        "repository": {"full_name": module.REPOSITORY},
        "head_repository": {"full_name": module.REPOSITORY},
        "status": "in_progress",
    }
    job = {
        "id": 88,
        "run_id": 123,
        "run_attempt": 1,
        "name": "cleanup",
        "status": "in_progress",
        "head_sha": "a" * 40,
        "conclusion": None,
        "started_at": dt.datetime.fromtimestamp(now[0], dt.UTC).isoformat(),
        "steps": [
            {"name": "Publish armed receipt", "status": "completed", "conclusion": "success"},
            {"name": "Watch deadline", "status": "in_progress", "conclusion": None},
        ],
    }
    receipt = {
        "schema_version": 1,
        "kind": "lambda-deadline-watchdog",
        "status": "armed",
        "dry_run": False,
        "armed_unix": now[0],
        "launch_not_after_unix": now[0] + 600,
        "deadline_unix": now[0] + 900,
        "cleanup_deadline_unix": now[0] + 900 + module.CLEANUP_SECONDS,
        "job_started_unix": now[0],
        "job_timeout_seconds": module.JOB_TIMEOUT_SECONDS,
        "evidence_reserve_seconds": module.EVIDENCE_RESERVE_SECONDS,
        "request": request,
        "request_sha256": module.sha256(request),
        "github": {
            "repository": module.REPOSITORY,
            "run_id": "123",
            "run_attempt": "1",
            "head_sha": "a" * 40,
            "ref": "refs/heads/main",
        },
        "launch_gate_requires_running_step": "Watch deadline",
    }
    receipt["expected"] = {
        key: request[key] for key in ("instance_name", "workspace_id", "worker_commit", "region", "instance_type")
    }
    receipt["expected"]["tags"] = {
        "project": "scalar-pi",
        "commit": request["worker_commit"],
        "watchdog-run": "123-1",
        "watchdog-deadline": str(int(receipt["deadline_unix"])),
    }
    state = SimpleNamespace(
        guard=guard,
        module=module,
        request=request,
        receipt=receipt,
        run=run,
        job=job,
        calls=calls,
        now=now,
        dispatch_response=b"https://github.com/xyleus1/scalar-pi/actions/runs/123\n",
        extra_members={},
        artifact_override={},
        error=None,
    )

    def archive():
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as zipped:
            zipped.writestr("armed.json", json.dumps(state.receipt))
            for name, data in state.extra_members.items():
                zipped.writestr(name, data)
        return stream.getvalue()

    def gh(arguments, *, stdin=None, mutation=False):
        calls.append((arguments, stdin, mutation))
        if state.error:
            raise state.error
        if arguments[:2] == ["workflow", "run"]:
            return state.dispatch_response
        if arguments[:2] == ["run", "cancel"]:
            return b"Cancellation requested"
        endpoint = arguments[1]
        if endpoint.endswith("/actions/workflows/lambda-watchdog.yml"):
            value = {"id": 77, "path": module.WORKFLOW_PATH, "state": "active"}
        elif endpoint.endswith("/git/ref/heads/main"):
            value = {"object": {"sha": "a" * 40}}
        elif endpoint.endswith("/actions/runs/123"):
            value = state.run
        elif endpoint.endswith("/artifacts?per_page=100"):
            data = archive()
            artifact = {
                "id": 456,
                "name": "lambda-watchdog-armed-123-1",
                "expired": False,
                "size_in_bytes": len(data),
                "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
                "workflow_run": {"id": 123, "head_sha": "a" * 40, "head_branch": "main"},
                **state.artifact_override,
            }
            value = {"total_count": 1, "artifacts": [artifact]}
        elif endpoint.endswith("/actions/artifacts/456/zip"):
            return archive()
        elif endpoint.endswith("/attempts/1/jobs?per_page=100"):
            value = {"total_count": 1, "jobs": [state.job]}
        else:
            pytest.fail(f"Unexpected fake gh call: {arguments}")
        return json.dumps(value).encode()

    monkeypatch.setattr(guard, "_gh", gh)
    return state


def dispatch(setup, *, dry_run=False):
    setup.receipt["dry_run"] = dry_run
    return setup.guard.dispatch(setup.request, dry_run=dry_run)


def test_dispatch_captures_exact_returned_run_and_string_inputs(setup):
    path = dispatch(setup)
    record = json.loads(path.read_text())
    assert record["run_id"] == "123"
    arguments, stdin, mutation = next(call for call in setup.calls if call[0][:2] == ["workflow", "run"])
    assert arguments[-1] == "--json"
    assert mutation is True
    assert json.loads(stdin)["dry_run"] == "false"
    assert json.loads(json.loads(stdin)["request"]) == setup.request


@pytest.mark.parametrize("response", [b"", b"Created dispatch", b"https://github.com/other/repo/actions/runs/123"])
def test_unknown_dispatch_result_is_never_redispatched(setup, response):
    setup.dispatch_response = response
    with pytest.raises(setup.module.GitHubCommandError) as error:
        dispatch(setup)
    assert error.value.uncertain is True
    path = setup.guard.evidence_root / setup.request["instance_name"] / "guard.json"
    assert json.loads(path.read_text())["status"] == "dispatch_uncertain"
    with pytest.raises(FileExistsError):
        dispatch(setup)
    assert sum(call[0][:2] == ["workflow", "run"] for call in setup.calls) == 1


def test_ready_then_revalidate_records_one_launch_intent(setup):
    path = dispatch(setup)
    assert setup.guard.wait_ready(path) == setup.receipt
    assert not (path.parent / "launch-intent.json").exists()
    assert setup.guard.revalidate(path, expected_request=setup.request) == setup.receipt
    assert (path.parent / "launch-intent.json").is_file()
    with pytest.raises(ValueError, match="twice"):
        setup.guard.revalidate(path)
    with pytest.raises(ValueError, match="launch may have been attempted"):
        setup.guard.cancel_no_launch(path)


def test_dry_run_is_separate_and_never_authorizes_launch(setup):
    path = dispatch(setup, dry_run=True)
    assert setup.guard.wait_ready(path)["dry_run"] is True
    assert setup.guard.validate_dry_run(path)["dry_run"] is True
    with pytest.raises(ValueError, match="never authorize"):
        setup.guard.revalidate(path)
    assert not (path.parent / "launch-intent.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("head_sha", "d" * 40),
        ("run_attempt", 2),
        ("workflow_id", 99),
        ("head_branch", "other"),
        ("event", "pull_request"),
    ],
)
def test_wrong_run_identity_is_rejected(setup, field, value):
    path = dispatch(setup)
    setup.run[field] = value
    with pytest.raises(ValueError, match="different workflow run"):
        setup.guard.revalidate(path)
    assert not (path.parent / "launch-intent.json").exists()


def test_artifact_alone_cannot_authorize_when_watch_step_is_pending(setup):
    path = dispatch(setup)
    setup.job["steps"][1]["status"] = "pending"
    with pytest.raises(setup.module.NotReadyError, match="foreground"):
        setup.guard.revalidate(path)
    assert not (path.parent / "launch-intent.json").exists()


def test_completed_run_is_not_ready(setup):
    path = dispatch(setup)
    setup.run["status"] = "completed"
    with pytest.raises(ValueError, match="already ended"):
        setup.guard.revalidate(path)


def test_changed_request_inside_artifact_is_rejected(setup):
    path = dispatch(setup)
    setup.receipt["request"] = {**setup.request, "workspace_id": "d" * 32}
    with pytest.raises(ValueError, match="does not match"):
        setup.guard.revalidate(path)


def test_changed_expected_request_from_caller_is_rejected(setup):
    path = dispatch(setup)
    with pytest.raises(ValueError, match="Caller's expected"):
        setup.guard.revalidate(path, expected_request={**setup.request, "instance_name": "scalar-pi-111111111111"})


def test_actual_job_start_must_cover_cleanup_and_upload_reserve(setup):
    path = dispatch(setup)
    setup.job["started_at"] = dt.datetime.fromtimestamp(setup.now[0] - 5 * 3600, dt.UTC).isoformat()
    with pytest.raises(ValueError, match="Actual GitHub job timeout"):
        setup.guard.revalidate(path)


def test_expired_launch_window_is_rejected_after_api_reads(setup):
    path = dispatch(setup)
    setup.now[0] = setup.receipt["launch_not_after_unix"]
    with pytest.raises(TimeoutError, match="admission expired"):
        setup.guard.revalidate(path)


def test_artifact_digest_mismatch_is_rejected(setup):
    path = dispatch(setup)
    setup.artifact_override["digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="digest differs"):
        setup.guard.revalidate(path)


def test_artifact_path_traversal_is_never_extracted(setup):
    path = dispatch(setup)
    setup.extra_members["../escaped.txt"] = "not allowed"
    with pytest.raises(ValueError, match="members"):
        setup.guard.revalidate(path)
    assert not (path.parent / "escaped.txt").exists()


def test_tampered_cached_receipt_is_rejected(setup):
    path = dispatch(setup)
    setup.guard.wait_ready(path)
    (path.parent / "armed-456" / "armed.json").write_text("{}")
    with pytest.raises(ValueError, match="bytes changed"):
        setup.guard.revalidate(path)


def test_no_launch_cancellation_blocks_any_later_admission(setup):
    path = dispatch(setup)
    assert setup.guard.cancel_no_launch(path)["status"] == "cancellation_requested"
    assert sum(call[0][:2] == ["run", "cancel"] for call in setup.calls) == 1
    with pytest.raises(ValueError, match="cancelled"):
        setup.guard.revalidate(path)
    with pytest.raises(ValueError, match="already requested"):
        setup.guard.cancel_no_launch(path)


def test_gh_child_receives_only_github_auth_and_no_lambda_keys(module, monkeypatch, tmp_path):
    monkeypatch.setenv("LAMBDA_API_KEY", "sensitive-parent-key")
    monkeypatch.setenv("LAMBDA_CREATE_API_KEY", "sensitive-launch-key")
    monkeypatch.setenv("LAMBDA_TERMINATE_API_KEY", "sensitive-cleanup-key")
    monkeypatch.setenv("UNUSUALLY_NAMED_CREATOR_CREDENTIAL", "also-secret")
    executable = tmp_path / "gh.exe"
    executable.touch()
    seen = []

    def run(arguments, **kwargs):
        seen.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0, b"{}", b"")

    monkeypatch.setattr(module.subprocess, "run", run)
    guard = module.GitHubGuard(executable, tmp_path, "a" * 40)
    guard._api(module.API_ROOT + "/actions/workflows/lambda-watchdog.yml")  # noqa: SLF001
    arguments, kwargs = seen[0]
    assert kwargs["env"]["GH_TOKEN"] == "fixture-github-token"
    assert not any("LAMBDA" in key or "CREATOR" in key for key in kwargs["env"])
    assert "GH_TOKEN" not in arguments
    assert "shell" not in kwargs
    assert kwargs["capture_output"] is True


def test_missing_gh_token_does_not_fall_back_to_stored_credentials(module, monkeypatch):
    monkeypatch.delenv("GH_TOKEN")
    with pytest.raises(ValueError, match="must supply GH_TOKEN"):
        module.gh_environment()


def test_subprocess_errors_do_not_expose_output_or_secrets(module, monkeypatch, tmp_path):
    executable = tmp_path / "gh.exe"
    executable.touch()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 1, b"private response", b"HTTP 403 fixture-github-token sensitive-parent-key"
        ),
    )
    guard = module.GitHubGuard(executable, tmp_path, "a" * 40)
    with pytest.raises(module.GitHubCommandError) as error:
        guard._api(module.API_ROOT + "/actions/workflows/lambda-watchdog.yml")  # noqa: SLF001
    assert error.value.http_status == 403
    assert "fixture-github-token" not in str(error.value)
    assert "sensitive-parent-key" not in str(error.value)
