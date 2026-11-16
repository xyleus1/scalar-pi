"""Admit a Lambda launch only behind one exact, running GitHub deadline watcher.

Authentication is supplied as GH_TOKEN by the private caller. No credential is
retrieved here, and gh children receive no Lambda credentials. A dispatch with
an uncertain response is never redispatched. All launch callers must use
revalidate immediately before POST; its durable intent marker prevents a later
no-launch cancellation from racing that POST.
"""

import argparse
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time
import zipfile

from cloud import project_lock
from watchdog import CLEANUP_SECONDS
from watchdog import EVIDENCE_RESERVE_SECONDS
from watchdog import JOB_TIMEOUT_SECONDS
from watchdog import REPOSITORY
from watchdog import sha256
from watchdog import validate_request
from watchdog import write_json

WORKFLOW = "lambda-watchdog.yml"
WORKFLOW_PATH = ".github/workflows/" + WORKFLOW
API_ROOT = "repos/" + REPOSITORY
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024
MAX_RECEIPT_BYTES = 64 * 1024


class GitHubCommandError(RuntimeError):
    def __init__(self, *, status=None, uncertain=False):
        super().__init__("GitHub command failed; no raw credential-bearing subprocess output is exposed")
        self.http_status = status
        self.uncertain = uncertain


class NotReadyError(RuntimeError):
    """The exact watcher has not reached its required running step yet."""


def gh_environment():
    # An allowlist, rather than guessing every possible Lambda key variable name.
    keep = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "GH_CONFIG_DIR",
    }
    result = {key: value for key, value in os.environ.items() if key.upper() in keep}
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise ValueError("The private caller must supply GH_TOKEN; this module does not retrieve credentials")
    result.update(
        GH_TOKEN=token,
        GH_HOST="github.com",
        GH_PROMPT_DISABLED="1",
        GIT_TERMINAL_PROMPT="0",
        GH_NO_UPDATE_NOTIFIER="1",
        GH_PAGER="cat",
        NO_COLOR="1",
    )
    return result


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("GitHub timestamp must have a timezone")
    return parsed.timestamp()


def read_json(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RECEIPT_BYTES:
        raise ValueError("Expected a small regular evidence JSON file")
    return json.loads(path.read_text(encoding="utf-8"))


class GitHubGuard:
    def __init__(self, gh_path, evidence_root, controller_commit, *, now=time.time, sleep=time.sleep):
        self.gh_path = Path(gh_path).resolve(strict=True)
        self.evidence_root = Path(evidence_root).resolve()
        if not self.gh_path.is_file() or not re.fullmatch(r"[a-f0-9]{40}", controller_commit):
            raise ValueError("Provide an installed gh executable and an immutable controller commit")
        self.controller_commit = controller_commit
        self.now = now
        self.sleep = sleep

    def _gh(self, arguments, *, stdin=None, mutation=False):
        try:
            result = subprocess.run(
                [str(self.gh_path), *arguments],
                input=stdin,
                capture_output=True,
                env=gh_environment(),
                timeout=60,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired):
            raise GitHubCommandError(uncertain=mutation) from None
        if result.returncode:
            status = re.search(rb"HTTP (\d{3})", result.stderr)
            raise GitHubCommandError(status=int(status[1]) if status else None, uncertain=mutation)
        return result.stdout

    def _api(self, endpoint, *, binary=False):
        if not endpoint.startswith(API_ROOT + "/") or ".." in endpoint:
            raise ValueError("GitHub API requests must stay in the project repository")
        data = self._gh(["api", endpoint, "--method", "GET", "-H", "Accept: application/vnd.github+json"])
        if len(data) > MAX_ARCHIVE_BYTES:
            raise ValueError("GitHub evidence response exceeds its size limit")
        return data if binary else json.loads(data)

    def dispatch(self, request, *, dry_run=False):
        if not isinstance(dry_run, bool):
            raise ValueError("dry_run must be a boolean")
        started = self.now()
        request, _ = validate_request(request, started)
        folder = self.evidence_root / request["instance_name"]
        folder.mkdir(parents=True, exist_ok=False)
        guard_path = folder / "guard.json"
        record = {
            "schema_version": 1,
            "repository": REPOSITORY,
            "workflow": WORKFLOW_PATH,
            "controller_commit": self.controller_commit,
            "request": request,
            "request_sha256": sha256(request),
            "dry_run": dry_run,
            "dispatched_unix": started,
            "status": "prepared",
        }
        write_json(guard_path, record)
        workflow = self._api(API_ROOT + "/actions/workflows/" + WORKFLOW)
        branch = self._api(API_ROOT + "/git/ref/heads/main")
        if workflow.get("path") != WORKFLOW_PATH or workflow.get("state") != "active":
            raise ValueError("The trusted watchdog workflow is not active")
        if branch.get("object", {}).get("sha") != self.controller_commit:
            raise ValueError("Published main does not match the reviewed controller commit")
        record.update(workflow_id=workflow["id"], status="dispatch_pending")
        write_json(guard_path, record)
        inputs = json.dumps({"request": json.dumps(request, sort_keys=True), "dry_run": str(dry_run).lower()}).encode()
        try:
            response = (
                self._gh(
                    ["workflow", "run", WORKFLOW, "--repo", REPOSITORY, "--ref", "main", "--json"],
                    stdin=inputs,
                    mutation=True,
                )
                .decode("utf-8")
                .strip()
            )
            match = re.fullmatch(r"https://github\.com/xyleus1/scalar-pi/actions/runs/([0-9]+)", response)
            if match is None:
                raise GitHubCommandError(uncertain=True)
        except (GitHubCommandError, UnicodeDecodeError):
            record["status"] = "dispatch_uncertain"
            write_json(guard_path, record)
            raise GitHubCommandError(uncertain=True) from None
        record.update(run_id=match[1], run_url=response, run_attempt=1, status="dispatched")
        write_json(guard_path, record)
        return guard_path

    def _load(self, guard_path):
        path = Path(guard_path).resolve()
        if not path.is_relative_to(self.evidence_root) or path.name != "guard.json":
            raise ValueError("Guard evidence must stay under the configured private evidence root")
        record = read_json(path)
        if record.get("schema_version") != 1 or record.get("repository") != REPOSITORY:
            raise ValueError("Unknown guard evidence")
        if record.get("controller_commit") != self.controller_commit or record.get("workflow") != WORKFLOW_PATH:
            raise ValueError("Guard belongs to another controller or workflow")
        if record.get("status") != "dispatched" or not str(record.get("run_id", "")).isdigit():
            raise ValueError("Dispatch is incomplete or uncertain; do not redispatch or launch")
        if record.get("run_attempt") != 1:
            raise ValueError("Only the exact initial dispatch attempt is supported")
        canonical, _ = validate_request(record["request"], record["dispatched_unix"])
        if canonical != record["request"] or sha256(canonical) != record["request_sha256"]:
            raise ValueError("Stored dispatch request was changed")
        if path.parent.name != canonical["instance_name"] or not isinstance(record["dry_run"], bool):
            raise ValueError("Guard path or mode differs from its request")
        expected_url = f"https://github.com/{REPOSITORY}/actions/runs/{record['run_id']}"
        if record.get("run_url") != expected_url:
            raise ValueError("Guard URL differs from its exact returned run ID")
        return path, record

    def _run(self, record):
        run = self._api(API_ROOT + "/actions/runs/" + record["run_id"])
        if (
            str(run.get("id")) != record["run_id"]
            or run.get("run_attempt") != record["run_attempt"]
            or run.get("workflow_id") != record["workflow_id"]
            or run.get("event") != "workflow_dispatch"
            or run.get("head_sha") != self.controller_commit
            or run.get("head_branch") != "main"
            or run.get("path", "").split("@", 1)[0] != WORKFLOW_PATH
            or run.get("repository", {}).get("full_name") != REPOSITORY
            or run.get("head_repository", {}).get("full_name") != REPOSITORY
        ):
            raise ValueError("GitHub returned a different workflow run, attempt, repository, or controller commit")
        return run

    def _artifact(self, record, kind):
        response = self._api(API_ROOT + "/actions/runs/" + record["run_id"] + "/artifacts?per_page=100")
        if response["total_count"] > 100:
            raise ValueError("Unexpected number of workflow artifacts; refusing a partial listing")
        name = f"lambda-watchdog-{kind}-{record['run_id']}-{record['run_attempt']}"
        matches = [item for item in response["artifacts"] if item.get("name") == name]
        if not matches:
            raise NotReadyError("The exact watchdog artifact has not been published")
        if len(matches) != 1:
            raise ValueError("Artifact name is ambiguous")
        artifact = matches[0]
        source = artifact.get("workflow_run", {})
        if (
            artifact.get("expired") is not False
            or str(source.get("id")) != record["run_id"]
            or source.get("head_sha") != self.controller_commit
            or source.get("head_branch") != "main"
            or not 0 < artifact.get("size_in_bytes", 0) <= MAX_ARCHIVE_BYTES
            or re.fullmatch(r"sha256:[a-f0-9]{64}", artifact.get("digest", "")) is None
        ):
            raise ValueError("Artifact identity, digest, expiry, or size failed validation")
        return artifact

    def _download(self, record, folder, kind):
        artifact = self._artifact(record, kind)
        destination = folder / f"{kind}-{artifact['id']}"
        metadata_path = destination / "download.json"
        if destination.exists():
            metadata = read_json(metadata_path)
            if metadata["artifact"] != artifact:
                raise ValueError("Published artifact metadata changed after download")
            for name, expected_hash in metadata["files"].items():
                path = destination / name
                if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                    raise ValueError("Downloaded artifact bytes changed")
            return destination
        data = self._api(API_ROOT + f"/actions/artifacts/{artifact['id']}/zip", binary=True)
        if "sha256:" + hashlib.sha256(data).hexdigest() != artifact["digest"]:
            raise ValueError("Downloaded artifact digest differs from GitHub's recorded digest")
        allowed = (
            {"armed.json"} if kind == "armed" else {"armed.json", "binding.json", "result.json", "result.events.jsonl"}
        )
        files = {}
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            if not members or len(members) > len(allowed) or len({item.filename for item in members}) != len(members):
                raise ValueError("Unexpected or duplicate artifact members")
            for item in members:
                if item.filename not in allowed or item.is_dir() or stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Artifact contains an unexpected path or file type")
                if item.file_size > MAX_RECEIPT_BYTES or item.flag_bits & 1:
                    raise ValueError("Artifact member exceeds the evidence size limit")
                with archive.open(item) as stream:
                    contents = stream.read(MAX_RECEIPT_BYTES + 1)
                if len(contents) > MAX_RECEIPT_BYTES:
                    raise ValueError("Expanded artifact member exceeds the evidence size limit")
                files[item.filename] = contents
        if kind == "armed" and set(files) != {"armed.json"}:
            raise ValueError("Armed artifact must contain exactly armed.json")
        destination.mkdir()
        for name, contents in files.items():
            (destination / name).write_bytes(contents)
        (destination / "artifact.zip").write_bytes(data)
        write_json(
            metadata_path,
            {
                "artifact": artifact,
                "files": {name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()},
            },
        )
        return destination

    def _receipt(self, record, path):
        receipt = read_json(path)
        github = {
            "repository": REPOSITORY,
            "run_id": record["run_id"],
            "run_attempt": str(record["run_attempt"]),
            "head_sha": self.controller_commit,
            "ref": "refs/heads/main",
        }
        if (
            receipt.get("schema_version") != 1
            or receipt.get("kind") != "lambda-deadline-watchdog"
            or receipt.get("status") != "armed"
            or receipt.get("dry_run") is not record["dry_run"]
            or receipt.get("github") != github
            or receipt.get("request") != record["request"]
            or receipt.get("request_sha256") != record["request_sha256"]
        ):
            raise ValueError("Armed receipt does not match the dispatched request and run")
        request, deadline = validate_request(receipt["request"], receipt["armed_unix"])
        expected = {
            key: request[key] for key in ("instance_name", "workspace_id", "worker_commit", "region", "instance_type")
        }
        expected["tags"] = {
            "project": "scalar-pi",
            "commit": request["worker_commit"],
            "watchdog-run": record["run_id"] + "-" + str(record["run_attempt"]),
            "watchdog-deadline": str(deadline),
        }
        if (
            receipt["expected"] != expected
            or receipt["deadline_unix"] != deadline
            or receipt["cleanup_deadline_unix"] != deadline + CLEANUP_SECONDS
            or receipt["launch_not_after_unix"] != receipt["armed_unix"] + request["launch_window_seconds"]
            or receipt["job_timeout_seconds"] != JOB_TIMEOUT_SECONDS
            or receipt["evidence_reserve_seconds"] != EVIDENCE_RESERVE_SECONDS
            or receipt["launch_gate_requires_running_step"] != "Watch deadline"
        ):
            raise ValueError("Armed identity, deadline, or timeout bounds differ from the request")
        if not record["dispatched_unix"] - 30 <= receipt["armed_unix"] <= self.now() + 30:
            raise ValueError("Armed timestamp is outside this dispatch's execution window")
        return receipt

    def _validate(self, path, record, *, expected_request=None):
        if (path.parent / "cancel-requested.json").exists():
            raise ValueError("This guard was cancelled or cancellation is uncertain; do not launch")
        if expected_request is not None:
            canonical, _ = validate_request(expected_request, record["dispatched_unix"])
            if canonical != record["request"]:
                raise ValueError("Caller's expected request differs from the dispatched guard")
        run = self._run(record)
        if run["status"] == "completed":
            raise ValueError("The watchdog run already ended; do not launch")
        if run["status"] != "in_progress":
            raise NotReadyError("The watchdog run is not running yet")
        artifact_dir = self._download(record, path.parent, "armed")
        receipt = self._receipt(record, artifact_dir / "armed.json")
        response = self._api(
            API_ROOT + f"/actions/runs/{record['run_id']}/attempts/{record['run_attempt']}/jobs?per_page=100"
        )
        if response["total_count"] > 100:
            raise ValueError("Unexpected number of jobs")
        jobs = [job for job in response["jobs"] if job.get("name") == "cleanup"]
        if len(jobs) != 1:
            raise NotReadyError("The exact cleanup job is not available")
        job = jobs[0]
        if (
            job.get("status") != "in_progress"
            or job.get("head_sha") != self.controller_commit
            or str(job.get("run_id")) != record["run_id"]
            or job.get("conclusion") is not None
            or job.get("run_attempt", record["run_attempt"]) != record["run_attempt"]
        ):
            raise NotReadyError("The cleanup job is not running the expected controller")
        matches = [step for step in job["steps"] if step.get("name") == "Watch deadline"]
        if len(matches) != 1 or matches[0].get("status") != "in_progress" or matches[0].get("conclusion") is not None:
            raise NotReadyError("The foreground Watch deadline step is not running")
        publish = [step for step in job["steps"] if step.get("name") == "Publish armed receipt"]
        if len(publish) != 1 or publish[0].get("conclusion") != "success" or publish[0].get("status") != "completed":
            raise NotReadyError("The armed artifact publication has not completed successfully")
        started = timestamp(job["started_at"])
        if receipt["cleanup_deadline_unix"] + EVIDENCE_RESERVE_SECONDS > started + JOB_TIMEOUT_SECONDS:
            raise ValueError("Actual GitHub job timeout cannot cover cleanup and evidence publication")
        if self._run(record)["status"] != "in_progress":
            raise ValueError("The watchdog ended during validation; do not launch")
        if self.now() >= receipt["launch_not_after_unix"]:
            raise TimeoutError("Launch admission expired; do not launch")
        write_json(
            path.parent / "last-validation.json",
            {
                "observed_unix": self.now(),
                "run_id": record["run_id"],
                "run_attempt": record["run_attempt"],
                "controller_commit": self.controller_commit,
                "job_id": job["id"],
                "job_started_unix": started,
                "watch_step_status": "in_progress",
                "request_sha256": record["request_sha256"],
                "armed_path": str(artifact_dir / "armed.json"),
            },
        )
        return receipt

    def wait_ready(self, guard_path, *, timeout_seconds=600):
        if not 0 < timeout_seconds <= 900:
            raise ValueError("Readiness polling must be bounded to at most fifteen minutes")
        path, record = self._load(guard_path)
        until = min(self.now() + timeout_seconds, timestamp(record["request"]["deadline_utc"]) - 60)
        while self.now() < until:
            try:
                return self._validate(path, record)
            except NotReadyError:
                pass
            except GitHubCommandError as error:
                if error.http_status not in {None, 404, 408, 429, 500, 502, 503, 504}:
                    raise
            self.sleep(min(5, max(0, until - self.now())))
        raise TimeoutError("The exact watchdog did not become ready within the polling allowance")

    def revalidate(self, guard_path, *, expected_request=None):
        path, record = self._load(guard_path)
        if record["dry_run"]:
            raise ValueError("A dry-run guard can never authorize a GPU launch")
        with project_lock(path.with_suffix(".lock")):
            marker = path.parent / "launch-intent.json"
            if marker.exists():
                raise ValueError("This guard already authorized a launch attempt; do not consume it twice")
            receipt = self._validate(path, record, expected_request=expected_request)
            write_json(
                marker,
                {"recorded_unix": self.now(), "run_id": record["run_id"], "request_sha256": record["request_sha256"]},
            )
            if self.now() >= receipt["launch_not_after_unix"]:
                raise TimeoutError("Launch admission expired during validation; do not launch")
            return receipt

    def validate_dry_run(self, guard_path, *, expected_request=None):
        path, record = self._load(guard_path)
        if not record["dry_run"]:
            raise ValueError("Dry-run validation requires a no-cloud guard")
        return self._validate(path, record, expected_request=expected_request)

    def cancel_no_launch(self, guard_path):
        path, record = self._load(guard_path)
        with project_lock(path.with_suffix(".lock")):
            if (path.parent / "launch-intent.json").exists():
                raise ValueError("A launch may have been attempted; keep the remote cleanup guard running")
            marker = path.parent / "cancel-requested.json"
            if marker.exists():
                raise ValueError("Cancellation was already requested; reconcile its exact run without repeating it")
            run = self._run(record)
            write_json(
                marker,
                {
                    "recorded_unix": self.now(),
                    "run_id": record["run_id"],
                    "reason": "No launch authorization was issued by this guard",
                },
            )
            if run["status"] != "completed":
                self._gh(["run", "cancel", record["run_id"], "--repo", REPOSITORY], mutation=True)
            return {
                "run_id": record["run_id"],
                "status": "already_completed" if run["status"] == "completed" else "cancellation_requested",
            }

    def download_result(self, guard_path):
        path, record = self._load(guard_path)
        self._run(record)
        folder = self._download(record, path.parent, "result")
        result_path = folder / "result.json"
        if result_path.exists():
            result = read_json(result_path)
            github = result.get("github", {})
            if (
                result.get("request_sha256") != record["request_sha256"]
                or result.get("dry_run") is not record["dry_run"]
                or github.get("run_id") != record["run_id"]
                or github.get("run_attempt") != str(record["run_attempt"])
                or github.get("head_sha") != self.controller_commit
            ):
                raise ValueError("Cleanup result belongs to another request or run")
        return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("dispatch", "wait-ready", "validate-dry-run", "revalidate", "cancel-no-launch", "download-result"),
    )
    parser.add_argument("--gh", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--controller-commit", required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--guard", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    guard = GitHubGuard(args.gh, args.evidence_root, args.controller_commit)
    if args.action == "dispatch":
        if args.request is None:
            parser.error("dispatch requires --request")
        result = guard.dispatch(read_json(args.request), dry_run=args.dry_run)
    else:
        if args.guard is None:
            parser.error("this action requires --guard")
        result = getattr(guard, args.action.replace("-", "_"))(args.guard)
    print(json.dumps({"evidence_path": str(result)} if isinstance(result, Path) else result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
