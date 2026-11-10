"""Reuse producer artifacts only when source, inputs, outputs and image still match."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def inventory(path):
    path = Path(path)
    if path.is_symlink() or not path.exists():
        raise ValueError(f"Artifact must exist and not be a symlink: {path}")
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    result = {}
    for file in files:
        if file.is_symlink():
            raise ValueError(f"Symlink in staged artifact: {file}")
        if file.is_file():
            name = "." if path.is_file() else file.relative_to(path).as_posix()
            result[name] = {"sha256": digest(file), "bytes": file.stat().st_size}
    if not result:
        raise ValueError(f"Staged artifact is empty: {path}")
    return result


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def pairs(values):
    result = {}
    for value in values:
        label, separator, content = value.partition("=")
        if not separator or not label or not content or label in result:
            raise ValueError("Use unique label=value arguments")
        result[label] = content
    return result


def producer():
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip()
    image = os.environ.get("SCALAR_PI_IMAGE_ID", "")
    if dirty or re.fullmatch(r"[a-f0-9]{40}", commit) is None or re.fullmatch(r"sha256:[a-f0-9]{64}", image) is None:
        raise ValueError("Staging requires a clean source commit and actual SCALAR_PI_IMAGE_ID")
    return {"code_commit": commit, "container_image_id": image}


def specification(args):
    inputs, outputs = pairs(args.input), pairs(args.output)
    if not outputs:
        raise ValueError("A stage must declare its producer outputs")
    for value in outputs.values():
        if args.receipt.resolve().is_relative_to(Path(value).resolve()):
            raise ValueError("Keep receipts outside their own artifact directories")
    return {
        "schema_version": 1,
        "stage": args.stage,
        "producer": producer(),
        "parameters": pairs(args.parameter),
        "inputs": {key: {"path": str(Path(path).resolve()), "files": inventory(path)} for key, path in inputs.items()},
        "outputs": {key: str(Path(path).resolve()) for key, path in outputs.items()},
    }


def verify(receipt, expected):
    record = json.loads(receipt.read_text())
    if record.get("specification") != expected:
        raise ValueError(f"Stage source, inputs or parameters changed: {receipt}")
    actual = {key: inventory(path) for key, path in expected["outputs"].items()}
    if record.get("output_files") != actual:
        raise ValueError(f"Stage output bytes changed: {receipt}")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "run"))
    parser.add_argument("--stage", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--output", action="append", default=[])
    parser.add_argument("--parameter", action="append", default=[])
    parser.add_argument("--allow-existing-output", action="store_true", help="Only for resumable upstream downloads")
    args, command = parser.parse_known_args(argv)
    if command and command[0] == "--":
        command = command[1:]
    expected = specification(args)
    if args.receipt.exists():
        verify(args.receipt, expected)
        print(f"Reused verified stage: {args.stage}")
        return
    if args.action == "verify":
        raise FileNotFoundError(f"Stage receipt is missing: {args.receipt}")
    if not command:
        parser.error("run requires the producer command after --")
    pending = args.receipt.with_suffix(".pending.json")
    owned_retry = pending.exists() and json.loads(pending.read_text()) == expected
    if pending.exists() and not owned_retry:
        raise ValueError("Interrupted stage belongs to different inputs or source")
    if (
        not owned_retry
        and not args.allow_existing_output
        and any(Path(path).exists() for path in expected["outputs"].values())
    ):
        raise FileExistsError("Unreceipted output exists; preserve and inspect it before retrying")
    write_json(pending, expected)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    record = {
        "specification": expected,
        "output_files": {key: inventory(path) for key, path in expected["outputs"].items()},
        "producer_seconds": time.perf_counter() - started,
    }
    write_json(args.receipt, record)
    pending.unlink()
    print(f"Recorded verified stage: {args.stage}")


if __name__ == "__main__":
    main()
