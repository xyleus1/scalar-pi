"""Record and verify the two exact Docker images retained between instances."""

import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity(commit, model_id, simulator_id):
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("An immutable source commit is required")
    if any(not re.fullmatch(r"sha256:[a-f0-9]{64}", item) for item in (model_id, simulator_id)):
        raise ValueError("Actual Docker image IDs are required")
    return {"commit": commit, "model_id": model_id, "simulator_id": simulator_id}


def verify(receipt, archive, expected, *, hash_archive=False):
    saved = json.loads(receipt.read_text())
    if saved.get("schema_version") != 1 or any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("Retained Docker images differ from their source/image receipt")
    if hash_archive and (
        archive.stat().st_size != saved.get("archive_bytes") or digest(archive) != saved.get("archive_sha256")
    ):
        raise ValueError("Retained Docker archive is incomplete or corrupt")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("record", "verify"))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--simulator-id")
    parser.add_argument("--hash-archive", action="store_true")
    args = parser.parse_args()
    if args.action == "record":
        expected = identity(args.commit, args.model_id or "", args.simulator_id or "")
        if args.receipt.exists():
            raise FileExistsError("Image receipts are immutable")
        saved = {
            "schema_version": 1,
            **expected,
            "archive_sha256": digest(args.archive),
            "archive_bytes": args.archive.stat().st_size,
        }
        temporary = args.receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(saved, indent=2) + "\n")
        temporary.replace(args.receipt)
    else:
        saved = json.loads(args.receipt.read_text())
        expected = identity(args.commit, args.model_id or saved["model_id"], args.simulator_id or saved["simulator_id"])
        verify(args.receipt, args.archive, expected, hash_archive=args.hash_archive)
    print(json.dumps(saved))


if __name__ == "__main__":
    main()
