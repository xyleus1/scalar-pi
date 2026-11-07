"""Download (optional) and byte-verify the full official LIBERO dataset."""

import argparse
import json
import os
from pathlib import Path
import re

from huggingface_hub import HfApi
from huggingface_hub import snapshot_download

from openpi.research.data import build_data_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Immutable Hugging Face dataset commit SHA")
    parser.add_argument("--repo-id", default="physical-intelligence/libero")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/libero_manifest.json"))
    parser.add_argument("--download", action="store_true", help="Download missing snapshot files before verification")
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40,64}", args.revision) is None:
        parser.error("--revision must be an immutable commit SHA, not a branch or tag")
    cache_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    lerobot_home = Path(os.environ.get("HF_LEROBOT_HOME", cache_home / "lerobot"))
    root = (args.dataset_root or lerobot_home / args.repo_id).resolve()
    info = HfApi().dataset_info(args.repo_id, revision=args.revision, files_metadata=True)
    if info.sha != args.revision:
        raise ValueError("Hub returned a different revision from the pinned commit")
    if args.download:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=info.sha,
            local_dir=root,
            allow_patterns=["meta/**", "data/**", "videos/**"],
        )
    manifest = build_data_manifest(root, repo_id=args.repo_id, revision=info.sha, remote_files=info.siblings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Verified {len(manifest['files'])} files at {info.sha}; wrote {args.output}")
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
