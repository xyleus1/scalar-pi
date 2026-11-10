"""Download (optional) and byte-verify the full official LIBERO dataset."""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re

from openpi.research.data import audit_libero_dataset
from openpi.research.data import build_data_manifest
from openpi.training.potential import prepare_offline_dataset
from openpi.training.potential import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="Immutable Hugging Face dataset commit SHA")
    parser.add_argument("--repo-id", default="physical-intelligence/libero")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/libero_manifest.json"))
    parser.add_argument("--download", action="store_true", help="Download missing snapshot files before verification")
    parser.add_argument("--audit", action="store_true", help="Audit all decoded windows and matched transforms offline")
    parser.add_argument(
        "--verify-audit", action="store_true", help="Verify an existing audit/corpus without decoding again"
    )
    parser.add_argument("--manifest", type=Path, help="Use an existing verified manifest; no Hub access")
    parser.add_argument("--audit-output", type=Path, default=Path("data/libero_audit.json"))
    parser.add_argument("--corpus-output", type=Path, help="Export one raw policy observation per task as a new NPZ")
    args = parser.parse_args(argv)
    if args.verify_audit:
        if not args.manifest or not args.corpus_output or args.audit or args.download or args.revision:
            parser.error(
                "--verify-audit requires --manifest and --corpus-output; cannot combine with audit/download/revision"
            )
        report = verify_audit(args.manifest, args.audit_output, args.corpus_output)
        print(f"Verified existing audit and corpus for {report['decoded_windows']} decoded windows")
        return
    if args.manifest:
        if args.download or args.revision or (args.corpus_output and not args.audit):
            parser.error("--manifest cannot combine with --download/--revision; corpus export requires --audit")
        if args.audit:
            audit(args.manifest, args.audit_output, args.corpus_output)
        else:
            manifest = prepare_offline_dataset(args.manifest)
            print(f"Verified {len(manifest['files'])} local files at {manifest['revision']}")
        return
    if args.audit or args.corpus_output:
        parser.error("Offline audit/corpus export requires --audit --manifest after normalization")
    if not args.revision:
        parser.error("--revision is required when preparing a snapshot")
    if re.fullmatch(r"[0-9a-f]{40,64}", args.revision) is None:
        parser.error("--revision must be an immutable commit SHA, not a branch or tag")
    cache_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    lerobot_home = Path(os.environ.get("HF_LEROBOT_HOME", cache_home / "lerobot"))
    root = (args.dataset_root or lerobot_home / args.repo_id).resolve()
    from huggingface_hub import HfApi
    from huggingface_hub import snapshot_download

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


def verify_audit(manifest_path: Path, output: Path, corpus_output: Path, *, assets_root: Path | None = None) -> dict:
    """Recheck retained evidence and current bytes without loading or decoding LeRobot."""
    manifest = prepare_offline_dataset(manifest_path)
    import numpy as np

    from openpi.research.policy_checks import load_corpus
    from openpi.shared import normalize

    report = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Audit must be a JSON object")
    assets_root = assets_root or Path(__file__).resolve().parents[1] / "assets"
    arms = ("pi05_libero_b0", "pi05_libero_b1", "pi05_libero_potential")
    norm_hash = None
    for name in arms:
        norms = normalize.load(assets_root / name / manifest["repo_id"])
        current = hashlib.sha256(normalize.serialize_json(norms).encode()).hexdigest()
        if norm_hash is not None and current != norm_hash:
            raise ValueError(f"Normalization differs for {name}")
        norm_hash = current
    identity = {
        "data_manifest_sha256": sha256_file(manifest_path),
        "revision": manifest["revision"],
        "normalization_hash": norm_hash,
    }
    if any(report.get(key) != value for key, value in identity.items()):
        raise ValueError("Audit identity differs from the current manifest/revision/normalization")
    if report.get("corpus_sha256") != sha256_file(corpus_output):
        raise ValueError("Corpus bytes differ from the audit digest")
    counts = manifest["counts"]
    frames, tasks, episodes = (counts[key] for key in ("total_frames", "total_tasks", "total_episodes"))
    if any(type(value) is not int or value < 1 for value in (frames, tasks, episodes)):
        raise ValueError("Manifest counts must be positive integers")
    for key, expected in (
        ("decoded_windows", frames),
        ("normalization_windows", frames // 256 * 256),
        ("normalization_omitted_tail", frames % 256),
    ):
        if type(report.get(key)) is not int or report[key] != expected:
            raise ValueError(f"Audit {key} differs from manifest counts")
    task_counts = report.get("task_window_counts", {})
    if (
        not isinstance(task_counts, dict)
        or set(task_counts) != {str(index) for index in range(tasks)}
        or any(type(count) is not int or count < 1 for count in task_counts.values())
        or sum(task_counts.values()) != frames
    ):
        raise ValueError("Audit task counts do not cover the manifest")
    if sorted(report.get("transform_variants", [])) != sorted(("pi05_libero", *arms)):
        raise ValueError("Audit does not cover all matched transform variants")
    corpus = load_corpus(corpus_output)
    metadata = corpus.get("metadata_json")
    if (
        metadata is None
        or metadata.shape != ()
        or metadata.dtype.kind != "U"
        or json.loads(metadata.item()) != identity
    ):
        raise ValueError("Corpus metadata differs from the audited identity")
    if corpus["images"].shape != (tasks, 256, 256, 3):
        raise ValueError("Corpus must contain one native-resolution observation per task")
    for key, limit in (
        ("task_indices", tasks),
        ("dataset_indices", frames),
        ("episode_indices", episodes),
        ("frame_indices", frames),
    ):
        values = corpus.get(key)
        if (
            values is None
            or values.shape != (tasks,)
            or values.dtype.kind not in "iu"
            or np.any((values < 0) | (values >= limit))
        ):
            raise ValueError(f"Invalid audited corpus {key}")
    if (
        not np.array_equal(corpus["task_indices"], np.arange(tasks))
        or len(np.unique(corpus["dataset_indices"])) != tasks
    ):
        raise ValueError("Corpus task/window identities are duplicated or incomplete")
    return report


def audit(manifest_path: Path, output: Path, corpus_output: Path | None) -> None:
    manifest = prepare_offline_dataset(manifest_path)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from openpi.research.policy_checks import require_staged_tokenizer
    from openpi.shared import normalize
    from openpi.training import config as configs
    from openpi.training import data_loader

    require_staged_tokenizer()
    config = configs.get_config("pi05_libero_b0")
    data = config.data.create(config.assets_dirs, config.model)
    if data.norm_stats is None:
        raise ValueError("Compute the official normalization asset before auditing transformed samples")
    norm_json = normalize.serialize_json(data.norm_stats)
    dataset = data_loader.create_torch_dataset(data, config.model.action_horizon, config.model)
    source = dataset
    while hasattr(source, "_dataset"):
        source = source._dataset  # noqa: SLF001
    if source.root.resolve() != Path(manifest["dataset_root"]).resolve():
        raise ValueError("The actual loader root differs from the verified dataset")
    variants = {}
    for name in ("pi05_libero", "pi05_libero_b0", "pi05_libero_b1", "pi05_libero_potential"):
        variant = configs.get_config(name)
        variant_data = variant.data.create(variant.assets_dirs, variant.model)
        if name == "pi05_libero":
            variant_data = dataclasses.replace(variant_data, norm_stats=data.norm_stats)
        elif variant_data.norm_stats is None or normalize.serialize_json(variant_data.norm_stats) != norm_json:
            raise ValueError(f"Normalization differs for {name}")
        variants[name] = data_loader.transform_dataset(dataset, variant_data)
    provenance = {
        "data_manifest_sha256": sha256_file(manifest_path),
        "revision": manifest["revision"],
        "normalization_hash": hashlib.sha256(norm_json.encode()).hexdigest(),
    }
    report = audit_libero_dataset(dataset, variants, corpus_output=corpus_output, corpus_metadata=provenance)
    report.update(provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Audited {report['decoded_windows']} windows; wrote {output}")


if __name__ == "__main__":
    main()
