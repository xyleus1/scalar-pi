"""Verify local LeRobot bytes against an immutable Hugging Face snapshot."""

import hashlib
import json
from pathlib import Path
import re
from typing import Any


def build_data_manifest(root: Path, *, repo_id: str, revision: str, remote_files: list[Any]) -> dict:
    """Check remote LFS SHA256 / Git blob SHA1, then hash every loaded file.

    A revision label alone is not provenance: these checks bind the local bytes
    to the remote commit, including small metadata files not stored in LFS.
    """
    if re.fullmatch(r"[0-9a-f]{40,64}", revision) is None:
        raise ValueError("An immutable hexadecimal dataset revision is required")
    root = root.resolve()
    remote = {
        item.rfilename: item for item in remote_files if item.rfilename.split("/")[0] in {"meta", "data", "videos"}
    }
    local = {
        path.relative_to(root).as_posix(): path
        for directory in ("meta", "data", "videos")
        for path in (root / directory).rglob("*")
        if path.is_file()
    }
    if "meta/info.json" not in remote or not any(name.startswith("data/") for name in remote):
        raise ValueError("Remote snapshot is not a complete LeRobot dataset")
    if set(local) != set(remote):
        raise ValueError(
            f"Snapshot file mismatch: missing={sorted(remote.keys() - local.keys())}, "
            f"extra={sorted(local.keys() - remote.keys())}"
        )
    hashes = {}
    for relative, path in sorted(local.items()):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Dataset symlink escapes root: {relative}")
        item = remote[relative]
        size = path.stat().st_size
        if item.size is not None and item.size != size:
            raise ValueError(f"Remote size mismatch: {relative}")
        sha256 = hashlib.sha256()
        git_blob = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                sha256.update(block)
                git_blob.update(block)
        lfs = item.lfs
        expected = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        if expected is not None:
            matches = sha256.hexdigest() == expected
        elif item.blob_id:
            matches = git_blob.hexdigest() == item.blob_id
        else:
            raise ValueError(f"Remote API supplied no content digest: {relative}")
        if not matches:
            raise ValueError(f"Local bytes differ from requested revision: {relative}")
        hashes[relative] = sha256.hexdigest()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "repo_id": repo_id,
        "revision": revision,
        "dataset_root": str(root),
        "files": hashes,
        "counts": {key: info.get(key) for key in ("total_episodes", "total_frames", "total_tasks")},
        "verified_against": "huggingface_git_blob_or_lfs_digest",
    }


def audit_libero_dataset(
    dataset,
    variants: dict[str, Any],
    *,
    corpus_output: Path | None = None,
    corpus_metadata: dict | None = None,
    expected_counts: dict[str, int] | None = None,
) -> dict:
    """Audit decoded official-loader samples and transformed episode-boundary fixtures.

    ``dataset`` is OpenPI's task-prompt wrapper; ``variants`` maps configuration
    names to the corresponding officially transformed views of that dataset.
    This streams all windows without retaining images or changing model inputs.
    """
    import numpy as np

    source = dataset
    while hasattr(source, "_dataset"):
        source = source._dataset  # noqa: SLF001 -- official transform wrapper.
    info = source.meta.info
    expected_counts = expected_counts or {"total_episodes": 1693, "total_frames": 273465, "total_tasks": 40}
    if any(info.get(key) != count for key, count in expected_counts.items()):
        raise ValueError("Dataset counts do not match the official LIBERO snapshot")
    if len(dataset) != info["total_frames"] or len(source.meta.tasks) != info["total_tasks"]:
        raise ValueError("Loaded dataset length or task metadata disagrees with info.json")
    starts = np.asarray(source.episode_data_index["from"])
    ends = np.asarray(source.episode_data_index["to"])
    if (
        len(starts) != info["total_episodes"]
        or len(ends) != len(starts)
        or starts[0] != 0
        or ends[-1] != len(dataset)
        or np.any(ends <= starts)
        or not np.array_equal(ends[:-1], starts[1:])
    ):
        raise ValueError("Episode boundaries do not partition the dataset")
    boundaries = {
        int(i) for start, end in zip(starts, ends, strict=True) for i in (start, (start + end - 1) // 2, end - 1)
    }
    task_counts = dict.fromkeys(source.meta.tasks, 0)
    fixture_ids = []
    corpus = {}

    def compare(expected, actual, prefix=""):
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or expected.keys() != actual.keys():
                raise ValueError(f"Transformed keys differ at {prefix}")
            for key in expected:
                compare(expected[key], actual[key], f"{prefix}/{key}")
        else:
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected), err_msg=prefix)

    for index in range(len(dataset)):
        sample = dataset[index]
        episode = int(np.asarray(sample["episode_index"]).item())
        frame = int(np.asarray(sample["frame_index"]).item())
        task = int(np.asarray(sample["task_index"]).item())
        timestamp = float(np.asarray(sample["timestamp"]).item())
        if episode < 0 or episode >= len(starts) or not starts[episode] <= index < ends[episode]:
            raise ValueError(f"Wrong episode identity at window {index}")
        if frame != index - starts[episode] or int(np.asarray(sample["index"]).item()) != index:
            raise ValueError(f"Wrong frame identity at window {index}")
        if task not in task_counts or sample["prompt"] != source.meta.tasks[task]:
            raise ValueError(f"Wrong task prompt at window {index}")
        if not np.isfinite(timestamp) or abs(timestamp - frame / info["fps"]) > 1e-3:
            raise ValueError(f"Wrong timestamp at window {index}")
        task_counts[task] += 1
        for name, shape in (("state", (8,)), ("actions", (10, 7))):
            value = np.asarray(sample[name])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name} at window {index}")
        for name in ("image", "wrist_image"):
            value = np.asarray(sample[name])
            if value.shape != (3, 256, 256) or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                raise ValueError(f"Invalid decoded {name} at window {index}")
        if index not in boundaries:
            continue
        # Compare the loader's chunk to raw actions, clamping at this episode's end.
        indices = np.minimum(np.arange(index, index + 10), ends[episode] - 1)
        actions = np.stack([np.asarray(source.hf_dataset[int(i)]["actions"]) for i in indices])
        np.testing.assert_array_equal(np.asarray(sample["actions"]), actions)
        expected_pad = np.arange(index, index + 10) >= ends[episode]
        np.testing.assert_array_equal(np.asarray(sample["actions_is_pad"]), expected_pad)
        reference = variants["pi05_libero"][index]
        for name, transformed in variants.items():
            actual = transformed[index]
            compare(reference, actual, name)
        if np.asarray(reference["actions"]).shape != (10, 32) or np.count_nonzero(reference["actions"][..., 7:]):
            raise ValueError("Wrong model action shape or padding")
        if reference["image_mask"] != {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False}:
            raise ValueError("Wrong LIBERO camera masks")
        fixture_ids.append({"index": index, "episode_index": episode, "frame_index": frame, "timestamp": timestamp})
        if corpus_output is not None and task not in corpus and index == (starts[episode] + ends[episode] - 1) // 2:
            corpus[task] = {
                "images": np.moveaxis((np.asarray(sample["image"]) * 255).astype(np.uint8), 0, -1),
                "wrist_images": np.moveaxis((np.asarray(sample["wrist_image"]) * 255).astype(np.uint8), 0, -1),
                "states": np.asarray(sample["state"], dtype=np.float32),
                "prompts": sample["prompt"],
                "dataset_indices": index,
                "task_indices": task,
                "episode_indices": episode,
                "frame_indices": frame,
            }
    if any(count == 0 for count in task_counts.values()):
        raise ValueError("A declared task has no training frames")
    report = {
        "decoded_windows": len(dataset),
        "task_window_counts": task_counts,
        "boundary_fixtures": fixture_ids,
        "transform_variants": list(variants),
        "normalization_windows": len(dataset) // 256 * 256,
        "normalization_omitted_tail": len(dataset) % 256,
        "terminal_actions": "repeated within episode; original padding loss treatment retained",
    }
    if corpus_output is not None:
        if len(corpus) != info["total_tasks"]:
            raise ValueError("Policy corpus does not cover every task")
        corpus_output = Path(corpus_output)
        if corpus_output.suffix != ".npz" or corpus_output.exists():
            raise ValueError("Corpus output must be a new .npz file")
        corpus_output.parent.mkdir(parents=True, exist_ok=True)
        ordered = [corpus[task] for task in sorted(corpus)]
        arrays = {key: np.stack([sample[key] for sample in ordered]) for key in ordered[0]}
        arrays["noise"] = np.random.default_rng(7).standard_normal((len(ordered), 10, 32)).astype(np.float32)
        arrays["metadata_json"] = np.asarray(json.dumps(corpus_metadata or {}, sort_keys=True))
        np.savez(corpus_output, **arrays)
        report["corpus_sha256"] = hashlib.sha256(corpus_output.read_bytes()).hexdigest()
        report["corpus_path"] = str(corpus_output)
    return report
