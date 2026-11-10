"""Fixed-input policy checks shared by conversion and latency diagnostics."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def require_staged_tokenizer() -> Path:
    """HF offline flags do not cover OpenPI's GCS SentencePiece download."""
    path = (
        Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
        / "big_vision/paligemma_tokenizer.model"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Stage the PaliGemma tokenizer before offline policy execution: {path}")
    return path


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpus(path: Path) -> dict[str, np.ndarray]:
    """Read the numeric/string NPZ emitted by the dataset audit; never unpickle."""
    with np.load(path, allow_pickle=False) as archive:
        corpus = {name: archive[name] for name in archive.files}
    required = {"images", "wrist_images", "states", "prompts", "noise"}
    if not required.issubset(corpus):
        raise ValueError(f"Missing corpus arrays: {sorted(required - corpus.keys())}")
    count = len(corpus["states"])
    if count < 1 or any(len(corpus[name]) != count for name in required):
        raise ValueError("Corpus arrays must have the same positive sample count")
    for name in ("images", "wrist_images"):
        values = corpus[name]
        if values.dtype != np.uint8 or values.ndim != 4 or values.shape[-1] != 3:
            raise ValueError(f"{name} must be uint8 [N,H,W,3]")
    if corpus["images"].shape != corpus["wrist_images"].shape:
        raise ValueError("Camera image shapes must match")
    for name, shape in (("states", (count, 8)), ("noise", (count, 10, 32))):
        if corpus[name].dtype != np.float32 or corpus[name].shape != shape or not np.isfinite(corpus[name]).all():
            raise ValueError(f"{name} must be finite float32 with shape {shape}")
    if corpus["prompts"].shape != (count,) or corpus["prompts"].dtype.kind != "U":
        raise ValueError("prompts must be a one-dimensional Unicode array")
    return corpus


def corpus_observation(corpus: dict, index: int) -> dict:
    return {
        "observation/image": corpus["images"][index],
        "observation/wrist_image": corpus["wrist_images"][index],
        "observation/state": corpus["states"][index],
        "prompt": str(corpus["prompts"][index]),
    }


def input_digest(inputs: dict) -> str:
    """Identify actual transformed input arrays, including dtype and shape."""
    digest = hashlib.sha256()

    def visit(value, path):
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], (*path, key))
        else:
            array = np.asarray(value)
            digest.update(json.dumps([path, str(array.dtype), list(array.shape)]).encode())
            digest.update(np.ascontiguousarray(array).tobytes())

    visit(inputs, ())
    return digest.hexdigest()


def array_error(actual, expected, *, atol: float, rtol: float) -> dict:
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} versus {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {"passed": False, "reason": "nonfinite output", "atol": atol, "rtol": rtol}
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "passed": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
        "max_abs": float(np.abs(difference).max()),
        "rms": float(np.sqrt(np.mean(difference**2))),
        "atol": atol,
        "rtol": rtol,
    }


def timing_summary(samples: list[dict], *, horizon: int = 10, replan: int = 5) -> dict:
    if not samples:
        raise ValueError("At least one measured call is required")
    result = {}
    for key in ("infer_ms", "total_ms", "cuda_ms"):
        if all(key in sample for sample in samples):
            values = np.asarray([sample[key] for sample in samples], dtype=np.float64)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"Invalid {key} measurements")
            result[key] = {"median": float(np.median(values)), "p95": float(np.quantile(values, 0.95))}
    if not {"infer_ms", "total_ms"}.issubset(result):
        raise ValueError("Model and total policy measurements are required")
    seconds = result["total_ms"]["median"] / 1000
    result["generated_actions_per_second"] = horizon / seconds
    result["executed_actions_per_second"] = replan / seconds
    return result
