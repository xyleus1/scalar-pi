"""Small Torch-only building blocks for reproducible potential-expert training."""

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Any

import numpy as np
from safetensors import safe_open
import safetensors.torch
import torch
from torch.optim.swa_utils import AveragedModel
from torch.optim.swa_utils import get_ema_multi_avg_fn

from openpi.shared.potential_checkpoint import METADATA_FILENAME
from openpi.shared.potential_checkpoint import SCHEMA_VERSION
from openpi.shared.potential_checkpoint import validate_checkpoint_metadata


def stable_seed(seed: int, *parts: object) -> int:
    payload = json.dumps([seed, *parts], separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def torch_rng_scope(seed: int, device: torch.device):
    """Independent corruption/augmentation streams without advancing global RNG."""
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA checkpoint RNG cannot be restored without CUDA")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


class ResumableBatchSampler(torch.utils.data.Sampler):
    """Deterministic shuffled epochs, resuming from *consumed* microbatches.

    The trainer owns the committed cursor. DataLoader prefetch never changes it.
    Epoch tails are dropped at the effective-batch boundary, matching a loader
    with batch_size=effective_batch_size and drop_last=True.
    """

    def __init__(
        self,
        dataset_size: int,
        microbatch_size: int,
        effective_batch_size: int,
        seed: int,
        *,
        start_microbatch: int = 0,
    ):
        if microbatch_size <= 0 or effective_batch_size % microbatch_size:
            raise ValueError("Positive microbatch must divide effective batch")
        if dataset_size < effective_batch_size or start_microbatch < 0:
            raise ValueError("Dataset must contain a complete effective batch; cursor must be nonnegative")
        self.dataset_size = dataset_size
        self.microbatch_size = microbatch_size
        self.effective_batch_size = effective_batch_size
        self.seed = seed
        self.start_microbatch = start_microbatch
        self.microbatches_per_epoch = (dataset_size // effective_batch_size) * (effective_batch_size // microbatch_size)

    def __iter__(self) -> Iterator[list[int]]:
        epoch, offset = divmod(self.start_microbatch, self.microbatches_per_epoch)
        while True:
            generator = torch.Generator().manual_seed(stable_seed(self.seed, "data", epoch))
            indices = torch.randperm(self.dataset_size, generator=generator).tolist()
            for batch in range(offset, self.microbatches_per_epoch):
                first = batch * self.microbatch_size
                yield indices[first : first + self.microbatch_size]
            epoch += 1
            offset = 0


def learning_rate(step: int, *, warmup_steps: int, peak_lr: float, decay_steps: int, decay_lr: float) -> float:
    """The upstream OpenPI warmup/cosine schedule, indexed by optimizer update."""
    if step < 0 or warmup_steps < 0:
        raise ValueError("Schedule step counts must be nonnegative")
    if step < warmup_steps:
        initial = peak_lr / (warmup_steps + 1)
        return initial + (peak_lr - initial) * step / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
    return decay_lr + (peak_lr - decay_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_ema(expert: torch.nn.Module, decay: float = 0.999) -> AveragedModel:
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0, 1)")
    ema = AveragedModel(expert, multi_avg_fn=get_ema_multi_avg_fn(decay))
    ema.requires_grad_(requires_grad=False)
    # AveragedModel otherwise copies theta_1 on its first call. Seed theta_0
    # now so the first optimizer update uses the exact OpenPI recurrence.
    ema.update_parameters(expert)
    return ema


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def validate_data_manifest(path: Path, *, repo_id: str, dataset_root: Path) -> dict[str, Any]:
    """Verify an immutable HF revision and all data/metadata bytes actually loaded."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("repo_id") != repo_id:
        raise ValueError("Dataset manifest schema or repository mismatch")
    if re.fullmatch(r"[0-9a-fA-F]{40,64}", str(manifest.get("revision", ""))) is None:
        raise ValueError("Dataset revision must be an immutable commit hash")
    root = Path(dataset_root).resolve()
    if Path(manifest.get("dataset_root", "")).resolve() != root:
        raise ValueError("Manifest dataset_root differs from the dataset actually loaded")
    files = manifest.get("files")
    if not isinstance(files, dict) or "meta/info.json" not in files:
        raise ValueError("Dataset manifest must hash meta/info.json and all data files")
    required = {
        item.relative_to(root).as_posix()
        for folder in ("meta", "data", "videos")
        for item in (root / folder).rglob("*")
        if item.is_file()
    }
    if not any(name.startswith("data/") for name in required) or not required.issubset(files):
        raise ValueError("Dataset manifest does not cover the complete loaded dataset")
    for relative, expected_hash in files.items():
        item = (root / relative).resolve()
        if not item.is_relative_to(root) or not item.is_file():
            raise ValueError(f"Invalid dataset manifest path: {relative}")
        if sha256_file(item) != expected_hash:
            raise ValueError(f"Dataset file hash mismatch: {relative}")
    return manifest


def validate_base_conversion(directory: Path, *, require_receipt: bool) -> dict[str, Any] | None:
    """Reject irreversibly rounded base weights and require canonical provenance."""
    directory = Path(directory)
    receipt_path = directory / "config.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else None
    if receipt is not None and not isinstance(receipt, dict):
        raise ValueError("Base conversion config.json must be an object")
    if receipt is not None and receipt.get("precision") not in (None, "float32"):
        raise ValueError(
            "Base conversion must use --precision float32; casting rounded weights cannot recover precision"
        )
    if require_receipt:
        if receipt is None or receipt.get("precision") != "float32":
            raise ValueError("Canonical training requires an FP32 conversion config.json receipt")
        hashes = receipt.get("source_files_sha256")
        if not receipt.get("source_checkpoint") or not isinstance(hashes, dict) or not hashes:
            raise ValueError("Canonical conversion receipt must record the source checkpoint and source file hashes")
        if any(re.fullmatch(r"[0-9a-fA-F]{64}", str(value)) is None for value in hashes.values()):
            raise ValueError("Conversion source hashes must be SHA256 digests")
    # Read headers only; do not allocate another complete model. The receipt is
    # insufficient if a later conversion actually changed the stored precision.
    with safe_open(directory / "model.safetensors", framework="pt", device="cpu") as weights:
        rounded = [
            name
            for name in weights.keys()  # noqa: SIM118 -- safe_open is not an iterable mapping.
            if weights.get_slice(name).get_dtype() in {"BF16", "F16", "F64", "F8_E4M3", "F8_E5M2"}
        ]
    if rounded:
        raise ValueError(f"Base weights are not all FP32: {rounded[:3]}")
    return receipt


@contextmanager
def use_ema_weights(expert: torch.nn.Module, ema: AveragedModel):
    # CPU backup avoids a second full expert copy in scarce GPU memory.
    original = {name: value.detach().cpu().clone() for name, value in expert.state_dict().items()}
    try:
        expert.load_state_dict(ema.module.state_dict(), strict=True)
        yield
    finally:
        expert.load_state_dict(original, strict=True)


def save_checkpoint(
    directory: Path,
    *,
    model: torch.nn.Module,
    expert: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: AveragedModel,
    state: dict[str, Any],
    metadata: dict[str, Any],
    assets: dict[str, str] | None = None,
) -> Path:
    """Publish a complete directory by same-filesystem rename, never overwrite.

    model.safetensors contains EMA inference weights; raw_model.safetensors and
    training_state.pt are for exact trusted-local resumption.
    """
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {directory}")
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        embedded = {
            "trainer": "potential_pi",
            "schema_version": str(SCHEMA_VERSION),
            **{key: str(metadata[key]) for key in ("velocity_mode", "action_horizon", "action_dim")},
        }
        safetensors.torch.save_model(
            model, staging / "raw_model.safetensors", metadata={**embedded, "weight_kind": "raw"}
        )
        with use_ema_weights(expert, ema):
            safetensors.torch.save_model(
                model, staging / "model.safetensors", metadata={**embedded, "weight_kind": "ema"}
            )
        torch.save(
            {**state, "optimizer": optimizer.state_dict(), "ema": ema.state_dict(), "rng": capture_rng_state()},
            staging / "training_state.pt",
        )
        for relative, content in (assets or {}).items():
            target = (staging / relative).resolve()
            if not target.is_relative_to(staging.resolve()):
                raise ValueError(f"Asset escapes checkpoint: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        payload = {
            **metadata,
            "schema_version": SCHEMA_VERSION,
            "trainer": "potential_pi",
            "optimizer_updates": state["optimizer_updates"],
            "inference_weights": "ema",
            "model_sha256": sha256_file(staging / "model.safetensors"),
            "raw_model_sha256": sha256_file(staging / "raw_model.safetensors"),
        }
        write_json(staging / METADATA_FILENAME, payload)
        validate_checkpoint_metadata(
            staging,
            velocity_mode=payload["velocity_mode"],
            action_horizon=payload["action_horizon"],
            action_dim=payload["action_dim"],
        )
        os.rename(staging, directory)
    except BaseException:
        # staging is created by us under the exact checkpoint parent.
        if staging.exists() and staging.resolve().parent == directory.parent.resolve():
            shutil.rmtree(staging)
        raise
    return directory


def prune_checkpoints(root: Path, *, keep_last: int, preserve: tuple[int, ...] = ()) -> list[Path]:
    """Prune only numbered, owned training artifacts under this exact run root."""
    if keep_last < 1:
        raise ValueError("At least one recovery checkpoint must be retained")
    root = Path(root).resolve()
    owned = []
    for candidate in root.iterdir():
        if not candidate.name.isdigit() or not candidate.is_dir() or candidate.is_symlink():
            continue
        sidecar = candidate / METADATA_FILENAME
        if not sidecar.is_file():
            continue
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue  # An invalid or unrelated artifact is not demonstrably ours.
        if (
            isinstance(metadata, dict)
            and metadata.get("trainer") == "potential_pi"
            and metadata.get("optimizer_updates") == int(candidate.name)
        ):
            owned.append(candidate)
    owned.sort(key=lambda item: int(item.name))
    removed = []
    for candidate in owned[:-keep_last]:
        if int(candidate.name) in preserve:
            continue
        if candidate.resolve().parent != root:
            raise ValueError("Refusing to prune a checkpoint outside the run directory")
        shutil.rmtree(candidate)
        removed.append(candidate)
    return removed


def load_training_checkpoint(
    directory: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: AveragedModel,
    velocity_mode: str,
    action_horizon: int,
    action_dim: int,
) -> dict[str, Any]:
    """Resume a checkpoint produced locally by this trainer (contains pickle)."""
    directory = Path(directory)
    metadata = validate_checkpoint_metadata(
        directory,
        velocity_mode=velocity_mode,
        action_horizon=action_horizon,
        action_dim=action_dim,
    )
    if metadata is None or metadata.get("trainer") != "potential_pi":
        raise ValueError("Resumption requires a complete potential_pi training checkpoint")
    if sha256_file(directory / "raw_model.safetensors") != metadata["raw_model_sha256"]:
        raise ValueError("Raw model checkpoint hash mismatch")
    safetensors.torch.load_model(model, directory / "raw_model.safetensors", strict=True)
    state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=False)
    if state["optimizer_updates"] != metadata["optimizer_updates"]:
        raise ValueError("Metadata and training state update counts disagree")
    optimizer.load_state_dict(state["optimizer"])
    ema.load_state_dict(state["ema"], strict=True)
    restore_rng_state(state["rng"])
    return state


def cost_forecast(
    seconds_per_update: float, updates: int, hourly_rate: float | None, *, runs: int = 1
) -> dict[str, float | None]:
    if not math.isfinite(seconds_per_update) or seconds_per_update <= 0 or updates < 0 or runs < 1:
        raise ValueError("Invalid cost forecast inputs")
    if hourly_rate is not None and (not math.isfinite(hourly_rate) or hourly_rate < 0):
        raise ValueError("Invalid cost forecast inputs")
    hours = seconds_per_update * updates * runs / 3600
    return {
        "gpu_hours": hours,
        "gpu_cost_usd": hours * hourly_rate if hourly_rate is not None else None,
        "cost_with_20pct_headroom_usd": hours * hourly_rate * 1.2 if hourly_rate is not None else None,
    }
