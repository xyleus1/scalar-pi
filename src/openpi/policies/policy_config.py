import hashlib
import importlib.metadata
import json
import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
from openpi.shared import normalize
import openpi.shared.download as download
from openpi.shared.potential_checkpoint import validate_checkpoint_metadata
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def checkpoint_digest(checkpoint_dir: pathlib.Path) -> str:
    """Hash actual weights once at policy load, independently of caller labels."""
    weight = checkpoint_dir / "model.safetensors"
    files = [weight] if weight.is_file() else sorted((checkpoint_dir / "params").rglob("*"))
    digests = {}
    for path in files:
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            digests[path.relative_to(checkpoint_dir).as_posix()] = digest.hexdigest()
    if not digests:
        raise ValueError("Checkpoint contains no weights to identify")
    if weight.is_file():
        return digests["model.safetensors"]
    return hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()


def _runtime_metadata(*, is_pytorch: bool, device: str | None) -> dict:
    if is_pytorch:
        import torch

        result = {
            "backend": "pytorch",
            "torch": torch.__version__,
            "device": device,
            "container_image": os.environ.get("SCALAR_PI_MODEL_IMAGE"),
        }
        if torch.device(device).type == "cuda":
            result.update(
                gpu=torch.cuda.get_device_name(device),
                cuda_runtime=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),
            )
        return result
    import jax

    return {
        "backend": "jax",
        "jax": importlib.metadata.version("jax"),
        "devices": [
            {"device": str(device), "kind": device.device_kind, "platform": device.platform} for device in jax.devices()
        ],
        "container_image": os.environ.get("SCALAR_PI_MODEL_IMAGE"),
    }


def _matmul_precision(*, is_pytorch: bool) -> str:
    if is_pytorch:
        import torch

        return torch.get_float32_matmul_precision()
    import jax

    return jax.config.jax_default_matmul_precision or "default"


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    if getattr(train_config.model, "freeze_vlm", False):
        from openpi.research.policy_checks import require_staged_tokenizer

        require_staged_tokenizer()
    checkpoint_dir = pathlib.Path(download.maybe_download(str(checkpoint_dir)))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    research_metadata = validate_checkpoint_metadata(
        checkpoint_dir,
        velocity_mode=getattr(train_config.model, "velocity_mode", "direct"),
        action_horizon=train_config.model.action_horizon,
        action_dim=train_config.model.action_dim,
    )
    checkpoint_sha256 = checkpoint_digest(checkpoint_dir)
    if (
        research_metadata
        and research_metadata.get("trainer") == "potential_pi"
        and checkpoint_sha256 != research_metadata.get("model_sha256")
    ):
        raise ValueError("Research inference weights differ from the checkpoint's EMA receipt")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        if getattr(train_config.model, "freeze_vlm", False):
            model.configure_expert_only()
            # The constructor sets "high". Research serving must match the
            # "highest" arithmetic qualified and used by the matched trainer.
            import torch

            torch.set_float32_matmul_precision("highest")
        else:
            model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    normalization_sha256 = hashlib.sha256(normalize.serialize_json(norm_stats).encode()).hexdigest()
    if (
        research_metadata
        and research_metadata.get("trainer") == "potential_pi"
        and normalization_sha256 != research_metadata.get("normalization_hash")
    ):
        raise ValueError("Research normalization differs from the checkpoint's training receipt")

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata={
            **(train_config.policy_metadata or {}),
            "config_name": train_config.name,
            "checkpoint_id": str(checkpoint_dir),
            "checkpoint_sha256": checkpoint_sha256,
            "normalization_sha256": normalization_sha256,
            "matmul_precision": _matmul_precision(is_pytorch=is_pytorch),
            "runtime": _runtime_metadata(is_pytorch=is_pytorch, device=pytorch_device),
            "num_steps": (
                None
                if train_config.model.model_type == _model.ModelType.PI0_FAST
                else (sample_kwargs or {}).get("num_steps", 10)
            ),
            "velocity_mode": getattr(train_config.model, "velocity_mode", "direct"),
            "research": research_metadata,
        },
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
