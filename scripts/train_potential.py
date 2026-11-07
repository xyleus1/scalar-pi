"""Single-GPU, expert-only pi05 training with reproducible accumulation and EMA.

Example (after conversion and compute_norm_stats):
  uv run scripts/train_potential.py pi05_libero_potential --exp-name p_seed42 \
      --weights /data/pi05_base_torch --microbatch-size 4 \
      --data-manifest /data/libero_manifest.json

Run --profile-updates 100 first in a separate experiment directory. The script
does not provision hardware, download checkpoints, or claim a measured budget
until it has timed actual complete optimizer updates.
"""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

# The official loader uses JAX tree utilities; its GPU runtime is unnecessary.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import safetensors.torch
import torch

from openpi.shared.potential_checkpoint import validate_checkpoint_metadata
from openpi.training.potential import ResumableBatchSampler
from openpi.training.potential import cost_forecast
from openpi.training.potential import learning_rate
from openpi.training.potential import load_training_checkpoint
from openpi.training.potential import make_ema
from openpi.training.potential import prune_checkpoints
from openpi.training.potential import save_checkpoint
from openpi.training.potential import seed_all
from openpi.training.potential import sha256_file
from openpi.training.potential import stable_seed
from openpi.training.potential import torch_rng_scope
from openpi.training.potential import validate_base_conversion
from openpi.training.potential import validate_data_manifest
from openpi.training.potential import write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", choices=("pi05_libero_b0", "pi05_libero_b1", "pi05_libero_potential"))
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--weights", type=Path, help="Converted direct pi05_base directory, required for a fresh run")
    parser.add_argument("--resume", type=Path, help="Trusted complete checkpoint produced by this trainer")
    parser.add_argument("--output-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--updates", type=int, help="Explicit non-reference override; default is configured 30,000")
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--keep-last-checkpoints", type=int, default=2)
    parser.add_argument("--profile-updates", type=int, default=0)
    parser.add_argument("--profile-warmup", type=int, default=10)
    parser.add_argument(
        "--hourly-rate", type=float, help="Actual GPU price; required for profiling, otherwise costs remain unknown"
    )
    parser.add_argument("--max-hours", type=float, help="Stop and save at an optimizer boundary after this run time")
    parser.add_argument("--data-manifest", type=Path, help="Immutable dataset manifest; required outside profiling")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--matmul-precision", choices=("highest", "high"), default="highest")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if Path(args.exp_name).name != args.exp_name or args.exp_name in (".", ".."):
        parser.error("--exp-name must be a single directory name")
    if args.microbatch_size < 1 or args.num_workers < 0 or args.save_every < 1 or args.keep_last_checkpoints < 1:
        parser.error("microbatch/save interval must be positive and workers nonnegative")
    if args.profile_updates < 0 or args.profile_warmup < 0:
        parser.error("Profile counts must be nonnegative")
    if args.hourly_rate is not None and (not math.isfinite(args.hourly_rate) or args.hourly_rate < 0):
        parser.error("Hourly rate must be finite and nonnegative")
    if args.profile_updates and args.hourly_rate is None:
        parser.error("Profiling requires --hourly-rate with your actual GPU price")
    if args.updates is not None and args.updates < 1:
        parser.error("--updates must be positive")
    if args.max_hours is not None and args.max_hours <= 0:
        parser.error("--max-hours must be positive")
    if args.resume is not None and (args.weights is not None or args.profile_updates):
        parser.error("--resume cannot be combined with --weights or profiling")
    if not args.profile_updates and args.data_manifest is None:
        parser.error("Non-profile training requires --data-manifest from prepare_potential_data.py")
    return args


def _git_state():
    repository = Path(__file__).resolve().parents[1]

    def run(*arguments):
        result = subprocess.run(
            ["git", *arguments],
            capture_output=True,
            text=True,
            check=False,
            cwd=repository,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    paths = run("ls-files", "--cached", "--others", "--exclude-standard", "-z") or ""
    # A commit plus "dirty" is insufficient for reproducing uncommitted research.
    # Include exact source bytes, including newly added files, without hashing
    # data/checkpoints or relying on a later commit to identify this execution.
    hashes = {
        name: sha256_file(repository / name)
        for name in sorted(set(paths.split("\0")))
        if name and (name.endswith(".py") or name in {"pyproject.toml", "uv.lock"}) and (repository / name).is_file()
    }
    return {
        "commit": run("rev-parse", "HEAD"),
        "status": run("status", "--porcelain"),
        "source_files_sha256": hashes,
        "source_snapshot_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
    }


def _packages():
    versions = {}
    for name in ("torch", "transformers", "safetensors", "jax", "lerobot", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def _append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, default=str) + "\n")


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("Full pi05 training requires CUDA. Run the lightweight helper tests on CPU.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This higher-derivative reference trainer supports exactly one GPU process")

    # Imports intentionally remain after CLI parsing and capability checks.
    import jax

    from openpi.models import model as model_api
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.shared import normalize
    from openpi.training import config as configs
    from openpi.training import data_loader

    config = dataclasses.replace(configs.get_config(args.config), seed=args.seed, exp_name=args.exp_name)
    if config.batch_size % args.microbatch_size:
        raise ValueError("Microbatch must divide the configured effective batch size")
    if not config.model.freeze_vlm or config.model.dtype != "float32":
        raise ValueError("Research configs must freeze the VLM and construct FP32 weights")
    if config.ema_decay != 0.999:
        raise ValueError("The reference protocol requires EMA decay 0.999")
    updates = args.updates or config.num_train_steps
    accumulation = config.batch_size // args.microbatch_size
    output = args.output_root / args.config / args.exp_name
    if output.exists() and args.resume is None:
        raise FileExistsError(f"Use a new experiment name or --resume; refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(args.deterministic)
    torch.backends.cudnn.benchmark = False

    model = PI0Pytorch(config.model)  # FP32 construction before any source weights are loaded.
    source = args.resume or args.weights or Path(config.pytorch_weight_path or "")
    weight_file = source / ("raw_model.safetensors" if args.resume else "model.safetensors")
    if not weight_file.is_file():
        raise FileNotFoundError(f"Converted weights not found: {weight_file}")
    base_conversion = None
    if args.resume:
        source_metadata = validate_checkpoint_metadata(
            source,
            velocity_mode=config.model.velocity_mode,
            action_horizon=config.model.action_horizon,
            action_dim=config.model.action_dim,
        )
    else:
        source_metadata = validate_checkpoint_metadata(
            source,
            velocity_mode="direct",
            action_horizon=config.model.action_horizon,
            action_dim=config.model.action_dim,
        )
        if source_metadata is not None and source_metadata.get("trainer") == "potential_pi":
            raise ValueError(
                "Fresh research training must start from converted pi05_base, not an adapted research checkpoint"
            )
        base_conversion = validate_base_conversion(source, require_receipt=not bool(args.profile_updates))
    safetensors.torch.load_model(model, weight_file, strict=True)
    model.configure_expert_only(vlm_dtype=torch.bfloat16)
    if args.resume is None and config.model.initialization == "zero_output":
        model.zero_action_output()
    model.to(device)
    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    torch.set_float32_matmul_precision(args.matmul_precision)  # Override upstream constructor setting.
    expert = model.expert_modules()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if {id(p) for p in trainable} != {id(p) for p in expert.parameters() if p.requires_grad}:
        raise RuntimeError("Trainable parameter allowlist disagrees with expert_modules")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    ema = make_ema(expert, config.ema_decay)
    schedule = dataclasses.asdict(config.lr_schedule)
    signature = {
        "config": args.config,
        "seed": args.seed,
        "microbatch_size": args.microbatch_size,
        "effective_batch_size": config.batch_size,
        "schedule": schedule,
        "ema_decay": config.ema_decay,
        "matmul_precision": args.matmul_precision,
        "deterministic": args.deterministic,
        "initialization": config.model.initialization,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    completed = 0
    consumed = 0
    cumulative_seconds = 0.0
    if args.resume:
        state = load_training_checkpoint(
            source,
            model=model,
            optimizer=optimizer,
            ema=ema,
            velocity_mode=config.model.velocity_mode,
            action_horizon=config.model.action_horizon,
            action_dim=config.model.action_dim,
        )
        if state["resume_signature"] != signature:
            raise ValueError("Resume protocol differs (seed, batching, schedule, initialization, or arithmetic)")
        completed, consumed = state["optimizer_updates"], state["consumed_microbatches"]
        cumulative_seconds = state["training_seconds"]
        if consumed != completed * accumulation:
            raise ValueError("Checkpoint is not at a complete effective-batch boundary")
    if completed >= updates:
        raise ValueError(f"Checkpoint already has {completed} updates; requested target is {updates}")

    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    source_dataset = dataset
    while hasattr(source_dataset, "_dataset"):
        source_dataset = source_dataset._dataset  # noqa: SLF001
    dataset_manifest = None
    if args.data_manifest:
        if not hasattr(source_dataset, "root"):
            raise ValueError("Cannot verify loaded dataset root against manifest")
        dataset_manifest = validate_data_manifest(
            args.data_manifest,
            repo_id=data_config.repo_id,
            dataset_root=source_dataset.root,
        )
        manifest_hash = sha256_file(args.data_manifest)
        if args.resume and source_metadata.get("data_manifest_sha256") != manifest_hash:
            raise ValueError("Dataset manifest changed since the checkpoint")
    dataset = data_loader.transform_dataset(dataset, data_config)
    sampler = ResumableBatchSampler(
        len(dataset),
        args.microbatch_size,
        config.batch_size,
        args.seed,
        start_microbatch=consumed,
    )
    loader_kwargs = {"multiprocessing_context": "spawn", "persistent_workers": True} if args.num_workers else {}
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=data_loader._collate_fn,  # noqa: SLF001
        worker_init_fn=data_loader._worker_init_fn,  # noqa: SLF001
        generator=torch.Generator().manual_seed(stable_seed(args.seed, "loader")),
        **loader_kwargs,
    )
    iterator = iter(loader)
    assets = {}
    if data_config.norm_stats is not None and data_config.asset_id is not None:
        norm_json = normalize.serialize_json(data_config.norm_stats)
        assets[f"assets/{data_config.asset_id}/norm_stats.json"] = norm_json
    else:
        raise ValueError("LIBERO research training requires explicit normalization assets")
    norm_hash = hashlib.sha256(norm_json.encode()).hexdigest()
    if args.resume and source_metadata.get("normalization_hash") != norm_hash:
        raise ValueError("Normalization assets changed since the checkpoint")
    manifest = {
        "arguments": vars(args),
        "configuration": dataclasses.asdict(config),
        "protocol": signature,
        "code": _git_state(),
        "packages": _packages(),
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(device),
        "cuda_runtime": torch.version.cuda,
        "trainable_parameters": {name: list(p.shape) for name, p in model.named_parameters() if p.requires_grad},
        "trainable_parameter_count": sum(p.numel() for p in trainable),
        "dataset_repo": data_config.repo_id,
        "dataset_windows": len(dataset),
        "dataset_revision": dataset_manifest["revision"] if dataset_manifest else None,
        "data_manifest_sha256": sha256_file(args.data_manifest) if args.data_manifest else None,
        "normalization_hash": norm_hash,
        "source_weights": str(weight_file),
        "source_weights_sha256": sha256_file(weight_file),
        "base_conversion": base_conversion,
        "base_conversion_receipt_sha256": sha256_file(source / "config.json") if base_conversion else None,
        "reference_exposure": config.batch_size * config.num_train_steps,
        "requested_exposure": config.batch_size * updates,
        "mode": "profile" if args.profile_updates else "training",
    }
    if not args.data_manifest:
        logging.warning("No immutable dataset manifest supplied; dataset revision is not independently pinned")
    manifest_path = output / (f"resume_{completed}_manifest.json" if args.resume else "run_manifest.json")
    write_json(manifest_path, manifest)
    metadata = {
        "velocity_mode": config.model.velocity_mode,
        "initialization": config.model.initialization,
        "freeze_vlm": True,
        "action_horizon": config.model.action_horizon,
        "action_dim": config.model.action_dim,
        "expert_precision": "float32",
        "vlm_precision": "bfloat16",
        "normalization_hash": norm_hash,
        "training_seed": args.seed,
        "code_commit": manifest["code"]["commit"],
        "source_snapshot_sha256": manifest["code"]["source_snapshot_sha256"],
        "base_checkpoint_hash": manifest["source_weights_sha256"],
        "base_conversion_receipt_sha256": manifest["base_conversion_receipt_sha256"],
        "data_manifest_sha256": manifest["data_manifest_sha256"],
        "resume_signature": signature,
    }
    if args.resume:
        metadata["base_checkpoint_hash"] = source_metadata["base_checkpoint_hash"]
        metadata["base_conversion_receipt_sha256"] = source_metadata.get("base_conversion_receipt_sha256")
    target = args.profile_warmup + args.profile_updates if args.profile_updates else updates
    timings = []
    start_wall = time.perf_counter()
    last_saved = completed
    torch.cuda.reset_peak_memory_stats(device)

    def persist():
        nonlocal last_saved
        if completed == last_saved:
            return
        checkpoint = save_checkpoint(
            output / str(completed),
            model=model,
            expert=expert,
            optimizer=optimizer,
            ema=ema,
            state={
                "optimizer_updates": completed,
                "consumed_microbatches": consumed,
                "training_seconds": cumulative_seconds,
                "resume_signature": signature,
            },
            metadata=metadata,
            assets=assets,
        )
        last_saved = completed
        logging.info("Checkpoint committed: %s", checkpoint)
        removed = prune_checkpoints(
            output,
            keep_last=args.keep_last_checkpoints,
            preserve=(config.num_train_steps,),
        )
        if removed:
            logging.info("Pruned older recovery checkpoints: %s", removed)

    while completed < target:
        torch.cuda.synchronize(device)
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        rate = learning_rate(completed, **schedule)
        for group in optimizer.param_groups:
            group["lr"] = rate
        loss_total = 0.0
        physical_total = 0.0
        for micro in range(accumulation):
            batch = next(iterator)
            batch = jax.tree.map(lambda value: torch.as_tensor(value, device=device), batch)
            observation = model_api.Observation.from_dict(batch)
            actions = batch["actions"].float()
            with torch_rng_scope(stable_seed(args.seed, "corruption", completed, micro), device):
                noise = model.sample_noise(actions.shape, device)
                flow_time = model.sample_time(actions.shape[0], device)
            with torch_rng_scope(stable_seed(args.seed, "augmentation", completed, micro), device):
                losses = model(observation, actions, noise=noise, time=flow_time)
                loss = losses.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at update {completed}, microbatch {micro}")
                (loss / accumulation).backward()
            loss_total += loss.detach().item() / accumulation
            physical_total += losses.detach()[..., :7].mean().item() / accumulation
        norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            config.optimizer.clip_gradient_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        ema.update_parameters(expert)
        completed += 1
        consumed += accumulation
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - tick
        cumulative_seconds += elapsed
        if not args.profile_updates or completed > args.profile_warmup:
            timings.append(elapsed)
        record = {
            "optimizer_updates": completed,
            "consumed_microbatches": consumed,
            "loss": loss_total,
            "physical_loss": physical_total,
            "learning_rate": rate,
            "gradient_norm": float(norm),
            "seconds_per_update": elapsed,
            "training_seconds": cumulative_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        _append(output / "metrics.jsonl", record)
        if completed % config.log_interval == 0 or completed == 1:
            logging.info("update=%d loss=%.5f lr=%.3g seconds=%.3f", completed, loss_total, rate, elapsed)
        if not args.profile_updates and completed % args.save_every == 0:
            persist()
        if args.max_hours is not None and time.perf_counter() - start_wall >= args.max_hours * 3600:
            logging.warning("Run time limit reached; saving a partial run at an optimizer boundary")
            break
    if not args.profile_updates:
        persist()
    if timings:
        mean_seconds = statistics.mean(timings)
        report = {
            "measured_gpu": True,
            "model_kind": "pi05_full",
            "hourly_rate_usd": args.hourly_rate,
            "hardware": torch.cuda.get_device_name(device),
            "measurement_source": str(output / "metrics.jsonl"),
            "effective_batch_size": config.batch_size,
            "warmup_optimizer_updates": args.profile_warmup if args.profile_updates else 0,
            "timed_optimizer_updates": len(timings),
            "seconds_per_optimizer_update": {args.config: mean_seconds},
            "mode": manifest["mode"],
            "completed_updates": completed,
            "requested_updates": updates,
            "completed_reference_run": not args.profile_updates and completed == config.num_train_steps,
            "warmup_updates": args.profile_warmup if args.profile_updates else 0,
            "timed_updates": len(timings),
            "mean_seconds_per_update": mean_seconds,
            "median_seconds_per_update": statistics.median(timings),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "forecast_one_reference_run": cost_forecast(mean_seconds, config.num_train_steps, args.hourly_rate),
            "forecast_three_same_method_seeds": cost_forecast(
                mean_seconds, config.num_train_steps, args.hourly_rate, runs=3
            ),
            "forecast_excludes": ["setup", "checkpoint I/O", "evaluation", "storage", "other architectures"],
        }
        write_json(output / "profile.json", report)
        logging.info("Measured forecast: %s", json.dumps(report))


if __name__ == "__main__":
    main()
