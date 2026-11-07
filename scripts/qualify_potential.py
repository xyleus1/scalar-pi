"""Qualify full-size potential pi05 CUDA code paths using synthetic observations.

This is a correctness diagnostic, never a training-cost or LIBERO benchmark.
It does not download assets, update weights, or provision compute.
"""

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-kind", choices=("base", "research"), default="base")
    parser.add_argument("--config", default="pi05_libero_potential")
    parser.add_argument("--report", type=Path, default=Path("data/gpu_qualification.json"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=2, help="Diagnostic repetitions; not a benchmark standard")
    parser.add_argument("--repeats", type=int, default=10, help="Recorded diagnostic repetitions per solver count")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.warmup < 1 or args.repeats < 2:
        parser.error("batch-size and warmup must be positive; repeats must be at least two")
    if not (args.checkpoint / "model.safetensors").is_file():
        parser.error("--checkpoint must contain model.safetensors")
    if args.report.exists():
        parser.error("--report already exists; choose a new report path")
    return args


def _git_head():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _qualify(args, report):
    # Keep --help usable on machines without Torch, CUDA, JAX, or model assets.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Full-size qualification requires CUDA; CPU fixtures are separate tests.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This diagnostic supports one CUDA process only")
    from safetensors import safe_open
    import safetensors.torch

    from openpi.models import model as model_api
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.shared.potential_checkpoint import validate_checkpoint_metadata
    from openpi.training import config as configs

    config = configs.get_config(args.config)
    if not config.model.pi05 or not config.model.freeze_vlm or config.model.velocity_mode != "potential_dot":
        raise ValueError("Qualification requires the potential pi05 expert-only config")
    if config.model.dtype != "float32" or config.model.pytorch_compile_mode is not None:
        raise ValueError("Qualification requires FP32 construction and eager execution")
    source_metadata = validate_checkpoint_metadata(
        args.checkpoint,
        velocity_mode="direct" if args.checkpoint_kind == "base" else "potential_dot",
        action_horizon=config.model.action_horizon,
        action_dim=config.model.action_dim,
    )
    if args.checkpoint_kind == "base" and source_metadata and source_metadata.get("trainer") == "potential_pi":
        raise ValueError("Base qualification must start from converted pi05_base, not a research checkpoint")
    weight_file = args.checkpoint / "model.safetensors"
    if args.checkpoint_kind == "base":
        with safe_open(weight_file, framework="pt", device="cpu") as weights:
            non_fp32 = [name for name in weights.keys() if weights.get_slice(name).get_dtype() != "F32"]  # noqa: SIM118
        if non_fp32:
            raise ValueError(f"Convert the base checkpoint in FP32; other dtypes found: {non_fp32[:8]}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(mode=True)
    torch.backends.cudnn.benchmark = False
    model = PI0Pytorch(config.model)
    safetensors.torch.load_model(model, weight_file, strict=True)
    model.configure_expert_only(vlm_dtype=torch.bfloat16)
    model.to(device)
    torch.set_float32_matmul_precision("highest")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    shape = (args.batch_size, config.model.action_horizon, config.model.action_dim)
    token_shape = (args.batch_size, config.model.max_token_len)
    observation = model_api.Observation(
        images={key: torch.rand(args.batch_size, 3, 224, 224, device=device) * 2 - 1 for key in model_api.IMAGE_KEYS},
        image_masks={
            key: torch.full((args.batch_size,), key != "right_wrist_0_rgb", dtype=torch.bool, device=device)
            for key in model_api.IMAGE_KEYS
        },
        state=torch.zeros(args.batch_size, config.model.action_dim, device=device),
        tokenized_prompt=torch.ones(token_shape, dtype=torch.int64, device=device),
        tokenized_prompt_mask=torch.ones(token_shape, dtype=torch.bool, device=device),
    )
    actions = torch.randn(shape, device=device)
    noise = torch.randn(shape, device=device)
    time = torch.full((args.batch_size,), 0.5, device=device)
    report.update(
        {
            "measured_gpu": True,
            "hardware": torch.cuda.get_device_name(device),
            "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
            "cuda_runtime": torch.version.cuda,
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors")},
            "checkpoint_metadata": source_metadata,
            "checkpoint_bytes": weight_file.stat().st_size,
            "action_shape": list(shape),
            "image_layout": "NCHW",
            "image_shape": [args.batch_size, 3, 224, 224],
            "token_shape": list(token_shape),
            "precision": {
                "expert": "float32",
                "frozen_vlm": "bfloat16_with_native_fp32_exceptions",
                "matmul": "highest",
            },
            "checks": {},
        }
    )

    model.train()
    loss = model(observation, actions, noise=noise, time=time).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Full-size potential loss is non-finite")
    loss.backward()
    gradients = list(model.expert_modules().named_parameters())
    missing = [name for name, parameter in gradients if parameter.grad is None]
    nonfinite = [
        name for name, parameter in gradients if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if missing or nonfinite:
        raise RuntimeError(f"Expert higher-derivative gradients invalid: missing={missing}, nonfinite={nonfinite}")
    if any(
        parameter.grad is not None or parameter.requires_grad
        for parameter in model.paligemma_with_expert.paligemma.parameters()
    ):
        raise RuntimeError("Frozen VLM unexpectedly has gradients or trainable parameters")
    report["checks"]["full_expert_double_backward"] = {
        "passed": True,
        "loss": loss.item(),
        "parameter_tensors": len(gradients),
    }
    report["checks"]["frozen_vlm_has_no_gradients"] = True
    del loss, gradients
    model.zero_grad(set_to_none=True)
    model.eval()

    context = model.encode_context(observation)
    cache_before = [(key.clone(), value.clone()) for key, value in context.past_key_values]
    x = noise.clone().requires_grad_(requires_grad=True)
    first = model.predict_velocity(x, time, context, create_graph=False).detach()
    second = model.predict_velocity(x, time, context, create_graph=False).detach()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    if not torch.isfinite(first).all():
        raise RuntimeError("Repeated full-size velocity is non-finite")
    for (key, value), (old_key, old_value) in zip(context.past_key_values, cache_before, strict=True):
        torch.testing.assert_close(key, old_key, rtol=0, atol=0)
        torch.testing.assert_close(value, old_value, rtol=0, atol=0)
    report["checks"]["immutable_cache_and_repeatable_velocity"] = True
    del first, second, cache_before, x

    # Probe both controlled initializations on the same in-memory model; restore
    # the loaded head before sampling. No checkpoint is modified or saved.
    head_weight = model.action_out_proj.weight.detach().clone()
    head_bias = model.action_out_proj.bias.detach().clone()
    model.zero_action_output()
    zero_results = {}
    for mode in ("potential_dot", "direct"):
        model.velocity_mode = mode
        x = noise.clone().requires_grad_(requires_grad=True)
        velocity = model.predict_velocity(x, time, context, create_graph=True)
        if torch.count_nonzero(velocity):
            raise RuntimeError(f"Zero-output {mode} control did not produce zero velocity")
        (velocity - 1).square().mean().backward()
        head_grad = model.action_out_proj.weight.grad
        if head_grad is None or not torch.isfinite(head_grad).all() or not torch.count_nonzero(head_grad):
            raise RuntimeError(f"Zero-output {mode} control cannot begin learning")
        zero_results[mode] = {"zero_velocity": True, "nonzero_finite_head_gradient": True}
        model.zero_grad(set_to_none=True)
        del x, velocity, head_grad
    with torch.no_grad():
        model.action_out_proj.weight.copy_(head_weight)
        model.action_out_proj.bias.copy_(head_bias)
    model.velocity_mode = "potential_dot"
    report["checks"]["zero_output_controls"] = zero_results
    del head_weight, head_bias, context

    memory = {}
    for steps in (1, 10):
        for _ in range(args.warmup):
            output = model.sample_actions(device, observation, noise=noise, num_steps=steps)
            del output
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        series = []
        reference = None
        for index in range(args.repeats):
            output = model.sample_actions(device, observation, noise=noise, num_steps=steps)
            if output.requires_grad or not torch.isfinite(output).all():
                raise RuntimeError("Inference output retains a graph or contains non-finite values")
            cpu_output = output.detach().cpu()
            if reference is None:
                reference = cpu_output.clone()
            else:
                torch.testing.assert_close(cpu_output, reference, rtol=0, atol=0)
            del output, cpu_output
            gc.collect()
            torch.cuda.synchronize(device)
            series.append(
                {
                    "repeat": index + 1,
                    "allocated_bytes": torch.cuda.memory_allocated(device),
                    "reserved_bytes": torch.cuda.memory_reserved(device),
                }
            )
        memory[str(steps)] = {
            "series": series,
            "allocated_last_minus_first_bytes": series[-1]["allocated_bytes"] - series[0]["allocated_bytes"],
            "allocated_range_bytes": max(row["allocated_bytes"] for row in series)
            - min(row["allocated_bytes"] for row in series),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "memory_stability_pass": None,
            "interpretation": "Inspect the series; no arbitrary leak threshold or long-run guarantee is assigned.",
        }
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Inference unexpectedly populated parameter gradients")
    report["checks"]["k1_k10_repeatable_finite_detached_inference"] = True
    report["inference_memory"] = memory
    report["correctness_checks_passed"] = True


def main(argv=None):
    args = parse_args(argv)
    report = {
        "schema_version": 1,
        "purpose": "full_size_cuda_synthetic_correctness",
        "synthetic_input": True,
        "valid_training_cost_profile": False,
        "valid_benchmark_evaluation": False,
        "measured_gpu": False,
        "correctness_checks_passed": False,
        "git_head": _git_head(),
        "config": args.config,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_kind": args.checkpoint_kind,
        "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "warmup_repetitions": args.warmup,
        "recorded_repetitions": args.repeats,
        "limitations": [
            "Synthetic inputs are not LIBERO observations or tokenized tasks.",
            "No optimizer update, dataset iteration, rollout, or training throughput is measured.",
            "Short inference memory series do not establish long-run stability.",
        ],
    }
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    try:
        _qualify(args, report)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote synthetic correctness report: {args.report}")
    return 0 if report["correctness_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
