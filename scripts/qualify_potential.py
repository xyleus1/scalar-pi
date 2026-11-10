"""Qualify potential pi05 CUDA derivatives and the actual policy serving loader.

This is a correctness diagnostic, never a training-cost or LIBERO benchmark.
Synthetic derivative probes temporarily change and restore the in-memory output
head. No checkpoint is modified, training run started, or resource provisioned.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time as walltime
import traceback


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-kind", choices=("base", "research"), default="base")
    parser.add_argument("--config", default="pi05_libero_potential")
    parser.add_argument(
        "--math-only", action="store_true", help="Early synthetic derivative gate; no data/tokenizer needed"
    )
    parser.add_argument("--serving-only", action="store_true", help="Reload and serve a produced research checkpoint")
    parser.add_argument(
        "--parity-report", type=Path, help="Passing base JAX/Torch conversion report for full qualification"
    )
    parser.add_argument("--corpus", type=Path, help="Audited real-observation NPZ; required for CUDA qualification")
    parser.add_argument("--norm-stats", type=Path, help="Staged LIBERO norms for base-checkpoint serving qualification")
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
    if args.serving_only and (args.math_only or args.checkpoint_kind != "research" or args.corpus is None):
        parser.error("--serving-only requires a research checkpoint and --corpus, without --math-only")
    if not (args.checkpoint / "model.safetensors").is_file():
        parser.error("--checkpoint must contain model.safetensors")
    if args.report.exists():
        parser.error("--report already exists; choose a new report path")
    return args


def _parameter_digest(module):
    import torch

    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode())
        raw = parameter.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy()
        digest.update(raw.tobytes())
    return digest.hexdigest()


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
    if not args.math_only and (args.corpus is None or not args.corpus.is_file()):
        raise ValueError("--corpus must identify the audited observation NPZ for actual-policy checks")
    if (
        not args.math_only
        and args.checkpoint_kind == "base"
        and (args.norm_stats is None or not args.norm_stats.is_file())
    ):
        raise ValueError("Base qualification requires the staged LIBERO --norm-stats asset")
    if args.checkpoint_kind == "research" and args.norm_stats is not None:
        raise ValueError("Research serving must use normalization assets inside its checkpoint")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This diagnostic supports one CUDA process only")
    from safetensors import safe_open
    import safetensors.torch

    from openpi.models import model as model_api
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.research.evaluation import file_digest
    from openpi.research.policy_checks import require_staged_tokenizer
    from openpi.research.qualification import execution_identity
    from openpi.shared.potential_checkpoint import validate_checkpoint_metadata
    from openpi.training import config as configs
    from openpi.training.potential import validate_base_conversion

    if not args.math_only:
        require_staged_tokenizer()

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
    report["identity"] = (
        execution_identity(checkpoint=args.checkpoint)
        if args.checkpoint_kind == "base"
        else execution_identity(
            base_hash=source_metadata["base_checkpoint_hash"],
            conversion_hash=source_metadata["base_conversion_receipt_sha256"],
        )
    )
    if not args.math_only and args.checkpoint_kind == "base":
        if args.parity_report is None:
            raise ValueError("Full base qualification requires --parity-report")
        parity = json.loads(args.parity_report.read_text())
        if (
            parity.get("passed") is not True
            or parity.get("purpose") != "direct_model_conversion_and_cache_parity"
            or parity.get("torch", {}).get("checkpoint_sha256") != report["identity"]["base_checkpoint_sha256"]
            or parity.get("torch", {}).get("corpus_sha256") != file_digest(args.corpus)
            or parity.get("torch", {}).get("normalization_sha256") != file_digest(args.norm_stats)
        ):
            raise ValueError("Conversion parity receipt differs from current base/corpus/normalization")
        report["conversion_parity"] = {"passed": True, "report_sha256": file_digest(args.parity_report)}
    if args.checkpoint_kind == "base":
        validate_base_conversion(args.checkpoint, require_receipt=True)
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

    vlm_digest = _parameter_digest(model.paligemma_with_expert.paligemma)
    model.train()
    torch.cuda.synchronize(device)
    backward_start = walltime.perf_counter()
    loss = model(observation, actions, noise=noise, time=time).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Full-size potential loss is non-finite")
    loss.backward()
    torch.cuda.synchronize(device)
    report["probe_backward_seconds"] = walltime.perf_counter() - backward_start
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
        model.zero_action_output()
        model.velocity_mode = mode
        x = noise.clone().requires_grad_(requires_grad=True)
        velocity = model.predict_velocity(x, time, context, create_graph=True)
        if torch.count_nonzero(velocity):
            raise RuntimeError(f"Zero-output {mode} control did not produce zero velocity")
        (velocity - 1).square().mean().backward()
        head_grad = model.action_out_proj.weight.grad
        if head_grad is None or not torch.isfinite(head_grad).all() or not torch.count_nonzero(head_grad):
            raise RuntimeError(f"Zero-output {mode} control cannot begin learning")
        # One temporary head update tests that the zero initialization does not
        # permanently disconnect the pretrained trunk. Restore below; no artifact
        # or optimizer state from this synthetic probe is used for training.
        with torch.no_grad():
            model.action_out_proj.weight.add_(head_grad, alpha=-1e-3)
        model.zero_grad(set_to_none=True)
        del x, velocity, head_grad
        x = noise.clone().requires_grad_(requires_grad=True)
        velocity = model.predict_velocity(x, time, context, create_graph=True)
        (velocity - 1).square().mean().backward()
        trunk = [
            parameter.grad
            for name, parameter in model.expert_modules().named_parameters()
            if not name.startswith("action_out_proj.")
        ]
        if not trunk or any(gradient is None or not torch.isfinite(gradient).all() for gradient in trunk):
            raise RuntimeError(f"{mode} trunk gradients are missing or nonfinite after a head update")
        if not any(torch.count_nonzero(gradient) for gradient in trunk):
            raise RuntimeError(f"{mode} trunk did not begin learning after a head update")
        zero_results[mode] = {
            "zero_velocity": True,
            "nonzero_finite_head_gradient": True,
            "nonzero_finite_trunk_gradients_after_head_update": True,
        }
        model.zero_grad(set_to_none=True)
        del x, velocity, trunk
    with torch.no_grad():
        model.action_out_proj.weight.copy_(head_weight)
        model.action_out_proj.bias.copy_(head_bias)
    model.velocity_mode = "potential_dot"
    report["checks"]["zero_output_controls"] = zero_results
    if _parameter_digest(model.paligemma_with_expert.paligemma) != vlm_digest:
        raise RuntimeError("Frozen VLM weights changed during derivative qualification")
    report["checks"]["frozen_vlm_weights_unchanged"] = True
    del head_weight, head_bias, context
    if args.math_only:
        return

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


def _qualify_policy(args, report):
    import numpy as np
    import torch

    from openpi.policies.policy_config import create_trained_policy
    from openpi.research.policy_checks import corpus_observation
    from openpi.research.policy_checks import file_digest
    from openpi.research.policy_checks import load_corpus
    from openpi.shared import normalize
    from openpi.training.config import get_config

    gc.collect()
    torch.cuda.empty_cache()
    corpus = load_corpus(args.corpus)
    config = get_config("pi05_libero_b0" if args.checkpoint_kind == "base" else args.config)
    norms = normalize.deserialize_json(args.norm_stats.read_text()) if args.norm_stats is not None else None
    serving = {}
    for steps in (1, 10):
        policy = create_trained_policy(
            config,
            args.checkpoint,
            norm_stats=norms,
            sample_kwargs={"num_steps": steps},
            pytorch_device="cuda:0",
        )
        if policy.metadata["matmul_precision"] != "highest" or torch.get_float32_matmul_precision() != "highest":
            raise RuntimeError("Actual serving loader disagrees with qualified arithmetic")
        first = None
        for index in range(len(corpus["states"])):
            output = policy.infer(corpus_observation(corpus, index), noise=corpus["noise"][index])
            if output.get("policy_error") or not np.isfinite(output["actions"]).all():
                raise RuntimeError("Actual serving produced invalid actions on the audited corpus")
            if index == 0:
                first = output["actions"].copy()
        repeated = policy.infer(corpus_observation(corpus, 0), noise=corpus["noise"][0])
        np.testing.assert_array_equal(first, repeated["actions"])
        if any(parameter.requires_grad or parameter.grad is not None for parameter in policy._model.parameters()):  # noqa: SLF001
            raise RuntimeError("Serving action derivatives unexpectedly enabled parameter gradients")
        serving[str(steps)] = {"passed": True, "observations": len(corpus["states"]), "policy": policy.metadata}
        del policy
        gc.collect()
        torch.cuda.empty_cache()
    report["actual_policy_serving"] = {
        "passed": True,
        "corpus_sha256": file_digest(args.corpus),
        "num_steps": serving,
        "scope": "direct base bridge" if args.checkpoint_kind == "base" else "trained potential checkpoint",
    }
    report["corpus_sha256"] = file_digest(args.corpus)
    report["normalization_sha256"] = serving["1"]["policy"]["normalization_sha256"]


def main(argv=None):
    args = parse_args(argv)
    report = {
        "schema_version": 1,
        "purpose": "full_size_cuda_derivatives_and_serving",
        "synthetic_input": True,
        "valid_training_cost_profile": False,
        "valid_benchmark_evaluation": False,
        "measured_gpu": False,
        "correctness_checks_passed": False,
        "math_only": args.math_only,
        "serving_only": args.serving_only,
        "git_head": _git_head(),
        "config": args.config,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_kind": args.checkpoint_kind,
        "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "warmup_repetitions": args.warmup,
        "recorded_repetitions": args.repeats,
        "limitations": [
            "Derivative probes use synthetic inputs; serving checks use the audited real-observation corpus.",
            "No training run, rollout, or training throughput is measured.",
            "Base serving qualification checks the direct loader; repeat on a produced potential checkpoint before evaluation.",
            "Short inference memory series do not establish long-run stability.",
        ],
    }
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    try:
        if not args.serving_only:
            _qualify(args, report)
        if not args.math_only:
            _qualify_policy(args, report)
        if args.serving_only:
            report["measured_gpu"] = True
        report["correctness_checks_passed"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote derivative and serving qualification: {args.report}")
    return 0 if report["correctness_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
