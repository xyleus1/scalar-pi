# ruff: noqa: PLC0415 -- Keep CLI and receipt inspection usable without the GPU stack.
"""Qualify mixed-precision cached execution against an independent uncached oracle.

The original FP32 JAX/Torch conversion bridge is a separate prerequisite. Here,
both original converted checkpoints first qualify the oracle in FP32 against
native joint execution. Only then are hybrid cache/oracle comparisons run.
Cross-precision drift is descriptive and cannot pass or fail the matched gate.
"""

import argparse
import dataclasses
import gc
import json
import os
from pathlib import Path
import traceback

import numpy as np

from openpi.research.hybrid_qualification import ATOL
from openpi.research.hybrid_qualification import PURPOSE
from openpi.research.hybrid_qualification import RTOL
from openpi.research.hybrid_qualification import validate_hybrid_report
from openpi.research.policy_checks import array_error
from openpi.research.policy_checks import corpus_observation
from openpi.research.policy_checks import file_digest
from openpi.research.policy_checks import input_digest
from openpi.research.policy_checks import load_corpus
from openpi.research.policy_checks import require_staged_tokenizer
from openpi.research.uncached_reference import reference_inputs
from openpi.research.uncached_reference import torch_joint_velocity
from openpi.research.uncached_reference import trajectory
from openpi.research.uncached_reference import uncached_reference
from openpi.shared.potential_precision import FP32_BRIDGE_PRECISION
from openpi.shared.potential_precision import RESEARCH_PRECISION


def transformed_inputs(corpus, norm_path):
    from openpi import transforms
    from openpi.shared import normalize
    from openpi.training.config import get_config

    config = get_config("pi05_libero")
    model_config = dataclasses.replace(config.model, dtype="float32", pytorch_compile_mode=None)
    data = config.data.create(config.assets_dirs, model_config)
    norms = normalize.deserialize_json(norm_path.read_text(encoding="utf-8"))
    transform = transforms.compose(
        [
            transforms.InjectDefaultPrompt(None),
            *data.data_transforms.inputs,
            transforms.Normalize(norms, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ]
    )
    return [transform(corpus_observation(corpus, index)) for index in range(40)]


def check_stage(checkpoint, transformed, corpus, mode):
    import jax
    from safetensors import safe_open
    import safetensors.torch
    import torch

    from openpi.models import model as model_api
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.training.config import get_config

    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as weights:
        if any(weights.get_slice(name).get_dtype() != "F32" for name in weights.keys()):  # noqa: SIM118
            raise ValueError("Oracle qualification starts from original converted FP32 weights")
    model = PI0Pytorch(get_config("pi05_libero_b0").model)
    safetensors.torch.load_model(model, checkpoint / "model.safetensors", strict=True)
    model.configure_expert_only(vlm_dtype=torch.float32 if mode == "fp32" else torch.bfloat16)
    if mode == "hybrid":
        model.validate_research_precision()
    model.eval().requires_grad_(requires_grad=False).to("cuda:0")
    torch.set_float32_matmul_precision("highest")
    rows, outputs = [], []
    with torch.no_grad():
        for index, inputs in enumerate(transformed):
            observation = model_api.Observation.from_dict(
                jax.tree.map(lambda value: torch.as_tensor(np.array(value), device="cuda:0")[None], inputs)
            )
            noise = torch.as_tensor(corpus["noise"][index][None], device="cuda:0")
            time = torch.tensor([0.5], device="cuda:0")
            prepared = reference_inputs(model, observation)
            context = model.encode_context(observation)
            before = [(key.clone(), value.clone()) for key, value in context.past_key_values]

            def cached(value, time_, ctx=context, current_model=model):
                return current_model.predict_velocity(value, time_, ctx)

            def reference(value, time_, prepared=prepared, current_model=model):
                return uncached_reference(current_model, prepared, value, time_)

            checks = {}
            actual = cached(noise, time).cpu().numpy()
            oracle = reference(noise, time).cpu().numpy()
            checks["velocity_t05"] = array_error(actual, oracle, atol=ATOL, rtol=RTOL)
            sample_outputs = {"velocity_t05": actual}
            if mode == "fp32":
                native = torch_joint_velocity(model, observation, noise, time).cpu().numpy()
                checks["native_velocity"] = array_error(oracle, native, atol=ATOL, rtol=RTOL)
            for steps in (1, 10):
                actual = trajectory(cached, noise, steps)
                oracle = trajectory(reference, noise, steps)
                checks[f"trajectory_k{steps}"] = array_error(actual, oracle, atol=ATOL, rtol=RTOL)
                sample_outputs[f"trajectory_k{steps}"] = actual
                sampled = model.sample_actions("cuda:0", observation, noise=noise, num_steps=steps).cpu().numpy()
                checks[f"sample_k{steps}"] = array_error(sampled, actual[:, -1], atol=ATOL, rtol=RTOL)
                if mode == "fp32":
                    native = trajectory(
                        lambda value, time_, obs=observation, current_model=model: torch_joint_velocity(
                            current_model, obs, value, time_
                        ),
                        noise,
                        steps,
                    )
                    checks[f"native_trajectory_k{steps}"] = array_error(oracle, native, atol=ATOL, rtol=RTOL)
            unchanged = all(
                torch.equal(actual, old)
                for pair, previous in zip(context.past_key_values, before, strict=True)
                for actual, old in zip(pair, previous, strict=True)
            )
            rows.append({"index": index, "checks": checks, "cache_unchanged": unchanged})
            if not unchanged or any(check["passed"] is not True for check in checks.values()):
                return {"passed": False, "observations": rows}, outputs
            outputs.append(sample_outputs)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"passed": True, "observations": rows}, outputs


def run_precision_stages(report, checkpoints, inputs, corpus):
    """Finish both FP32 gates before admitting either hybrid stage."""
    fp32_outputs = {}
    for mode in ("fp32", "hybrid"):
        for label in ("base", "reference"):
            report["stage_order"].append(f"{label}_{mode}")
            result, outputs = check_stage(checkpoints[label], inputs[label], corpus, mode)
            report["checkpoints"][label][mode] = result
            if not result["passed"]:
                raise ValueError(f"{label} {mode} oracle gate failed; no later precision stage is admitted")
            if mode == "fp32":
                fp32_outputs[label] = outputs
            else:
                # No tolerance or pass field: quantization drift is not a cache-equivalence test.
                report["checkpoints"][label]["cross_precision_drift"] = {
                    field: {
                        "max_absolute_error": float(
                            np.max(
                                np.abs(
                                    np.concatenate([row[field] for row in outputs]).astype(np.float64)
                                    - np.concatenate([row[field] for row in fp32_outputs[label]]).astype(np.float64)
                                )
                            )
                        )
                    }
                    for field in ("velocity_t05", "trajectory_k1", "trajectory_k10")
                }


def qualify(args, report):
    import torch

    from openpi.research.qualification import execution_identity
    from openpi.training.potential import validate_base_conversion

    if not torch.cuda.is_available():
        raise RuntimeError("Full-size hybrid qualification requires CUDA; tiny CPU fixtures are separate")
    corpus = load_corpus(args.corpus)
    if len(corpus["states"]) != 40 or not np.array_equal(corpus.get("task_indices"), np.arange(40)):
        raise ValueError("Hybrid qualification requires all forty audited task observations")
    require_staged_tokenizer()
    validate_base_conversion(args.base_checkpoint, require_receipt=True)
    validate_base_conversion(args.reference_checkpoint, require_receipt=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(mode=True)
    torch.backends.cudnn.benchmark = False
    report.update(
        identity=execution_identity(checkpoint=args.base_checkpoint),
        measured_gpu=True,
        hardware=torch.cuda.get_device_name(0),
        cuda_runtime=torch.version.cuda,
        corpus_sha256=file_digest(args.corpus),
        corpus_observations=40,
        checkpoints={},
        stage_order=[],
    )
    inputs = {}
    for label in ("base", "reference"):
        checkpoint, norms = getattr(args, f"{label}_checkpoint"), getattr(args, f"{label}_norm_stats")
        inputs[label] = transformed_inputs(corpus, norms)
        report["checkpoints"][label] = {
            "checkpoint_sha256": report["identity"]["base_checkpoint_sha256"]
            if label == "base"
            else file_digest(checkpoint / "model.safetensors"),
            "normalization_sha256": file_digest(norms),
            "conversion_receipt_sha256": file_digest(checkpoint / "config.json"),
            "input_sha256": [input_digest(value) for value in inputs[label]],
        }
    run_precision_stages(
        report, {label: getattr(args, f"{label}_checkpoint") for label in ("base", "reference")}, inputs, corpus
    )
    report["passed"] = True
    validate_hybrid_report(
        report,
        identity=report["identity"],
        corpus_sha256=report["corpus_sha256"],
        base_normalization_sha256=file_digest(args.base_norm_stats),
        reference_checkpoint_sha256=report["checkpoints"]["reference"]["checkpoint_sha256"],
        reference_normalization_sha256=file_digest(args.reference_norm_stats),
        reference_conversion_sha256=file_digest(args.reference_checkpoint / "config.json"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "base-checkpoint",
        "reference-checkpoint",
        "base-norm-stats",
        "reference-norm-stats",
        "corpus",
        "report",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error("Choose a new report path; numerical evidence is not overwritten")
    report = {
        "schema_version": 1,
        "purpose": PURPOSE,
        "passed": False,
        "measured_gpu": False,
        "precision": dict(RESEARCH_PRECISION),
        "fp32_bridge_precision": dict(FP32_BRIDGE_PRECISION),
        "valid_training_cost_profile": False,
        "valid_benchmark_evaluation": False,
    }
    try:
        qualify(args, report)
    except Exception as exc:
        report["passed"] = False
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    raise SystemExit(main())
