"""Export fixed-input direct-model outputs in separate processes, then compare.

Run `export --backend jax` on the staged original checkpoint and `export
--backend torch` on its FP32 conversion. Both use the same audited corpus and
normalization asset. `compare` is CPU-only and never loads a model. Exports
require a GPU; they neither download assets nor perform training or simulation.
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np

from openpi.research.policy_checks import array_error
from openpi.research.policy_checks import corpus_observation
from openpi.research.policy_checks import file_digest
from openpi.research.policy_checks import input_digest
from openpi.research.policy_checks import load_corpus
from openpi.research.policy_checks import require_staged_tokenizer


def torch_joint_velocity(model, observation, actions, time):
    """Native joint attention path on fixed inference inputs, before scalarization."""
    import torch

    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    images, image_masks, tokens, token_masks, state = model._preprocess_observation(observation, train=False)  # noqa: SLF001
    prefix, prefix_masks, prefix_ar = model.embed_prefix(images, image_masks, tokens, token_masks)
    suffix, suffix_masks, suffix_ar, condition = model.embed_suffix(state, actions, time)
    if model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype != torch.float32:
        raise ValueError("Native joint parity requires all-FP32 weights; mixed cached arithmetic is checked separately")
    masks = torch.cat((prefix_masks, suffix_masks), dim=1)
    autoregressive = torch.cat((prefix_ar, suffix_ar), dim=1)
    (_, output), _ = model.paligemma_with_expert.forward(
        attention_mask=model._prepare_attention_masks_4d(make_att_2d_masks(masks, autoregressive)),  # noqa: SLF001
        position_ids=torch.cumsum(masks, dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[prefix, suffix],
        use_cache=False,
        adarms_cond=[None, condition],
    )
    return model.action_out_proj(output[:, -model.config.action_horizon :].float())


def jax_joint_velocity(model, observation, actions, time):
    import jax.numpy as jnp

    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask

    observation = model_api.preprocess_observation(None, observation, train=False)
    prefix, prefix_mask, prefix_ar = model.embed_prefix(observation)
    suffix, suffix_mask, suffix_ar, condition = model.embed_suffix(observation, actions, time)
    masks = jnp.concatenate((prefix_mask, suffix_mask), axis=1)
    autoregressive = jnp.concatenate((prefix_ar, suffix_ar), axis=0)
    (_, output), _ = model.PaliGemma.llm(
        [prefix, suffix],
        mask=make_attn_mask(masks, autoregressive),
        positions=jnp.cumsum(masks, axis=1) - 1,
        adarms_cond=[None, condition],
    )
    return model.action_out_proj(output[:, -model.action_horizon :])


def export_outputs(args):
    import jax
    import jax.numpy as jnp

    from openpi import transforms
    from openpi.models import model as model_api
    from openpi.policies.policy_config import checkpoint_digest
    from openpi.shared import nnx_utils
    from openpi.shared import normalize
    from openpi.training.config import get_config

    corpus = load_corpus(args.corpus)
    require_staged_tokenizer()
    config = get_config("pi05_libero")
    model_config = dataclasses.replace(config.model, dtype="float32", pytorch_compile_mode=None)
    data_config = config.data.create(config.assets_dirs, model_config)
    norms = normalize.deserialize_json(args.norm_stats.read_text(encoding="utf-8"))
    transform = transforms.compose(
        [
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norms, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    transformed = [transform(corpus_observation(corpus, index)) for index in range(len(corpus["states"]))]
    result = {"input_sha256": np.asarray([input_digest(inputs) for inputs in transformed])}
    metadata = {
        "schema_version": 1,
        "backend": args.backend,
        "precision": "float32",
        "corpus_sha256": file_digest(args.corpus),
        "normalization_sha256": file_digest(args.norm_stats),
        "checkpoint_sha256": checkpoint_digest(args.checkpoint),
        "flow_time": 0.5,
        "container_image": os.environ.get("SCALAR_PI_MODEL_IMAGE"),
    }
    outputs = {"velocity_t05": [], "actions_k1": [], "actions_k10": []}
    if args.backend == "jax":
        if not any(device.platform == "gpu" for device in jax.devices()):
            raise RuntimeError("JAX export requires a GPU backend; unset JAX_PLATFORMS=cpu")
        jax.config.update("jax_default_matmul_precision", "highest")
        model = model_config.load(model_api.restore_params(args.checkpoint / "params", dtype=jnp.float32))
        sample = nnx_utils.module_jit(model.sample_actions)
        for index, inputs in enumerate(transformed):
            observation = model_api.Observation.from_dict(jax.tree.map(lambda value: jnp.asarray(value)[None], inputs))
            noise = jnp.asarray(corpus["noise"][index][None])
            outputs["velocity_t05"].append(
                np.asarray(jax_joint_velocity(model, observation, noise, jnp.asarray([0.5])))
            )
            for steps in (1, 10):
                outputs[f"actions_k{steps}"].append(
                    np.asarray(sample(jax.random.key(0), observation, noise=noise, num_steps=steps))
                )
        metadata["hardware"] = str(jax.devices()[0])
    else:
        import safetensors.torch
        import torch

        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

        if not torch.cuda.is_available():
            raise RuntimeError("Torch export requires CUDA")
        model_config = dataclasses.replace(model_config, freeze_vlm=True)
        model = PI0Pytorch(model_config)
        safetensors.torch.load_model(model, args.checkpoint / "model.safetensors", strict=True)
        model.configure_expert_only(vlm_dtype=torch.float32)
        model.eval().requires_grad_(requires_grad=False).to("cuda:0")
        torch.set_float32_matmul_precision("highest")
        observations = [
            model_api.Observation.from_dict(
                jax.tree.map(lambda value: torch.as_tensor(np.array(value), device="cuda:0")[None], inputs)
            )
            for inputs in transformed
        ]
        cache_checks = []
        with torch.no_grad():
            for index, observation in enumerate(observations):
                noise = torch.as_tensor(corpus["noise"][index][None], device="cuda:0")
                time = torch.tensor([0.5], device="cuda:0")
                joint = torch_joint_velocity(model, observation, noise, time)
                cached = model.predict_velocity(noise, time, model.encode_context(observation))
                cache_checks.append(array_error(cached.cpu().numpy(), joint.cpu().numpy(), atol=1e-4, rtol=1e-4))
                outputs["velocity_t05"].append(joint.cpu().numpy())
                for steps in (1, 10):
                    outputs[f"actions_k{steps}"].append(
                        model.sample_actions("cuda:0", observation, noise=noise, num_steps=steps).cpu().numpy()
                    )
            model.configure_expert_only(vlm_dtype=torch.bfloat16)
            model.requires_grad_(requires_grad=False)
            mixed_checks = []
            for index, observation in enumerate(observations):
                noise = torch.as_tensor(corpus["noise"][index][None], device="cuda:0")
                time = torch.tensor([0.5], device="cuda:0")
                cached = model.predict_velocity(noise, time, model.encode_context(observation))
                mixed_checks.append(
                    array_error(cached.cpu().numpy(), outputs["velocity_t05"][index], atol=2e-2, rtol=2e-2)
                )
        metadata.update(
            hardware=torch.cuda.get_device_name(0),
            cuda_runtime=torch.version.cuda,
            cached_joint_fp32=cache_checks,
            mixed_cached_vs_fp32=mixed_checks,
            mixed_joint_supported=False,
        )
    result.update({name: np.concatenate(values) for name, values in outputs.items()})
    result["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        np.savez_compressed(stream, **result)
    print(f"Wrote {args.backend} parity outputs: {args.output}")


def compare_exports(jax_path: Path, torch_path: Path) -> dict:
    with np.load(jax_path, allow_pickle=False) as archive:
        jax_values = {name: archive[name] for name in archive.files}
    with np.load(torch_path, allow_pickle=False) as archive:
        torch_values = {name: archive[name] for name in archive.files}
    source = json.loads(str(jax_values["metadata_json"]))
    converted = json.loads(str(torch_values["metadata_json"]))
    if source["backend"] != "jax" or converted["backend"] != "torch":
        raise ValueError("Supply JAX reference and Torch conversion exports in that order")
    for key in ("schema_version", "precision", "corpus_sha256", "normalization_sha256", "flow_time"):
        if source[key] != converted[key]:
            raise ValueError(f"Export protocol differs: {key}")
    if not np.array_equal(jax_values["input_sha256"], torch_values["input_sha256"]):
        raise ValueError("Exports did not receive identical transformed observations")
    errors = {
        name: array_error(torch_values[name], jax_values[name], atol=1e-4, rtol=1e-4)
        for name in ("velocity_t05", "actions_k1", "actions_k10")
    }
    count = len(jax_values["input_sha256"])
    if count < 1 or any(len(converted[key]) != count for key in ("cached_joint_fp32", "mixed_cached_vs_fp32")):
        raise ValueError("Every exported observation requires both cache and precision checks")
    cache_checks = [*converted["cached_joint_fp32"], *converted["mixed_cached_vs_fp32"]]
    return {
        "schema_version": 1,
        "purpose": "direct_model_conversion_and_cache_parity",
        "passed": all(value["passed"] for value in [*errors.values(), *cache_checks]),
        "jax_export_sha256": file_digest(jax_path),
        "torch_export_sha256": file_digest(torch_path),
        "jax": source,
        "torch": converted,
        "conversion_errors": errors,
        "limitations": [
            "Engineering numerical tolerances, not a task-performance guarantee; no scalar model is compared here.",
            "Native joint attention does not support mixed BF16 VLM/FP32 expert weights. Mixed precision is measured against the all-FP32 reference.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--backend", choices=("jax", "torch"), required=True)
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--norm-stats", type=Path, required=True)
    export.add_argument("--corpus", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--jax-output", type=Path, required=True)
    compare.add_argument("--torch-output", type=Path, required=True)
    compare.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    destination = args.output if args.command == "export" else args.report
    if destination.exists():
        parser.error("Choose a new output path; parity evidence is not overwritten")
    if args.command == "export":
        expected = "params" if args.backend == "jax" else "model.safetensors"
        if not (args.checkpoint / expected).exists():
            parser.error(f"Checkpoint must already contain {expected}")
        export_outputs(args)
        return 0
    report = compare_exports(args.jax_output, args.torch_output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote parity comparison: {args.report}; passed={report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    raise SystemExit(main())
