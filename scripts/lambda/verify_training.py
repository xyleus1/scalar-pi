"""Verify a terminal checkpoint before skipping its completed training run.

Read-only: validates source, staged inputs and saved bytes without loading a model
or deserializing training_state.pt. The existing batch-selection verifier supplies
the current experiment identity.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy

CONFIGS = ("pi05_libero_b0", "pi05_libero_b1", "pi05_libero_potential")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_terminal(checkpoint, config, seed, selection):
    """Compare a terminal artifact with an already verified batch-selection report."""
    from openpi.shared.potential_checkpoint import validate_checkpoint_metadata

    if config not in CONFIGS or type(seed) is not int or seed not in (17, 42, 93):
        raise ValueError("Unsupported research config or training seed")
    mode = "potential_dot" if config == "pi05_libero_potential" else "direct"
    initialization = "pretrained_output" if config == "pi05_libero_b0" else "zero_output"
    metadata = validate_checkpoint_metadata(checkpoint, velocity_mode=mode, action_horizon=10, action_dim=32)
    if not isinstance(metadata, dict):
        raise ValueError("Terminal training checkpoint requires research metadata")
    expected = {
        "schema_version": 1,
        "trainer": "potential_pi",
        "run_mode": "training",
        "training_seed": seed,
        "optimizer_updates": 30000,
        "inference_weights": "ema",
        "initialization": initialization,
        "freeze_vlm": True,
        "expert_precision": "float32",
        "vlm_precision": "bfloat16",
        "matmul_precision": "highest",
        "code_commit": selection["code"]["commit"],
        "source_snapshot_sha256": selection["code"]["source_snapshot_sha256"],
        "container_image_id": selection["container_image_id"],
        "base_checkpoint_hash": selection["inputs_sha256"]["weights"],
        "base_conversion_receipt_sha256": selection["inputs_sha256"]["conversion_receipt"],
        "data_manifest_sha256": selection["inputs_sha256"]["data_manifest"],
        "normalization_hash": selection["normalization_sha256"],
    }
    for field, value in expected.items():
        if type(metadata.get(field)) is not type(value) or metadata.get(field) != value:
            raise ValueError(f"Terminal checkpoint differs from the selected experiment: {field}")
    signature = metadata.get("resume_signature", {})
    expected_signature = {
        "config": config,
        "seed": seed,
        "microbatch_size": selection["microbatch_size"],
        "effective_batch_size": 256,
        "gradient_checkpointing": selection["gradient_checkpointing"],
        "deterministic": True,
        **{
            key: expected[key]
            for key in (
                "initialization",
                "matmul_precision",
                "run_mode",
                "code_commit",
                "source_snapshot_sha256",
                "container_image_id",
            )
        },
    }
    for field, value in expected_signature.items():
        if type(signature.get(field)) is not type(value) or signature.get(field) != value:
            raise ValueError(f"Terminal checkpoint signature differs: {field}")
    for filename, field in (("model.safetensors", "model_sha256"), ("raw_model.safetensors", "raw_model_sha256")):
        if digest(checkpoint / filename) != metadata.get(field):
            raise ValueError(f"Terminal checkpoint weight hash mismatch: {filename}")
    norms = checkpoint / "assets/physical-intelligence/libero/norm_stats.json"
    if digest(norms) != expected["normalization_hash"]:
        raise ValueError("Terminal checkpoint normalization asset differs")
    state = checkpoint / "training_state.pt"
    if not state.is_file() or state.stat().st_size == 0:
        raise ValueError("Terminal checkpoint lacks its saved training state")
    return {
        "verified": True,
        "checkpoint": str(checkpoint),
        "config": config,
        "training_seed": seed,
        "optimizer_updates": 30000,
        "metadata_sha256": digest(checkpoint / "potential_metadata.json"),
        "model_sha256": metadata["model_sha256"],
        "code_commit": expected["code_commit"],
        "container_image_id": expected["container_image_id"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--seed", type=int, choices=(17, 42, 93), required=True)
    parser.add_argument("--selection", dest="verify_report", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--qualification-report", type=Path, required=True)
    args = parser.parse_args(argv)
    verifier = runpy.run_path(str(Path(__file__).resolve().parents[1] / "select_potential_batch.py"))
    selection = verifier["verify_report"](args)
    print(json.dumps(verify_terminal(args.checkpoint, args.config, args.seed, selection), indent=2))


if __name__ == "__main__":
    main()
