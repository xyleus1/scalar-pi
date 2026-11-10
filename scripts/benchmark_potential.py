"""Measure warmed batch-one policy latency on a fixed audited observation corpus.

This runs inference only when invoked by the operator. No simulator, training,
downloads, or cloud provisioning is performed. K=1 and K=10 use separate reports.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from openpi.research.policy_checks import corpus_observation
from openpi.research.policy_checks import file_digest
from openpi.research.policy_checks import load_corpus
from openpi.research.policy_checks import require_staged_tokenizer
from openpi.research.policy_checks import timing_summary


def measure_calls(policy, corpus, *, warmup: int, calls: int, synchronize, reset_peak):
    if warmup < 1 or calls < 1:
        raise ValueError("Warmup and measured call counts must be positive")
    samples = []
    count = len(corpus["states"])
    for index in range(warmup + calls):
        if index == warmup:
            synchronize()
            reset_peak()
        # Measured calls always start at corpus item zero, independently of warmup.
        sample_index = (index if index < warmup else index - warmup) % count
        synchronize()
        output = policy.infer(corpus_observation(corpus, sample_index), noise=corpus["noise"][sample_index])
        synchronize()
        if output.get("policy_error") or not np.isfinite(output["actions"]).all():
            raise RuntimeError(f"Policy produced invalid actions during timing: {output.get('policy_error')}")
        if index >= warmup:
            samples.append({"corpus_index": sample_index, **output["policy_timing"]})
    return samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", choices=("pi05_libero", "pi05_libero_b0", "pi05_libero_b1", "pi05_libero_potential"), required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, choices=(1, 10), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--calls", type=int, default=100)
    args = parser.parse_args(argv)
    if not args.checkpoint.is_dir() or args.report.exists():
        parser.error("Use a staged local checkpoint and a new report path")
    if args.warmup < 1 or args.calls < 1:
        parser.error("warmup and calls must be positive")
    corpus = load_corpus(args.corpus)
    require_staged_tokenizer()

    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config

    policy = create_trained_policy(
        get_config(args.config), args.checkpoint, sample_kwargs={"num_steps": args.num_steps}, pytorch_device="cuda:0"
    )
    if policy.metadata["runtime"]["backend"] == "pytorch":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("A real CUDA host is required for this latency benchmark")

        def synchronize():
            torch.cuda.synchronize(0)

        def reset_peak():
            torch.cuda.reset_peak_memory_stats(0)
    else:
        import jax

        if not any(device.platform == "gpu" for device in jax.devices()):
            raise RuntimeError("The released-reference benchmark requires the JAX GPU backend")
        synchronize = jax.effects_barrier  # Policy.infer already blocks on action results.

        def reset_peak():
            pass

    samples = measure_calls(
        policy, corpus, warmup=args.warmup, calls=args.calls, synchronize=synchronize, reset_peak=reset_peak
    )
    if policy.metadata["runtime"]["backend"] == "pytorch":
        memory = {
            "allocator": "torch",
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        }
    else:
        memory = {"allocator": "jax", "stats": jax.devices()[0].memory_stats()}
    report = {
        "schema_version": 1,
        "purpose": "isolated_warmed_policy_latency",
        "valid_training_cost_profile": False,
        "valid_benchmark_evaluation": False,
        "policy": policy.metadata,
        "corpus_sha256": file_digest(args.corpus),
        "corpus_observations": len(corpus["states"]),
        "warmup_calls": args.warmup,
        "measured_calls": args.calls,
        "batch_size": 1,
        "action_horizon": 10,
        "replan_horizon": 5,
        "num_steps": args.num_steps,
        "summary": timing_summary(samples),
        "memory": memory,
        "calls": samples,
        "limitations": ["Requires an otherwise idle GPU; native JAX versus eager Torch is a system comparison."],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote warmed policy timing: {args.report}")


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()
