"""Reproduce the descriptive study using the original statistical function bodies.

Python 3.10+, standard library only. This does not confer hardware acceptance.
"""

import argparse
import ast
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import types
import typing

HERE = Path(__file__).resolve().parent
FIELDS = (
    "checkpoint_id",
    "group",
    "training_seed",
    "num_steps",
    "suite",
    "task_id",
    "initial_state_id",
    "episode_id",
    "rollout_seed",
    "paired_noise_seed",
    "protocol_id",
    "replan_steps",
    "action_horizon",
    "reference",
    "success",
    "failure_tags",
    "artifact_errors",
    "failure_kind",
    "attempt_id",
    "task_attempt_id",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ast_fingerprint(node):
    """Use the retained Python 3.13 dump representation on Python 3.10+.

    Python 3.13 omits empty list fields by default; older ast.dump versions do
    not. No source or AST nodes are changed. This preserves the existing pinned
    historical proof instead of weakening or replacing it.
    """

    def render(value):
        if isinstance(value, ast.AST):
            fields = []
            for name, child in ast.iter_fields(value):
                if child is None and getattr(type(value), name, Ellipsis) is None:
                    continue
                if isinstance(child, list) and not child:
                    continue
                fields.append(f"{name}={render(child)}")
            return f"{type(value).__name__}({', '.join(fields)})"
        if isinstance(value, list):
            return "[" + ", ".join(render(child) for child in value) + "]"
        return repr(value)

    return hashlib.sha256(render(node).encode()).hexdigest()


def statistics_api(expected):
    pins = json.loads((HERE / "reproducibility/sources.json").read_bytes())

    def pinned(name):
        path = HERE / "reproducibility" / name
        require(digest(path) == pins[name]["sha256"], "Pinned source changed: " + name)
        return path.read_bytes().decode("utf-8-sig")

    tree = ast.parse(pinned("statistics.py"))
    functions = {
        name: next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        for name in ("_percentile", "summarize", "paired_bootstrap")
    }
    namespace = {
        "math": math,
        "statistics": statistics,
        "random": random,
        "Counter": collections.Counter,
        "defaultdict": collections.defaultdict,
        "Any": typing.Any,
    }
    exec(compile(ast.Module(body=list(functions.values()), type_ignores=[]), "frozen_statistics.py", "exec"), namespace)
    entry = ast.parse(pinned("analysis_entrypoint.py"))
    original = next(n for n in entry.body if isinstance(n, ast.FunctionDef) and n.name == "analyze")
    compare = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "compare")
    loop = next(n for n in original.body if isinstance(n, ast.For))
    integration = next(n for n in original.body if isinstance(n, ast.If))
    wrapper = ast.parse(
        "def comparisons(records, manifest, *, comparisons, resamples, random_seed):\n"
        '    report={"comparisons":[]}\n    return report["comparisons"]\n'
    ).body[0]
    wrapper.body[1:1] = [compare, loop, integration]
    namespace["evaluation"] = types.SimpleNamespace(paired_bootstrap=namespace["paired_bootstrap"])
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), "frozen_comparisons.py", "exec"
        ),
        namespace,
    )

    proof = {
        "original_functions": {name: ast_fingerprint(n) for name, n in functions.items()},
        "original_comparison_bodies": [ast_fingerprint(n) for n in (compare, loop, integration)],
    }
    require(all(proof[k] == expected["source_proof"][k] for k in proof), "Original statistical body differs")
    return namespace, json.loads(pinned("manifest.json")), proof


def inputs(dataset):
    index = json.loads((dataset / "dataset-index.json").read_bytes())
    require(index["kind"] == "public_curated_study_dataset_v1", "Expected curated public dataset")
    require(
        (index["episode_count"], index["task_count"], index["hardware_acceptance_claimed"]) == (40000, 800, False),
        "Wrong dataset scope",
    )
    for line in (dataset / "checksums.sha256").read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        path = (dataset / name.lstrip("*")).resolve()
        require(path.is_relative_to(dataset.resolve()), "Checksum path escapes dataset")
        require(digest(path) == sha, "Dataset checksum mismatch: " + name)
    records, identities = [], set()
    with (dataset / "episodes.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            require(row.get("failure_kind") != "infrastructure", "Infrastructure row in selected data")
            key = tuple(row[k] for k in ("checkpoint_id", "num_steps", "suite", "task_id", "initial_state_id"))
            require(key not in identities, "Duplicate scientific episode")
            identities.add(key)
            records.append({k: row[k] for k in FIELDS if k in row})
    require(len(records) == 40000, "Expected 40,000 distinct episodes")
    counts = collections.Counter(k[:4] for k in identities)
    require(len(counts) == 800 and set(counts.values()) == {50}, "Expected 800 tasks of 50 episodes")
    require(len(index["excluded_infrastructure_attempts"]) == 4, "Expected four retained infrastructure exclusions")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    expected = json.loads((HERE.parent / "results/study-v0.1.0.json").read_bytes())
    records = inputs(args.dataset)
    api, manifest, proof = statistics_api(expected)
    summary = api["summarize"](records)
    require(summary == expected["summary"], "80-cell/800-task summaries differ from the delivered study")
    result = {
        "kind": "public_descriptive_reproduction_v1",
        "author": expected["author"],
        "episode_count": 40000,
        "task_count": 800,
        "suite_cells": 80,
        "summary_matches": True,
        "source_proof": proof,
        "hardware_acceptance_claimed": False,
    }
    if args.check_only:
        result["status"] = "dataset_checksums_and_original_sources_and_all_summaries_match"
    else:
        require(args.output is not None and not args.output.exists(), "Provide a new --output file")
        comparisons = api["comparisons"](
            records, manifest, comparisons=["P:B1", "P:B0"], resamples=10000, random_seed=0
        )
        require(len(comparisons) == len(expected["comparisons"]) == 8, "Expected eight contrasts")
        for actual, saved in zip(comparisons, expected["comparisons"], strict=True):
            require(all(actual[k] == saved[k] for k in actual), "A contrast or confidence interval changed")
        result.update(
            status="all_80_cells_and_8_contrasts_match_exactly",
            comparisons=comparisons,
            bootstrap_resamples=10000,
            random_seed=0,
        )
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("comparisons", "source_proof")}, sort_keys=True))


if __name__ == "__main__":
    main()
