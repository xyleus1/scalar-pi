"""Original fixed-task summaries and paired bootstrap estimators.

Keep these estimators value-compatible with the published study reproduction.
"""

# ruff: noqa: UP038 -- The simulator also imports this module on Python 3.8.

from __future__ import annotations

from collections import Counter
from collections import defaultdict
import math
import random
import statistics
from typing import Any


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-checkpoint/seed/suite summaries; released references stay separate."""
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["checkpoint_id"], record["num_steps"], record["suite"])].append(record)
    result = {"trained": [], "reference": []}
    for (checkpoint_id, steps, suite), rows in sorted(grouped.items()):
        first = rows[0]
        timings = defaultdict(list)
        for row in rows:
            for timing in row.get("latencies", []):
                for name, value in timing.items():
                    if isinstance(value, (float, int)) and math.isfinite(value):
                        timings[name].append(float(value))
        item = {
            "checkpoint_id": checkpoint_id,
            "group": first["group"],
            "training_seed": first.get("training_seed"),
            "num_steps": steps,
            "suite": suite,
            "episodes": len(rows),
            "successes": sum(row["success"] for row in rows),
            "success_rate": statistics.mean(float(row["success"]) for row in rows),
            "failure_tags": dict(Counter(tag for row in rows for tag in row.get("failure_tags", []))),
            "artifact_errors": dict(Counter(error["kind"] for row in rows for error in row.get("artifact_errors", []))),
            "latency_ms": {
                name: {"count": len(values), "mean": statistics.mean(values), "p95": _percentile(values, 0.95)}
                for name, values in sorted(timings.items())
            },
        }
        result["reference" if first.get("reference", False) else "trained"].append(item)
    checkpoint_groups = defaultdict(list)
    for section in ("trained", "reference"):
        for item in result[section]:
            checkpoint_groups[(section, item["checkpoint_id"], item["num_steps"])].append(item)
    result["checkpoint_macro_averages"] = [
        {
            "reference": section == "reference",
            "checkpoint_id": checkpoint_id,
            "num_steps": steps,
            "group": items[0]["group"],
            "training_seed": items[0]["training_seed"],
            "suites": [item["suite"] for item in items],
            "success_rate": statistics.mean(item["success_rate"] for item in items),
        }
        for (section, checkpoint_id, steps), items in sorted(checkpoint_groups.items())
    ]
    task_groups = defaultdict(list)
    for row in records:
        task_groups[(row["checkpoint_id"], row["num_steps"], row["suite"], row["task_id"])].append(row)
    result["tasks"] = [
        {
            "checkpoint_id": checkpoint_id,
            "num_steps": steps,
            "suite": suite,
            "task_id": task,
            "episodes": len(rows),
            "success_rate": statistics.mean(float(row["success"]) for row in rows),
        }
        for (checkpoint_id, steps, suite, task), rows in sorted(task_groups.items())
    ]
    return result


def paired_bootstrap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    resamples: int = 10000,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Paired seed/state bootstrap for equally weighted, fixed benchmark tasks.

    Initial-state draws are shared across sampled training seeds and both methods.
    A one-seed interval measures state variation, not training-seed uncertainty.
    """
    if resamples < 100:
        raise ValueError("Require at least 100 resamples")

    def index(rows):
        indexed = {}
        for row in rows:
            key = (row.get("training_seed"), row["suite"], row["task_id"], row["initial_state_id"])
            if key in indexed:
                raise ValueError(f"Duplicate paired key: {key}; select one group and K per side")
            indexed[key] = row
        return indexed

    li, ri = index(left), index(right)
    if not li or li.keys() != ri.keys():
        raise ValueError("Paired comparison requires identical nonempty seed/task/initial-state keys")
    for key in li:
        for field in (
            "episode_id",
            "rollout_seed",
            "paired_noise_seed",
            "protocol_id",
            "replan_steps",
            "action_horizon",
        ):
            if li[key].get(field) != ri[key].get(field):
                raise ValueError(f"Pairing mismatch in {field}: {key}")
        if bool(li[key].get("reference", False)) != bool(ri[key].get("reference", False)):
            raise ValueError("Do not treat a released reference as an independently trained seed")
    seeds = sorted({key[0] for key in li}, key=str)
    grid = defaultdict(lambda: defaultdict(set))
    for _, suite, task, state in li:
        grid[suite][task].add(state)
    for seed in seeds:
        for suite, tasks in grid.items():
            for task, states in tasks.items():
                if any((seed, suite, task, state) not in li for state in states):
                    raise ValueError("Unbalanced training seed/task/episode grid")
    differences = {key: float(li[key]["success"]) - float(ri[key]["success"]) for key in li}
    rng = random.Random(random_seed)
    draws = []
    suite_names = sorted(grid)
    for _ in range(resamples):
        sampled_seeds = rng.choices(seeds, k=len(seeds))
        suite_means = []
        for suite in suite_names:
            tasks = sorted(grid[suite])
            task_means = []
            for task in tasks:
                states = sorted(grid[suite][task])
                sampled_states = rng.choices(states, k=len(states))
                task_means.append(
                    sum(differences[(seed, suite, task, state)] for seed in sampled_seeds for state in sampled_states)
                    / (len(sampled_seeds) * len(sampled_states))
                )
            suite_means.append(statistics.mean(task_means))
        draws.append(statistics.mean(suite_means))
    ci95 = [_percentile(draws, 0.025), _percentile(draws, 0.975)]
    point = statistics.mean(
        statistics.mean(
            statistics.mean(differences[(seed, suite, task, state)] for seed in seeds for state in states)
            for task, states in grid[suite].items()
        )
        for suite in suite_names
    )
    return {
        "difference_left_minus_right": point,
        "ci95": ci95,
        "resamples": resamples,
        "random_seed": random_seed,
        "training_seeds": seeds,
        "paired_episodes": len(li),
        "method": "paired training-seed/initial-state percentile bootstrap; fixed equally weighted benchmark tasks",
        "training_seed_variability_estimated": len(seeds) > 1 and seeds != [None],
        "scope": (
            "conditional on one checkpoint; training-seed variability is not estimated"
            if len(seeds) == 1
            else "training-seed and initial-state variation on the fixed benchmark tasks; few-seed estimate"
        ),
    }
