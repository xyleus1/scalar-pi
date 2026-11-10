# Evaluation timing and task recovery

Run calibration after the audited corpus, numerical/serving qualification, complete
100-update cost profiles, and the four released-policy compatibility episodes pass.
These commands are operator-run on the supervised Lambda session. They do not start
a cloud instance themselves. Keep the Windows controller running throughout.

The calibration uses eight existing policy benchmarks (B0/B1/P/reference × K1/K10)
and the native LIBERO runner in all 40 scenes. Research timing uses the diagnostic
profile checkpoints; those checkpoints remain ineligible for confirmatory scores.
The four-episode smoke supplies observed WebSocket overhead. No success rate or
early-success episode length enters the cost estimate.

## Render the concrete specification

From `/lambda/nfs/scalar-pi/repo`, set `SMOKE_MANIFEST` to the retained manifest from
the completed smoke session, then run this once. It resolves each research checkpoint
from its verified canonical profile report, including the actual completed update.
The controller supplies `SCALAR_PI_ATTEMPT`.

```bash
export SMOKE_MANIFEST=/lambda/nfs/scalar-pi/data/experiments/reports/smoke/SESSION/matrix.json
python3 - <<'PY'
import json, os
from pathlib import Path
root = Path('/lambda/nfs/scalar-pi')
reports = root / 'data/experiments/reports'
spec = json.loads(Path('configs/potential/evaluation_calibration.example.json').read_text())
spec['output_dir'] = str(root / 'data/experiments/calibration' / os.environ['SCALAR_PI_ATTEMPT'])
spec['smoke_manifest'] = os.environ['SMOKE_MANIFEST']
for policy in spec['policies']:
    if policy['group'] == 'reference':
        continue
    profile = json.loads((reports / 'profiles' / (policy['config'] + '.json')).read_text())
    run = Path(profile['measurement_source']).parent
    if run.is_relative_to('/data'):
        run = root / 'data' / run.relative_to('/data')
    policy['checkpoint_dir'] = str(run / str(profile['completed_updates']))
target = reports / 'evaluation_calibration_spec.json'
with target.open('x') as stream:
    json.dump(spec, stream, indent=2)
print(target)
PY
```

Preview the exact twelve commands without loading a model or simulator:

```bash
python3 scripts/calibrate_potential_evaluation.py \
  --spec /lambda/nfs/scalar-pi/data/experiments/reports/evaluation_calibration_spec.json \
  --output /lambda/nfs/scalar-pi/data/experiments/reports/evaluation_calibration.json
```

Execute inside a supervised lease through the container wrapper so both actual
container identities are supplied. The wrapper orchestrates on the host and invokes
the existing model/simulator containers:

```bash
bash scripts/lambda/run.sh calibrate --execute \
  --spec /lambda/nfs/scalar-pi/data/experiments/reports/evaluation_calibration_spec.json \
  --output /lambda/nfs/scalar-pi/data/experiments/reports/evaluation_calibration.json
```

For a later session, reuse the same spec and add `--resume`; complete component
receipts are revalidated. A failed component's log is retained. Investigate it and
use a fresh component output directory/spec rather than overwriting failed evidence.
An aggregate timing report is published only after every component validates.

## Budget and evaluation inputs

The report provides `evaluation_seconds_per_episode[group]["1"|"10"][suite]`,
`task_budget_seconds` with the same keys, `server_start_budget_seconds`, source and
container identities, and SHA256 hashes for its evidence files. Point the measured
budget input's `calibration_report` to this report. The budget stage reads both timing
maps directly, verifies the component evidence and compares its declared identity.

Set `timing_report` in the full or pilot evaluation matrix to the same absolute path.
Keep `task_shards: true` and `save_videos: false`. Use:

```bash
bash scripts/lambda/run.sh evaluate --execute MATRIX.json --resume-completed-suites
```

The simulator keeps each task's original ID and runs all 50 initial states in their
original order. One simulator process handles several pending tasks within a suite;
a new simulator process starts for another suite. The policy server stays loaded
across suites at the same checkpoint/K. Before a task starts, its complete measured
time allowance and the shutdown reserve must fit the absolute controller deadline.

Task files live under `SUITE.tasks/task-XX/ATTEMPT.jsonl`. A complete task can be
reused after a session ends. An interrupted task must be rerun in full with an
explicit fresh entry such as:

```json
{"task_attempts": {"P-seed42/k10/libero_goal/task-03": "retry-02"}}
```

The original attempt remains available. Once all ten tasks validate, the aggregator
publishes the canonical 500-episode suite and a receipt identifying its task inputs.
It rejects missing/duplicate states, renumbered tasks, infrastructure failures and
inconsistent provenance. The existing full-matrix statistics are unchanged.

The scene calibration also compares each task's initial observation and reset RNG
between contiguous and standalone initialization. These runtime checks must pass;
CPU mocks alone do not establish simulator parity. Component timing uses the native
maximum control-step/query counts and measured p95 costs, charging cold starts per
task conservatively. It is a forecast, not a wall-time guarantee. Task admission adds
20% operational headroom; the overall project budget retains its separate 20%
contingency. Recheck actual completed-task durations before extending paid work.
