# Running scalar-pi on Lambda

This runbook implements the matched frozen-backbone pi0.5 study. The scalar arm
changes the velocity to the full gradient of a dot-product potential; B0 retains
the pretrained direct head and B1 resets that head exactly like the scalar arm.
P versus B1 at K=10 is primary. P versus B0 measures practical benefit. This is
not a reproduction of full-VLM fine-tuning, and improvement is not assumed.

The project limit is **$2,000**, including preparation and storage. The initial
preparation/qualification allowance is $100. The complete study has three seeds
and 40,000 evaluation episodes. Its fallback has seed 42 and 16,000 episodes;
both retain all tasks and 30,000 updates per arm. If neither fits, stop.

## What runs where

| Component | Implementation |
| --- | --- |
| Windows control | `scripts/lambda/Invoke-Lambda.ps1`, Python standard-library Lambda API client, OpenSSH |
| Authentication | Existing private DPAPI key loader; never copied into this repository or the worker |
| Compute | One x86-64 H100 PCIe 80 GB in `us-west-3` |
| Storage | Project filesystem `scalar-pi`, attached when the instance is created |
| Model environment | Existing locked Dockerfile, Python 3.11.9, Torch 2.7.1, patched Transformers 4.53.2 |
| Simulator | Existing Python 3.8 LIBERO image, headless EGL, native MuJoCo/robosuite dependencies |
| Research | Existing training, policy, evaluation, and analysis entrypoints |

The helper defaults to dry runs for every mutating command. You invoke the
commands with `--execute` when ready. No startup script automatically begins
training. No GPU, training, or real simulator run was needed to implement this
tooling.

## 1. Private account settings

Copy `configs/potential/lambda.example.json` outside Git, normally to
`~/.config/scalar-pi/lambda.json`. On the current workstation this private file
has already been prepared with the supplied key-loader path, a project SSH-key
path, the observed operator IPv4 address, and an available GPU Base 22.04 image
ID. **`credits_verified` remains false.**

Before paid preparation, check Lambda Billing for credit eligibility and expiry,
and the console's filesystem price. Record actual noncompute spending and a
remaining storage/tax reserve in the private settings, then set
`credits_verified` to true. The API inspection does not verify credit terms.
Recheck `operator_cidr` if your public IP changes. The image is pinned by ID and
checked against Lambda's current regional catalog at launch.

From the repository in PowerShell:

```powershell
$lambda = '.\scripts\lambda\Invoke-Lambda.ps1'
& $lambda inspect
& $lambda prepare                 # preview; no API mutation
& $lambda prepare --execute       # SSH key registration + persistent storage
$commit = git rev-parse HEAD
```

`prepare` generates a dedicated local Ed25519 key if absent. It records resource
IDs privately and refuses to replace unrecorded resources with the same name.
It does not launch a GPU or change the shared account firewall. Instance
cloud-init restricts incoming host traffic to the configured SSH /32. The
policy server also binds to localhost.

The observed H100 PCIe quote on 2026-11-10 was $3.29/hour. This is not a future
price guarantee. Inspection and launch recheck the price and capacity; the
controller refuses a higher rate or substitute hardware. Lambda bills idle
instances and bills storage after instances terminate. See the official
[API](https://docs.lambda.com/api/cloud),
[filesystem](https://docs.lambda.com/public-cloud/filesystems/), and
[billing](https://docs.lambda.com/public-cloud/billing/) documentation.

## 2. Session lifecycle

Every session clones/checks out the supplied immutable commit, attaches the
same persistent filesystem, runs your shell script over SSH, downloads small
files from `data/experiments/reports`, and terminates the recorded instance in a
`finally` block. Logs and the cost ledger live beside the private settings.

```powershell
& $lambda run --script scripts/lambda/prepare.sh --commit $commit --max-hours 4
# Review the preview, then explicitly launch:
& $lambda run --script scripts/lambda/prepare.sh --commit $commit --max-hours 4 --execute
```

Choose a session duration inside the remaining phase allowance. Four hours is
an example lease, not a measured preparation estimate. The controller counts
boot, setup, idle time, and execution, rechecks price during the session, and
keeps 20% compute headroom. It refuses concurrent controllers and another
session while an earlier recorded instance or uncertain launch is unresolved.

**Keep Windows awake and the foreground controller running.** A local process
cannot guarantee termination if the computer loses power/network access. The
ledger is an estimate, not a provider-enforced spending cap. Linux shutdown
does not end Lambda billing. If interrupted, inspect Lambda and terminate the
exact recorded instance:

```powershell
& $lambda terminate --instance-id RECORDED_ID                 # preview
& $lambda terminate --instance-id RECORDED_ID --execute
```

An uncertain launch is recorded before the request and blocks another launch.
The helper attempts to find and terminate the unique recorded name after a
lost response. Reconcile any unresolved request in the console; do not reset
the ledger or repeatedly submit launches. Manual termination of an already
absent recorded instance reconciles the ledger conservatively.

## 3. Staging and data audit

`prepare.sh` builds both existing images and saves them on persistent storage
for later instances. It stages:

- Dataset `physical-intelligence/libero` at immutable revision
  `a4336d589d589045d1c56423ffdf3b88a0e19b1f`.
- Original `pi05_base` and released `pi05_libero` checkpoints and tokenizer.
- Verified FP32 Torch conversions of both checkpoints.
- One official normalization asset copied identically to B0/B1/P.
- A complete decoded-data audit and a fixed 40-task observation corpus.

The dataset has 1,693 episodes, 273,465 frames, and 40 tasks. Snapshot files are
bound to remote Git/LFS digests. The audit checks images, finite values,
episode/task/frame identities, beginning/middle/terminal windows, native
terminal-action repetition, and upstream/research transform parity. It does
not filter demonstrations or invent a training split.

After staging, normalization, audits, training, and profiling use the fixed
`/data/lerobot` root with Hugging Face offline flags. Dataset bytes are verified
**before** LeRobot construction and the loaded root is checked afterward.
Research commands separately require the staged GCS tokenizer. The native
normalization procedure processes 273,408 windows (1,068 full batches), omitting
the final 57-window partial batch. Its quantile estimator is unchanged.

The source checkout is mounted at `/app`; persistent data at `/data` and at its
host path `/lambda/nfs/scalar-pi/data`. That second mount lets host orchestration
and both containers refer to the same result/checkpoint paths. Exact Docker
image IDs are supplied automatically to training and evaluation provenance.

Preserve failed-attempt outputs. Staging downloads are resumable, but audit,
parity, and experiment files are evidence; use a new attempt path when an
existing command refuses overwrite.
On repeated staging, `--verify-audit` checks the existing corpus against its
audit, the verified dataset, and current normalization before reusing it.

## 4. Numerical and simulator qualification

Launch these as separate bounded sessions using the same `run` command:

```powershell
& $lambda run --script scripts/lambda/qualify.sh --commit $commit --max-hours 4 --execute
& $lambda run --script scripts/lambda/smoke_libero.sh --commit $commit --max-hours 1 --execute
```

`qualify.sh` exports original JAX and converted Torch outputs in separate
processes with identical preprocessed observations, explicit noise, and flow
time. It compares direct velocity and K=1/K=10 chunks for both checkpoints.
FP32 tolerances are `atol=rtol=1e-4`; mixed-precision checks use `2e-2`. These are
declared engineering tolerances. Investigate failures rather than relaxing
them after observing success rates.

Native joint execution cannot combine a BF16 VLM with an FP32 expert. Therefore
joint/cached parity is checked in FP32, followed by a separate mixed-versus-FP32
cached drift check. Qualification also checks finite higher derivatives, frozen
VLM bytes, immutable caches, reset-head learning, and the actual policy loader.
Training and research serving explicitly use matmul precision `highest`.
Qualification and simulator smoke runs receive unique attempt paths from the
controller. Their earlier outputs are retained; qualification JSON reports are
also copied to the directory downloaded back to Windows.

The simulator smoke script evaluates one official initial state of the first
task in each suite using the released policy. Its four outcomes establish
compatibility only. Confirmatory results still require every prescribed task
and state. A numerical qualification report is not a training-cost profile.

## 5. Select batching, then measure cost

```powershell
& $lambda run --script scripts/lambda/profile.sh --commit $commit --max-hours 4 --execute
```

The batch selector runs two complete optimizer updates per candidate, including
optimizer/EMA allocation. It searches power-of-two microbatches dividing 256,
keeps 10% memory headroom, and tries gradient checkpointing only if microbatch 1
cannot satisfy the requirement without it. Unknown failures abort; only a
recognized memory failure permits a smaller candidate. Its short profiles are
explicitly ineligible as cost measurements.

The selected microbatch and checkpointing setting apply to every arm. The
profile phase then measures ten warmup and 100 complete batch-256 updates for
each arm, plus checkpoint publication time/size. It saves diagnostic profile
checkpoints, which cannot be resumed or evaluated as completed research runs.
Before reusing a batch selection, both profiling and training verify its source,
container, checkpoint, dataset manifest, and all three normalization assets.
Repeated profiling uses a new attempt directory. The completed reports are
copied to `data/experiments/reports/profiles/<config>.json` for cost review.

Use `configs/potential/measured_budget.example.json` as the measured-cost input.
It intentionally contains nulls and `measured_gpu=false`, so it cannot authorize
a run unchanged. Populate its update times from retained profile reports.
Populate per-arm/K/suite evaluation durations from actual rollout/timing
measurements, allowing the full task timeout. Do not extrapolate weak policies
from quick successful reference episodes. Set the checkpoint interval to
`max(1, min(5000, floor(3600 / mean_update_seconds)))` and use the measured
publication time per save. Include incurred costs and remaining storage/tax
allowances in `fixed_cost_usd`.

Inside a user-launched shell phase, the decision command is:

```bash
bash scripts/lambda/run.sh model --execute python scripts/potential_preflight.py \
  --profile /data/experiments/measured_budget.json --select-study \
  --hourly-rate-usd "$SCALAR_PI_HOURLY_RATE_USD" --budget-usd 2000 \
  --require-profile --report /data/experiments/reports/budget.json
```

This chooses the complete three-seed study, then the complete seed-42 pilot,
or neither. It does not shorten 30,000-update exposure. The report is downloaded
to the private Windows `reports/<session-name>` directory on session completion.
Regenerate the decision before subsequent research sessions, listing completed
training checkpoints and evaluation-suite manifests; their actual artifacts
are validated before subtracting completed work. Never report planned work as
completed merely to make the estimate fit.

## 6. Training and exact resume

Use the downloaded measured decision to authorize a research session:

```powershell
& $lambda run --stage research --script scripts/lambda/train.sh --commit $commit `
  --budget-report PATH_TO_DOWNLOADED_BUDGET_JSON --max-hours 8 --execute
```

The controller copies that exact decision to persistent storage at
`/data/experiments/reports/budget.json`. The script processes seed 42 first in
B0/B1/P order, then seeds 17 and 93 if the full study was selected. Each arm
starts from the same base conversion, uses effective batch 256 and the native
30,000-update recipe, and selects terminal EMA. It resumes the newest complete
owned recovery checkpoint if a bounded session ended early.
Completed runs pass byte and protocol verification before being skipped. If a
startup failed before producing any resumable checkpoint, its directory moves
to `experiments/interrupted-starts` before retrying the same seed.

The trainer stops at an optimizer boundary before the controller deadline,
reserving ten minutes for saving/cleanup. SIGTERM/SIGINT also request a boundary
stop. Checkpoints retain raw weights, EMA, optimizer, RNG, data cursor, and
protocol metadata. Resume rejects different source/image/dependencies,
optimizer settings, dataset, normalization, or numerical settings.
The ten-minute margin must cover model/data initialization and checkpoint I/O;
check those timings during qualification and increase it if necessary. The
controller's hard deadline can otherwise interrupt unsaved work. Checkpoint
cadence is read from the same measured budget report used to admit the session.

A partial run is never a completed 30k result. Profile artifacts live in a
separate directory and are rejected as research resumes.

## 7. Evaluation, latency, and analysis

First qualify each produced potential checkpoint through the actual serving
loader, using `qualify_potential.py --checkpoint-kind research --corpus
/data/policy_corpus.npz`. It uses checkpoint-owned normalization. Repeat with
the selected gradient-checkpointing flag when qualifying the training path.

Render the selected complete matrix on the host:

```bash
python3 scripts/lambda/matrix.py render --study pilot \
  --output /lambda/nfs/scalar-pi/data/experiments/pilot.json
python3 scripts/evaluate_potential.py /lambda/nfs/scalar-pi/data/experiments/pilot.json
bash scripts/lambda/run.sh evaluate --execute /lambda/nfs/scalar-pi/data/experiments/pilot.json
```

Use `--study full` for the full decision. The first evaluator command is a dry
run. Use `--resume-completed-suites` only to skip validated complete suites.
For an interrupted suite, preserve its old directory and set the manifest's
`suite_attempts` entry to a fresh attempt name. Choose replacement attempts for
infrastructure recovery, never for higher success.

Both protocols use all four suites, ten tasks/suite, 50 official initial
states/task, native camera processing, ten settling actions, action horizon 10,
replan horizon 5, environment seed 7, and native limits 220/280/300/520. K=10 is
primary; K=1 is secondary. Native success/timeouts and invalid policy actions
count as outcomes. Infrastructure errors interrupt a suite; video errors do not
change robot success.

Every suite has a provenance sidecar binding actual weight/norm hashes,
source/container identities, simulator/task/init-state assets, and inference
settings. Episode records contain its digest. All research arms must match the
shared training protocol. The released reference retains its own normalization
and is outside paired training-seed statistics.

Run isolated latency measurements with the simulator stopped:

```bash
bash scripts/lambda/run.sh model --execute python scripts/benchmark_potential.py \
  --config pi05_libero_potential --checkpoint /data/experiments/checkpoints/pi05_libero_potential/p_seed42/30000 \
  --corpus /data/policy_corpus.npz --num-steps 10 --report /data/experiments/reports/p42_k10_latency.json
```

Repeat for each checkpoint and K. Defaults are ten warmups and 100 synchronized
calls. Reports distinguish model/policy latency, median/p95, peak memory,
generated-action throughput (10/chunk latency), executed-action throughput
(5/chunk latency), and rollout round-trip latency. Equal K is not equal compute.
For the released `pi05_libero` reference, use `run.sh model-jax` so its original
JAX policy runs on the GPU.

```bash
bash scripts/lambda/run.sh model --execute python scripts/analyze_potential.py \
  /lambda/nfs/scalar-pi/data/experiments/pilot.json \
  --compare P:B1 --compare P:B0 --output /data/experiments/reports/analysis.json
bash scripts/lambda/run.sh archive --execute \
  --output /lambda/nfs/scalar-pi/data/experiments/artifact_hashes.json
```

Analysis reports individual seeds, tasks, suites, fixed-task macro averages,
and paired 95% bootstrap intervals. One-seed intervals describe evaluation
variability conditional on that checkpoint, not training-seed variability.
There is no default equivalence margin or equivalence claim from nonsignificance.
The primary contrast is P-B1 at K10; P-B0 and K1 contrasts are secondary.

## Evidence and interpretation

The inherited protocol is anchored in the pinned
[OpenPI configuration](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py)
and [LIBERO runner](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py).
[OpenVLA-OFT](https://arxiv.org/html/2502.19645v2) supports the 500-trial/suite
evaluation and separate inference measurements, but its other training choices
are not substituted for OpenPI's. Three training seeds, bootstrap settings,
memory reserve, and warmup counts are study/engineering decisions, not universal
SOTA laws.

The scalar construction is supported by
[Composing Flow-Matching Energies with Known Physics](https://arxiv.org/html/2608.18004v1#S9)
and [Equilibrium Matching](https://arxiv.org/html/2510.02300v3). Their results do
not guarantee a benefit here. Learned flow potential is not a calibrated
physical energy, reward, or safety score. Native LIBERO tasks occur in the
training dataset; report benchmark adaptation, not unseen-task or real-world
generalization. LIBERO-Plus remains a separately budgeted later extension.

Local CPU tests cover mathematics, small actual pi0.5 updates/resume, data and
evaluation validation, and mocked cloud lifecycle. They do not establish full
checkpoint parity, GPU memory fit, real rendering, convergence, or success rate.
Those gates remain explicit user-launched phases above.

Implementation validation on 2026-11-10: **268 CPU tests passed** with one upstream
`pynvml` deprecation warning. Ruff checks passed, all six Lambda shell scripts
passed Bash syntax checks, and the Windows wrapper passed its resource-free
dry run. Docker builds, CUDA qualification, full dataset decoding, and actual
LIBERO rollouts were not run on this workstation.
