# Scalar-potential pi0.5: architecture, provenance, and execution

This research extension changes the **whole action expert's output parameterization** while retaining the pretrained pi0.5 network. It does not replace the action expert with a small scalar head trained in isolation.

The source baseline is OpenPI commit `215abfb217dbac7d5f1273282331b9b1866c0479`. The checked-in implementation and CPU tests are not evidence of GPU fit, training convergence, or LIBERO performance. No real-model training budget is established until a complete GPU update and rollout profile have been measured. The current local Windows machine has no NVIDIA GPU available for that qualification.

## 1. Architectural graph and scientific intervention

```text
images + language -> original PaliGemma -> original conditioning cache
                                                   |
noisy normalized actions x [B,10,32]                |
        |                                          |
        v                                          v
original input linear 32 -> 1024 -> original 18-layer action expert
                                      ^            |
                                      |            v
time -> original sinusoid -> time MLP -> AdaRMS     original output linear
                                                   1024 -> 32
                                                       |
                                                       v
                                            raw field f(x,t,c) [B,10,32]
                                                       |
                                  original x ----------+ elementwise product
                                                       |
                                                       v
                                         sum over horizon AND dimensions
                                                       |
                                                       v
                                               one scalar Psi [B]
                                                       |
                                                       v
                                           FULL input gradient dPsi/dx
                                                       |
                                                       v
                                      velocity = f + Jacobian(f).T @ x
                                                       |
                                                       v
                                      original reverse-time Euler update
```

The action expert has width 1024, depth 18, and MLP width 4096. The time path remains the original sinusoidal embedding followed by Linear, SiLU, Linear, SiLU and the native adaptive RMS conditioning. The image-language model and all original action dependencies remain intact. These dimensions and operations are inherited from [OpenPI's model configuration](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/gemma.py) and [PyTorch model](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models_pytorch/pi0_pytorch.py).

The potential is

\[
\Psi_\theta(x,\tau,c)=\sum_{h,d}x_{hd}f_{\theta,hd}(x,\tau,c),\qquad
v_\theta=\nabla_x\Psi_\theta=f_\theta+J_f^\mathsf{T}x.
\]

The sum is not an average: dividing by 320 would change velocity scale. There is one scalar per example, not one independent energy per motor. Each raw output can depend on the complete action chunk, so the final potential retains cross-time and cross-coordinate interactions. Differentiation must include the input projection, every action-dependent expert layer, and the output projection. Detaching `f` would remove the Jacobian term and invalidate the experiment.

### Architectural decision ledger

| Choice | Evidence and interpretation |
|---|---|
| Dot product of input with backbone output | Direct precedent in *Composing Flow-Matching Energies with Known Physics*, Appendix B, and EqM, Section 3.2. |
| Retain the pi transformer instead of introducing a U-Net | Pretrained host architecture; isolates the scalar constraint without simultaneously replacing capacity or conditioning. This exact combination is our research adaptation. |
| Zero final linear projection for potential initialization | Adaptation of the PDE paper's zero-initialized final convolution. An identically reset vector baseline is our causal control. |
| Train complete action expert plus input/output/time projections | Allows action-dependent features and derivatives to adapt. Head-only training would test a different restricted architecture. |
| Freeze VLM in all research groups | Explicit study/compute choice, not a mathematical requirement of conservativeness or the official full fine-tuning recipe. |
| Keep native smooth GELU/SiLU and AdaRMS | Preserves pretrained computation while supporting higher derivatives. No new activation or normalization is introduced. |
| Keep 32 model channels and horizon 10 | Official LIBERO configuration and checkpoint-compatible per-token projections. Only seven action dimensions are executed. |
| Preserve native masks, positions, context and time | Prevents unrelated architectural changes from explaining the result. |
| Frozen BF16 VLM, FP32 expert, eager kernels | Engineering reference for higher derivatives, not a published universal precision standard. Must be GPU-qualified. |
| Architecture metadata in each research checkpoint | Required because direct and potential weight tensors have the same shapes. Loading the wrong interpretation can silently deploy the wrong policy. |

The fixed graph uses the official LIBERO setting `discrete_state_input=False`. For pi0.5 this means numerical proprioception is not fed into the network; its learned conditioning is images and language. The state remains in observation plumbing. Adding state tokens is a separate experiment.

## 2. Closest scalar architectures and literature review

| Source | Scalar architecture | What to reuse |
|---|---|---|
| [Composing Flow-Matching Energies with Known Physics](https://arxiv.org/html/2608.18004v1), Appendix B | ADM U-Net, smooth SiLU, zero output convolution, dot with noisy input, full input gradient | Dot-potential construction and initialization precedent. Its uniform time distribution and PDE sampling counts are not pi defaults. |
| [EnergyFlow](https://arxiv.org/html/2605.00623v1), Appendix C | Temporal U-Net, global temporal pooling, scalar MLP, action gradient | Higher-order autograd pattern and a shared-backbone comparison. It uses denoising score matching, not our pi flow target. |
| [EnergyFlow released model](https://github.com/sotaagi/EnergyFlow/blob/53cf227026185965f6726f8c8da3ccc6f8477183/model.py) | Mean pooling, spectrally normalized scalar MLP with no final bias, Softplus | Practical training/inference graph handling. The paper says Mish, whereas this code uses Softplus; do not silently merge their descriptions. |
| [Equilibrium Matching](https://arxiv.org/html/2510.02300v3), Sections 3.2 and 5.3 | Dot-product or negative squared-norm scalar from an existing backbone | Matched-parameter architecture comparison and initialization sensitivity. Explicit scalar variants underperformed its unrestricted field. |
| [Energy Matching](https://arxiv.org/html/2504.10612v5), Appendix D | U-Net followed by a small ViT producing a time-independent scalar | An alternative architecture; its equilibrium and contrastive objectives are outside this intervention. |
| [Velociraptor](https://openreview.net/pdf?id=tdIwEN0fCi), Sections 4–5 | Separate scalar adapter to a frozen vector model plus a reweighted residual | Evidence that eliminating nonconservative residuals is not automatically beneficial; report actual compute as well as step count. |

The review ledger must record paper version, code commit, actual scalar graph, initialization, objective, solver, tuning policy, seeds, metrics and limitations. Check actual pi modifications such as [GeCO](https://arxiv.org/html/2603.17834v2) and [pi0-EqM](https://arxiv.org/html/2605.23128v1) when positioning novelty: they study time-unconditional fields, not this exact time-conditioned scalar architecture.

This is not the first scalar-energy robot policy or the invention of potential flow matching. The study asks whether the constraint is beneficial in a pretrained pi0.5 expert under controlled initialization and compute.

## 3. Objective, sampling, and limitations

Retain OpenPI's convention:

\[
x_\tau=(1-\tau)A+\tau\epsilon,\quad u=\epsilon-A,\quad
\epsilon\sim\mathcal N(0,I),\quad
\tau=.001+.999\,\mathrm{Beta}(1.5,1).
\]

Train with the original mean squared velocity error over all 32 action channels. Keep independent action/noise pairing, native time conditioning and unit loss weighting. Changing to the PDE paper's uniform time distribution would introduce a second intervention. Inference starts at tau=1 and uses negative Euler increments to reach tau=0.

For the independent Gaussian path, the population regression optimum is conservative. That fact does not guarantee a finite pretrained model will optimize better. The derived energy readout is

\[
E_\theta(x,\tau,c)=\frac{\|x\|^2+2(1-\tau)\Psi_\theta(x,\tau,c)}{2\tau}+b(\tau,c).
\]

This is singular at tau=0 and has an unidentified context-dependent offset. It is not automatically a calibrated reward, cross-state confidence, or physical energy. A time-conditioned flow is not an autonomous descent process: do not stop merely because velocity is small or continue descending after the integration endpoint.

The dot form fixes `Psi(0)=0`, but does not force linearity, convexity, or independent action coordinates. Conservativeness is defined in normalized coordinates; unequal physical-unit rescaling changes the associated metric. Padding accounts for 25 of 32 channels, so report physical-coordinate error separately from full-space error.

## 4. Experimental recipe and comparison groups

| Config | Initialization | Velocity |
|---|---|---|
| `pi05_libero_b0` | Pretrained action trunk and pretrained output projection | Direct field |
| `pi05_libero_b1` | Same pretrained trunk; zero output weight and bias | Direct field |
| `pi05_libero_potential` | Identical initial tensors to B1 | Full gradient of dot potential |

B1 and P start at the same zero velocity. The head can update immediately; zero trunk gradients on the first update are expected. Trunk gradients must appear after the head begins learning. This reset control is our experimental design, not a claimed standard published pi protocol.

Use the reference [pi05_libero training configuration](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py#L744):

- Full official LIBERO training set, with no invented 90/10 demonstration split.
- Official base checkpoint; do not initialize the research groups from the already LIBERO-finetuned checkpoint.
- 30,000 optimizer updates and effective batch 256.
- AdamW, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `1e-10`.
- Global gradient norm clipping at 1.0.
- 10,000-update linear warmup to `5e-5`, then constant `5e-5`.
- EMA decay 0.999; predeclared terminal EMA checkpoint.
- Native tokenizer, image processing, action normalization, horizon 10 and replan horizon 5.
- Three paired training seeds, with the exact seed values recorded in the run manifest.

The frozen VLM, scalar output interpretation, precision and cache path are declared departures from upstream training. Preserve effective batch by accumulation; a microbatch of one requires 256 microbatches per optimizer update. A short `--updates` run is a smoke test or explicitly limited-budget experiment, not the reference result.

### Evaluation protocol

Ten Euler steps are the official OpenPI sampler default. They are a reference setting, not a universal optimum. [When One Step Is Enough](https://arxiv.org/html/2610.30735v1) motivates the mandatory comparison at **K=1 and K=10** on the same trained checkpoints, initial states and model-noise schedule.

- All four suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.
- All 40 tasks; 50 official initial states per task.
- Official settling steps, camera orientation, action transformations and per-suite step limits.
- Same environment seeds and deterministic per-query noise pairing across groups.
- Report every training seed, suite success, overall success and paired uncertainty.
- Report synchronized model latency and end-to-end client latency; an input-gradient evaluation is more expensive than a direct field evaluation.
- Retain per-episode JSONL records, failure reasons and checkpoint identity.
- Three groups by three seeds by two solver settings by 2,000 episodes gives **36,000 research rollouts**. The supplied matrix adds the released reference at both K values, giving **40,000 total episodes**.

All 40 task identities occur in the training data. Do not describe the result as unseen-task generalization. Curl/Hessian checks validate construction; task success validates behavior. A loss-only gain does not establish a control gain. A nonsignificant difference is not evidence of equivalence.

## 5. Canonical environment

Use Linux for the model environment: Ubuntu 22.04, Python 3.11.9, the repository `uv.lock`, Torch 2.7.1 and Transformers 4.53.2 with OpenPI's replacement files. Keep the simulator in its existing Python 3.8 environment/container. Do not upgrade one component independently.

```bash
bash scripts/setup_potential.sh
```

This installs the locked environment, applies the exact transformer replacements, audits the stack and runs mathematical tests. It does not download robot data or weights, launch cloud resources, or validate full-model GPU training.

Equivalent model-container build:

```bash
docker build -f scripts/docker/train_potential.Dockerfile -t potential-pi .
docker run --rm --gpus all -v "$PWD:/app" potential-pi \
  python scripts/potential_preflight.py --strict-stack --require-cuda
```

The base image follows the official recipe. Torch's lock-resolved CUDA runtime is distinct from the version in that image's name; record `torch.version.cuda`, cuDNN and the host driver at execution. No Docker build or CUDA run has been established merely by checking in this recipe.

## 6. Data and checkpoint staging

Use persistent storage and explicitly select cache roots:

```bash
export OPENPI_DATA_HOME="$PWD/data/openpi-cache"
export HF_LEROBOT_HOME="$PWD/data/lerobot"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline
```

Download using the existing providers, and pin the resolved dataset revision in a manifest. The preparation utility requires an immutable revision, hashes the dataset files and writes the trainer's manifest. It performs no download without the explicit `--download` flag. Resolve and record a full dataset commit, then use it in this command:

```bash
uv run --no-sync python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('physical-intelligence/libero').sha)"
uv run --no-sync python scripts/prepare_potential_data.py \
  --revision FULL_DATASET_COMMIT_FROM_ABOVE \
  --dataset-root "$HF_LEROBOT_HOME/physical-intelligence/libero" \
  --output data/libero_manifest.json --download
uv run --no-sync python -c "from openpi.shared.download import maybe_download; print(maybe_download('gs://openpi-assets/checkpoints/pi05_base'))"
```

The source data currently advertises 1,693 episodes, 273,465 frames and 40 tasks. Verify downloaded metadata and retain file hashes rather than assuming a future repository head has identical content. Preserve dataset and pretrained-model license notices.

Convert with the original direct `pi05_libero` configuration and FP32 conversion. The converter constructs FP32 weights before loading the source parameters; research initialization is applied later by the trainer.

```bash
uv run --no-sync python examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base" \
  --config-name pi05_libero \
  --output-path checkpoints/pi05_base_torch \
  --precision float32
```

Audit the conversion key/shape report and hash the resulting checkpoint. Successful file creation alone does not establish conversion parity. The action-expert constructor must be FP32 before weights are loaded. The upstream converter's generic `config.json` is not architecture metadata for a trained potential policy.

Compute quantile statistics with the official script on the full training data, once, then copy the same small statistics file to the other research config asset locations:

```bash
uv run --no-sync python scripts/compute_norm_stats.py --config-name pi05_libero_b0
uv run --no-sync python - <<'PY'
from pathlib import Path
from shutil import copy2
source = Path("assets/pi05_libero_b0/physical-intelligence/libero/norm_stats.json")
for config in ("pi05_libero_b1", "pi05_libero_potential"):
    target = Path("assets") / config / "physical-intelligence/libero/norm_stats.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    copy2(source, target)
PY
```

## 7. Preflight, profiling, training and resume

Inspect prerequisites before a GPU smoke test:

```bash
uv run --no-sync python scripts/potential_preflight.py \
  --checkpoint checkpoints/pi05_base_torch --checkpoint-kind base \
  --dataset "$HF_LEROBOT_HOME/physical-intelligence/libero" \
  --norm-stats assets/pi05_libero_b0/physical-intelligence/libero/norm_stats.json \
  --strict-stack --require-cuda --require-assets \
  --report data/preflight.json
```

For a produced research checkpoint, omit `--checkpoint-kind base`; `potential_metadata.json` is required. Missing architecture metadata must never be inferred from weight shapes.

Run the executable full-size GPU correctness diagnostic before training. It loads the converted base, exercises the actual potential action expert, checks finite double-backward parameter gradients, immutable caches, frozen VLM gradients and both zero-output controls, then samples repeatedly at K=1 and K=10 with identical explicit noise:

```bash
uv run --no-sync python scripts/qualify_potential.py \
  --checkpoint checkpoints/pi05_base_torch --checkpoint-kind base \
  --report data/gpu_qualification.json
```

If gradient checkpointing will be used, repeat with `--gradient-checkpointing` and a new report filename. Use `--checkpoint-kind research` to check a trained potential checkpoint with mandatory architecture metadata. This diagnostic uses full-shape synthetic images, token masks and actions, so its JSON records `synthetic_input: true` and explicitly excludes training-cost and benchmark validity. The default two warmup and ten recorded calls per K are adjustable diagnostic counts, not literature-derived evaluation settings. Inspect its allocated/reserved memory series; it intentionally assigns no arbitrary memory-leak pass threshold. No full-size GPU qualification has been run on the current CPU-only machine.

Profile each group in a separate experiment directory. The following performs actual complete optimizer updates; it is not an inference-only timing loop:

```bash
uv run --no-sync python scripts/train_potential.py pi05_libero_potential \
  --exp-name profile_potential --weights checkpoints/pi05_base_torch \
  --microbatch-size 1 --seed 42 --profile-warmup 10 --profile-updates 100 \
  --hourly-rate ACTUAL_PROVIDER_RATE
```

Repeat for B0 and B1. Inspect peak allocated/reserved memory, complete update time, finite loss and gradients, frozen-weight checks and output metadata. Choose the common microbatch based on the potential model's measured fit, then use that same microbatch, accumulation, effective batch and numerical settings for all groups.

Reference training example:

```bash
uv run --no-sync python scripts/train_potential.py pi05_libero_potential \
  --exp-name potential_seed42 --weights checkpoints/pi05_base_torch \
  --microbatch-size 1 --seed 42 --data-manifest data/libero_manifest.json
```

Use the corresponding B0/B1 config names with the same seeds and data. An explicit `--updates 2` is suitable only for a smoke test. A wall-clock `--max-hours` stop produces an incomplete run and cannot be reported as the full 30k result.

Resume from a complete trusted checkpoint with the same config and experiment identity:

```bash
uv run --no-sync python scripts/train_potential.py pi05_libero_potential \
  --exp-name potential_seed42 --resume /absolute/path/to/recovery/checkpoint \
  --microbatch-size 1 --seed 42 --data-manifest data/libero_manifest.json
```

The checkpoint must retain optimizer, EMA, RNG, scheduler/data position, normalization and architecture metadata. Do not treat weights-only reload as an exact resume.

The trainer keeps the two most recent owned recovery checkpoints by default (`--keep-last-checkpoints 2`) and preserves the terminal reference checkpoint. Retention changes storage use, not which checkpoint is selected for evaluation.

## 8. Evaluation environment and serving

Initialize the pinned simulator submodule and use the existing simulator Dockerfile. Keep this container separate from the model container:

```bash
git submodule update --init --recursive
docker build -f examples/libero/Dockerfile -t potential-libero .
```

Serve a terminal potential checkpoint with an explicit solver count:

```bash
uv run --no-sync python scripts/serve_policy.py --env LIBERO --num-steps 10 \
  policy:checkpoint --policy.config pi05_libero_potential \
  --policy.dir /absolute/path/to/terminal/checkpoint
```

Run the official simulator client, with the opt-in research record fields, in the Python 3.8 container. The client records K but does not override the server's sampler: set K on both sides and check metadata. Repeat at K=1, with the same checkpoint and episode/noise identities. Full-suite orchestration and analysis commands are documented with the evaluation scripts; never count a task-limited diagnostic run as the full benchmark.

The checked-in matrix is a template: replace every trained checkpoint path with the actual terminal EMA checkpoint, and set its server/client command prefixes to the two environments. Its default simulator prefix expects `examples/libero/.venv/bin/python`; to use containers, replace that prefix with the corresponding `docker run` invocation and mounted paths. Inspect the exact commands before execution:

```bash
python scripts/evaluate_potential.py configs/potential/libero_matrix.example.json
python scripts/evaluate_potential.py configs/potential/libero_matrix.example.json --execute
python scripts/evaluate_potential.py configs/potential/libero_matrix.example.json --execute --resume-completed-suites
python scripts/analyze_potential.py configs/potential/libero_matrix.example.json \
  --compare P:B1 --compare P:B0 --output results/potential/analysis.json
```

The first command is a dry run. Use the resume command only after an interrupted matrix: it skips a suite only after validating its complete log against the exact expected cell, rejects partial/duplicate/error/mismatched logs and validates the full matrix at completion. Partial suite results must be retained for diagnosis and rerun into a fresh results location; they are never silently counted as complete. Primary matrices require B0, B1, P and the released reference; reduced matrices must be explicitly diagnostic. Analysis must reject incomplete, duplicate or mismatched episode records. The evaluator's latency summaries include cold requests; a separate warmed, synchronized timing run is needed for steady-state latency claims. The implemented 90% confidence-interval equivalence criterion is not a computed TOST p-value.

## 9. Budget gate

The total cloud ceiling is $2,000. Do not assert that the full reference recipe fits before measuring it. A gradient-accumulated batch of 256 is substantially more expensive than timing a microbatch.

`potential_preflight.py --profile` accepts a combined measurement record with:

```text
measured_gpu: true
model_kind: pi05_full
hardware: actual GPU model
measurement_source: retained train/evaluation profiler logs
warmup_optimizer_updates: positive count
timed_optimizer_updates: at least 100
effective_batch_size: 256
optimizer_updates: 30000
training_seeds: planned seed count
seconds_per_optimizer_update: mapping from each planned group to measured seconds
evaluation_episodes: complete planned rollout count
episodes_per_gpu_hour: measured complete rollout throughput
fixed_cost_usd: setup, storage and other already-budgeted costs
```

There are intentionally no invented timing values in this document. Compose the record from measured per-group training profiles and actual episode throughput. Synthetic/CPU profiles are rejected. The audit adds 20% compute headroom and compares the result with the supplied ceiling:

```bash
uv run --no-sync python scripts/potential_preflight.py \
  --profile data/measured_budget.json --hourly-rate-usd ACTUAL_PROVIDER_RATE \
  --budget-usd 2000 --require-profile --report data/budget_gate.json
```

If the reference study does not fit, record that limitation and reduce the experiment matrix explicitly. Do not silently lower training exposure, omit hard tasks, or call a short run a standard reproduction.

## 10. Verification and reporting

```bash
uv run --no-sync pytest -q tests/potential
```

The mathematical tests cover the full Jacobian term, FP64 gradient/double-gradient checks, zero-head learnability, an analytic Gaussian mixture, the reverse-time sign, the one-step conditional-mean limit, batch independence, Hessian symmetry, contraction precision and inference graph detachment. They use standard PyTorch and pytest without JAX/model downloads.

Seven CPU integration tests using small dimensions of the actual pi0.5 implementation have also passed: action-expert double backward and frozen VLM parameters; BF16 frozen VLM with FP32 expert double backward; immutable conditioning cache; checkpointed versus ordinary suffix output/gradient parity; local inference gradient handling; joint versus cached vector-path parity; and matched zero-head controls that begin learning. These validate the code paths at small scale. They do not establish full-size GPU fit, kernel behavior, training convergence or simulator success.

Local validation on 2026-11-06: **121 tests passed**, Ruff checks passed, and `git diff --check` passed. The CPU test environment used WSL Linux, Python 3.11.18, Torch 2.7.1+cpu, Transformers 4.53.2 with the exact OpenPI replacements, and NumPy 1.26.4. The official configuration/data-loader import and collation smoke checks and trainer/server CLI checks also passed. This CPU environment is separate from the locked GPU/container recipe. The local JUnit report is `research_outputs/cpu_tests.xml`; the prerequisite audit is `research_outputs/local_preflight.json`. The audit correctly reports CUDA unavailable and full-model readiness false. Docker build, full pretrained checkpoint conversion, full-size CUDA qualification, training, and real LIBERO rollouts remain unexecuted.

Before claiming real-model readiness, additionally verify:

1. Repeat cache-path and conditioning parity on the full pretrained GPU model.
2. Check full-size pi expert parameter gradients through the second derivative.
3. Verify frozen VLM weights remain unchanged during a GPU optimizer update.
4. Qualify checkpoint interpretation, save/reload and exact resume with the full model and real data, beyond the small CPU fixtures.
5. Stable repeated GPU inference memory.
6. Official released-policy rollout qualification.
7. Complete paired K=1/K=10 results and honest infrastructure/model failure accounting.

Report source, environment and data revisions; all seeds; terminal training exposure; every rollout outcome; task success with uncertainty; measured memory, latency and cloud cost. The meaningful result can be positive, negative or inconclusive. Zero curl alone is an implementation fact, not evidence that robot behavior improved.
