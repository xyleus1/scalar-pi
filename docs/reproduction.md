# Reproducing Scalar-Pi

There are two paths: replay the released statistics on a CPU, or rerun the model experiment. The [methods](potential_pi05.md) define the scientific settings; the commands below use the provider-neutral tools in this repository.

## Replay the published results

Download the [current editable source](https://github.com/xyleus1/scalar-pi/releases/latest/download/scalar-pi-source.zip) and the [unchanged v0.1.1 dataset](https://github.com/xyleus1/scalar-pi/releases/download/v0.1.1/scalar-pi-data.zip). Verify each archive against the checksums in its own release, then extract them beside each other. Paper revisions use the same dataset. From the extracted source directory:

```bash
python paper/reproduce_descriptive.py --dataset ../data --check-only
python paper/reproduce_descriptive.py --dataset ../data --output reproduced-contrasts.json
```

Use the actual extracted data directory if your archive tool adds another parent folder. The replay uses the Python standard library and the pinned statistical sources. It checks data identity and reproduces all eight contrasts with 10,000 bootstrap resamples and seed 0. It does not upgrade the hardware qualification of the observations.

The editable paper has its own build instructions in `paper/README.md`. Model checkpoints are not included in the public data release.

## Model and simulator environments

Use Linux with a compatible NVIDIA driver, Docker with NVIDIA Container Toolkit, Git LFS, and Python 3.11 on the host. The model image uses Python 3.11.9; the pinned LIBERO image uses Python 3.8. Build from a clean committed checkout:

```bash
git submodule update --init --recursive
export SCALAR_PI_ROOT="$PWD/.research"
bash scripts/docker/run_potential.sh build --execute
```

The runner records actual image identities and provides the cache, EGL, and precision environment expected by the research tools. It uses `$SCALAR_PI_ROOT/data` as persistent storage, mounted at `/data` inside containers. See [Docker](docker.md) for its commands.

For compactness, the examples below use these shell functions:

```bash
model() { bash scripts/docker/run_potential.sh model --execute "$@"; }
online() { bash scripts/docker/run_potential.sh model-online --execute "$@"; }
jax_model() { bash scripts/docker/run_potential.sh model-jax --execute "$@"; }
DATA="$SCALAR_PI_ROOT/data"
REPORTS="$DATA/experiments/reports"
BASE="$DATA/weights/pi05_base_torch"
REFERENCE="$DATA/weights/pi05_libero_torch"
NORMS="assets/pi05_libero_b0/physical-intelligence/libero/norm_stats.json"
REFERENCE_NORMS="$REFERENCE/assets/physical-intelligence/libero/norm_stats.json"
```

Keep this shell open for the following commands. Output paths use the absolute host root, which is also mounted at the same path in each container. This lets host-side calibration verify the paths recorded by the trainer.

## Prepare the data and base weights

The study pins the `physical-intelligence/libero` dataset revision to `a4336d589d589045d1c56423ffdf3b88a0e19b1f`. It contains 1,693 demonstrations, 273,465 frames, and 40 tasks. Download and record the file hashes:

```bash
online python scripts/prepare_potential_data.py \
  --revision a4336d589d589045d1c56423ffdf3b88a0e19b1f \
  --dataset-root /data/lerobot/physical-intelligence/libero \
  --output "$DATA/libero_manifest.json" --download
online python -c "from openpi.shared.download import maybe_download; maybe_download('gs://openpi-assets/checkpoints/pi05_base'); maybe_download('gs://openpi-assets/checkpoints/pi05_libero')"
online python -c "from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer()"
model python examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /data/openpi-cache/openpi-assets/checkpoints/pi05_base \
  --config-name pi05_libero --output-path "$BASE" \
  --precision float32
model python examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /data/openpi-cache/openpi-assets/checkpoints/pi05_libero \
  --config-name pi05_libero --output-path "$REFERENCE" \
  --precision float32
```

The research arms start from the base model. The converted released weights are used for the hybrid-cache qualification; the benchmark's separate released reference uses the original JAX checkpoint. Preserve upstream model and dataset terms.

Compute normalization once with the original batch-256 iterator. Its 273,408 processed windows omit the final 57 windows, matching the upstream behavior:

```bash
model python scripts/compute_norm_stats.py --config-name pi05_libero_b0
for config in pi05_libero_b1 pi05_libero_potential; do
  mkdir -p "assets/$config/physical-intelligence/libero"
  cp assets/pi05_libero_b0/physical-intelligence/libero/norm_stats.json \
    "assets/$config/physical-intelligence/libero/norm_stats.json"
done
model python scripts/prepare_potential_data.py \
  --manifest "$DATA/libero_manifest.json" --audit \
  --audit-output "$DATA/libero_audit.json" --corpus-output "$DATA/policy_corpus.npz"
```

The audit checks decoded windows, episode boundaries, transforms, and action padding. The fixed corpus contains one observation per task for numerical tests.

## Qualification and training

The trainer requires reports for the current source, images, and assets. Reuse a report only when its recorded inputs match. CPU tests do not replace these full-model checks.

First produce the FP32 conversion comparison and the separate hybrid-cache oracle report:

```bash
jax_model python scripts/check_potential_parity.py export --backend jax \
  --checkpoint "$DATA/openpi-cache/openpi-assets/checkpoints/pi05_base" \
  --norm-stats "$NORMS" --corpus "$DATA/policy_corpus.npz" \
  --output "$REPORTS/base-jax.npz"
model python scripts/check_potential_parity.py export --backend torch \
  --checkpoint "$BASE" --norm-stats "$NORMS" --corpus "$DATA/policy_corpus.npz" \
  --output "$REPORTS/base-torch.npz"
model python scripts/check_potential_parity.py compare \
  --jax-output "$REPORTS/base-jax.npz" --torch-output "$REPORTS/base-torch.npz" \
  --report "$REPORTS/conversion-parity.json"
model python scripts/check_hybrid_cache.py \
  --base-checkpoint "$BASE" --reference-checkpoint "$REFERENCE" \
  --base-norm-stats "$NORMS" --reference-norm-stats "$REFERENCE_NORMS" \
  --corpus "$DATA/policy_corpus.npz" --report "$REPORTS/hybrid-cache.json"
model python scripts/qualify_potential.py \
  --checkpoint "$BASE" --checkpoint-kind base \
  --corpus "$DATA/policy_corpus.npz" --norm-stats "$NORMS" \
  --parity-report "$REPORTS/conversion-parity.json" \
  --hybrid-report "$REPORTS/hybrid-cache.json" \
  --reference-checkpoint "$REFERENCE" --reference-norm-stats "$REFERENCE_NORMS" \
  --report "$REPORTS/gpu_qualification.json"
```

Set `HOURLY_RATE_USD` to the positive hourly GPU price for this run. The selector and profiles record this value for cost estimates; it does not change model settings. Then select the common batch and test checkpoint recovery:

```bash
: "${HOURLY_RATE_USD:?Set the actual hourly GPU price before selecting a batch}"
model python scripts/select_potential_batch.py \
  --checkpoint "$BASE" --data-manifest "$DATA/libero_manifest.json" \
  --qualification-report "$REPORTS/gpu_qualification.json" \
  --output-root "$DATA/experiments/batch-probes" \
  --report "$DATA/experiments/batch_selection.json" \
  --hourly-rate "$HOURLY_RATE_USD" --execute
model python scripts/qualify_potential_recovery.py \
  --checkpoint "$BASE" --data-manifest "$DATA/libero_manifest.json" \
  --selection "$DATA/experiments/batch_selection.json" \
  --qualification-report "$REPORTS/gpu_qualification.json" --corpus "$DATA/policy_corpus.npz" \
  --output-root "$DATA/experiments/recovery-probes" --report "$REPORTS/recovery.json" --execute
```

Use the selected common microbatch and gradient-checkpointing setting for all arms. For example, after selecting the settings from `batch_selection.json`:

```bash
MICROBATCH=$(python3 -c 'import json,os; print(json.load(open(os.environ["SCALAR_PI_ROOT"]+"/data/experiments/batch_selection.json"))["microbatch_size"])')
CHECKPOINTING=$(python3 -c 'import json,os; v=json.load(open(os.environ["SCALAR_PI_ROOT"]+"/data/experiments/batch_selection.json"))["gradient_checkpointing"]; print("--gradient-checkpointing" if v else "--no-gradient-checkpointing")')
model python scripts/train_potential.py pi05_libero_potential \
  --exp-name p_seed42 --weights "$BASE" --output-root "$DATA/experiments/checkpoints" \
  --data-manifest "$DATA/libero_manifest.json" --microbatch-size "$MICROBATCH" \
  "$CHECKPOINTING" --seed 42 \
  --qualification-report "$REPORTS/gpu_qualification.json" \
  --recovery-report "$REPORTS/recovery.json"
```

Repeat with B0/B1/P at seeds 17, 42, and 93. Use experiment names `b0_seed17`, `b1_seed17`, `p_seed17`, and the corresponding names for the other seeds so the matrix renderer resolves the terminal checkpoint paths.

The default is 30,000 updates with effective batch 256. An explicit shorter run is a diagnostic, not a completed study run. The terminal EMA is the evaluation checkpoint. An exact resume uses `--resume` in place of `--weights` and retains optimizer, EMA, RNG, sample cursor, metadata, and the same run identity.

## Evaluation

The native simulator must also pass its small compatibility run on the target hardware. A diagnostic matrix contains one released-policy episode per suite:

```bash
python3 scripts/potential_matrix.py render --root "$SCALAR_PI_ROOT" \
  --diagnostic --attempt smoke --output "$SCALAR_PI_ROOT/data/smoke.json"
bash scripts/docker/run_potential.sh evaluate --execute "$SCALAR_PI_ROOT/data/smoke.json"
```

Before full evaluation, follow [evaluation calibration](evaluation_calibration.md) to produce the measured timing report. The full runner requires measured task budgets; the four-episode diagnostic has a separate finite default. Verify each terminal checkpoint, substituting its actual config, seed, and run name in this example:

```bash
model python scripts/verify_potential_training.py \
  --checkpoint "$DATA/experiments/checkpoints/pi05_libero_potential/p_seed42/30000" \
  --config pi05_libero_potential --seed 42 \
  --selection "$DATA/experiments/batch_selection.json" --weights "$BASE" \
  --data-manifest "$DATA/libero_manifest.json" \
  --qualification-report "$REPORTS/gpu_qualification.json"
```

Then render the complete matrix:

```bash
python3 scripts/potential_matrix.py render --root "$SCALAR_PI_ROOT" \
  --study full --timing-report "$REPORTS/evaluation_calibration.json" \
  --output "$SCALAR_PI_ROOT/data/full-matrix.json"
python3 scripts/evaluate_potential.py "$SCALAR_PI_ROOT/data/full-matrix.json"
bash scripts/docker/run_potential.sh evaluate --execute "$SCALAR_PI_ROOT/data/full-matrix.json"
```

The second command previews 20 checkpoint/solver batches, 80 suites, and 40,000 episodes without loading weights. Rendering and preview do not require the timing report to exist yet; execution verifies it. The historical one-seed pilot template is retained for reproducibility and is not a substitute for this grid.

Use `--resume-completed-suites` to reuse complete validated work. Incomplete attempts remain separate; explicitly select fresh attempt IDs rather than overwriting their logs. Keep checkpoint, task, initial-state, seed, and solver identities unchanged.

For a newly qualified complete run, use the strict analysis entrypoint:

```bash
model python scripts/analyze_potential.py "$SCALAR_PI_ROOT/data/full-matrix.json" \
  --compare P:B1 --compare P:B0 --output /data/experiments/analysis.json
```

For the released mixed-device observations, use the descriptive replay above. Its evidence category differs from that of a fully hardware-qualified run. Timing measurements should be compared within hardware/runtime families; see [evaluation calibration](evaluation_calibration.md).
