# Evaluation timing

Calibration measures the actual model and simulator paths before planning a new run. It does not change the benchmark's task limits, seeds, number of trials, or acceptance criteria.

The specification covers eight policy settings (B0/B1/P/reference at K=1/K=10) and the native simulator in all 40 scenes. Research timing uses explicitly marked diagnostic profile checkpoints. Those checkpoints are not eligible for the final success-rate comparison.

## Prepare and run

Start with the qualification, batch selection, recovery report, and completed four-episode compatibility run from [Reproduction](reproduction.md). In the same shell, produce one diagnostic profile for each arm using the selected settings. These are separate from the nine full training runs:

```bash
mkdir -p "$REPORTS/profiles"
for config in pi05_libero_b0 pi05_libero_b1 pi05_libero_potential; do
  model python scripts/train_potential.py "$config" \
    --exp-name timing --weights "$BASE" --output-root "$DATA/experiments/profiles" \
    --data-manifest "$DATA/libero_manifest.json" --microbatch-size "$MICROBATCH" \
    "$CHECKPOINTING" --seed 42 --profile-warmup 10 --profile-updates 100 \
    --hourly-rate "$HOURLY_RATE_USD" \
    --qualification-report "$REPORTS/gpu_qualification.json" \
    --recovery-report "$REPORTS/recovery.json"
  cp "$DATA/experiments/profiles/$config/timing/profile.json" "$REPORTS/profiles/$config.json"
done
bash scripts/docker/run_potential.sh prepare-calibration --execute --attempt timing \
  --smoke-manifest "$SCALAR_PI_ROOT/data/smoke.json"
```

The helper verifies all 110 optimizer updates and the final diagnostic checkpoint for each arm. It uses the reproduction layout: `data/weights/pi05_base_torch`, `data/experiments/batch_selection.json`, and the reports under `data/experiments/reports`. Profile evidence paths must remain readable on the host; the absolute paths in the commands above are mounted unchanged inside containers. Use a fresh root for a different source/image/profile set rather than replacing existing reports.

The helper writes `$REPORTS/evaluation_calibration_spec.json`. Preview its commands, then execute through the runner, which supplies the actual model and simulator image identities:

```bash
python3 scripts/calibrate_potential_evaluation.py \
  --spec "$REPORTS/evaluation_calibration_spec.json" --output "$REPORTS/evaluation_calibration.json"
bash scripts/docker/run_potential.sh calibrate --execute \
  --spec "$REPORTS/evaluation_calibration_spec.json" --output "$REPORTS/evaluation_calibration.json"
```

Each completed component retains its inputs, elapsed time, and result hash. Add `--resume` to reuse completed components with matching inputs. Use a new aggregate output path; do not overwrite previous measurements. Pass the completed aggregate to `potential_matrix.py render --timing-report` before executing the full matrix. For a different file layout, the explicit specification in `configs/potential/evaluation_calibration.example.json` is available instead of the profile-aware helper.

## Interpretation

Separate warmed synchronized model latency from end-to-end client latency, startup, and simulator time. Record the GPU family, runtime, precision, and concurrency with every timing result. Hardware-dependent timings are not directly interchangeable.

The scalar action path requires an input gradient, so a matched number of Euler steps does not imply equal inference cost. Calibrate both direct and potential experts. Preserve the original scientific timeouts when measuring throughput.

The timing estimator uses the benchmark's allowed execution lengths rather than assuming every task ends early. Success-rate estimates belong to the evaluation dataset, not the calibration report.
