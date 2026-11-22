# Evaluation timing

Calibration measures the actual model and simulator paths before planning a new run. It does not change the benchmark's task limits, seeds, number of trials, or acceptance criteria.

The specification covers eight policy settings (B0/B1/P/reference at K=1/K=10) and the native simulator in all 40 scenes. Research timing uses explicitly marked diagnostic profile checkpoints. Those checkpoints are not eligible for the final success-rate comparison.

## Prepare and run

Start with the verified policy corpus, completed per-arm profile reports, and a completed four-episode released-policy compatibility run. Fill `configs/potential/evaluation_calibration.example.json` with their paths, or use the profile-aware helper:

```bash
python3 scripts/prepare_potential_calibration.py \
  --root "$SCALAR_PI_ROOT" --attempt timing \
  --smoke-manifest "$SCALAR_PI_ROOT/data/smoke.json"
```

The helper reports its output specification. Use that path for a dry run, then execute it through the container runner:

```bash
python3 scripts/calibrate_potential_evaluation.py \
  --spec PATH_TO_SPEC --output "$SCALAR_PI_ROOT/data/evaluation-timing.json"
bash scripts/docker/run_potential.sh calibrate --execute \
  --spec PATH_TO_SPEC --output "$SCALAR_PI_ROOT/data/evaluation-timing.json"
```

Each completed component retains its inputs, elapsed time, and result hash. Add `--resume` to reuse completed components with matching inputs. Use a new aggregate output path; do not overwrite previous measurements.

## Interpretation

Separate warmed synchronized model latency from end-to-end client latency, startup, and simulator time. Record the GPU family, runtime, precision, and concurrency with every timing result. Hardware-dependent timings are not directly interchangeable.

The scalar action path requires an input gradient, so a matched number of Euler steps does not imply equal inference cost. Calibrate both direct and potential experts. Preserve the original scientific timeouts when measuring throughput.

The timing estimator uses the benchmark's allowed execution lengths rather than assuming every task ends early. Success-rate estimates belong to the evaluation dataset, not the calibration report.
