#!/usr/bin/env bash
# User-launched fit selection and complete effective-batch optimizer timing.
set -euo pipefail
runner=(bash scripts/lambda/run.sh)
rate="${SCALAR_PI_HOURLY_RATE_USD:?The Lambda controller supplies the live hourly quote}"
attempt="${SCALAR_PI_ATTEMPT:?Launch this phase through the Lambda controller}"
selection=/lambda/nfs/scalar-pi/data/experiments/batch_selection.json
if [[ ! -f "$selection" ]]; then
    "${runner[@]}" model --execute python scripts/select_potential_batch.py \
        --checkpoint /data/weights/pi05_base_torch --data-manifest /data/libero_manifest.json \
        --hourly-rate "$rate" --output-root "/data/experiments/batch_selection/$attempt" \
        --report /data/experiments/batch_selection.json --execute
fi
"${runner[@]}" model --execute python scripts/select_potential_batch.py \
    --checkpoint /data/weights/pi05_base_torch --data-manifest /data/libero_manifest.json \
    --verify-report /data/experiments/batch_selection.json
batch="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["microbatch_size"])' "$selection")"
checkpointing="$(python3 -c 'import json,sys; print("--gradient-checkpointing" if json.load(open(sys.argv[1]))["gradient_checkpointing"] else "--no-gradient-checkpointing")' "$selection")"
for config in pi05_libero_b0 pi05_libero_b1 pi05_libero_potential; do
    "${runner[@]}" model --execute python scripts/train_potential.py "$config" \
        --exp-name "$attempt" --weights /data/weights/pi05_base_torch \
        --output-root /data/experiments/profiles --data-manifest /data/libero_manifest.json \
        --microbatch-size "$batch" "$checkpointing" --seed 42 \
        --profile-warmup 10 --profile-updates 100 --hourly-rate "$rate"
done
mkdir -p /lambda/nfs/scalar-pi/data/experiments/reports/profiles
for config in pi05_libero_b0 pi05_libero_b1 pi05_libero_potential; do
    cp "/lambda/nfs/scalar-pi/data/experiments/profiles/$config/$attempt/profile.json" \
        "/lambda/nfs/scalar-pi/data/experiments/reports/profiles/$config.json"
done
cp "$selection" /lambda/nfs/scalar-pi/data/experiments/reports/batch_selection.json
