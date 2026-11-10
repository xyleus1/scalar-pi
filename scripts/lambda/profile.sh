#!/usr/bin/env bash
# Select fit once, prove exact recovery, then reuse independently verified arm profiles.
set -euo pipefail
runner=(bash scripts/lambda/run.sh)
rate="${SCALAR_PI_HOURLY_RATE_USD:?The controller supplies the live quote}"
attempt="${SCALAR_PI_ATTEMPT:?Launch this phase through the Lambda controller}"
deadline="${SCALAR_PI_DEADLINE_UNIX:?The controller supplies the absolute deadline}"
host=/lambda/nfs/scalar-pi/data
qualification=/data/experiments/reports/gpu_qualification.json
recovery=/data/experiments/reports/recovery.json
inputs=(--checkpoint /data/weights/pi05_base_torch --data-manifest /data/libero_manifest.json
    --qualification-report "$qualification")
selection="$host/experiments/batch_selection.json"
if [[ ! -f "$selection" ]]; then
    "${runner[@]}" model --execute python scripts/select_potential_batch.py "${inputs[@]}" \
        --hourly-rate "$rate" --output-root "/data/experiments/batch_selection/$attempt" \
        --report /data/experiments/batch_selection.json --deadline-unix "$deadline" --execute
fi
"${runner[@]}" model --execute python scripts/select_potential_batch.py "${inputs[@]}" \
    --verify-report /data/experiments/batch_selection.json
recovery_inputs=("${inputs[@]}" --selection /data/experiments/batch_selection.json --corpus /data/policy_corpus.npz)
if [[ -f "$host/experiments/reports/recovery.json" ]] && python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("passed") is False else 1)' "$host/experiments/reports/recovery.json"; then
    if ! python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("error",{}).get("type") in ("TimeoutError","TimeoutExpired") else 1)' "$host/experiments/reports/recovery.json"; then
        echo 'Recovery qualification failed. Inspect its retained report and identify the fix before another attempt; no automatic numerical retry is allowed.' >&2
        exit 1
    fi
    mv -- "$host/experiments/reports/recovery.json" "$host/experiments/reports/recovery.failed-$attempt.json"
fi
if [[ -f "$host/experiments/reports/recovery.json" ]]; then
    "${runner[@]}" model --execute python scripts/qualify_potential_recovery.py "${recovery_inputs[@]}" \
        --verify-report "$recovery"
else
    "${runner[@]}" model --execute python scripts/qualify_potential_recovery.py "${recovery_inputs[@]}" \
        --output-root "/data/experiments/recovery/$attempt" --report "$recovery" \
        --deadline-unix "$deadline" --execute
fi
batch="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["microbatch_size"])' "$selection")"
checkpointing="$(python3 -c 'import json,sys; print("--gradient-checkpointing" if json.load(open(sys.argv[1]))["gradient_checkpointing"] else "--no-gradient-checkpointing")' "$selection")"
reports="$host/experiments/reports/profiles"
mkdir -p "$reports"
copy_attempt_reports() {
    mkdir -p "$reports/attempts/$attempt"
    for config in pi05_libero_b0 pi05_libero_b1 pi05_libero_potential; do
        source="$host/experiments/profiles/$config/$attempt/profile.json"
        if [[ -f "$source" ]]; then cp "$source" "$reports/attempts/$attempt/$config.json"; fi
    done
}
trap copy_attempt_reports EXIT
for config in pi05_libero_b0 pi05_libero_b1 pi05_libero_potential; do
    canonical="/data/experiments/reports/profiles/$config.json"
    if [[ -f "$reports/$config.json" ]]; then
        "${runner[@]}" model --execute python scripts/select_potential_batch.py "${inputs[@]}" \
            --recovery-report "$recovery" --config "$config" --verify-profile "$canonical"
        continue
    fi
    "${runner[@]}" model --execute python scripts/train_potential.py "$config" \
        --exp-name "$attempt" --weights /data/weights/pi05_base_torch \
        --output-root /data/experiments/profiles --data-manifest /data/libero_manifest.json \
        --qualification-report "$qualification" --recovery-report "$recovery" \
        --microbatch-size "$batch" "$checkpointing" --seed 42 \
        --profile-warmup 10 --profile-updates 100 --hourly-rate "$rate" --deadline-unix "$deadline"
    profile="/data/experiments/profiles/$config/$attempt/profile.json"
    "${runner[@]}" model --execute python scripts/select_potential_batch.py "${inputs[@]}" \
        --recovery-report "$recovery" --config "$config" --verify-profile "$profile"
    cp "$host/experiments/profiles/$config/$attempt/profile.json" "$reports/$config.json"
done
cp "$selection" "$host/experiments/reports/batch_selection.json"
