#!/usr/bin/env bash
# Reuse independent conversion-parity and real-policy qualification evidence.
set -euo pipefail
runner=(bash scripts/lambda/run.sh)
stage=(python scripts/lambda/staging.py run)
attempt="${SCALAR_PI_ATTEMPT:?Launch this phase through the Lambda controller}"
output=/data/experiments/qualification
receipts=/data/experiments/reports/staging
preserve_failed() {
    local file="/lambda/nfs/scalar-pi$1" receipt="/lambda/nfs/scalar-pi$2"
    if [[ -f "$file" && ! -f "$receipt" ]]; then
        if [[ "$file" != *.json ]] || ! python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("error",{}).get("type")=="TimeoutError" else 1)' "$file"; then
            echo "Qualification output has no passing producer receipt: $file. Inspect and identify the cause before retrying." >&2
            exit 1
        fi
        mv -- "$file" "$file.deadline-$attempt"
    fi
}
for name in pi05_base pi05_libero; do
    norms=/app/assets/pi05_libero_b0/physical-intelligence/libero/norm_stats.json
    if [[ "$name" == pi05_libero ]]; then
        norms=/data/openpi-cache/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json
    fi
    for backend in jax torch; do
        phase=model
        checkpoint="/data/weights/${name}_torch"
        if [[ "$backend" == jax ]]; then
            phase=model-jax
            checkpoint="/data/openpi-cache/openpi-assets/checkpoints/$name"
        fi
        preserve_failed "$output/${name}_$backend.npz" "$receipts/${name}_$backend.json"
        "${runner[@]}" "$phase" --execute "${stage[@]}" --stage "${name}_$backend" \
            --receipt "$receipts/${name}_$backend.json" --parameter backend="$backend" \
            --input checkpoint="$checkpoint" --input norms="$norms" --input corpus=/data/policy_corpus.npz \
            --output export="$output/${name}_$backend.npz" -- python scripts/check_potential_parity.py export \
            --backend "$backend" --checkpoint "$checkpoint" --norm-stats "$norms" \
            --corpus /data/policy_corpus.npz --output "$output/${name}_$backend.npz"
    done
    parity="/data/experiments/reports/${name}_parity.json"
    preserve_failed "$parity" "$receipts/${name}_parity.json"
    "${runner[@]}" model --execute "${stage[@]}" --stage "${name}_parity" \
        --receipt "$receipts/${name}_parity.json" --input jax="$output/${name}_jax.npz" \
        --input torch="$output/${name}_torch.npz" --output report="$parity" -- \
        python scripts/check_potential_parity.py compare --jax-output "$output/${name}_jax.npz" \
        --torch-output "$output/${name}_torch.npz" --report "$parity"
done
qualification=/data/experiments/reports/gpu_qualification.json
preserve_failed "$qualification" "$receipts/full-qualification.json"
"${runner[@]}" model --execute "${stage[@]}" --stage full-qualification \
    --receipt "$receipts/full-qualification.json" --input base=/data/weights/pi05_base_torch \
    --input parity=/data/experiments/reports/pi05_base_parity.json --input corpus=/data/policy_corpus.npz \
    --input norms=/app/assets/pi05_libero_b0/physical-intelligence/libero/norm_stats.json \
    --output report="$qualification" -- python scripts/qualify_potential.py \
    --checkpoint /data/weights/pi05_base_torch --checkpoint-kind base \
    --parity-report /data/experiments/reports/pi05_base_parity.json --corpus /data/policy_corpus.npz \
    --norm-stats /app/assets/pi05_libero_b0/physical-intelligence/libero/norm_stats.json --report "$qualification"
