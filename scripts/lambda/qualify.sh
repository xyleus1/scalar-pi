#!/usr/bin/env bash
# User-launched numerical qualification; no experiment training or LIBERO rollouts.
set -euo pipefail
runner=(bash scripts/lambda/run.sh)
attempt="${SCALAR_PI_ATTEMPT:?Launch this phase through the Lambda controller}"
output="/data/experiments/qualification/$attempt"
reports="/lambda/nfs/scalar-pi/data/experiments/reports/qualification/$attempt"
norms=/app/assets/pi05_libero_b0/physical-intelligence/libero/norm_stats.json
for name in pi05_base pi05_libero; do
    if [[ "$name" == pi05_libero ]]; then
        norms=/data/openpi-cache/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json
    fi
    "${runner[@]}" model-jax --execute python scripts/check_potential_parity.py export \
        --backend jax --checkpoint "/data/openpi-cache/openpi-assets/checkpoints/$name" \
        --norm-stats "$norms" --corpus /data/policy_corpus.npz --output "$output/${name}_jax.npz"
    "${runner[@]}" model --execute python scripts/check_potential_parity.py export \
        --backend torch --checkpoint "/data/weights/${name}_torch" \
        --norm-stats "$norms" --corpus /data/policy_corpus.npz --output "$output/${name}_torch.npz"
    "${runner[@]}" model --execute python scripts/check_potential_parity.py compare \
        --jax-output "$output/${name}_jax.npz" --torch-output "$output/${name}_torch.npz" \
        --report "$output/${name}_parity.json"
    mkdir -p "$reports"
    cp "/lambda/nfs/scalar-pi$output/${name}_parity.json" "$reports/"
done
"${runner[@]}" model --execute python scripts/qualify_potential.py \
    --checkpoint /data/weights/pi05_base_torch --checkpoint-kind base \
    --corpus /data/policy_corpus.npz \
    --norm-stats /app/assets/pi05_libero_b0/physical-intelligence/libero/norm_stats.json \
    --report "$output/gpu_qualification.json"
cp "/lambda/nfs/scalar-pi$output/gpu_qualification.json" "$reports/"
