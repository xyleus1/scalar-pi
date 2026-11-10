#!/usr/bin/env bash
# User-launched stages. Verified producer receipts make completed work reusable.
set -euo pipefail
runner=(bash scripts/lambda/run.sh)
stage=(python scripts/lambda/staging.py run)
attempt="${SCALAR_PI_ATTEMPT:?Launch this phase through the Lambda controller}"
host=/lambda/nfs/scalar-pi/data
receipts=/data/experiments/reports/staging
"${runner[@]}" build --execute

"${runner[@]}" model-online --execute "${stage[@]}" --stage base-download \
    --receipt "$receipts/base-download.json" --parameter source=gs://openpi-assets/checkpoints/pi05_base \
    --output base=/data/openpi-cache/openpi-assets/checkpoints/pi05_base \
    --output tokenizer=/data/openpi-cache/big_vision/paligemma_tokenizer.model --allow-existing-output -- python - <<'PY'
from openpi.shared.download import maybe_download
from openpi.models.tokenizer import PaligemmaTokenizer
maybe_download("gs://openpi-assets/checkpoints/pi05_base")
PaligemmaTokenizer()
PY
"${runner[@]}" model --execute "${stage[@]}" --stage base-conversion \
    --receipt "$receipts/base-conversion.json" --parameter precision=float32 --parameter config=pi05_libero \
    --input source=/data/openpi-cache/openpi-assets/checkpoints/pi05_base \
    --output converted=/data/weights/pi05_base_torch -- python examples/convert_jax_model_to_pytorch.py \
    --checkpoint-dir /data/openpi-cache/openpi-assets/checkpoints/pi05_base \
    --config-name pi05_libero --output-path /data/weights/pi05_base_torch --precision float32

# Test scalar double backward before the data audit or reference conversion.
if [[ ! -f "$host/experiments/reports/staging/math.json" && -f "$host/experiments/reports/math_qualification.json" ]]; then
    if ! python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("error",{}).get("type")=="TimeoutError" else 1)' "$host/experiments/reports/math_qualification.json"; then
        echo 'Scalar qualification failed or lacks a publication receipt. Inspect the retained report and identify the fix; this phase will not repeat it blindly.' >&2
        exit 1
    fi
    mv -- "$host/experiments/reports/math_qualification.json" "$host/experiments/reports/math-failed-$attempt.json"
fi
"${runner[@]}" model --execute "${stage[@]}" --stage potential-math \
    --receipt "$receipts/math.json" --input base=/data/weights/pi05_base_torch \
    --output report=/data/experiments/reports/math_qualification.json -- python scripts/qualify_potential.py \
    --math-only --checkpoint /data/weights/pi05_base_torch --checkpoint-kind base \
    --report /data/experiments/reports/math_qualification.json

"${runner[@]}" model-online --execute "${stage[@]}" --stage libero-download \
    --receipt "$receipts/data.json" --parameter revision=a4336d589d589045d1c56423ffdf3b88a0e19b1f \
    --output manifest=/data/libero_manifest.json --allow-existing-output -- python scripts/prepare_potential_data.py \
    --revision a4336d589d589045d1c56423ffdf3b88a0e19b1f \
    --dataset-root /data/lerobot/physical-intelligence/libero --output /data/libero_manifest.json --download
"${runner[@]}" model --execute python scripts/prepare_potential_data.py --manifest /data/libero_manifest.json
"${runner[@]}" model --execute "${stage[@]}" --stage normalization \
    --receipt "$receipts/normalization.json" --input manifest=/data/libero_manifest.json \
    --parameter config=pi05_libero_b0 \
    --output b0=/app/assets/pi05_libero_b0/physical-intelligence/libero/norm_stats.json \
    --output b1=/app/assets/pi05_libero_b1/physical-intelligence/libero/norm_stats.json \
    --output p=/app/assets/pi05_libero_potential/physical-intelligence/libero/norm_stats.json -- bash -c '
set -euo pipefail
python scripts/compute_norm_stats.py --config-name pi05_libero_b0
for config in pi05_libero_b1 pi05_libero_potential; do
    mkdir -p "assets/$config/physical-intelligence/libero"
    cp assets/pi05_libero_b0/physical-intelligence/libero/norm_stats.json "assets/$config/physical-intelligence/libero/norm_stats.json"
done'

# Preserve incomplete export evidence before retrying the native exporter.
if [[ ! -f "$host/experiments/reports/staging/audit.json" && ( -f "$host/policy_corpus.npz" || -f "$host/libero_audit.json" ) ]]; then
    interrupted="$host/experiments/interrupted-staging/$attempt"
    mkdir -p "$interrupted"
    for file in policy_corpus.npz libero_audit.json; do
        if [[ -f "$host/$file" ]]; then mv -- "$host/$file" "$interrupted/$file"; fi
    done
fi
"${runner[@]}" model --execute "${stage[@]}" --stage decoded-data-audit \
    --receipt "$receipts/audit.json" --input manifest=/data/libero_manifest.json \
    --input norms=/app/assets/pi05_libero_b0/physical-intelligence/libero/norm_stats.json \
    --output audit=/data/libero_audit.json --output corpus=/data/policy_corpus.npz -- python scripts/prepare_potential_data.py \
    --audit --manifest /data/libero_manifest.json --audit-output /data/libero_audit.json --corpus-output /data/policy_corpus.npz
"${runner[@]}" model --execute python scripts/prepare_potential_data.py --verify-audit \
    --manifest /data/libero_manifest.json --audit-output /data/libero_audit.json --corpus-output /data/policy_corpus.npz

"${runner[@]}" model-online --execute "${stage[@]}" --stage reference-download \
    --receipt "$receipts/reference-download.json" --parameter source=gs://openpi-assets/checkpoints/pi05_libero \
    --output source=/data/openpi-cache/openpi-assets/checkpoints/pi05_libero --allow-existing-output -- \
    python -c 'from openpi.shared.download import maybe_download; maybe_download("gs://openpi-assets/checkpoints/pi05_libero")'
"${runner[@]}" model --execute "${stage[@]}" --stage reference-conversion \
    --receipt "$receipts/reference-conversion.json" --parameter precision=float32 --parameter config=pi05_libero \
    --input source=/data/openpi-cache/openpi-assets/checkpoints/pi05_libero \
    --output converted=/data/weights/pi05_libero_torch -- python examples/convert_jax_model_to_pytorch.py \
    --checkpoint-dir /data/openpi-cache/openpi-assets/checkpoints/pi05_libero \
    --config-name pi05_libero --output-path /data/weights/pi05_libero_torch --precision float32
