#!/usr/bin/env bash
# User-launched staging phase. Builds images and downloads public assets.
set -euo pipefail
runner=(bash scripts/lambda/run.sh)
"${runner[@]}" build --execute
"${runner[@]}" model-online --execute python scripts/prepare_potential_data.py \
    --revision a4336d589d589045d1c56423ffdf3b88a0e19b1f \
    --dataset-root /data/lerobot/physical-intelligence/libero \
    --output /data/libero_manifest.json --download
"${runner[@]}" model-online --execute python - <<'PY'
from openpi.shared.download import maybe_download
from openpi.models.tokenizer import PaligemmaTokenizer
for name in ("pi05_base", "pi05_libero"):
    print(maybe_download(f"gs://openpi-assets/checkpoints/{name}"))
PaligemmaTokenizer()
PY
for name in pi05_base pi05_libero; do
    "${runner[@]}" model --execute python examples/convert_jax_model_to_pytorch.py \
        --checkpoint-dir "/data/openpi-cache/openpi-assets/checkpoints/$name" \
        --config-name pi05_libero --output-path "/data/weights/${name}_torch" --precision float32
done
"${runner[@]}" model --execute python scripts/prepare_potential_data.py --manifest /data/libero_manifest.json
"${runner[@]}" model --execute python scripts/compute_norm_stats.py --config-name pi05_libero_b0
"${runner[@]}" model --execute python - <<'PY'
from pathlib import Path
from shutil import copy2
source = Path("assets/pi05_libero_b0/physical-intelligence/libero/norm_stats.json")
for name in ("pi05_libero_b1", "pi05_libero_potential"):
    destination = Path("assets") / name / "physical-intelligence/libero/norm_stats.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy2(source, destination)
PY
audit_mode=--audit
if [[ -f /lambda/nfs/scalar-pi/data/policy_corpus.npz || -f /lambda/nfs/scalar-pi/data/libero_audit.json ]]; then
    audit_mode=--verify-audit
fi
"${runner[@]}" model --execute python scripts/prepare_potential_data.py \
    "$audit_mode" --manifest /data/libero_manifest.json --audit-output /data/libero_audit.json \
    --corpus-output /data/policy_corpus.npz
