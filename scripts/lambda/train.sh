#!/usr/bin/env bash
# User-launched, resumable study. Budget selection precedes comparative outcomes.
set -euo pipefail
root=/lambda/nfs/scalar-pi
data="$root/data"
report="$data/experiments/reports/budget.json"
study="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("budget",d)["selected_study"])' "$report")"
case "$study" in
    full) seeds=(42 17 93) ;;
    pilot) seeds=(42) ;;
    *) echo 'No affordable complete study has been selected.' >&2; exit 1 ;;
esac
selection="$data/experiments/batch_selection.json"
bash scripts/lambda/run.sh model --execute python scripts/select_potential_batch.py \
    --checkpoint /data/weights/pi05_base_torch --data-manifest /data/libero_manifest.json \
    --verify-report /data/experiments/batch_selection.json
batch="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["microbatch_size"])' "$selection")"
checkpointing="$(python3 -c 'import json,sys; print("--gradient-checkpointing" if json.load(open(sys.argv[1]))["gradient_checkpointing"] else "--no-gradient-checkpointing")' "$selection")"
for seed in "${seeds[@]}"; do
    for group in b0 b1 p; do
        config="pi05_libero_$group"
        if [[ "$group" == p ]]; then config=pi05_libero_potential; fi
        run="${group}_seed${seed}"
        directory="$data/experiments/checkpoints/$config/$run"
        if [[ -f "$directory/30000/potential_metadata.json" ]]; then
            bash scripts/lambda/run.sh model --execute python scripts/lambda/verify_training.py \
                --checkpoint "$directory/30000" --config "$config" --seed "$seed" \
                --selection /data/experiments/batch_selection.json \
                --weights /data/weights/pi05_base_torch --data-manifest /data/libero_manifest.json
            continue
        fi
        remaining="$(python3 -c 'import os,time; print(max(0,(float(os.environ["SCALAR_PI_DEADLINE_UNIX"])-time.time()-600)/3600))')"
        if ! python3 -c 'import sys; sys.exit(0 if float(sys.argv[1])>0 else 1)' "$remaining"; then exit 0; fi
        save_every="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("budget",d)["checkpoint_interval_updates"][sys.argv[2].upper()])' "$report" "$group")"
        latest="$(python3 -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); ds=[d for d in p.glob("*") if d.name.isdigit() and (d/"training_state.pt").is_file()]; print(max(ds,key=lambda d:int(d.name)) if ds else "")' "$directory")"
        if [[ -d "$directory" && -z "$latest" ]]; then
            # A failed startup has no resumable update. Keep its evidence before
            # retrying the same seed; never overwrite an existing attempt.
            interrupted="$data/experiments/interrupted-starts/$config/${run}-${SCALAR_PI_ATTEMPT:?}"
            [[ ! -e "$interrupted" ]] || { echo 'Interrupted-startup destination exists.' >&2; exit 1; }
            mkdir -p "$(dirname "$interrupted")"
            mv -- "$directory" "$interrupted"
        fi
        source=(--weights /data/weights/pi05_base_torch)
        if [[ -n "$latest" ]]; then source=(--resume "$latest"); fi
        bash scripts/lambda/run.sh model --execute python scripts/train_potential.py "$config" \
            --exp-name "$run" "${source[@]}" --output-root /data/experiments/checkpoints \
            --data-manifest /data/libero_manifest.json --microbatch-size "$batch" "$checkpointing" \
            --seed "$seed" --save-every "$save_every" --max-hours "$remaining" \
            --hourly-rate "${SCALAR_PI_HOURLY_RATE_USD:?}"
        if [[ ! -f "$directory/30000/potential_metadata.json" ]]; then exit 0; fi
    done
done
