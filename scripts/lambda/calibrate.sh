#!/usr/bin/env bash
# Complete native timing diagnostics; retain one exact spec across bounded sessions.
set -euo pipefail
root=/lambda/nfs/scalar-pi
reports="$root/data/experiments/reports"
: "${SCALAR_PI_ATTEMPT:?Launch this phase through the Lambda controller}"
: "${SCALAR_PI_DEADLINE_UNIX:?The controller supplies the absolute deadline}"
arguments=()
if [[ -n "${SMOKE_MANIFEST:-}" ]]; then arguments+=(--smoke-manifest "$SMOKE_MANIFEST"); fi
bash scripts/lambda/run.sh model --execute python scripts/lambda/calibration_spec.py "${arguments[@]}"
if [[ -f "$reports/evaluation_calibration.json" ]]; then
    echo 'Reused verified calibration aggregate and component evidence.'
    exit 0
fi
bash scripts/lambda/run.sh calibrate --execute \
    --spec "$reports/evaluation_calibration_spec.json" \
    --output "$reports/evaluation_calibration.json" --resume
