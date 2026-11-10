#!/usr/bin/env bash
# Infrastructure only: no model, dataset, training or simulator.
set -euo pipefail
root=/lambda/nfs/scalar-pi
attempt="${SCALAR_PI_ATTEMPT:?Use the Lambda controller}"
python3 scripts/lambda/canary.py probe --root "$root" \
    --report "$root/data/experiments/reports/canary/$attempt.json"
