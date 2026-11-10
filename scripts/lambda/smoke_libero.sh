#!/usr/bin/env bash
# User-launched four-episode compatibility check, not a benchmark score.
set -euo pipefail
attempt="${SCALAR_PI_ATTEMPT:?Launch this phase through the Lambda controller}"
matrix="/lambda/nfs/scalar-pi/data/experiments/diagnostic-$attempt.json"
python3 scripts/lambda/matrix.py render --diagnostic --attempt "$attempt" --output "$matrix"
bash scripts/lambda/run.sh evaluate --execute "$matrix"
