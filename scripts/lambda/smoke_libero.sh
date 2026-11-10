#!/usr/bin/env bash
# User-launched four-episode compatibility check, not a benchmark score.
set -euo pipefail
attempt="${SCALAR_PI_ATTEMPT:?Launch this phase through the Lambda controller}"
bash scripts/lambda/run.sh model --execute python - <<'PY'
from pathlib import Path
from openpi.research.qualification import verify_qualification
verify_qualification(Path("/data/experiments/reports/gpu_qualification.json"), checkpoint=Path("/data/weights/pi05_base_torch"))
PY
matrix="/lambda/nfs/scalar-pi/data/experiments/diagnostic-$attempt.json"
python3 scripts/lambda/matrix.py render --diagnostic --attempt "$attempt" --output "$matrix"
reports="/lambda/nfs/scalar-pi/data/experiments/reports/smoke/$attempt"
mkdir -p "$reports"
cp "$matrix" "$reports/matrix.json"
bash scripts/lambda/run.sh evaluate --execute "$matrix"
