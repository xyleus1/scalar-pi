#!/usr/bin/env bash
# Assemble actual measurements and explicit accounting, then lock full/pilot scope.
set -euo pipefail
attempt="${SCALAR_PI_ATTEMPT:?Launch this phase through the Lambda controller}"
rate="${SCALAR_PI_HOURLY_RATE_USD:?The controller supplies the live quote}"
forecast="${SCALAR_PI_FORECAST_RATE_USD:?The controller supplies the forecast quote}"
report="/data/experiments/reports/budgets/$attempt.json"
bash scripts/lambda/run.sh model --execute python scripts/lambda/measured_budget.py \
    --inputs /data/experiments/budget_inputs.json --output "$report" --forecast-rate "$forecast"
bash scripts/lambda/run.sh model --execute python scripts/potential_preflight.py \
    --profile "$report" --select-study --study-lock /data/experiments/reports/study_decision.json \
    --hourly-rate-usd "$rate" --budget-usd 2000 --require-profile \
    --report /data/experiments/reports/budget.json
