#!/usr/bin/env bash
# Install the audited OpenPI stack. This does not download weights or data.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    printf '%s\n' 'Use Linux/WSL for the canonical model environment.' >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' 'Install uv first, following https://docs.astral.sh/uv/getting-started/installation/.' >&2
    exit 1
fi

task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$task_root"
export UV_LINK_MODE=copy
export GIT_LFS_SKIP_SMUDGE=1
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

uv sync --frozen --python 3.11.9
uv run --no-sync python scripts/install_transformers_patch.py

uv run --no-sync python scripts/potential_preflight.py --strict-stack
uv run --no-sync pytest -q tests/potential
printf '%s\n' 'Environment and CPU mathematical checks completed. Full-model GPU, data, and rollout qualification remain required.'
