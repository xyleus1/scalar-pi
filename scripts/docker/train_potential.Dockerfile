# Adapted from the official OpenPI serving image at 215abfb217dbac7d5f1273282331b9b1866c0479.
# The lockfile's Torch wheel supplies its own CUDA runtime; the base-image name
# does not describe torch.version.cuda. A compatible host NVIDIA driver is needed.
# Build: docker build -f scripts/docker/train_potential.Dockerfile -t potential-pi .
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

LABEL org.opencontainers.image.title="Potential pi05 research environment"
LABEL org.opencontainers.image.base.upstream="215abfb217dbac7d5f1273282331b9b1866c0479"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential clang ca-certificates libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/.venv \
    GIT_LFS_SKIP_SMUDGE=1 \
    JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    WANDB_MODE=offline \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY packages/ ./packages/
RUN uv venv --python 3.11.9 /.venv
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY examples/ ./examples/
COPY configs/ ./configs/
COPY tests/potential/ ./tests/potential/
COPY docs/potential_pi05.md ./docs/potential_pi05.md
RUN --mount=type=cache,target=/root/.cache/uv bash scripts/setup_potential.sh

ENV PATH="/.venv/bin:${PATH}"
# Inspection only by default. Training/evaluation must be invoked explicitly.
CMD ["python", "scripts/potential_preflight.py"]
