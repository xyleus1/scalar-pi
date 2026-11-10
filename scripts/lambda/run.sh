#!/usr/bin/env bash
# Thin container commands for the existing research entrypoints. Preview by default.
set -euo pipefail
phase="${1:-inspect}"
shift || true
execute=false
if [[ "${1:-}" == --execute ]]; then execute=true; shift; fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
root="${SCALAR_PI_ROOT:-/lambda/nfs/scalar-pi}"
data="$root/data"
commit="$(git -C "$repo" rev-parse HEAD)"
model="scalar-pi-model:$commit"
simulator="scalar-pi-libero:$commit"
docker=(sudo -n docker)

show() { printf '%q ' "$@"; printf '\n'; }
if [[ "$phase" == inspect ]]; then
    printf 'commit=%s\nroot=%s\nmodel=%s\nsimulator=%s\n' "$commit" "$root" "$model" "$simulator"
    exit 0
fi
if ! $execute; then
    show bash "$repo/scripts/lambda/run.sh" "$phase" --execute "$@"
    printf 'Preview only: no container, training, or simulation started.\n'
    exit 0
fi
[[ -z "$(git -C "$repo" status --porcelain)" ]] || { echo 'Commit source changes before execution.' >&2; exit 1; }
mkdir -p "$data/experiments" "$root/images"
cd "$repo"

if [[ "$phase" == build ]]; then
    git submodule update --init --recursive
    "${docker[@]}" build -f scripts/docker/train_potential.Dockerfile -t "$model" .
    "${docker[@]}" build -f examples/libero/Dockerfile -t "$simulator" .
    # Instances are disposable; retain the exact images on the project filesystem.
    archive="$root/images/$commit.tar.gz"
    "${docker[@]}" image save "$model" "$simulator" | gzip -1 > "$archive.tmp"
    mv -- "$archive.tmp" "$archive"
    exit 0
fi

if ! "${docker[@]}" image inspect "$model" "$simulator" >/dev/null 2>&1; then
    [[ -f "$root/images/$commit.tar.gz" ]] || { echo 'Run the build phase first.' >&2; exit 1; }
    gzip -dc "$root/images/$commit.tar.gz" | "${docker[@]}" image load
fi
model_id="$("${docker[@]}" image inspect --format '{{.Id}}' "$model")"
simulator_id="$("${docker[@]}" image inspect --format '{{.Id}}' "$simulator")"
export SCALAR_PI_IMAGE_ID="$model_id"
export SCALAR_PI_MODEL_IMAGE="$model_id"
export SCALAR_PI_SIMULATOR_IMAGE="$simulator_id"

if [[ "$phase" == evaluate ]]; then
    exec python3 scripts/evaluate_potential.py "$@" --execute
fi
if [[ "$phase" == archive ]]; then
    # Logs and final checkpoints already live on the persistent filesystem.
    # Export their hashes without duplicating multi-GB checkpoints.
    exec python3 scripts/lambda/matrix.py archive --root "$root" "$@"
fi

common=(run --rm --init -i --gpus all --network host --stop-timeout 120
    -v "$repo:/app" -v "$data:/data" -v "$root:$root" -w /app
    -e OPENPI_DATA_HOME=/data/openpi-cache -e HF_LEROBOT_HOME=/data/lerobot -e HF_HOME=/data/hf
    -e WANDB_MODE=offline -e XLA_PYTHON_CLIENT_PREALLOCATE=false
    -e GIT_CONFIG_COUNT=2 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/app
    -e GIT_CONFIG_KEY_1=safe.directory -e GIT_CONFIG_VALUE_1=/app/third_party/libero
    -e "SCALAR_PI_IMAGE_ID=$model_id" -e "SCALAR_PI_MODEL_IMAGE=$model_id"
    -e "SCALAR_PI_SIMULATOR_IMAGE=$simulator_id")
jax_platform=cpu
managed_container=false
case "$phase" in
    model|model-online|model-jax) ;;
    server)
        # The released JAX reference needs GPU JAX; research Torch serving does not.
        for argument in "$@"; do
            if [[ "$argument" == pi05_libero ]]; then jax_platform=cuda; fi
        done
        common+=(--name scalar-pi-policy)
        managed_container=true
        set -- python scripts/serve_policy.py "$@"
        ;;
    simulator)
        common+=(--name scalar-pi-simulator -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl
            -e MUJOCO_EGL_DEVICE_ID=0 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics)
        model_id="$simulator_id"
        managed_container=true
        set -- /.venv/bin/python examples/libero/main.py "$@"
        ;;
    *) echo "Unknown phase: $phase" >&2; exit 2 ;;
esac
if [[ "$phase" == model-jax ]]; then jax_platform=cuda; fi
common+=(-e "JAX_PLATFORMS=$jax_platform")
if [[ "$phase" != model-online ]]; then
    common+=(-e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1)
fi
if $managed_container; then
    cid_dir="$(mktemp -d)"
    cleanup() {
        if [[ -s "$cid_dir/id" ]]; then
            cid="$(cat "$cid_dir/id")"
            if [[ "$cid" =~ ^[a-f0-9]{64}$ ]]; then
                "${docker[@]}" stop --time 5 "$cid" >/dev/null 2>&1 || true
                "${docker[@]}" rm -f "$cid" >/dev/null 2>&1 || true
            fi
        fi
        rm -f -- "$cid_dir/id"
        rmdir -- "$cid_dir"
    }
    trap cleanup EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    "${docker[@]}" "${common[@]}" --cidfile "$cid_dir/id" "$model_id" "$@" &
    child=$!
    wait "$child"
    exit $?
fi
exec "${docker[@]}" "${common[@]}" "$model_id" "$@"
