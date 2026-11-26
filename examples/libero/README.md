# LIBERO Benchmark

For Scalar-Pi's matched experiment and reported results, use the
[study reproduction guide](../../docs/reproduction.md). The commands below describe
the underlying OpenPI simulator integration.

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

The Python 3.8 simulator uses `requirements.lock.txt`, which includes the dependencies
of this example, the pinned LIBERO submodule, and `openpi-client`. Syncing the three
input files directly omits transitive dependencies. The Docker build checks the
installed dependency graph and imports the dependency modules without starting a
simulator or GPU workload.

To regenerate the lock from the repository root, keep the existing direct pins and
CUDA 11.3 index:

```bash
uvx --from uv==0.5.1 uv pip compile \
    examples/libero/requirements.txt third_party/libero/requirements.txt \
    packages/openpi-client/pyproject.toml --python-version 3.8 \
    --build-constraint examples/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match -o examples/libero/requirements.lock.txt
```

The build constraint retains the existing `setuptools==75.3.0` pin in isolated
source-package builds, including `pathtools`. The image installs CMake for the
native `egl-probe` build. The LIBERO input is from submodule
commit `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the repository root):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.lock.txt --build-constraint examples/libero/requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install --no-deps -e packages/openpi-client -e third_party/libero
uv pip check
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Upstream reference results

The following table is retained from OpenPI's example. It is not the measured
Scalar-Pi result; the study's reference and matched controls are reported in the
[Scalar-Pi README](../../README.md#results).

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
