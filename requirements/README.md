# CPU scientific tests

`ci-cpu.lock` is a separate Linux x86_64 / CPython 3.11.9 environment for the
retained scientific tests, including tiny instances of the production models.
Ubuntu 24.04 satisfies its glibc requirement (at least 2.31). It does not replace
the canonical training/serving `uv.lock` or qualify a new serving runtime.

The environment retains the canonical dependency versions. The deliberate
wheel substitutions are Torch 2.7.1+cpu, torchvision 0.22.1+cpu and plain
JAX/JAXlib 0.5.3, without the CUDA extras. Transformers remains 4.53.2 and receives
the same checked-in replacement files. There are no NVIDIA, Triton or TensorFlow
packages. The LeRobot source is pinned to the same full Git commit as `uv.lock`;
registry distributions carry hashes in the CPU lock.

From the repository root on Linux, with uv 0.13.0:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends build-essential linux-libc-dev ffmpeg
uv venv --python 3.11.9 .venv-cpu
uv pip sync --python .venv-cpu/bin/python --torch-backend cpu requirements/ci-cpu.lock
uv pip install --python .venv-cpu/bin/python --no-deps --no-build-isolation -e . -e packages/openpi-client
.venv-cpu/bin/python scripts/install_transformers_patch.py
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
.venv-cpu/bin/python .github/scripts/check_cpu_environment.py
.venv-cpu/bin/python -m pytest --strict-markers -q tests/potential
```

The editable installs intentionally use `--no-deps`: installing the root project
with dependency resolution would reintroduce its canonical CUDA requirement.
The CI environment check verifies the actual wheel versions, CPU backends and
patch bytes before tests. The tests use fixtures rather than downloaded weights,
robot data or simulators. Package installation itself requires network access.

`ci-canonical-constraints.txt` records the resolved registry versions
from the canonical lock, including its Python 3.11 branch for dm-tree. To update
the CPU lock after an intentional canonical
dependency change, update those constraints and the exact roots in `ci-cpu.in`,
then run:

```bash
uv pip compile requirements/ci-cpu.in --python-version 3.11.9 \
  --python-platform x86_64-manylinux_2_31 --torch-backend cpu \
  --generate-hashes --emit-index-annotation --output-file requirements/ci-cpu.lock
```

Review the lock diff and rerun the complete suite. CPU test success establishes
the software checks exercised here, not hardware numerical eligibility or a
replacement for the original study's validation criteria.
