# Research containers

Scalar-Pi keeps the model and LIBERO simulator in separate pinned environments. The model image uses Python 3.11.9 and the repository lockfile; the simulator uses Python 3.8 and its existing dependency lock. Both need a compatible NVIDIA driver and NVIDIA Container Toolkit on a Linux host.

The provider-neutral runner uses Docker as the current user. Configure Docker access before running it. It creates no cloud resources.

```bash
export SCALAR_PI_ROOT="$PWD/.research"
bash scripts/docker/run_potential.sh inspect
bash scripts/docker/run_potential.sh build --execute
```

Build from a clean committed checkout. The runner saves both image identities and an image archive under the selected root. The model image is defined in `scripts/docker/train_potential.Dockerfile`; the simulator is defined in `examples/libero/Dockerfile`.

## Commands

Every command previews its action unless `--execute` follows the phase name.

| Phase | Purpose |
|---|---|
| `inspect` | Print source, storage, and image names |
| `build` | Build or restore the pinned model and simulator images |
| `model` | Run a model command with offline dataset access |
| `model-online` | Prepare explicitly requested model/data downloads |
| `model-jax` | Enable CUDA JAX for the released reference |
| `server` | Start the existing policy server |
| `simulator` | Run the native LIBERO client with EGL |
| `evaluate` | Execute a prepared evaluation matrix |
| `calibrate` | Execute a prepared timing specification |

```bash
bash scripts/docker/run_potential.sh model --execute python scripts/potential_preflight.py
```

The checkout is mounted at `/app`, persistent data at `/data`, and the selected root at the same absolute host path. These mappings let the model and simulator resolve the same manifest and checkpoint paths. The wrapper supplies actual image identities and retains per-process container cleanup on exit.

Research serving uses Torch with CPU JAX initialization. The released `pi05_libero` reference uses CUDA JAX. Keep this distinction when constructing commands outside the wrapper.

For a complete experiment, follow [Reproduction](reproduction.md). Upstream examples remain available for other robot integrations, but their generic deployment recipes are separate from the Scalar-Pi study protocol.
