# Scalar-Pi

[![Research tests](https://github.com/xyleus1/scalar-pi/actions/workflows/research-cpu.yml/badge.svg)](https://github.com/xyleus1/scalar-pi/actions/workflows/research-cpu.yml)
[![Code checks](https://github.com/xyleus1/scalar-pi/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/xyleus1/scalar-pi/actions/workflows/pre-commit.yml)

**Nima Kamali** · 22 November 2026 · [Methods](docs/potential_pi05.md) · [Reproduction](docs/reproduction.md)

**Paper:** [Scalar-potential action experts in pi0.5](https://github.com/xyleus1/scalar-pi/releases/download/v0.2.0/scalar-pi-paper.pdf) · [Editable source (v0.2.0)](https://github.com/xyleus1/scalar-pi/releases/download/v0.2.0/scalar-pi-source.zip) · [Curated data (unchanged v0.1.0)](https://github.com/xyleus1/scalar-pi/releases/download/v0.1.0/scalar-pi-data.zip)

Scalar-Pi studies a scalar-potential action expert for π0.5. It retains the pretrained vision-language backbone and action transformer, forms a scalar from the noisy action chunk and the transformer's output, and uses its full input gradient as the flow velocity:

$$\Psi_\theta(x,t,c)=\langle x,f_\theta(x,t,c)\rangle,\qquad v_\theta=\nabla_x\Psi_\theta.$$

The comparison holds the data, training exposure, backbone, precision, and evaluation settings fixed. A zero-output direct-field control separates the scalar constraint from the effect of resetting the output layer. The implementation extends [OpenPI](https://github.com/Physical-Intelligence/openpi); it keeps the `openpi` Python namespace and checkpoint conventions.

## Results

The completed experiment contains **nine training runs and 40,000 selected LIBERO episodes**: three research arms, three training seeds, K=1 and K=10 Euler steps, and a separately evaluated released policy. Each checkpoint/step setting covers 40 tasks with 50 initial states per task.

| Model | K=10 success |
|---|---:|
| B1: reset direct-field control | 91.08% |
| P: scalar-potential expert | 91.38% |
| Released π0.5 LIBERO reference | 95.85% |

The primary P−B1 difference is **+0.30 percentage points**, with a descriptive paired 95% interval of **[−1.27, +1.75]**. This does not establish an improvement. The released policy uses a different training protocol and is a reference, not the matched control.

All 800 selected task logs passed row validation. Cross-device numerical qualification is a separate limitation: A100 PCIe did not pass the original H100 comparison, and A100 SXM remains unqualified. Mixed-device estimates are therefore descriptive. The study did not measure motion smoothness or establish a speed advantage. See the [expanded paper](https://github.com/xyleus1/scalar-pi/releases/tag/v0.2.0) for the complete comparisons and hardware breakdown.

The v0.2.0 paper adds eight figures, a broader literature discussion, and five proposed follow-up studies. It adds **no experiments or new outcomes**: the 40,000-record dataset and reported statistics remain those of v0.1.0. The [paper guide](paper/README.md) provides figure-generation commands and primary-source evidence ledgers; other papers' results and proposed studies are explicitly separated from this study's measurements.

## Installation

The model environment uses Linux, Python 3.11.9, and the pinned dependencies in `uv.lock`. LIBERO runs in a separate Python 3.8 environment. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git LFS, then:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/xyleus1/scalar-pi.git
cd scalar-pi
bash scripts/setup_potential.sh
```

This installs the model environment and runs CPU research tests. It does not download model weights or demonstrations. For GPU execution, use the [container instructions](docs/docker.md) and [data preparation guide](docs/reproduction.md).

## Use the research tools

The three configurations are `pi05_libero_b0`, `pi05_libero_b1`, and `pi05_libero_potential`. The [reproduction guide](docs/reproduction.md) covers dataset preparation, training prerequisites, terminal checkpoints, and evaluation.

Preview the evaluation matrix without starting a model or simulator:

```bash
python scripts/evaluate_potential.py configs/potential/libero_matrix.example.json
```

Replace the template checkpoint paths and environment prefixes before execution. The original matrix preserves seeds 17, 42, and 93, paired rollout/noise seed 7, and both solver settings.

The release also includes a CPU-only replay of the reported statistics. Download and extract the [immutable v0.1.0 data ZIP](https://github.com/xyleus1/scalar-pi/releases/download/v0.1.0/scalar-pi-data.zip), then run:

```bash
python paper/reproduce_descriptive.py --dataset /path/to/data --check-only
python paper/reproduce_descriptive.py --dataset /path/to/data --output reproduced.json
```

These commands need Python 3.10+ and its standard library, not a GPU or model checkpoint. The full replay reproduces all 80 suite cells and eight original paired contrasts. See [paper/README.md](paper/README.md) for PDF and figure builds.

## Repository guide

- `src/openpi/`: model, training, evaluation, and checkpoint code.
- `scripts/`: data preparation, training, qualification, evaluation, and analysis entrypoints.
- `configs/potential/`: experiment and calibration templates.
- `examples/libero/`: pinned simulator and benchmark client.
- `tests/potential/`: research regression tests, including full-gradient and checkpoint contracts.
- `paper/`: editable paper, figures, and reproduction sources.

## Citation and licenses

Use [CITATION.cff](CITATION.cff) to cite this project. Code is distributed under [Apache 2.0](LICENSE), with the existing [Gemma terms](LICENSE_GEMMA.txt) and third-party notices retained. The original paper and curated results use [CC BY 4.0](LICENSE-RESEARCH). Pretrained model weights and source datasets retain their own terms.

Maintained by **Nima Kamali** ([xyleus1](https://github.com/xyleus1)). See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions.
