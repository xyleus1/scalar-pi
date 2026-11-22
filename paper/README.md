# Paper and descriptive reproduction

**Nima Kamali and [Mert Turna](https://github.com/mmturna).** *Scalar-Potential Action Experts in a Pretrained pi0.5 Policy:
A Controlled LIBERO Study* (2026).

The completed observed grid contains 40,000 episodes, 800 tasks, 20 checkpoint/K
combinations and 80 checkpoint/K/suite cells. All selected tasks have retained
original row-validation and closed-execution evidence. The original phase
archives separately account for 16,000 episodes. The primary descriptive P minus
B1 contrast at K=10 is +0.30 percentage points (95% paired bootstrap interval
[-1.267, +1.750]). A100 PCIe failed the unchanged H100 numerical gate and A100 SXM
rollouts remain unqualified. These mixed-hardware results do not establish
confirmatory superiority or hardware equivalence.

## Reproduce the results without a GPU

Download and extract `scalar-pi-data.zip` from the versioned release.
Python 3.10+ and its standard library are sufficient. From the repository root:

```bash
python paper/reproduce_descriptive.py --dataset /path/to/data --check-only
python paper/reproduce_descriptive.py --dataset /path/to/data --output reproduced.json
```

The first command verifies all public dataset checksums, unique task/state keys,
the pinned original statistical function bodies, and every saved 80-cell and
800-task summary. The second also reruns the original eight paired contrasts
(10,000 resamples, random seed 0) and requires exact agreement with all saved
estimates and confidence intervals. It can take several minutes. It does not
rerun a simulator or convert task validation into hardware acceptance.

The public dataset is an explicit curated derivative, with new checksums and
source digests. Private machine identities, filesystem paths, command lines and
cloud-control receipts are omitted. Scientific outcomes, failure categories,
pairing identities, solver settings, actual runtime cohorts, and four consumed
retry selections are retained. The four original infrastructure rows remain
separate and do not enter the selected 40,000 outcomes. Raw private originals are
unchanged. No per-step action/state trajectories or physical jerk measurements
were recorded.

## Build the PDF

Install a TeX distribution providing `pdflatex` and `bibtex`, with geometry,
amsmath, amssymb, graphicx, booktabs, array, xcolor and hyperref. Then:

```bash
python -m pip install -r paper/requirements-render.txt
python paper/build.py --output scalar-pi-paper.pdf
```

The build regenerates the original design/coverage figures and descriptive
tables, then compiles the manuscript. Intermediate files stay in `paper/build/`.
The bundled figures/tables are also available without running Matplotlib.

## Attribution and provenance

Credit Nima Kamali and Mert Turna when citing the original paper and results under CC BY 4.0. See `../LICENSE-RESEARCH`.
Code and the retained OpenPI source snapshots: Apache-2.0; see `../LICENSE`.
Third-party model/data terms remain unchanged. The scientific worker commit is
`a9c0bdbf5f20356246f25ec303c51e466fa4c3a3`. `reproducibility/sources.json` records
the original source hashes and the explicitly projected scientific manifest.
`../results/study-v0.1.0.json` contains the saved descriptive statistics.
