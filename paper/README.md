# Scalar-potential action experts in π₀.₅

**Nima Kamali and [Mert Turna](https://github.com/mmturna)** · 22 November 2026

[Paper PDF (v0.1.3)](https://github.com/xyleus1/scalar-pi/releases/download/v0.1.3/scalar-pi-paper.pdf) · [Editable source (v0.1.3)](https://github.com/xyleus1/scalar-pi/releases/download/v0.1.3/scalar-pi-source.zip) · [Curated data (unchanged v0.1.1)](https://github.com/xyleus1/scalar-pi/releases/download/v0.1.1/scalar-pi-data.zip)

The v0.1.3 revision redraws all ten figures using a shared publication style,
with consistent model colors, readable labels, and editable vector exports. It introduces no new experiments, outcomes or statistical
estimates. The public dataset remains the immutable v0.1.1 release.

The completed observed grid contains 40,000 episodes, 800 task-evaluation runs, 20 checkpoint/K
combinations and 80 checkpoint/K/suite cells. All selected tasks have retained
original row-validation and closed-execution evidence. The original phase
archives separately account for 16,000 episodes. The primary descriptive P minus
B1 contrast at K=10 is +0.30 percentage points (95% paired bootstrap interval
[-1.267, +1.750]). A100 PCIe failed the unchanged H100 numerical gate and A100 SXM
rollouts remain unqualified. These mixed-hardware results do not establish
confirmatory superiority or hardware equivalence.

## Reproduce the results without a GPU

Download and extract [`scalar-pi-data.zip` from v0.1.1](https://github.com/xyleus1/scalar-pi/releases/download/v0.1.1/scalar-pi-data.zip).
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
amsmath, amssymb, graphicx, booktabs, array, xcolor, natbib and hyperref. Then:

```bash
python -m pip install -r paper/requirements-render.txt
python paper/build.py --output scalar-pi-paper.pdf
```

The build regenerates ten figures and the descriptive tables, then compiles
the manuscript. Intermediate files stay in `paper/build/`. The bundled figures/tables are also available
without running Matplotlib.

## Figures, literature and proposed studies

Regenerate all ten figures from the saved statistics and bundled task
descriptions:

```bash
python paper/expanded_figures.py
```

Each figure has vector PDF, editable SVG, and 300-dpi PNG versions in `figures/`.
`figure_style.py` defines the palette and physical dimensions;
`architecture_figures.py` and `result_figures.py` draw the diagrams and charts.
The [visual reference study](figure-design-notes.md) documents the primary-paper
references, Applied Compute artifacts, and the design decisions.

| File stem | Content |
| --- | --- |
| `comparison_map` | Direct, input-dot and scalar-head architecture relationships |
| `method` | Full-gradient computation and closed-loop sampling |
| `coverage` | Selected episode grid and original phase archives |
| `study_design` | Training arms, seeds, solver settings and evaluation grid |
| `success_overview` | Suite and overall success, with individual training-seed points |
| `contrast_intervals` | All eight saved paired differences and descriptive intervals |
| `task_differences` | Long-horizon task differences and Moka/mugs seed breakdowns |
| `task_heatmap_all` | All 40 tasks at both solver settings |
| `literature_evidence` | Other studies' within-paper comparisons, with distinct protocols |
| `future_studies` | Five proposed studies and their matched controls |

[`figure-data.json`](figure-data.json) contains the plotted data, source hashes,
task descriptions, captions and literature references. Figure generation does
not rerun evaluation or estimate new confidence intervals. Results from other
papers are not pooled into a cross-paper leaderboard.

The five proposals concern training-only trajectory supervision, event/waypoint
conditioning, geometric invariance, future-feature prediction, and
distributional action generation. None has been evaluated here. Their evidence
and limitations are documented in the [structural-model ledger](literature_structural.md)
and [distributional/predictive-supervision ledger](literature_distributional.md),
with versioned primary references in `references_structural.bib` and
`references_distributional.bib`. Neither being scalar nor the observed success
rates establish smoother motion, lower jerk, safety, or a mechanism for the
task-level differences.

## Attribution and provenance

Credit Nima Kamali and Mert Turna when citing the original paper and results under CC BY 4.0. See `../LICENSE-RESEARCH`.
Code and the retained OpenPI source snapshots: Apache-2.0; see `../LICENSE`.
Third-party model/data terms remain unchanged. The scientific worker commit is
`a9c0bdbf5f20356246f25ec303c51e466fa4c3a3`. `reproducibility/sources.json` records
the original source hashes and the explicitly projected scientific manifest.
`../results/study-v0.1.0.json` contains the saved descriptive statistics.
