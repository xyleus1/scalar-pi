# Structural action-model evidence ledger

Checked 2026-11-22. This is a primary-source audit for proposed extensions to
**Scalar-potential action experts in pi0.5**. These are other authors' reported
results, not additional Scalar-Pi experiments. Entries are not a common
leaderboard: training, checkpoint selection, baseline architecture, camera
conditions, and evaluation populations differ. Bibliographic authors and exact
versions are in `references_structural.bib`.

## 1. Hermite trajectory supervision

**Source:** Lv et al., [arXiv:2608.01265v2](https://arxiv.org/html/2608.01265v2),
submitted 2026-08-02, revised 2026-08-09; Sections III-C3, III-D, IV-A/B/D,
Tables II, V, VI. Citation key `lv2026hermite`.

| Reported comparison | Baseline | Hermite-Reg |
| --- | ---: | ---: |
| Four-suite LIBERO average success (%) | 95.9 | 98.7 |
| LIBERO-10/Long success (%) | 91.4 | 96.8 |

Each task has 50 rollouts, totaling 2,000 per method. The continuous models use
30,000 training steps, batch 256, horizon 10, execution window 5, and 10 denoising
steps. Table V selects auxiliary weight 10; Table VI reports 93.7% for one
segment versus 98.7% for two. These are reported point estimates, not evidence
that our potential expert gains the same improvement.

The regularized variant retains the original inference head and sampler; its
Hermite branch is training-only. Section III-D explicitly limits the seam result:
it bounds errors, and regularization provides an inductive bias rather than hard
runtime continuity. Do not claim exact continuity of independently predicted
chunks or of a scaffold plus unrestricted residual.

**Original redraw:** hidden action tokens split into the existing velocity head
and a pooled auxiliary MLP; the latter predicts endpoint/tangent coefficients,
followed by a fixed Hermite matrix and trajectory loss. Mark that branch
“training only”; use newly drawn curves, not the paper's images.

### Released head count, distinct from an implemented Scalar-Pi extension

The [official repository](https://github.com/aopolin-lv/Hermite-VLA) was inspected
at commit `f0d3b0bbcc8fd9e8da1fcb947c3cbd8744e96f2e` (Apache-2.0 code;
third-party components retain their own terms):

- [`hermite_head.py`, `HermiteContinuousHead`](https://github.com/aopolin-lv/Hermite-VLA/blob/f0d3b0bbcc8fd9e8da1fcb947c3cbd8744e96f2e/src/openpi/models/hermite_head.py#L128): mean pooling and three biased linear layers.
- [`pi0.py`, auxiliary construction](https://github.com/aopolin-lv/Hermite-VLA/blob/f0d3b0bbcc8fd9e8da1fcb947c3cbd8744e96f2e/src/openpi/models/pi0.py): input width from the expert, `tail_dim=0`, output size `K*4*6`.
- [`pi0_config.py`](https://github.com/aopolin-lv/Hermite-VLA/blob/f0d3b0bbcc8fd9e8da1fcb947c3cbd8744e96f2e/src/openpi/models/pi0_config.py) and [`training/config.py`](https://github.com/aopolin-lv/Hermite-VLA/blob/f0d3b0bbcc8fd9e8da1fcb947c3cbd8744e96f2e/src/openpi/training/config.py): width 512, two segments, six auxiliary channels, weight 10; `pi05_libero_hermite_aux` enables these defaults.

Thus the released auxiliary architecture is **1024 → 512 → 512 → 48**, with
GELU between linear layers and no auxiliary gripper branch. Its derived count is

```text
(1024 + 1)*512 + (512 + 1)*512 + (512 + 1)*48
= 524800 + 262656 + 24624 = 812080 parameters.
```

This is source arithmetic, not a measured Scalar-Pi checkpoint inventory. Our
current `gemma_300m` expert also has width 1024, so this is a concrete proposed
budget if the same head is added. It has not been added or evaluated here.

The proposed **0.19%** requires a stated denominator. From the local
`src/openpi/models/gemma.py` architecture with pi0.5 AdaRMS, the action-expert
transformer has

```text
18*(4718592 attention + 12582912 feed-forward)
+ 37*(1024*3072 + 3072 AdaRMS modulation) = 427932672.
812080 / 427932672 * 100 = 0.189768... percent.
```

Including the 32-channel action input/output projections and two time MLPs gives
430,098,464 parameters and 0.188813%. These are analytical architecture counts,
excluding VLM/vision weights and optimizer state. Do not describe the percentage
as a checkpoint measurement or a fraction of the whole VLA. The historical
“311M” Gemma comment alone omits the pi0.5 adaptive normalization parameters.

## 2. Waypoint planning and conditioning

**Source:** Xie et al., [arXiv:2609.30833v1](https://arxiv.org/html/2609.30833v1),
2026-09-25, Sections 4–6, especially Table 2 and Section 6.1. Citation key
`xie2026fastplans`.

| Custom hierarchy, reported success (%) | Four-suite average | Long |
| --- | ---: | ---: |
| Block-AR + suffix conditioning | 95.85 | 91.0 |
| Block-AR + normalized goal modulation (NGM) | 98.45 | 96.2 |

The baseline is a custom pi0.5-derived hierarchy, **not plain pi0.5**. The cited
released-pi0.5 row (96.85%) is not a matched rerun. Custom variants share data,
initialization, training seed, and step count; evaluation uses 50 official
initial states per task, 500 episodes per suite. Each variant is trained once.

NGM combines displacement conditioning, a denoising-phase gate, and anti-shortcut
noise/dropout; do not attribute its gain to waypoints alone. Section 6.3 tests
endpoint erasure with other inputs and flow noise fixed. An official
implementation was not identified in the paper or targeted search.

**Original redraw:** planner emits `(configuration, gripper, duration)` blocks;
executor also sees visual-language context. Add a distinct gated goal path and
an endpoint-erasure intervention. Avoid reproducing the original figure layout.

## 3. Semantic and visual subgoals in pi0.7

**Sources:** Physical Intelligence, [official technical report](https://www.pi.website/download/pi07.pdf),
Sections IV, V-A/B and Figure 2; [announcement dated 2026-04-16](https://www.pi.website/blog/pi07).
Citation key `physicalintelligence2026pi07`. The PDF is publisher-hosted;
an arXiv identifier or numbered revision was not verified.

The report supports intermediate text instructions that may come from a human,
a learned high-level policy, or be omitted. It separately describes multi-view
subgoal images generated by a learned world model. This is evidence for richer
conditioning, not a controlled LIBERO gain from adding one subgoal head to
pi0.5. Its approximately 5B model and 860M action expert are different systems.

**Original redraw:** observation history plus three separately labeled prompt
inputs—task/subtask text, near-future image, episode metadata—condition the
action expert. Depict optional inputs and distinguish semantic intent from a
geometric waypoint.

The [official OpenPI repository](https://github.com/Physical-Intelligence/openpi)
did not list pi0.7 in the inspected README. A pi0.7-specific implementation,
checkpoint release, and paper/figure reuse license were not verified. OpenPI's
code license is not permission to reproduce the report's figures.

## 4. Symmetry-aware action heads

**Source:** Li et al., [SAVLA, arXiv:2609.16641v1](https://arxiv.org/html/2609.16641v1),
2026-09-15, Sections IV and V, Tables I, IV, V. Citation key `li2026savla`.

At 10% demonstrations, Table V reports 48.3% for the comparison model and 57.1%
for SAVLA. The comparison is **budget-matched reduced GR00T N1.5**, with 100.8M
versus 103.3M trainable parameters; both heads train from scratch on a frozen
backbone. Reduced-data runs use identical per-task subsets and report 200
episodes with three seeds. Checkpoints at 20K/40K/80K undergo common screening.
This is not a stock-GR00T or pi0.5 comparison.

The head is SO(3)-equivariant, but the canonicalizer and rotation experiments
address gravity-axis rotations. Approximate planar image warping limits
end-to-end equivariance; the paper does not establish arbitrary viewpoint or
crop invariance.

**Original redraw:** camera geometry and rotation estimate feed a canonicalized
view; typed scalar/vector channels feed an equivariant action head. Show the
image transformation as approximate.

The paper links [EVIEHub/SAVLA](https://github.com/EVIEHub/SAVLA); that URL returned
404 during this audit. Code and code-license availability are unverified. Do not
substitute the unrelated “SA-VLA” reinforcement-learning repository.

## 5. Camera adaptation: correcting the quoted AnyCam result

**Source:** Heo et al., [AnyCamVLA, arXiv:2603.05868v3](https://arxiv.org/html/2603.05868v3),
submitted 2026-03-06, revised 2026-11-11, Section IV-A and Table I. Citation key
`heo2026anycam`.

| Unseen agent-camera views, all-suite average success (%) | Reported value |
| --- | ---: |
| Unadapted pi0.5 | 69.9 |
| Viewpoint-augmented pi0.5* | 87.2 |
| AnyCamVLA + frozen pi0.5 | 94.5 |

**87.2% belongs to the augmentation baseline, not AnyCamVLA.** The result averages
small/medium/large camera perturbations over four suites; the wrist camera is
held fixed. It is not ordinary fixed-view LIBERO. The VLA stays frozen, but the
LVSM view-synthesis module is fine-tuned on multi-view synthetic images. Thus
“zero-shot” must not be paraphrased as no training anywhere. An exact rollout
count was not verified from the paper text inspected.

Version pinning matters: [v1 Table I](https://arxiv.org/html/2603.05868v1) listed
the unadapted aggregate as 67.9%; v3 lists 69.9%. Both place 87.2% on pi0.5*.

**Original redraw:** calibrated observed views enter a view-synthesis adapter,
which renders the expected camera view for a frozen policy. Separate camera
pose adaptation from rotation equivariance of action coordinates.

The [official code](https://github.com/heo0224/AnyCamVLA#license) assigns different
licenses: OpenPI Apache-2.0, LVSM and its fine-tuned checkpoint CC BY-NC-SA 4.0,
LIBERO MIT; Gemma weights have additional terms. Do not apply one repository-wide
license to all components.

## Figure provenance and reuse

| Work | Paper's displayed license | Recommended manuscript treatment |
| --- | --- | --- |
| Hermite-VLA v2 | CC BY 4.0 | Original schematic, cite the paper; no copied robot images |
| Fast Plans v1 | arXiv perpetual non-exclusive license | Original schematic; no assumption of figure-republication permission |
| pi0.7 official PDF | Not verified | Original schematic; no copied figure |
| SAVLA v1 | arXiv perpetual non-exclusive license | Original schematic; no assumption of figure-republication permission |
| AnyCamVLA v3 | CC BY 4.0 | Original camera schematic; distinguish paper and code licenses |

Use independently designed boxes, curves, axes, and synthetic geometry. Caption
them as conceptual illustrations of cited methods, not reproduced experimental
outputs. For a quantitative literature figure, place comparisons in separate
within-paper panels with their actual comparator names and protocols. Do not
pool scores, create an overall ranking, or imply Scalar-Pi reproduction.

## Conditional symmetry statement for the proposed potential

The following is a mathematical derivation for a proposed extension, not a
measured property of the current trained checkpoints.

Let `x` denote the complete noisy action chunk, `c` its conditioning, and `t`
transport time. Let `rho(g)` be an orthogonal representation acting on action
coordinates and `T_g` the corresponding action on context. Assume a domain
closed under these maps, differentiability in `x`, and exact invariance

```text
Psi(rho(g) x, t, T_g c) = Psi(x, t, c).
```

The chain rule gives

```text
rho(g)^T grad_x Psi(rho(g)x, t, T_g c) = grad_x Psi(x,t,c),
grad_x Psi(rho(g)x, t, T_g c) = rho(g) grad_x Psi(x,t,c).
```

The same holds for the negative gradient. Scalar output alone does not establish
the premise: invariance must be constructed or tested. For a general invertible
linear map `A`, a Euclidean gradient transforms as `A^{-T}`, not `A`.

### Normalization and typed channels

For physical coordinates `a = D z + mu`, a physical rotation `R` induces
`z' = D^{-1} R D z + D^{-1}(R mu - mu)`. Equal normalization scales on axes mixed
by tabletop yaw make its linear action orthogonal. Independent quantile scaling
generally does not. Nonzero centers must transform consistently or be handled
as an affine action. Translation and rotation increments form separate vector
blocks; gripper, padding, and unrelated scalar channels use identity actions.
For anisotropic scaling, a properly transformed metric can replace the naive
Euclidean gradient: with `G = D^T D`, `A^T G A = G` and the metric gradient
`G^{-1} grad Psi` transforms as a vector under `A`.

### Gripper-relative coordinates versus physical yaw

Choose the representation explicitly. With calibrated gripper orientation `Q`
and position `p_g`, body-relative displacement is `Q^T(p-p_g)`. Under a global
rigid rotation of both gripper and scene it is unchanged, because
`(R Q)^T(R p - R p_g) = Q^T(p-p_g)`. In that canonical frame, the global yaw acts
trivially on these local components; decoding back to world coordinates rotates
the output. A passive change of local coordinate basis can instead act by a
nontrivial orthogonal matrix. Do not use both interpretations interchangeably.

Rotation increments need a specified convention: spatial and body increments
transform differently; finite orientations lie on SO(3), not unconstrained
Euclidean triples. Small axis-angle vectors can be treated as vectors under
proper rotations within a consistent chart; this does not remove chart
singularities or justify arbitrary rotation of joint coordinates.

For an active scene yaw, transform all relevant geometry, camera extrinsics,
and semantic conditions consistently. Robot-base constraints, gravity,
occlusion, workspace boundaries, and “left/right” instructions may break the
symmetry. Gravity-preserving tabletop yaw is narrower than arbitrary SO(3).
Changing an oblique view is not generally an in-plane image rotation, and an
arbitrary crop is not an invertible physical symmetry. Canonicalization or
novel-view synthesis requires its own assumptions and error assessment.

Even an equivariant instantaneous velocity field does not by itself prove an
equivariant sampled policy: the initial-noise law and numerical sampler must
also respect the representation. Nor is a gradient in noisy action-chunk space
a robot-state Lyapunov function or proof of physical stability, collision
avoidance, or closed-loop safety.

**Original theorem diagram:** a commuting square with action/context transform
on the horizontal arrows and potential-gradient evaluation on the vertical
arrows; a separate inset shows world vectors encoded into a calibrated gripper
frame. Label the former “conditional mathematical property” and the latter
“coordinate convention.”

## Unverified claims to keep out of completed results

- No Hermite, waypoint, canonicalization, or pi0.7 extension was trained here.
- The proposed 812,080-parameter auxiliary head is supported by released source
  dimensions, but it is not present in our released checkpoints.
- No primary artifact was supplied to this audit for the proposed local
  26,581-chunk / 1,493-demonstration basis study or its explained-variance
  percentages. Do not present those numbers as verified measurements.
- Literature accuracy does not resolve the existing mixed-device numerical
  acceptance limitation of Scalar-Pi's completed descriptive study.
