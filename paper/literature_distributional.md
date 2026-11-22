# Distributional objectives, predictive supervision, and confirmation protocol

Verified against primary sources on 22 November 2026. This is a citation and
claim ledger for the expanded manuscript, not an account of new experiments.
ArXiv versions below are explicit; repository and documentation links describe
the retrieved implementation and must be commit-pinned before a replication.
Bibliography keys are defined in `references_distributional.bib`.

## Predictive supervision: what the comparisons establish

### DYNA-2 (`dyna2`)

The corporate technical report *Dyna-2: A 1-Million-Hour Scaling Law for
World-Action Models* is published by Dyna Robotics, August 2026. Figure 7 in
“Does a scaling law on human data imply a scaling law on robot data?” evaluates
held-out **robot action prediction** on 39 tasks: 12 internal YAM and 27 xdof
ABC tasks. These robot trajectories were excluded from training. Figure 10,
“Understanding what makes it work,” compares action-only, joint video/action,
and additional-video co-training recipes at matched training steps and several
human action-data scales. Joint prediction beats action-only on all 39 tasks
at the reported scales. These are offline prediction results, not 39 successful
zero-shot closed-loop robot tasks. Deployment and robot adaptation are separate
experiments. Cite it as a technical report, not a peer-reviewed theorem or a
matched scalar-head ablation. [Official report](https://www.dyna.co/dyna-2).

### Cosmos Policy (`cosmospolicy`)

Kim et al., arXiv:2601.16163v1, 22 January 2026, report **98.5%** mean LIBERO
success in the LIBERO table of §5.2 (98.1/100.0/98.2/97.6 across the four suites).
The headline result uses direct policy execution; it does not require the
separate planning procedure. The method fine-tunes a pretrained video model to
jointly generate actions, future states, and values. Appendix A.4.1 compares the
full method with removing auxiliary losses (97.0%) and removing pretraining
(94.6%). This supports the particular combined recipe, not a controlled claim
that our frozen-backbone scalar head would benefit equally. Its architecture,
pretraining, and action execution protocol differ from this study.
[Paper, §5.2 and Appendix A.4.1](https://arxiv.org/html/2601.16163v1).

The official implementation is [NVlabs/cosmos-policy](https://github.com/NVlabs/cosmos-policy).
Its retrieved [LIBERO runner](https://github.com/NVlabs/cosmos-policy/blob/main/cosmos_policy/experiments/robot/libero/run_libero_eval.py)
defaults to five action-denoising steps and a 16-action chunk/execution window.
These settings must not be silently substituted into this study's protocol.

### SaiVLA-JEPA (`saivlajepa`)

Xiao et al., arXiv:2610.38508v1, 28 October 2026, isolate future-latent
supervision from current-JEPA-feature conditioning. Section 5.1 and Table 2
report a paired, uniform-300-step comparison on **10,030** LIBERO-Plus episodes:
Future 79.00%, Current 78.97%, difference +0.03 percentage points, 95% CI
[-0.73, +0.77], p=0.958. Future versus BaseFlow is much larger, but also changes
the conditioning representation. The final mixed-limit table instead reports
Future 81.99% and Current 82.45%; the paired 300-step interval cannot be attached
to those numbers. Appendix A, Table 8 specifies four Euler steps and a
16-action replan window. Thus future supervision produced no detectable
aggregate advantage over current-feature conditioning at this budget; this is
not evidence that predictive supervision never helps. The future-prediction
heads are training-only. [Versioned primary PDF](https://arxiv.org/pdf/2610.38508v1).

## One-step distributional generation

### ESP (`esp`)

Makabe, Kim, and Matsushita (Microsoft Research Asia–Tokyo and the University of
Osaka), arXiv:2610.07696v1, 6 October 2026: §IV-C/Figure 4 reports one-step ESP
**91.8 ± 1.4%** versus ten-step flow **89.1 ± 1.2%**, across three training seeds.
The frozen backbone, randomly initialized action modules, training schedule,
and trainable parameter counts are matched. All train for 60k steps; each
predicts ten actions and executes five. Table I reports A100 expert latency
3.35 ms for ESP versus **28.39 ms** for flow; uncached end-to-end is
59.01 versus 82.54 ms. Table II is a different comparison: widening ESP's FFN
from 4,096 to 73,728 raises success to **93.7 ± 0.4%** and latency to **21.15 ms**.
The 3.35→21.15 ms change is a capacity tradeoff within ESP, not flow→ESP speedup.
Neither result guarantees physical smoothness.
[Paper, §IV-C/D, Figure 4, Tables I/II](https://arxiv.org/html/2610.07696v1);
[official project](https://lilika-makabe.github.io/energy-score-policy-site/).

### DriftingVLA (`driftingvla`)

Gao et al., arXiv:2608.29749v1, 30 August 2026: §V-B/Table I reports one-step
DriftingVLA **98.32 ± 0.13%**, versus ten-step pi0.5 **97.10 ± 0.13%** and
one-step-truncated pi0.5 94.02 ± 0.13%. However, the full recipe updates the VLM
and initializes a new action head. Section V-D/Table III(b)'s frozen-VLM,
expert-only variant achieves 94.57 ± 0.15%; warm-starting the flow head gives
93.33 ± 0.29%. The headline is therefore not an isolated frozen-backbone
objective substitution. Dimensionwise temporal drifting changes the training
geometry, while the shared model still generates a joint action chunk; it
should not be described as proving independent action dimensions.
[Paper, §V-B/D, Tables I/III](https://arxiv.org/html/2608.29749v1).

## Benchmark corrections and pairing

### LIBERO-Plus (`liberoplus`, `lajepa`)

Fei et al.'s final CVPR 2026 title is *LIBERO-Plus: A Progressive Robustness
Benchmark for Visual-Language-Action Models*; the arXiv/repository title is
*In-depth Robustness Analysis of Vision-Language-Action Models*. It has 10,030
variants across seven perturbation dimensions. This differs from repeatedly
evaluating the 40 original tasks. [Final primary paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.pdf);
[official repository](https://github.com/sylvestf/LIBERO-plus).

Clémente et al., *La-JEPA*, arXiv:2611.08156v1, 5 November 2026, Appendix B,
Tables 13–14, identifies unintended instruction changes in the released
LIBERO-Plus evaluator: variant filenames append perturbation parameters to
non-language prompts. Its corrected protocol restores the base training
instruction for all 8,493 non-language variants and preserves the intended
1,537 language variants. Table 14 holds the harness, MuJoCo version, initial
state, settling, and limits fixed; corrections affect models differently and
can change rankings. Flex-pi's BDDL-annotation alternative is not identical to
verbatim training instructions. Report released and corrected protocols
separately rather than merging their scores. [Primary correction](https://arxiv.org/html/2611.08156v1#A2);
[official code](https://github.com/mt-clemente/lajepa).

### Reset identity (`lerobotreset`, `lerobotlibero`)

LeRobot issue #4152, opened 26 July 2026 and still open at retrieval, provides
an implementation-specific reproduction: internal terminal resets combined
with vector-environment autoresets advance initial-state indices according to
termination timing. Equal seeds and episode indices can consequently mask
different initial conditions. The report reproduces synchronous vector
execution; it does not establish the same failure in every LIBERO harness.
It is not evidence that this study's explicit-state runner suffered this bug.
[Primary issue and reproduction](https://github.com/huggingface/lerobot/issues/4152).

LeRobot's maintained documentation also distinguishes hard resets from faster
soft resets: settled states/images can differ, so the modes are not bitwise
interchangeable. [Official LIBERO documentation](https://huggingface.co/docs/lerobot/libero).
For confirmation, record each initial-state identifier and content hash, the
actual prompt, reset mode, settling count, and action-noise schedule per arm.
Pair on those recorded inputs rather than seeds alone.

## Mathematical claims: precise scope

### A convex endpoint map can represent a conditional distribution

For a fixed conditioning value `g`, let `mu = N(0, I_d)` and let
`nu_g = Law(A | G=g)` have finite second moment. The quadratic transport problem
has a unique optimal map `T_g`, up to `mu`-null sets, and

\[
  (T_g)_\#\mu=\nu_g,\qquad T_g(z)=\nabla_z\Psi_g(z),
\]

for a convex potential. Gaussian absolute continuity is essential to the
standard source-side hypothesis; target absolute continuity is unnecessary.
Uniqueness is of the map almost everywhere, not of an arbitrarily normalized
potential. Cite Brenier for the original result, McCann's Main Theorem
(p.310) for general probability measures, and Gangbo–McCann Theorem 3.7
(p.136) for the finite-cost optimal-transport formulation.
[Brenier 1991](https://www.ceremade.dauphine.fr/~carlier/Brenier91.pdf),
[McCann 1995](https://www.math.toronto.edu/~mccann/papers/monotone.pdf),
[Gangbo–McCann 1996](https://www.math.toronto.edu/mccann/assignments/477/GangboMcCann96.pdf).

This is a pointwise conditional existence statement for almost every `g` with
the required moments. It does not establish a smooth jointly parameterized
family in `g`, finite-network exact representability, recovery by SGD, or
finite-sample generalization. Smoothness of the transport requires additional
assumptions; convexity alone does not bound acceleration or jerk in an executed
robot trajectory. A disconnected or discrete target is allowed, so continuity
of the map is not automatic.

The proposed endpoint `+grad_z Psi` differs from this repository's velocity
`+grad_x Phi(x,t,G)`. The existing dot-potential head is not constrained to be
convex, and differentiating a scalar does not itself supply convexity. An
energy-score loss is also distinct from sampling by the negative gradient of
an energy-based model. These distinctions should remain explicit in diagrams
and notation.

For an implementable convexity constraint, Amos, Xu, and Kolter's ICML 2017
ICNN construction (§3.1, Proposition 1) uses nonnegative hidden-to-hidden
weights and convex, nondecreasing activations, with unrestricted affine input
connections. Its partially input-convex variant (§3.2) permits nonconvex
conditioning dependence while preserving convexity in the designated input.
For the proposed transport, that designated input is noise `z`, with `G` fixed.
This is an architectural option, not a property of our existing transformer.
Input convexity also does not make training convex in the parameters.
[Primary ICNN paper](https://proceedings.mlr.press/v70/amos17b/amos17b.pdf)
(`icnn`).

### The energy score identifies a law at population optimum

Using the minimization convention and Euclidean norm on the **joint flattened
action chunk**, define

\[
 L(Q,a)=\mathbb E\|X-a\|-\tfrac12\mathbb E\|X-X'\|,
 \quad X,X'\overset{\mathrm{iid}}\sim Q.
\]

For finite-first-moment target and forecast laws, expected score is uniquely
minimized by the target law. Gneiting–Raftery §4.3, Eq.(22), and §5.1/Theorem 5
state the result for exponent `0 < beta < 2`; their score uses the opposite,
maximization sign. Squared norm (`beta=2`) identifies only the mean, not the
whole law. [Primary paper](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).

Condition on `G`: use independent noise draws and a target independent of those
draws given `G`. For `M >= 2`, an unbiased Monte Carlo loss is

\[
 \widehat L=\frac1M\sum_i\|X_i-a\|
 -\frac1{2M(M-1)}\sum_{i\ne j}\|X_i-X_j\|.
\]

This proposed estimator follows directly by taking expectations of independent
pairs. Scoring separate coordinate marginals would not identify their joint
dependence. Integrability over contexts is needed for a finite global risk.
Population propriety does not guarantee optimization, finite model capacity,
mode coverage with finite data, or closed-loop success. Combined with a
convex endpoint parameterization, it identifies the Brenier transport only
if the target law is representable and the population minimum is attained.

### Why the ideal one-step conditional-mean argument needs an endpoint assumption

The following is a derivation for this repository's noise-to-data time
convention, not an empirical claim about every trained flow network. Let

\[
 X_t=(1-t)A+tZ,\quad Z\perp A\mid G,\quad
 v^*(x,t,g)=\mathbb E[Z-A\mid X_t=x,G=g].
\]

The conditional expectation is the unrestricted square-loss optimum at a
specified time. If this regression condition is imposed at `t=1`, then

\[
 v^*(z,1,g)=z-\mathbb E[A\mid G=g],\qquad
 z-v^*(z,1,g)=\mathbb E[A\mid G=g].
\]

Thus a single reverse Euler step from the ideal endpoint field returns the
conditional mean. But a continuous training-time distribution assigns the
endpoint zero mass: the integrated loss alone does **not** fix `v(.,1,.)`.
Using the formula for a continuously extended population field requires an
appropriate limiting/regularity assumption; merely saying “trained by flow
matching” is insufficient. Finite capacity, optimization error, and endpoint
extrapolation can leave noise dependence. A different noise/data coupling also
changes the calculation. This is not a theorem that finite learned one-step
policies must collapse to a mean.

The local implementation uses `Beta(1.5,1) * 0.999 + 0.001`, the above
interpolation, target `noise - actions`, and reverse Euler updates. Its ideal
continuous distribution has no atom at `t=1`; finite floating-point sampling
does not repair the general theoretical implication. See
[`sample_time`, training, and sampling](../src/openpi/models_pytorch/pi0_pytorch.py).
The conditional-velocity framework is described in
[Lipman et al., Flow Matching](https://arxiv.org/abs/2210.02747), already cited
as `flowmatching` in the main bibliography.

## Proposed confirmation protocol, not completed results

1. **Separate objectives from architecture.** Compare the existing direct and
   dot-potential velocity heads under the same flow objective. Separately
   compare direct endpoint and explicitly input-convex endpoint generators
   under the same energy score. Match frozen encoder, data, trainable capacity,
   update budget, optimizer, precision, normalization, and execution horizon.
   Include a capacity-matched control before attributing a gain to an objective.
2. **Isolate predictive supervision.** Compare current-feature conditioning
   against the same model plus a training-only future-latent loss; preserve the
   inference path. Specify whether the encoder is frozen or updated. A weaker
   conditioning baseline alone cannot identify the benefit of prediction.
3. **Test distributions before making multimodality claims.** Use a held-out
   synthetic conditional mixture with known law, independent generator draws,
   and both distributional and mode-coverage diagnostics. For demonstrations
   with one recorded action per context, acknowledge the conditional estimation
   assumption; multiple noise samples are not additional target observations.
4. **Pin the closed-loop comparison.** Predeclare original LIBERO and
   corrected LIBERO-Plus as separate evaluations. Save actual prompts and
   initial-state hashes, enforce outcome-independent state selection, retain
   failed and infrastructure attempts distinctly, and match reset mode, task
   limits, noise, and training seeds. Use a common qualified device/runtime for
   each paired contrast, or predeclare hardware strata. Do not retroactively
   relabel existing mixed-hardware outcomes as confirmatory equivalence.
5. **Measure proposed benefits.** Save commanded actions, observed joint and
   end-effector states, and control timestamps if smoothness/jerk is a goal.
   Define units, derivatives, filtering, replan boundaries, and success
   conditioning before evaluation. Report collisions and completion alongside
   smoothness, so immobility cannot appear beneficial. The delivered study's
   episode logs lack these trajectories; fixed-input solver captures cannot
   establish executed-motion smoothness.
6. **Report matched budgets and uncertainty.** Predeclare primary contrasts,
   paired resampling units, seed aggregation, multiplicity treatment, and
   stopping rules. Report expert-only and complete-policy latency separately
   on the same hardware, including the input-gradient cost. Success, latency,
   distributional fidelity, and physical smoothness are separate endpoints.

No claim in this ledger authorizes new computation or changes the completed
study's numerical eligibility criteria, empirical results, or acceptance state.
