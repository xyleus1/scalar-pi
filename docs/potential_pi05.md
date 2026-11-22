# Scalar-potential π0.5: methods

Scalar-Pi changes the output parameterization of the complete action expert while retaining the pretrained π0.5 architecture. The implementation is based on OpenPI commit `215abfb217dbac7d5f1273282331b9b1866c0479`. The completed study used worker commit `a9c0bdbf5f20356246f25ec303c51e466fa4c3a3`; the release records its manifest and artifact hashes.

## Architecture

Images and language pass through the original PaliGemma backbone. The action expert receives its conditioning cache, noisy normalized actions of shape `[B, 10, 32]`, and the original time embedding. Its width 1024, depth 18, MLP width 4096, masks, positions, smooth activations, and adaptive RMS conditioning are unchanged.

For the original action expert output $f_\theta$, define

$$\Psi_\theta(x,\tau,c)=\sum_{h,d}x_{hd}f_{\theta,hd}(x,\tau,c),$$
$$v_\theta=\nabla_x\Psi_\theta=f_\theta+J_f^\mathsf{T}x.$$

There is one scalar per example, summed over the entire action chunk. Averaging instead would change the velocity scale. The full derivative includes the input projection, every action-dependent transformer layer, and the output projection. Detaching the raw field would remove the Jacobian term and implement a different model.

The action expert, input/output projections, and time conditioning are trained together. The vision-language backbone is frozen in every research arm. The LIBERO configuration uses `discrete_state_input=False`; numerical proprioception remains in observation plumbing but is not added as a new network input. Seven of the 32 model action channels are executed.

## Training objective and controls

The native flow-matching convention is retained:

$$x_\tau=(1-\tau)A+\tau\epsilon,\quad u=\epsilon-A,\quad \epsilon\sim\mathcal N(0,I),$$
$$\tau=0.001+0.999\,\mathrm{Beta}(1.5,1).$$

Training minimizes mean squared velocity error over all 32 channels. Sampling starts at $\tau=1$ and uses negative Euler increments to reach zero.

| Arm | Configuration | Output initialization | Velocity |
|---|---|---|---|
| B0 | `pi05_libero_b0` | Pretrained projection | Direct field |
| B1 | `pi05_libero_b1` | Zero projection | Direct field |
| P | `pi05_libero_potential` | Same tensors as B1 | Full potential gradient |

B1 and P begin at zero velocity. The output projection can update immediately; zero trunk gradients on the first update are expected. Architecture metadata is required because the direct and potential models have the same weight shapes.

The nine completed runs use seeds **17, 42, and 93**, the full official LIBERO training set, and the base π0.5 checkpoint. Each run has:

- 30,000 optimizer updates and effective batch 256, preserved through accumulation.
- AdamW with betas `(0.9, 0.95)`, epsilon `1e-8`, and weight decay `1e-10`.
- Global gradient clipping at 1.0.
- Linear warmup for 10,000 updates to `5e-5`, followed by a constant learning rate.
- EMA decay 0.999 and the terminal EMA checkpoint for evaluation.
- The same tokenizer, image transforms, normalization, action horizon 10, and replanning horizon 5.

All research arms use `potential_pi_bf16_vlm_fp32_expert_v1`: BF16 frozen VLM weights with the declared native FP32 embedding and normalization exceptions; FP32 action expert, optimizer, and EMA; eager execution and `highest` FP32 matmul precision. Earlier all-FP32 diagnostics are a separate protocol and are not pooled into these results. Mixed checkpoint dtypes, tied tensors, and exact recovery state are preserved.

This is a matched frozen-backbone adaptation study, not a reproduction of upstream full-VLM fine-tuning.

## Evaluation and statistics

The matrix covers `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`: 40 tasks, each with 50 official initial states. Every checkpoint is evaluated at **K=1 and K=10**. The research arms contribute 36,000 episodes; the released LIBERO policy contributes 4,000 reference episodes. The complete grid has 20 checkpoint/K batches, 80 suite cells, and 800 task records.

Rollout seed and paired-noise seed are both 7. Per-query noise is paired by task, initial state, and query index. Settling, camera orientation, action processing, and suite-specific limits remain those of the pinned runner. Policy-caused invalid actions count as failures. Infrastructure-failed attempts are excluded through explicit retry selection, with no duplicated episode identity.

The primary contrast is **P−B1 at K=10**. P−B0, K=1 comparisons, and within-model step comparisons are secondary. The released policy is a separate reference with a different training history.

The reported descriptive intervals use 10,000 paired percentile-bootstrap resamples with analysis seed 0. Training seeds are resampled jointly across the compared methods; initial-state draws within each fixed task are shared across sampled seeds and methods. Tasks and suites receive equal weight. Three training seeds provide limited information about training variability, and a nonsignificant contrast is not evidence of equivalence.

All tasks appear in the training data, so the experiment does not measure unseen-task generalization.

## Results and numerical limitations

All 40,000 selected episode rows are present, and all 800 selected task logs passed the original row validator. Four excluded infrastructure-failure records and their retry selections are retained in the release's provenance. The original accepted phase archives cover 16,000 episodes; row validation of the full grid is a distinct evidence category.

At K=10, P has 91.3833% success and B1 has 91.0833%. The paired difference is **+0.30 percentage points**, with descriptive 95% interval **[−1.2667, +1.75]**. This does not establish a benefit from the scalar constraint.

A100 PCIe fixed-input actions failed the unchanged cross-H100 numerical criterion. A100 SXM rollouts lack accepted cross-H100 qualification. The complete mixed-device statistics are therefore descriptive rather than the registered confirmatory result. The primary P/B1 pairs use matched hardware/runtime families; timing comparisons remain stratified by hardware. Same-device repeatability and matching weights do not establish cross-device equivalence.

Conservativeness holds in normalized action-chunk coordinates at fixed context and time. It does not imply convexity, physical energy, a robot Lyapunov function, smoother motion, or faster inference. The retained study does not contain the trajectory measurements required to establish a smoothness advantage.

## Related work

The dot-potential construction and zero-output initialization follow the scalar architectures discussed in *Composing Flow-Matching Energies with Known Physics* and *Equilibrium Matching*. *EnergyFlow* provides a related scalar robot-policy architecture with a different objective. These precedents motivate testing the constraint; they do not establish that it improves a pretrained π0.5 expert.

See the [paper](../paper/manuscript.tex) for the versioned bibliography and full discussion. Installation, data preparation, training, and replay commands are in [Reproduction](reproduction.md).
