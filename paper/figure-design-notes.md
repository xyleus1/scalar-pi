# Figure design study

22 November 2026. Design notes for the Scalar-Pi v0.1.3 paper revision.

This revision changes how the existing evidence is presented. It does not add
experiments or alter success counts, paired intervals, evaluation identities,
hardware qualification, or statistical conclusions. The examples below are
primary robotics papers inspected visually at page scale, not a ranking of
current benchmark performance. Older influential implementations are included
because their visual vocabulary is directly relevant to this action expert.

## Primary examples inspected

| Paper | Pages and figures inspected | Visual observation | Application here |
| --- | --- | --- | --- |
| [pi0.5](https://arxiv.org/pdf/2504.16054) | Page 4, Fig. 3; page 8, Figs. 6–7 | The architecture figure separates training and inference with a vertical rule. Token strips, image inputs, an action expert, and concise arrows expose what each branch computes. Its evaluation panels group related conditions and put details in the caption. | Show the actual conditioning, action-tensor, scalar-reduction, and derivative paths. Separate shared computation from the changed readout. Use captions for qualifications rather than large banners inside the drawing. |
| [OpenVLA](https://arxiv.org/pdf/2406.09246) | Page 4, Fig. 2; page 7, Fig. 3 | Architecture modules have short labels, token glyphs make their interfaces visible, and a small number of colors identify inputs and components. The results use consistent model colors across grouped tasks, with one shared legend. | Label tensor interfaces and the operation that changes. Keep one model palette across every Scalar-Pi result panel. Use a shared legend and align comparisons on common axes. |
| [Octo](https://arxiv.org/pdf/2405.12213) | Page 3, Fig. 2; page 8, Fig. 6 | Task, observation, and readout tokens keep the same colors across pretraining and adaptation. Dashed outlines distinguish new modules. The scaling plot uses a small number of clearly marked curves and direct annotations. | Give repeated concepts stable visual identities. Distinguish a training-only auxiliary head from the deployed action path with line style as well as color. Prefer direct labels when a separate legend would require repeated lookup. |
| [ACT](https://arxiv.org/pdf/2304.13705) | Page 4, Figs. 4–5 | The architecture is divided into encoder and decoder computations. The action-chunk figure uses a literal time grid to explain which predictions overlap; it does not substitute a prose box for the mechanism. | Show the ten predicted actions and five executed actions as discrete steps. Use explicit branches and arrows for the full input derivative and for proposed auxiliary losses. Do not imply that action-chunk illustrations are measured robot trajectories. |
| [Diffusion Policy](https://arxiv.org/pdf/2303.04137) | Page 1, Fig. 1; page 5, Figs. 3–5 | The policy-representation comparison places equations next to the quantity each method produces. Small ablation plots use compact axes and a limited palette. Qualitative trajectories are actual experimental evidence, separate from conceptual representations. | Use equations to make direct-field, dot-potential, and scalar-head differences visible. Keep result plots compact and precise. Do not invent qualitative rollouts or smoothness traces: Scalar-Pi's retained data do not provide those measurements. |
| [Cosmos Policy](https://arxiv.org/pdf/2601.16163) | Page 4, Fig. 2; page 8, Fig. 4 | A structured grid exposes the roles of conditioning, target, latent, and noised frames. The results combine compact tables with a shared-legend task plot rather than giving every number a decorative card. | Align comparable stages and quantities. Use a table where the evidence consists of a few reported values under different protocols; reserve plotted comparisons for common quantities with an interpretable axis. |

These are observations from the linked figures. The adoption decisions are our
design judgments. No third-party figure, robot photograph, or rollout is copied
into the Scalar-Pi paper.

## Problems addressed in the preceding figures

The earlier diagrams put long descriptions into equally weighted rounded boxes.
Those boxes explained the topic but did not reveal tensor structure, branching,
or the distinction between a forward pass and an input derivative. Repeating
the pattern in a five-row future-work graphic made it read like a presentation
slide rather than an architectural comparison.

The earlier result set also used several different color roles, oversized panel
headings, and annotations that duplicated the captions. The ten-panel success
figure and the all-task heatmap needed to be judged at their final page width,
not only as large PNG files. A readable source graphic is insufficient if
scaling it into the manuscript makes labels or seed markers too small.

The PDF dimensions made this concrete. The preceding success figure was 11.33
inches wide with eight-point labels, so embedding it at the manuscript's
6.8-inch text width reduced those labels to about 4.8 points. The all-task
heatmap was 11.53 inches wide with 7.7-point labels, reducing them to about
4.5 points. Exporting at the intended publication dimensions addresses this
problem directly; higher PNG resolution alone does not.

## Shared visual specification

- **Model colors:** Scalar-Pi P uses blue `#0072B2`; the matched B1 direct control
  uses vermilion `#D55E00`; B0 uses gray `#999999`; the released reference uses
  near-black `#222222`. Model identity is also encoded through labels or marker
  shape, so the comparison does not depend on color alone.
- **Color meaning:** blue identifies the scalar arm or its intervention. It is
  not a generic synonym for success, a positive effect, or the improved model
  in every external paper. A positive/negative heatmap uses an explicitly
  labeled diverging scale centered on zero. Unrelated external studies use
  neutral marks or a table rather than inheriting the research-arm colors.
- **Typography and scale:** fixed publication dimensions, approximately
  seven inches for a full-width figure; compact, consistent sans-serif labels;
  mathematical symbols matched to the manuscript. Check legibility after
  embedding at the actual text width. Panel labels identify (a), (b), and so on
  without becoming separate poster headings.
- **Geometry:** thin rules and arrows, white space between functional groups,
  rectangular computational modules, and explicit interfaces. No shadows,
  decorative background cards, giant all-caps notices, or generic iconography.
- **Statistical marks:** preserve all three seed observations and distinguish
  them from a group mean. Plot the saved paired confidence intervals directly;
  do not turn seed ranges into confidence intervals or add inferred error bars.
  Draw a clear zero reference for differences. Bar axes start at zero; focused
  point-plot limits are labeled rather than disguised as full-range bars.
- **Captions:** retain sample size, uncertainty definitions, the separate
  released checkpoint, and the descriptive-only hardware limitation. Proposed
  architectures and literature measurements remain clearly identified in
  captions and section text, without repeating a paragraph inside each panel.
- **Output:** vector PDF plus PNG, a common style module, source-linked plotted
  arrays, and the same code path for individual figure export and paper build.

## Placement and interpretation

The main text should prioritize the intervention, matched experimental design,
overall and suite-level outcomes, paired uncertainty, and task heterogeneity.
The complete 40-task view belongs in the appendix where labels can use the
required space. A literature comparison graphic is not a leaderboard: the
source protocols differ. A table or clearly separated small panels is more
honest than a common ranked axis. Likewise, the five planned studies should
show their computational differences without receiving the visual emphasis
of completed results.

Illustrative action blocks are symbols for tensors, not reconstructed
demonstrations. No image implies that Scalar-Pi has measured smoother motion,
physical safety, lower latency, or statistically established superiority.

## Visual review procedure

Inspect both the individual figures and the compiled pages. Check arrow
endpoints, equation baselines, label collisions, line clipping, consistent
colors, zero references, number rounding, and legibility at final scale.
Review a grayscale rendering for important distinctions. Compare every plot
against the saved figure data and retain the unchanged source-report digest.
Re-render after corrections; review the delivered PDF rather than an earlier
working image. The release notes record what changed and the source package
contains the generators needed to reproduce the figures.


## User references and Applied Compute research artifacts

The six supplied retrieval-study JPEGs were inspected directly. Their useful
features are aligned marginal and paired-effect panels, direct labels with
short leaders, restrained grid lines, visible uncertainty, and a clear
hierarchy between the focal method and contextual controls. The warm canvas,
large standalone titles, and monospaced typography are not requirements for a
robotics paper. Following the user's later direction, the revision does **not**
copy the screenshot font: it uses conventional DejaVu Sans figure text with
Computer Modern mathematical notation. Layout and statistical clarity are the
transferable ideas.

The Applied Compute research pages were read and their actual embedded chart
and diagram artifacts were retrieved. Browser access was unavailable, so the
original SVG geometry and chart-generated SVG were rendered offline. This
inspection used Helvetica/Courier fallbacks for the site's unavailable fonts;
it assesses geometry, marks, color roles, and alignment rather than exact web
font metrics or animation. No reference artifact is redistributed in the paper.

| Research page and actual artifact | What was inspected | Decision for Scalar-Pi |
| --- | --- | --- |
| [Production RL](https://www.appliedcompute.com/research/production-rl), [data/weights loop artifact](https://www.appliedcompute.com/embeds/sanity/efc5c4c90c0927a50bc37d489d180b962da5ae40.html) | A static frame of the original deployment/trainer drawing: spatially separated systems, thin routes, explicit requests/responses, checkpoint labels, and a return path for quantized weights. The animated chart and moving traces were not treated as static evidence. | Distinguish a compute route from an explanatory annotation; make the full input-derivative path and the forward readout identifiable. Do not copy the broad rounded system cards or animation into a compact print figure. |
| [Quantized models](https://www.appliedcompute.com/research/quantized-models), [training/inference mismatch artifact](https://www.appliedcompute.com/embeds/sanity/f08da37be380207d79a3589e3b94a1bc81001378.html) | The batching/kernel panel uses two aligned lanes. The shared input appears at the same horizontal position, while a single batched operation is contrasted with three kernels and a distinct reduction. Pale blue and orange identify computational differences; short labels sit inside thin outlined modules. | Align direct and scalar paths and localize the changed readout. Show multiple action slots or transformer layers with literal repeated geometry. A panel should expose a difference in computation, not merely display two prose summaries. |
| [Shift-1 overview](https://www.appliedcompute.com/research/shift-1-research-overview), [offline cost/quality artifact](https://www.appliedcompute.com/embeds/sanity/4dde1ffc2fa4ae1a720d537e60533b4c93b7b965.html) | Two aligned point-and-interval panels repeat the same method encodings. One orange treatment is emphasized against dark and muted controls. Direct numerical labels, marker shapes, light grids, and uncluttered axes carry the comparison. | Use stable treatment colors and marker identities across results, aligned effect labels and intervals, and sparse value annotations. Preserve our existing blue scalar identity. Do not invent latency or cost measurements to reproduce this chart's axes. |

Applied Compute's inspected artifacts use predominantly white figure canvases;
the user's cream retrieval plots are a related reference, not an exact template
for every Applied Compute figure. Their strongest common feature is that a
reader can identify the compared quantities before reading a paragraph of
annotation. For Scalar-Pi this favors the new compute-path diagram, small
multiples with a shared legend, and a forest plot with aligned numerical
intervals. It does not justify a decorative typeface or additional colors.


## Iteration record

The first new rendering was reviewed at approximately 700 pixels across,
matching a full-width figure on a normal paper display rather than a poster
preview. Corrections included the future-work panel's bottom padding, an
explicit conditioning-context input for stage prediction, a labeled five-action
prefix for future prediction, a corrupted multiplication symbol, dense plot
labels, and colorbar seams in the all-task heatmap. The method figure also
received a distinct direct-control branch so the intervention can be compared
within the same compute graph.

The second rendering was inspected again after those changes. Method, study,
success, contrast, and all-task figures have 7.0-inch PDF widths and body labels
of at least 7.5 points, approximately 7.3 points when embedded at 6.8 inches.
Mathematical subscripts are smaller by design. No text spans extended beyond
the inspected figure pages. The corrected context/action inputs, interval
labels, heatmap scale, and panel boundaries were checked visually. Grayscale
renders of the success and task-detail figures retained distinguishable square,
triangle, circle, and diamond identities. This is a substantial legibility
improvement over the preceding 4.5?4.8-point effective labels, independent of
PNG resolution.
