# Learning objectives

This page explains the scientific semantics of the objectives. The numeric
example values below describe the original staged Synthetic reference; exact
settings for the final study are recorded in its frozen protocol and per-run
provenance. Current output labels use `soft`, `infonce`,
`time_contrastive_blocks`, and `behavior_contrastive_blocks`.

## Representation target

For each neural window `x_i`, an encoder produces `z_i = f_theta(x_i)` in
three dimensions. The encoder is not regressed directly onto the simulator
latent during training. Instead, the objective specifies relations among
windows and the learned embedding is evaluated afterwards against the known
latent components.

The current benchmark compares four objectives:

1. `soft`: pairwise soft contrastive objective using only time and condition;
2. `infonce`: supervised label-based InfoNCE;
3. `time_contrastive_blocks`: sampled triplets with a same-trial positive at
   the configured temporal offset;
4. `behavior_contrastive_blocks`: sampled triplets with a same-condition
   positive.

PCA is the non-neural baseline and has no loss.

## Soft structured contrastive loss

Let a minibatch contain `B` windows. Metadata are not input features; they are
used only to build a target relation. For windows `i,j`:

```math
\\Delta t_{ij}=|t_i-t_j|/(T-1)
```

The condition distance is circular:

```math
\\Delta c_{ij}=\\min(|c_i-c_j|, K-|c_i-c_j|)/(K/2)
```

with `K=8`. The current protocol uses equal weights:

```math
D_{ij}=\\frac{0.5\\Delta t_{ij}+0.5\\Delta c_{ij}}{0.5+0.5}.
```

The target affinity and row-normalized target distribution are:

```math
S_{ij}=\\exp(-D_{ij}/\\tau_{metadata}),
\\qquad
Q_{ij}=S_{ij}/\\sum_{k\\ne i}S_{ik},
```

where `tau_metadata=0.5`; the diagonal is removed.

The encoder distribution uses cosine similarity:

```math
P_{ij}=\\frac{\\exp(\\cos(z_i,z_j)/\\tau_{embedding})}
{\\sum_{k\\ne i}\\exp(\\cos(z_i,z_k)/\\tau_{embedding})},
```

with `tau_embedding=0.1`. The loss is the row-wise cross-entropy:

```math
\\mathcal L_{soft}=-\\frac1B\\sum_i\\sum_{j\\ne i}Q_{ij}\\log P_{ij}.
```

There is no separate progress gate in the current implementation. The only
metadata supplied to the target are time and condition. The loss is pairwise,
not a triplet loss.

## Other objectives

### Supervised InfoNCE

All non-self windows with the same direction label are positives; other batch
members are negatives. This is label-supervised and can preserve condition
identity without preserving the exact temporal geometry.

### Time-offset objective

The sampler chooses a reference window, a positive window from the same trial
at temporal offset `+10`, and an independently sampled negative. The triplet
is resampled before every optimizer step. This explicitly encodes the temporal
prior used for the lag-recovery benchmark.

### Behavior objective

The sampler chooses a reference and a positive with the same discrete
condition. It does not impose a temporal offset, so it is not expected to be
the best objective for recovering the imposed inter-subject lag.

For both sampled objectives, the triplet objective normalizes the three
encoder outputs for cosine scores, uses the configured temperature, and
contrasts each reference with the negative batch.

## What is supervised?

- `soft` is task-informed/weakly supervised because condition and time labels
  define the target geometry;
- `infonce` is supervised by direction labels;
- `time_contrastive_blocks` uses trial identity and a researcher-chosen
  temporal offset;
- `behavior_contrastive_blocks` uses condition labels;
- PCA has no task labels during fitting.

Calling all four neural objectives unsupervised would therefore be incorrect.

## Dimensions and computation

In the original staged Synthetic reference, the neural input is
`(batch, 21, N)`, with `N=160` for A and `N=120` for B; the embedding is
`(batch, 3)`. The soft and supervised objectives build dense `B x B`
similarity matrices, so their loss computation is approximately `O(B^2)` in
time and memory. Sampled objectives use three encoder forwards per optimizer
update and an explicit triplet batch.

That historical reference used AdamW with fixed settings and a 2000-update
budget. Those numbers do **not** define the final HPO/final-fit protocol. For
the current frozen protocol, use the Phase-2A sealed candidate table and the
spec/provenance under `outputs/final_thesis_v1/freeze/`; do not copy the
historical settings into a new run. In all branches, test observations are
held out from fitting and model selection.

## Scientific interpretation

The objective itself injects a prior. High recovery is therefore interpreted
as evidence that the neural observations support the specified relational
structure, not as proof that the brain contains exactly that geometry.

The complete shared latent `Z` is used only for evaluation. The main controls
are recovery of `Z`, cross-subject lag estimation, held-out generalization, and
computational cost. Any internal decomposition used by the simulator is a
secondary diagnostic, not a separate representation-learning target.

Relevant ablations are removing time or condition, changing the offset,
shuffling labels, varying window size and temperatures, and repeating the
experiment with independent seeds.

## Real-data use of the objectives

The final Real study uses the same four scientific objective labels. The
positive training offset remains 10 bins at 1 ms sampling; this is a training
sampling rule, not the controlled 100/160/200 ms digital interventions and not
a claimed biological lag. Exact Real objective settings and the selected
window are recorded in the frozen experiment spec.

Because no ground-truth latent process is observed in the monkey recording,
objective comparisons there concern task decoding, behavioural regression,
input alignment, robustness, and efficiency—not recovery of `Z`.
