# Learning objectives

## Representation target

For each neural window `x_i`, an encoder produces `z_i = f_theta(x_i)` in
three dimensions. The encoder is not regressed directly onto the simulator
latent during training. Instead, the objective specifies relations among
windows and the learned embedding is evaluated afterwards against the known
latent components.

The current benchmark compares four objectives:

1. `soft`: pairwise soft contrastive objective using only time and condition;
2. `infonce`: supervised label-based InfoNCE;
3. `cebra_time`: sampled triplets with a same-trial positive at offset 10;
4. `cebra_behavior`: sampled triplets with a same-condition positive.

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

For both sampled objectives, `cebra_infonce_loss` normalizes the three encoder
outputs for cosine scores, uses temperature `1.0`, and contrasts each
reference with the full negative batch.

## What is supervised?

- `soft` is task-informed/weakly supervised because condition and time labels
  define the target geometry;
- `infonce` is supervised by direction labels;
- `cebra_time` uses trial identity and a researcher-chosen offset;
- `cebra_behavior` uses condition labels;
- PCA has no task labels during fitting.

Calling all four neural objectives unsupervised would therefore be incorrect.

## Dimensions and computation

The current neural input is `(batch, 21, N)`, with `N=160` for A and `N=120`
for B. The embedding is `(batch, 3)`. The soft and supervised objectives build
dense `B x B` similarity matrices, so their loss computation is approximately
`O(B^2)` in time and memory (`B=1024`). Sampled objectives use three encoder
forwards per optimizer update and a batch of explicit triplets.

Training uses AdamW (`lr=1e-3`, `weight_decay=1e-4`) for exactly 2000 optimizer
steps per subject/model/loss. The held-out fit uses trial-level training data;
the test trials are never used for fitting. The validation trials are used for
checkpoint selection. The full-sample branch is descriptive and has no
independent validation estimate.

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

The real-monkey branch reuses the same objective identifiers and checkpoint
logic. Its controlled time offset is `10` bins, which equals 10 ms for the
1 ms-binned recording; this is an experimental sampling offset, not a claimed
biological lag. In scientific prose, `cebra_time` and `cebra_behavior` are
called **Time Contrastive Blocks** and **Behavior Contrastive Blocks**; the
identifiers are retained only for cache and file compatibility.

Because no ground-truth latent process is observed in the monkey recording,
objective comparisons there concern task decoding, behavioural regression,
input alignment, robustness, and efficiency—not recovery of `Z`.
