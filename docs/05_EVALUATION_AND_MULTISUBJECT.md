# Evaluation, Alignment, And Multiple Subjects

## Why More Than One Metric Is Needed

Representation learning is identifiable only up to transformations allowed by
the objective. A learned embedding can be rotated, reflected, rescaled, or
nonlinearly distorted relative to the known latent state. NeuroBridge therefore
combines coordinate alignment, relational geometry, and trajectory plots.

## Procrustes Alignment

Let `Z` be the known latent matrix and `E` the learned embedding, with rows
corresponding to the same observations. Procrustes alignment:

1. centers both matrices;
2. divides by a global Frobenius norm;
3. finds the orthogonal rotation or reflection minimizing squared error;
4. transforms `E` into the coordinate system of `Z`.

The reported score is:

```math
R^2_{\mathrm{Proc}}
=
1-
\frac{
\| Z-E_{\mathrm{aligned}}\|_F^2
}{
\| Z-\bar Z\|_F^2
}.
```

Here:

- $Z$ is the known latent matrix;
- $E_{\mathrm{aligned}}$ is the embedding after Procrustes transformation;
- $\bar Z$ repeats the column means of $Z$;
- $\|\cdot\|_F^2$ sums the squared values of every matrix entry.

The numerator is residual mismatch after alignment. The denominator is the
total variation of the known latent around its mean. A score near one means
small residual mismatch; zero means the aligned embedding is no better than
using the latent mean.

The operation does not reorder observations and does not delete a coordinate.
If a candidate lag is applied, only nonoverlapping boundary rows are excluded
before alignment.

When embedding and latent dimensions differ, the current evaluation helper
reduces the embedding to the latent dimension with PCA before Procrustes. This
is a practical diagnostic but must be disclosed because it adds a fitted
projection to the evaluation.

## Representational Similarity Analysis

RSA avoids direct coordinate matching:

1. compute all pairwise Euclidean distances among held-out known states;
2. compute the corresponding distances among held-out embeddings;
3. correlate the upper triangles of the two distance matrices.

Spearman correlation tests agreement in rank ordering; Pearson correlation
tests linear agreement of distances.

A Spearman value of `0.927` means that pairs ranked as near or far in latent
space tend to retain that ordering in embedding space. It does not mean 92.7%
classification accuracy and does not prove that a visible loop or common
center has the correct shape.

## Full State And Motor Core

For enriched experiments:

- **full-state RSA** uses every simulated coordinate;
- **motor-core RSA** uses task-defining coordinates only.

The circular motor core is X, Y, and progress. The linear motor core is position
and direction. Reporting both distinguishes recovery of the main task geometry
from recovery of velocity and context.

## Trajectory Plots

Window embeddings are grouped by condition and target time, then averaged
across trials. This reduces spike noise and reveals the mean dynamical path.

Plots should state whether they display:

- native embedding coordinates;
- PCA coordinates;
- a visualization projection of a higher-dimensional embedding;
- Procrustes-aligned coordinates.

These are different objects. A native three-dimensional embedding should not be
called PCA unless PCA was actually applied.

## Two Subjects Performing The Same Task

For subjects `A` and `B`, a controlled simulation can use:

```text
shared task state Z
subject-specific B_A, c_A, neural noise -> X_A
subject-specific B_B, c_B, neural noise -> X_B.
```

The ground-truth task geometry is shared, while neural populations and spike
realizations differ. An imposed delay can shift subject `B` later within each
trial.

This setting tests two separate questions:

1. do both encoders recover a compatible task geometry?
2. does cross-subject temporal alignment recover the imposed delay?

## Lag-Aware Alignment Scan

For each candidate lag $\ell$, the current utility compares:

```math
E_A(r,t)
\quad\mathrm{with}\quad
E_B(r,t+\ell).
```

$r$ identifies the same trial in both subjects, $t$ is reference time, and
$\ell$ is the candidate temporal displacement measured in bins.

Only valid overlapping rows from the same trial are retained. No sample wraps
across trial boundaries. Procrustes alignment is fit for that candidate and an
alignment score is recorded:

```math
\mathrm{score}(\ell)
=
R^2_{\mathrm{Proc}}
\left(
E_A(r,t),
E_B(r,t+\ell)
\right).
```

The estimated lag is:

```math
\widehat\ell
=
\arg\max_{\ell}\;
\mathrm{score}(\ell).
```

`arg max` returns the candidate lag at which the score is largest; it returns
the lag value, not the score itself.

With this convention, a positive best lag means the second population is best
matched at later time indices. The convention must always be restated on plots
because reversing axes or arguments reverses the verbal interpretation.

## What Happens At The Boundaries

In the current run each trial has 200 bins. For a candidate lag of 10, the
comparison uses 190 paired bins:

```text
reference times 0..189
other times     10..199.
```

The final 10 reference bins and first 10 other bins have no partner for that
lag and are omitted. No embedding dimension is lost; only unmatched temporal
rows are excluded.

## Lag-Lag Matrices

A lag-lag matrix compares temporally aggregated representation profiles for
many lag choices in population A against many lag choices in population B.
Off-diagonal structure can indicate temporal asymmetry.

Its interpretation depends on axis convention and on the exact aggregation.
It is a diagnostic visualization, not by itself evidence of information
transfer or causal influence. Smooth trajectories and autocorrelation can
produce broad ridges even without directional interaction.

## Required Controls For A Strong Multisubject Claim

- recover the known imposed lag across multiple seeds;
- report confidence intervals or bootstrap uncertainty;
- use held-out trials for alignment scoring;
- include zero-lag and shuffled-trial nulls;
- reverse subject order and verify sign convention;
- test several imposed delays;
- compare task-informed and time-only objectives;
- show performance when subjects have different neuron counts and mappings;
- verify that alignment is not explained only by condition averages.

The utilities for subject-specific mappings and lag scans exist. A complete
multisubject benchmark satisfying all controls is future work.

## Real-data evaluation boundary

The local monkey recording is a single active Area-2 session. Its staged
evaluation therefore separates four questions rather than collapsing them:

- direction decoding and regression of progress, position, and velocity;
- temporal/input structure through CKA to the observed input windows;
- post-hoc embedding-noise robustness and participation ratio;
- computational cost recorded with the model artifacts.

The real branch has no observed latent `Z`, no second subject/session, and no
biological lag label. Any future temporal shift or lag-lag experiment must be
described as a controlled perturbation of the observed data. The current
`full_sample` output is descriptive; held-out trial evaluation remains the
generalization result.
