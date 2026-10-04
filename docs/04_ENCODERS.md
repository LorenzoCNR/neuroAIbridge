# Encoders

## Common Interface

Every temporal encoder receives:

```text
x: (batch size, window size, neural features)
```

and returns:

```text
z: (batch size, embedding dimension).
```

One row of `z` summarizes one complete neural window. The controlled notebooks
set the embedding dimension equal to the known latent dimension to make direct
coordinate recovery diagnostics possible. This is an experimental choice, not
a requirement of the models.

## PCA

PCA is the linear baseline. Each window is flattened:

```text
(W,N) -> (W*N).
```

PCA centers the training data and finds orthogonal directions that maximize
explained variance. It has no contrastive loss and no explicit temporal
architecture: temporal order is represented only by the fixed position of
each bin in the flattened vector.

### Why Include It

- It is deterministic and easy to inspect.
- It establishes whether nonlinear learning is necessary.
- It can recover a low-rank linear observation model surprisingly well.

### Limitations

- High variance need not correspond to task-relevant structure.
- Flattening does not encode translation-invariant temporal motifs.
- It is sensitive to scaling and cannot model nonlinear observation geometry.

PCA is fit only on training windows and then applied unchanged to test windows.

## Residual CNN1D

The validated nonlinear encoder is `TemporalCNNEncoder`.

### Tensor Flow

The external tensor:

```text
(B,W,N)
```

is transposed for PyTorch `Conv1d`:

```text
(B,N,W).
```

Neurons are channels and convolution kernels move across time.

The default architecture is:

```text
Conv1d(N -> 64, kernel=3, padding=1)
GELU
two residual Conv1d(64 -> 64, kernel=3, padding=1) blocks
global average pooling over time
Linear(64 -> embedding_dim)
L2 normalization
```

With three convolutions of kernel size three, stride one, and no dilation, the
effective receptive field is seven bins. Padding preserves window length.

Residual additions help optimization and preserve information while the
receptive field grows. Global average pooling produces one hidden vector per
window; the loss therefore compares windows, not individual bins.

### Strengths

- Strong inductive bias for local temporal patterns.
- Parameter sharing across positions inside a window.
- Lower cost than full self-attention for long sequences.
- Direct compatibility with cosine contrastive objectives.

### Limitations

- Global average pooling discards explicit output timing within the window.
- The receptive field may be smaller than the full window.
- Symmetric padding permits use of bins on both sides of the target and is not
  appropriate for strictly causal prediction.
- L2 normalization restricts embeddings to the unit hypersphere and removes
  radial information.

## MLP

`TemporalMLPEncoder` flattens the window and applies:

```text
Linear(W*N -> hidden)
GELU
Linear(hidden -> embedding_dim).
```

It is a nonlinear baseline without an explicit temporal inductive bias. It can
show whether improvements come from nonlinearity alone or from convolutional
structure. It is implemented but not included in the four validated notebooks.

## LSTM

`TemporalLSTMEncoder` reads the `W` bins sequentially. The final hidden state of
the top recurrent layer summarizes the window and is projected into embedding
space.

### Strengths

- Order is intrinsic to recurrence.
- Suitable for variable-length or causal formulations.
- Hidden state can represent history-dependent dynamics.

### Limitations

- Sequential computation limits parallelism.
- The final state can become an information bottleneck.
- It is harder to optimize for long contexts.

The LSTM is implemented but not yet validated in the controlled experiment
matrix.

## Transformer

`TemporalTransformerEncoder` treats each time bin inside a window as one token.
The `N` neural counts at that bin are the token features.

### Tensor Flow

```text
(B,W,N)
-> linear feature projection (B,W,model_dim)
-> sinusoidal positional encoding
-> Transformer encoder layers
-> mean pooling across W tokens
-> linear projection (B,embedding_dim)
-> optional L2 normalization.
```

Self-attention compares all pairs of time bins **inside the same window**. It
does not directly compare tokens belonging to different windows. Different
windows interact only after pooling, when their embedding vectors enter the
contrastive loss.

Positional encoding is necessary because self-attention alone is permutation
equivariant and does not know temporal order.

### Strengths

- Direct interaction between every pair of bins in the window.
- Flexible modeling of nonlocal temporal dependencies.
- Highly parallel sequence processing.

### Limitations

- Attention cost is quadratic in window length.
- The current 21-bin windows provide more context, but the quadratic attention
  cost still grows with window length.
- Mean pooling can hide token-specific temporal structure.
- It has more hyperparameters and a greater overfitting risk than PCA or CNN1D.

The Transformer is part of the current shared-latent comparison in
`notebooks/experiment_05_shared_latent_staged.py`.

## Real-monkey encoder configuration

The first real-data run uses the same three-way comparison—PCA, CNN1D, and
Transformer—but fits it to `(21, 65)` neural windows and produces three
embedding coordinates. Every neural checkpoint produces both `embedding_raw`
and its L2-normalized `embedding_unit` counterpart. The two forms are saved
from the same checkpoint and are evaluated separately; normalization is not a
second training run.

## Normalization And Geometry

With L2 normalization:

```math
\widetilde z_i
=
\frac{z_i}{\| z_i\|_2},
\qquad
\|\widetilde z_i\|_2=1.
```

$\| z_i\|_2$ is the Euclidean length of embedding vector $z_i$.
Normalization divides every coordinate by that length, preserving direction
but discarding magnitude.

The dot product used by the contrastive loss then equals cosine similarity.
This stabilizes the objective but constrains the representation to a sphere.
A projected 2D or 3D plot can appear compressed even when pairwise angular
relations are useful.

Turning normalization off restores radial degrees of freedom but changes the
optimization geometry and allows embedding norms to affect logits. This must
be treated as an ablation, not a cosmetic plotting choice.

## Why Visualizations May Disagree With Metrics

- A 2D view can omit informative dimensions.
- A 3D view of a higher-dimensional embedding is still a projection.
- RSA tests pairwise distance ordering, not a common center or exact trajectory
  shape.
- Procrustes can align rotation, reflection, and global scale but cannot repair
  nonlinear distortion.
- Trial averaging can reveal a trajectory hidden by single-window noise.

Therefore every model should be assessed using held-out scalar metrics,
condition-averaged trajectories, and task-specific geometric diagnostics.
