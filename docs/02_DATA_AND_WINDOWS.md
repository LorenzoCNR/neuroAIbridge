# Data, Metadata, And Temporal Windows

## Observable Unit

The simulator returns

```text
X.shape = (trials, time bins, neurons).
```

`X[r,t,j]` is the spike count of neuron `j` during time bin `t` of trial `r`.
The basic scientific unit is therefore a complete trial with a multivariate
neural time series, not an isolated scalar observation.

## Why Use Windows

An encoder needs local temporal context around a target time bin. A window of
length `W` collects:

```text
W consecutive time bins x N neurons.
```

For the current run, `W=21`; one model observation has shape `(21,N)`, with
`N=160` for subject A and `N=120` for subject B. The window
does not create ten independent embeddings: PCA and the neural encoders reduce
the complete window to one embedding vector associated with its target time.

Window length is a model-selection parameter. A short window can omit relevant
dynamics; a long window can mix distinct phases and add redundant information.
It must be evaluated by held-out performance and temporal-resolution
diagnostics rather than justified only by architectural convenience.

## Centered Padding And Trial Boundaries

The current experiment uses centered windows, `window_size=21`, and
`stride=1`. Padding preserves all 200 target time bins in every trial.

Two rules are essential:

1. a window never crosses a trial boundary;
2. padding uses edge information within the same trial rather than data from
   the preceding or following trial.

Consequently, 200 trials of 200 bins produce 40,000 window-level observations
per subject (before excluding B's ten lag-padding centers from fitting).

## Metadata Returned With Each Window

The PyTorch dataset returns a mapping containing:

```text
x               neural window, shape (W,N)
time_id         target time within the trial
global_time_id  flattened time index
trial_id        trial identity
label           task condition
progress        normalized movement progress
```

Metadata accompanies the window through DataLoader shuffling. Shuffling changes
minibatch membership but does not detach a window from its time, trial, or
condition. The pairwise contrastive target is reconstructed from the metadata
inside each minibatch.

## Train/Test Split

The split is performed on complete trials before model fitting:

```text
200 total trials
140 training trials
 40 held-out test trials
 20 validation trials
```

No window from a held-out trial is used to fit PCA, CNN1D, or Transformer. This avoids the
strong leakage that would occur if overlapping windows from the same trial
were divided randomly between training and test sets.

For the circular task, the split is stratified by condition so every direction
is represented. The test set evaluates new stochastic realizations from the
same simulator and task distribution. It does not test transfer to unseen task
conditions or a new simulator.

## Model-Specific Views Of The Same Window

All validated models receive the same conceptual observation.

### PCA

PCA reshapes:

```text
(batch, W, N) -> (batch, W*N)
```

and finds linear directions of maximum variance using training windows only.

### CNN1D

CNN1D transposes:

```text
(batch, W, N) -> (batch, N, W)
```

so neurons are channels and convolution moves across time.

### LSTM

The LSTM reads the original `(batch,W,N)` sequence in temporal order and uses
the final hidden state as the window summary.

### Transformer

The Transformer treats each of the `W` time bins as one token whose features
are the `N` neural channels. A learned linear projection maps each token into
model dimension, positional encoding identifies its order inside the window,
self-attention relates all bins inside that window, and mean pooling produces
one vector.

There is no attention directly between different windows in the current
encoder. Windows interact during training through the contrastive loss:
their final embedding vectors are compared pairwise within a minibatch.

## Observation Counts In A Minibatch

With the current `batch_size=1024`, the CNN receives:

```text
x: (1024,21,N)
```

and emits, for a three-dimensional experiment:

```text
z: (1024,3).
```

The loss then forms a `(1024,1024)` metadata target and a `(1024,1024)`
embedding similarity matrix. The number 21 belongs to the temporal context of
one observation; the number 1024 is how many such observations are compared in
one optimization step.

## Limitations

- Strongly overlapping windows are statistically dependent.
- Padding changes boundary observations.
- Batch composition changes which pairwise relations are visible in one step.
- Dense pairwise objectives scale quadratically with batch size.
- Trial-level splitting prevents direct leakage but does not replace
  multi-seed or cross-session validation.

## Real-monkey input branch

The real-data preparation is implemented separately in
`src/neurobridge/experiments/real_monkey.py`. The local Area-2 recording has
193 trials, 600 bins per trial, 65 neural channels, and eight direction
classes. The loader documents 1 ms bins and 40 ms smoothing. NeuroBridge
creates a stratified trial-level split of 134/20/39 train/validation/test
trials, then constructs centered, trial-safe windows of shape `(21, 65)` with
stride 1. No window crosses a trial boundary.

These observed-data windows do not carry a simulated `Z`; they support
downstream behavioural decoding and temporal/robustness diagnostics instead.
