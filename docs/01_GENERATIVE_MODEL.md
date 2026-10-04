# Generative Model

## Purpose And Scope

NeuroBridge generates neural observations from a known low-dimensional task
state. The known state provides ground truth for testing representation
learning:

```text
task state -> latent trajectory Z -> neural drive u
           -> firing rate lambda -> spike counts X
```

This is an observation-level statistical simulator. It does not simulate ion
channels, membrane voltage, or the biophysical cause of individual action
potentials.

## Tensor Definitions

Let:

- `R` be the number of trials;
- `T` be the number of time bins per trial;
- `K` be the latent dimension;
- `N` be the number of simulated neurons.

The main tensors are:

```text
Z       : (R, T, K)  known task-level latent state
B       : (K, N)     latent-to-neuron loading matrix
c       : (N,)       neuron baseline
u       : (R, T, N)  neural drive
lambda  : (R, T, N)  firing rate in spikes/second
X       : (R, T, N)  observed spike counts
```

For the current shared-latent center-out run, `R=200`, `T=200`, and the
subject-specific neuron counts are `N_A=160` and `N_B=120`.

### How To Read The Equations

Subscripts identify an object:

- $r$: trial;
- $t$: time bin inside that trial;
- $j$: neuron;
- $k$: latent coordinate.

For example, $X_{r,t,j}$ is one number: the spike count of neuron $j$ at
time $t$ in trial $r$. A bold or unsubscripted capital such as $Z$
denotes the complete array of those values.

## Deterministic Task State And Trial Variability

For trial $r$, the latent trajectory is:

```math
Z_r(t)=m_r(t)+\eta_r(t).
```

Here:

- $Z_r(t)$ is the complete $K$-dimensional latent vector at one time;
- $m_r(t)$ is the deterministic trajectory prescribed by the task;
- $\eta_r(t)$ is a random $K$-dimensional perturbation specific to that
  trial.

The perturbation is temporally correlated:

```math
\eta_r(t)
=
\phi\,\eta_r(t-1)+\epsilon_r(t),
\qquad
\epsilon_r(t)\sim
\mathcal{N}\!\left(0,\sigma_{\mathrm{noise}}^2 I_K\right).
```

$\phi$ controls memory: at zero, perturbations are independent over time; as
$\phi$ approaches one, deviations persist longer. $I_K$ is the
$K\times K$ identity matrix and
$\sigma_{\mathrm{noise}}$ is `noise_scale`.

The default controlled configuration uses `phi=0.4` and
`noise_scale=0.05`. This creates smooth trial-to-trial perturbations without
changing the known task family.

## Circular Reaching Task

Each trial starts from a common center and reaches one of eight targets. Let
$\theta$ be the target angle and let $s(t)$ be movement progress. The
planar position is:

```math
x(t)=s(t)\cos\theta,
\qquad
y(t)=s(t)\sin\theta.
```

When $s(t)=0$, every condition is at the common center. When $s(t)=1$, the
point reaches the unit-circle target $(\cos\theta,\sin\theta)$.

The smooth movement profile is:

```math
s(t)=10t^3-15t^4+6t^5,
\qquad 0\le t\le1.
```

In this equation $t$ is normalized trial time, not seconds. The polynomial
maps normalized time to normalized progress.

This profile starts and ends with zero velocity and acceleration. It is a
controlled minimum-jerk-like interpolation, not a complete model of biological
reaching.

The essential circular latent is:

```text
Z(t) = [x(t), y(t), progress(t)].
```

The enriched version adds velocity and trial context:

```text
Z(t) = [x(t), y(t), progress(t), velocity(t), context].
```

## Linear Track Task

One trial contains both movement phases:

```text
outbound: position 0 -> 1, direction +1
return:   position 1 -> 0, direction -1
```

The physical behavior is one-dimensional, but the essential latent is
two-dimensional:

```text
Z(t) = [position(t), direction(t)].
```

This distinction separates two visits to the same position made in opposite
directions. In the position-direction plane, the outbound and return branches
form a cycle. The enriched state adds velocity and context.

## Neural Population Map

The linear part of the population observation model is:

```math
u_r(t)=Z_r(t)B+c+g_r(t).
```

The matrix multiplication has explicit dimensions:

```math
\underbrace{Z_r(t)}_{1\times K}
\underbrace{B}_{K\times N}
=
\underbrace{Z_r(t)B}_{1\times N}.
```

Therefore the result contains one drive value for each of the $N$ neurons.
The baseline $c$ and optional place-field term $g_r(t)$ are also
$N$-dimensional vectors. Column $j$ of $B$ specifies how neuron $j$
weights all $K$ latent coordinates.

The population is heterogeneous. Neurons can emphasize direction, progress,
position, velocity, context, or mixtures. The
`first_coordinates_multiplier` parameter increases loading magnitude for the
task-defining coordinates; the controlled default is `3.0`.

This is an experimental control, not a discovered biological law. Increasing
it raises the signal-to-noise ratio of the motor core and can make recovery
easier.

## Linear-Track Place Fields

A subset of linear-track neurons receives a localized Gaussian drive:

```math
g_j(p)
=
a_j
\exp\left[
-\frac{(p-\mu_j)^2}{2\sigma_j^2}
\right].
```

Here:

- $p\in[0,1]$ is current track position;
- `mu_j` is neuron `j`'s preferred track position;
- `sigma_j` is its field width;
- `a_j` is its gain.

At $p=\mu_j$, the squared distance is zero and $g_j(p)=a_j$, its maximum.
Moving away from $\mu_j$ increases the negative exponent and lowers the
drive. Larger $\sigma_j$ produces a broader preferred region.

The controlled defaults are `place_fraction=0.25`,
`place_width=0.10`, and `place_scale=3.0`.

The place term remains separate from `ZB` because a localized Gaussian field
is not linear in position. It also provides neuron-level ground truth for a
future explainability test: an attribution method should identify the neurons
whose preferred fields cover the queried position. That XAI analysis is not
yet part of the validated suite.

## From Drive To Firing Rate

The controlled notebooks use:

```math
\lambda_{r,t,j}
=
r_{\mathrm{scale}}\,
\mathrm{softplus}(u_{r,t,j}),
```

with

```math
\mathrm{softplus}(u)=\log(1+\exp u).
```

`softplus` guarantees positive rates while remaining smooth. In the current
shared-latent run, `rate_scale=1.0` and `lambda` is expressed in spikes/second.
This operation is
element-wise: every drive value $u_{r,t,j}$ becomes one nonnegative firing
rate $\lambda_{r,t,j}$.

The baseline vector is sampled around `baseline_mean=1.0` with
`baseline_std=0.10`. Changing these quantities changes the overall rate regime
and therefore the difficulty of recovery.

## From Rate To Spike Counts

For bin width $\Delta t$, the expected number of spikes in one bin is:

```math
\mu_{r,t,j}=\lambda_{r,t,j}\Delta t.
```

The observed count is sampled as:

```math
X_{r,t,j}\sim
\mathrm{Poisson}(\mu_{r,t,j})
=
\mathrm{Poisson}(\lambda_{r,t,j}\Delta t).
```

The symbol $\sim$ means "is randomly sampled from," not "is equal to."
For example, a rate of 20 spikes/second and a bin width of 0.02 seconds give an
expected count of $20\times0.02=0.4$ spikes in that bin. The realized count
can be 0, 1, 2, and so on.

The current shared-latent run uses `dt=0.02` seconds, so one trial contains 200 bins
covering four seconds. `X` contains integer counts per bin, not exact spike
times.

Optional mechanisms are implemented:

- **Overdispersion:** a Gamma-distributed multiplicative rate gain is sampled
  before the Poisson count, producing variance above the Poisson mean.
- **Bursting:** a nonzero count can trigger additional short-lived counts with
  configurable probability and intensity.
- **Refractory suppression:** after a nonzero bin, subsequent bins can be
  suppressed for a normally distributed number of bins.

These mechanisms are phenomenological. They make observations statistically
richer but do not make the simulator biophysical. In the current
shared-latent run, `overdispersion=4.0`, `refractory_mean_bins=1`, and
`refractory_std_bins=0.25`; `burst_probability=0` and `burst_size_mean=0`
disable only the burst mechanism. Thus the Poisson equation above is the base
emission model, but the run is not pure Poisson: a Gamma rate gain is applied
before Poisson sampling and refractory suppression is applied afterwards.

## Subject-Specific Populations And Lag

The same task generator can produce a shared task state for two subjects while
using different loading matrices, baselines, neuron counts, and stochastic
emissions. Thus the task geometry is common but the neural observations are
not identical.

A positive temporal lag shifts one subject's latent trajectory later within
each trial. Boundary values are padded and samples never wrap into another
trial. If the lag is 10 bins and `dt=0.02`, the imposed delay is 200 ms.

This creates controlled ground truth for asking whether learned representations
recover both shared geometry and temporal asymmetry.

## Main Assumptions And Limits

- The latent state is specified by the researcher rather than inferred from
  biology.
- The linear map `B` is deliberately interpretable but simplified.
- The base Poisson layer alone has Fano factor near one and lacks history
  dependence; the current run adds Gamma overdispersion and refractory
  suppression.
- Burst mechanisms are available but disabled in the current run.
- Results depend on rate scale, neuron tuning mixture, latent noise, bin width,
  and trial count.
- Good recovery can demonstrate estimator behavior under known conditions; it
  cannot establish that the same latent process generated real neural data.
