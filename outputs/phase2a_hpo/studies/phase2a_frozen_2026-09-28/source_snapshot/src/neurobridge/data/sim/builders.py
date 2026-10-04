"""Builders for latent trajectories, loadings, rates, spikes, and lags."""
import collections
import numpy as np
import math


def build_linear_loading_and_place_fields(
    *, k, n_neurons, position, place_fraction, place_width, place_scale,
    first_coordinates_multiplier, random_state, direction=None,
    n_position_bins=20, gradient_fraction=0.0,
    nonpreferred_direction_gain=0.10, return_metadata=False,
):
    """Build the linear loading matrix and nonlinear spatial fields."""
    if k < 2:
        raise ValueError("Linear loading requires k >= 2.")
    position = np.asarray(position)
    if position.ndim != 2:
        raise ValueError("position must have shape (n_trials, trial_length).")
    if n_position_bins < 2:
        raise ValueError("n_position_bins must be at least 2.")
    if not 0.0 <= place_fraction < 1.0:
        raise ValueError("place_fraction must satisfy 0 <= value < 1.")
    if not 0.0 <= gradient_fraction < 1.0:
        raise ValueError("gradient_fraction must satisfy 0 <= value < 1.")
    if place_fraction + gradient_fraction >= 1.0:
        raise ValueError("place_fraction + gradient_fraction must be smaller than 1.")
    if not 0.0 <= nonpreferred_direction_gain <= 1.0:
        raise ValueError("nonpreferred_direction_gain must lie between 0 and 1.")

    if direction is None:
        direction_binary = np.zeros(position.shape, dtype=int)
        for trial in range(position.shape[0]):
            direction_binary[trial, : int(np.argmax(position[trial])) + 1] = 1
    else:
        direction_array = np.asarray(direction)
        if direction_array.shape != position.shape:
            raise ValueError("direction must have the same shape as position.")
        unique_direction = set(np.unique(direction_array).tolist())
        if unique_direction.issubset({0, 1}):
            direction_binary = direction_array.astype(int)
        elif unique_direction.issubset({-1, 1}):
            direction_binary = (direction_array > 0).astype(int)
        else:
            raise ValueError("direction must contain only 0/1 or -1/+1.")

    rng = np.random.default_rng(random_state)
    B = np.zeros((k, n_neurons), dtype=float)
    names = np.array(["positional", "directional", "mixed", "gradient"])
    remaining = 1.0 - place_fraction - gradient_fraction
    probabilities = [place_fraction, remaining / 2, remaining / 2, gradient_fraction]
    neuron_types = rng.choice(names, size=n_neurons, p=probabilities)
    preferred_bins = np.full(n_neurons, -1, dtype=int)
    preferred_directions = np.full(n_neurons, -1, dtype=int)
    gradient_sign = np.zeros(n_neurons, dtype=int)
    spatial_mask = np.isin(neuron_types, ["positional", "mixed"])
    directional_mask = np.isin(neuron_types, ["directional", "mixed"])
    preferred_bins[spatial_mask] = rng.integers(0, n_position_bins, spatial_mask.sum())
    preferred_directions[directional_mask] = rng.integers(0, 2, directional_mask.sum())

    directional_indices = np.flatnonzero(neuron_types == "directional")
    B[1, directional_indices] = 2 * preferred_directions[directional_indices] - 1
    gradient_indices = np.flatnonzero(neuron_types == "gradient")
    if gradient_indices.size:
        gradient_sign[gradient_indices] = rng.choice([-1, 1], gradient_indices.size)
        B[0, gradient_indices] = gradient_sign[gradient_indices]
    B[:2] *= first_coordinates_multiplier

    bin_edges = np.linspace(0.0, 1.0, n_position_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    position_bins = np.clip(
        np.digitize(position, bin_edges[1:-1]), 0, n_position_bins - 1
    )
    centers = np.full(n_neurons, np.nan)
    centers[spatial_mask] = bin_centers[preferred_bins[spatial_mask]]
    width_bins = max(place_width * n_position_bins, 0.5)
    place_drive = np.zeros((*position.shape, n_neurons), dtype=float)
    for neuron in np.flatnonzero(spatial_mask):
        distance = position_bins - preferred_bins[neuron]
        field = place_scale * np.exp(-(distance**2) / (2 * width_bins**2))
        if neuron_types[neuron] == "mixed":
            field *= np.where(
                direction_binary == preferred_directions[neuron],
                1.0,
                nonpreferred_direction_gain,
            )
        place_drive[..., neuron] = field

    metadata = {
        "neuron_id": np.arange(n_neurons, dtype=int),
        "neuron_type": neuron_types.copy(),
        "preferred_bin": preferred_bins,
        "preferred_direction": preferred_directions,
        "gradient_sign": gradient_sign,
        "position_bin_edges": bin_edges,
        "position_bin_centers": bin_centers,
    }
    result = (B, neuron_types, centers, place_drive)
    return (*result, metadata) if return_metadata else result


def deterministic_builder(
    condition,
    k,
    condition_mode="circular",
    n_conditions=8,
    s=None,
    p=None,
    velocity=None,
    context_value=None,
):
    """
    Build the deterministic component of a latent trajectory.

    Parameters
    ----------
    condition : int or None
        Condition label used in circular mode to select the movement direction.
        It is not used in linear mode.

    k : int
        Number of latent dimensions.

    condition_mode : str
        Type of deterministic trajectory.
        Supported modes:
            - "circular"
            - "linear"

    n_conditions : int
        Total number of possible circular conditions.

    s : array-like or None
        Monotonic movement profile used in circular mode.
        It describes movement progress from 0 to 1.

    p : array-like or None
        Complete position profile used in linear mode.
        It describes outward and return movement: 0 -> 1 -> 0.

    velocity : array-like or None
        Trial-specific instantaneous velocity profile.

        It is used only when the selected latent dimensionality includes
        the velocity coordinate:
            - circular mode: k > 3
            - linear mode: k > 2

    context_value : float or None
        Trial-level context value, constant throughout one trial.

        It is used only when the selected latent dimensionality includes
        the context coordinate:
            - circular mode: k > 4
            - linear mode: k > 3

    Returns
    -------
    m_i : ndarray, shape (L, k)
        Deterministic latent trajectory.
    """

    # Linear transformation applied to the latent trajectory.
    # The identity matrix leaves the geometry unchanged.
    A = np.eye(k)

    # Translation vector in latent space.
    # A zero vector leaves the trajectory untranslated.
    b = np.zeros(k)

    if condition_mode == "circular":

        if s is None:
            raise ValueError(
                "circular condition_mode requires s"
            )

        if condition is None:
            raise ValueError(
                "circular condition_mode requires a condition"
            )

        if k < 3:
            raise ValueError(
                "circular condition_mode requires k >= 3"
            )

        # Raw deterministic trajectory before transformation.
        v_vec = np.zeros((len(s), k))

        # Angle associated with the selected condition.
        theta = 2 * np.pi * condition / n_conditions

        # Radial movement in the first two latent dimensions.
        v_vec[:, 0] = s * np.cos(theta)
        v_vec[:, 1] = s * np.sin(theta)

        # Explicit movement progress in the third latent dimension.
        v_vec[:, 2] = s

        if k > 3:
            if velocity is None:
                raise ValueError(
                    "circular condition_mode with k > 3 requires velocity"
                )

            # Fourth latent coordinate:
            # trial-specific instantaneous movement speed.
            v_vec[:, 3] = velocity

        if k > 4:
            if context_value is None:
                raise ValueError(
                    "circular condition_mode with k > 4 "
                    "requires context_value"
                )

            # Fifth latent coordinate:
            # trial-level context, constant throughout one trial.
            v_vec[:, 4] = context_value

        # Any additional latent dimensions remain equal to zero.

    elif condition_mode == "linear":

        if p is None:
            raise ValueError(
                "linear condition_mode requires p"
            )

        if k < 2:
            raise ValueError(
                "linear condition_mode requires k >= 2"
            )

        # Raw deterministic trajectory before transformation.
        v_vec = np.zeros((len(p), k))

        # Position along the linear track: 0 -> 1 -> 0.
        v_vec[:, 0] = p

        # Turning point between outward and return movement.
        turn = np.argmax(p)

        # Direction of movement:
        # +1 during outward movement, including the turning point,
        # -1 during return movement.
        v_vec[:turn + 1, 1] = 1.0
        v_vec[turn + 1:, 1] = -1.0

        if k > 2:
            if velocity is None:
                raise ValueError(
                    "linear condition_mode with k > 2 requires velocity"
                )

            # Third latent coordinate:
            # trial-specific instantaneous signed movement velocity.
            v_vec[:, 2] = velocity

        if k > 3:
            if context_value is None:
                raise ValueError(
                    "linear condition_mode with k > 3 "
                    "requires context_value"
                )

            # Fourth latent coordinate:
            # trial-level context, constant throughout one trial.
            v_vec[:, 3] = context_value

        # Any additional latent dimensions remain equal to zero.

    else:
        raise ValueError(
            f"condition_mode '{condition_mode}' is not supported"
        )

    # Apply the linear transformation and translation.
    m_i = v_vec @ A.T + b

    return m_i

def stochastic_builder(phi, L, k, noise_scale):
    """
    Build the stochastic component of a latent process.

    AR(1) process:
        eta_t = phi * eta_{t-1} + eps_t

    Parameters
    ----------
    phi : float
        Temporal persistence parameter. It must satisfy -1 < phi < 1.

    L : int
        Trial length in time bins.

    k : int
        Number of latent dimensions.

    noise_scale : float
        Standard deviation of the innovation noise.

    Returns
    -------
    eta : ndarray, shape (L, k)
        Stochastic latent trajectory.
    """

    if not (-1 < phi < 1):
        raise ValueError("phi must satisfy -1 < phi < 1")

    if L <= 0:
        raise ValueError("L must be positive")

    if k <= 0:
        raise ValueError("k must be positive")

    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative")

    eta = np.zeros((L, k))

    # Stationary standard deviation of the AR(1) process.
    sigma0 = noise_scale / np.sqrt(1 - phi**2)

    eta[0, :] = np.random.normal(
        loc=0.0,
        scale=sigma0,
        size=k,
    )

    for t in range(1, L):
        innovation = noise_scale * np.random.standard_normal(k)

        eta[t, :] = (
            phi * eta[t - 1, :]
            + innovation
        )

    return eta
        
        
def latent_to_drive(Z,B, c):

    """
    function to map latent trajectories to neural drive

    Inputs:
        Z: latent process, shape (n_trials, L, k)
        B: loading matrix, shape (k, n_neurons)
        c: baseline, shape (n_neurons,)

    Output:
        u: neural drive, shape (n_trials, L, n_neurons)
    """
    if Z.ndim != 3:
        raise ValueError("Z must have shape (n_trials, L, k)")

    if B.ndim != 2:
        raise ValueError("B must have shape (k, n_neurons)")

    if Z.shape[2] != B.shape[0]:
        raise ValueError("Z and B have incompatible shapes")

    if c.shape != (B.shape[1],):
        raise ValueError("c must have shape (n_neurons,)")


    u=Z@B+c 
    return u


def drive_to_rate(u, non_linearity="softplus"):
    '''
    function to convert neural drive to rate of firing of group of neuron

    inputs:
        u: neurla drive mapping neurons to latent space/dynamics (shape:n_trial, L, n_neurons)

    output:    
        lam_ (rate of firing) (same shpae as u)

    '''
    if non_linearity=="softplus":
        lam = np.log1p(np.exp(-np.abs(u))) + np.maximum(u, 0) 


    elif non_linearity=='exponential':
        lam=np.exp(u)

    #elif

    else:
        raise ValueError(f"Non-linearity '{non_linearity}' not supported.")


    return lam


def rate_to_spike(
    lam,
    dt=1.0,
    overdispersion=0.0,
    refractory_mean_bins=None,
    refractory_std_bins=0.0,
    burst_probability=0.0,
    burst_size_mean=0.0,
    burst_window_bins=1,
):

    '''
    function to get neural spiek for group neurons giving a firing rate

    inputs: 
        lam rate of firing  (shape: n_trials, L, n_neurons )
        dt tme interval (scalar)
        overdispersion:
            Gamma variability on lambda. If 0, pure Poisson.
            If > 0, the conditional rate is multiplied by a Gamma random factor
            with mean 1 and variance overdispersion.
        refractory_mean_bins:
            Mean refractory duration in bins after a nonzero spike count.
            If None or <= 0, no refractory correction is applied.
        refractory_std_bins:
            Standard deviation of refractory duration in bins.
        burst_probability:
            Probability that a nonzero spike count starts a short burst.
        burst_size_mean:
            Mean number of extra spikes added by a burst.
        burst_window_bins:
            Number of following bins over which burst spikes can be added.


    output:    
         X = Matrix of Spikes

    '''
    if (np.any(lam<0)): 
        raise ValueError(f"lambda must be non negative")

    if dt <= 0:
        raise ValueError("dt must be positive")

    if overdispersion < 0:
        raise ValueError("overdispersion must be non-negative")

    if refractory_std_bins < 0:
        raise ValueError("refractory_std_bins must be non-negative")

    if not (0 <= burst_probability <= 1):
        raise ValueError("burst_probability must be in [0, 1]")

    if burst_size_mean < 0:
        raise ValueError("burst_size_mean must be non-negative")

    if burst_window_bins <= 0:
        raise ValueError("burst_window_bins must be positive")

    lam_bin=lam*dt

    if overdispersion > 0:
        gamma_shape = 1.0 / overdispersion
        gamma_scale = overdispersion
        gamma_gain = np.random.gamma(gamma_shape, gamma_scale, size=lam.shape)
        lam_bin = lam_bin * gamma_gain

    X=np.random.poisson(lam_bin)

    if burst_probability > 0 and burst_size_mean > 0:
        X = _add_bursts(
            X,
            burst_probability=burst_probability,
            burst_size_mean=burst_size_mean,
            burst_window_bins=burst_window_bins,
        )

    if refractory_mean_bins is not None and refractory_mean_bins > 0:
        X = _apply_refractory_period(
            X,
            refractory_mean_bins=refractory_mean_bins,
            refractory_std_bins=refractory_std_bins,
        )

    return X


def apply_temporal_lag(Z, lag_bins=0, pad_mode="edge"):
    """
    Shift a trial-wise time series by a lag without wrapping across time.

    Parameters
    ----------
    Z : ndarray
        Tensor with shape (n_trials, L, n_features).
    lag_bins : int
        Positive values delay the signal. Negative values advance it.
    pad_mode : str
        Currently only "edge" is supported. Missing samples are filled by
        repeating the first or last valid time bin.

    Returns
    -------
    Z_lagged : ndarray
        Lagged tensor with the same shape as Z.
    """
    Z = np.asarray(Z)

    if Z.ndim != 3:
        raise ValueError("Z must have shape (n_trials, L, n_features)")

    if pad_mode != "edge":
        raise ValueError("Only pad_mode='edge' is currently supported")

    lag_bins = int(lag_bins)

    if lag_bins == 0:
        return Z.copy()

    Z_lagged = np.empty_like(Z)

    if lag_bins > 0:
        if lag_bins >= Z.shape[1]:
            Z_lagged[:] = Z[:, :1, :]
        else:
            Z_lagged[:, :lag_bins, :] = Z[:, :1, :]
            Z_lagged[:, lag_bins:, :] = Z[:, :-lag_bins, :]
    else:
        lag_abs = abs(lag_bins)
        if lag_abs >= Z.shape[1]:
            Z_lagged[:] = Z[:, -1:, :]
        else:
            Z_lagged[:, :-lag_abs, :] = Z[:, lag_abs:, :]
            Z_lagged[:, -lag_abs:, :] = Z[:, -1:, :]

    return Z_lagged


def _sample_refractory_bins(refractory_mean_bins, refractory_std_bins):
    if refractory_std_bins == 0:
        refractory_bins = refractory_mean_bins
    else:
        refractory_bins = np.random.normal(refractory_mean_bins, refractory_std_bins)

    refractory_bins = int(np.round(refractory_bins))
    return max(refractory_bins, 0)


def _apply_refractory_period(X, refractory_mean_bins, refractory_std_bins=0.0):
    """
    Apply a bin-level refractory correction.

    If a neuron spikes in a bin, following bins are forced to zero for a
    refractory duration sampled from a normal distribution and rounded to bins.
    """
    X = np.asarray(X).copy()

    if X.ndim != 3:
        raise ValueError("X must have shape (n_trials, L, n_neurons)")

    n_trials, L, n_neurons = X.shape
    refractory_counter = np.zeros((n_trials, n_neurons), dtype=int)

    for t in range(L):
        in_refractory = refractory_counter > 0
        X[:, t, :][in_refractory] = 0

        fired = X[:, t, :] > 0

        refractory_counter[refractory_counter > 0] -= 1

        if np.any(fired):
            fired_trials, fired_neurons = np.where(fired)
            for trial_idx, neuron_idx in zip(fired_trials, fired_neurons):
                refractory_counter[trial_idx, neuron_idx] = _sample_refractory_bins(
                    refractory_mean_bins,
                    refractory_std_bins,
                )

    return X


def _add_bursts(X, burst_probability, burst_size_mean, burst_window_bins=1):
    """
    Add short spike clusters after some nonzero spike bins.

    This is a simple phenomenological burst model: a spike bin can trigger
    extra spikes distributed over the current or following bins.
    """
    X = np.asarray(X).copy()

    if X.ndim != 3:
        raise ValueError("X must have shape (n_trials, L, n_neurons)")

    n_trials, L, n_neurons = X.shape
    burst_starts = (X > 0) & (np.random.random(size=X.shape) < burst_probability)
    burst_trials, burst_times, burst_neurons = np.where(burst_starts)

    for trial_idx, time_idx, neuron_idx in zip(burst_trials, burst_times, burst_neurons):
        n_extra = np.random.poisson(burst_size_mean)
        for _ in range(n_extra):
            lag = np.random.randint(0, burst_window_bins)
            add_time = time_idx + lag
            if add_time < L:
                X[trial_idx, add_time, neuron_idx] += 1

    return X


def build_structured_B(
    k,
    n_neurons,
    conditions,
    n_conditions,
    condition_mode="circular",
    cluster_spikes=True,
    directional_scale=1.0,
    extra_scale=0.1,
    position_scale=1.0,
    velocity_scale=1.0,
    context_scale=1.0,
    neuron_type_probabilities=None,
    random_state=None,
    return_neuron_types=False,
):
    """
    Build a structured loading matrix B.

    Each row of B corresponds to one latent coordinate.
    Each column of B corresponds to one neuron.

    The neural drive is computed as:

        u = Z @ B + c

    Therefore, B[r, j] indicates how strongly neuron j is influenced
    by latent coordinate r.

    Circular latent coordinates
    ---------------------------
    row 0:
        s * cos(theta)

    row 1:
        s * sin(theta)

    row 2:
        movement progress

    row 3, if k > 3:
        movement velocity

    row 4, if k > 4:
        trial-level context

    Linear latent coordinates
    -------------------------
    row 0:
        position

    row 1:
        movement direction:
            +1 = outward movement
            -1 = return movement

    row 2, if k > 2:
        movement velocity

    row 3, if k > 3:
        trial-level context

    Neuron types
    ------------
    "direction":
        Sensitive only to movement direction.

    "position_or_progress":
        Sensitive only to position in the linear task or progress in the
        circular task.

    "velocity":
        Sensitive only to movement velocity.

    "context":
        Sensitive only to trial-level context.

    "mixed":
        Sensitive to at least two task factors.

    "none":
        No task-related loading. Its complete B column remains zero.
        Its activity can still depend on the baseline c and spike noise.

    Parameters
    ----------
    k : int
        Number of latent dimensions.

    n_neurons : int
        Number of simulated neurons.

    conditions : array-like or None
        Allowed circular conditions.

        With cluster_spikes=True, directional preferences are sampled from
        these discrete conditions.

        It is not used in linear mode.

    n_conditions : int
        Total number of circular directions.

    condition_mode : str
        Supported task modes:
            - "circular"
            - "linear"

    cluster_spikes : bool
        In circular mode:

        If True:
            preferred directions are selected from the discrete conditions.

        If False:
            preferred angles are sampled continuously over the circle.

    directional_scale : float
        Scale of directional weights.

    extra_scale : float
        Preserved for compatibility with existing calls.

        It is not used to fill all extra dimensions automatically anymore,
        because velocity and context now have explicit meanings and their own
        scales.

    position_scale : float
        Scale of position or progress weights.

    velocity_scale : float
        Scale of velocity weights.

    context_scale : float
        Scale of context weights.

    neuron_type_probabilities : dict or None
        Probability assigned to each neuron type.

        Expected keys:
            - "direction"
            - "position_or_progress"
            - "velocity"
            - "context"
            - "mixed"
            - "none"

    random_state : int or None
        Seed used for reproducible random assignments.

    return_neuron_types : bool
        If True, return both B and the assigned neuron types.

    Returns
    -------
    B : ndarray, shape (k, n_neurons)
        Loading matrix.

    neuron_types : ndarray, shape (n_neurons,), optional
        Type assigned to each neuron.
        Returned only when return_neuron_types=True.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive")

    if n_conditions <= 0:
        raise ValueError("n_conditions must be positive")

    if condition_mode not in {"circular", "linear"}:
        raise ValueError(
            "condition_mode must be 'circular' or 'linear'"
        )

    if directional_scale < 0:
        raise ValueError(
            "directional_scale must be non-negative"
        )

    if position_scale < 0:
        raise ValueError(
            "position_scale must be non-negative"
        )

    if velocity_scale < 0:
        raise ValueError(
            "velocity_scale must be non-negative"
        )

    if context_scale < 0:
        raise ValueError(
            "context_scale must be non-negative"
        )

    if condition_mode == "circular" and k < 3:
        raise ValueError(
            "circular condition_mode requires k >= 3"
        )

    if condition_mode == "linear" and k < 2:
        raise ValueError(
            "linear condition_mode requires k >= 2"
        )

    # Random-number generator.
    #
    # Supplying random_state makes the assignment reproducible.
    rng = np.random.default_rng(random_state)

    # Initialize every loading to zero.
    #
    # Neuron-specific weights are assigned below according to neuron type.
    B = np.zeros((k, n_neurons))

    neuron_type_names = np.array([
        "direction",
        "position_or_progress",
        "velocity",
        "context",
        "mixed",
        "none",
    ])

    if neuron_type_probabilities is None:

        # Default population composition.
        #
        # These probabilities are modelling choices rather than biological
        # constants. They can be changed when testing different simulated
        # population structures.
        neuron_type_probabilities = {
            "direction": 0.20,
            "position_or_progress": 0.20,
            "velocity": 0.15,
            "context": 0.10,
            "mixed": 0.30,
            "none": 0.05,
        }

    missing_types = [
        neuron_type
        for neuron_type in neuron_type_names
        if neuron_type not in neuron_type_probabilities
    ]

    if len(missing_types) > 0:
        raise ValueError(
            "neuron_type_probabilities is missing the following keys: "
            + ", ".join(missing_types)
        )

    # Convert the probability dictionary into an array.
    #
    # The order must match neuron_type_names because rng.choice associates
    # each probability with the item at the same position.
    probabilities = np.zeros(
        len(neuron_type_names),
        dtype=float,
    )

    for i, neuron_type in enumerate(neuron_type_names):
        probabilities[i] = neuron_type_probabilities[
            neuron_type
        ]

    if np.any(probabilities < 0):
        raise ValueError(
            "neuron-type probabilities must be non-negative"
        )

    if np.any(probabilities > 1):
        raise ValueError(
            "neuron-type probabilities must not exceed 1"
        )

    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError(
            "neuron-type probabilities must sum to 1"
        )

    # Some neuron types cannot exist when the corresponding latent coordinate
    # is absent.
    #
    # Their probability is therefore set to zero and the remaining
    # probabilities are normalized again. This prevents, for example, a
    # velocity neuron from being selected when the latent space contains no
    # velocity coordinate.
    velocity_type_index = np.where(
        neuron_type_names == "velocity"
    )[0][0]

    context_type_index = np.where(
        neuron_type_names == "context"
    )[0][0]

    if condition_mode == "circular":

        velocity_is_available = k > 3
        context_is_available = k > 4

    else:

        velocity_is_available = k > 2
        context_is_available = k > 3

    if not velocity_is_available:
        probabilities[velocity_type_index] = 0.0

    if not context_is_available:
        probabilities[context_type_index] = 0.0

    probability_sum = probabilities.sum()

    if probability_sum <= 0:
        raise ValueError(
            "no available neuron type has positive probability"
        )

    probabilities = probabilities / probability_sum

    # Assign one type to every neuron.
    neuron_types = rng.choice(
        neuron_type_names,
        size=n_neurons,
        p=probabilities,
    )

    if condition_mode == "circular":

        if conditions is None:
            conditions = np.arange(n_conditions)

        conditions = np.asarray(conditions)

        if conditions.ndim != 1:
            raise ValueError(
                "conditions must be one-dimensional"
            )

        if len(conditions) == 0:
            raise ValueError(
                "circular condition_mode requires at least one condition"
            )

        if np.any(conditions < 0) or np.any(
            conditions >= n_conditions
        ):
            raise ValueError(
                "circular conditions must satisfy "
                "0 <= condition < n_conditions"
            )

        for j in range(n_neurons):

            neuron_type = neuron_types[j]

            if neuron_type == "direction":

                if cluster_spikes:

                    # Select one preferred discrete movement condition.
                    preferred_condition = rng.choice(
                        conditions
                    )

                    # Convert the preferred condition into an angle in radians.
                    preferred_angle = (
                        2.0
                        * np.pi
                        * preferred_condition
                        / n_conditions
                    )

                else:

                    # Select one continuous preferred angle.
                    preferred_angle = rng.uniform(
                        0.0,
                        2.0 * np.pi,
                    )

                # Directional tuning is represented by a two-dimensional
                # preferred-direction vector.
                B[0, j] = (
                    directional_scale
                    * np.cos(preferred_angle)
                )

                B[1, j] = (
                    directional_scale
                    * np.sin(preferred_angle)
                )

            elif neuron_type == "position_or_progress":

                # Circular progress is stored in latent coordinate 2.
                #
                # The positive weight makes the neuron's task-related drive
                # increase as movement progresses.
                B[2, j] = rng.uniform(
                    0.5 * position_scale,
                    1.5 * position_scale,
                )

            elif neuron_type == "velocity":

                # Circular velocity is stored in latent coordinate 3.
                #
                # Positive and negative weights allow neurons whose firing
                # increases or decreases with movement speed.
                sign = rng.choice([-1.0, 1.0])

                B[3, j] = (
                    sign
                    * rng.uniform(
                        0.5 * velocity_scale,
                        1.5 * velocity_scale,
                    )
                )

            elif neuron_type == "context":

                # Circular context is stored in latent coordinate 4.
                #
                # Positive and negative weights allow opposite context
                # modulations across neurons.
                sign = rng.choice([-1.0, 1.0])

                B[4, j] = (
                    sign
                    * rng.uniform(
                        0.5 * context_scale,
                        1.5 * context_scale,
                    )
                )

            elif neuron_type == "mixed":

                if cluster_spikes:

                    # Select one preferred discrete movement condition.
                    preferred_condition = rng.choice(
                        conditions
                    )

                    preferred_angle = (
                        2.0
                        * np.pi
                        * preferred_condition
                        / n_conditions
                    )

                else:

                    # Select one continuous preferred angle.
                    preferred_angle = rng.uniform(
                        0.0,
                        2.0 * np.pi,
                    )

                # Every mixed circular neuron receives directional tuning.
                B[0, j] = (
                    directional_scale
                    * np.cos(preferred_angle)
                )

                B[1, j] = (
                    directional_scale
                    * np.sin(preferred_angle)
                )

                # Build the list of additional factors that are available.
                additional_factors = ["progress"]

                if k > 3:
                    additional_factors.append("velocity")

                if k > 4:
                    additional_factors.append("context")

                # Select at least one additional factor.
                #
                # A mixed neuron therefore contains directional tuning plus
                # one or more other task-related sensitivities.
                n_additional = rng.integers(
                    1,
                    len(additional_factors) + 1,
                )

                selected_factors = rng.choice(
                    additional_factors,
                    size=n_additional,
                    replace=False,
                )

                for selected_factor in selected_factors:

                    if selected_factor == "progress":

                        B[2, j] = rng.uniform(
                            0.5 * position_scale,
                            1.5 * position_scale,
                        )

                    elif selected_factor == "velocity":

                        sign = rng.choice([-1.0, 1.0])

                        B[3, j] = (
                            sign
                            * rng.uniform(
                                0.5 * velocity_scale,
                                1.5 * velocity_scale,
                            )
                        )

                    elif selected_factor == "context":

                        sign = rng.choice([-1.0, 1.0])

                        B[4, j] = (
                            sign
                            * rng.uniform(
                                0.5 * context_scale,
                                1.5 * context_scale,
                            )
                        )

            elif neuron_type == "none":

                # The complete column remains equal to zero.
                #
                # This neuron has no deterministic task-related tuning in B.
                pass

    elif condition_mode == "linear":

        for j in range(n_neurons):

            neuron_type = neuron_types[j]

            if neuron_type == "direction":

                # Select whether the neuron prefers outward or return movement.
                #
                # Positive loading:
                #     outward preference
                #
                # Negative loading:
                #     return preference
                preferred_direction = rng.choice([
                    -1.0,
                    1.0,
                ])

                B[1, j] = (
                    directional_scale
                    * preferred_direction
                )

            elif neuron_type == "position_or_progress":

                # Linear position is stored in latent coordinate 0.
                #
                # Positive and negative weights produce different monotonic
                # relationships between position and neural drive.
                sign = rng.choice([-1.0, 1.0])

                B[0, j] = (
                    sign
                    * rng.uniform(
                        0.5 * position_scale,
                        1.5 * position_scale,
                    )
                )

            elif neuron_type == "velocity":

                # Linear velocity is stored in latent coordinate 2.
                sign = rng.choice([-1.0, 1.0])

                B[2, j] = (
                    sign
                    * rng.uniform(
                        0.5 * velocity_scale,
                        1.5 * velocity_scale,
                    )
                )

            elif neuron_type == "context":

                # Linear context is stored in latent coordinate 3.
                sign = rng.choice([-1.0, 1.0])

                B[3, j] = (
                    sign
                    * rng.uniform(
                        0.5 * context_scale,
                        1.5 * context_scale,
                    )
                )

            elif neuron_type == "mixed":

                # List the task factors available in the current latent space.
                available_factors = [
                    "position",
                    "direction",
                ]

                if k > 2:
                    available_factors.append("velocity")

                if k > 3:
                    available_factors.append("context")

                # A mixed neuron must depend on at least two factors.
                n_selected = rng.integers(
                    2,
                    len(available_factors) + 1,
                )

                selected_factors = rng.choice(
                    available_factors,
                    size=n_selected,
                    replace=False,
                )

                for selected_factor in selected_factors:

                    if selected_factor == "position":

                        sign = rng.choice([-1.0, 1.0])

                        B[0, j] = (
                            sign
                            * rng.uniform(
                                0.5 * position_scale,
                                1.5 * position_scale,
                            )
                        )

                    elif selected_factor == "direction":

                        preferred_direction = rng.choice([
                            -1.0,
                            1.0,
                        ])

                        B[1, j] = (
                            directional_scale
                            * preferred_direction
                        )

                    elif selected_factor == "velocity":

                        sign = rng.choice([-1.0, 1.0])

                        B[2, j] = (
                            sign
                            * rng.uniform(
                                0.5 * velocity_scale,
                                1.5 * velocity_scale,
                            )
                        )

                    elif selected_factor == "context":

                        sign = rng.choice([-1.0, 1.0])

                        B[3, j] = (
                            sign
                            * rng.uniform(
                                0.5 * context_scale,
                                1.5 * context_scale,
                            )
                        )

            elif neuron_type == "none":

                # The complete column remains equal to zero.
                pass

    # Normalize only the mixed-neuron columns.
    #
    # Without normalization, mixed neurons would tend to receive a stronger
    # total drive simply because they contain more nonzero weights.
    mixed_indices = np.where(
        neuron_types == "mixed"
    )[0]

    for j in mixed_indices:

        column_norm = np.linalg.norm(
            B[:, j]
        )

        if column_norm > 0:
            B[:, j] = (
                B[:, j]
                / column_norm
            )

    if return_neuron_types:
        return B, neuron_types

    return B


def build_direction_dominant_B(
    k,
    n_neurons,
    conditions,
    n_conditions,
    directional_scale=3.0,
    extra_scale=0.051,
    random_state=None,
):
    """
    Build the direction-dominant loading used by the original circular suite.

    Every neuron receives a preferred-direction loading in the first two
    latent coordinates. Remaining coordinates receive weak random loadings.
    Preferred conditions are distributed as evenly as possible across the
    population.
    """
    if k < 2:
        raise ValueError("direction-dominant loading requires k >= 2")
    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive")
    if n_conditions <= 0:
        raise ValueError("n_conditions must be positive")
    if directional_scale < 0 or extra_scale < 0:
        raise ValueError("loading scales must be non-negative")

    conditions = np.asarray(conditions, dtype=int)
    if conditions.ndim != 1 or len(conditions) == 0:
        raise ValueError("conditions must be a non-empty one-dimensional array")

    rng = np.random.default_rng(random_state)
    preferred_conditions = np.resize(conditions, n_neurons)
    preferred_angles = (
        2.0 * np.pi * preferred_conditions / n_conditions
    )

    B = np.zeros((k, n_neurons), dtype=float)
    B[0, :] = directional_scale * np.cos(preferred_angles)
    B[1, :] = directional_scale * np.sin(preferred_angles)

    if k > 2:
        B[2:, :] = extra_scale * rng.standard_normal(
            size=(k - 2, n_neurons)
        )

    return B
