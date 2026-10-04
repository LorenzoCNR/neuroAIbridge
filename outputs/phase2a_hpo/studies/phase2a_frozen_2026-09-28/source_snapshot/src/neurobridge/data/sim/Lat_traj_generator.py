"""Latent trajectory generator for synthetic motor-task simulations."""
import os
import numpy as np
import sys
from random import choice
from .builders import deterministic_builder, stochastic_builder


class LatentTrajectoryGenerator:
    """
    Generate synthetic latent trajectories for motor-task simulations.

    Each latent trajectory is obtained as:

        Z_i = m_i + eta_i

    where:
        m_i   is the deterministic task trajectory;
        eta_i is a stochastic AR(1) component.

    Parameters
    ----------
    n_trials : int
        Number of simulated trials.

    L : int
        Trial length in time bins.

    k : int
        Number of latent dimensions.

        Circular task:
            k = 3 -> x, y, progress
            k = 4 -> x, y, progress, velocity
            k >= 5 -> x, y, progress, velocity, context

        Linear task:
            k = 2 -> position, direction
            k = 3 -> position, direction, velocity
            k >= 4 -> position, direction, velocity, context

    phi : float
        AR(1) persistence parameter for the stochastic component.

    conditions : array-like or None
        Available trial conditions.

        In circular mode, if conditions is None, the default conditions are:
            0, 1, ..., n_conditions - 1

        In linear mode, conditions is normally None.

    condition_mode : str
        Type of task:
            - "circular"
            - "linear"

    n_conditions : int
        Total number of possible circular directions.

    noise_scale : float
        Standard deviation of the AR(1) innovation noise.

    condition_type : str
        Method used to assign conditions to trials:
            - "balanced"
            - "randomized"

    speed_range : tuple of float
        Range used to sample one trial-specific speed scale.
        The sampled scale is used only when the selected latent dimensionality
        includes the velocity coordinate.

    context_range : tuple of float
        Range used to sample one trial-level context value.
        The sampled value is used only when the selected latent dimensionality
        includes the context coordinate.
    """

    def __init__(
        self,
        n_trials,
        L,
        k,
        phi,
        conditions=None,
        condition_mode="circular",
        n_conditions=8,
        noise_scale=0.05,
        condition_type="balanced",
        speed_range=(0.8, 1.2),
        context_range=(0.7, 1.3),
    ):

        if n_trials <= 0:
            raise ValueError("n_trials must be positive")

        if L <= 0:
            raise ValueError("L must be positive")

        if k <= 0:
            raise ValueError("k must be positive")

        if n_conditions <= 0:
            raise ValueError("n_conditions must be positive")

        if condition_mode not in {"circular", "linear"}:
            raise ValueError(
                "condition_mode must be 'circular' or 'linear'"
            )

        if condition_type not in {"balanced", "randomized"}:
            raise ValueError(
                "condition_type must be 'balanced' or 'randomized'"
            )

        if (
            speed_range[0] <= 0
            or speed_range[0] >= speed_range[1]
        ):
            raise ValueError(
                "speed_range must contain two positive increasing values"
            )

        if context_range[0] >= context_range[1]:
            raise ValueError(
                "context_range must contain two increasing values"
            )

        if condition_mode == "circular" and k < 3:
            raise ValueError(
                "circular condition_mode requires k >= 3"
            )

        if condition_mode == "linear" and k < 2:
            raise ValueError(
                "linear condition_mode requires k >= 2"
            )

        self.n_trials = n_trials
        self.trial_length = L
        self.latent_space_dimension = k
        self.AR_parameter = phi
        self.condition_mode = condition_mode
        self.n_conditions = n_conditions
        self.condition_type = condition_type
        self.noise_scale = noise_scale
        self.speed_range = speed_range
        self.context_range = context_range

        if condition_mode == "circular":
            if conditions is None:
                self.labels = np.arange(n_conditions)
            else:
                self.labels = np.asarray(conditions)

            if len(self.labels) == 0:
                raise ValueError(
                    "circular condition_mode requires at least one condition"
                )

            if np.any(self.labels < 0) or np.any(
                self.labels >= n_conditions
            ):
                raise ValueError(
                    "circular conditions must be between "
                    "0 and n_conditions - 1"
                )

        else:
            # The linear task does not use discrete trial conditions.
            self.labels = None

    def _build_movement_profile(self, speed_scale=1.0):
        """
        Build the deterministic temporal profiles used by the two tasks.

        When the latent dimensionality includes velocity, speed_scale changes
        movement timing across trials while preserving the same endpoints.

        Returns
        -------
        t : ndarray, shape (L,)
            Normalized trial time.

        s : ndarray, shape (L,)
            Circular movement progress:
                0 -> 1

        p : ndarray, shape (L,)
            Linear-track position:
                0 -> 1 -> 0

        circular_velocity : ndarray, shape (L,)
            Instantaneous circular movement speed.

        linear_velocity : ndarray, shape (L,)
            Instantaneous signed linear movement velocity.
        """
        t = np.linspace(0.0, 1.0, self.trial_length)

        # Trial-specific temporal progression.
        # The power transformation preserves the interval endpoints 0 and 1.
        t_scaled = t ** (1.0 / speed_scale)

        # Profilo radiale: 0 -> 1
        s = 10*t_scaled**3 - 15*t_scaled**4 + 6*t_scaled**5

        # Instantaneous circular movement speed.
        circular_velocity = np.gradient(s, t)

        # Profilo lineare smooth: 0 -> 1 -> 0
        half = self.trial_length // 2

        t_out = np.linspace(0.0, 1.0, half, endpoint=False)
        t_back = np.linspace(0.0, 1.0, self.trial_length - half)

        # Apply the same trial-specific temporal scaling to both phases.
        t_out_scaled = t_out ** (1.0 / speed_scale)
        t_back_scaled = t_back ** (1.0 / speed_scale)

        # Andata smooth: 0 -> 1
        p_out = (
            10*t_out_scaled**3
            - 15*t_out_scaled**4
            + 6*t_out_scaled**5
        )

        # Ritorno smooth: 1 -> 0
        progress_back = (
            10*t_back_scaled**3
            - 15*t_back_scaled**4
            + 6*t_back_scaled**5
        )
        p_back = 1.0 - progress_back

        p = np.concatenate([p_out, p_back])

        # Instantaneous signed linear movement velocity.
        linear_velocity = np.gradient(p, t)

        return t, s, p, circular_velocity, linear_velocity

    def _build_deterministic(
        self,
        condition,
        s,
        p,
        velocity,
        context_value,
    ):
        """
        Build the deterministic latent trajectory for one trial.
        """
        return deterministic_builder(
            condition=condition,
            s=s,
            p=p,
            velocity=velocity,
            context_value=context_value,
            k=self.latent_space_dimension,
            condition_mode=self.condition_mode,
            n_conditions=self.n_conditions,
        )

    def _build_task_state(
        self,
        condition,
        s,
        p,
        velocity,
        context_value,
        speed_scale,
    ):
        """
        Build interpretable task variables associated with one trial.
        """

        if self.condition_mode == "circular":

            if condition is None:
                raise ValueError(
                    "circular condition_mode requires a condition"
                )

            phase = s.copy()

            direction_angle = (
                2 * np.pi * condition / self.n_conditions
            )

            direction_vector = np.array([
                np.cos(direction_angle),
                np.sin(direction_angle),
            ])

            position = s[:, None] * direction_vector[None, :]

        elif self.condition_mode == "linear":

            # For the linear task, phase follows position: 0 -> 1 -> 0.
            phase = p.copy()

            direction_angle = np.nan
            direction_vector = np.full(2, np.nan)

            position = np.column_stack([
                p,
                np.zeros_like(p),
            ])

        else:
            raise ValueError(
                f"condition_mode '{self.condition_mode}' is not supported"
            )

        return {
            "phase": phase,
            "position": position,
            "direction_angle": direction_angle,
            "direction_vector": direction_vector,
            "velocity": velocity.copy(),
            "context": context_value,
            "speed_scale": speed_scale,
        }

    # @staticmethod
    def _build_stochastic(self):
        """
        Build the stochastic AR(1) trajectory for one trial.
        """
        return stochastic_builder(
            self.AR_parameter,
            self.trial_length,
            self.latent_space_dimension,
            self.noise_scale,
        )

    def _select_condition(self, i):
        """
        Select the condition of one trial.
        """
        if self.labels is None:
            return None

        if self.condition_type == "balanced":
            return self.labels[i % len(self.labels)]

        elif self.condition_type == "randomized":
            return choice(self.labels)

        else:
            raise ValueError(
                "condition_type must be either 'balanced' or 'randomized'"
            )

    def _sample_speed_scale(self):
        """
        Sample one movement-speed scale for the current trial.

        Values greater than 1 correspond to faster movement progression.
        Values smaller than 1 correspond to slower movement progression.
        """

        return np.random.uniform(
            self.speed_range[0],
            self.speed_range[1],
        )

    def _sample_context_value(self):
        """
        Sample one trial-level context value.

        The sampled value remains constant throughout the complete trial.
        """

        return np.random.uniform(
            self.context_range[0],
            self.context_range[1],
        )

    def generate_latent(self, return_state=False):
        """
        Generate all latent trajectories.

        Parameters
        ----------
        return_state : bool
            If True, also return interpretable task variables.

        Returns
        -------
        Z : ndarray, shape (n_trials, L, k)
            Complete latent trajectories.

        cond : ndarray or None
            Condition assigned to each circular trial.
            It is None for the linear task.

        state : dict, optional
            Returned only when return_state=True.
        """
        L = self.trial_length
        k = self.latent_space_dimension

        Z = np.zeros((self.n_trials, L, k))
        phase = np.zeros((self.n_trials, L))
        position = np.zeros((self.n_trials, L, 2))
        direction_angle = np.full(self.n_trials, np.nan)
        direction_vector = np.full((self.n_trials, 2), np.nan)
        velocity = np.zeros((self.n_trials, L))
        context = np.full(self.n_trials, np.nan)
        speed_scale = np.ones(self.n_trials)

        if self.labels is None:
            cond = None
        else:
            cond = np.zeros(self.n_trials, dtype=int)

        for i in range(self.n_trials):

            cond_i = self._select_condition(i)

            # Velocity is included only when k exceeds the baseline dimension.
            if (
                self.condition_mode == "circular"
                and k > 3
            ) or (
                self.condition_mode == "linear"
                and k > 2
            ):
                speed_scale_i = self._sample_speed_scale()
            else:
                # Preserve the original movement profile in the baseline case.
                speed_scale_i = 1.0

            (
                t,
                s,
                p,
                circular_velocity,
                linear_velocity,
            ) = self._build_movement_profile(
                speed_scale=speed_scale_i
            )

            if self.condition_mode == "circular":
                velocity_i = circular_velocity

                # Context is the fifth circular coordinate, index 4.
                if k > 4:
                    context_value_i = self._sample_context_value()
                else:
                    context_value_i = None

            else:
                velocity_i = linear_velocity

                # Context is the fourth linear coordinate, index 3.
                if k > 3:
                    context_value_i = self._sample_context_value()
                else:
                    context_value_i = None

            m_i = self._build_deterministic(
                cond_i,
                s,
                p,
                velocity_i,
                context_value_i,
            )
            eta_i = self._build_stochastic()
            state_i = self._build_task_state(
                cond_i,
                s,
                p,
                velocity_i,
                context_value_i,
                speed_scale_i,
            )

            Z[i] = m_i + eta_i
            phase[i] = state_i["phase"]
            position[i] = state_i["position"]
            direction_angle[i] = state_i["direction_angle"]
            direction_vector[i] = state_i["direction_vector"]
            velocity[i] = state_i["velocity"]
            speed_scale[i] = state_i["speed_scale"]

            if state_i["context"] is not None:
                context[i] = state_i["context"]

            if cond is not None:
                cond[i] = cond_i

        if return_state:
            state = {
                "phase": phase,
                "position": position,
                "direction_angle": direction_angle,
                "direction_vector": direction_vector,
                "velocity": velocity,
                "context": context,
                "speed_scale": speed_scale,
            }
            return Z, cond, state

        return Z, cond


# arr=np.arange(2,100)
# choice(arr)
# from random import choice
