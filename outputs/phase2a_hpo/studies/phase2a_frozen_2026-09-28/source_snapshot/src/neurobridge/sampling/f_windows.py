#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal window extraction for trial-structured time series.

The central function, ``build_windows``, never allows a window to cross
a trial boundary. It supports fixed-length trials, variable-length trials,
explicit trial bounds, or one implicit trial spanning the complete input.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _validate_positive_int(name: str, value: int) -> None:
    """Validate a strictly positive integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _build_trial_bounds(
    total_time: int,
    *,
    trial_len: int | None,
    trial_lengths: Sequence[int] | None,
    trial_bounds: Sequence[tuple[int, int]] | None,
) -> list[tuple[int, int]]:
    """Resolve the supported trial specifications into explicit bounds."""
    provided = (
        trial_len is not None,
        trial_lengths is not None,
        trial_bounds is not None,
    )

    if sum(provided) > 1:
        raise ValueError(
            "Provide only one of trial_len, trial_lengths, or trial_bounds."
        )

    if not any(provided):
        return [(0, total_time)]

    if trial_len is not None:
        _validate_positive_int("trial_len", trial_len)

        if total_time % trial_len != 0:
            raise ValueError(
                "The total number of time points must be divisible by "
                "trial_len."
            )

        return [
            (start, start + trial_len)
            for start in range(0, total_time, trial_len)
        ]

    if trial_lengths is not None:
        lengths = list(trial_lengths)

        if len(lengths) == 0:
            raise ValueError("trial_lengths must not be empty.")

        for length in lengths:
            _validate_positive_int("Each trial length", length)

        if sum(lengths) != total_time:
            raise ValueError(
                "sum(trial_lengths) must equal the total number of time points."
            )

        bounds: list[tuple[int, int]] = []
        start = 0

        for length in lengths:
            end = start + length
            bounds.append((start, end))
            start = end

        return bounds

    bounds = list(trial_bounds)

    if len(bounds) == 0:
        raise ValueError("trial_bounds must not be empty.")

    previous_end = None

    for index, bound in enumerate(bounds):
        if not isinstance(bound, Sequence) or len(bound) != 2:
            raise ValueError(
                f"trial_bounds[{index}] must be a (start, end) pair."
            )

        start, end = bound

        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError("Trial-bound values must be integers.")

        if not 0 <= start < end <= total_time:
            raise ValueError(
                "Each trial bound must satisfy 0 <= start < end <= total_time."
            )

        # Overlapping trials would duplicate observations and make trial IDs
        # ambiguous. Gaps are allowed because explicit bounds may select only
        # specific segments of a longer recording.
        if previous_end is not None and start < previous_end:
            raise ValueError("trial_bounds must be sorted and non-overlapping.")

        previous_end = end

    return [(int(start), int(end)) for start, end in bounds]


def build_windows(
    X,
    window_size,
    stride,
    labels=None,
    trial_len=None,
    trial_lengths=None,
    trial_bounds=None,
    time_mode="relative",
    padding="valid",
    pad_value=0.0,
):
    """
    Extract temporal windows without crossing trial boundaries.

    Parameters
    ----------
    X:
        Data matrix with shape ``(total_time, n_features)``.

    window_size:
        Number of time bins in every output window.

    stride:
        Distance between consecutive window centers.

    labels:
        Optional trial-level labels. The first dimension must equal the
        number of trials. Scalar or vector-valued trial labels are supported.

    trial_len:
        Fixed trial length when all trials have equal size.

    trial_lengths:
        Sequence of contiguous variable trial lengths.

    trial_bounds:
        Explicit ``(start, end)`` pairs using Python's end-exclusive
        convention.

    time_mode:
        ``"relative"`` returns a coordinate in ``[0, 1]`` inside each trial.
        ``"absolute"`` returns the integer center index inside each trial.

    padding:
        ``"valid"`` returns only complete windows.
        ``"center"`` returns a centered window at each selected trial time
        and pads values that fall outside the trial.

    pad_value:
        Constant used by ``padding="center"``.

    Returns
    -------
    X_windows:
        Array with shape ``(n_windows, window_size, n_features)``.

    time_id:
        Relative or absolute center coordinate inside each trial.

    global_time_id:
        Center coordinate in the original concatenated input.

    trial_id:
        Trial index associated with each window.

    labels_windows:
        Trial label repeated for every window, or ``None``.
    """
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(
            "X must have shape (total_time, n_features)."
        )

    _validate_positive_int("window_size", window_size)
    _validate_positive_int("stride", stride)

    if time_mode not in {"relative", "absolute"}:
        raise ValueError(
            "time_mode must be either 'relative' or 'absolute'."
        )

    if padding not in {"valid", "center"}:
        raise ValueError(
            "padding must be either 'valid' or 'center'."
        )

    total_time, n_features = X.shape

    bounds = _build_trial_bounds(
        total_time,
        trial_len=trial_len,
        trial_lengths=trial_lengths,
        trial_bounds=trial_bounds,
    )
    n_trials = len(bounds)

    labels_array = None

    if labels is not None:
        if np.isscalar(labels):
            raise ValueError(
                "labels must contain one entry per trial. "
                "Use np.repeat(label, n_trials) when all trials share a label."
            )

        labels_array = np.asarray(labels)

        if labels_array.ndim == 0:
            raise ValueError("labels must have a trial dimension.")

        if labels_array.shape[0] != n_trials:
            raise ValueError(
                "labels.shape[0] must equal the number of trials."
            )

    windows: list[np.ndarray] = []
    time_ids: list[float | int] = []
    global_time_ids: list[int] = []
    trial_ids: list[int] = []
    window_labels: list[np.ndarray | np.generic] = []

    left_width = window_size // 2
    right_width = window_size - left_width

    for trial_index, (trial_start, trial_end) in enumerate(bounds):
        trial_length = trial_end - trial_start

        if padding == "valid":
            # Valid center c satisfies:
            # c - left_width >= 0
            # c + right_width <= trial_length
            first_center = left_width
            last_center_exclusive = trial_length - right_width + 1

            if first_center >= last_center_exclusive:
                continue

            local_centers = range(
                first_center,
                last_center_exclusive,
                stride,
            )
        else:
            local_centers = range(0, trial_length, stride)

        for local_center in local_centers:
            local_start = local_center - left_width
            local_end = local_center + right_width
            global_center = trial_start + local_center

            if padding == "valid":
                global_start = trial_start + local_start
                global_end = trial_start + local_end
                window = X[global_start:global_end]
            else:
                valid_local_start = max(local_start, 0)
                valid_local_end = min(local_end, trial_length)

                valid_global_start = trial_start + valid_local_start
                valid_global_end = trial_start + valid_local_end

                window = np.full(
                    (window_size, n_features),
                    pad_value,
                    dtype=X.dtype,
                )

                insert_start = valid_local_start - local_start
                insert_end = insert_start + (
                    valid_local_end - valid_local_start
                )

                window[insert_start:insert_end] = X[
                    valid_global_start:valid_global_end
                ]

            if window.shape != (window_size, n_features):
                raise RuntimeError(
                    "Internal windowing error: generated window has shape "
                    f"{window.shape}, expected "
                    f"{(window_size, n_features)}."
                )

            windows.append(window)
            global_time_ids.append(global_center)
            trial_ids.append(trial_index)

            if time_mode == "relative":
                denominator = max(trial_length - 1, 1)
                time_ids.append(local_center / denominator)
            else:
                time_ids.append(local_center)

            if labels_array is not None:
                window_labels.append(labels_array[trial_index])

    if windows:
        X_windows = np.stack(windows, axis=0).astype(
            X.dtype,
            copy=False,
        )
    else:
        X_windows = np.empty(
            (0, window_size, n_features),
            dtype=X.dtype,
        )

    time_dtype = np.float64 if time_mode == "relative" else np.int64
    time_id = np.asarray(time_ids, dtype=time_dtype)
    global_time_id = np.asarray(global_time_ids, dtype=np.int64)
    trial_id = np.asarray(trial_ids, dtype=np.int64)

    if labels_array is None:
        labels_windows = None
    elif window_labels:
        labels_windows = np.asarray(window_labels)
    else:
        labels_windows = np.empty(
            (0, *labels_array.shape[1:]),
            dtype=labels_array.dtype,
        )

    return (
        X_windows,
        time_id,
        global_time_id,
        trial_id,
        labels_windows,
    )
