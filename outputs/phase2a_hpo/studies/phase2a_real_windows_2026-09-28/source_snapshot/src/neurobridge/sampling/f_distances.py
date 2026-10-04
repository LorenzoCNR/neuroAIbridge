#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legacy NumPy pairwise-distance utilities.

Training code should prefer ``batch_similarity.py`` because it operates
directly on PyTorch batches and can run on GPU.

This module remains for:
- exploratory NumPy workflows;
- compatibility with older experiment scripts;
- construction of full diagnostic distance matrices.

All implementations are vectorized. Materializing a full matrix still
requires O(n^2) time and memory.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _as_vector(values, name: str) -> np.ndarray:
    """Convert scalar metadata to a one-dimensional NumPy array."""
    values = np.asarray(values)

    if values.ndim == 0:
        raise ValueError(f"{name} must contain at least one value.")

    return values.reshape(-1)


def _validate_square_nonnegative(
    matrix,
    *,
    name: str,
) -> np.ndarray:
    """Validate a square non-negative matrix."""
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    if np.any(matrix < 0):
        raise ValueError(f"{name} must contain non-negative values.")

    return matrix


def pairwise_temporal_distance(temporal_id):
    """
    Compute absolute pairwise temporal distance.

    Complexity
    ----------
    Time:  O(n^2)
    Space: O(n^2)
    """
    temporal_id = _as_vector(temporal_id, "temporal_id").astype(float)
    return np.abs(
        temporal_id[:, None] - temporal_id[None, :]
    )


def pairwise_label_distance(labels_id):
    """
    Compute categorical pairwise distance.

    Equal labels receive distance 0; different labels receive distance 1.
    """
    labels_id = _as_vector(labels_id, "labels_id")
    return (
        labels_id[:, None] != labels_id[None, :]
    ).astype(float)


def circular_distance(labels_id, num_labels=8):
    """
    Compute geodesic pairwise distance on a discrete circular label space.

    Labels must lie in ``[0, num_labels - 1]``.
    """
    if not isinstance(num_labels, int) or isinstance(num_labels, bool):
        raise TypeError("num_labels must be an integer.")
    if num_labels < 2:
        raise ValueError("num_labels must be at least 2.")

    labels = _as_vector(labels_id, "labels_id")

    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.equal(labels, np.round(labels))):
            raise ValueError("labels_id must contain integer values.")
        labels = np.round(labels)

    labels = labels.astype(np.int64)

    # Support both conventions used by existing NeuroBridge experiments:
    # zero-based labels (0..K-1) and one-based labels (1..K).
    if np.all((labels >= 1) & (labels <= num_labels)):
        labels = labels - 1
    elif not np.all((labels >= 0) & (labels < num_labels)):
        raise ValueError(
            "labels_id must use either zero-based values "
            f"[0, {num_labels - 1}] or one-based values "
            f"[1, {num_labels}]."
        )

    raw = np.abs(labels[:, None] - labels[None, :])
    return np.minimum(raw, num_labels - raw).astype(float)


def normalize_distance(D, eps=1e-12):
    """
    Normalize a non-negative square distance matrix to ``[0, 1]``.

    A zero matrix is returned unchanged.
    """
    if eps <= 0:
        raise ValueError("eps must be strictly positive.")

    D = _validate_square_nonnegative(D, name="D")
    max_distance = D.max(initial=0.0)

    if max_distance <= eps:
        return D.copy()

    return D / max_distance


def combine_distances(
    distance_dict: Mapping[str, np.ndarray],
    weight_dict: Mapping[str, float],
    verbose: bool = False,
):
    """
    Combine distance matrices using normalized non-negative weights.

    The original API required weights to sum exactly to one. This version
    accepts any positive total and normalizes the weights internally.

    Parameters
    ----------
    distance_dict:
        Mapping from component name to a square distance matrix.

    weight_dict:
        Mapping with the same keys and non-negative scalar weights.

    verbose:
        Print the normalized component weights.

    Returns
    -------
    np.ndarray
        Weighted pairwise distance matrix.
    """
    if not isinstance(distance_dict, Mapping):
        raise TypeError("distance_dict must be a mapping.")
    if not isinstance(weight_dict, Mapping):
        raise TypeError("weight_dict must be a mapping.")
    if len(distance_dict) == 0:
        raise ValueError("distance_dict must not be empty.")
    if set(distance_dict) != set(weight_dict):
        raise ValueError(
            "distance_dict and weight_dict must have identical keys."
        )

    matrices = {}
    reference_shape = None

    for name, matrix in distance_dict.items():
        validated = _validate_square_nonnegative(
            matrix,
            name=f"distance_dict[{name!r}]",
        )

        if reference_shape is None:
            reference_shape = validated.shape
        elif validated.shape != reference_shape:
            raise ValueError(
                "All distance matrices must have the same shape."
            )

        matrices[name] = validated

    weights = {}

    for name, value in weight_dict.items():
        weight = float(value)

        if not np.isfinite(weight):
            raise ValueError(f"Weight {name!r} must be finite.")
        if weight < 0:
            raise ValueError(f"Weight {name!r} must be non-negative.")

        weights[name] = weight

    total_weight = sum(weights.values())

    if total_weight <= 0:
        raise ValueError("At least one weight must be positive.")

    normalized_weights = {
        name: weight / total_weight
        for name, weight in weights.items()
    }

    result = np.zeros(reference_shape, dtype=float)

    for name, matrix in matrices.items():
        weight = normalized_weights[name]
        result += weight * matrix

        if verbose:
            print(f"{name}: normalized weight={weight:.6f}")

    return result
