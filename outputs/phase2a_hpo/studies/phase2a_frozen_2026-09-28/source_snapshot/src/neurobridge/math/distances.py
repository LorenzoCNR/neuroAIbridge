# -*- coding: utf-8 -*-
"""Pairwise distance and similarity utilities.

This module contains reusable PyTorch primitives for building structured
contrastive objectives. Every public function accepts tensors on CPU or GPU and
returns tensors on the same device.

Shape convention
----------------
Feature inputs:
    x1: [N, D]
    x2: [M, D]

Label or time inputs:
    v1: [N]
    v2: [M]

Pairwise outputs always have shape [N, M].
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional
import warnings

import torch


__all__ = [
    "adaptive_gaussian_similarity",
    "categorical_distance",
    "categorical_similarity",
    "circular_distance",
    "combine_distances",
    "direction_distance",
    "minkowski_distance",
    "normalize_distance",
    "temporal_distance",
]


def _validate_feature_pair(x1: torch.Tensor, x2: torch.Tensor) -> None:
    """Validate two feature matrices used in pairwise computations."""
    if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
        raise TypeError("x1 and x2 must be torch.Tensor objects.")
    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError(
            f"x1 and x2 must be 2D tensors; got {x1.shape} and {x2.shape}."
        )
    if x1.shape[1] != x2.shape[1]:
        raise ValueError(
            "x1 and x2 must have the same feature dimension; "
            f"got {x1.shape[1]} and {x2.shape[1]}."
        )
    if x1.device != x2.device:
        raise ValueError("x1 and x2 must be on the same device.")
    if not x1.is_floating_point() or not x2.is_floating_point():
        raise TypeError("x1 and x2 must use floating-point dtypes.")


def _as_vector(x: torch.Tensor, name: str) -> torch.Tensor:
    """Return a tensor as a one-dimensional vector."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if x.ndim == 0:
        raise ValueError(f"{name} must contain at least one element.")
    return x.reshape(-1)


def _safe_std(
    x: torch.Tensor,
    dim: int = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute a finite population standard deviation with a numerical floor."""
    if eps <= 0:
        raise ValueError("eps must be greater than zero.")
    std = torch.std(x, dim=dim, unbiased=False, keepdim=True)
    return std.clamp_min(eps)


def _shared_standardization(
    x1: torch.Tensor,
    x2: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardize two matrices using statistics from their pooled rows."""
    pooled = torch.cat((x1, x2), dim=0)
    mean = pooled.mean(dim=0, keepdim=True)
    std = _safe_std(pooled, dim=0, eps=eps)
    return (x1 - mean) / std, (x2 - mean) / std


def minkowski_distance(
    x1: torch.Tensor,
    x2: torch.Tensor,
    p: float = 2.0,
    *,
    normalize: bool = False,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute pairwise Minkowski distances.

    Normalization modes
    -------------------
    normalize=False:
        Use inputs as provided.

    normalize=True and mean/std are omitted:
        Standardize both inputs using shared pooled statistics. This preserves a
        common coordinate system and avoids separately warping x1 and x2.

    normalize=True and mean/std are provided:
        Use externally fitted statistics, for example statistics computed on a
        training split. Both mean and std must be provided together.
    """
    _validate_feature_pair(x1, x2)

    if p <= 0:
        raise ValueError("p must be greater than zero.")
    if eps <= 0:
        raise ValueError("eps must be greater than zero.")
    if (mean is None) != (std is None):
        raise ValueError("mean and std must either both be provided or both omitted.")

    if normalize:
        if mean is None:
            x1, x2 = _shared_standardization(x1, x2, eps=eps)
        else:
            mean = torch.as_tensor(mean, device=x1.device, dtype=x1.dtype)
            std = torch.as_tensor(std, device=x1.device, dtype=x1.dtype)
            if mean.numel() != x1.shape[1] or std.numel() != x1.shape[1]:
                raise ValueError(
                    "mean and std must contain one value per feature; "
                    f"expected {x1.shape[1]}."
                )
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1).clamp_min(eps)
            x1 = (x1 - mean) / std
            x2 = (x2 - mean) / std

    return torch.cdist(x1, x2, p=float(p))


@torch.no_grad()
def adaptive_gaussian_similarity(
    x1: torch.Tensor,
    x2: torch.Tensor,
    p: float = 2.0,
    *,
    bandwidth: Optional[float | torch.Tensor] = None,
    normalize: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute an RBF-style pairwise similarity matrix.

    The similarity is ``exp(-distance**2 / (2 * bandwidth**2))``.

    When ``bandwidth`` is omitted, it is estimated symmetrically from the pooled
    rows of x1 and x2 as the mean per-feature population standard deviation.
    """
    _validate_feature_pair(x1, x2)

    dists = minkowski_distance(
        x1,
        x2,
        p=p,
        normalize=normalize,
        eps=eps,
    )

    if bandwidth is None:
        pooled = torch.cat((x1, x2), dim=0)
        bandwidth_tensor = _safe_std(pooled, dim=0, eps=eps).mean()
    else:
        bandwidth_tensor = torch.as_tensor(
            bandwidth,
            device=x1.device,
            dtype=x1.dtype,
        )
        if bandwidth_tensor.numel() != 1:
            raise ValueError("bandwidth must be a scalar.")
        bandwidth_tensor = bandwidth_tensor.reshape(()).clamp_min(eps)

    return torch.exp(-(dists.square()) / (2.0 * bandwidth_tensor.square()))


@torch.no_grad()
def temporal_distance(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Compute absolute pairwise temporal distances."""
    t1 = _as_vector(t1, "t1")
    t2 = _as_vector(t2, "t2")
    if t1.device != t2.device:
        raise ValueError("t1 and t2 must be on the same device.")
    return torch.abs(t1[:, None] - t2[None, :])


@torch.no_grad()
def categorical_similarity(l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
    """Return 1 for equal labels and 0 otherwise."""
    l1 = _as_vector(l1, "l1")
    l2 = _as_vector(l2, "l2")
    if l1.device != l2.device:
        raise ValueError("l1 and l2 must be on the same device.")
    return (l1[:, None] == l2[None, :]).to(dtype=torch.float32)


@torch.no_grad()
def categorical_distance(l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
    """Return 0 for equal labels and 1 otherwise."""
    return 1.0 - categorical_similarity(l1, l2)


@torch.no_grad()
def direction_distance(l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
    """Deprecated compatibility alias for :func:`categorical_similarity`.

    The historical function name was misleading because it returned a
    similarity rather than a distance.
    """
    warnings.warn(
        "direction_distance is deprecated; use categorical_similarity instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return categorical_similarity(l1, l2)


@torch.no_grad()
def circular_distance(
    l1: torch.Tensor,
    l2: torch.Tensor,
    num_directions: int = 8,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    """Compute pairwise geodesic distance on a discrete circular label space.

    Labels must belong to ``{0, ..., num_directions - 1}``.
    With ``normalize=True``, distances are divided by the largest possible
    circular distance, ``floor(num_directions / 2)``.
    """
    if num_directions < 2:
        raise ValueError("num_directions must be at least 2.")

    l1 = _as_vector(l1, "l1").to(dtype=torch.long)
    l2 = _as_vector(l2, "l2").to(dtype=torch.long)
    if l1.device != l2.device:
        raise ValueError("l1 and l2 must be on the same device.")

    for labels, name in ((l1, "l1"), (l2, "l2")):
        if labels.numel() and (
            torch.any(labels < 0) or torch.any(labels >= num_directions)
        ):
            raise ValueError(
                f"{name} contains labels outside [0, {num_directions - 1}]."
            )

    raw = torch.abs(l1[:, None] - l2[None, :])
    distances = torch.minimum(raw, num_directions - raw).to(torch.float32)

    if normalize:
        distances = distances / float(num_directions // 2)
    return distances


def normalize_distance(
    distance: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scale a non-negative distance matrix to the interval [0, 1]."""
    if not isinstance(distance, torch.Tensor):
        raise TypeError("distance must be a torch.Tensor.")
    if distance.ndim != 2:
        raise ValueError("distance must be a 2D matrix.")
    if torch.any(distance < 0):
        raise ValueError("distance must contain only non-negative values.")
    if eps <= 0:
        raise ValueError("eps must be greater than zero.")

    max_distance = distance.max()
    if bool(max_distance <= eps):
        return distance.clone()
    return distance / max_distance


def combine_distances(
    distances: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    *,
    normalize_inputs: bool = False,
    atol: float = 1e-6,
) -> torch.Tensor:
    """Combine distance matrices using a convex weighted sum.

    Parameters
    ----------
    distances:
        Mapping from component name to a pairwise distance matrix.
    weights:
        Non-negative weights with the same keys as ``distances``. Weights must
        sum to one within ``atol``.
    normalize_inputs:
        Normalize every component to [0, 1] before combining it.
    """
    if not distances:
        raise ValueError("distances must not be empty.")
    if set(distances) != set(weights):
        raise ValueError("distances and weights must have identical keys.")
    if atol <= 0:
        raise ValueError("atol must be greater than zero.")

    first_name = next(iter(distances))
    reference = distances[first_name]
    if not isinstance(reference, torch.Tensor) or reference.ndim != 2:
        raise ValueError(f"distances['{first_name}'] must be a 2D tensor.")

    total_weight = 0.0
    prepared: dict[str, torch.Tensor] = {}

    for name, matrix in distances.items():
        if not isinstance(matrix, torch.Tensor):
            raise TypeError(f"distances['{name}'] must be a torch.Tensor.")
        if matrix.ndim != 2:
            raise ValueError(f"distances['{name}'] must be a 2D matrix.")
        if matrix.shape != reference.shape:
            raise ValueError("all distance matrices must have the same shape.")
        if matrix.device != reference.device:
            raise ValueError("all distance matrices must be on the same device.")
        if torch.any(matrix < 0):
            raise ValueError(f"distances['{name}'] contains negative values.")

        weight = float(weights[name])
        if weight < 0:
            raise ValueError(f"weights['{name}'] must be non-negative.")
        total_weight += weight
        prepared[name] = normalize_distance(matrix) if normalize_inputs else matrix

    if abs(total_weight - 1.0) > atol:
        raise ValueError(f"weights must sum to 1.0; got {total_weight:.8f}.")

    result_dtype = reference.dtype if reference.is_floating_point() else torch.float32
    result = torch.zeros(
        reference.shape,
        device=reference.device,
        dtype=result_dtype,
    )

    for name, matrix in prepared.items():
        result = result + float(weights[name]) * matrix.to(dtype=result_dtype)

    return result
