# -*- coding: utf-8 -*-
"""
Batch-wise metadata distances and structured similarities.

These functions operate on PyTorch tensors and are intended for training.
All pairwise matrices have shape ``(batch_size, batch_size)``.

The full pairwise computation is quadratic in batch size. This is
intentional in the current implementation because the contrastive
objectives consume dense pairwise relations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _require_tensor(name: str, value: Any) -> torch.Tensor:
    """Validate and return a tensor."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    return value


def _as_2d_float(values: torch.Tensor) -> torch.Tensor:
    """
    Convert scalar or vector metadata into a 2D floating-point matrix.

    Shape conversions:
        (B,)    -> (B, 1)
        (B, D)  -> unchanged
    """
    values = _require_tensor("values", values)

    if values.ndim == 1:
        values = values[:, None]
    elif values.ndim != 2:
        raise ValueError(
            "values must have shape (batch,) or (batch, features), "
            f"got {tuple(values.shape)}."
        )

    return values.float()


def batch_temporal_distance(time_id: torch.Tensor) -> torch.Tensor:
    """
    Compute absolute pairwise temporal distance.

    Parameters
    ----------
    time_id:
        Tensor with one scalar time value per batch element.

    Returns
    -------
    torch.Tensor
        Matrix with shape ``(B, B)``.
    """
    time_id = _require_tensor("time_id", time_id)

    if time_id.ndim not in (1, 2):
        raise ValueError(
            "time_id must have shape (batch,) or (batch, 1)."
        )

    time_id = time_id.float().reshape(-1)
    return torch.abs(time_id[:, None] - time_id[None, :])


def batch_circular_label_distance(
    labels: torch.Tensor,
    num_labels: int = 8,
) -> torch.Tensor:
    """
    Compute geodesic distance between labels on a discrete circle.

    Labels must lie in ``[0, num_labels - 1]``.

    Example for eight labels:
        distance(0, 7) = 1
        distance(0, 4) = 4
    """
    labels = _require_tensor("labels", labels)

    if not isinstance(num_labels, int) or isinstance(num_labels, bool):
        raise TypeError("num_labels must be an integer.")
    if num_labels < 2:
        raise ValueError("num_labels must be at least 2.")

    labels = labels.reshape(-1)

    if labels.is_floating_point():
        rounded = labels.round()
        if not torch.allclose(labels, rounded):
            raise ValueError("Circular labels must contain integer values.")
        labels = rounded

    labels = labels.long()

    if labels.numel() > 0:
        # NeuroBridge historically used both zero-based labels (0..K-1)
        # and one-based labels (1..K). Circular distances are invariant to
        # a global shift, so both conventions can be supported safely.
        if torch.all((labels >= 1) & (labels <= num_labels)):
            labels = labels - 1
        elif not torch.all((labels >= 0) & (labels < num_labels)):
            raise ValueError(
                "Circular labels must use either zero-based values "
                f"[0, {num_labels - 1}] or one-based values "
                f"[1, {num_labels}]."
            )

    raw = torch.abs(labels[:, None] - labels[None, :])
    wrapped = num_labels - raw
    return torch.minimum(raw, wrapped).float()


def batch_continuous_distance(
    values: torch.Tensor,
    p: float = 2.0,
) -> torch.Tensor:
    """
    Compute a pairwise Minkowski distance for continuous metadata.

    ``values`` may contain one scalar or multiple continuous features
    per sample.
    """
    if not isinstance(p, (int, float)) or isinstance(p, bool):
        raise TypeError("p must be a positive real number.")
    if p <= 0:
        raise ValueError("p must be strictly positive.")

    values = _as_2d_float(values)
    return torch.cdist(values, values, p=float(p))


def normalize_batch_distance(
    distance: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Scale a non-negative distance matrix to the interval ``[0, 1]``.

    A zero matrix is returned unchanged.
    """
    distance = _require_tensor("distance", distance)

    if distance.ndim != 2:
        raise ValueError("distance must be a 2D matrix.")
    if distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be square.")
    if torch.any(distance < 0):
        raise ValueError("distance must contain non-negative values.")
    if eps <= 0:
        raise ValueError("eps must be strictly positive.")

    max_value = distance.max()

    if bool(max_value <= eps):
        return distance.clone()

    return distance / max_value


def batch_distance_from_spec(
    batch: Mapping[str, torch.Tensor],
    spec: Mapping[str, Any],
) -> torch.Tensor:
    """
    Build one pairwise distance matrix from a metadata specification.

    Supported geometries
    --------------------
    temporal:
        Absolute scalar time difference.

    circular:
        Geodesic distance on a periodic discrete label space.

    continuous / euclidean:
        Minkowski distance between continuous metadata vectors.

    categorical:
        Zero for equal rows and one for different rows.

    Required specification field
    ----------------------------
    key:
        Name of the metadata tensor in ``batch``.
    """
    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping.")
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a mapping.")
    if "key" not in spec:
        raise ValueError("Each specification must contain a 'key' field.")

    key = spec["key"]

    if key not in batch:
        raise ValueError(f"batch must contain {key!r}.")

    geometry = str(spec.get("geometry", "continuous")).lower()
    values = _require_tensor(f"batch[{key!r}]", batch[key])

    if geometry == "temporal":
        return batch_temporal_distance(values)

    if geometry == "circular":
        return batch_circular_label_distance(
            values,
            num_labels=int(spec.get("num_labels", 8)),
        )

    if geometry in {"continuous", "euclidean"}:
        p = 2.0 if geometry == "euclidean" else float(spec.get("p", 2.0))
        return batch_continuous_distance(values, p=p)

    if geometry == "categorical":
        if values.ndim == 1:
            values = values[:, None]
        elif values.ndim != 2:
            raise ValueError(
                "Categorical metadata must have shape (batch,) "
                "or (batch, features)."
            )

        equal_rows = torch.all(
            values[:, None, :] == values[None, :, :],
            dim=-1,
        )
        return (~equal_rows).float()

    raise ValueError(
        f"Unsupported metadata geometry {geometry!r}. "
        "Expected temporal, circular, continuous, euclidean, "
        "or categorical."
    )


def batch_structured_similarity_from_specs(
    batch: Mapping[str, torch.Tensor],
    specs: Sequence[Mapping[str, Any]],
    tau: float = 0.5,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Build a soft pairwise similarity from multiple metadata geometries.

    The weighted distance is

        D_total = sum_k normalized_weight_k * D_k

    and the final similarity is

        S = exp(-D_total / tau)

    Each specification may contain:

    - ``key``: metadata key in the batch;
    - ``geometry``: distance geometry;
    - ``weight``: non-negative component weight;
    - geometry-specific parameters such as ``num_labels`` or ``p``.

    Weights are normalized internally, so they do not need to sum to one.
    """
    if tau <= 0:
        raise ValueError("tau must be strictly positive.")
    if not isinstance(specs, Sequence) or len(specs) == 0:
        raise ValueError("specs must contain at least one specification.")

    weights = []

    for index, spec in enumerate(specs):
        if not isinstance(spec, Mapping):
            raise TypeError(f"specs[{index}] must be a mapping.")

        weight = float(spec.get("weight", 1.0))

        if weight < 0:
            raise ValueError("Metadata weights must be non-negative.")

        weights.append(weight)

    total_weight = sum(weights)

    if total_weight <= 0:
        raise ValueError("At least one metadata weight must be positive.")

    combined_distance = None

    for spec, weight in zip(specs, weights):
        if weight == 0:
            continue

        component = batch_distance_from_spec(batch, spec)

        if normalize:
            component = normalize_batch_distance(component)

        weighted_component = (weight / total_weight) * component

        if combined_distance is None:
            combined_distance = weighted_component
        else:
            if component.shape != combined_distance.shape:
                raise ValueError(
                    "All metadata distance matrices must have the same shape."
                )
            combined_distance = combined_distance + weighted_component

    if combined_distance is None:
        raise RuntimeError("No positive-weight distance component was built.")

    return torch.exp(-combined_distance / float(tau))


def batch_structured_similarity(
    batch: Mapping[str, torch.Tensor],
    time_weight: float = 0.5,
    label_weight: float = 0.5,
    tau: float = 0.5,
    num_labels: int = 8,
) -> torch.Tensor:
    """
    Convenience wrapper for temporal plus circular-label similarity.

    This preserves the original NeuroBridge API while delegating the
    computation to ``batch_structured_similarity_from_specs``.
    """
    if time_weight < 0 or label_weight < 0:
        raise ValueError("Weights must be non-negative.")
    if time_weight + label_weight <= 0:
        raise ValueError("At least one weight must be positive.")

    required_keys = ("time_id", "label")
    missing = [key for key in required_keys if key not in batch]

    if missing:
        raise ValueError(
            f"batch is missing required keys: {', '.join(missing)}."
        )

    specs = [
        {
            "key": "time_id",
            "geometry": "temporal",
            "weight": time_weight,
        },
        {
            "key": "label",
            "geometry": "circular",
            "num_labels": num_labels,
            "weight": label_weight,
        },
    ]

    return batch_structured_similarity_from_specs(
        batch,
        specs,
        tau=tau,
        normalize=True,
    )
