# -*- coding: utf-8 -*-
"""
Utilities for soft positive-pair weights.

A dense similarity matrix can be transformed into row-wise target
probabilities for soft contrastive learning. These helpers centralize
that normalization and its edge-case handling.
"""

from __future__ import annotations

import torch


def _validate_weight_matrix(weights: torch.Tensor) -> torch.Tensor:
    """Validate a finite non-negative square tensor."""
    if not isinstance(weights, torch.Tensor):
        raise TypeError("weights must be a torch.Tensor.")
    if weights.ndim != 2:
        raise ValueError("weights must be a 2D matrix.")
    if weights.shape[0] != weights.shape[1]:
        raise ValueError("weights must be square.")
    if not torch.isfinite(weights).all():
        raise ValueError("weights must contain finite values.")
    if torch.any(weights < 0):
        raise ValueError("weights must be non-negative.")

    return weights


def normalize_positive_weights(
    weights: torch.Tensor,
    *,
    exclude_self: bool = True,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize each row into a probability distribution.

    Rows with no positive mass remain zero and are marked as invalid.

    Returns
    -------
    normalized:
        Row-normalized weight matrix.

    valid_rows:
        Boolean vector identifying rows with positive total mass.
    """
    if eps <= 0:
        raise ValueError("eps must be strictly positive.")

    weights = _validate_weight_matrix(weights).clone()

    if exclude_self:
        weights.fill_diagonal_(0.0)

    row_sum = weights.sum(dim=1, keepdim=True)
    valid_rows = row_sum.squeeze(1) > eps

    normalized = torch.zeros_like(weights)

    if valid_rows.any():
        normalized[valid_rows] = (
            weights[valid_rows] / row_sum[valid_rows]
        )

    return normalized, valid_rows


def distance_to_positive_weights(
    distance: torch.Tensor,
    tau: float,
    *,
    exclude_self: bool = True,
    normalize_rows: bool = False,
    eps: float = 1e-12,
):
    """
    Convert a distance matrix into soft positive weights.

    Formula
    -------
    weights = exp(-distance / tau)

    With ``normalize_rows=True``, the function returns
    ``(weights, valid_rows)``. Otherwise it returns the raw weight matrix.
    """
    if tau <= 0:
        raise ValueError("tau must be strictly positive.")

    distance = _validate_weight_matrix(distance)

    weights = torch.exp(-distance / float(tau))

    if exclude_self:
        weights.fill_diagonal_(0.0)

    if normalize_rows:
        return normalize_positive_weights(
            weights,
            exclude_self=False,
            eps=eps,
        )

    return weights
