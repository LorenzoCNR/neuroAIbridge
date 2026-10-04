# -*- coding: utf-8 -*-
"""
Positive-pair masks derived from labels and temporal metadata.

These masks are intended for contrastive objectives such as
``masked_infonce_loss``. The diagonal is excluded by default so a sample
is never treated as its own positive.
"""

from __future__ import annotations

import torch


def _as_batch_vector(values: torch.Tensor, name: str) -> torch.Tensor:
    """Validate metadata with one scalar value per batch element."""
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")

    if values.ndim not in (1, 2):
        raise ValueError(
            f"{name} must have shape (batch,) or (batch, 1)."
        )

    if values.ndim == 2 and values.shape[1] != 1:
        raise ValueError(
            f"{name} must contain one scalar per sample."
        )

    return values.reshape(-1)


def _remove_diagonal(mask: torch.Tensor) -> torch.Tensor:
    """Set self-pairs to False."""
    eye = torch.eye(
        mask.shape[0],
        dtype=torch.bool,
        device=mask.device,
    )
    return mask & ~eye


def categorical_positive_mask(
    labels: torch.Tensor,
    *,
    exclude_self: bool = True,
) -> torch.Tensor:
    """
    Mark samples with the same label as positive pairs.

    Parameters
    ----------
    labels:
        One scalar categorical label per batch element.

    exclude_self:
        Remove diagonal self-pairs.

    Returns
    -------
    torch.Tensor
        Boolean matrix with shape ``(B, B)``.
    """
    labels = _as_batch_vector(labels, "labels")
    mask = labels[:, None] == labels[None, :]

    if exclude_self:
        mask = _remove_diagonal(mask)

    return mask


def time_offset_positive_mask(
    trial_id: torch.Tensor,
    time_id: torch.Tensor,
    offset: float,
    *,
    atol: float = 1e-6,
    exclude_self: bool = True,
) -> torch.Tensor:
    """
    Mark pairs from the same trial separated by a target time offset.

    For integer time IDs, ``atol`` has no practical effect. For floating
    time coordinates it provides numerical tolerance.
    """
    if offset <= 0:
        raise ValueError("offset must be strictly positive.")
    if atol < 0:
        raise ValueError("atol must be non-negative.")

    trial_id = _as_batch_vector(trial_id, "trial_id")
    time_id = _as_batch_vector(time_id, "time_id").float()

    if trial_id.shape[0] != time_id.shape[0]:
        raise ValueError(
            "trial_id and time_id must contain the same number of samples."
        )
    if trial_id.device != time_id.device:
        raise ValueError(
            "trial_id and time_id must be on the same device."
        )

    same_trial = trial_id[:, None] == trial_id[None, :]
    time_difference = torch.abs(
        time_id[:, None] - time_id[None, :]
    )
    target_offset = torch.as_tensor(
        offset,
        device=time_id.device,
        dtype=time_id.dtype,
    )

    correct_offset = torch.isclose(
        time_difference,
        target_offset,
        atol=atol,
        rtol=0.0,
    )

    mask = same_trial & correct_offset

    if exclude_self:
        mask = _remove_diagonal(mask)

    return mask
