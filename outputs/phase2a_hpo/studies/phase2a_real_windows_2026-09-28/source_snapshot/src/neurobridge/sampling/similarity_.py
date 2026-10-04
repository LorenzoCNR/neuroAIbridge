# -*- coding: utf-8 -*-
"""
Conversion from pairwise distances to soft similarities.

The filename is kept for compatibility with existing experiment imports.
New code may use the clearer function name ``dist_to_similarity``.
"""

from __future__ import annotations

import numpy as np


def dist_to_similarity(D_total, tau):
    """
    Convert a non-negative square distance matrix into an RBF similarity.

    Formula
    -------
    W = exp(-D_total / tau)

    Parameters
    ----------
    D_total:
        Square pairwise distance matrix with shape ``(B, B)``.

    tau:
        Strictly positive temperature / scale parameter.

    Returns
    -------
    np.ndarray
        Similarity matrix with values in ``(0, 1]``.

    Complexity
    ----------
    Time:  O(B^2)
    Space: O(B^2)
    """
    D_total = np.asarray(D_total, dtype=float)

    if D_total.ndim != 2:
        raise ValueError("D_total must be a 2D matrix.")
    if D_total.shape[0] != D_total.shape[1]:
        raise ValueError("D_total must be square, with shape (B, B).")
    if not np.all(np.isfinite(D_total)):
        raise ValueError("D_total must contain finite values.")
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if np.any(D_total < 0):
        raise ValueError(
            "D_total must contain non-negative distances."
        )

    return np.exp(-D_total / float(tau))


def dist_to_simi(D_total, tau):
    """
    Backward-compatible alias for ``dist_to_similarity``.

    Existing scripts can keep importing ``dist_to_simi``.
    """
    return dist_to_similarity(D_total, tau)
