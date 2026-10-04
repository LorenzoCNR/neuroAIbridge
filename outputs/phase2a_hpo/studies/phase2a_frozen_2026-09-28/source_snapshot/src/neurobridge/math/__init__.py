# -*- coding: utf-8 -*-
"""
Distance and similarity utilities for contrastive learning.

Includes:
1) adaptive_gaussian_similarity: RBF-style similarity using empirical variance
2) minkowski_distance: generic Lp distance (optionally normalized)
3) direction_distance: categorical "same/different" indicator
4) circular_distance: distance on a periodic label space (e.g., 8 directions)

Design notes
------------
- Return shapes are always [N, M] given x1=[N,D], x2=[M,D] or l1=[N], l2=[M].
- No in-place ops, pure functions: safe for autograd when used on tensors
  that require grad (only the embeddings, not labels, typically).
- Keep CPU/GPU neutrality: results live on the same device as the inputs.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Optional


def _safe_std(x: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    """Compute unbiased std and clamp to avoid division-by-zero."""
    std = torch.std(x, dim=dim, unbiased=True, keepdim=True)
    return std.clamp_min(eps)


@torch.no_grad()
def adaptive_gaussian_similarity(
    x1: torch.Tensor,
    x2: torch.Tensor,
    p: int = 2,
) -> torch.Tensor:
    """
    RBF-like similarity using empirical (average) std as bandwidth.
    sim[i,j] = exp(- ||x1[i]-x2[j]||_p^2 / (2*sigma^2))

    Args
    ----
    x1, x2 : [N,D] and [M,D] tensors (float32/float64)
    p      : Minkowski order (2 = Euclidean)

    Returns
    -------
    sim : [N,M] tensor in [0,1]
    """
    # bandwidth from x1 only; variants could use concat or average of stds
    sigma = _safe_std(x1, dim=0).mean()  # scalar tensor
    # torch.cdist supports p in {1, 2, ...}
    dists = torch.cdist(x1, x2, p=p)     # [N,M]
    sim = torch.exp(-(dists ** 2) / (2 * sigma ** 2))
    return sim


def minkowski_distance(
    x1: torch.Tensor,
    x2: torch.Tensor,
    p: int = 2,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Lp distance; if normalize=True, per-feature z-score before distance.

    Args
    ----
    x1, x2 : [N,D], [M,D]
    p      : Minkowski order (1=L1, 2=L2)
    normalize : if True, standardize features independently for x1 and x2
    eps    : numerical floor for std

    Returns
    -------
    dists : [N,M]
    """
    if normalize:
        mean1, std1 = x1.mean(dim=0, keepdim=True), _safe_std(x1, dim=0, eps=eps)
        mean2, std2 = x2.mean(dim=0, keepdim=True), _safe_std(x2, dim=0, eps=eps)
        x1 = (x1 - mean1) / std1
        x2 = (x2 - mean2) / std2
    return torch.cdist(x1, x2, p=p)


@torch.no_grad()
def direction_distance(l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
    """
    Binary categorical "distance": 1 if labels are equal, else 0.

    Args
    ----
    l1, l2 : [N], [M] integer (or float castable) label vectors

    Returns
    -------
    mat : [N,M] with {0,1}
    """
    l1 = l1.view(-1)
    l2 = l2.view(-1)
    return (l1[:, None] == l2[None, :]).to(torch.int)


@torch.no_grad()
def circular_distance(
    l1: torch.Tensor,
    l2: torch.Tensor,
    num_directions: int = 8,
) -> torch.Tensor:
    """
    Distance on a circular label space Z_K (e.g., 8-way movement directions).

    d_circ(a,b) = min(|a-b|, K-|a-b|)

    Args
    ----
    l1, l2 : [N], [M] integer labels in [0, K-1]
    num_directions : K (period)

    Returns
    -------
    dists : [N,M] in [0, floor(K/2)]
    """
    l1 = l1.view(-1).to(torch.long)
    l2 = l2.view(-1).to(torch.long)
    diff = torch.abs(l1[:, None] - l2[None, :])
    return torch.minimum(diff, num_directions - diff)
