## -*- coding: utf-8 -*-
# Building blocks adapted from the CEBRA model architecture.
# Original project:
# Schneider, Lee, Mathis et al., CEBRA.
#
# This project reuses and extends these temporal convolutional patterns.
#
# These modules are lightweight building blocks for 1D temporal nets:
# - _Skip: residual connection with optional temporal crop
# - Squeeze: drop singleton dimensions (robust to 3D/4D tensors)
# - _Norm: L2 feature normalization along channel dimension
# - _MeanAndConv: downsampled “skip” concatenated with a Conv1d branch

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn


class _Skip(nn.Module):
    """
    Residual/skip wrapper around an arbitrary submodule stack.

    Idea
    ----
    Apply self.module to a cropped temporal slice of the input, then add it
    back to the (possibly cropped) original input to form a residual sum:
       y = module(x[..., crop]) + x[..., crop]

    This helps when the temporal length changes (e.g., due to conv strides)
    and you want to align shapes before residual addition.

    Args
    ----
    *modules : nn.Module
        One or more modules combined into an internal nn.Sequential.
    crop : Tuple[int, int]
        Number of time steps to crop from (left, right). If right==0 or <=0,
        no right crop is applied.

    Input
    -----
    inp : Tensor [..., T]
        Temporal tensor (N,C,T) or (N,C,T,...) where the last axis is time.

    Output
    ------
    out : Tensor [..., T'] where T' = T - crop_left - crop_right
    """

    def __init__(self, *modules: nn.Module, crop=(1, 1)):
        super().__init__()
        if len(crop) != 2:
            raise ValueError("crop must contain exactly two elements: (left, right).")

        left, right = crop

        if not isinstance(left, int) or left < 0:
            raise ValueError("crop[0] must be a non-negative integer.")

        if right is not None and (not isinstance(right, int) or right < 0):
            raise ValueError("crop[1] must be None or a non-negative integer.")

        self.module = nn.Sequential(*modules)
        self.crop = slice(
            left,
            -right if right is not None and right > 0 else None,
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        cropped = inp[..., self.crop]
        branch = self.module(cropped)

        if branch.shape != cropped.shape:
            raise ValueError(
                "Residual branch and cropped input must have the same shape, "
                f"got branch={tuple(branch.shape)} and input={tuple(cropped.shape)}."
            )

        return cropped + branch


class Squeeze(nn.Module):
    """
    Remove a singleton dimension (robust to 3D/4D inputs).

    Rules
    -----
    - If input has shape (N, C, 1, T) → squeeze the third dim → (N, C, T)
    - Else if input has ≥3 dims and dim=2 is 1 → squeeze that
    - Else pass-through

    This is useful after temporal convs that leave an extra singleton axis.
    """

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if inp.ndim < 3:
            return inp

        if inp.size(2) == 1:
            return inp.squeeze(2)

        return inp


class _Norm(nn.Module):
    """
    L2-normalize features along channel dimension.

    Input:  x in R^{N, C, T}
    Output: x / ||x||_2 per time step (numerically safe)

    Notes
    -----
    - eps prevents division-by-zero for all-zero vectors.
    - Keepdim ensures broadcasting over (N, C, T).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be strictly positive.")
        self.eps = eps

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # Norm over channels (dim=1), keep dims for safe broadcasting
        denom = torch.norm(inp, dim=1, keepdim=True).clamp_min(self.eps)
        return inp / denom


class _MeanAndConv(nn.Module):
    """
    Conv1d branch + downsampled shortcut concatenated on channels.

    Pattern
    -------
    - Left path: Conv1d with stride=s (reduces temporal length by s)
    - Right path: interpolate(·, scale_factor=1/s) on the input (downsample)
    - Concatenate along channel dim: cat([conv(x), ds(x)[:,:,:,aligned_T]], dim=1)

    Args
    ----
    inp : int
        In channels for Conv1d.
    output : int
        Out channels for Conv1d branch (NOT counting the downsample concat).
    kernel : int
        Kernel size for Conv1d.
    stride : int
        Downsampling factor (must be >= 1).

    Input
    -----
    inp : Tensor [N, C, T]

    Output
    ------
    out : Tensor [N, C+output, T'] where T' ≈ ceil(T/stride)
    The temporal output length is determined by the Conv1d branch.
    The interpolated shortcut is cropped to match that length.
    """

    def __init__(self, inp: int, output: int, kernel: int, *, stride: int):
        super().__init__()
        if not isinstance(inp, int) or inp < 1:
            raise ValueError("inp must be a positive integer.")

        if not isinstance(output, int) or output < 1:
            raise ValueError("output must be a positive integer.")

        if not isinstance(kernel, int) or kernel < 1:
            raise ValueError("kernel must be a positive integer.")

        if not isinstance(stride, int) or stride < 1:
            raise ValueError("stride must be a positive integer.")
        self.downsample = stride
        self.layer = nn.Conv1d(inp, output, kernel, stride=stride, padding=0)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if inp.ndim != 3:
            raise ValueError(f"Expected input with shape (batch, channels, time), got {tuple(inp.shape)}.")
        conv = self.layer(inp)  # [N, output, T']
        # Downsample shortcut; use 'nearest' to avoid artifacts (no grads through interpolate issue here)
        ds = F.interpolate(inp, scale_factor=1.0 / self.downsample, mode="nearest")
        # Align time lengths in case of off-by-one
        T_out = conv.size(-1)
        ds = ds[..., :T_out]
        return torch.cat([conv, ds], dim=1)
#