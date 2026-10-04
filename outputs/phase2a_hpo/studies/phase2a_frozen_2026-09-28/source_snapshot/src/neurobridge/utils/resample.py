# -*- coding: utf-8 -*-
"""Array resampling helpers."""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


def resample_array(
    data,
    original_rate: float,
    target_rate: float,
    *,
    axis: int = 0,
) -> np.ndarray:
    """
    Resample an array using polyphase filtering.

    Parameters
    ----------
    data:
        Input NumPy-compatible array.

    original_rate:
        Original sampling frequency.

    target_rate:
        Requested sampling frequency.

    axis:
        Time axis.

    Returns
    -------
    np.ndarray
        Resampled array.
    """
    data = np.asarray(data)

    if data.ndim == 0:
        raise ValueError("data must have at least one dimension.")
    if original_rate <= 0 or target_rate <= 0:
        raise ValueError("Sampling rates must be strictly positive.")

    axis = np.core.numeric.normalize_axis_index(axis, data.ndim)

    if np.isclose(original_rate, target_rate):
        return data.copy()

    # Convert rates to a bounded rational ratio. This supports both integer
    # and common decimal sampling rates without creating huge filter factors.
    from fractions import Fraction

    ratio = Fraction(
        float(target_rate) / float(original_rate)
    ).limit_denominator(10000)

    return resample_poly(
        data,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=axis,
    )
