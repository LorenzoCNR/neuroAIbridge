# riuso vecchie fuznioni 
# src/neurobridge/utils/io.py
import os
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.io import loadmat
import joblib as jl


def swap_labels(labels: np.ndarray, mapping: dict) -> np.ndarray:
    """
    Swap labels according to a mapping dictionary (e.g., {3: 6, 6: 3}).

    Args:
        labels: 1D array of integer labels.
        mapping: dict mapping old_label -> new_label.

    Returns:
        A copy of labels with values remapped.
    """
    out = labels.copy()
    for old, new in mapping.items():
        out[labels == old] = new
    return out

def build_trial_intervals(trial_ids: np.ndarray):
    """
    Build (start, end) inclusive intervals for each contiguous block of the same trial id.

    Args:
        trial_ids: 1D array of integers (trial identifier per sample).

    Returns:
        List of (start, end) tuples, inclusive.
    """
    change_idx = np.where(np.diff(trial_ids) != 0)[0] + 1
    boundaries = np.concatenate([[0], change_idx, [len(trial_ids)]])
    return [(int(boundaries[i]), int(boundaries[i+1] - 1)) for i in range(len(boundaries) - 1)]
