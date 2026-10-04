# -*- coding: utf-8 -*-
"""
Sampling and pair-construction utilities for NeuroBridge.

This package contains three related groups of tools:

- temporal window extraction;
- pairwise metadata distances and similarities;
- positive-pair masks and weights for contrastive learning.

The package-level imports expose the stable public API. Legacy NumPy
utilities remain available from ``f_distances.py`` and ``similarity_.py``
for older experiment scripts.
"""

from .batch_similarity import (
    batch_circular_label_distance,
    batch_continuous_distance,
    batch_distance_from_spec,
    batch_structured_similarity,
    batch_structured_similarity_from_specs,
    batch_temporal_distance,
    normalize_batch_distance,
)
from .cebra_behavior import CEBRASupervisedWindowDataset
from .cebra_time import CEBRATripletWindowDataset
from .f_windows import build_windows
from .labelled import (
    categorical_positive_mask,
    time_offset_positive_mask,
)
from .positive_weights import (
    distance_to_positive_weights,
    normalize_positive_weights,
)
from .similarity_ import dist_to_simi, dist_to_similarity

__all__ = [
    "batch_circular_label_distance",
    "batch_continuous_distance",
    "batch_distance_from_spec",
    "batch_structured_similarity",
    "batch_structured_similarity_from_specs",
    "batch_temporal_distance",
    "build_windows",
    "CEBRASupervisedWindowDataset",
    "CEBRATripletWindowDataset",
    "categorical_positive_mask",
    "distance_to_positive_weights",
    "dist_to_simi",
    "dist_to_similarity",
    "normalize_batch_distance",
    "normalize_positive_weights",
    "time_offset_positive_mask",
]
