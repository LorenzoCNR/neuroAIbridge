# -*- coding: utf-8 -*-
"""Visualization utilities for embeddings, trajectories, and diagnostics."""

from .manifold_plots import (
    condition_averaged_trajectories,
    plot_condition_centroids_2d,
    plot_condition_trajectories_2d,
    plot_condition_trajectories_3d,
    plot_condition_trajectories_sphere,
    plot_direction_averaged_embedding,
    plot_direction_averaged_embedding_raw,
    plot_embedding_2d,
    plot_embedding_3d,
    plot_embedding_sphere,
)
from .plots import (
    plot_confusion_matrix,
    plot_raster,
    plot_training_curve,
)

__all__ = [
    "condition_averaged_trajectories",
    "plot_condition_centroids_2d",
    "plot_condition_trajectories_2d",
    "plot_condition_trajectories_3d",
    "plot_condition_trajectories_sphere",
    "plot_confusion_matrix",
    "plot_direction_averaged_embedding",
    "plot_direction_averaged_embedding_raw",
    "plot_embedding_2d",
    "plot_embedding_3d",
    "plot_embedding_sphere",
    "plot_raster",
    "plot_training_curve",
]
