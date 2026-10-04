# -*- coding: utf-8 -*-
"""
Static Matplotlib diagnostics for training and simulation experiments.

Interactive manifold visualizations remain in ``manifold_plots.py``.
This module contains compact figures that are convenient for PNG/PDF export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _save_or_show(
    figure,
    output_path: str | Path | None,
    *,
    show: bool,
    dpi: int,
) -> None:
    """Apply the shared save/show policy."""
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(figure)


def plot_training_curve(
    losses: Sequence[float],
    *,
    validation_losses: Sequence[float] | None = None,
    title: str = "Training loss",
    output_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
):
    """
    Plot training and optional validation loss by epoch.
    """
    train = np.asarray(losses, dtype=float).reshape(-1)

    if train.size == 0:
        raise ValueError("losses must contain at least one value.")
    if not np.all(np.isfinite(train)):
        raise ValueError("losses must contain finite values.")

    figure, axis = plt.subplots()
    epochs = np.arange(1, train.size + 1)
    axis.plot(epochs, train, label="train")

    if validation_losses is not None:
        validation = np.asarray(
            validation_losses,
            dtype=float,
        ).reshape(-1)

        if validation.shape != train.shape:
            raise ValueError(
                "validation_losses must match the number of training epochs."
            )
        if not np.all(np.isfinite(validation)):
            raise ValueError(
                "validation_losses must contain finite values."
            )

        axis.plot(epochs, validation, label="validation")
        axis.legend()

    axis.set(
        xlabel="Epoch",
        ylabel="Loss",
        title=title,
    )
    axis.grid(alpha=0.25)

    _save_or_show(
        figure,
        output_path,
        show=show,
        dpi=dpi,
    )
    return figure


def plot_confusion_matrix(
    matrix,
    *,
    class_names: Sequence[str] | None = None,
    normalize: bool = False,
    title: str = "Confusion matrix",
    output_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
):
    """
    Plot a square confusion matrix.

    With ``normalize=True``, each row is divided by its total.
    """
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")
    if np.any(matrix < 0) or not np.all(np.isfinite(matrix)):
        raise ValueError(
            "matrix must contain finite non-negative values."
        )

    displayed = matrix.copy()

    if normalize:
        row_sum = displayed.sum(axis=1, keepdims=True)
        displayed = np.divide(
            displayed,
            row_sum,
            out=np.zeros_like(displayed),
            where=row_sum > 0,
        )

    n_classes = displayed.shape[0]

    if class_names is None:
        class_names = [str(index) for index in range(n_classes)]
    elif len(class_names) != n_classes:
        raise ValueError(
            "class_names length must match matrix size."
        )

    figure, axis = plt.subplots()
    image = axis.imshow(displayed, aspect="equal")
    figure.colorbar(image, ax=axis)

    axis.set(
        xlabel="Predicted class",
        ylabel="True class",
        title=title,
    )
    axis.set_xticks(np.arange(n_classes), class_names)
    axis.set_yticks(np.arange(n_classes), class_names)

    threshold = displayed.max(initial=0.0) / 2.0

    for row in range(n_classes):
        for column in range(n_classes):
            value = displayed[row, column]
            text = f"{value:.2f}" if normalize else f"{value:.0f}"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    _save_or_show(
        figure,
        output_path,
        show=show,
        dpi=dpi,
    )
    return figure


def plot_raster(
    spikes,
    *,
    trial_index: int = 0,
    max_neurons: int | None = None,
    title: str | None = None,
    output_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
):
    """
    Plot a spike raster for one trial.

    Accepted shapes:
        (time, neurons)
        (trials, time, neurons)
    """
    spikes = np.asarray(spikes)

    if spikes.ndim == 3:
        if not 0 <= trial_index < spikes.shape[0]:
            raise ValueError("trial_index is outside the available trials.")
        spikes = spikes[trial_index]
    elif spikes.ndim != 2:
        raise ValueError(
            "spikes must have shape (time, neurons) or "
            "(trials, time, neurons)."
        )

    if max_neurons is not None:
        if max_neurons < 1:
            raise ValueError("max_neurons must be positive.")
        spikes = spikes[:, :max_neurons]

    time_index, neuron_index = np.nonzero(spikes > 0)

    figure, axis = plt.subplots()
    axis.scatter(
        time_index,
        neuron_index,
        s=5,
        marker="|",
    )
    axis.set(
        xlabel="Time bin",
        ylabel="Neuron",
        title=title or f"Spike raster — trial {trial_index}",
    )

    _save_or_show(
        figure,
        output_path,
        show=show,
        dpi=dpi,
    )
    return figure
