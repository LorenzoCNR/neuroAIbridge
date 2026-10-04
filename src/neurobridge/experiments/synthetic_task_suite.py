"""Controlled circular and linear synthetic-task experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset

from neurobridge.data.dataset import TemporalWindowDataset
from neurobridge.data.sim import (
    LatentTrajectoryGenerator,
    build_linear_loading_and_place_fields,
    build_structured_B,
)
from neurobridge.data.sim.builders import drive_to_rate, rate_to_spike
from neurobridge.eval.representation import evaluate_latent_recovery
from neurobridge.losses.infonce import soft_contrastive_loss
from neurobridge.models.temporal_cnn import TemporalCNNEncoder
from neurobridge.sampling.batch_similarity import (
    batch_distance_from_spec,
    batch_temporal_distance,
    normalize_batch_distance,
)
from neurobridge.sampling.f_windows import build_windows
from neurobridge.train.loop import encode_windows, train_epoch
from neurobridge.viz import (
    condition_averaged_trajectories,
    plot_condition_trajectories_2d,
    plot_condition_trajectories_3d,
    plot_embedding_2d,
    plot_embedding_3d,
)


@dataclass(frozen=True)
class SyntheticTaskConfig:
    """Complete configuration for one controlled baseline experiment."""

    name: str
    condition_mode: str
    latent_dim: int
    n_trials: int = 160
    trial_length: int = 100
    n_neurons: int = 100
    n_conditions: int = 8
    window_size: int = 10
    stride: int = 1
    cnn_epochs: int = 30
    cnn_embedding_dim: int | None = None
    batch_size: int = 256
    learning_rate: float = 1e-3
    temperature: float = 0.1
    similarity_tau: float = 0.5
    time_weight: float = 0.5
    label_weight: float = 0.5
    phi: float = 0.4
    noise_scale: float = 0.05
    dt: float = 0.02
    baseline_mean: float = 1.0
    baseline_std: float = 0.10
    rate_scale: float = 10.0
    first_coordinates_multiplier: float = 3.0
    place_fraction: float = 0.25
    place_width: float = 0.10
    place_scale: float = 3.0
    n_position_bins: int = 20
    gradient_fraction: float = 0.0
    nonpreferred_direction_gain: float = 0.10
    train_fraction: float = 0.8
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.condition_mode not in {"circular", "linear"}:
            raise ValueError("condition_mode must be circular or linear.")
        minimum_dim = 3 if self.condition_mode == "circular" else 2
        if self.latent_dim < minimum_dim:
            raise ValueError(
                f"{self.condition_mode} requires latent_dim >= {minimum_dim}."
            )
        if not 0.0 <= self.place_fraction < 1.0:
            raise ValueError("place_fraction must satisfy 0 <= value < 1.")
        if self.place_width <= 0 or self.place_scale < 0:
            raise ValueError("Place-field width and scale must be positive.")
        if self.n_position_bins < 2:
            raise ValueError("n_position_bins must be at least 2.")
        if not 0.0 <= self.gradient_fraction < 1.0:
            raise ValueError("gradient_fraction must satisfy 0 <= value < 1.")
        if self.place_fraction + self.gradient_fraction >= 1.0:
            raise ValueError(
                "place_fraction + gradient_fraction must be smaller than 1."
            )
        if not 0.0 <= self.nonpreferred_direction_gain <= 1.0:
            raise ValueError(
                "nonpreferred_direction_gain must lie between 0 and 1."
            )
        if self.rate_scale <= 0:
            raise ValueError("rate_scale must be strictly positive.")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must lie strictly between 0 and 1.")
        if self.temperature <= 0 or self.similarity_tau <= 0:
            raise ValueError("Temperatures must be strictly positive.")
        if self.time_weight < 0 or self.label_weight < 0:
            raise ValueError("Distance weights cannot be negative.")
        if self.time_weight + self.label_weight <= 0:
            raise ValueError("At least one distance weight must be positive.")

    @property
    def embedding_dim(self) -> int:
        return self.latent_dim if self.cnn_embedding_dim is None else (
            self.cnn_embedding_dim
        )


def circular_neuron_type_probabilities(k: int) -> dict[str, float]:
    """Return the neuron-type mixture used by a circular-task experiment."""
    if k == 3:
        return {
            "direction": 0.45,
            "position_or_progress": 0.20,
            "velocity": 0.0,
            "context": 0.0,
            "mixed": 0.34,
            "none": 0.01,
        }
    return {
        "direction": 0.45,
        "position_or_progress": 0.20,
        "velocity": 0.02,
        "context": 0.02,
        "mixed": 0.30,
        "none": 0.01,
    }


def build_windows_and_labels(
    X: np.ndarray,
    condition: np.ndarray | None,
    state: dict[str, np.ndarray],
    config: SyntheticTaskConfig,
) -> tuple[TemporalWindowDataset, dict[str, np.ndarray]]:
    """
    Create centered trial-safe windows and aligned task metadata.

    Returns one :class:`TemporalWindowDataset` sample per retained time bin.
    Each sample contains the neural window plus trial, time, label, and
    movement-progress metadata used by the contrastive target.
    """
    X_flat = X.reshape(-1, X.shape[-1])
    labels = condition if config.condition_mode == "circular" else None

    (
        X_windows,
        time_id,
        global_time_id,
        trial_id,
        labels_windows,
    ) = build_windows(
        X_flat,
        config.window_size,
        config.stride,
        labels=labels,
        trial_len=config.trial_length,
        time_mode="absolute",
        padding="center",
    )

    if config.condition_mode == "linear":
        local_time = np.asarray(time_id, dtype=int)
        velocity = state["velocity"][trial_id, local_time]
        labels_windows = (velocity < 0).astype(int)

    progress = state["phase"][
        np.asarray(trial_id, dtype=int),
        np.asarray(time_id, dtype=int),
    ]
    dataset = TemporalWindowDataset(
        X_windows,
        time_id,
        global_time_id,
        trial_id,
        labels_windows,
        extra_metadata={"progress": progress},
    )
    metadata = {
        "time_id": np.asarray(time_id),
        "global_time_id": np.asarray(global_time_id),
        "trial_id": np.asarray(trial_id),
        "labels": np.asarray(labels_windows),
        "progress": np.asarray(progress),
    }
    if config.condition_mode == "linear":
        metadata["movement_phase"] = np.asarray(labels_windows)
    return dataset, metadata


def build_similarity_matrix(
    batch: dict[str, torch.Tensor],
    config: SyntheticTaskConfig,
) -> torch.Tensor:
    """
    Build the batch-wise soft target from temporal and task metadata.

    The returned square matrix has one row and column per mini-batch sample.
    Time distance and categorical label distance are normalized, weighted,
    and converted to similarity through an exponential kernel.
    """
    temporal_distance = normalize_batch_distance(
        batch_temporal_distance(batch["time_id"])
    )
    condition_distance = batch_distance_from_spec(
        batch,
        {"key": "label", "geometry": "categorical"},
    )

    weight_sum = config.time_weight + config.label_weight
    total_distance = (
        config.time_weight * temporal_distance
        + config.label_weight * condition_distance
    ) / weight_sum
    return torch.exp(-total_distance / config.similarity_tau)


def split_trials(
    metadata: dict[str, np.ndarray],
    condition: np.ndarray | None,
    config: SyntheticTaskConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a reproducible trial-level train/test split."""
    rng = np.random.default_rng(config.random_state + 3)
    all_trials = np.arange(config.n_trials)

    train_trials: list[int] = []
    if condition is None:
        shuffled = rng.permutation(all_trials)
        n_train = int(round(config.train_fraction * len(shuffled)))
        train_trials.extend(shuffled[:n_train].tolist())
    else:
        for label in np.unique(condition):
            candidates = all_trials[condition == label]
            candidates = rng.permutation(candidates)
            n_train = int(round(config.train_fraction * len(candidates)))
            train_trials.extend(candidates[:n_train].tolist())

    train_trials_array = np.sort(np.asarray(train_trials, dtype=int))
    test_trials_array = np.setdiff1d(all_trials, train_trials_array)
    train_mask = np.isin(metadata["trial_id"], train_trials_array)
    test_mask = np.isin(metadata["trial_id"], test_trials_array)
    return train_trials_array, test_trials_array, train_mask, test_mask


def fit_models(
    dataset: TemporalWindowDataset,
    config: SyntheticTaskConfig,
    device: torch.device,
    train_mask: np.ndarray,
) -> dict[str, object]:
    """
    Fit the standard PCA and CNN1D baselines.

    This convenience function is used by non-interactive scripts. The
    documented notebooks instead expose PCA construction, data loaders,
    optimizer, target function, loss, epoch loop, and ordered encoding in
    separate cells.
    """
    X_windows = dataset.X_windows.numpy()
    flattened = X_windows.reshape(len(dataset), -1)
    train_indices = np.flatnonzero(train_mask)
    pca_dim = min(config.latent_dim, flattened.shape[1])
    pca = PCA(n_components=pca_dim, random_state=config.random_state)
    pca.fit(flattened[train_indices])
    pca_embedding = pca.transform(flattened)

    model = TemporalCNNEncoder(
        n_features=X_windows.shape[-1],
        embedding_dim=config.embedding_dim,
        hidden_dim=64,
        kernel_size=3,
        n_layers=3,
        normalize=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4,
    )
    training_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    losses: list[float] = []
    for _ in range(config.cnn_epochs):
        loss = train_epoch(
            model,
            training_loader,
            optimizer,
            lambda z, similarity: soft_contrastive_loss(
                z,
                similarity,
                temperature=config.temperature,
            ),
            device=device,
            similarity_builder=lambda batch: build_similarity_matrix(
                batch,
                config,
            ),
        )
        losses.append(float(loss))

    ordered_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    cnn_embedding, _ = encode_windows(model, ordered_loader, device=device)
    return {
        "pca_model": pca,
        "pca_embedding": pca_embedding,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        "cnn_model": model,
        "cnn_embedding": cnn_embedding.numpy(),
        "cnn_losses": np.asarray(losses),
        "cnn_training_steps": config.cnn_epochs * len(training_loader),
    }


def evaluate_models(
    *,
    Z: np.ndarray,
    metadata: dict[str, np.ndarray],
    fitted: dict[str, object],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    config: SyntheticTaskConfig,
    max_samples: int = 2000,
) -> dict[str, dict[str, float]]:
    """
    Compare PCA and CNN embeddings with the known latent state.

    Metrics are reported for all sampled windows, fitting windows, untouched
    test windows, trial-averaged trajectories, and task-defining motor-core
    coordinates.
    """
    trial_id = metadata["trial_id"].astype(int)
    time_id = metadata["time_id"].astype(int)
    latent_target = Z[trial_id, time_id]

    def evaluation_indices(mask: np.ndarray) -> np.ndarray:
        candidates = np.flatnonzero(mask)
        if len(candidates) <= max_samples:
            return candidates
        selected = np.linspace(
            0,
            len(candidates) - 1,
            max_samples,
            dtype=int,
        )
        return candidates[selected]

    all_indices = evaluation_indices(
        np.ones(len(latent_target), dtype=bool)
    )
    train_indices = evaluation_indices(train_mask)
    test_indices = evaluation_indices(test_mask)

    latent_trajectories = condition_averaged_trajectories(
        latent_target,
        metadata["labels"],
        metadata["time_id"],
    )
    latent_trajectory = np.concatenate(
        list(latent_trajectories.values()),
        axis=0,
    )
    core_dimensions = 3 if config.condition_mode == "circular" else 2

    metrics: dict[str, dict[str, float]] = {}
    for model_name, embedding_key in [
        ("pca", "pca_embedding"),
        ("cnn1d", "cnn_embedding"),
    ]:
        embedding = np.asarray(fitted[embedding_key])
        embedding_trajectories = condition_averaged_trajectories(
            embedding,
            metadata["labels"],
            metadata["time_id"],
        )
        embedding_trajectory = np.concatenate(
            [
                embedding_trajectories[label]
                for label in latent_trajectories
            ],
            axis=0,
        )

        all_scores = evaluate_latent_recovery(
            embedding[all_indices],
            latent_target[all_indices],
        )
        train_scores = evaluate_latent_recovery(
            embedding[train_indices],
            latent_target[train_indices],
        )
        test_scores = evaluate_latent_recovery(
            embedding[test_indices],
            latent_target[test_indices],
        )
        trajectory_scores = evaluate_latent_recovery(
            embedding_trajectory,
            latent_trajectory,
        )
        core_test_scores = evaluate_latent_recovery(
            embedding[test_indices],
            latent_target[test_indices, :core_dimensions],
        )
        core_trajectory_scores = evaluate_latent_recovery(
            embedding_trajectory,
            latent_trajectory[:, :core_dimensions],
        )

        start_points = np.stack([
            trajectory[0]
            for trajectory in embedding_trajectories.values()
        ])
        end_points = np.stack([
            trajectory[-1]
            for trajectory in embedding_trajectories.values()
        ])
        start_spread = float(pdist(start_points).mean())
        end_spread = float(pdist(end_points).mean())
        metrics[model_name] = {
            **all_scores,
            **{
                f"train_{key}": value
                for key, value in train_scores.items()
            },
            **{
                f"test_{key}": value
                for key, value in test_scores.items()
            },
            **{
                f"trajectory_{key}": value
                for key, value in trajectory_scores.items()
            },
            **{
                f"core_test_{key}": value
                for key, value in core_test_scores.items()
            },
            **{
                f"core_trajectory_{key}": value
                for key, value in core_trajectory_scores.items()
            },
            "trajectory_start_end_spread_ratio": (
                start_spread / max(end_spread, 1e-12)
            ),
        }

    return metrics


def save_experiment_figures(
    results: dict[str, object],
    output_root: Path,
) -> None:
    """Save trajectory-first figures and secondary scatter diagnostics."""
    metadata = results["metadata"]
    labels = metadata["labels"]
    time_id = metadata["time_id"]
    trial_id = metadata["trial_id"]
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    Z = np.asarray(results["Z"])
    latent_target = Z[trial_id.astype(int), time_id.astype(int)]
    condition_mode = results["config"]["condition_mode"]
    if condition_mode == "linear":
        track_position_target = np.asarray(
            results["state"]["position"]
        )[
            trial_id.astype(int),
            time_id.astype(int),
            0,
        ]
        movement_direction_target = np.where(
            np.asarray(labels, dtype=int) == 0,
            1.0,
            -1.0,
        )
        plotted_ground_truth = np.column_stack([
            track_position_target,
            movement_direction_target,
        ])
    else:
        track_position_target = None
        plotted_ground_truth = latent_target
    display_labels = (
        np.where(
            np.asarray(labels, dtype=int) == 0,
            "Outbound phase (0 -> 1)",
            "Return phase (1 -> 0)",
        )
        if condition_mode == "linear"
        else labels
    )
    latent_axis_labels = (
        ("Position X", "Position Y")
        if condition_mode == "circular"
        else (
            "Normalized track position",
            "Movement direction (+1 outbound, -1 return)",
        )
    )
    cnn_embedding = np.asarray(results["cnn_embedding"])
    cnn_axis_labels_2d = ("Embedding 1", "Embedding 2")
    cnn_axis_labels_3d = (
        "Embedding 1",
        "Embedding 2",
        "Embedding 3",
    )
    cnn_display_title = "CNN1D"

    trajectory_specs = [
        (
            plotted_ground_truth,
            "ground_truth_trajectories_2d.html",
            "Ground-truth condition-averaged trajectories",
            latent_axis_labels,
        ),
        (
            np.asarray(results["pca_embedding"]),
            "pca_condition_averaged_trajectories_2d.html",
            "PCA condition-averaged trajectories",
            ("PC1", "PC2"),
        ),
        (
            cnn_embedding,
            "cnn1d_condition_averaged_trajectories_2d.html",
            f"{cnn_display_title} condition-averaged trajectories",
            cnn_axis_labels_2d,
        ),
    ]

    for embedding, filename, title, axis_labels in trajectory_specs:
        plot_condition_trajectories_2d(
            embedding=embedding,
            labels=display_labels,
            trial_id=trial_id,
            time_id=time_id,
            output_folder=figure_root,
            name=filename,
            title=f"{title}: {results['config']['name']}",
            dims=(0, 1),
            axis_labels=axis_labels,
            show=False,
        )

    trajectory_specs_3d = [
        (
            plotted_ground_truth,
            "ground_truth_trajectories_3d.html",
            "Ground-truth condition-averaged trajectories",
            ("Latent 1", "Latent 2", "Latent 3"),
        ),
        (
            np.asarray(results["pca_embedding"]),
            "pca_condition_averaged_trajectories_3d.html",
            "PCA condition-averaged trajectories",
            ("PC1", "PC2", "PC3"),
        ),
        (
            cnn_embedding,
            "cnn1d_condition_averaged_trajectories_3d.html",
            f"{cnn_display_title} condition-averaged trajectories",
            cnn_axis_labels_3d,
        ),
    ]
    for embedding, filename, title, axis_labels in trajectory_specs_3d:
        if embedding.shape[1] < 3:
            continue
        plot_condition_trajectories_3d(
            embedding=embedding,
            labels=display_labels,
            trial_id=trial_id,
            time_id=time_id,
            output_folder=figure_root,
            name=filename,
            title=f"{title}: {results['config']['name']}",
            axis_labels=axis_labels,
            show=False,
        )

    for embedding, filename, title, axis_labels in [
        (
            np.asarray(results["pca_embedding"]),
            "diagnostic_pca_all_windows_scatter_2d.html",
            "Diagnostic PCA scatter of all windows",
            ("PC1", "PC2"),
        ),
        (
            cnn_embedding,
            "diagnostic_cnn1d_all_windows_scatter_2d.html",
            f"Diagnostic {cnn_display_title} scatter of all windows",
            cnn_axis_labels_2d,
        ),
    ]:
        plot_embedding_2d(
            embedding=embedding,
            labels=display_labels,
            output_folder=figure_root,
            name=filename,
            title=f"{title}: {results['config']['name']}",
            dims=(0, 1),
            axis_labels=axis_labels,
            show=False,
        )

    for embedding, filename, title, axis_labels in [
        (
            np.asarray(results["pca_embedding"]),
            "diagnostic_pca_all_windows_scatter_3d.html",
            "Diagnostic PCA scatter of all windows",
            ("PC1", "PC2", "PC3"),
        ),
        (
            cnn_embedding,
            "diagnostic_cnn1d_all_windows_scatter_3d.html",
            f"Diagnostic {cnn_display_title} scatter of all windows",
            cnn_axis_labels_3d,
        ),
    ]:
        if embedding.shape[1] < 3:
            continue
        plot_embedding_3d(
            embedding=embedding,
            labels=display_labels,
            output_folder=figure_root,
            name=filename,
            title=f"{title}: {results['config']['name']}",
            axis_labels=axis_labels,
            show=False,
        )

    if condition_mode == "linear":
        fig, all_axes = plt.subplots(
            1,
            4,
            figsize=(19, 5),
            constrained_layout=True,
        )
        behavior_axis = all_axes[0]
        axes = all_axes[1:]

        behavior_axis.plot(
            [0.0, 1.0],
            [0.0, 0.0],
            color="#17202a",
            linewidth=7,
            solid_capstyle="round",
        )
        behavior_axis.scatter(
            [0.0, 1.0],
            [0.0, 0.0],
            color=[plt.cm.viridis(0.0), plt.cm.viridis(1.0)],
            edgecolor="#17202a",
            linewidth=0.8,
            s=70,
            zorder=3,
        )
        behavior_axis.annotate(
            "Outbound: 0 -> 1",
            xy=(0.88, 0.22),
            xytext=(0.12, 0.22),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#176b55",
                "linewidth": 2.2,
            },
            ha="center",
            color="#176b55",
            fontsize=9,
        )
        behavior_axis.annotate(
            "Return: 1 -> 0",
            xy=(0.12, -0.22),
            xytext=(0.88, -0.22),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#b63c35",
                "linewidth": 2.2,
            },
            ha="center",
            color="#b63c35",
            fontsize=9,
        )
        behavior_axis.set_title("Physical task")
        behavior_axis.set_xlabel("Normalized track position")
        behavior_axis.set_xlim(-0.12, 1.12)
        behavior_axis.set_ylim(-0.42, 0.42)
        behavior_axis.set_yticks([])
        behavior_axis.spines[["left", "right", "top"]].set_visible(False)
    else:
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5),
            constrained_layout=True,
        )
        all_axes = axes
    static_specs = [
        (
            plotted_ground_truth,
            (
                "Expected position-direction cycle"
                if condition_mode == "linear"
                else "Ground-truth latent state"
            ),
            latent_axis_labels,
        ),
        (np.asarray(results["pca_embedding"]), "PCA", ("PC1", "PC2")),
        (
            cnn_embedding,
            cnn_display_title,
            cnn_axis_labels_2d,
        ),
    ]
    for spec_index, (
        axis,
        (embedding, title, axis_labels),
    ) in enumerate(zip(axes, static_specs)):
        trajectories = condition_averaged_trajectories(
            embedding,
            display_labels,
            time_id,
        )
        if condition_mode == "linear":
            for label, trajectory in trajectories.items():
                label_mask = display_labels == label
                phase_times = np.unique(time_id[label_mask])
                positions = np.array([
                    track_position_target[
                        label_mask & (time_id == time_value)
                    ].mean()
                    for time_value in phase_times
                ])
                points = trajectory[:, :2].reshape(-1, 1, 2)
                segments = np.concatenate(
                    [points[:-1], points[1:]],
                    axis=1,
                )
                line_style = "--" if str(label).startswith("Return") else "-"
                collection = LineCollection(
                    segments,
                    cmap="viridis",
                    norm=plt.Normalize(0.0, 1.0),
                    linewidth=2.6,
                    linestyle=line_style,
                )
                collection.set_array(
                    (positions[:-1] + positions[1:]) / 2.0
                )
                axis.add_collection(collection)
                axis.scatter(
                    trajectory[0, 0],
                    trajectory[0, 1],
                    color=plt.cm.viridis(positions[0]),
                    edgecolor="black",
                    linewidth=0.5,
                    s=38,
                    zorder=3,
                )
                axis.scatter(
                    trajectory[-1, 0],
                    trajectory[-1, 1],
                    color="black",
                    marker="x",
                    s=48,
                    zorder=4,
                )
                axis.autoscale_view()
            if spec_index == 0:
                axis.clear()
                axis.annotate(
                    "Outbound: direction +1",
                    xy=(0.95, 1.0),
                    xytext=(0.05, 1.0),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "#176b55",
                        "linewidth": 3.0,
                    },
                    ha="center",
                    va="bottom",
                    color="#176b55",
                    fontsize=9,
                )
                axis.annotate(
                    "Return: direction -1",
                    xy=(0.05, -1.0),
                    xytext=(0.95, -1.0),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "#b63c35",
                        "linewidth": 3.0,
                    },
                    ha="center",
                    va="top",
                    color="#b63c35",
                    fontsize=9,
                )
                axis.plot(
                    [1.0, 1.0],
                    [1.0, -1.0],
                    color="#17202a",
                    linewidth=1.8,
                )
                axis.annotate(
                    "turnaround",
                    xy=(1.0, 0.0),
                    xytext=(0.72, 0.0),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "#17202a",
                    },
                    fontsize=8,
                )
                axis.plot(
                    [0.0, 0.0],
                    [-1.0, 1.0],
                    color="#7d8790",
                    linewidth=1.5,
                    linestyle=":",
                )
                axis.text(
                    0.03,
                    0.0,
                    "next trial",
                    color="#58636f",
                    fontsize=8,
                    va="center",
                )
                axis.set_xlim(-0.12, 1.12)
                axis.set_ylim(-1.35, 1.35)
                axis.set_yticks([-1.0, 1.0])
                axis.set_yticklabels(["Return", "Outbound"])
        else:
            colors = plt.cm.hsv(
                np.linspace(0.0, 0.88, max(len(trajectories), 1))
            )
            for color, (label, trajectory) in zip(
                colors,
                trajectories.items(),
            ):
                axis.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    color=color,
                    linewidth=2.2,
                    label=f"cond {label}",
                )
                axis.scatter(
                    trajectory[0, 0],
                    trajectory[0, 1],
                    color=color,
                    s=32,
                )
                axis.scatter(
                    trajectory[-1, 0],
                    trajectory[-1, 1],
                    color="black",
                    marker="x",
                    s=42,
                )
        axis.set_title(title)
        axis.set_xlabel(axis_labels[0])
        axis.set_ylabel(axis_labels[1])
        if condition_mode == "linear" and spec_index == 0:
            axis.set_aspect("auto")
        else:
            axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)

    if condition_mode == "linear":
        handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=2.4,
                linestyle="-",
                label="Outbound phase (0 -> 1)",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=2.4,
                linestyle="--",
                label="Return phase (1 -> 0)",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                marker="x",
                linestyle="none",
                label="Arrival",
            ),
        ]
        legend_labels = [handle.get_label() for handle in handles]
        position_mappable = plt.cm.ScalarMappable(
            norm=plt.Normalize(0.0, 1.0),
            cmap="viridis",
        )
        fig.colorbar(
            position_mappable,
            ax=all_axes,
            label="Track position (0 -> 1)",
            shrink=0.72,
            pad=0.02,
        )
    else:
        handles, legend_labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="outside lower center",
            ncol=min(len(handles), 8),
        )
    figure_kind = (
        (
            "Linear-track recovery: position + direction"
            if results["config"]["latent_dim"] == 2
            else (
                "Linear-track recovery: position + direction "
                "+ velocity + context"
            )
        )
        if condition_mode == "linear"
        else "Condition-averaged trajectories"
    )
    fig.suptitle(f"{figure_kind}: {results['config']['name']}")
    fig.savefig(
        figure_root / "condition_averaged_trajectories_2d.png",
        dpi=180,
    )
    plt.close(fig)

    if all(embedding.shape[1] >= 3 for embedding, _, _ in static_specs):
        fig = plt.figure(figsize=(16, 5), constrained_layout=True)
        axes_3d = [
            fig.add_subplot(1, 3, index + 1, projection="3d")
            for index in range(3)
        ]
        labels_3d = [
            ("Latent 1", "Latent 2", "Latent 3"),
            ("PC1", "PC2", "PC3"),
            cnn_axis_labels_3d,
        ]
        for axis, (embedding, title, _), axis_labels in zip(
            axes_3d,
            static_specs,
            labels_3d,
        ):
            trajectories = condition_averaged_trajectories(
                embedding,
                display_labels,
                time_id,
            )
            if condition_mode == "linear":
                for label, trajectory in trajectories.items():
                    label_mask = display_labels == label
                    phase_times = np.unique(time_id[label_mask])
                    positions = np.array([
                        track_position_target[
                            label_mask & (time_id == time_value)
                        ].mean()
                        for time_value in phase_times
                    ])
                    points = trajectory[:, :3].reshape(-1, 1, 3)
                    segments = np.concatenate(
                        [points[:-1], points[1:]],
                        axis=1,
                    )
                    line_style = (
                        "--" if str(label).startswith("Return") else "-"
                    )
                    collection = Line3DCollection(
                        segments,
                        cmap="viridis",
                        norm=plt.Normalize(0.0, 1.0),
                        linewidth=2.4,
                        linestyle=line_style,
                    )
                    collection.set_array(
                        (positions[:-1] + positions[1:]) / 2.0
                    )
                    axis.add_collection3d(collection)
                    axis.scatter(
                        trajectory[0, 0],
                        trajectory[0, 1],
                        trajectory[0, 2],
                        color=plt.cm.viridis(positions[0]),
                        edgecolor="black",
                        linewidth=0.5,
                        s=30,
                    )
                    axis.scatter(
                        trajectory[-1, 0],
                        trajectory[-1, 1],
                        trajectory[-1, 2],
                        color="black",
                        marker="x",
                        s=42,
                    )
                    axis.auto_scale_xyz(
                        trajectory[:, 0],
                        trajectory[:, 1],
                        trajectory[:, 2],
                    )
            else:
                colors = plt.cm.hsv(
                    np.linspace(0.0, 0.88, max(len(trajectories), 1))
                )
                for color, (label, trajectory) in zip(
                    colors,
                    trajectories.items(),
                ):
                    axis.plot(
                        trajectory[:, 0],
                        trajectory[:, 1],
                        trajectory[:, 2],
                        color=color,
                        linewidth=2.0,
                        label=f"cond {label}",
                    )
                    axis.scatter(
                        trajectory[0, 0],
                        trajectory[0, 1],
                        trajectory[0, 2],
                        color=color,
                        s=24,
                    )
                    axis.scatter(
                        trajectory[-1, 0],
                        trajectory[-1, 1],
                        trajectory[-1, 2],
                        color="black",
                        marker="x",
                        s=36,
                    )
            axis.set_title(title)
            axis.set_xlabel(axis_labels[0])
            axis.set_ylabel(axis_labels[1])
            axis.set_zlabel(axis_labels[2])
            axis.set_box_aspect((1, 1, 1))

        if condition_mode == "linear":
            handles = [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=2.4,
                    linestyle="-",
                    label="Outbound phase (0 -> 1)",
                ),
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=2.4,
                    linestyle="--",
                    label="Return phase (1 -> 0)",
                ),
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker="x",
                    linestyle="none",
                    label="Arrival",
                ),
            ]
            legend_labels = [handle.get_label() for handle in handles]
            position_mappable = plt.cm.ScalarMappable(
                norm=plt.Normalize(0.0, 1.0),
                cmap="viridis",
            )
            fig.colorbar(
                position_mappable,
                ax=axes_3d,
                label="Track position (0 -> 1)",
                shrink=0.66,
                pad=0.02,
            )
        else:
            handles, legend_labels = (
                axes_3d[-1].get_legend_handles_labels()
            )
        if handles:
            fig.legend(
                handles,
                legend_labels,
                loc="outside lower center",
                ncol=min(len(handles), 8),
            )
        fig.suptitle(f"{figure_kind}: {results['config']['name']}")
        fig.savefig(
            figure_root / "condition_averaged_trajectories_3d.png",
            dpi=180,
        )
        plt.close(fig)

    if condition_mode == "linear":
        neuron_types = np.asarray(results["neuron_types"])
        place_indices = np.flatnonzero(
            np.isin(neuron_types, ["positional", "mixed", "place"])
        )
        if len(place_indices) > 0:
            centers = np.asarray(results["place_centers"])[place_indices]
            order = np.argsort(centers)
            place_indices = place_indices[order]
            centers = centers[order]

            track_position = np.asarray(
                results["state"]["position"]
            )[:, :, 0]
            rates = np.asarray(results["lam"])[:, :, place_indices]
            bin_edges = np.linspace(0.0, 1.0, 31)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            tuning = np.zeros((len(place_indices), len(bin_centers)))
            for bin_index in range(len(bin_centers)):
                mask = (
                    (track_position >= bin_edges[bin_index])
                    & (track_position < bin_edges[bin_index + 1])
                )
                if bin_index == len(bin_centers) - 1:
                    mask |= track_position == 1.0
                if np.any(mask):
                    tuning[:, bin_index] = rates[mask].mean(axis=0)

            tuning_min = tuning.min(axis=1, keepdims=True)
            tuning_range = np.ptp(tuning, axis=1, keepdims=True)
            tuning_normalized = (
                (tuning - tuning_min) / np.maximum(tuning_range, 1e-12)
            )

            fig, axes = plt.subplots(
                1,
                2,
                figsize=(12, 4.5),
                constrained_layout=True,
            )
            example_count = min(8, len(place_indices))
            example_indices = np.linspace(
                0,
                len(place_indices) - 1,
                example_count,
                dtype=int,
            )
            colors = plt.cm.viridis(
                np.linspace(0.05, 0.95, example_count)
            )
            width = results["config"]["place_width"]
            position_grid = np.linspace(0.0, 1.0, 400)
            for color, index in zip(colors, example_indices):
                field = np.exp(
                    -((position_grid - centers[index]) ** 2)
                    / (2.0 * width**2)
                )
                axes[0].plot(
                    position_grid,
                    field,
                    color=color,
                    linewidth=2.0,
                )
            axes[0].set_title("Assigned spatial fields")
            axes[0].set_xlabel("Normalized track position")
            axes[0].set_ylabel("Normalized place drive")
            axes[0].set_xlim(0.0, 1.0)
            axes[0].grid(alpha=0.18)

            image = axes[1].imshow(
                tuning_normalized,
                origin="lower",
                aspect="auto",
                extent=(0.0, 1.0, 0, len(place_indices)),
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            axes[1].set_title("Realized position tuning in simulated rates")
            axes[1].set_xlabel("Normalized track position")
            axes[1].set_ylabel("Place-selective neurons, sorted by center")
            fig.colorbar(
                image,
                ax=axes[1],
                label="Within-neuron normalized mean rate",
            )
            fig.suptitle(
                f"Linear-track place-field check: "
                f"{results['config']['name']}"
            )
            fig.savefig(
                figure_root / "realized_place_field_tuning.png",
                dpi=180,
            )
            plt.close(fig)


def run_synthetic_task_experiment(
    config: SyntheticTaskConfig,
    *,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    """Generate spikes, fit PCA/CNN, save artifacts, and return results."""
    random.seed(config.random_state)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path.cwd().resolve()
    )
    output_root = root / "outputs" / config.name
    output_root.mkdir(parents=True, exist_ok=True)

    conditions = (
        np.arange(config.n_conditions)
        if config.condition_mode == "circular"
        else None
    )
    generator = LatentTrajectoryGenerator(
        config.n_trials,
        config.trial_length,
        config.latent_dim,
        config.phi,
        conditions=conditions,
        condition_mode=config.condition_mode,
        n_conditions=config.n_conditions,
        noise_scale=config.noise_scale,
        condition_type="balanced",
    )
    Z, condition, state = generator.generate_latent(return_state=True)

    if config.condition_mode == "circular":
        B, neuron_types = build_structured_B(
            k=config.latent_dim,
            n_neurons=config.n_neurons,
            conditions=conditions,
            n_conditions=config.n_conditions,
            condition_mode="circular",
            directional_scale=1.0,
            position_scale=1.0,
            velocity_scale=1.0,
            context_scale=1.0,
            neuron_type_probabilities=circular_neuron_type_probabilities(
                config.latent_dim
            ),
            random_state=config.random_state + 1,
            return_neuron_types=True,
        )
        B[:3, :] *= config.first_coordinates_multiplier
        place_centers = np.full(config.n_neurons, np.nan)
        place_drive = np.zeros(
            (config.n_trials, config.trial_length, config.n_neurons)
        )
        neuron_metadata = None
    else:
        (
            B,
            neuron_types,
            place_centers,
            place_drive,
            neuron_metadata,
        ) = build_linear_loading_and_place_fields(
            k=config.latent_dim,
            n_neurons=config.n_neurons,
            position=state["position"][:, :, 0],
            place_fraction=config.place_fraction,
            place_width=config.place_width,
            place_scale=config.place_scale,
            first_coordinates_multiplier=config.first_coordinates_multiplier,
            random_state=config.random_state + 1,
            n_position_bins=config.n_position_bins,
            gradient_fraction=config.gradient_fraction,
            nonpreferred_direction_gain=config.nonpreferred_direction_gain,
            return_metadata=True,
        )

    rng = np.random.default_rng(config.random_state + 2)
    baseline = rng.normal(
        config.baseline_mean,
        config.baseline_std,
        size=config.n_neurons,
    )
    u = Z @ B + baseline + place_drive
    lam = config.rate_scale * drive_to_rate(u, "softplus")
    X = rate_to_spike(lam, config.dt)

    dataset, metadata = build_windows_and_labels(
        X,
        condition,
        state,
        config,
    )
    (
        train_trials,
        test_trials,
        train_mask,
        test_mask,
    ) = split_trials(metadata, condition, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fitted = fit_models(
        dataset,
        config,
        device,
        train_mask,
    )
    metrics = evaluate_models(
        Z=Z,
        metadata=metadata,
        fitted=fitted,
        train_mask=train_mask,
        test_mask=test_mask,
        config=config,
    )

    results: dict[str, object] = {
        "config": asdict(config),
        "device": str(device),
        "Z": Z,
        "condition": condition,
        "state": state,
        "B": B,
        "baseline": baseline,
        "neuron_types": neuron_types,
        "neuron_metadata": neuron_metadata,
        "place_centers": place_centers,
        "u": u,
        "lam": lam,
        "X": X,
        "metadata": metadata,
        "train_trials": train_trials,
        "test_trials": test_trials,
        "metrics": metrics,
        **fitted,
    }

    model_root = output_root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted["pca_model"], model_root / "pca.joblib")
    torch.save(
        fitted["cnn_model"].state_dict(),
        model_root / "cnn1d_state_dict.pt",
    )

    serializable_results = {
        key: value
        for key, value in results.items()
        if key not in {"pca_model", "cnn_model"}
    }
    joblib.dump(serializable_results, output_root / "results.joblib")
    with (output_root / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)
    save_experiment_figures(results, output_root)
    return results
