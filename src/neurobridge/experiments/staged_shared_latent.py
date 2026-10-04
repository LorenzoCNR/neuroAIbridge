"""Compartmental shared-latent benchmark.

The benchmark is deliberately split into cacheable stages.  Data generation,
window construction, model fitting, encoding, evaluation, and plotting write
different artifacts, so changing one stage does not silently rerun or
overwrite the others.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from neurobridge.data.dataset import TemporalWindowDataset
from neurobridge.experiments.audit_contracts import (
    check_cached_config, validate_split, write_checkpoint_provenance,
)
from neurobridge.data.sim.Lat_traj_generator import LatentTrajectoryGenerator
from neurobridge.data.sim.builders import (
    apply_temporal_lag,
    build_structured_B,
    drive_to_rate,
    rate_to_spike,
)
from neurobridge.eval.representation import (
    evaluate_latent_recovery,
    lagged_alignment_by_trial_time,
)
from neurobridge.losses.infonce import (
    cebra_infonce_loss,
    soft_contrastive_loss,
    supervised_infonce_loss,
)
from neurobridge.models.temporal_cnn import (
    CEBRAOffset10Encoder,
    TemporalCNNEncoder,
    TemporalTransformerEncoder,
)
from neurobridge.sampling.f_windows import build_windows
from neurobridge.sampling.cebra_time import CEBRATripletWindowDataset
from neurobridge.sampling.cebra_behavior import CEBRASupervisedWindowDataset
from neurobridge.train.loop import (
    encode_windows,
    profile_training_steps,
    train_steps,
    train_triplet_steps,
)


@dataclass(frozen=True)
class SharedLatentStageConfig:
    """Fixed protocol for the two-subject center-out benchmark."""

    name: str = "shared_latent_center_out"
    latent_dim: int = 3
    n_trials: int = 200
    n_conditions: int = 8
    trial_length: int = 200
    dt: float = 0.02
    n_neurons_A: int = 160
    n_neurons_B: int = 120
    lag_bins: int = 10
    cebra_time_offset: int = 10
    window_size: int = 21
    stride: int = 1
    phi: float = 0.4
    noise_scale: float = 0.05
    baseline_mean: float = 1.0
    baseline_std: float = 0.10
    # Calibrated against the rat count matrices: approximately 96--97% zero
    # bins, Fano around 1.2, and a maximum of roughly 5--10 spikes/bin.
    rate_scale: float = 1.0
    first_coordinates_multiplier: float = 3.0
    overdispersion: float = 4.0
    refractory_mean_bins: int = 1
    refractory_std_bins: float = 0.25
    train_per_condition: int = 17
    test_per_condition: int = 5
    validation_per_condition: int = 3
    train_fraction: float = 0.70
    test_fraction: float = 0.20
    validation_fraction: float = 0.10
    # Human-readable run directory. Change this label when starting a new
    # protocol; the full parameter table is written to PARAMETERS.md.
    run_label: str = "clean_rebuild_2026-09-23_4000_seed42"
    hidden_dim: int = 64
    cnn_layers: int = 3
    transformer_dim: int = 64
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_dropout: float = 0.1
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    # CEBRA counts optimizer updates rather than epochs.  The benchmark uses
    # the same fixed budget for every neural model/loss combination.
    max_iterations: int = 4000
    max_epochs: int = 100
    validation_interval: int = 400
    min_iterations: int = 800
    relative_min_delta: float = 0.001
    early_stopping_patience: int = 3
    embedding_temperature: float = 0.1
    cebra_temperature: float = 1.0
    metadata_temperature: float = 0.5
    time_weight: float = 0.5
    condition_weight: float = 0.5
    seed: int = 42
    split_seed: int | None = None
    training_seed: int | None = None
    output_root: str = "outputs/runs"

    def __post_init__(self) -> None:
        for name in ("seed", "split_seed", "training_seed"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.latent_dim not in {3, 5}:
            raise ValueError("latent_dim must be 3 or 5 for this protocol")
        if self.n_trials != self.n_conditions * (
            self.train_per_condition
            + self.test_per_condition
            + self.validation_per_condition
        ):
            raise ValueError(
                "trial counts must equal n_conditions times the three split counts"
            )
        fractions = (
            self.train_fraction,
            self.test_fraction,
            self.validation_fraction,
        )
        if any(value < 0 for value in fractions) or not np.isclose(sum(fractions), 1.0):
            raise ValueError("split fractions must be non-negative and sum to one")
        if self.trial_length < 150:
            raise ValueError("trial_length must be at least 150 bins")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.lag_bins < 0 or self.lag_bins >= self.trial_length:
            raise ValueError("lag_bins must satisfy 0 <= lag_bins < trial_length")
        if self.cebra_time_offset <= 0 or self.cebra_time_offset >= self.trial_length:
            raise ValueError("cebra_time_offset must satisfy 0 < cebra_time_offset < trial_length")
        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 3")
        if self.overdispersion < 0:
            raise ValueError("overdispersion must be non-negative")
        if self.refractory_mean_bins < 0 or self.refractory_std_bins < 0:
            raise ValueError("refractory parameters must be non-negative")
        if self.embedding_temperature <= 0 or self.cebra_temperature <= 0:
            raise ValueError("contrastive temperatures must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be strictly positive")
        if self.validation_interval < 1 or self.min_iterations < 0 or self.min_iterations > self.max_iterations:
            raise ValueError("validation_interval/min_iterations are outside the valid range")
        if not 0.0 <= self.relative_min_delta < 1.0:
            raise ValueError("relative_min_delta must lie in [0, 1)")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")

    @property
    def split_counts(self) -> dict[str, int]:
        names = ("train", "test", "validation")
        fractions = np.asarray(
            (self.train_fraction, self.test_fraction, self.validation_fraction),
            dtype=float,
        )
        raw = self.n_trials * fractions
        counts = np.floor(raw).astype(int)
        remainder = self.n_trials - int(counts.sum())
        order = np.argsort(-(raw - counts), kind="stable")
        for index in order[:remainder]:
            counts[index] += 1
        return {name: int(count) for name, count in zip(names, counts)}

    @property
    def run_dir_name(self) -> str:
        return self.run_label


def _protocol_fingerprint(config: SharedLatentStageConfig) -> str:
    """Short provenance identifier kept in the parameter note, not the path."""
    payload = asdict(config)
    payload.pop("run_label", None)
    for key in ("split_seed", "training_seed"):
        if payload[key] is None:
            payload.pop(key)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _run_root(project_root: str | Path, config: SharedLatentStageConfig) -> Path:
    return Path(project_root).resolve() / config.output_root / config.run_dir_name


def _stage_dir(project_root: str | Path, config: SharedLatentStageConfig, name: str) -> Path:
    if config.run_label == "output_2026-08-27_comparison_2000":
        raise ValueError(
            "The reference run is immutable. Read its artifacts directly or use "
            "the frozen evaluator with a separate output namespace."
        )
    cached_config = _run_root(project_root, config) / "stage01_data" / "config.json"
    if cached_config.exists():
        check_cached_config(cached_config, config)
    path = _run_root(project_root, config) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_parameters_note(project_root: str | Path, config: SharedLatentStageConfig) -> Path:
    """Write a human-readable manifest at the root of every staged run."""
    run_root = _run_root(project_root, config)
    run_root.mkdir(parents=True, exist_ok=True)
    note = run_root / "PARAMETERS.md"
    lines = [
        "# Shared-latent benchmark — run parameters",
        "",
        f"Protocol fingerprint: `{_protocol_fingerprint(config)}`",
        f"Run label: `{config.run_label}`",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]
    for key, value in _public_config(config).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Stage outputs",
            "",
            "- `stage01_data/`: shared latent process and spike counts",
            "- `stage02_windows/`: trial-safe windows and split",
            "- `stage03_models/`: one fitted model per subject/branch/loss",
            "- `stage04_embeddings/`: frozen transforms",
            "- `stage05_metrics/`: recovery and lag metrics",
            "- `stage06_figures/`: planar and spherical plots",
            "- `stage07_profiles/`: computational profiles",
        ]
    )
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return note


def _canonical_loss(model_name: str, loss_name: str) -> str:
    """Return the artifact key used for a model/loss combination."""
    if model_name == "pca":
        return "none"
    if loss_name not in {"soft", "infonce", "cebra_time", "cebra_behavior"}:
        raise ValueError("loss_name must be soft, infonce, cebra_time, or cebra_behavior")
    return loss_name


def _public_loss_name(loss_name: str) -> str:
    """Return the thesis-facing objective name used in generated outputs.

    Legacy cache identifiers remain unchanged internally for reproducibility;
    filenames, figure titles, and other user-facing artifacts use the project
    terminology instead.
    """
    return {
        "cebra_time": "time_contrastive_blocks",
        "cebra_behavior": "behavior_contrastive_blocks",
    }.get(loss_name, loss_name)


def _public_config(config: SharedLatentStageConfig) -> dict[str, object]:
    """Serialize configuration with neutral thesis-facing objective fields."""
    values = asdict(config)
    values["time_offset_bins"] = values.pop("cebra_time_offset")
    values["temporal_objective_temperature"] = values.pop("cebra_temperature")
    return values


def _loss_definition(loss_name: str) -> str:
    """Human-readable definition stored with each result artifact."""
    if loss_name == "soft":
        return "metadata-weighted contrastive loss (time plus circular condition similarity)"
    if loss_name == "infonce":
        return "supervised InfoNCE: same-direction windows are positives; other directions are negatives"
    if loss_name == "cebra_time":
        return "Time Contrastive Blocks: reference/positive windows are separated by the configured temporal offset"
    if loss_name == "cebra_behavior":
        return "Behavior Contrastive Blocks: reference/positive windows share the discrete direction label"
    return "not applicable (PCA)"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _deterministic_state(state: dict[str, np.ndarray], latent_dim: int) -> np.ndarray:
    """Reconstruct M from the interpretable state returned by the generator."""
    position = np.asarray(state["position"], dtype=float)
    progress = np.asarray(state["phase"], dtype=float)[..., None]
    columns = [position, progress]
    if latent_dim >= 5:
        velocity = np.asarray(state["velocity"], dtype=float)[..., None]
        context = np.asarray(state["context"], dtype=float)
        context = np.repeat(context[:, None, None], position.shape[1], axis=1)
        columns.extend([velocity, context])
    return np.concatenate(columns, axis=-1)


def _neuron_probabilities(latent_dim: int) -> dict[str, float]:
    if latent_dim == 3:
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


def _spike_diagnostics(X: np.ndarray, dt: float) -> dict[str, object]:
    mean_per_neuron = X.mean(axis=(0, 1))
    var_per_neuron = X.var(axis=(0, 1))
    fano = var_per_neuron / np.maximum(mean_per_neuron, 1e-12)
    return {
        "zero_fraction": float(np.mean(X == 0)),
        "mean_count_per_bin": float(X.mean()),
        "mean_rate_hz": float(X.mean() / dt),
        "median_rate_hz": float(np.median(mean_per_neuron / dt)),
        "fano_mean": float(np.mean(fano)),
        "fano_median": float(np.median(fano)),
        "q95_count": float(np.quantile(X, 0.95)),
        "q99_count": float(np.quantile(X, 0.99)),
        "q999_count": float(np.quantile(X, 0.999)),
        "max_count": int(X.max()),
        "nearly_silent_neuron_fraction": float(np.mean(mean_per_neuron < 1e-3)),
    }


def stage_generate(
    project_root: str | Path,
    config: SharedLatentStageConfig,
    *,
    force: bool = False,
) -> Path:
    """Stage 1: generate and cache the shared latent and two populations."""
    stage = _stage_dir(project_root, config, "stage01_data")
    output = stage / "shared_data.npz"
    if output.exists() and not force:
        return output

    _write_parameters_note(project_root, config)

    _seed_everything(config.seed)
    conditions = np.arange(config.n_conditions, dtype=int)
    generator = LatentTrajectoryGenerator(
        config.n_trials,
        config.trial_length,
        config.latent_dim,
        config.phi,
        conditions=conditions,
        condition_mode="circular",
        n_conditions=config.n_conditions,
        noise_scale=config.noise_scale,
        condition_type="balanced",
    )
    Z_shared, labels, state = generator.generate_latent(return_state=True)
    M = _deterministic_state(state, config.latent_dim)
    eta = Z_shared - M
    probabilities = _neuron_probabilities(config.latent_dim)

    subject_arrays: dict[str, np.ndarray] = {}
    diagnostics: dict[str, object] = {}
    for subject, n_neurons, lag in (
        ("A", config.n_neurons_A, 0),
        ("B", config.n_neurons_B, config.lag_bins),
    ):
        Z_subject = apply_temporal_lag(Z_shared, lag_bins=lag)
        valid = np.ones((config.n_trials, config.trial_length), dtype=bool)
        if lag > 0:
            valid[:, :lag] = False

        B, neuron_types = build_structured_B(
            k=config.latent_dim,
            n_neurons=n_neurons,
            conditions=conditions,
            n_conditions=config.n_conditions,
            condition_mode="circular",
            directional_scale=1.0,
            position_scale=1.0,
            velocity_scale=1.0,
            context_scale=1.0,
            neuron_type_probabilities=probabilities,
            random_state=config.seed + (1 if subject == "A" else 2),
            return_neuron_types=True,
        )
        B[:3] *= config.first_coordinates_multiplier
        rng = np.random.default_rng(config.seed + (11 if subject == "A" else 12))
        baseline = rng.normal(config.baseline_mean, config.baseline_std, n_neurons)
        u = Z_subject @ B + baseline
        lam = config.rate_scale * drive_to_rate(u, "softplus")
        X = rate_to_spike(
            lam,
            config.dt,
            overdispersion=config.overdispersion,
            refractory_mean_bins=config.refractory_mean_bins,
            refractory_std_bins=config.refractory_std_bins,
            burst_probability=0.0,
            burst_size_mean=0.0,
        )

        subject_arrays.update(
            {
                f"Z_{subject}": Z_subject,
                f"valid_{subject}": valid,
                f"B_{subject}": B,
                f"baseline_{subject}": baseline,
                f"u_{subject}": u,
                f"lam_{subject}": lam,
                f"X_{subject}": X,
                f"neuron_types_{subject}": neuron_types,
            }
        )
        diagnostics[subject] = _spike_diagnostics(X, config.dt)

    np.savez_compressed(
        output,
        M=M,
        eta=eta,
        Z_shared=Z_shared,
        labels=labels,
        phase=state["phase"],
        position=state["position"],
        velocity=state["velocity"],
        context=state["context"],
        speed_scale=state["speed_scale"],
        **subject_arrays,
    )
    _write_json(stage / "config.json", _public_config(config))
    _write_json(stage / "spike_diagnostics.json", diagnostics)
    return output


def _balanced_trial_split(config: SharedLatentStageConfig) -> dict[str, list[int]]:
    rng = np.random.default_rng(config.seed + 101 if config.split_seed is None else config.split_seed)
    labels = np.arange(config.n_trials) % config.n_conditions
    split = {"train": [], "test": [], "validation": []}
    target = config.split_counts
    train_counts = np.full(config.n_conditions, config.train_per_condition, dtype=int)
    test_counts = np.full(config.n_conditions, config.test_per_condition, dtype=int)
    validation_counts = np.full(
        config.n_conditions, config.validation_per_condition, dtype=int
    )
    # Twenty-five trials per direction cannot be split into exact 70/20/10
    # percentages per direction. Keep five test trials per direction and
    # distribute the four train/validation rounding adjustments across four
    # directions, giving exact global totals of 140/40/20.
    train_delta = target["train"] - int(train_counts.sum())
    test_delta = target["test"] - int(test_counts.sum())
    validation_delta = target["validation"] - int(validation_counts.sum())
    if test_delta != 0 or train_delta != -validation_delta:
        raise ValueError(
            "the configured per-condition counts cannot realize the requested global split"
        )
    if train_delta > 0:
        chosen = rng.permutation(config.n_conditions)[:train_delta]
        train_counts[chosen] += 1
        validation_counts[chosen] -= 1
    elif train_delta < 0:
        chosen = rng.permutation(config.n_conditions)[: -train_delta]
        train_counts[chosen] -= 1
        validation_counts[chosen] += 1
    for condition in range(config.n_conditions):
        trials = np.flatnonzero(labels == condition)
        trials = rng.permutation(trials)
        n_train = int(train_counts[condition])
        n_test = int(test_counts[condition])
        split["train"].extend(trials[:n_train].tolist())
        start = n_train
        split["test"].extend(
            trials[start : start + n_test].tolist()
        )
        start += n_test
        split["validation"].extend(trials[start:].tolist())
    for values in split.values():
        values.sort()
    validate_split(split, np.arange(config.n_trials), config.split_counts)
    return split


def _window_subject(
    X: np.ndarray,
    labels: np.ndarray,
    progress: np.ndarray,
    valid: np.ndarray,
    config: SharedLatentStageConfig,
) -> dict[str, np.ndarray]:
    flat = X.reshape(config.n_trials * config.trial_length, X.shape[-1])
    windows, time_id, global_time_id, trial_id, labels_windows = build_windows(
        flat,
        config.window_size,
        config.stride,
        labels=labels,
        trial_len=config.trial_length,
        time_mode="absolute",
        padding="center",
        pad_value=0.0,
    )
    center_progress = progress[trial_id, time_id.astype(int)]
    radius = config.window_size // 2
    strict_window_valid = np.zeros((config.n_trials, config.trial_length), dtype=bool)
    for center in range(radius, config.trial_length - radius):
        strict_window_valid[:, center] = valid[
            :, center - radius:center + radius + 1
        ].all(axis=1)
    center_valid = strict_window_valid[trial_id, time_id.astype(int)]
    return {
        "X_windows": windows,
        "time_id": time_id.astype(np.int64),
        "global_time_id": global_time_id.astype(np.int64),
        "trial_id": trial_id.astype(np.int64),
        "labels": np.asarray(labels_windows, dtype=np.int64),
        "progress": center_progress.astype(np.float32),
        "lag_valid": center_valid.astype(bool),
    }


def stage_windows(
    project_root: str | Path,
    config: SharedLatentStageConfig,
    *,
    force: bool = False,
) -> Path:
    """Stage 2: build trial-safe windows and cache a single split."""
    stage = _stage_dir(project_root, config, "stage02_windows")
    output = stage / "windows_A.npz"
    if (
        output.exists()
        and (stage / "windows_B.npz").exists()
        and (stage / "split.json").exists()
        and not force
    ):
        check_cached_config(stage / "config.json", config)
        cached_split = json.loads((stage / "split.json").read_text(encoding="utf-8"))
        validate_split(cached_split, np.arange(config.n_trials), config.split_counts)
        return output

    data_path = stage_generate(project_root, config, force=force)
    data = np.load(data_path, allow_pickle=False)
    labels = data["labels"]
    progress = data["M"][..., 2]
    for subject in ("A", "B"):
        windows = _window_subject(
            data[f"X_{subject}"],
            labels,
            progress,
            data[f"valid_{subject}"],
            config,
        )
        np.savez_compressed(stage / f"windows_{subject}.npz", **windows)

    split = _balanced_trial_split(config)
    _write_json(stage / "split.json", split)
    _write_json(stage / "config.json", _public_config(config))
    return output


def _load_window_dataset(path: Path) -> TemporalWindowDataset:
    values = np.load(path, allow_pickle=False)
    return TemporalWindowDataset(
        values["X_windows"],
        values["time_id"],
        values["global_time_id"],
        values["trial_id"],
        labels_windows=values["labels"],
        extra_metadata={
            "progress": values["progress"],
            "lag_valid": values["lag_valid"].astype(np.int64),
        },
    )


def _soft_metadata_similarity(batch: dict[str, torch.Tensor], config: SharedLatentStageConfig) -> torch.Tensor:
    time_distance = torch.abs(batch["time_id"][:, None] - batch["time_id"][None, :])
    time_distance = time_distance / max(config.trial_length - 1, 1)
    labels = batch["label"].long().view(-1)
    categorical = torch.abs(labels[:, None] - labels[None, :]).float()
    categorical = torch.minimum(
        categorical,
        float(config.n_conditions) - categorical,
    ) / (config.n_conditions / 2.0)
    distance = (
        config.time_weight * time_distance
        + config.condition_weight * categorical
    ) / (config.time_weight + config.condition_weight)
    return torch.exp(-distance / config.metadata_temperature)


def _make_model(model_name: str, n_features: int, config: SharedLatentStageConfig, normalize: bool) -> torch.nn.Module:
    if model_name == "cebra_offset10":
        return CEBRAOffset10Encoder(
            n_features=n_features,
            embedding_dim=config.latent_dim,
            hidden_dim=32,
            normalize=normalize,
        )
    if model_name == "cnn1d":
        return TemporalCNNEncoder(
            n_features=n_features,
            embedding_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            kernel_size=3,
            n_layers=config.cnn_layers,
            normalize=normalize,
        )
    if model_name == "transformer":
        return TemporalTransformerEncoder(
            n_features=n_features,
            embedding_dim=config.latent_dim,
            model_dim=config.transformer_dim,
            n_heads=config.transformer_heads,
            n_layers=config.transformer_layers,
            dropout=config.transformer_dropout,
            normalize=normalize,
        )
    raise ValueError("model_name must be 'cnn1d', 'transformer', or 'cebra_offset10'")


def _subset_indices(
    dataset: TemporalWindowDataset,
    trials: Iterable[int],
    valid: np.ndarray | None = None,
) -> list[int]:
    allowed = set(int(value) for value in trials)
    indices = []
    for index, trial in enumerate(dataset.trial_id.tolist()):
        if int(trial) not in allowed:
            continue
        if valid is not None and not bool(valid[index]):
            continue
        indices.append(index)
    return indices


def _evaluate_training_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_name: str,
    config: SharedLatentStageConfig,
    device: torch.device,
) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            embedding = model(batch["x"])
            if loss_name == "soft":
                loss = soft_contrastive_loss(
                    embedding,
                    _soft_metadata_similarity(batch, config),
                    temperature=config.embedding_temperature,
                )
            else:
                loss = supervised_infonce_loss(
                    embedding,
                    batch["label"],
                    temperature=config.embedding_temperature,
                )
            values.append(float(loss.detach().cpu()))
    return float(np.mean(values)) if values else float("nan")


def _evaluate_triplet_loss(
    model: torch.nn.Module,
    loader: DataLoader | None,
    temperature: float,
    device: torch.device,
) -> float:
    """Evaluate CEBRA-style InfoNCE on explicit triplet batches."""
    if loader is None:
        return float("nan")
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            reference = model(batch["reference_x"])
            positive = model(batch["positive_x"])
            negative = model(batch["negative_x"])
            loss = cebra_infonce_loss(
                reference,
                positive,
                negative,
                temperature=temperature,
            )
            batch_size = int(batch["reference_x"].shape[0])
            total += float(loss.detach().cpu()) * batch_size
            count += batch_size
    return total / count if count else float("nan")


def stage_fit(
    project_root: str | Path,
    config: SharedLatentStageConfig,
    *,
    subject: str,
    model_name: str,
    loss_name: str,
    branch: str,
    force: bool = False,
) -> Path:
    """Stage 3: fit one model/loss/subject/branch combination."""
    if subject not in {"A", "B"}:
        raise ValueError("subject must be A or B")
    if branch not in {"full_sample", "held_out"}:
        raise ValueError("branch must be full_sample or held_out")
    loss_name = _canonical_loss(model_name, loss_name)
    public_loss_name = _public_loss_name(loss_name)

    stage_windows(project_root, config)
    windows_path = _run_root(project_root, config) / "stage02_windows" / f"windows_{subject}.npz"
    dataset = _load_window_dataset(windows_path)
    split = json.loads((_run_root(project_root, config) / "stage02_windows" / "split.json").read_text(encoding="utf-8"))
    validate_split(split, dataset.trial_id.numpy(), config.split_counts)
    valid_windows = None
    if subject == "B":
        # The first lagged bins are padding, not observations of the shared
        # process.  They remain cached for alignment/plotting, but cannot be
        # used to fit a model.
        valid_windows = dataset.extra_metadata["lag_valid"].numpy().astype(bool)
    if branch == "full_sample":
        train_indices = list(range(len(dataset)))
        if valid_windows is not None:
            train_indices = [index for index in train_indices if valid_windows[index]]
        validation_indices = []
    else:
        train_indices = _subset_indices(dataset, split["train"], valid_windows)
        validation_indices = _subset_indices(dataset, split["validation"], valid_windows)

    max_updates = config.max_iterations
    if branch == "full_sample" and model_name != "pca":
        heldout_compute = (
            _stage_dir(project_root, config, "stage03_models") / "held_out"
            / f"{subject}_{model_name}_{public_loss_name}" / "compute.json"
        )
        if not heldout_compute.is_file():
            raise FileNotFoundError(
                "full-sample training requires the corresponding held-out fit first: "
                f"{heldout_compute}"
            )
        max_updates = int(json.loads(heldout_compute.read_text(encoding="utf-8"))["best_validation_step"])
        if max_updates < 1:
            raise RuntimeError("held-out checkpoint did not record a valid N_best")

    model_dir = _stage_dir(project_root, config, "stage03_models") / branch / f"{subject}_{model_name}_{public_loss_name}"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "model.pt"
    history_path = model_dir / "training_history.csv"

    if model_name == "pca":
        values = np.load(windows_path, allow_pickle=False)
        flattened = values["X_windows"].reshape(len(values["X_windows"]), -1)
        pca_path = model_dir / "pca.joblib"
        if pca_path.exists() and (model_dir / "compute.json").exists() and not force:
            check_cached_config(model_dir / "config.json", config)
            return pca_path
        start = time.perf_counter()
        pca = PCA(n_components=config.latent_dim, random_state=config.seed)
        pca.fit(flattened[train_indices])
        joblib.dump(pca, pca_path)
        elapsed = time.perf_counter() - start
        _write_json(
            model_dir / "compute.json",
            {
                "fit_seconds": elapsed,
                "n_parameters": int(flattened.shape[1] * config.latent_dim),
                "n_fit_windows": len(train_indices),
                "model": "pca",
                "loss": "none",
                "loss_definition": _loss_definition("none"),
            },
        )
        _write_json(model_dir / "config.json", {**_public_config(config), "subject": subject, "model": model_name, "loss": "none", "branch": branch})
        write_checkpoint_provenance(project_root, config, pca_path, windows_path, windows_path.with_name("split.json"))
        return pca_path

    if checkpoint.exists() and history_path.exists() and not force:
        check_cached_config(model_dir / "config.json", config)
        return checkpoint

    training_seed = config.seed if config.training_seed is None else config.training_seed
    _seed_everything(training_seed + (0 if subject == "A" else 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(model_name, dataset.X_windows.shape[-1], config, normalize=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader_generator = torch.Generator().manual_seed(
        training_seed + (0 if subject == "A" else 1)
    )
    if loss_name in {"cebra_time", "cebra_behavior"}:
        if loss_name == "cebra_time":
            triplet_dataset = CEBRATripletWindowDataset(
                dataset,
                train_indices,
                offset=config.cebra_time_offset,
                valid_mask=valid_windows,
            )
            validation_triplet_dataset = (
                CEBRATripletWindowDataset(
                    dataset,
                    validation_indices,
                    offset=config.cebra_time_offset,
                    valid_mask=valid_windows,
                )
                if validation_indices
                else None
            )
        else:
            triplet_dataset = CEBRASupervisedWindowDataset(
                dataset,
                train_indices,
                valid_mask=valid_windows,
            )
            validation_triplet_dataset = (
                CEBRASupervisedWindowDataset(
                    dataset,
                    validation_indices,
                    valid_mask=valid_windows,
                )
                if validation_indices
                else None
            )
        train_loader = DataLoader(
            triplet_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            generator=loader_generator,
        )
        validation_loader = (
            DataLoader(validation_triplet_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
            if validation_triplet_dataset is not None
            else None
        )
    else:
        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            generator=loader_generator,
        )
        validation_loader = DataLoader(Subset(dataset, validation_indices), batch_size=config.batch_size, shuffle=False, drop_last=True) if validation_indices else None
    if len(train_loader) == 0:
        raise RuntimeError(f"{branch} training has fewer than one full B={config.batch_size} batch")
    history: list[dict[str, object]] = []
    best_state = None
    stopping_state = None
    best_value = float("inf")
    best_step: int | None = None
    meaningful_reference: float | None = None
    stale_checks = 0
    optimizer_steps = 0
    block = 0
    stop_reason = "max_updates_reached"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    while optimizer_steps < max_updates:
        block += 1
        steps_to_check = config.validation_interval - (optimizer_steps % config.validation_interval)
        block_steps = min(steps_to_check, max_updates - optimizer_steps)
        if loss_name in {"cebra_time", "cebra_behavior"}:
            train_loss = train_triplet_steps(
                model, train_loader, optimizer,
                lambda reference, positive, negative: cebra_infonce_loss(
                    reference, positive, negative, temperature=config.cebra_temperature,
                ),
                steps=block_steps, device=device,
                resample=lambda: triplet_dataset.resample(generator=loader_generator),
            )
        elif loss_name == "soft":
            train_loss = train_steps(
                model, train_loader, optimizer,
                lambda embedding, similarity: soft_contrastive_loss(
                    embedding, similarity, temperature=config.embedding_temperature,
                ),
                steps=block_steps, device=device,
                similarity_builder=lambda batch: _soft_metadata_similarity(batch, config),
            )
        else:
            train_loss = train_steps(
                model, train_loader, optimizer,
                lambda embedding, labels: supervised_infonce_loss(
                    embedding, labels, temperature=config.embedding_temperature,
                ),
                steps=block_steps, device=device,
            )
        optimizer_steps += block_steps
        validation_loss = (
            _evaluate_triplet_loss(model, validation_loader, config.cebra_temperature, device)
            if validation_loader is not None and loss_name in {"cebra_time", "cebra_behavior"}
            else _evaluate_training_loss(model, validation_loader, loss_name, config, device)
            if validation_loader is not None
            else float("nan")
        )
        history.append({
            "epoch": block,
            "iteration": optimizer_steps,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        })
        if branch == "held_out":
            if validation_loss < best_value:
                best_value = validation_loss
                best_step = optimizer_steps
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if optimizer_steps >= config.min_iterations:
                if meaningful_reference is None:
                    meaningful_reference = validation_loss
                    stale_checks = 0
                else:
                    relative_gain = (meaningful_reference - validation_loss) / max(abs(meaningful_reference), 1e-12)
                    if relative_gain >= config.relative_min_delta:
                        meaningful_reference = validation_loss
                        stale_checks = 0
                    else:
                        stale_checks += 1
            if stale_checks >= config.early_stopping_patience and optimizer_steps >= config.min_iterations and optimizer_steps < max_updates:
                stop_reason = "patience_exhausted"
                stopping_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                break
        if optimizer_steps == max_updates:
            stop_reason = "max_updates_reached" if branch == "held_out" else "fixed_N_best_full_sample"
        stopping_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"[{config.run_label}] {branch} {subject}/{model_name}/{public_loss_name}: "
            f"{optimizer_steps}/{max_updates} updates; train={train_loss:.5f}; "
            f"validation={validation_loss:.5f}; reason={stop_reason}",
            flush=True,
        )

    if branch == "held_out" and best_state is None:
        raise RuntimeError("held-out training produced no validation-selected checkpoint")
    if stopping_state is None:
        stopping_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    selected_state = best_state if branch == "held_out" else stopping_state
    model.load_state_dict(selected_state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    final_checkpoint = {
        "state_dict": selected_state,
        "model_name": model_name,
        "loss_name": loss_name,
        "device": str(device),
        "branch": branch,
        "selected_validation_step": best_step,
        "full_sample_updates": max_updates if branch == "full_sample" else None,
    }
    torch.save(final_checkpoint, checkpoint)
    if branch == "held_out":
        torch.save(final_checkpoint, model_dir / "best_validation_checkpoint.pt")
    torch.save({
        **final_checkpoint,
        "state_dict": stopping_state,
        "optimizer_steps": optimizer_steps,
        "stop_reason": stop_reason,
    }, model_dir / "stopping_checkpoint.pt")
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        columns = ["epoch", "iteration", "train_loss", "validation_loss"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    _write_json(model_dir / "compute.json", {
        "training_seconds": elapsed,
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer_steps": optimizer_steps,
        "max_iterations": max_updates,
        "best_validation_step": best_step if branch == "held_out" else None,
        "best_validation_loss": best_value if branch == "held_out" else None,
        "stopping_step": optimizer_steps,
        "stopping_reason": stop_reason,
        "checkpoints": len(history),
        "mean_seconds_per_step": elapsed / max(optimizer_steps, 1),
        "seconds_per_100_updates": elapsed / max(optimizer_steps, 1) * 100,
        "updates_per_second": optimizer_steps / max(elapsed, 1e-12),
        "n_fit_windows": len(train_indices),
        "training_windows_per_update": config.batch_size * (3 if loss_name in {"cebra_time", "cebra_behavior"} else 1),
        "training_windows_per_second": optimizer_steps * config.batch_size * (3 if loss_name in {"cebra_time", "cebra_behavior"} else 1) / max(elapsed, 1e-12),
        "batch_size": config.batch_size,
        "train_tail_queries_dropped_per_pass": len(train_indices) % config.batch_size,
        "loss": public_loss_name,
        "loss_definition": _loss_definition(loss_name),
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "pytorch": torch.__version__,
        "cuda_version": torch.version.cuda,
    })
    _write_json(model_dir / "config.json", {
        **_public_config(config), "subject": subject, "model": model_name,
        "loss": public_loss_name, "branch": branch,
        "optimizer_updates": optimizer_steps,
        "best_validation_step": best_step if branch == "held_out" else None,
        "N_best": max_updates if branch == "full_sample" else None,
        "stop_reason": stop_reason,
    })
    write_checkpoint_provenance(project_root, config, checkpoint, windows_path, windows_path.with_name("split.json"))
    return checkpoint


def stage_transform(
    project_root: str | Path,
    config: SharedLatentStageConfig,
    *,
    subject: str,
    model_name: str,
    loss_name: str,
    branch: str,
    force: bool = False,
) -> Path:
    """Stage 4: transform every cached window with a frozen fit."""
    loss_name = _canonical_loss(model_name, loss_name)
    public_loss_name = _public_loss_name(loss_name)
    stage_fit(
        project_root,
        config,
        subject=subject,
        model_name=model_name,
        loss_name=loss_name,
        branch=branch,
        force=force,
    )
    output_dir = _stage_dir(project_root, config, "stage04_embeddings") / branch
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{subject}_{model_name}_{public_loss_name}.npz"
    if output.exists() and not force:
        return output
    run_root = _run_root(project_root, config)
    windows_path = run_root / "stage02_windows" / f"windows_{subject}.npz"
    dataset = _load_window_dataset(windows_path)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    model_dir = run_root / "stage03_models" / branch / f"{subject}_{model_name}_{public_loss_name}"
    if model_name == "pca":
        start_time = time.perf_counter()
        pca = joblib.load(model_dir / "pca.joblib")
        raw = pca.transform(dataset.X_windows.numpy().reshape(len(dataset), -1))
        unit = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-8)
        encoding_seconds = time.perf_counter() - start_time
    else:
        start_time = time.perf_counter()
        checkpoint = torch.load(model_dir / "model.pt", map_location="cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        raw_model = _make_model(model_name, dataset.X_windows.shape[-1], config, normalize=False)
        raw_model.load_state_dict(checkpoint["state_dict"])
        unit_model = _make_model(model_name, dataset.X_windows.shape[-1], config, normalize=True)
        unit_model.load_state_dict(checkpoint["state_dict"])
        raw, _ = encode_windows(raw_model, loader, device=device)
        unit, _ = encode_windows(unit_model, loader, device=device)
        raw = raw.numpy()
        unit = unit.numpy()
        encoding_seconds = time.perf_counter() - start_time
    values = np.load(windows_path, allow_pickle=False)
    np.savez_compressed(
        output,
        embedding_raw=raw,
        embedding_unit=unit,
        trial_id=values["trial_id"],
        time_id=values["time_id"],
        global_time_id=values["global_time_id"],
        labels=values["labels"],
        progress=values["progress"],
        lag_valid=values["lag_valid"],
    )
    _write_json(
        output.with_name(output.stem + "_compute.json"),
        {
            "encoding_seconds": encoding_seconds,
            "n_windows": len(values["X_windows"]),
            "seconds_per_window": encoding_seconds / max(len(values["X_windows"]), 1),
            "model": model_name,
            "loss": public_loss_name,
            "loss_definition": _loss_definition(loss_name),
            "branch": branch,
            "subject": subject,
        },
    )
    return output


def stage_encode(
    project_root: str | Path,
    config: SharedLatentStageConfig,
    **kwargs,
) -> Path:
    """Backward-compatible alias for :func:`stage_transform`.

    The public terminology is now ``transform`` because PCA uses
    ``PCA.transform`` and neural models are applied in inference mode.
    """
    return stage_transform(project_root, config, **kwargs)


def stage_evaluate(
    project_root: str | Path,
    config: SharedLatentStageConfig,
    *,
    model_name: str,
    loss_name: str,
    branch: str,
    force: bool = False,
) -> Path:
    """Stage 5: evaluate both subjects and the cross-subject lag."""
    loss_name = _canonical_loss(model_name, loss_name)
    public_loss_name = _public_loss_name(loss_name)
    run_root = _run_root(project_root, config)
    stage = _stage_dir(project_root, config, "stage05_metrics") / branch
    stage.mkdir(parents=True, exist_ok=True)
    metrics_path = stage / f"{model_name}_{public_loss_name}.csv"
    if metrics_path.exists() and not force:
        return metrics_path
    data = np.load(run_root / "stage01_data" / "shared_data.npz", allow_pickle=False)
    split = json.loads((run_root / "stage02_windows" / "split.json").read_text(encoding="utf-8"))
    test_trials = set(split["test"])
    rows: list[dict[str, object]] = []
    embeddings: dict[str, dict[str, np.ndarray]] = {}
    for subject in ("A", "B"):
        path = stage_transform(
            project_root,
            config,
            subject=subject,
            model_name=model_name,
            loss_name=loss_name,
            branch=branch,
            # Evaluation consumes the frozen transform artifact.  A forced
            # metric recomputation must not encode every window a second time.
            force=False,
        )
        encoded = np.load(path, allow_pickle=False)
        embeddings[subject] = {key: encoded[key] for key in encoded.files}
        values = embeddings[subject]
        if branch == "full_sample":
            mask = np.ones(len(values["trial_id"]), dtype=bool)
        else:
            mask = np.isin(values["trial_id"], list(test_trials))
        if subject == "B":
            mask &= values["lag_valid"].astype(bool)
        trial_indices = values["trial_id"][mask]
        time_indices = values["time_id"][mask]
        embedding = values["embedding_unit"][mask]

        # Report recovery of the deterministic task geometry, the stochastic
        # AR(1) component, and their sum separately.  For B the two components
        # are shifted by the same controlled lag as its neural observations.
        if subject == "B":
            deterministic = apply_temporal_lag(data["M"], config.lag_bins)
            stochastic = apply_temporal_lag(data["eta"], config.lag_bins)
        else:
            deterministic = data["M"]
            stochastic = data["eta"]
        references = {
            "M": deterministic,
            "eta": stochastic,
            "Z": data[f"Z_{subject}"],
        }
        for reference_name, reference_array in references.items():
            latent = reference_array[trial_indices, time_indices]
            recovered = evaluate_latent_recovery(embedding, latent)
            for metric, value in recovered.items():
                rows.append(
                    {
                        "branch": branch,
                        "subject": subject,
                        "model": model_name,
                        "loss": public_loss_name,
                        "reference": reference_name,
                        "metric": metric,
                        "value": value,
                    }
                )

    a = embeddings["A"]
    b = embeddings["B"]
    if branch == "full_sample":
        a_mask = a["lag_valid"].astype(bool)
        b_mask = b["lag_valid"].astype(bool)
    else:
        a_mask = np.isin(a["trial_id"], list(test_trials)) & a["lag_valid"].astype(bool)
        b_mask = np.isin(b["trial_id"], list(test_trials)) & b["lag_valid"].astype(bool)
    lag_range = tuple(range(-20, 21))
    best_lag, lag_scores, aligned_pairs = lagged_alignment_by_trial_time(
        a["embedding_unit"][a_mask],
        b["embedding_unit"][b_mask],
        a["trial_id"][a_mask],
        a["time_id"][a_mask],
        b["trial_id"][b_mask],
        b["time_id"][b_mask],
        lag_range,
        common_support=True,
    )
    rows.append({"branch": branch, "subject": "A_B", "model": model_name, "loss": public_loss_name, "metric": "estimated_lag_bins", "value": best_lag})
    lag_path = stage / f"{model_name}_{public_loss_name}_lag.csv"
    pair_counts = {int(lag): int(len(aligned_pairs[int(lag)][0])) for lag in lag_range}
    if len(set(pair_counts.values())) != 1:
        raise RuntimeError("strict lag scan returned unequal comparison counts")
    support_coordinates = []
    b_coordinates = {
        (int(trial), int(time))
        for trial, time, keep in zip(b["trial_id"], b["time_id"], b_mask)
        if keep
    }
    for trial, time, keep in zip(a["trial_id"], a["time_id"], a_mask):
        if keep and all((int(trial), int(time) + lag) in b_coordinates for lag in lag_range):
            support_coordinates.append([int(trial), int(time)])
    _write_json(
        stage / f"{model_name}_{public_loss_name}_lag_support.json",
        {
            "branch": branch,
            "support_definition": "intersection of matched trial/time pairs valid for every candidate lag and every full temporal window",
            "lag_range_bins": list(lag_range),
            "n_comparisons_per_lag": pair_counts,
            "common_reference_trial_time_indices": support_coordinates,
            "n_common_reference_indices": len(support_coordinates),
        },
    )
    with lag_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lag_bins", "procrustes_r2", "n_comparisons"])
        writer.writerows([[lag, score, pair_counts[int(lag)]] for lag, score in sorted(lag_scores.items())])
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "branch",
                "subject",
                "model",
                "loss",
                "reference",
                "metric",
                "value",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    return metrics_path


def stage_plot(
    project_root: str | Path,
    config: SharedLatentStageConfig,
    *,
    model_name: str,
    loss_name: str,
    branch: str,
    force: bool = False,
) -> Path:
    """Stage 6: produce figures only from saved embeddings/metrics."""
    loss_name = _canonical_loss(model_name, loss_name)
    public_loss_name = _public_loss_name(loss_name)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    stage = _stage_dir(project_root, config, "stage06_figures") / branch
    stage.mkdir(parents=True, exist_ok=True)
    output = stage / f"{model_name}_{public_loss_name}_embeddings.png"
    sphere_outputs = [
        stage / f"{subject}_{model_name}_{public_loss_name}_sphere.html"
        for subject in ("A", "B")
    ]
    planar_outputs = [
        stage / f"{subject}_{model_name}_{public_loss_name}_trajectories_2d.html"
        for subject in ("A", "B")
    ]
    plot_manifest = stage / "plot_manifest.json"
    plot_version = 3
    try:
        cached_plot_version = json.loads(plot_manifest.read_text(encoding="utf-8")).get("version")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        cached_plot_version = None
    if (
        output.exists()
        and all(path.exists() for path in sphere_outputs + planar_outputs)
        and cached_plot_version == plot_version
        and not force
    ):
        return output
    run_root = _run_root(project_root, config)
    data = np.load(run_root / "stage01_data" / "shared_data.npz", allow_pickle=False)
    # Sample the continuous HSV map at i/n.  ``get_cmap(name, n)`` includes
    # both endpoints, making conditions 0 and n-1 share the same red.
    cmap = plt.get_cmap("hsv")
    norm = plt.Normalize(vmin=0, vmax=max(config.n_conditions - 1, 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(left=0.06, right=0.80, bottom=0.12, top=0.88, wspace=0.25)
    from neurobridge.viz import (
        plot_condition_trajectories_2d,
        plot_direction_averaged_embedding,
    )

    for axis, subject in zip(axes, ("A", "B")):
        path = stage_transform(
            project_root,
            config,
            subject=subject,
            model_name=model_name,
            loss_name=loss_name,
            branch=branch,
            # Regenerating a figure must reuse the frozen embedding.  The
            # figure's ``force`` flag must never propagate upstream to fit.
            force=False,
        )
        values = np.load(path, allow_pickle=False)
        mask = values["lag_valid"].astype(bool)
        if branch == "held_out":
            split = json.loads((run_root / "stage02_windows" / "split.json").read_text(encoding="utf-8"))
            mask &= np.isin(values["trial_id"], split["test"])
        embedding = values["embedding_unit"][mask]
        labels = values["labels"][mask]
        # The generator stores zero-based labels; figures use the human-facing
        # convention condition 1--8 without changing training/evaluation data.
        labels_for_plot = labels.astype(int) + 1
        time_values = values["time_id"][mask]
        axis.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            s=1,
            alpha=0.08,
            cmap=cmap,
            norm=norm,
        )
        axis.set_title(f"Subject {subject}: {branch}")
        axis.set_xlabel("embedding 1")
        axis.set_ylabel("embedding 2")
        axis.set_aspect("equal", adjustable="box")

        # Overlay condition-averaged trajectories and explicit start/end
        # markers on the faint window cloud.  This is the planar analogue of
        # the spherical CEBRA-style figure below.
        for condition in range(config.n_conditions):
            condition_mask = labels == condition
            times = np.unique(time_values[condition_mask])
            trajectory = np.asarray([
                embedding[(labels == condition) & (time_values == time)].mean(axis=0)
                for time in times
            ])
            if len(trajectory) < 2:
                continue
            color = cmap(condition / max(config.n_conditions, 1))
            axis.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=color,
                linewidth=1.5,
                alpha=0.9,
            )
            axis.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                color="maroon",
                marker="o",
                s=22,
                zorder=3,
            )
            axis.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                color="black",
                marker="x",
                s=28,
                zorder=3,
            )

        plot_condition_trajectories_2d(
            embedding=embedding,
            labels=labels_for_plot,
            trial_id=values["trial_id"][mask],
            time_id=values["time_id"][mask],
            output_folder=stage,
            name=planar_outputs[0 if subject == "A" else 1].name,
            title=f"Subject {subject}: {branch} condition-averaged trajectories",
            dims=(0, 1),
            axis_labels=("Embedding 1", "Embedding 2"),
            show=False,
        )

        # Preserve the established spherical-trajectory style, but reconstruct
        # each condition by explicit trial/time coordinates and the saved
        # validity mask (rather than reshaping by an assumed trial length).
        plot_direction_averaged_embedding(
            values["embedding_unit"],
            values["labels"].astype(int) + 1,
            original_label_order=np.sort(np.unique(labels_for_plot)),
            c_s="maroon",
            output_folder=stage,
            name=sphere_outputs[0 if subject == "A" else 1].name,
            quiescent_label=None,
            show=False,
            trial_id=values["trial_id"],
            time_id=values["time_id"],
            valid_mask=mask,
        )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=cmap(condition / max(config.n_conditions, 1)),
            markeredgecolor="none",
            markersize=6,
            label=f"condition {condition + 1}",
        )
        for condition in range(config.n_conditions)
    ]
    legend_handles.extend([
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="maroon",
            markeredgecolor="maroon",
            markersize=7,
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="",
            color="black",
            markersize=7,
            label="End",
        ),
    ])
    fig.legend(
        handles=legend_handles,
        title="Task condition",
        loc="center left",
        bbox_to_anchor=(0.82, 0.5),
        frameon=True,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)
    _write_json(
        plot_manifest,
        {
            "version": plot_version,
            "condition_labels_in_figures": "1-based",
            "start_marker": "maroon circle",
            "end_marker": "black x",
        },
    )
    return output


def stage_plot_raw(
    project_root: str | Path,
    config: SharedLatentStageConfig,
    *,
    model_name: str,
    loss_name: str,
    branch: str,
    force: bool = False,
) -> Path:
    """Stage 6b: plot saved raw embeddings without unit-sphere normalization.

    This stage only reads ``stage04_embeddings``.  It never fits a model or
    recomputes metrics, and writes explicit ``*_raw`` filenames alongside the
    established normalized figures.
    """
    loss_name = _canonical_loss(model_name, loss_name)
    public_loss_name = _public_loss_name(loss_name)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from neurobridge.viz import (
        plot_condition_trajectories_2d,
        plot_direction_averaged_embedding_raw,
    )

    stage = _stage_dir(project_root, config, "stage06_figures") / branch
    stage.mkdir(parents=True, exist_ok=True)
    output = stage / f"{model_name}_{public_loss_name}_embeddings_raw.png"
    raw_3d_outputs = [
        stage / f"{subject}_{model_name}_{public_loss_name}_trajectories_raw.html"
        for subject in ("A", "B")
    ]
    raw_2d_outputs = [
        stage / f"{subject}_{model_name}_{public_loss_name}_trajectories_2d_raw.html"
        for subject in ("A", "B")
    ]
    if output.exists() and all(path.exists() for path in raw_3d_outputs + raw_2d_outputs) and not force:
        return output

    run_root = _run_root(project_root, config)
    data = np.load(run_root / "stage01_data" / "shared_data.npz", allow_pickle=False)
    cmap = plt.get_cmap("hsv")
    norm = plt.Normalize(vmin=0, vmax=max(config.n_conditions - 1, 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(left=0.06, right=0.80, bottom=0.12, top=0.88, wspace=0.25)

    for axis, subject in zip(axes, ("A", "B")):
        # Raw plotting is a read-only post-processing stage: never trigger a
        # transform (and therefore never trigger a fit) implicitly.
        path = (
            _stage_dir(project_root, config, "stage04_embeddings")
            / branch
            / f"{subject}_{model_name}_{public_loss_name}.npz"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Missing cached embedding for raw plot: {path}. "
                "Run stage_transform explicitly first."
            )
        values = np.load(path, allow_pickle=False)
        mask = values["lag_valid"].astype(bool)
        if branch == "held_out":
            split = json.loads((run_root / "stage02_windows" / "split.json").read_text(encoding="utf-8"))
            mask &= np.isin(values["trial_id"], split["test"])
        embedding = values["embedding_raw"][mask]
        labels = values["labels"][mask]
        labels_for_plot = labels.astype(int) + 1
        time_values = values["time_id"][mask]
        if embedding.shape[1] < 2:
            raise ValueError("raw embedding must have at least two dimensions")
        axis.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=1, alpha=0.08, cmap=cmap, norm=norm)
        axis.set_title(f"Subject {subject}: {branch} (raw embedding)")
        axis.set_xlabel("embedding 1")
        axis.set_ylabel("embedding 2")
        axis.set_aspect("equal", adjustable="box")
        for condition in range(config.n_conditions):
            cmask = labels == condition
            times = np.unique(time_values[cmask])
            trajectory = np.asarray([
                embedding[(labels == condition) & (time_values == time)].mean(axis=0)
                for time in times
            ])
            if len(trajectory) < 2:
                continue
            color = cmap(condition / max(config.n_conditions, 1))
            axis.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=1.5, alpha=0.9)
            axis.scatter(*trajectory[0, :2], color="maroon", marker="o", s=22, zorder=3)
            axis.scatter(*trajectory[-1, :2], color="black", marker="x", s=28, zorder=3)

        plot_condition_trajectories_2d(
            embedding=embedding, labels=labels_for_plot,
            trial_id=values["trial_id"][mask], time_id=time_values,
            output_folder=stage, name=raw_2d_outputs[0 if subject == "A" else 1].name,
            title=f"Subject {subject}: {branch} condition-averaged raw trajectories",
            dims=(0, 1), axis_labels=("Raw embedding 1", "Raw embedding 2"), show=False,
        )
        plot_direction_averaged_embedding_raw(
            values["embedding_raw"], values["labels"].astype(int) + 1,
            original_label_order=np.sort(np.unique(labels_for_plot)), c_s="maroon",
            output_folder=stage, name=raw_3d_outputs[0 if subject == "A" else 1].name,
            quiescent_label=None, show=False,
            trial_id=values["trial_id"], time_id=values["time_id"],
            valid_mask=mask,
        )

    legend_handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=cmap(i / max(config.n_conditions, 1)), markeredgecolor="none", markersize=6, label=f"condition {i + 1}") for i in range(config.n_conditions)]
    legend_handles.extend([
        Line2D([0], [0], marker="o", linestyle="", color="maroon", markersize=7, label="Start"),
        Line2D([0], [0], marker="x", linestyle="", color="black", markersize=7, label="End"),
    ])
    fig.legend(handles=legend_handles, title="Task condition", loc="center left", bbox_to_anchor=(0.82, 0.5), frameon=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def stage_plot_ground_truth_latent(
    project_root: str | Path,
    config: SharedLatentStageConfig,
    *,
    branch: str,
    force: bool = False,
) -> tuple[Path, Path]:
    """Plot the simulated shared latent as a visual reference, not a metric.

    The reference is the original ``Z_shared`` before the imposed A/B lag.
    Rows use the common valid trial/time support of the two cached populations;
    held-out plots are further restricted to the test trials.  Both raw
    coordinates and the established unit-sphere view are saved.
    """
    if branch not in {"held_out", "full_sample"}:
        raise ValueError("branch must be 'held_out' or 'full_sample'")

    stage = _stage_dir(project_root, config, "stage06_figures") / branch
    stage.mkdir(parents=True, exist_ok=True)
    outputs = (
        stage / "Ground-truth shared latent Z pre-shift, unit sphere.html",
        stage / "Ground-truth shared latent Z pre-shift, raw.html",
    )
    manifest_path = stage / "ground_truth_Z_shared_manifest.json"
    if all(path.exists() for path in outputs) and manifest_path.exists() and not force:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        if manifest.get("version") == 1 and manifest.get("branch") == branch:
            return outputs

    run_root = _run_root(project_root, config)
    data_path = run_root / "stage01_data" / "shared_data.npz"
    windows_root = run_root / "stage02_windows"
    data = np.load(data_path, allow_pickle=False)
    windows_a = np.load(windows_root / "windows_A.npz", allow_pickle=False)
    windows_b = np.load(windows_root / "windows_B.npz", allow_pickle=False)

    Z_shared = np.asarray(data["Z_shared"])
    if Z_shared.ndim != 3 or Z_shared.shape[-1] != 3:
        raise ValueError("Z_shared must have shape (trials, time, 3)")
    if Z_shared.shape[:2] != (config.n_trials, config.trial_length):
        raise ValueError("Z_shared shape does not match the cached run configuration")

    for key in ("trial_id", "time_id", "labels", "lag_valid"):
        if key not in windows_a.files or key not in windows_b.files:
            raise ValueError(f"cached windows are missing required metadata: {key}")
        if windows_a[key].shape != windows_b[key].shape:
            raise ValueError(f"A/B cached metadata cardinality differs for {key}")
    for key in ("trial_id", "time_id", "labels"):
        if not np.array_equal(windows_a[key], windows_b[key]):
            raise ValueError(f"A/B cached metadata do not align for {key}")

    trial_id = np.asarray(windows_a["trial_id"], dtype=int)
    time_id = np.asarray(windows_a["time_id"], dtype=int)
    labels = np.asarray(windows_a["labels"], dtype=int)
    valid = windows_a["lag_valid"].astype(bool) & windows_b["lag_valid"].astype(bool)
    if not (trial_id.size == time_id.size == labels.size == valid.size):
        raise ValueError("cached trial/time/label/validity arrays have different cardinalities")
    if np.any((trial_id < 0) | (trial_id >= Z_shared.shape[0])):
        raise ValueError("cached trial_id falls outside Z_shared")
    if np.any((time_id < 0) | (time_id >= Z_shared.shape[1])):
        raise ValueError("cached time_id falls outside Z_shared")

    if branch == "held_out":
        split = json.loads((windows_root / "split.json").read_text(encoding="utf-8"))
        valid &= np.isin(trial_id, np.asarray(split["test"], dtype=int))
    valid_pairs = np.column_stack((trial_id[valid], time_id[valid]))
    if valid_pairs.shape[0] == 0:
        raise ValueError(f"no common valid latent samples are available for {branch}")
    if np.unique(valid_pairs, axis=0).shape[0] != valid_pairs.shape[0]:
        raise ValueError("common valid latent support contains duplicate trial/time pairs")

    Z_rows = Z_shared[trial_id, time_id]
    if not np.isfinite(Z_rows[valid]).all():
        raise ValueError("Z_shared has non-finite values on the selected plot support")

    from neurobridge.viz import (
        plot_direction_averaged_embedding,
        plot_direction_averaged_embedding_raw,
    )

    shared_kwargs = {
        "original_label_order": np.sort(np.unique(labels[valid] + 1)),
        "c_s": "maroon",
        "output_folder": stage,
        "quiescent_label": None,
        "show": False,
        "trial_id": trial_id,
        "time_id": time_id,
        "valid_mask": valid,
    }
    plot_direction_averaged_embedding(
        Z_rows,
        labels + 1,
        name=outputs[0].name,
        **shared_kwargs,
    )
    plot_direction_averaged_embedding_raw(
        Z_rows,
        labels + 1,
        name=outputs[1].name,
        **shared_kwargs,
    )
    _write_json(
        manifest_path,
        {
            "version": 1,
            "run_label": config.run_label,
            "branch": branch,
            "source_artifact": str(data_path.relative_to(run_root)),
            "source_array": "Z_shared",
            "latent_reference": "original shared latent before imposed A/B lag",
            "is_evaluation_metric": False,
            "support": "common valid A/B trial_id/time_id pairs",
            "n_valid_samples": int(valid.sum()),
            "n_unique_trials": int(np.unique(trial_id[valid]).size),
            "n_unique_time_bins": int(np.unique(time_id[valid]).size),
            "plot_files": [path.name for path in outputs],
        },
    )
    return outputs


def stage_plot_training_history(
    project_root: str | Path,
    config: SharedLatentStageConfig,
    *,
    model_name: str,
    loss_name: str,
    branch: str,
    force: bool = False,
) -> Path:
    """Plot the saved train/validation histories for both subjects."""
    loss_name = _canonical_loss(model_name, loss_name)
    public_loss_name = _public_loss_name(loss_name)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = (
        _stage_dir(project_root, config, "stage06_figures") / branch
        / f"{model_name}_{public_loss_name}_loss.png"
    )
    if output.exists() and not force:
        return output

    run_root = _run_root(project_root, config)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, subject in zip(axes, ("A", "B")):
        history_path = (
            run_root / "stage03_models" / branch
            / f"{subject}_{model_name}_{public_loss_name}" / "training_history.csv"
        )
        with history_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        iterations = np.asarray(
            [float(row.get("iteration", index + 1)) for index, row in enumerate(rows)]
        )
        train = np.asarray([float(row["train_loss"]) for row in rows])
        validation = np.asarray([float(row["validation_loss"]) for row in rows])
        axis.plot(iterations, train, label="train")
        if branch == "held_out":
            axis.plot(iterations, validation, label="validation", linestyle="--", alpha=0.9)
            axis.fill_between(
                iterations,
                train,
                validation,
                color="tab:orange",
                alpha=0.15,
                linewidth=0,
                label="train-validation gap",
            )
            axis.set_title(f"Subject {subject}: {branch}")
        else:
            axis.set_title(f"Subject {subject}: {branch} (train only)")
        axis.set_xlabel("Optimizer steps")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Contrastive objective")
    fig.suptitle(f"Training history: {model_name} / {public_loss_name}")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def stage_plot_lag_profile(
    project_root: str | Path,
    config: SharedLatentStageConfig,
    *,
    model_name: str,
    loss_name: str,
    branch: str,
    force: bool = False,
) -> Path:
    """Plot Procrustes score across candidate inter-subject lags."""
    loss_name = _canonical_loss(model_name, loss_name)
    public_loss_name = _public_loss_name(loss_name)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = (
        _stage_dir(project_root, config, "stage06_figures") / branch
        / f"{model_name}_{public_loss_name}_lag.png"
    )
    if output.exists() and not force:
        return output

    run_root = _run_root(project_root, config)
    lag_path = run_root / "stage05_metrics" / branch / f"{model_name}_{public_loss_name}_lag.csv"
    metric_path = run_root / "stage05_metrics" / branch / f"{model_name}_{public_loss_name}.csv"
    with lag_path.open(encoding="utf-8") as handle:
        lag_rows = list(csv.DictReader(handle))
    lags = np.asarray([int(row["lag_bins"]) for row in lag_rows])
    scores = np.asarray([float(row["procrustes_r2"]) for row in lag_rows])
    estimated = None
    with metric_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] == "estimated_lag_bins":
                estimated = int(float(row["value"]))
                break

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(lags, scores, marker="o", markersize=3, label="Procrustes $R^2$")
    axis.axvline(config.lag_bins, color="black", linestyle="--", label="imposed lag")
    if estimated is not None:
        axis.axvline(estimated, color="tab:red", linestyle=":", label="estimated lag")
    axis.set(
        xlabel="Candidate inter-subject lag (bins)",
        ylabel="Procrustes $R^2$",
        title=f"Lag profile: {model_name} / {public_loss_name} ({branch})",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def stage_profile(
    project_root: str | Path,
    config: SharedLatentStageConfig,
    *,
    branches: tuple[str, ...] = ("held_out", "full_sample"),
    models: tuple[str, ...] = ("pca", "cnn1d", "transformer"),
    losses: tuple[str, ...] = ("soft", "infonce", "cebra_time", "cebra_behavior"),
    warmup_steps: int = 2,
    active_steps: int = 6,
    force: bool = False,
) -> Path:
    """Profile cached models and write a compact computational report."""
    if warmup_steps < 0 or active_steps <= 0:
        raise ValueError("warmup_steps must be non-negative and active_steps positive")
    stage = _stage_dir(project_root, config, "stage07_profiles")
    summary_path = stage / "profiles.csv"
    if summary_path.exists() and not force:
        return summary_path
    run_root = _run_root(project_root, config)
    stage_windows(project_root, config)
    split = json.loads((run_root / "stage02_windows" / "split.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for branch in branches:
        for model_name in models:
            for raw_loss in (("none",) if model_name == "pca" else losses):
                loss_name = _canonical_loss(model_name, raw_loss)
                public_loss_name = _public_loss_name(loss_name)
                for subject in ("A", "B"):
                    model_dir = run_root / "stage03_models" / branch / f"{subject}_{model_name}_{public_loss_name}"
                    compute_path = model_dir / "compute.json"
                    if not compute_path.exists():
                        continue
                    compute = json.loads(compute_path.read_text(encoding="utf-8"))
                    windows_path = run_root / "stage02_windows" / f"windows_{subject}.npz"
                    dataset = _load_window_dataset(windows_path)
                    valid = dataset.extra_metadata["lag_valid"].numpy().astype(bool) if subject == "B" else None
                    train_indices = list(range(len(dataset))) if branch == "full_sample" else _subset_indices(dataset, split["train"], valid)
                    if valid is not None:
                        train_indices = [i for i in train_indices if valid[i]]
                    record: dict[str, object] = {
                        "branch": branch, "subject": subject, "model": model_name, "loss": public_loss_name,
                        "training_seconds": float(compute.get("training_seconds", compute.get("fit_seconds", 0.0))),
                        "optimizer_steps": int(compute.get("optimizer_steps", 0)),
                        "mean_seconds_per_step": float(compute.get("mean_seconds_per_step", float("nan"))),
                        "device": str(compute.get("device", "cpu")), "peak_cuda_memory_bytes": 0,
                    }
                    if model_name == "pca":
                        pca = joblib.load(model_dir / "pca.joblib")
                        values = np.load(windows_path, allow_pickle=False)["X_windows"]
                        count = min(256, len(train_indices))
                        batch = values[np.asarray(train_indices[:count])].reshape(count, -1)
                        for _ in range(warmup_steps):
                            pca.transform(batch)
                        start = time.perf_counter()
                        for _ in range(active_steps):
                            pca.transform(batch)
                        elapsed = time.perf_counter() - start
                        record.update({"profile_mode": "pca_transform", "profile_seconds": elapsed, "profile_mean_seconds_per_step": elapsed / active_steps})
                    else:
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        _seed_everything(config.seed + (0 if subject == "A" else 1))
                        model = _make_model(model_name, dataset.X_windows.shape[-1], config, normalize=True)
                        checkpoint = torch.load(model_dir / "model.pt", map_location="cpu")
                        model.load_state_dict(checkpoint["state_dict"])
                        if loss_name in {"cebra_time", "cebra_behavior"}:
                            if loss_name == "cebra_time":
                                sampled = CEBRATripletWindowDataset(dataset, train_indices, offset=config.cebra_time_offset, valid_mask=valid)
                            else:
                                sampled = CEBRASupervisedWindowDataset(dataset, train_indices, valid_mask=valid)
                            batch = next(iter(DataLoader(sampled, batch_size=min(config.batch_size, len(sampled)), shuffle=False)))
                            objective = lambda m, b: cebra_infonce_loss(m(b["reference_x"]), m(b["positive_x"]), m(b["negative_x"]), temperature=config.cebra_temperature)
                        else:
                            batch = next(iter(DataLoader(Subset(dataset, train_indices), batch_size=min(config.batch_size, len(train_indices)), shuffle=False)))
                            if loss_name == "soft":
                                objective = lambda m, b: soft_contrastive_loss(m(b["x"]), _soft_metadata_similarity(b, config), temperature=config.embedding_temperature)
                            else:
                                objective = lambda m, b: supervised_infonce_loss(m(b["x"]), b["label"], temperature=config.embedding_temperature)
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                        profiled = profile_training_steps(model, batch, objective, device=device, warmup_steps=warmup_steps, active_steps=active_steps)
                        record.update({"profile_mode": "torch_profiler_fixed_batch", "profile_seconds": float(profiled["profile_seconds"]), "profile_mean_seconds_per_step": float(profiled["profile_mean_seconds_per_step"]), "peak_cuda_memory_bytes": int(profiled.get("peak_cuda_memory_bytes", 0)), "profile_device": profiled.get("device", str(device))})
                    output = stage / branch / f"{subject}_{model_name}_{public_loss_name}.json"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    _write_json(output, record)
                    rows.append(record)
    if rows:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    _write_json(stage / "summary.json", {"n_profiles": len(rows), "warmup_steps": warmup_steps, "active_steps": active_steps})
    return summary_path


def _append_decoder_metrics(
    rows: list[dict[str, object]],
    *,
    base: dict[str, object],
    train_embedding: np.ndarray,
    validation_embedding: np.ndarray,
    test_embedding: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    test_labels: np.ndarray,
    train_progress: np.ndarray,
    validation_progress: np.ndarray,
    test_progress: np.ndarray,
    noise_scales: tuple[float, ...],
    random_state: int,
) -> dict[str, object]:
    """Fit lightweight probes and report clean and corrupted-test scores.

    Noise is defined relative to each embedding coordinate's training-set
    standard deviation.  This probes downstream information stability; it is
    deliberately not described as robustness of the neural encoder itself.
    """
    classes = np.unique(train_labels)
    if len(classes) < 3:
        raise ValueError("condition decoding requires at least three training classes")
    condition_grid = (0.1, 1.0, 10.0)
    progress_grid = (0.1, 1.0, 10.0)

    # scikit-learn 1.9 removed ``multi_class``.  With >=3 classes and the
    # lbfgs solver, LogisticRegression optimizes the multinomial loss.  The
    # class-count guard above prevents an accidental binary formulation.
    condition_candidates = []
    for regularization in condition_grid:
        candidate = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=regularization,
                solver="lbfgs",
                max_iter=1000,
                random_state=random_state,
            ),
        )
        candidate.fit(train_embedding, train_labels)
        validation_score = balanced_accuracy_score(
            validation_labels,
            candidate.predict(validation_embedding),
        )
        condition_candidates.append((validation_score, candidate, regularization))
    _, condition_decoder, selected_condition_c = max(
        condition_candidates,
        key=lambda item: item[0],
    )

    progress_candidates = []
    for alpha in progress_grid:
        candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        candidate.fit(train_embedding, train_progress)
        validation_score = r2_score(
            validation_progress,
            candidate.predict(validation_embedding),
        )
        progress_candidates.append((validation_score, candidate, alpha))
    _, progress_decoder, selected_progress_alpha = max(
        progress_candidates,
        key=lambda item: item[0],
    )

    coordinate_scale = np.std(train_embedding, axis=0, ddof=0)
    coordinate_scale = np.where(coordinate_scale > 1e-12, coordinate_scale, 1.0)
    for noise_scale in (0.0, *noise_scales):
        if noise_scale == 0.0:
            evaluated = test_embedding
            category = "downstream_task_decoding"
        else:
            rng = np.random.default_rng(random_state + int(round(noise_scale * 10_000)))
            evaluated = test_embedding + rng.normal(
                scale=noise_scale * coordinate_scale,
                size=test_embedding.shape,
            )
            category = "posthoc_embedding_noise_robustness"
        condition_prediction = condition_decoder.predict(evaluated)
        progress_prediction = progress_decoder.predict(evaluated)
        for metric, value in (
            ("condition_balanced_accuracy", balanced_accuracy_score(test_labels, condition_prediction)),
            ("progress_r2", r2_score(test_progress, progress_prediction)),
            ("progress_mae", mean_absolute_error(test_progress, progress_prediction)),
        ):
            rows.append(
                {
                    **base,
                    "category": category,
                    "metric": metric,
                    "perturbation": "none" if noise_scale == 0.0 else "gaussian_embedding_noise",
                    "perturbation_level": noise_scale,
                    "value": float(value),
                }
            )
    return {
        **base,
        "n_classes": int(len(classes)),
        "condition_solver": "lbfgs",
        "condition_formulation": "multinomial_softmax",
        "preprocessing": "StandardScaler fit on TRAIN within each candidate pipeline",
        "condition_c_grid": condition_grid,
        "selected_condition_c": selected_condition_c,
        "progress_alpha_grid": progress_grid,
        "selected_progress_alpha": selected_progress_alpha,
        "selection_split": "VALIDATION",
        "final_evaluation_split": "TEST",
    }


def stage_quality_report(
    project_root: str | Path,
    config: SharedLatentStageConfig,
    *,
    noise_scales: tuple[float, ...] = (0.1, 0.25),
    force: bool = False,
) -> Path:
    """Stage 8: audit distinct representation-quality dimensions.

    This stage is evaluation-only.  It consumes cached held-out embeddings,
    stage-5 lag profiles, and stage-3 compute records; it never calls fitting
    or transformation and therefore cannot retrain or replace reference runs.
    """
    if any(scale <= 0 for scale in noise_scales):
        raise ValueError("noise_scales must contain only positive values")
    run_root = _run_root(project_root, config)
    stage = _stage_dir(project_root, config, "stage08_quality_report")
    output = stage / "quality_metrics.csv"
    manifest = stage / "manifest.json"
    report_version = 2
    if output.exists() and manifest.exists() and not force:
        cached = json.loads(manifest.read_text(encoding="utf-8"))
        if cached.get("version") == report_version and tuple(cached.get("noise_scales", ())) == noise_scales:
            return output

    embedding_root = run_root / "stage04_embeddings" / "held_out"
    split_path = run_root / "stage02_windows" / "split.json"
    if not embedding_root.exists() or not split_path.exists():
        raise FileNotFoundError("stage 8 requires cached held-out embeddings and split.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_trials = np.asarray(split["train"])
    validation_trials = np.asarray(split["validation"])
    test_trials = np.asarray(split["test"])
    split_sets = [set(train_trials), set(validation_trials), set(test_trials)]
    if any(len(values) == 0 for values in split_sets):
        raise ValueError("TRAIN, VALIDATION, and TEST trial sets must all be non-empty")
    if any(split_sets[left] & split_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("TRAIN, VALIDATION, and TEST trial IDs must be disjoint")
    rows: list[dict[str, object]] = []
    decoder_selections: list[dict[str, object]] = []

    for embedding_path in sorted(embedding_root.glob("*.npz")):
        stem_parts = embedding_path.stem.split("_", 2)
        if len(stem_parts) != 3:
            continue
        subject, model_name, loss_name = stem_parts
        values = np.load(embedding_path, allow_pickle=False)
        valid = values["lag_valid"].astype(bool) if subject == "B" else np.ones(len(values["trial_id"]), dtype=bool)
        train_mask = np.isin(values["trial_id"], train_trials) & valid
        validation_mask = np.isin(values["trial_id"], validation_trials) & valid
        test_mask = np.isin(values["trial_id"], test_trials) & valid
        if not np.any(train_mask) or not np.any(validation_mask) or not np.any(test_mask):
            raise ValueError(f"{embedding_path.name} has an empty TRAIN, VALIDATION, or TEST partition")
        base = {
            "branch": "held_out",
            "subject": subject,
            "model": model_name,
            "loss": loss_name,
        }
        decoder_selections.append(_append_decoder_metrics(
            rows,
            base=base,
            train_embedding=values["embedding_unit"][train_mask],
            validation_embedding=values["embedding_unit"][validation_mask],
            test_embedding=values["embedding_unit"][test_mask],
            train_labels=values["labels"][train_mask],
            validation_labels=values["labels"][validation_mask],
            test_labels=values["labels"][test_mask],
            train_progress=values["progress"][train_mask],
            validation_progress=values["progress"][validation_mask],
            test_progress=values["progress"][test_mask],
            noise_scales=noise_scales,
            random_state=config.seed + (0 if subject == "A" else 1),
        ))

        input_dimensions = config.window_size * (config.n_neurons_A if subject == "A" else config.n_neurons_B)
        embedding_dimensions = int(values["embedding_raw"].shape[1])
        model_dir = run_root / "stage03_models" / "held_out" / f"{subject}_{model_name}_{loss_name}"
        compute_path = model_dir / "compute.json"
        compute = json.loads(compute_path.read_text(encoding="utf-8")) if compute_path.exists() else {}
        checkpoint = model_dir / ("pca.joblib" if model_name == "pca" else "model.pt")
        efficiency_values = {
            "input_dimensions": input_dimensions,
            "embedding_dimensions": embedding_dimensions,
            "embedding_to_input_ratio": embedding_dimensions / input_dimensions,
            "parameter_count": compute.get("n_parameters", 0),
            "training_seconds": compute.get("training_seconds", compute.get("fit_seconds", float("nan"))),
            "artifact_bytes": checkpoint.stat().st_size if checkpoint.exists() else float("nan"),
        }
        for metric, value in efficiency_values.items():
            rows.append({**base, "category": "computational_representational_efficiency", "metric": metric, "perturbation": "none", "perturbation_level": 0.0, "value": value})

    metric_root = run_root / "stage05_metrics" / "held_out"
    for metric_path in sorted(metric_root.glob("*.csv")):
        if metric_path.stem.endswith("_lag"):
            continue
        with metric_path.open(encoding="utf-8") as handle:
            metric_rows = list(csv.DictReader(handle))
        for row in metric_rows:
            category = "temporal_lag_structure" if row["metric"] == "estimated_lag_bins" else "latent_process_recovery"
            rows.append(
                {
                    "branch": "held_out",
                    "subject": row["subject"],
                    "model": row["model"],
                    "loss": row["loss"],
                    "category": category,
                    "reference": row.get("reference", ""),
                    "metric": row["metric"],
                    "perturbation": "none",
                    "perturbation_level": 0.0,
                    "value": float(row["value"]),
                }
            )
            if row["metric"] == "estimated_lag_bins":
                rows.append(
                    {
                        "branch": "held_out", "subject": row["subject"], "model": row["model"], "loss": row["loss"],
                        "category": "temporal_lag_structure", "reference": "", "metric": "absolute_lag_error_bins",
                        "perturbation": "none", "perturbation_level": 0.0,
                        "value": abs(float(row["value"]) - config.lag_bins),
                    }
                )

    fieldnames = ["branch", "subject", "model", "loss", "category", "reference", "metric", "perturbation", "perturbation_level", "value"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        manifest,
        {
            "version": report_version,
            "protocol_fingerprint": _protocol_fingerprint(config),
            "noise_scales": noise_scales,
            "source": "cached held-out embeddings, lag metrics, and compute records",
            "n_rows": len(rows),
            "decoder_protocol": {
                "condition": "multinomial softmax logistic regression; lbfgs; C selected on VALIDATION",
                "progress": "ridge regression; alpha selected on VALIDATION",
                "preprocessing": "StandardScaler fit on TRAIN independently for every candidate",
                "test_usage": "final clean and perturbed scores only",
            },
            "decoder_selections": decoder_selections,
            "limitations": [
                "single simulation seed",
                "noise robustness is post-hoc embedding perturbation, not neural-input or encoder robustness",
                "linear probes quantify decodability, not latent-process recovery",
            ],
        },
    )
    return output


def run_staged_benchmark(
    project_root: str | Path,
    config: SharedLatentStageConfig,
    *,
    branches: tuple[str, ...] = ("held_out", "full_sample"),
    models: tuple[str, ...] = ("pca", "cnn1d", "transformer"),
    # Main comparison: our soft objective, standard supervised InfoNCE, and
    # the two CEBRA-style samplers.
    losses: tuple[str, ...] = ("soft", "infonce", "cebra_time", "cebra_behavior"),
    force: bool = False,
) -> Path:
    """Run all selected stages while reusing existing artifacts."""
    stage_generate(project_root, config, force=force)
    stage_windows(project_root, config, force=force)
    for branch in branches:
        stage_plot_ground_truth_latent(
            project_root, config, branch=branch, force=force,
        )
    for branch in branches:
        for model in models:
            model_losses = ("none",) if model == "pca" else losses
            for loss in model_losses:
                loss = _canonical_loss(model, loss)
                for subject in ("A", "B"):
                    stage_fit(project_root, config, subject=subject, model_name=model, loss_name=loss, branch=branch, force=force)
                    stage_transform(project_root, config, subject=subject, model_name=model, loss_name=loss, branch=branch, force=force)
                stage_evaluate(project_root, config, model_name=model, loss_name=loss, branch=branch, force=force)
                stage_plot(project_root, config, model_name=model, loss_name=loss, branch=branch, force=force)
                # Keep the unnormalized/raw view as a first-class companion to
                # the unit-sphere figure.  This is read-only post-processing:
                # it consumes the cached ``embedding_raw`` transform and never
                # changes a fit or an evaluation artifact.
                stage_plot_raw(
                    project_root,
                    config,
                    model_name=model,
                    loss_name=loss,
                    branch=branch,
                    force=force,
                )
                if model != "pca":
                    stage_plot_training_history(
                        project_root,
                        config,
                        model_name=model,
                        loss_name=loss,
                        branch=branch,
                        force=force,
                    )
                    stage_plot_lag_profile(
                        project_root,
                        config,
                        model_name=model,
                        loss_name=loss,
                        branch=branch,
                        force=force,
                    )
    stage_profile(project_root, config, branches=branches, models=models, losses=losses, force=force)
    return _run_root(project_root, config)
