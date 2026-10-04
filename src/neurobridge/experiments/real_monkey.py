"""Cacheable preparation and baseline embedding for the real monkey data.

This module deliberately does not reuse the synthetic generator.  The local
preloaded recording is a single active Area-2 reaching session with observed
behavioural variables, but no ground-truth latent process or imposed lag.
"""

from __future__ import annotations

import hashlib
import json
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neurobridge.eval.representation import linear_cka
from neurobridge.experiments.staged_shared_latent import _append_decoder_metrics
from neurobridge.losses.infonce import (
    cebra_infonce_loss,
    soft_contrastive_loss,
    supervised_infonce_loss,
)
from neurobridge.models.temporal_cnn import TemporalCNNEncoder, TemporalTransformerEncoder
from neurobridge.sampling.cebra_behavior import CEBRASupervisedWindowDataset
from neurobridge.sampling.cebra_time import CEBRATripletWindowDataset
from neurobridge.data.dataset import TemporalWindowDataset
from neurobridge.sampling.f_windows import build_windows
from neurobridge.train.loop import encode_windows, train_steps, train_triplet_steps
from torch.utils.data import DataLoader, Subset


@dataclass(frozen=True)
class RealMonkeyConfig:
    """Protocol for the first real-data preparation pass.

    The split proportions match the synthetic benchmark's global protocol
    (70/10/20) but are stratified by target direction at the trial level.
    They are recorded in the cache and must be changed explicitly for a new
    real-data run.
    """

    source_relative: str = "data/monkey_reaching_preload_smth_40/macaque_data.jl"
    output_root: str = "outputs"
    run_label: str = "real_monkey_area2_active_staged_2026-09-22"
    setting_name: str = "real_0"
    population_name: str = "all"
    channel_split_seed: int = 42
    channel_indices: tuple[int, ...] | None = None
    imposed_shift_bins: int = 0
    embedding_dim: int = 3
    window_size: int = 21
    stride: int = 1
    padding: str = "center"
    split_seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.10
    test_fraction: float = 0.20
    n_conditions: int = 8
    hidden_dim: int = 64
    cnn_layers: int = 3
    transformer_dim: int = 64
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_dropout: float = 0.1
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_iterations: int = 2000
    cebra_time_offset: int = 10
    embedding_temperature: float = 0.1
    cebra_temperature: float = 1.0
    metadata_temperature: float = 0.5
    time_weight: float = 0.5
    condition_weight: float = 0.5
    training_seed: int = 42

    def __post_init__(self) -> None:
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 3")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if self.padding not in {"valid", "center"}:
            raise ValueError("padding must be 'valid' or 'center'")
        if self.hidden_dim < 1 or self.cnn_layers < 1:
            raise ValueError("neural hidden dimensions and layer counts must be positive")
        if self.transformer_dim < 1 or self.transformer_heads < 1 or self.transformer_layers < 1:
            raise ValueError("transformer dimensions and layer counts must be positive")
        if self.transformer_dim % self.transformer_heads:
            raise ValueError("transformer_dim must be divisible by transformer_heads")
        if self.batch_size < 1 or self.max_iterations < 1:
            raise ValueError("batch_size and max_iterations must be positive")
        if self.channel_split_seed < 0:
            raise ValueError("channel_split_seed must be non-negative")
        if self.channel_indices is not None:
            if not self.channel_indices or len(set(self.channel_indices)) != len(self.channel_indices):
                raise ValueError("channel_indices must be non-empty and unique")
            if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in self.channel_indices):
                raise ValueError("channel_indices must contain non-negative integers")
        if isinstance(self.imposed_shift_bins, bool) or not isinstance(self.imposed_shift_bins, int):
            raise ValueError("imposed_shift_bins must be an integer")
        if self.cebra_time_offset < 1:
            raise ValueError("cebra_time_offset must be positive")
        if self.n_conditions < 2:
            raise ValueError("n_conditions must be at least two")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.embedding_temperature <= 0 or self.cebra_temperature <= 0 or self.metadata_temperature <= 0:
            raise ValueError("contrastive temperatures must be positive")
        if self.split_seed < 0:
            raise ValueError("split_seed must be non-negative")
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(value <= 0 for value in fractions):
            raise ValueError("split fractions must be positive")
        if not np.isclose(sum(fractions), 1.0):
            raise ValueError("split fractions must sum to one")


def _public_loss_name(loss_name: str) -> str:
    """Map legacy cache identifiers to thesis-facing output names."""
    return {
        "cebra_time": "time_contrastive_blocks",
        "cebra_behavior": "behavior_contrastive_blocks",
    }.get(loss_name, loss_name)


def _public_config(config: RealMonkeyConfig) -> dict[str, object]:
    """Serialize protocol values using neutral, user-facing field labels."""
    values = asdict(config)
    if values["channel_indices"] is not None:
        values["channel_indices"] = list(values["channel_indices"])
    values["time_offset_bins"] = values.pop("cebra_time_offset")
    values["temporal_objective_temperature"] = values.pop("cebra_temperature")
    return values


def _config_matches_serialized(values: dict, config: RealMonkeyConfig) -> bool:
    """Accept both historical internal keys and neutral serialized keys."""
    return values == asdict(config) or values == _public_config(config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_figure_provenance(
    project_root: str | Path,
    config: RealMonkeyConfig,
    *,
    branch: str,
    model_name: str,
    objective_name: str,
    figure_dir: Path,
    parent_artifacts: dict[str, Path],
    plotting_config: dict[str, object],
) -> Path:
    """Register a figure bundle and its immutable parent artifacts."""
    root = _run_root(project_root, config)
    split_path = root / "stage01_data" / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    pattern = f"*{model_name}_{objective_name}*"
    artifacts = sorted(
        path for path in figure_dir.glob(pattern)
        if path.is_file() and not path.name.endswith(".provenance.json")
    )
    if model_name == "pca":
        primary = figure_dir / "pca_raw_vs_unit.png"
        if primary.is_file() and primary not in artifacts:
            artifacts.append(primary)
    payload = {
        "setting": config.setting_name,
        "population": config.population_name,
        "architecture": model_name,
        "objective": objective_name,
        "training_seed": config.training_seed,
        "channel_split_seed": config.channel_split_seed,
        "branch": branch,
        "split_trial_ids": split,
        "config": _public_config(config),
        "plotting_config": plotting_config,
        "artifact_sha256": {path.name: _sha256(path) for path in artifacts},
        "artifact_ids": {path.name: _sha256(path) for path in artifacts},
        "parent_artifacts": {
            key: {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for key, path in parent_artifacts.items()
        },
    }
    manifest = figure_dir / f"{model_name}_{objective_name}_figures.provenance.json"
    _write_json(manifest, payload)
    return manifest


def _figure_bundle_is_current(
    manifest_path: Path,
    *,
    config: RealMonkeyConfig,
    branch: str,
    model_name: str,
    objective_name: str,
    figure_dir: Path,
    parent_artifacts: dict[str, Path],
    plotting_config: dict[str, object],
) -> bool:
    if not manifest_path.is_file():
        return False
    cached = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(cached.get(key) != value for key, value in {
        "setting": config.setting_name,
        "population": config.population_name,
        "architecture": model_name,
        "objective": objective_name,
        "training_seed": config.training_seed,
        "channel_split_seed": config.channel_split_seed,
        "branch": branch,
        "config": _public_config(config),
        "plotting_config": plotting_config,
    }.items()):
        return False
    run_root = manifest_path.parents[2]
    expected_parents = {
        key: {"path": str(path.relative_to(run_root)), "sha256": _sha256(path)}
        for key, path in parent_artifacts.items()
    }
    if cached.get("parent_artifacts") != expected_parents:
        return False
    for name, digest in cached.get("artifact_sha256", {}).items():
        artifact = figure_dir / name
        if not artifact.is_file() or _sha256(artifact) != digest:
            return False
    return bool(cached.get("artifact_sha256"))


def _run_root(project_root: str | Path, config: RealMonkeyConfig) -> Path:
    return Path(project_root).resolve() / config.output_root / config.run_label


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _participation_ratio(embedding: np.ndarray) -> float:
    """Estimate effective embedding dimension with float64 SVD arithmetic."""
    values = np.asarray(embedding, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("embedding must be a 2D array with at least two rows and one feature")
    centered = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = singular_values**2
    return float(eigenvalues.sum() ** 2 / max((eigenvalues**2).sum(), 1e-12))


_DATA_CONFIG_KEYS = (
    "source_relative",
    "output_root",
    "run_label",
    "window_size",
    "stride",
    "padding",
    "split_seed",
    "train_fraction",
    "validation_fraction",
    "test_fraction",
    "setting_name",
    "population_name",
    "channel_split_seed",
    "channel_indices",
    "imposed_shift_bins",
)


def _config_matches(cached: dict, config: RealMonkeyConfig, keys: tuple[str, ...]) -> bool:
    current = _public_config(config)
    return all(cached.get(key) == current.get(key) for key in keys)


def _load_source(project_root: str | Path, config: RealMonkeyConfig) -> tuple[Path, dict]:
    source = Path(project_root).resolve() / config.source_relative
    if not source.is_file():
        raise FileNotFoundError(f"Real monkey source not found: {source}")
    data = joblib.load(source)
    required = {"spikes_active", "active_target", "pos_active", "vel_active"}
    missing = sorted(required.difference(data))
    if missing:
        raise KeyError(f"Missing real monkey fields: {missing}")
    spikes = np.asarray(data["spikes_active"])
    target = np.asarray(data["active_target"]).reshape(-1)
    position = np.asarray(data["pos_active"])
    velocity = np.asarray(data["vel_active"])
    if spikes.ndim != 2 or position.ndim != 2 or velocity.ndim != 2:
        raise ValueError("spikes, position, and velocity must be 2D")
    n_time, n_neurons = spikes.shape
    if position.shape[0] != n_time or velocity.shape[0] != n_time:
        raise ValueError("neural and behavioural arrays must share time length")
    if target.shape[0] != n_time:
        raise ValueError("active_target must have one label per time bin")
    trial_len = int(data.get("trial_len") or 0)
    n_trials = int(data.get("num_trials") or 0)
    if trial_len <= 0 or n_trials <= 0 or trial_len * n_trials != n_time:
        raise ValueError("source trial metadata is missing or inconsistent")
    source_neuron_count = n_neurons
    channel_indices = (
        np.arange(n_neurons, dtype=np.int64)
        if config.channel_indices is None
        else np.asarray(config.channel_indices, dtype=np.int64)
    )
    if channel_indices.ndim != 1 or not len(channel_indices) or np.any(channel_indices >= source_neuron_count):
        raise ValueError("channel_indices must identify existing source channels")
    spikes_trials = spikes[:, channel_indices].reshape(n_trials, trial_len, len(channel_indices))
    valid_bins = np.ones((n_trials, trial_len), dtype=bool)
    shift = int(config.imposed_shift_bins)
    if abs(shift) >= trial_len:
        raise ValueError("imposed_shift_bins must be shorter than a trial")
    if shift > 0:
        shifted = np.zeros_like(spikes_trials)
        shifted[:, shift:, :] = spikes_trials[:, :-shift, :]
        valid_bins[:, :shift] = False
        spikes_trials = shifted
    elif shift < 0:
        amount = abs(shift)
        shifted = np.zeros_like(spikes_trials)
        shifted[:, :-amount, :] = spikes_trials[:, amount:, :]
        valid_bins[:, -amount:] = False
        spikes_trials = shifted
    spikes = spikes_trials.reshape(n_time, len(channel_indices))
    n_neurons = len(channel_indices)
    target_by_trial = target.reshape(n_trials, trial_len)
    if not np.all(target_by_trial == target_by_trial[:, :1]):
        raise ValueError("active_target must be constant within each trial")
    source_info = {
        "source": str(source),
        "source_relative": config.source_relative,
        "source_sha256": _sha256(source),
        "n_trials": n_trials,
        "trial_length_bins": trial_len,
        "n_neurons": n_neurons,
        "source_n_neurons": source_neuron_count,
        "channel_indices": channel_indices.astype(int).tolist(),
        "setting_name": config.setting_name,
        "population_name": config.population_name,
        "channel_split_seed": config.channel_split_seed,
        "imposed_shift_bins": shift,
        "n_time_bins": n_time,
        "spikes_dtype": str(spikes.dtype),
        "position_shape": list(position.shape),
        "velocity_shape": list(velocity.shape),
        "target_values": np.unique(target).astype(int).tolist(),
        "target_counts_by_trial": {
            str(int(label)): int(count)
            for label, count in zip(*np.unique(target_by_trial[:, 0], return_counts=True))
        },
        "bin_description": "preloaded Area-2 reaching data; loader documentation states 1 ms bins and 40 ms spike smoothing",
    }
    arrays = {
        "spikes": spikes.astype(np.float32, copy=False),
        "target": target.astype(np.int64, copy=False),
        "position": position.astype(np.float32, copy=False),
        "velocity": velocity.astype(np.float32, copy=False),
        "target_by_trial": target_by_trial[:, 0].astype(np.int64, copy=False),
        "valid_bins": valid_bins.reshape(-1),
    }
    return source, {"info": source_info, "arrays": arrays}


def _make_trial_split(labels: np.ndarray, config: RealMonkeyConfig) -> dict[str, list[int]]:
    labels = np.asarray(labels).reshape(-1)
    all_trials = np.arange(labels.size, dtype=np.int64)
    first = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.test_fraction,
        random_state=config.split_seed,
    )
    train_valid, test = next(first.split(all_trials, labels))
    train_valid = all_trials[train_valid]
    test = all_trials[test]
    validation_relative = config.validation_fraction / (
        config.train_fraction + config.validation_fraction
    )
    second = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_relative,
        random_state=config.split_seed + 1,
    )
    train, validation = next(second.split(train_valid, labels[train_valid]))
    split = {
        "train": sorted(train_valid[train].astype(int).tolist()),
        "validation": sorted(train_valid[validation].astype(int).tolist()),
        "test": sorted(test.astype(int).tolist()),
    }
    if len(set(split["train"]) | set(split["validation"]) | set(split["test"])) != labels.size:
        raise RuntimeError("trial split does not cover every trial exactly once")
    if len(set(split["train"]) & set(split["validation"])) or len(set(split["train"]) & set(split["test"])) or len(set(split["validation"]) & set(split["test"])):
        raise RuntimeError("trial split contains overlapping partitions")
    return split


def prepare_real_monkey(
    project_root: str | Path,
    config: RealMonkeyConfig | None = None,
    *,
    force: bool = False,
) -> Path:
    """Create and cache source metadata plus a stratified trial split."""
    config = config or RealMonkeyConfig()
    root = _run_root(project_root, config)
    stage = root / "stage01_data"
    source, loaded = _load_source(project_root, config)
    fingerprint = {
        "config": _public_config(config),
        "source_sha256": loaded["info"]["source_sha256"],
    }
    manifest_path = stage / "manifest.json"
    split_path = stage / "split.json"
    data_path = stage / "data.npz"
    if manifest_path.exists() and split_path.exists() and data_path.exists() and not force:
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        cached_fingerprint = cached.get("fingerprint", {})
        if (
            cached_fingerprint.get("source_sha256") != fingerprint["source_sha256"]
            or not _config_matches(cached_fingerprint.get("config", {}), config, _DATA_CONFIG_KEYS)
        ):
            raise ValueError("cached real-data protocol/source differs; use force for a new run label")
        return stage
    stage.mkdir(parents=True, exist_ok=True)
    split = _make_trial_split(loaded["arrays"]["target_by_trial"], config)
    np.savez_compressed(
        data_path,
        spikes=loaded["arrays"]["spikes"],
        target=loaded["arrays"]["target"],
        position=loaded["arrays"]["position"],
        velocity=loaded["arrays"]["velocity"],
        target_by_trial=loaded["arrays"]["target_by_trial"],
        valid_bins=loaded["arrays"]["valid_bins"],
    )
    _write_json(split_path, split)
    _write_json(
        manifest_path,
        {
            "fingerprint": fingerprint,
            "config": _public_config(config),
            "source": loaded["info"],
            "setting": config.setting_name,
            "population": config.population_name,
            "channel_split_seed": config.channel_split_seed,
            "training_seed": config.training_seed,
            "split": split,
            "artifact_sha256": {"data_npz": _sha256(data_path), "split_json": _sha256(split_path)},
            "parent_artifacts": {"raw_source_sha256": loaded["info"]["source_sha256"]},
        },
    )
    return stage


def stage_windows(
    project_root: str | Path,
    config: RealMonkeyConfig | None = None,
    *,
    force: bool = False,
) -> Path:
    """Build trial-safe windows using the same window contract as Synthetic v1."""
    config = config or RealMonkeyConfig()
    data_stage = prepare_real_monkey(project_root, config)
    root = _run_root(project_root, config)
    stage = root / "stage02_windows"
    output = stage / "windows.npz"
    metadata_path = stage / "metadata.json"
    if output.exists() and metadata_path.exists() and not force:
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not _config_matches(cached.get("config", {}), config, _DATA_CONFIG_KEYS):
            raise ValueError("cached real-data window protocol differs; use a new run label or force")
        return output

    values = np.load(data_stage / "data.npz", allow_pickle=False)
    split = json.loads((data_stage / "split.json").read_text(encoding="utf-8"))
    n_trials = int(len(values["target_by_trial"]))
    trial_len = int(values["spikes"].shape[0] // n_trials)
    windows, time_id, global_time_id, trial_id, labels = build_windows(
        values["spikes"],
        config.window_size,
        config.stride,
        labels=values["target_by_trial"],
        trial_len=trial_len,
        time_mode="absolute",
        padding=config.padding,
        pad_value=0.0,
    )
    center = time_id.astype(np.int64)
    position = values["position"].reshape(n_trials, trial_len, -1)[trial_id, center]
    velocity = values["velocity"].reshape(n_trials, trial_len, -1)[trial_id, center]
    progress = center.astype(np.float32) / max(trial_len - 1, 1)
    valid_bins = values["valid_bins"].reshape(n_trials, trial_len)
    radius = config.window_size // 2
    valid_centers = np.zeros((n_trials, trial_len), dtype=bool)
    for center_index in range(radius, trial_len - radius):
        valid_centers[:, center_index] = valid_bins[
            :, center_index - radius:center_index + radius + 1
        ].all(axis=1)
    window_valid = valid_centers[trial_id, center]
    trial_split = np.full(n_trials, "unassigned", dtype="U10")
    for name, trials in split.items():
        trial_split[np.asarray(trials, dtype=np.int64)] = name
    split_per_window = trial_split[trial_id]
    stage.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X_windows=windows.astype(np.float32, copy=False),
        time_id=center,
        global_time_id=global_time_id.astype(np.int64),
        trial_id=trial_id.astype(np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        progress=progress,
        position=position.astype(np.float32, copy=False),
        velocity=velocity.astype(np.float32, copy=False),
        split=split_per_window,
        lag_valid=window_valid,
    )
    _write_json(
        metadata_path,
        {
            "config": _public_config(config),
            "n_windows": int(len(windows)),
            "window_shape": list(windows.shape),
            "trial_length_bins": trial_len,
            "padding": config.padding,
            "setting": config.setting_name,
            "population": config.population_name,
            "channel_split_seed": config.channel_split_seed,
            "training_seed": config.training_seed,
            "split_trial_ids": split,
            "artifact_sha256": {"windows_npz": _sha256(output)},
            "parent_artifacts": {
                "data_npz_sha256": _sha256(data_stage / "data.npz"),
                "split_json_sha256": _sha256(data_stage / "split.json"),
            },
            "split": {key: len(value) for key, value in split.items()},
            "split_semantics": "trial-level stratified split; every window inherits its trial partition",
        },
    )
    return output


def fit_pca_embeddings(
    project_root: str | Path,
    config: RealMonkeyConfig | None = None,
    *,
    branch: str = "held_out",
    force: bool = False,
) -> Path:
    """Fit the Synthetic-v1 PCA baseline and export raw/unit embeddings."""
    config = config or RealMonkeyConfig()
    if branch not in {"full_sample", "held_out"}:
        raise ValueError("branch must be 'full_sample' or 'held_out'")
    windows_path = stage_windows(project_root, config)
    root = _run_root(project_root, config)
    output_dir = root / "stage04_embeddings" / branch
    output_path = output_dir / "pca_none.npz"
    metadata_path = output_dir / "metadata.json"
    if output_path.exists() and metadata_path.exists() and not force:
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            not _config_matches(cached.get("config", {}), config, _DATA_CONFIG_KEYS + ("embedding_dim",))
            or cached.get("branch") != branch
        ):
            raise ValueError("cached PCA protocol differs; use a new run label or force")
        return output_path
    values = np.load(windows_path, allow_pickle=False)
    split = json.loads(
        (root / "stage01_data" / "split.json").read_text(encoding="utf-8")
    )
    fit_trials = list(range(int(values["trial_id"].max()) + 1)) if branch == "full_sample" else split["train"]
    fit_mask = np.isin(values["trial_id"], np.asarray(fit_trials, dtype=np.int64))
    flattened = values["X_windows"].reshape(len(values["X_windows"]), -1)
    pca = PCA(n_components=config.embedding_dim, random_state=config.split_seed)
    pca.fit(flattened[fit_mask])
    raw = pca.transform(flattened)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    unit = np.divide(raw, norms, out=np.zeros_like(raw), where=norms > 0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embedding_raw=raw.astype(np.float32),
        embedding_unit=unit.astype(np.float32),
        trial_id=values["trial_id"],
        time_id=values["time_id"],
        global_time_id=values["global_time_id"],
        target=values["labels"],
        position=values["position"],
        velocity=values["velocity"],
        progress=values["progress"],
        split=values["split"],
    )
    model_dir = root / "stage03_models" / branch / "pca_none"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, model_dir / "pca.joblib")
    _write_json(model_dir / "config.json", {**_public_config(config), "branch": branch, "model": "pca", "loss": "none"})
    _write_json(model_dir / "compute.json", {"model": "pca", "loss": "none", "n_fit_windows": int(fit_mask.sum()), "n_parameters": int(flattened.shape[1] * config.embedding_dim)})
    _write_json(
        metadata_path,
        {
            "config": _public_config(config),
            "branch": branch,
            "model": "pca",
            "loss": "none",
            "setting": config.setting_name,
            "population": config.population_name,
            "training_seed": config.training_seed,
            "channel_split_seed": config.channel_split_seed,
            "split_trial_ids": split,
            "artifact_sha256": {
                "embedding_npz": _sha256(output_path),
                "pca_model": _sha256(model_dir / "pca.joblib"),
            },
            "parent_artifacts": {
                "windows_npz_sha256": _sha256(windows_path),
                "split_json_sha256": _sha256(root / "stage01_data" / "split.json"),
            },
            "fit_trials": fit_trials,
            "n_fit_windows": int(fit_mask.sum()),
            "embedding_raw_shape": list(raw.shape),
            "embedding_unit_shape": list(unit.shape),
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "normalization": "unit = embedding_raw / row-wise L2 norm; zero rows remain zero",
        },
    )
    _write_json(model_dir / "provenance.json", {
        "run_id": config.run_label,
        "setting": config.setting_name,
        "population": config.population_name,
        "architecture": "pca",
        "objective": "none",
        "branch": branch,
        "training_seed": config.training_seed,
        "channel_split_seed": config.channel_split_seed,
        "channel_indices": list(config.channel_indices) if config.channel_indices is not None else None,
        "split_trial_ids": split,
        "config": _public_config(config),
        "artifact_sha256": {
            path.name: _sha256(path)
            for path in (output_path, model_dir / "pca.joblib", model_dir / "config.json", model_dir / "compute.json")
        },
        "artifact_ids": {
            path.name: _sha256(path)
            for path in (output_path, model_dir / "pca.joblib", model_dir / "config.json", model_dir / "compute.json")
        },
        "parent_artifacts": {
            "windows_sha256": _sha256(windows_path),
            "split_sha256": _sha256(root / "stage01_data" / "split.json"),
        },
    })
    return output_path


def _load_real_window_dataset(path: Path) -> tuple[TemporalWindowDataset, dict[str, np.ndarray]]:
    values = np.load(path, allow_pickle=False)
    dataset = TemporalWindowDataset(
        values["X_windows"],
        values["time_id"],
        values["global_time_id"],
        values["trial_id"],
        labels_windows=values["labels"],
        extra_metadata={
            "progress": values["progress"],
            "position": values["position"],
            "velocity": values["velocity"],
            "lag_valid": values["lag_valid"].astype(np.int64),
        },
    )
    return dataset, {key: values[key] for key in values.files}


def _make_real_model(
    model_name: str,
    n_features: int,
    config: RealMonkeyConfig,
    *,
    normalize: bool,
) -> torch.nn.Module:
    if model_name == "cnn1d":
        return TemporalCNNEncoder(
            n_features=n_features,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            kernel_size=3,
            n_layers=config.cnn_layers,
            normalize=normalize,
        )
    if model_name == "transformer":
        return TemporalTransformerEncoder(
            n_features=n_features,
            embedding_dim=config.embedding_dim,
            model_dim=config.transformer_dim,
            n_heads=config.transformer_heads,
            n_layers=config.transformer_layers,
            dropout=config.transformer_dropout,
            normalize=normalize,
        )
    raise ValueError("model_name must be 'cnn1d' or 'transformer'")


def _seed_real_training(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _real_metadata_similarity(batch: dict[str, torch.Tensor], config: RealMonkeyConfig) -> torch.Tensor:
    time_distance = torch.abs(batch["time_id"][:, None] - batch["time_id"][None, :]) / 599.0
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


def _evaluate_real_standard_loss(
    model: torch.nn.Module,
    loader: DataLoader | None,
    loss_name: str,
    config: RealMonkeyConfig,
    device: torch.device,
) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            embedding = model(batch["x"])
            if loss_name == "soft":
                loss = soft_contrastive_loss(
                    embedding,
                    _real_metadata_similarity(batch, config),
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


def _evaluate_real_triplet_loss(
    model: torch.nn.Module,
    loader: DataLoader | None,
    config: RealMonkeyConfig,
    device: torch.device,
) -> float:
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
            loss = cebra_infonce_loss(
                model(batch["reference_x"]),
                model(batch["positive_x"]),
                model(batch["negative_x"]),
                temperature=config.cebra_temperature,
            )
            n = int(batch["reference_x"].shape[0])
            total += float(loss.detach().cpu()) * n
            count += n
    return total / max(count, 1)


def fit_neural_model(
    project_root: str | Path,
    config: RealMonkeyConfig | None = None,
    *,
    model_name: str,
    loss_name: str,
    branch: str = "held_out",
    force: bool = False,
) -> Path:
    """Fit one real-data CNN/Transformer configuration with staged caching."""
    config = config or RealMonkeyConfig()
    if model_name not in {"cnn1d", "transformer"}:
        raise ValueError("model_name must be 'cnn1d' or 'transformer'")
    if loss_name not in {"soft", "infonce", "cebra_time", "cebra_behavior"}:
        raise ValueError("unsupported real-data loss")
    if branch not in {"full_sample", "held_out"}:
        raise ValueError("branch must be 'full_sample' or 'held_out'")
    windows_path = stage_windows(project_root, config)
    root = _run_root(project_root, config)
    public_loss_name = _public_loss_name(loss_name)
    model_dir = root / "stage03_models" / branch / f"{model_name}_{public_loss_name}"
    checkpoint = model_dir / "model.pt"
    history_path = model_dir / "training_history.csv"
    config_path = model_dir / "config.json"
    if checkpoint.exists() and history_path.exists() and config_path.exists() and not force:
        cached = json.loads(config_path.read_text(encoding="utf-8"))
        if not _config_matches_serialized(cached.get("config", {}), config) or cached.get("branch") != branch:
            raise ValueError("cached neural protocol differs; use a new run label or force")
        return checkpoint

    dataset, values = _load_real_window_dataset(windows_path)
    split = json.loads((root / "stage01_data" / "split.json").read_text(encoding="utf-8"))
    valid_windows = values.get("lag_valid", np.ones(len(dataset), dtype=bool)).astype(bool)
    if branch == "full_sample":
        train_indices = np.flatnonzero(valid_windows).tolist()
        validation_indices: list[int] = []
    else:
        train_trials = set(split["train"])
        validation_trials = set(split["validation"])
        train_indices = [i for i, trial in enumerate(values["trial_id"]) if int(trial) in train_trials and valid_windows[i]]
        validation_indices = [i for i, trial in enumerate(values["trial_id"]) if int(trial) in validation_trials and valid_windows[i]]
    if not train_indices:
        raise RuntimeError("real-data neural training received no training windows")

    _seed_real_training(config.training_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_real_model(model_name, int(values["X_windows"].shape[-1]), config, normalize=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader_generator = torch.Generator().manual_seed(config.training_seed)
    validation_loader = None
    triplet_dataset = None
    validation_triplet_dataset = None
    if loss_name == "cebra_time":
        triplet_dataset = CEBRATripletWindowDataset(
            dataset,
            train_indices,
            offset=config.cebra_time_offset,
        )
        validation_triplet_dataset = (
            CEBRATripletWindowDataset(dataset, validation_indices, offset=config.cebra_time_offset)
            if validation_indices else None
        )
        train_loader = DataLoader(triplet_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, generator=loader_generator)
        validation_loader = (
            DataLoader(validation_triplet_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
            if validation_triplet_dataset is not None else None
        )
    elif loss_name == "cebra_behavior":
        triplet_dataset = CEBRASupervisedWindowDataset(dataset, train_indices)
        validation_triplet_dataset = (
            CEBRASupervisedWindowDataset(dataset, validation_indices)
            if validation_indices else None
        )
        train_loader = DataLoader(triplet_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, generator=loader_generator)
        validation_loader = (
            DataLoader(validation_triplet_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
            if validation_triplet_dataset is not None else None
        )
    else:
        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            generator=loader_generator,
        )
        validation_loader = (
            DataLoader(Subset(dataset, validation_indices), batch_size=config.batch_size, shuffle=False, drop_last=True)
            if validation_indices else None
        )

    history: list[dict[str, object]] = []
    best_state = None
    best_value = float("inf")
    optimizer_steps = 0
    checkpoint_block = max(1, len(train_loader))
    start_time = time.perf_counter()
    while optimizer_steps < config.max_iterations:
        block_steps = min(checkpoint_block, config.max_iterations - optimizer_steps)
        if loss_name in {"cebra_time", "cebra_behavior"}:
            train_loss = train_triplet_steps(
                model,
                train_loader,
                optimizer,
                lambda reference, positive, negative: cebra_infonce_loss(
                    reference,
                    positive,
                    negative,
                    temperature=config.cebra_temperature,
                ),
                steps=block_steps,
                device=device,
                resample=lambda: triplet_dataset.resample(generator=loader_generator),
            )
            validation_loss = _evaluate_real_triplet_loss(model, validation_loader, config, device) if validation_loader is not None else train_loss
        else:
            if loss_name == "soft":
                loss_fn = lambda embedding, similarity: soft_contrastive_loss(
                    embedding, similarity, temperature=config.embedding_temperature
                )
            else:
                loss_fn = lambda embedding, labels: supervised_infonce_loss(
                    embedding, labels, temperature=config.embedding_temperature
                )
            train_loss = train_steps(
                model,
                train_loader,
                optimizer,
                loss_fn,
                steps=block_steps,
                device=device,
                similarity_builder=(lambda batch: _real_metadata_similarity(batch, config)) if loss_name == "soft" else None,
            )
            validation_loss = _evaluate_real_standard_loss(model, validation_loader, loss_name, config, device) if validation_loader is not None else train_loss
        optimizer_steps += block_steps
        history.append({
            "iteration": optimizer_steps,
            "train_loss": train_loss,
            "validation_loss": validation_loss if validation_loader is not None else float("nan"),
        })
        if validation_loader is not None and validation_loss < best_value:
            best_value = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - start_time
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_name": model_name, "loss_name": loss_name, "device": str(device)}, checkpoint)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "train_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(history)
    _write_json(config_path, {"config": _public_config(config), "branch": branch, "model": model_name, "loss": public_loss_name})
    _write_json(
        model_dir / "compute.json",
        {
            "training_seconds": elapsed,
            "optimizer_steps": optimizer_steps,
            "mean_seconds_per_step": elapsed / max(optimizer_steps, 1),
            "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "n_fit_windows": len(train_indices),
            "train_queries_available": len(train_indices),
            "train_queries_used_per_pass": len(train_loader) * config.batch_size,
            "train_tail_queries_dropped_per_pass": len(train_indices) - len(train_loader) * config.batch_size,
            "device": str(device),
            "time_offset_bins": config.cebra_time_offset if loss_name == "cebra_time" else None,
        },
    )
    return checkpoint


def _resolve_neural_checkpoint(
    project_root: str | Path,
    config: RealMonkeyConfig,
    *,
    model_name: str,
    loss_name: str,
    branch: str,
    checkpoint_path: str | Path | None,
) -> Path:
    """Reuse a validated fit checkpoint or fall back to the legacy fitter."""
    if checkpoint_path is None:
        return fit_neural_model(
            project_root,
            config,
            model_name=model_name,
            loss_name=loss_name,
            branch=branch,
        )

    checkpoint = Path(checkpoint_path).resolve()
    root = _run_root(project_root, config)
    expected_model_dir = root / "stage03_models" / branch / f"{model_name}_{_public_loss_name(loss_name)}"
    if checkpoint.parent != expected_model_dir.resolve() or not checkpoint.is_file():
        raise ValueError("checkpoint must be the fitted artifact for this run/model/objective/branch")
    fit_config_path = checkpoint.parent / "config.json"
    if not fit_config_path.is_file():
        raise FileNotFoundError(f"checkpoint config is missing: {fit_config_path}")
    fit_config = json.loads(fit_config_path.read_text(encoding="utf-8"))
    if (
        fit_config.get("config") != _public_config(config)
        or fit_config.get("branch") != branch
        or fit_config.get("model") != model_name
        or fit_config.get("objective", fit_config.get("loss")) != _public_loss_name(loss_name)
    ):
        raise ValueError("checkpoint configuration does not match the requested frozen transform")
    checkpoint_values = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if checkpoint_values.get("model_name") not in {None, model_name}:
        raise ValueError("checkpoint architecture does not match the requested frozen transform")
    if checkpoint_values.get("branch") not in {None, branch}:
        raise ValueError("checkpoint branch does not match the requested frozen transform")
    saved_objective = checkpoint_values.get("objective")
    if saved_objective is None and checkpoint_values.get("loss_name") is not None:
        saved_objective = _public_loss_name(checkpoint_values["loss_name"])
    if saved_objective is not None and saved_objective != _public_loss_name(loss_name):
        raise ValueError("checkpoint objective does not match the requested frozen transform")
    return checkpoint


def transform_neural_embeddings(
    project_root: str | Path,
    config: RealMonkeyConfig | None = None,
    *,
    model_name: str,
    loss_name: str,
    branch: str = "held_out",
    checkpoint_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Stage 4 frozen raw/unit transforms for a fitted neural encoder."""
    config = config or RealMonkeyConfig()
    checkpoint = _resolve_neural_checkpoint(
        project_root,
        config,
        model_name=model_name,
        loss_name=loss_name,
        branch=branch,
        checkpoint_path=checkpoint_path,
    )
    root = _run_root(project_root, config)
    output_dir = root / "stage04_embeddings" / branch
    public_loss_name = _public_loss_name(loss_name)
    output = output_dir / f"{model_name}_{public_loss_name}.npz"
    metadata_path = output_dir / f"{model_name}_{public_loss_name}_metadata.json"
    if output.exists() and metadata_path.exists() and not force:
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        windows_path = stage_windows(project_root, config)
        expected_cache = {
            "config": _public_config(config),
            "branch": branch,
            "model": model_name,
            "loss": public_loss_name,
            "checkpoint_sha256": _sha256(checkpoint),
            "windows_sha256": _sha256(windows_path),
        }
        if all(cached.get(key) == value for key, value in expected_cache.items()):
            return output
        raise FileExistsError(
            f"cached embedding does not match its checkpoint/input; preserve it and use a new run label: {output}"
        )
    if not force and (output.exists() or metadata_path.exists()):
        raise FileExistsError(f"incomplete embedding cache; preserve it and use a new run label: {output_dir}")
    windows_path = stage_windows(project_root, config)
    dataset, values = _load_real_window_dataset(windows_path)
    loader = DataLoader(dataset, batch_size=max(config.batch_size, 512), shuffle=False)
    checkpoint_values = torch.load(checkpoint, map_location="cpu")
    raw_model = _make_real_model(model_name, int(values["X_windows"].shape[-1]), config, normalize=False)
    unit_model = _make_real_model(model_name, int(values["X_windows"].shape[-1]), config, normalize=True)
    raw_model.load_state_dict(checkpoint_values["state_dict"])
    unit_model.load_state_dict(checkpoint_values["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    raw, _ = encode_windows(raw_model, loader, device=device)
    unit, _ = encode_windows(unit_model, loader, device=device)
    elapsed = time.perf_counter() - start
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embedding_raw=raw.numpy().astype(np.float32),
        embedding_unit=unit.numpy().astype(np.float32),
        trial_id=values["trial_id"],
        time_id=values["time_id"],
        global_time_id=values["global_time_id"],
        target=values["labels"],
        position=values["position"],
        velocity=values["velocity"],
        progress=values["progress"],
        split=values["split"],
    )
    _write_json(
        metadata_path,
        {
            "config": _public_config(config),
            "branch": branch,
            "model": model_name,
            "loss": public_loss_name,
            "checkpoint": str(checkpoint),
            "setting": config.setting_name,
            "population": config.population_name,
            "architecture": model_name,
            "objective": public_loss_name,
            "training_seed": config.training_seed,
            "channel_split_seed": config.channel_split_seed,
            "split_trial_ids": json.loads((root / "stage01_data" / "split.json").read_text(encoding="utf-8")),
            "checkpoint_sha256": _sha256(checkpoint),
            "windows_sha256": _sha256(windows_path),
            "split_sha256": _sha256(root / "stage01_data" / "split.json"),
            "artifact_sha256": {"embedding_npz": _sha256(output)},
            "parent_artifacts": {
                "checkpoint_sha256": _sha256(checkpoint),
                "windows_sha256": _sha256(windows_path),
                "split_sha256": _sha256(root / "stage01_data" / "split.json"),
            },
            "encoding_seconds": elapsed,
            "embedding_shape": list(raw.shape),
            "normalization": "raw uses the checkpoint without output L2 normalization; unit uses the same checkpoint with row-wise L2 normalization",
        },
    )
    return output


def run_neural_suite(
    project_root: str | Path,
    config: RealMonkeyConfig | None = None,
    *,
    models: tuple[str, ...] = ("cnn1d", "transformer"),
    losses: tuple[str, ...] = ("soft", "infonce", "cebra_time", "cebra_behavior"),
    branches: tuple[str, ...] = ("held_out",),
    force: bool = False,
) -> list[Path]:
    """Fit/transform the real-data neural grid while preserving stage caches."""
    config = config or RealMonkeyConfig()
    outputs: list[Path] = []
    for branch in branches:
        for model_name in models:
            for loss_name in losses:
                outputs.append(
                    transform_neural_embeddings(
                        project_root,
                        config,
                        model_name=model_name,
                        loss_name=loss_name,
                        branch=branch,
                        force=force,
                    )
                )
    return outputs


def evaluate_pca(
    project_root: str | Path,
    config: RealMonkeyConfig | None = None,
    *,
    branch: str = "held_out",
    force: bool = False,
) -> Path:
    """Evaluate the frozen real-data PCA embedding."""
    config = config or RealMonkeyConfig()
    embedding_path = fit_pca_embeddings(project_root, config, branch=branch)
    return _evaluate_embedding_artifact(
        project_root,
        config,
        embedding_path=embedding_path,
        model_name="pca",
        loss_name="none",
        branch=branch,
        force=force,
    )


def evaluate_neural(
    project_root: str | Path,
    config: RealMonkeyConfig | None = None,
    *,
    model_name: str,
    loss_name: str,
    branch: str = "held_out",
    checkpoint_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Evaluate one frozen CNN/Transformer embedding without refitting it."""
    config = config or RealMonkeyConfig()
    embedding_path = transform_neural_embeddings(
        project_root,
        config,
        model_name=model_name,
        loss_name=loss_name,
        branch=branch,
        checkpoint_path=checkpoint_path,
    )
    return _evaluate_embedding_artifact(
        project_root,
        config,
        embedding_path=embedding_path,
        model_name=model_name,
        loss_name=loss_name,
        branch=branch,
        force=force,
    )


def _evaluate_embedding_artifact(
    project_root: str | Path,
    config: RealMonkeyConfig | None = None,
    *,
    embedding_path: Path,
    model_name: str,
    loss_name: str,
    branch: str = "held_out",
    force: bool = False,
) -> Path:
    """Run the existing Stage-8-style probes on one frozen real embedding.

    There is no latent ground truth in the monkey recording, so this stage
    reports downstream task/progress accessibility, post-hoc embedding-noise
    robustness, and input/embedding CKA as separate quantities.
    """
    config = config or RealMonkeyConfig()
    if branch not in {"full_sample", "held_out"}:
        raise ValueError("branch must be 'full_sample' or 'held_out'")
    root = _run_root(project_root, config)
    stage = root / "stage05_metrics" / branch
    public_loss_name = _public_loss_name(loss_name)
    metrics_path = stage / f"{model_name}_{public_loss_name}.csv"
    selection_path = stage / f"{model_name}_{public_loss_name}_decoder_selection.json"
    if metrics_path.exists() and selection_path.exists() and not force:
        return metrics_path
    windows = np.load(stage_windows(project_root, config), allow_pickle=False)
    embedding_values = np.load(embedding_path, allow_pickle=False)
    split = embedding_values["split"]
    masks = {name: split == name for name in ("train", "validation", "test")}
    if any(int(mask.sum()) == 0 for mask in masks.values()):
        raise RuntimeError("real-data split must contain non-empty train/validation/test windows")
    evaluation_scope = (
        "held_out_representation_generalization"
        if branch == "held_out"
        else "descriptive_in_sample_representation"
    )
    diagnostic_mask = masks["test"] if branch == "held_out" else np.ones(len(split), dtype=bool)
    diagnostic_split = "test" if branch == "held_out" else "all_windows"
    stage.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    diagnostic_inputs = windows["X_windows"][diagnostic_mask].reshape(int(diagnostic_mask.sum()), -1)
    for representation in ("raw", "unit"):
        embedding = embedding_values[f"embedding_{representation}"]
        base = {
            "branch": branch,
            "evaluation_scope": evaluation_scope,
            "representation_fit_scope": "training_trials_only" if branch == "held_out" else "all_trials",
            "model": model_name,
            "loss": public_loss_name,
            "representation": representation,
        }
        selection = _append_decoder_metrics(
            rows,
            base={**base, "evaluation_split": "test"},
            train_embedding=embedding[masks["train"]],
            validation_embedding=embedding[masks["validation"]],
            test_embedding=embedding[masks["test"]],
            train_labels=embedding_values["target"][masks["train"]],
            validation_labels=embedding_values["target"][masks["validation"]],
            test_labels=embedding_values["target"][masks["test"]],
            train_progress=embedding_values["progress"][masks["train"]],
            validation_progress=embedding_values["progress"][masks["validation"]],
            test_progress=embedding_values["progress"][masks["test"]],
            noise_scales=(0.1, 0.25),
            random_state=config.split_seed,
        )
        behavior_selections = {}
        for behavior_name in ("position", "velocity"):
            target = embedding_values[behavior_name]
            candidates = []
            for alpha in (0.1, 1.0, 10.0):
                candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                candidate.fit(embedding[masks["train"]], target[masks["train"]])
                validation_prediction = candidate.predict(embedding[masks["validation"]])
                validation_score = r2_score(
                    target[masks["validation"]],
                    validation_prediction,
                    multioutput="variance_weighted",
                )
                candidates.append((validation_score, candidate, alpha))
            _, decoder, selected_alpha = max(candidates, key=lambda item: item[0])
            test_prediction = decoder.predict(embedding[masks["test"]])
            rows.extend([
                {
                    **base,
                    "evaluation_split": "test",
                    "category": "downstream_behavior_decoding",
                    "metric": f"{behavior_name}_r2",
                    "perturbation": "none",
                    "perturbation_level": 0.0,
                    "value": float(r2_score(target[masks["test"]], test_prediction, multioutput="variance_weighted")),
                },
                {
                    **base,
                    "evaluation_split": "test",
                    "category": "downstream_behavior_decoding",
                    "metric": f"{behavior_name}_mae",
                    "perturbation": "none",
                    "perturbation_level": 0.0,
                    "value": float(mean_absolute_error(target[masks["test"]], test_prediction)),
                },
            ])
            behavior_selections[behavior_name] = {
                "alpha_grid": [0.1, 1.0, 10.0],
                "selected_alpha": selected_alpha,
                "selection_split": "VALIDATION",
                "final_evaluation_split": "TEST",
            }
        selection["behavior_decoders"] = behavior_selections
        selections.append(selection)
        # CKA is a complementary similarity to the observed neural input;
        # it is not a latent-recovery score and does not select a winner.
        max_rows = min(len(diagnostic_inputs), 20_000)
        sample = np.linspace(0, len(diagnostic_inputs) - 1, max_rows, dtype=int)
        rows.append({
            **base,
            "evaluation_split": diagnostic_split,
            "category": "representational_similarity",
            "metric": "linear_cka_to_input_windows",
            "perturbation": "none",
            "perturbation_level": 0.0,
            "value": float(linear_cka(embedding[diagnostic_mask][sample], diagnostic_inputs[sample])),
        })
        participation_ratio = _participation_ratio(embedding[diagnostic_mask])
        rows.append({
            **base,
            "evaluation_split": diagnostic_split,
            "category": "representational_diagnostics",
            "metric": "participation_ratio_test" if branch == "held_out" else "participation_ratio",
            "perturbation": "none",
            "perturbation_level": 0.0,
            "value": participation_ratio,
        })
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(selection_path, selections)
    split_path = root / "stage01_data" / "split.json"
    _write_json(metrics_path.with_name(f"{model_name}_{public_loss_name}_provenance.json"), {
        "setting": config.setting_name,
        "population": config.population_name,
        "architecture": model_name,
        "objective": public_loss_name,
        "training_seed": config.training_seed,
        "channel_split_seed": config.channel_split_seed,
        "branch": branch,
        "evaluation_scope": evaluation_scope,
        "split_trial_ids": json.loads(split_path.read_text(encoding="utf-8")),
        "config": _public_config(config),
        "artifact_sha256": {
            metrics_path.name: _sha256(metrics_path),
            selection_path.name: _sha256(selection_path),
        },
        "artifact_ids": {
            metrics_path.name: _sha256(metrics_path),
            selection_path.name: _sha256(selection_path),
        },
        "parent_artifacts": {
            "embedding": {"path": str(embedding_path.relative_to(root)), "sha256": _sha256(embedding_path)},
            "windows": {"path": str((root / "stage02_windows" / "windows.npz").relative_to(root)), "sha256": _sha256(root / "stage02_windows" / "windows.npz")},
            "split": {"path": str(split_path.relative_to(root)), "sha256": _sha256(split_path)},
        },
    })
    return metrics_path


def plot_pca_embeddings(
    project_root: str | Path,
    config: RealMonkeyConfig | None = None,
    *,
    branch: str = "held_out",
    force: bool = False,
) -> Path:
    """Create raw-vs-unit sphere PCA figures without changing upstream caches."""
    config = config or RealMonkeyConfig()
    embedding_path = fit_pca_embeddings(project_root, config, branch=branch)
    root = _run_root(project_root, config)
    figure_dir = root / "stage06_figures" / branch
    output = figure_dir / "pca_raw_vs_unit.png"
    figure_manifest = figure_dir / "pca_none_figures.provenance.json"
    figure_parents = {
        "embedding": embedding_path,
        "pca_model": root / "stage03_models" / branch / "pca_none" / "pca.joblib",
        "windows": root / "stage02_windows" / "windows.npz",
        "split": root / "stage01_data" / "split.json",
    }
    figure_config = {"dpi": 180, "marker_convention": "maroon circle=start; black x=end"}
    if output.exists() and not force:
        if _figure_bundle_is_current(
            figure_manifest, config=config, branch=branch, model_name="pca",
            objective_name="none", figure_dir=figure_dir,
            parent_artifacts=figure_parents, plotting_config=figure_config,
        ):
            return output
        raise FileExistsError(f"figure bundle provenance differs; use force=True to regenerate plots only: {figure_dir}")
    values = np.load(embedding_path, allow_pickle=False)
    mask = values["split"] == "test" if branch == "held_out" else np.ones(len(values["split"]), dtype=bool)
    indices = np.flatnonzero(mask)
    if len(indices) > 20_000:
        indices = indices[np.linspace(0, len(indices) - 1, 20_000, dtype=int)]
    labels = values["target"][indices]
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    for axis_index, representation in enumerate(("raw", "unit"), start=1):
        axis = fig.add_subplot(1, 2, axis_index, projection="3d")
        embedding = values[f"embedding_{representation}"][indices]
        scatter = axis.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels, s=2, alpha=0.35, cmap="tab10")
        axis.set_title(f"PCA {representation} ({branch})")
        axis.set_xlabel("Embedding 1")
        axis.set_ylabel("Embedding 2")
        axis.set_zlabel("Embedding 3")
    fig.colorbar(scatter, ax=fig.axes, shrink=0.65, label="movement direction")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    _plot_shared_embedding_views(
        figure_dir,
        values,
        model_name="pca",
        loss_name="none",
        branch=branch,
    )
    _write_figure_provenance(
        project_root, config, branch=branch, model_name="pca", objective_name="none",
        figure_dir=figure_dir, parent_artifacts=figure_parents,
        plotting_config=figure_config,
    )
    return output


def plot_neural_embeddings(
    project_root: str | Path,
    config: RealMonkeyConfig | None = None,
    *,
    model_name: str,
    loss_name: str,
    branch: str = "held_out",
    checkpoint_path: str | Path | None = None,
    dpi: int = 180,
    force: bool = False,
) -> Path:
    """Create raw-vs-unit sphere plots from one frozen neural embedding."""
    config = config or RealMonkeyConfig()
    embedding_path = transform_neural_embeddings(
        project_root,
        config,
        model_name=model_name,
        loss_name=loss_name,
        branch=branch,
        checkpoint_path=checkpoint_path,
    )
    root = _run_root(project_root, config)
    figure_dir = root / "stage06_figures" / branch
    public_loss_name = _public_loss_name(loss_name)
    output = figure_dir / f"{model_name}_{public_loss_name}_raw_vs_unit.png"
    figure_manifest = figure_dir / f"{model_name}_{public_loss_name}_figures.provenance.json"
    resolved_checkpoint = Path(checkpoint_path).resolve() if checkpoint_path is not None else (
        root / "stage03_models" / branch / f"{model_name}_{public_loss_name}" / "model.pt"
    )
    figure_parents = {
        "embedding": embedding_path,
        "checkpoint": resolved_checkpoint,
        "windows": root / "stage02_windows" / "windows.npz",
        "split": root / "stage01_data" / "split.json",
    }
    figure_config = {"dpi": dpi, "marker_convention": "maroon circle=start; black x=end"}
    if output.exists() and not force:
        if _figure_bundle_is_current(
            figure_manifest, config=config, branch=branch, model_name=model_name,
            objective_name=public_loss_name, figure_dir=figure_dir,
            parent_artifacts=figure_parents, plotting_config=figure_config,
        ):
            return output
        raise FileExistsError(f"figure bundle provenance differs; use force=True to regenerate plots only: {figure_dir}")
    values = np.load(embedding_path, allow_pickle=False)
    mask = values["split"] == "test" if branch == "held_out" else np.ones(len(values["split"]), dtype=bool)
    indices = np.flatnonzero(mask)
    if len(indices) > 20_000:
        indices = indices[np.linspace(0, len(indices) - 1, 20_000, dtype=int)]
    labels = values["target"][indices]
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    scatter = None
    for axis_index, representation in enumerate(("raw", "unit"), start=1):
        axis = fig.add_subplot(1, 2, axis_index, projection="3d")
        embedding = values[f"embedding_{representation}"][indices]
        scatter = axis.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels, s=2, alpha=0.35, cmap="tab10")
        axis.set_title(f"{model_name} {public_loss_name} {representation} ({branch})")
        axis.set_xlabel("Embedding 1")
        axis.set_ylabel("Embedding 2")
        axis.set_zlabel("Embedding 3")
    fig.colorbar(scatter, ax=fig.axes, shrink=0.65, label="movement direction")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    _plot_shared_embedding_views(
        figure_dir,
        values,
        model_name=model_name,
        loss_name=public_loss_name,
        branch=branch,
        dpi=dpi,
    )
    _write_figure_provenance(
        project_root, config, branch=branch, model_name=model_name,
        objective_name=public_loss_name, figure_dir=figure_dir,
        parent_artifacts=figure_parents, plotting_config=figure_config,
    )
    return output


def _plot_shared_embedding_views(
    figure_dir: Path,
    values: dict[str, np.ndarray] | np.lib.npyio.NpzFile,
    *,
    model_name: str,
    loss_name: str,
    branch: str,
    dpi: int = 180,
) -> None:
    """Write the same trajectory/sphere views used by the synthetic stage.

    This is a plotting-only adapter for the real-data cache.  It deliberately
    calls the established shared plotting functions, so real and synthetic
    figures have the same colors, start/end markers, labels, and raw/unit
    geometry.  It never touches a checkpoint or recomputes an embedding.
    """
    from neurobridge.viz import (
        plot_condition_trajectories_2d,
        plot_direction_averaged_embedding,
        plot_direction_averaged_embedding_raw,
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    mask = (
        values["split"] == "test"
        if branch == "held_out"
        else np.ones(len(values["split"]), dtype=bool)
    )
    labels = values["target"][mask].astype(int) + 1
    trial_id = values["trial_id"][mask]
    time_id = values["time_id"][mask]
    cmap = plt.get_cmap("hsv")
    norm = plt.Normalize(vmin=0, vmax=7)
    for representation, plotter in (
        ("unit", plot_direction_averaged_embedding),
        ("raw", plot_direction_averaged_embedding_raw),
    ):
        embedding = values[f"embedding_{representation}"][mask]
        suffix = "" if representation == "unit" else "_raw"
        panel_output = figure_dir / f"{model_name}_{loss_name}_embeddings{suffix}.png"
        fig, axis = plt.subplots(figsize=(7, 5))
        fig.subplots_adjust(left=0.10, right=0.96, bottom=0.12, top=0.88)
        zero_based_labels = labels - 1
        axis.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=zero_based_labels,
            s=1,
            alpha=0.08,
            cmap=cmap,
            norm=norm,
        )
        axis.set_title(f"Monkey: {branch} ({representation})")
        axis.set_xlabel("embedding 1")
        axis.set_ylabel("embedding 2")
        axis.set_aspect("equal", adjustable="box")
        for condition in range(8):
            condition_mask = zero_based_labels == condition
            times = np.unique(time_id[condition_mask])
            trajectory = np.asarray([
                embedding[(zero_based_labels == condition) & (time_id == time)].mean(axis=0)
                for time in times
            ])
            if len(trajectory) < 2:
                continue
            color = cmap(condition / 8.0)
            axis.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=1.5, alpha=0.9)
            axis.scatter(trajectory[0, 0], trajectory[0, 1], color="maroon", marker="o", s=22, zorder=3)
            axis.scatter(trajectory[-1, 0], trajectory[-1, 1], color="black", marker="x", s=28, zorder=3)
        fig.savefig(panel_output, dpi=dpi)
        plt.close(fig)
        plot_condition_trajectories_2d(
            embedding=embedding,
            labels=labels,
            trial_id=trial_id,
            time_id=time_id,
            output_folder=figure_dir,
            name=f"monkey_{model_name}_{loss_name}_trajectories_2d{suffix}.html",
            title=(
                f"Monkey {model_name} {loss_name} {branch} "
                f"{representation} condition-averaged trajectories"
            ),
            dims=(0, 1),
            axis_labels=("Embedding 1", "Embedding 2"),
            show=False,
        )
        plotter(
            embedding,
            labels,
            original_label_order=np.sort(np.unique(labels)),
            c_s="maroon",
            output_folder=figure_dir,
            name=f"monkey_{model_name}_{loss_name}_sphere{suffix}.html",
            trial_length=int(np.max(values["time_id"])) + 1,
            quiescent_length=0,
            constant_length=True,
            ww=0,
            quiescent_label=None,
            show=False,
        )


def plot_training_history(
    project_root: str | Path,
    config: RealMonkeyConfig | None = None,
    *,
    model_name: str,
    loss_name: str,
    branch: str = "held_out",
    force: bool = False,
) -> Path:
    """Plot the cached real-data train/validation loss history."""
    config = config or RealMonkeyConfig()
    if model_name not in {"cnn1d", "transformer"}:
        raise ValueError("model_name must be 'cnn1d' or 'transformer'")
    if branch not in {"full_sample", "held_out"}:
        raise ValueError("branch must be 'full_sample' or 'held_out'")
    public_loss_name = _public_loss_name(loss_name)
    root = _run_root(project_root, config)
    output = root / "stage06_figures" / branch / f"{model_name}_{public_loss_name}_loss.png"
    history_path = root / "stage03_models" / branch / f"{model_name}_{public_loss_name}" / "training_history.csv"
    checkpoint_path = root / "stage03_models" / branch / f"{model_name}_{public_loss_name}" / "model.pt"
    split_path = root / "stage01_data" / "split.json"
    parent_artifacts = {
        "training_history": history_path,
        "checkpoint": checkpoint_path,
        "windows": root / "stage02_windows" / "windows.npz",
        "split": split_path,
    }
    figure_dir = output.parent
    figure_manifest = figure_dir / f"{model_name}_{public_loss_name}_loss_figures.provenance.json"
    plot_config = {
        "dpi": 180,
        "train_validation_shading": branch == "held_out",
        "legend_order": ["train", "validation"] if branch == "held_out" else ["train"],
    }
    if output.exists() and not force:
        if _figure_bundle_is_current(
            figure_manifest, config=config, branch=branch, model_name=model_name,
            objective_name=public_loss_name, figure_dir=figure_dir,
            parent_artifacts=parent_artifacts, plotting_config=plot_config,
        ):
            return output
        raise FileExistsError(f"loss figure provenance differs; use force=True to regenerate plots only: {output}")
    with history_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    iterations = np.asarray([float(row["iteration"]) for row in rows])
    train = np.asarray([float(row["train_loss"]) for row in rows])
    validation = np.asarray([float(row["validation_loss"]) for row in rows])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4))
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
        title = f"Training history: {model_name} / {public_loss_name} ({branch})"
    else:
        title = f"Training history: {model_name} / {public_loss_name} ({branch}; train only)"
    axis.set_title(title)
    axis.set_xlabel("Optimizer steps")
    axis.set_ylabel("Contrastive objective")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    _write_figure_provenance(
        project_root, config, branch=branch, model_name=model_name,
        objective_name=public_loss_name, figure_dir=figure_dir,
        parent_artifacts=parent_artifacts, plotting_config=plot_config,
    )
    return output


__all__ = [
    "RealMonkeyConfig",
    "prepare_real_monkey",
    "stage_windows",
    "fit_pca_embeddings",
    "fit_neural_model",
    "transform_neural_embeddings",
    "run_neural_suite",
    "evaluate_pca",
    "evaluate_neural",
    "plot_pca_embeddings",
    "plot_neural_embeddings",
    "plot_training_history",
]
