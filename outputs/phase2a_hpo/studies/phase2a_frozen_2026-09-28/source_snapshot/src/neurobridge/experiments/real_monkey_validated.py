"""Versioned, held-out monkey experiment with update-based validation.

The historical implementation remains available in ``real_monkey``. This
module opts into a new schedule and writes only to a new run directory.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from neurobridge.eval.representation import distance_geometry_correlation
from neurobridge.experiments import real_monkey as legacy
from neurobridge.losses.infonce import (
    cebra_infonce_loss,
    soft_contrastive_loss,
    supervised_infonce_loss,
)


@dataclass(frozen=True)
class ValidationSchedule:
    max_steps: int = 4000
    validation_interval: int = 150
    min_steps: int = 750
    patience: int = 5
    relative_min_delta: float = 0.001

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.validation_interval < 1:
            raise ValueError("max_steps and validation_interval must be positive")
        if self.min_steps < 0 or self.min_steps > self.max_steps:
            raise ValueError("min_steps must lie between zero and max_steps")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if not 0.0 <= self.relative_min_delta < 1.0:
            raise ValueError("relative_min_delta must lie in [0, 1)")


DEFAULT_SCHEDULE = ValidationSchedule()
DEFAULT_RUN_LABEL = "clean_rebuild_2026-09-23_4000"
DEFAULT_PARENT_RUN = "real_monkey_area2_active_staged_2026-09-23_8000"
MODEL_NAMES = ("cnn1d", "transformer")
OBJECTIVE_NAMES = (
    "soft",
    "infonce",
    "time_contrastive_blocks",
    "behavior_contrastive_blocks",
)
_INTERNAL_OBJECTIVES = {
    "soft": "soft",
    "infonce": "infonce",
    "time_contrastive_blocks": "cebra_time",
    "behavior_contrastive_blocks": "cebra_behavior",
}
_PUBLIC_OBJECTIVES = {value: key for key, value in _INTERNAL_OBJECTIVES.items()}
_DATA_CACHE_KEYS = (
    "source_relative",
    "window_size",
    "stride",
    "padding",
    "split_seed",
    "train_fraction",
    "validation_fraction",
    "test_fraction",
)


def canonical_config(
    *,
    run_label: str = DEFAULT_RUN_LABEL,
    batch_size: int = 1024,
    max_steps: int = DEFAULT_SCHEDULE.max_steps,
    training_seed: int = 42,
    setting_name: str = "real_0",
    population_name: str = "all",
    channel_indices: tuple[int, ...] | None = None,
    channel_split_seed: int = 42,
    imposed_shift_bins: int = 0,
    output_root: str = "outputs/runs",
) -> legacy.RealMonkeyConfig:
    """Return the authorized prospective protocol without changing old runs."""
    return replace(
        legacy.RealMonkeyConfig(),
        run_label=run_label,
        batch_size=batch_size,
        max_iterations=max_steps,
        training_seed=training_seed,
        setting_name=setting_name,
        population_name=population_name,
        channel_indices=channel_indices,
        channel_split_seed=channel_split_seed,
        imposed_shift_bins=imposed_shift_bins,
        output_root=output_root,
    )


def make_random_channel_split(n_channels: int = 65, seed: int = 42) -> dict[str, Any]:
    """Create and validate the single channel partition shared by R0 and R10."""
    if n_channels != 65:
        raise ValueError(f"the canonical monkey recording has 65 channels, got {n_channels}")
    permutation = np.random.default_rng(seed).permutation(n_channels).astype(np.int64)
    channels_a = permutation[:32]
    channels_b = permutation[32:]
    if len(channels_a) != 32 or len(channels_b) != 33:
        raise RuntimeError("random channel split has incorrect population sizes")
    if set(channels_a.tolist()).intersection(channels_b.tolist()):
        raise RuntimeError("random channel split is not disjoint")
    if set(channels_a.tolist()).union(channels_b.tolist()) != set(range(n_channels)):
        raise RuntimeError("random channel split does not cover all source channels")
    return {
        "channel_split_seed": int(seed),
        "method": "numpy.default_rng(seed).permutation; A=first 32; B=remaining 33",
        "A_channel_indices": channels_a.astype(int).tolist(),
        "B_channel_indices": channels_b.astype(int).tolist(),
        "n_source_channels": int(n_channels),
        "verified_disjoint": True,
        "verified_complete_cover": True,
    }


def _json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _neutralize_metadata_keys(value: Any) -> Any:
    """Use neutral public names for internal protocol fields in output JSON."""
    replacements = {
        "cebra_time_offset": "time_offset_bins",
        "cebra_temperature": "temporal_objective_temperature",
    }
    if isinstance(value, dict):
        return {
            replacements.get(key, key): _neutralize_metadata_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_neutralize_metadata_keys(item) for item in value]
    return value


def _append_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _git_provenance(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=project_root, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "git_head": run("rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "worktree_status_line_count": len(status.splitlines()) if status else 0,
    }


def _cache_config_compatible(old: dict[str, Any], config: legacy.RealMonkeyConfig) -> bool:
    current = asdict(config)
    return all(old.get(key) == current.get(key) for key in _DATA_CACHE_KEYS)


def reuse_natural_input_cache(
    project_root: str | Path,
    config: legacy.RealMonkeyConfig,
    *,
    parent_run_label: str = DEFAULT_PARENT_RUN,
) -> dict[str, Path]:
    """Copy and verify exact source/split/window caches into the new run root.

    Copies are content-identical; their manifests are versioned for this run.
    A changed source, split, or window contract raises instead of recomputing or
    silently accepting a near match.
    """
    project_root = Path(project_root).resolve()
    root = legacy._run_root(project_root, config)
    parent_root = project_root / config.output_root / parent_run_label
    source = project_root / config.source_relative
    if not source.is_file():
        raise FileNotFoundError(source)
    if not parent_root.is_dir():
        raise FileNotFoundError(f"parent cache run not found: {parent_root}")

    parent_manifest_path = parent_root / "stage01_data" / "manifest.json"
    parent_split_path = parent_root / "stage01_data" / "split.json"
    parent_data_path = parent_root / "stage01_data" / "data.npz"
    parent_window_metadata = parent_root / "stage02_windows" / "metadata.json"
    parent_windows_path = parent_root / "stage02_windows" / "windows.npz"
    needed = (
        parent_manifest_path, parent_split_path, parent_data_path,
        parent_window_metadata, parent_windows_path,
    )
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"parent run lacks reusable cache files: {missing}")

    old_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    old_window_metadata = json.loads(parent_window_metadata.read_text(encoding="utf-8"))
    source_hash = _hash(source)
    parent_fingerprint = old_manifest.get("fingerprint", {})
    if parent_fingerprint.get("source_sha256") != source_hash:
        raise ValueError("source data hash differs from the parent cache; refusing reuse")
    if not _cache_config_compatible(parent_fingerprint.get("config", {}), config):
        raise ValueError("split/window protocol differs from parent cache; refusing reuse")
    if not _cache_config_compatible(old_window_metadata.get("config", {}), config):
        raise ValueError("window metadata differs from requested split/window protocol")

    root.mkdir(parents=True, exist_ok=True)
    destinations = {
        "data": root / "stage01_data" / "data.npz",
        "split": root / "stage01_data" / "split.json",
        "manifest": root / "stage01_data" / "manifest.json",
        "windows": root / "stage02_windows" / "windows.npz",
        "window_metadata": root / "stage02_windows" / "metadata.json",
    }
    sources = {
        "data": parent_data_path,
        "split": parent_split_path,
        "manifest": parent_manifest_path,
        "windows": parent_windows_path,
        "window_metadata": parent_window_metadata,
    }
    import shutil

    for key in ("data", "split", "windows"):
        destination = destinations[key]
        source_path = sources[key]
        if destination.exists():
            if _hash(destination) != _hash(source_path):
                raise FileExistsError(f"existing versioned cache differs: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)

    manifest_path = destinations["manifest"]
    if manifest_path.exists():
        original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = _neutralize_metadata_keys(original_manifest)
        if not _cache_config_compatible(manifest.get("config", {}), config):
            raise FileExistsError(f"existing versioned manifest uses another protocol: {manifest_path}")
        if manifest != original_manifest:
            _json_write(manifest_path, manifest)
    else:
        shutil.copy2(sources["manifest"], manifest_path)
        manifest = _neutralize_metadata_keys(json.loads(manifest_path.read_text(encoding="utf-8")))
        manifest["fingerprint"]["config"] = legacy._public_config(config)
        manifest["config"] = legacy._public_config(config)
        manifest["cache_parent_run"] = parent_run_label
        manifest["cache_parent_file_hashes"] = {
            key: _hash(path) for key, path in sources.items()
        }
        _json_write(manifest_path, manifest)

    metadata_path = destinations["window_metadata"]
    if metadata_path.exists():
        original_window_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        window_metadata = _neutralize_metadata_keys(original_window_metadata)
        if not _cache_config_compatible(window_metadata.get("config", {}), config):
            raise FileExistsError(f"existing window metadata uses another protocol: {metadata_path}")
        if window_metadata != original_window_metadata:
            _json_write(metadata_path, window_metadata)
    else:
        shutil.copy2(sources["window_metadata"], metadata_path)
        window_metadata = _neutralize_metadata_keys(json.loads(metadata_path.read_text(encoding="utf-8")))
        window_metadata["config"] = legacy._public_config(config)
        window_metadata["cache_parent_run"] = parent_run_label
        _json_write(metadata_path, window_metadata)
    reuse_record = {
        "run_id": config.run_label,
        "reused_from": parent_run_label,
        "source_sha256": source_hash,
        "source_file": config.source_relative,
        "arrays_content_identical": True,
        "split_sha256": _hash(destinations["split"]),
        "data_sha256": _hash(destinations["data"]),
        "windows_sha256": _hash(destinations["windows"]),
        "data_config_keys_verified": list(_DATA_CACHE_KEYS),
        "batch_size_and_optimizer_schedule_do_not_affect_data_or_windows": True,
    }
    reuse_path = root / "stage02_windows" / "cache_reuse.json"
    _json_write(reuse_path, reuse_record)
    return {**destinations, "cache_reuse": reuse_path, "root": root}


def reuse_train_only_pca(
    project_root: str | Path,
    config: legacy.RealMonkeyConfig,
    *,
    parent_run_label: str = DEFAULT_PARENT_RUN,
) -> Path:
    """Reuse PCA only after exact input, train split and dimension checks."""
    project_root = Path(project_root).resolve()
    parent_root = project_root / config.output_root / parent_run_label
    root = legacy._run_root(project_root, config)
    parent_embedding = parent_root / "stage04_embeddings" / "held_out" / "pca_none.npz"
    parent_metadata = parent_root / "stage04_embeddings" / "held_out" / "metadata.json"
    parent_model = parent_root / "stage03_models" / "held_out" / "pca_none" / "pca.joblib"
    parent_model_config = parent_root / "stage03_models" / "held_out" / "pca_none" / "config.json"
    parent_compute = parent_root / "stage03_models" / "held_out" / "pca_none" / "compute.json"
    required = (parent_embedding, parent_metadata, parent_model, parent_model_config, parent_compute)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("parent run lacks the train-only PCA cache")
    metadata = json.loads(parent_metadata.read_text(encoding="utf-8"))
    if metadata.get("branch") != "held_out" or metadata.get("model") != "pca":
        raise ValueError("parent PCA is not a held-out PCA fit")
    if not _cache_config_compatible(metadata.get("config", {}), config):
        raise ValueError("PCA source/split/window protocol differs; refusing reuse")
    if metadata.get("config", {}).get("embedding_dim") != config.embedding_dim:
        raise ValueError("PCA embedding dimension differs; refusing reuse")
    split = json.loads((root / "stage01_data" / "split.json").read_text(encoding="utf-8"))
    if metadata.get("fit_trials") != split["train"]:
        raise ValueError("parent PCA fit trials differ; refusing reuse")

    import shutil

    copies = {
        parent_embedding: root / "stage04_embeddings" / "held_out" / "pca_none.npz",
        parent_model: root / "stage03_models" / "held_out" / "pca_none" / "pca.joblib",
        parent_compute: root / "stage03_models" / "held_out" / "pca_none" / "compute.json",
    }
    for source, target in copies.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if _hash(source) != _hash(target):
                raise FileExistsError(f"existing PCA artifact differs: {target}")
        else:
            shutil.copy2(source, target)
    model_config_path = root / "stage03_models" / "held_out" / "pca_none" / "config.json"
    if not model_config_path.exists():
        model_config = json.loads(parent_model_config.read_text(encoding="utf-8"))
        model_config.update(legacy._public_config(config))
        model_config["branch"] = "held_out"
        model_config["model"] = "pca"
        model_config["loss"] = "none"
        _json_write(model_config_path, model_config)
    target_metadata = root / "stage04_embeddings" / "held_out" / "metadata.json"
    new_metadata = dict(metadata)
    new_metadata["config"] = legacy._public_config(config)
    new_metadata["reused_from"] = f"{parent_run_label}/stage04_embeddings/held_out/pca_none.npz"
    new_metadata["source_embedding_sha256"] = _hash(parent_embedding)
    _json_write(target_metadata, new_metadata)
    return copies[parent_embedding]


def _make_validation_loader(
    dataset: Any,
    indices: list[int],
    *,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, dict[str, int]]:
    if not indices:
        raise ValueError("held-out validation indices are empty")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(indices))
    fixed_indices = [indices[int(i)] for i in order]
    subset = Subset(dataset, fixed_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=True)
    count = len(fixed_indices)
    kept = (count // batch_size) * batch_size
    return loader, {
        "available_queries": count,
        "evaluated_queries": kept,
        "excluded_tail_queries": count - kept,
        "candidate_count_full_batch": batch_size - 1,
        "validation_order_seed": seed,
    }


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _batch_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    objective: str,
    config: legacy.RealMonkeyConfig,
) -> tuple[torch.Tensor, int, int]:
    if objective in {"cebra_time", "cebra_behavior"}:
        reference = model(batch["reference_x"])
        positive = model(batch["positive_x"])
        negative = model(batch["negative_x"])
        loss = cebra_infonce_loss(
            reference,
            positive,
            negative,
            temperature=config.cebra_temperature,
        )
        batch_size = int(batch["reference_x"].shape[0])
        return loss, batch_size, batch_size

    embedding = model(batch["x"])
    if objective == "soft":
        loss = soft_contrastive_loss(
            embedding,
            legacy._real_metadata_similarity(batch, config),
            temperature=config.embedding_temperature,
        )
    else:
        loss = supervised_infonce_loss(
            embedding,
            batch["label"],
            temperature=config.embedding_temperature,
        )
    batch_size = int(batch["x"].shape[0])
    return loss, batch_size, batch_size - 1


def _evaluate_fixed_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    objective: str,
    config: legacy.RealMonkeyConfig,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n_queries = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            loss, count, _ = _batch_loss(model, batch, objective, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            total += float(loss.detach().cpu()) * count
            n_queries += count
    if not n_queries:
        raise RuntimeError("fixed-size validation loader produced no queries")
    return total / n_queries


def _state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().float().view(-1).cpu() for parameter in model.parameters()])


def _source_provenance(project_root: Path) -> dict[str, Any]:
    sources = {
        "monkey_pipeline": "src/neurobridge/experiments/real_monkey.py",
        "validated_runner": "src/neurobridge/experiments/real_monkey_validated.py",
        "contrastive_loss": "src/neurobridge/losses/infonce.py",
        "temporal_encoders": "src/neurobridge/models/temporal_cnn.py",
        "temporal_pair_sampler": "src/neurobridge/sampling/cebra_time.py",
        "behavior_pair_sampler": "src/neurobridge/sampling/cebra_behavior.py",
        "optimizer_loop": "src/neurobridge/train/loop.py",
    }
    return {
        label: _hash(project_root / path)
        for label, path in sources.items() if (project_root / path).is_file()
    }


def _provenance_record(
    project_root: Path,
    config: legacy.RealMonkeyConfig,
    schedule: ValidationSchedule,
    *,
    subject: str,
    model_name: str,
    objective: str,
    branch: str,
    device: torch.device,
    checkpoint: Path,
) -> dict[str, Any]:
    public_objective = _PUBLIC_OBJECTIVES[objective]
    run_root = legacy._run_root(project_root, config)
    split_path = run_root / "stage01_data" / "split.json"
    windows_path = run_root / "stage02_windows" / "windows.npz"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    return {
        "run_id": config.run_label,
        "setting": config.setting_name,
        "population": config.population_name if config.population_name != "all" else subject,
        "architecture": model_name,
        "objective": public_objective,
        "branch": branch,
        "training_seed": config.training_seed,
        "channel_split_seed": config.channel_split_seed,
        "channel_indices": list(config.channel_indices) if config.channel_indices is not None else None,
        "imposed_shift_bins": config.imposed_shift_bins,
        "split_trial_ids": split,
        "split_sha256": _hash(split_path),
        "dataset_sha256": _hash(project_root / config.source_relative),
        "subject_or_population": config.population_name if config.population_name != "all" else subject,
        "global_batch_size": config.batch_size,
        "microbatch_size": None,
        "block_size": None,
        "candidate_count_formula": (
            "actual_batch_size" if objective in {"cebra_time", "cebra_behavior"}
            else "actual_batch_size - 1 (self excluded)"
        ),
        "window_length": config.window_size,
        "stride": config.stride,
        "embedding_dimension": config.embedding_dim,
        "optimizer": "AdamW",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "temperature": (
            config.cebra_temperature if objective in {"cebra_time", "cebra_behavior"}
            else config.embedding_temperature
        ),
        "schedule": asdict(schedule),
        "git": _git_provenance(project_root),
        "source_file_sha256": _source_provenance(project_root),
        "python": __import__("sys").version,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "selected_checkpoint": _relative(project_root, checkpoint),
        "artifact_id": _hash(checkpoint),
        "artifact_sha256": {"selected_checkpoint": _hash(checkpoint)},
        "parent_artifacts": {
            "raw_source_sha256": _hash(project_root / config.source_relative),
            "windows_sha256": _hash(windows_path),
            "split_sha256": _hash(split_path),
        },
    }


def fit_validated_neural_model(
    project_root: str | Path,
    config: legacy.RealMonkeyConfig,
    *,
    model_name: str,
    objective_name: str,
    schedule: ValidationSchedule = DEFAULT_SCHEDULE,
    subject: str = "all",
    branch: str = "held_out",
    full_sample_updates: int | None = None,
    windows_path: str | Path | None = None,
    split_path: str | Path | None = None,
) -> Path:
    """Fit one held-out model with persistent iteration and fixed-K validation."""
    project_root = Path(project_root).resolve()
    if model_name not in MODEL_NAMES or objective_name not in OBJECTIVE_NAMES:
        raise ValueError("unknown model or objective")
    if branch not in {"held_out", "full_sample"}:
        raise ValueError("branch must be held_out or full_sample")
    if config.batch_size != 1024:
        raise ValueError("the authorized prospective global batch size is 1024")
    if config.embedding_dim != 3 or config.window_size != 21 or config.stride != 1:
        raise ValueError("prospective representation protocol differs from the authorized settings")
    if branch == "held_out" and config.max_iterations != schedule.max_steps:
        raise ValueError("max_iterations must equal the schedule's max_steps")
    if branch == "full_sample" and (
        full_sample_updates is None or full_sample_updates < 1 or full_sample_updates > schedule.max_steps
    ):
        raise ValueError("full_sample_updates must be between 1 and the held-out update cap")
    if windows_path is None:
        windows_path = legacy.stage_windows(project_root, config)
    windows_path = Path(windows_path).resolve()
    root = legacy._run_root(project_root, config)
    if split_path is None:
        split_path = root / "stage01_data" / "split.json"
    split_path = Path(split_path).resolve()
    objective = _INTERNAL_OBJECTIVES[objective_name]
    artifact_label = legacy._public_loss_name(objective)
    model_label = f"{model_name}_{artifact_label}" if subject == "all" else f"{subject}_{model_name}_{artifact_label}"
    model_dir = root / "stage03_models" / branch / model_label
    checkpoint = model_dir / "model.pt"
    best_checkpoint = model_dir / "best_validation_checkpoint.pt"
    stopping_checkpoint = model_dir / "stopping_checkpoint.pt"
    history_path = model_dir / "training_history.csv"
    config_path = model_dir / "config.json"
    required_cache = [checkpoint, stopping_checkpoint, history_path, config_path]
    if branch == "held_out":
        required_cache.append(best_checkpoint)
    if all(path.exists() for path in required_cache):
        cached = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            cached.get("config") == legacy._public_config(config)
            and cached.get("schedule") == asdict(schedule)
            and cached.get("branch") == branch
            and cached.get("objective") == objective_name
            and cached.get("model") == model_name
            and cached.get("full_sample_updates") == full_sample_updates
        ):
            return checkpoint
        raise FileExistsError(f"cached validated fit uses a different protocol: {model_dir}")
    if model_dir.exists() and any(model_dir.iterdir()):
        raise FileExistsError(f"partial model output exists; use a new run ID: {model_dir}")

    dataset, values = legacy._load_real_window_dataset(windows_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    valid = values.get("lag_valid", np.ones(len(dataset), dtype=bool)).astype(bool)
    if branch == "held_out":
        train_trials = np.asarray(split["train"], dtype=np.int64)
        validation_trials = np.asarray(split["validation"], dtype=np.int64)
        train_indices = np.flatnonzero(np.isin(values["trial_id"], train_trials) & valid).tolist()
        validation_indices = np.flatnonzero(np.isin(values["trial_id"], validation_trials) & valid).tolist()
        if not train_indices or not validation_indices:
            raise RuntimeError("held-out training and validation windows must both be non-empty")
    else:
        train_indices = np.flatnonzero(valid).tolist()
        validation_indices = []
        if len(train_indices) < config.batch_size:
            raise RuntimeError("full-sample training has fewer eligible windows than one full batch")

    legacy._seed_real_training(config.training_seed)
    loader_generator = torch.Generator().manual_seed(config.training_seed)
    validation_seed = config.training_seed + 10_000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = legacy._make_real_model(
        model_name, int(values["X_windows"].shape[-1]), config, normalize=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if {id(parameter) for parameter in optimizer_parameters} != {id(parameter) for parameter in trainable_parameters}:
        raise RuntimeError("optimizer parameter set differs from the trainable model parameter set")

    triplet_dataset = None
    validation_triplets = None
    if objective == "cebra_time":
        triplet_dataset = legacy.CEBRATripletWindowDataset(
            dataset, train_indices, offset=config.cebra_time_offset,
            valid_mask=torch.as_tensor(valid),
        )
        if branch == "held_out":
            validation_triplets = legacy.CEBRATripletWindowDataset(
                dataset, validation_indices, offset=config.cebra_time_offset,
                valid_mask=torch.as_tensor(valid),
            )
        train_loader = DataLoader(
            triplet_dataset, batch_size=config.batch_size, shuffle=True,
            drop_last=True, generator=loader_generator,
        )
        if validation_triplets is not None:
            validation_order = np.random.default_rng(validation_seed).permutation(len(validation_triplets))
            validation_subset = Subset(validation_triplets, validation_order.tolist())
            validation_loader = DataLoader(
                validation_subset, batch_size=config.batch_size, shuffle=False, drop_last=True,
            )
            validation_raw_count = len(validation_triplets)
        else:
            validation_loader = None
            validation_raw_count = 0
    elif objective == "cebra_behavior":
        triplet_dataset = legacy.CEBRASupervisedWindowDataset(dataset, train_indices)
        if branch == "held_out":
            validation_triplets = legacy.CEBRASupervisedWindowDataset(dataset, validation_indices)
        train_loader = DataLoader(
            triplet_dataset, batch_size=config.batch_size, shuffle=True,
            drop_last=True, generator=loader_generator,
        )
        if validation_triplets is not None:
            validation_order = np.random.default_rng(validation_seed).permutation(len(validation_triplets))
            validation_subset = Subset(validation_triplets, validation_order.tolist())
            validation_loader = DataLoader(
                validation_subset, batch_size=config.batch_size, shuffle=False, drop_last=True,
            )
            validation_raw_count = len(validation_triplets)
        else:
            validation_loader = None
            validation_raw_count = 0
    else:
        train_loader = DataLoader(
            Subset(dataset, train_indices), batch_size=config.batch_size,
            shuffle=True, drop_last=True, generator=loader_generator,
        )
        if branch == "held_out":
            validation_loader, validation_summary = _make_validation_loader(
                dataset, validation_indices, batch_size=config.batch_size, seed=validation_seed,
            )
            validation_raw_count = validation_summary["available_queries"]
        else:
            validation_loader = None
            validation_raw_count = 0

    validation_evaluated = len(validation_loader) * config.batch_size if validation_loader is not None else 0
    validation_summary = {
        "available_queries": int(validation_raw_count),
        "evaluated_queries": int(validation_evaluated),
        "excluded_tail_queries": int(validation_raw_count - validation_evaluated),
        "validation_order_seed": validation_seed,
        "full_batch_size": config.batch_size,
        "effective_candidates_per_anchor": (
            config.batch_size if objective in {"cebra_time", "cebra_behavior"}
            else config.batch_size - 1
        ),
        "validation_order_fixed_across_checks": branch == "held_out",
        "triplet_validation_negatives_fixed_across_checks": branch == "held_out",
    }

    initial_parameters = _parameter_vector(model)
    best_raw_state: dict[str, torch.Tensor] | None = None
    best_raw_value = float("inf")
    best_raw_step = 0
    meaningful_reference: float | None = None
    meaningful_step = 0
    stale_checks = 0
    history: list[dict[str, Any]] = []
    batch_sizes: Counter[int] = Counter()
    candidate_counts: Counter[int] = Counter()
    first_batch_shapes: dict[str, list[int]] | None = None
    first_gradient_norm: float | None = None
    first_step_parameter_delta: float | None = None
    updates_since_validation: list[tuple[float, int]] = []
    training_iterator = iter(train_loader)
    optimizer_steps = 0
    stop_reason = "max_steps_reached"
    max_updates = schedule.max_steps if branch == "held_out" else int(full_sample_updates)
    start_time = time.perf_counter()
    progress_path = root / "progress.jsonl"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while optimizer_steps < max_updates:
        if triplet_dataset is not None:
            triplet_dataset.resample(generator=loader_generator)
        try:
            raw_batch = next(training_iterator)
        except StopIteration:
            training_iterator = iter(train_loader)
            if triplet_dataset is not None:
                triplet_dataset.resample(generator=loader_generator)
            raw_batch = next(training_iterator)
        batch = _move_batch(raw_batch, device)
        if first_batch_shapes is None:
            first_batch_shapes = {
                key: list(value.shape) for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, batch_size, effective_k = _batch_loss(model, batch, objective, config)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at update {optimizer_steps + 1}")
        loss.backward()
        if first_gradient_norm is None:
            grad_sq = sum(
                float(parameter.grad.detach().float().norm().cpu()) ** 2
                for parameter in trainable_parameters if parameter.grad is not None
            )
            first_gradient_norm = float(np.sqrt(grad_sq))
            if not np.isfinite(first_gradient_norm) or first_gradient_norm <= 0:
                raise RuntimeError("first optimizer update has no finite non-zero gradient")
        optimizer.step()
        optimizer_steps += 1
        if optimizer_steps == 1:
            first_step_parameter_delta = float(torch.linalg.vector_norm(_parameter_vector(model) - initial_parameters))
        batch_sizes[batch_size] += 1
        candidate_counts[effective_k] += 1
        updates_since_validation.append((float(loss.detach().cpu()), batch_size))

        validation_due = (
            branch == "held_out"
            and (optimizer_steps % schedule.validation_interval == 0 or optimizer_steps == max_updates)
        )
        history_due = validation_due or (
            branch == "full_sample"
            and (optimizer_steps % schedule.validation_interval == 0 or optimizer_steps == max_updates)
        )
        if not history_due:
            continue

        train_denominator = sum(count for _, count in updates_since_validation)
        train_interval_loss = sum(value * count for value, count in updates_since_validation) / max(train_denominator, 1)
        validation_loss = (
            _evaluate_fixed_validation(model, validation_loader, objective, config, device)
            if validation_loader is not None else float("nan")
        )
        check_elapsed = time.perf_counter() - start_time
        if branch == "held_out" and validation_loss < best_raw_value:
            best_raw_value = validation_loss
            best_raw_step = optimizer_steps
            best_raw_state = _state_cpu(model)

        eligible_for_patience = branch == "held_out" and optimizer_steps >= schedule.min_steps
        if eligible_for_patience:
            if meaningful_reference is None:
                meaningful_reference = validation_loss
                meaningful_step = optimizer_steps
                stale_checks = 0
            else:
                relative_gain = (meaningful_reference - validation_loss) / max(abs(meaningful_reference), 1e-12)
                if relative_gain >= schedule.relative_min_delta:
                    meaningful_reference = validation_loss
                    meaningful_step = optimizer_steps
                    stale_checks = 0
                else:
                    stale_checks += 1

        history.append({
            "iteration": optimizer_steps,
            "train_loss": train_interval_loss,
            "optimizer_step": optimizer_steps,
            "train_loss_interval": train_interval_loss,
            "validation_loss": validation_loss,
            "validation_queries": validation_evaluated,
            "effective_candidates_per_anchor": validation_summary["effective_candidates_per_anchor"],
            "stale_validation_checks": stale_checks,
            "early_stopping_eligible": eligible_for_patience,
        })
        seconds_per_update = check_elapsed / max(optimizer_steps, 1)
        _append_progress(progress_path, {
            "run_id": config.run_label,
            "phase": "neural_fit_validation",
            "model": model_name,
            "objective": objective_name,
            "subject_or_population": subject,
            "optimizer_step": optimizer_steps,
            "branch": branch,
            "max_steps": max_updates,
            "validation_checks": len(history),
            "validation_loss": validation_loss,
            "elapsed_minutes_this_fit": check_elapsed / 60.0,
            "estimated_remaining_minutes_to_max": seconds_per_update * (max_updates - optimizer_steps) / 60.0,
            "stop_reason_so_far": "patience_exhausted" if stale_checks >= schedule.patience and eligible_for_patience else "running",
        })
        if len(history) == 1 or len(history) % 5 == 0 or optimizer_steps == schedule.max_steps:
            print(
                f"[{config.run_label}] {branch} {model_label}: step {optimizer_steps}/{max_updates}; "
                f"elapsed {check_elapsed / 60.0:.1f} min; "
                f"estimated remaining to cap {seconds_per_update * (max_updates - optimizer_steps) / 60.0:.1f} min",
                flush=True,
            )
        updates_since_validation.clear()
        if eligible_for_patience and stale_checks >= schedule.patience and optimizer_steps < max_updates:
            stop_reason = "patience_exhausted"
            break
        if optimizer_steps == max_updates:
            stop_reason = "max_steps_reached" if branch == "held_out" else "fixed_N_best_full_sample"

    elapsed = time.perf_counter() - start_time
    if branch == "held_out" and best_raw_state is None:
        raise RuntimeError("no validation checkpoint was created")
    stopping_state = _state_cpu(model)
    final_parameters = _parameter_vector(model)
    total_parameter_delta = float(torch.linalg.vector_norm(final_parameters - initial_parameters))
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0

    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best_validation_checkpoint.pt"
    stop_path = model_dir / "stopping_checkpoint.pt"
    selected_state = best_raw_state if best_raw_state is not None else stopping_state
    checkpoint_payload = {
        "model_name": model_name,
        "objective": objective_name,
        "branch": branch,
        "subject_or_population": subject,
        "state_dict": selected_state,
        "selected_validation_step": best_raw_step if branch == "held_out" else None,
        "full_sample_updates": full_sample_updates,
    }
    if branch == "held_out":
        torch.save(checkpoint_payload, best_path)
    torch.save({
        "model_name": model_name,
        "objective": objective_name,
        "full_sample_updates": full_sample_updates,
        "branch": branch,
        "subject_or_population": subject,
        "state_dict": stopping_state,
        "optimizer_steps": optimizer_steps,
        "stop_reason": stop_reason,
    }, stop_path)
    torch.save(checkpoint_payload, checkpoint)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        columns = (
            "iteration", "train_loss", "optimizer_step", "train_loss_interval", "validation_loss",
            "validation_queries", "effective_candidates_per_anchor",
            "stale_validation_checks", "early_stopping_eligible",
        )
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    _json_write(config_path, {
        "config": legacy._public_config(config),
        "schedule": asdict(schedule),
        "branch": branch,
        "full_sample_updates": full_sample_updates,
        "model": model_name,
        "objective": objective_name,
        "global_batch_size": config.batch_size,
        "microbatch_size": None,
        "block_size": None,
        "validation": validation_summary,
    })
    compute = {
        "training_seconds": elapsed,
        "optimizer_steps": optimizer_steps,
        "max_steps": max_updates,
        "best_validation_step": best_raw_step if branch == "held_out" else None,
        "best_validation_loss": best_raw_value if branch == "held_out" else None,
        "meaningful_improvement_reference_step": meaningful_step,
        "meaningful_improvement_reference_loss": meaningful_reference,
        "stopping_step": optimizer_steps,
        "stopping_reason": stop_reason,
        "validation_checks": len(history) if branch == "held_out" else 0,
        "mean_seconds_per_update": elapsed / max(optimizer_steps, 1),
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "n_optimizer_parameters": sum(parameter.numel() for parameter in optimizer_parameters),
        "n_fit_queries": len(train_indices),
        "train_queries_available": len(train_indices),
        "train_queries_used_per_pass": len(train_loader) * config.batch_size,
        "train_tail_queries_dropped_per_pass": len(train_indices) - len(train_loader) * config.batch_size,
        "n_fit_triplets": len(triplet_dataset) if triplet_dataset is not None else None,
        "validation_summary": validation_summary,
        "training_batch_size_histogram": {str(key): value for key, value in sorted(batch_sizes.items())},
        "training_effective_candidate_histogram": {str(key): value for key, value in sorted(candidate_counts.items())},
        "first_batch_shapes": first_batch_shapes,
        "first_gradient_l2_norm": first_gradient_norm,
        "parameter_delta_after_first_step_l2": first_step_parameter_delta,
        "parameter_delta_after_training_l2": total_parameter_delta,
        "peak_cuda_memory_allocated_bytes": int(peak_allocated),
        "peak_cuda_memory_reserved_bytes": int(peak_reserved),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "candidate_count_formula": validation_summary["effective_candidates_per_anchor"],
        "validation_estimator": "fixed seeded full batches; incomplete tail excluded; sample-weighted over queries",
    }
    _json_write(model_dir / "compute.json", compute)
    provenance = _provenance_record(
        project_root, config, schedule, subject=subject, model_name=model_name,
        objective=objective, branch=branch, device=device,
        checkpoint=best_path if branch == "held_out" else checkpoint,
    )
    fit_artifacts = {
        path.name: _hash(path)
        for path in (
            checkpoint, best_path, stop_path, history_path,
            model_dir / "config.json", model_dir / "compute.json",
        )
        if path.is_file()
    }
    provenance.update({
        "split_sha256": _hash(split_path),
        "window_cache_sha256": _hash(windows_path),
        "best_checkpoint_sha256": _hash(best_path) if best_path.is_file() else None,
        "stopping_checkpoint_sha256": _hash(stop_path),
        "optimizer_updates": optimizer_steps,
        "artifact_sha256": fit_artifacts,
        "artifact_ids": fit_artifacts,
        "parent_artifact_ids": {
            "raw_source": _hash(project_root / config.source_relative),
            "windows": _hash(windows_path),
            "split": _hash(split_path),
        },
        "effective_candidate_counts_seen_in_training": compute["training_effective_candidate_histogram"],
    })
    _json_write(model_dir / "provenance.json", provenance)
    del model, optimizer, training_iterator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return checkpoint


def _trial_prototypes(
    embedding: np.ndarray,
    artifacts: dict[str, np.ndarray],
    split: dict[str, list[int]],
    *,
    sample_times: np.ndarray,
    selected_trials: np.ndarray | None = None,
) -> dict[str, Any]:
    selected_trials = np.asarray(
        split["test"] if selected_trials is None else selected_trials,
        dtype=np.int64,
    )
    trial_ids = artifacts["trial_id"]
    time_ids = artifacts["time_id"]
    is_selected = np.isin(trial_ids, selected_trials)
    prototypes = []
    positions = []
    velocities = []
    directions = []
    used_trials = []
    for trial in selected_trials:
        mask = is_selected & (trial_ids == trial) & np.isin(time_ids, sample_times)
        if int(mask.sum()) != len(sample_times):
            continue
        prototypes.append(np.mean(embedding[mask], axis=0))
        positions.append(np.mean(artifacts["position"][mask], axis=0))
        velocities.append(np.mean(artifacts["velocity"][mask], axis=0))
        directions.append(int(artifacts["target"][np.flatnonzero(mask)[0]]))
        used_trials.append(int(trial))
    return {
        "embedding": np.asarray(prototypes),
        "position": np.asarray(positions),
        "velocity": np.asarray(velocities),
        "direction": np.asarray(directions, dtype=np.int64),
        "trial_id": np.asarray(used_trials, dtype=np.int64),
    }


def _rsa_correlations(embedding: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    if len(embedding) < 4 or len(reference) != len(embedding):
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {
        "pearson": distance_geometry_correlation(
            embedding, reference, metric="euclidean", method="pearson", max_pairs=None,
        ),
        "spearman": distance_geometry_correlation(
            embedding, reference, metric="euclidean", method="spearman", max_pairs=None,
        ),
    }


def _bootstrap_rsa(
    prototypes: dict[str, np.ndarray],
    reference_name: str,
    *,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, dict[str, tuple[float, float]]]:
    embedding = prototypes["embedding"]
    reference = prototypes[reference_name]
    n_trials = len(embedding)
    rng = np.random.default_rng(random_state)
    distributions = {method: [] for method in ("pearson", "spearman")}
    for _ in range(n_bootstrap):
        sample = rng.integers(0, n_trials, size=n_trials)
        values = _rsa_correlations(embedding[sample], reference[sample])
        for method, value in values.items():
            if np.isfinite(value):
                distributions[method].append(value)
    return {
        method: (
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ) if values else (float("nan"), float("nan"))
        for method, values in distributions.items()
    }


def compute_representation_geometry(
    project_root: str | Path,
    config: legacy.RealMonkeyConfig,
    *,
    embedding_paths: list[tuple[str, str, Path]],
    windows_path: Path,
    split_path: Path,
    output_dir: Path,
    branch: str = "held_out",
    bootstrap_replicates: int = 200,
) -> Path:
    """Compute trial-prototype RSA separately from decoding and CKA."""
    if branch not in {"held_out", "full_sample"}:
        raise ValueError("branch must be 'held_out' or 'full_sample'")
    project_root = Path(project_root).resolve()
    windows = np.load(windows_path, allow_pickle=False)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    evaluation_scope = (
        "held_out_representation_generalization"
        if branch == "held_out"
        else "descriptive_in_sample_representation"
    )
    evaluation_trials = (
        np.asarray(split["test"], dtype=np.int64)
        if branch == "held_out"
        else np.unique(windows["trial_id"]).astype(np.int64)
    )
    train_trials = np.asarray(split["train"], dtype=np.int64)
    sample_times = np.linspace(40, 569, 10, dtype=np.int64)
    scaler_trials = train_trials if branch == "held_out" else evaluation_trials
    scaler_mask = np.isin(windows["trial_id"], scaler_trials) & np.isin(windows["time_id"], sample_times)
    behavior_scalers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in ("position", "velocity"):
        mean = windows[key][scaler_mask].mean(axis=0)
        scale = windows[key][scaler_mask].std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        behavior_scalers[key] = (mean, scale)

    rows: list[dict[str, Any]] = []
    for model_name, objective_name, embedding_path in embedding_paths:
        values = np.load(embedding_path, allow_pickle=False)
        for representation in ("raw", "unit"):
            prototypes = _trial_prototypes(
                values[f"embedding_{representation}"], values, split,
                sample_times=sample_times, selected_trials=evaluation_trials,
            )
            if len(prototypes["trial_id"]) < 10:
                raise RuntimeError("too few complete trial prototypes for RSA")
            direction_angle = 2 * np.pi * prototypes["direction"] / config.n_conditions
            direction_reference = np.column_stack((np.cos(direction_angle), np.sin(direction_angle)))
            references = {
                "circular_movement_direction": direction_reference,
                "position_geometry": (prototypes["position"] - behavior_scalers["position"][0]) / behavior_scalers["position"][1],
                "velocity_geometry": (prototypes["velocity"] - behavior_scalers["velocity"][0]) / behavior_scalers["velocity"][1],
            }
            for reference_name, reference in references.items():
                values_by_method = _rsa_correlations(prototypes["embedding"], reference)
                intervals = _bootstrap_rsa(
                    {**prototypes, reference_name: reference}, reference_name,
                    n_bootstrap=bootstrap_replicates,
                    random_state=config.split_seed + 311,
                )
                for method, value in values_by_method.items():
                    low, high = intervals[method]
                    rows.append({
                        "branch": branch,
                        "evaluation_scope": evaluation_scope,
                        "model": model_name,
                        "objective": objective_name,
                        "representation": representation,
                        "category": "representational_geometry",
                        "metric": f"rsa_{method}",
                        "reference_geometry": reference_name,
                        "value": value,
                        "trial_bootstrap_ci_low": low,
                        "trial_bootstrap_ci_high": high,
                        "bootstrap_replicates": bootstrap_replicates,
                        "n_evaluation_trials": len(prototypes["trial_id"]),
                        "n_pairwise_trial_distances": len(prototypes["trial_id"]) * (len(prototypes["trial_id"]) - 1) // 2,
                        "prototype_sample_times": sample_times.tolist(),
                        "interpretation": "geometry association, not latent recovery or task decoding",
                    })
            centered = prototypes["embedding"] - prototypes["embedding"].mean(axis=0, keepdims=True)
            covariance = centered.T @ centered / max(len(centered) - 1, 1)
            eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
            denominator = float(np.square(eigenvalues).sum())
            participation_ratio = float(eigenvalues.sum() ** 2 / denominator) if denominator > 1e-20 else float("nan")
            rows.append({
                "branch": branch,
                "evaluation_scope": evaluation_scope,
                "model": model_name,
                "objective": objective_name,
                "representation": representation,
                "category": "embedding_geometry_diagnostic",
                "metric": "participation_ratio",
                "reference_geometry": f"{branch}_trial_prototype_covariance",
                "value": participation_ratio,
                "trial_bootstrap_ci_low": float("nan"),
                "trial_bootstrap_ci_high": float("nan"),
                "bootstrap_replicates": 0,
                "n_evaluation_trials": len(prototypes["trial_id"]),
                "n_pairwise_trial_distances": len(prototypes["trial_id"]) * (len(prototypes["trial_id"]) - 1) // 2,
                "prototype_sample_times": sample_times.tolist(),
                "interpretation": (
                    "effective dimensions of trial-averaged embedding; descriptive full-sample refit"
                    if branch == "full_sample"
                    else "effective dimensions of held-out test-trial averaged embedding"
                ),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "representation_geometry.csv"
    columns = list(rows[0]) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    figure_path = _plot_rsa_summary(rows, output_dir.parent.parent / "stage06_figures" / branch, branch=branch)
    input_embeddings = {
        f"{model_name}_{objective_name}": path
        for model_name, objective_name, path in embedding_paths
    }
    figure_payload = {
        "setting": config.setting_name,
        "population": config.population_name,
        "architecture": "multiple",
        "objective": "multiple",
        "training_seed": config.training_seed,
        "channel_split_seed": config.channel_split_seed,
        "branch": branch,
        "evaluation_scope": evaluation_scope,
        "split_trial_ids": split,
        "config": legacy._public_config(config),
        "artifact_sha256": {output_path.name: _hash(output_path)},
        "artifact_ids": {output_path.name: _hash(output_path)},
        "parent_artifacts": {
            "windows": {"path": str(windows_path), "sha256": _hash(windows_path)},
            "split": {"path": str(split_path), "sha256": _hash(split_path)},
            "embeddings": {
                label: {"path": str(path), "sha256": _hash(path)}
                for label, path in input_embeddings.items()
            },
        },
    }
    _json_write(output_path.with_name("representation_geometry_provenance.json"), figure_payload)
    if figure_path.is_file():
        figure_payload["artifact_sha256"] = {figure_path.name: _hash(figure_path)}
        figure_payload["artifact_ids"] = {figure_path.name: _hash(figure_path)}
        _json_write(figure_path.with_name(f"{figure_path.stem}_provenance.json"), figure_payload)
    return output_path


def _plot_rsa_summary(rows: list[dict[str, Any]], figure_dir: Path, *, branch: str = "held_out") -> Path:
    subset = [row for row in rows if row["metric"] == "rsa_spearman"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    output = figure_dir / "behavioral_geometry_rsa_spearman.png"
    if not subset:
        return output
    references = list(dict.fromkeys(row["reference_geometry"] for row in subset))
    objectives = list(dict.fromkeys(row["objective"] for row in subset))
    fig, axes = plt.subplots(len(references), 1, figsize=(12, 3.6 * len(references)), squeeze=False)
    for axis, reference in zip(axes[:, 0], references):
        chosen = [row for row in subset if row["reference_geometry"] == reference]
        labels = [f"{row['model']}\n{row['objective']}\n{row['representation']}" for row in chosen]
        values = [row["value"] for row in chosen]
        low = [row["value"] - row["trial_bootstrap_ci_low"] for row in chosen]
        high = [row["trial_bootstrap_ci_high"] - row["value"] for row in chosen]
        x = np.arange(len(chosen))
        axis.bar(x, values, color=["tab:blue" if row["representation"] == "raw" else "tab:orange" for row in chosen])
        if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
            axis.errorbar(x, values, yerr=[low, high], fmt="none", ecolor="black", capsize=2)
        axis.set_xticks(x, labels, rotation=45, ha="right")
        axis.set_ylabel("Spearman RSA")
        axis.set_title(reference.replace("_", " "))
        axis.grid(axis="y", alpha=0.25)
    scope_label = "test trials" if branch == "held_out" else "all trials; descriptive in-sample"
    fig.suptitle(f"Trial-level representational geometry ({scope_label}; bootstrap by trial)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _bootstrap_trial_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    trial_id: np.ndarray,
    class_label: np.ndarray,
    metric_name: str,
    *,
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    trials = np.unique(trial_id)
    grouped = {
        int(label): [int(trial) for trial in trials if np.any((trial_id == trial) & (class_label == label))]
        for label in np.unique(class_label)
    }
    estimates = []
    for _ in range(n_bootstrap):
        sampled_trials = []
        for candidates in grouped.values():
            sampled_trials.extend(rng.choice(candidates, size=len(candidates), replace=True).tolist())
        indices = np.concatenate([np.flatnonzero(trial_id == trial) for trial in sampled_trials])
        y = truth[indices]
        p = prediction[indices]
        if metric_name == "balanced_accuracy":
            value = balanced_accuracy_score(y, p)
        elif metric_name == "r2":
            value = r2_score(y, p, multioutput="variance_weighted")
        elif metric_name == "mae":
            value = mean_absolute_error(y, p)
        else:
            raise ValueError(metric_name)
        if np.isfinite(value):
            estimates.append(float(value))
    if not estimates:
        return float("nan"), float("nan")
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def compute_trial_bootstrap_decoding(
    config: legacy.RealMonkeyConfig,
    *,
    model_name: str,
    objective_name: str,
    embedding_path: Path,
    windows_path: Path,
    split_path: Path,
    output_dir: Path,
    branch: str = "held_out",
    bootstrap_replicates: int = 200,
) -> Path:
    """Add trial-cluster intervals to task and kinematic probe point estimates."""
    if branch not in {"held_out", "full_sample"}:
        raise ValueError("branch must be 'held_out' or 'full_sample'")
    evaluation_scope = (
        "held_out_representation_generalization"
        if branch == "held_out"
        else "descriptive_in_sample_representation"
    )
    values = np.load(embedding_path, allow_pickle=False)
    windows = np.load(windows_path, allow_pickle=False)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train = values["split"] == "train"
    validation = values["split"] == "validation"
    test = values["split"] == "test"
    test_trials = values["trial_id"][test]
    test_labels = values["target"][test]
    rows: list[dict[str, Any]] = []
    for representation in ("raw", "unit"):
        embedding = values[f"embedding_{representation}"]
        train_z, val_z, test_z = embedding[train], embedding[validation], embedding[test]
        y_train, y_val, y_test = values["target"][train], values["target"][validation], values["target"][test]
        classifier_candidates = []
        for c_value in (0.1, 1.0, 10.0):
            candidate = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c_value, solver="lbfgs", max_iter=1000, random_state=config.split_seed),
            )
            candidate.fit(train_z, y_train)
            score = balanced_accuracy_score(y_val, candidate.predict(val_z))
            classifier_candidates.append((score, candidate, c_value))
        _, classifier, selected_c = max(classifier_candidates, key=lambda item: item[0])
        class_prediction = classifier.predict(test_z)
        for metric_name, statistic in (("condition_balanced_accuracy", "balanced_accuracy"),):
            low, high = _bootstrap_trial_metrics(
                y_test, class_prediction, test_trials, y_test,
                statistic, n_bootstrap=bootstrap_replicates,
                random_state=config.split_seed + 1701,
            )
            rows.append({
                "model": model_name, "objective": objective_name,
                "representation": representation, "metric": metric_name,
                "value": float(balanced_accuracy_score(y_test, class_prediction)),
                "trial_bootstrap_ci_low": low, "trial_bootstrap_ci_high": high,
                "bootstrap_replicates": bootstrap_replicates,
                "selected_hyperparameter": selected_c,
                "selection_split": "validation", "evaluation_split": "test",
            })

        progress_truth = values["progress"][test]
        progress_candidates = []
        for alpha in (0.1, 1.0, 10.0):
            candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            candidate.fit(train_z, values["progress"][train])
            progress_candidates.append((r2_score(values["progress"][validation], candidate.predict(val_z)), candidate, alpha))
        _, progress_decoder, selected_alpha = max(progress_candidates, key=lambda item: item[0])
        progress_prediction = progress_decoder.predict(test_z)
        for metric_name, statistic, point in (
            ("normalized_epoch_time_r2", "r2", r2_score(progress_truth, progress_prediction)),
            ("normalized_epoch_time_mae", "mae", mean_absolute_error(progress_truth, progress_prediction)),
        ):
            low, high = _bootstrap_trial_metrics(
                progress_truth, progress_prediction, test_trials, test_labels,
                statistic, n_bootstrap=bootstrap_replicates,
                random_state=config.split_seed + 1702,
            )
            rows.append({
                "model": model_name, "objective": objective_name,
                "representation": representation, "metric": metric_name,
                "value": float(point), "trial_bootstrap_ci_low": low,
                "trial_bootstrap_ci_high": high,
                "bootstrap_replicates": bootstrap_replicates,
                "selected_hyperparameter": selected_alpha,
                "selection_split": "validation", "evaluation_split": "test",
            })

        test_indices = np.flatnonzero(test)
        for behavior in ("position", "velocity"):
            target = values[behavior]
            candidates = []
            for alpha in (0.1, 1.0, 10.0):
                candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                candidate.fit(train_z, target[train])
                score = r2_score(target[validation], candidate.predict(val_z), multioutput="variance_weighted")
                candidates.append((score, candidate, alpha))
            _, decoder, selected_alpha = max(candidates, key=lambda item: item[0])
            prediction = decoder.predict(test_z)
            truth = target[test]
            for metric_name, statistic, point in (
                (f"{behavior}_r2", "r2", r2_score(truth, prediction, multioutput="variance_weighted")),
                (f"{behavior}_mae", "mae", mean_absolute_error(truth, prediction)),
            ):
                low, high = _bootstrap_trial_metrics(
                    truth, prediction, test_trials, test_labels,
                    statistic, n_bootstrap=bootstrap_replicates,
                    random_state=config.split_seed + (1703 if behavior == "position" else 1704),
                )
                rows.append({
                    "model": model_name, "objective": objective_name,
                    "representation": representation, "metric": metric_name,
                    "value": float(point), "trial_bootstrap_ci_low": low,
                    "trial_bootstrap_ci_high": high,
                    "bootstrap_replicates": bootstrap_replicates,
                    "selected_hyperparameter": selected_alpha,
                    "selection_split": "validation", "evaluation_split": "test",
                })

    for row in rows:
        row["branch"] = branch
        row["evaluation_scope"] = evaluation_scope
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_{objective_name}_trial_bootstrap.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _json_write(output_path.with_name(f"{output_path.stem}_provenance.json"), {
        "setting": config.setting_name,
        "population": config.population_name,
        "architecture": model_name,
        "objective": objective_name,
        "training_seed": config.training_seed,
        "channel_split_seed": config.channel_split_seed,
        "branch": branch,
        "evaluation_scope": evaluation_scope,
        "split_trial_ids": json.loads(Path(split_path).read_text(encoding="utf-8")),
        "config": legacy._public_config(config),
        "artifact_sha256": {output_path.name: _hash(output_path)},
        "artifact_ids": {output_path.name: _hash(output_path)},
        "parent_artifacts": {
            "embedding": {"path": str(Path(embedding_path)), "sha256": _hash(Path(embedding_path))},
            "windows": {"path": str(Path(windows_path)), "sha256": _hash(Path(windows_path))},
            "split": {"path": str(Path(split_path)), "sha256": _hash(Path(split_path))},
        },
    })
    return output_path


def _rename_epoch_time_metric(metrics_path: Path) -> None:
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        columns = list(rows[0]) if rows else []
    rename = {
        "progress_r2": "normalized_epoch_time_r2",
        "progress_mae": "normalized_epoch_time_mae",
    }
    for row in rows:
        row["metric"] = rename.get(row.get("metric", ""), row.get("metric", ""))
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    provenance_path = metrics_path.with_name(f"{metrics_path.stem}_provenance.json")
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        digest = _hash(metrics_path)
        provenance.setdefault("artifact_sha256", {})[metrics_path.name] = digest
        provenance.setdefault("artifact_ids", {})[metrics_path.name] = digest
        _json_write(provenance_path, provenance)


def _write_run_manifest(
    project_root: Path,
    config: legacy.RealMonkeyConfig,
    schedule: ValidationSchedule,
    *,
    status: str,
    completed: list[str],
    parent_run_label: str,
) -> Path:
    root = legacy._run_root(project_root, config)
    source = project_root / config.source_relative
    split_path = root / "stage01_data" / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8")) if split_path.is_file() else None
    windows_path = root / "stage02_windows" / "windows.npz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = {
        "run_id": config.run_label,
        "status": status,
        "completed_fits": completed,
        "parent_cache_run": None,
        "input_preparation": "rebuilt from the configured raw source in this run; no parent cache dependency",
        "source_data_sha256": _hash(source),
        "configuration": legacy._public_config(config),
        "setting": config.setting_name,
        "population": config.population_name,
        "channel_split_seed": config.channel_split_seed,
        "channel_indices": list(config.channel_indices) if config.channel_indices is not None else None,
        "training_seed": config.training_seed,
        "imposed_shift_bins": config.imposed_shift_bins,
        "split_trial_ids": split,
        "split_sha256": _hash(split_path) if split_path.is_file() else None,
        "parent_artifacts": {
            "raw_source_sha256": _hash(source),
            "windows_sha256": _hash(windows_path) if windows_path.is_file() else None,
            "split_sha256": _hash(split_path) if split_path.is_file() else None,
        },
        "validation_schedule": asdict(schedule),
        "branches": ["held_out", "full_sample"],
        "git": _git_provenance(project_root),
        "source_file_sha256": _source_provenance(project_root),
        "python": __import__("sys").version,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "scientific_scope": {
            "latent_recovery": "not available: no monkey ground-truth latent Z",
            "full_sample": "descriptive in-sample representation; not a generalization estimate",
            "task_decoding": "separate from representation geometry",
            "natural_lag": "not known; controlled perturbation is a separate run",
        },
    }
    manifest_path = root / "run_manifest.json"
    _json_write(manifest_path, payload)
    return manifest_path


def _validate_cached_metric_embedding(path: Path) -> int:
    """Check cached embedding shape and row metadata before evaluation-only work."""
    with np.load(path, allow_pickle=False) as values:
        required = ("embedding_raw", "embedding_unit", "trial_id", "time_id", "split", "target", "progress", "position", "velocity")
        missing = [key for key in required if key not in values.files]
        if missing:
            raise ValueError(f"cached embedding is missing fields {missing}: {path}")
        n_rows = len(values["trial_id"])
        if values["embedding_raw"].shape != (n_rows, 3) or values["embedding_unit"].shape != (n_rows, 3):
            raise ValueError(f"cached embedding must have matching N x 3 raw/unit arrays: {path}")
        for key in required[2:]:
            if len(values[key]) != n_rows:
                raise ValueError(f"cached embedding row count mismatch for {key}: {path}")
        if "valid_mask" in values.files and len(values["valid_mask"]) != n_rows:
            raise ValueError(f"cached embedding validity mask row count mismatch: {path}")
        if not np.all(np.isfinite(values["embedding_raw"])) or not np.all(np.isfinite(values["embedding_unit"])):
            raise ValueError(f"cached embedding has non-finite coordinates: {path}")
        return n_rows


def evaluate_cached_real_metrics(
    project_root: str | Path,
    config: legacy.RealMonkeyConfig,
    *,
    branches: tuple[str, ...] = ("held_out", "full_sample"),
    rsa_bootstrap_replicates: int = 200,
    decoding_bootstrap_replicates: int = 200,
) -> dict[str, Any]:
    """Run missing Stage-5 metrics from frozen cached embeddings only.

    This function never fits a model or transforms windows. Existing metric
    files are preserved; missing files are produced from their saved parent
    embeddings and the current cached windows/split.
    """
    project_root = Path(project_root).resolve()
    root = legacy._run_root(project_root, config)
    windows_path = root / "stage02_windows" / "windows.npz"
    split_path = root / "stage01_data" / "split.json"
    if not windows_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(f"cached real windows/split missing for metric-only evaluation: {root}")
    results: dict[str, Any] = {"run_id": config.run_label, "run_root": root, "branches": {}}
    for branch in branches:
        if branch not in {"held_out", "full_sample"}:
            raise ValueError("branch must be 'held_out' or 'full_sample'")
        stage = root / "stage05_metrics" / branch
        created: list[str] = []
        reused: list[str] = []

        pca_embedding = root / "stage04_embeddings" / branch / "pca_none.npz"
        if not pca_embedding.is_file():
            raise FileNotFoundError(f"cached {branch} PCA embedding missing: {pca_embedding}")
        _validate_cached_metric_embedding(pca_embedding)
        pca_metric = stage / "pca_none.csv"
        if pca_metric.is_file():
            reused.append(str(pca_metric.relative_to(root)))
        else:
            legacy._evaluate_embedding_artifact(
                project_root, config, embedding_path=pca_embedding,
                model_name="pca", loss_name="none", branch=branch,
            )
            created.append(str(pca_metric.relative_to(root)))

        embedding_paths: list[tuple[str, str, Path]] = []
        for model_name in MODEL_NAMES:
            for objective_name in OBJECTIVE_NAMES:
                embedding_path = root / "stage04_embeddings" / branch / f"{model_name}_{objective_name}.npz"
                checkpoint_path = root / "stage03_models" / branch / f"{model_name}_{objective_name}" / "model.pt"
                if not embedding_path.is_file() or not checkpoint_path.is_file():
                    raise FileNotFoundError(f"cached {branch} model/embedding pair missing: {checkpoint_path}, {embedding_path}")
                _validate_cached_metric_embedding(embedding_path)
                embedding_paths.append((model_name, objective_name, embedding_path))

                metric_path = stage / f"{model_name}_{objective_name}.csv"
                selection_path = stage / f"{model_name}_{objective_name}_decoder_selection.json"
                if metric_path.is_file() and selection_path.is_file():
                    reused.append(str(metric_path.relative_to(root)))
                elif metric_path.exists() or selection_path.exists():
                    raise FileExistsError(f"partial existing evaluation artifacts preserved; inspect before repair: {stage}")
                else:
                    legacy._evaluate_embedding_artifact(
                        project_root, config, embedding_path=embedding_path,
                        model_name=model_name,
                        loss_name=_INTERNAL_OBJECTIVES[objective_name],
                        branch=branch,
                    )
                    _rename_epoch_time_metric(metric_path)
                    created.append(str(metric_path.relative_to(root)))

                bootstrap_path = stage / f"{model_name}_{objective_name}_trial_bootstrap.csv"
                if bootstrap_path.is_file():
                    reused.append(str(bootstrap_path.relative_to(root)))
                else:
                    compute_trial_bootstrap_decoding(
                        config, model_name=model_name, objective_name=objective_name,
                        embedding_path=embedding_path, windows_path=windows_path,
                        split_path=split_path, output_dir=stage, branch=branch,
                        bootstrap_replicates=decoding_bootstrap_replicates,
                    )
                    created.append(str(bootstrap_path.relative_to(root)))

        geometry_path = stage / "representation_geometry.csv"
        if geometry_path.is_file():
            reused.append(str(geometry_path.relative_to(root)))
        else:
            compute_representation_geometry(
                project_root, config, embedding_paths=embedding_paths,
                windows_path=windows_path, split_path=split_path,
                output_dir=stage, branch=branch,
                bootstrap_replicates=rsa_bootstrap_replicates,
            )
            created.append(str(geometry_path.relative_to(root)))
        results["branches"][branch] = {"created": created, "reused": reused}
    return results


def run_natural_monkey_suite(
    project_root: str | Path,
    config: legacy.RealMonkeyConfig | None = None,
    *,
    schedule: ValidationSchedule = DEFAULT_SCHEDULE,
    parent_run_label: str = DEFAULT_PARENT_RUN,
    models: tuple[str, ...] = MODEL_NAMES,
    objectives: tuple[str, ...] = OBJECTIVE_NAMES,
    rsa_bootstrap_replicates: int = 200,
    decoding_bootstrap_replicates: int = 200,
) -> dict[str, Any]:
    """Run one population from raw data through held-out and descriptive stages."""
    project_root = Path(project_root).resolve()
    config = config or canonical_config(max_steps=schedule.max_steps)
    if config.max_iterations != schedule.max_steps:
        raise ValueError("canonical config max_iterations must equal schedule.max_steps")
    prospective_root = legacy._run_root(project_root, config)
    manifest_path = prospective_root / "run_manifest.json"
    old_manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            old_manifest.get("configuration") != legacy._public_config(config)
            or old_manifest.get("validation_schedule") != asdict(schedule)
        ):
            raise FileExistsError(f"run ID exists with a different protocol; choose a new run label: {prospective_root}")
    windows_path = legacy.stage_windows(project_root, config)
    root = prospective_root
    inputs = {
        "root": root,
        "windows": windows_path,
        "split": root / "stage01_data" / "split.json",
        "data": root / "stage01_data" / "data.npz",
        "cache_reuse": "rebuilt_or_resumed_from_raw_stage_cache",
    }
    completed = list(dict.fromkeys(old_manifest.get("completed_fits", [])))
    total_fit_units = 2 * len(models) * len(objectives)
    suite_start = time.perf_counter()
    fit_seconds: list[float] = []
    _write_run_manifest(project_root, config, schedule, status="running", completed=completed, parent_run_label=parent_run_label)

    pca_block_start = time.perf_counter()
    pca_path = legacy.fit_pca_embeddings(project_root, config, branch="held_out")
    pca_metric = legacy.evaluate_pca(project_root, config, branch="held_out")
    legacy.plot_pca_embeddings(project_root, config, branch="held_out")
    legacy.fit_pca_embeddings(project_root, config, branch="full_sample")
    legacy.plot_pca_embeddings(project_root, config, branch="full_sample")
    _append_progress(root / "progress.jsonl", {
        "run_id": config.run_label,
        "phase": "pca_and_upstream_outputs_complete",
        "elapsed_block_minutes": (time.perf_counter() - pca_block_start) / 60.0,
        "elapsed_total_minutes": (time.perf_counter() - suite_start) / 60.0,
        "input_source": config.source_relative,
        "pca_source": "fit from current run training trials",
    })
    embedding_paths: list[tuple[str, str, Path]] = []
    for model_name in models:
        for objective_name in objectives:
            block_start = time.perf_counter()
            checkpoint = fit_validated_neural_model(
                project_root, config, model_name=model_name,
                objective_name=objective_name, schedule=schedule,
            )
            objective = _INTERNAL_OBJECTIVES[objective_name]
            embedding_path = legacy.transform_neural_embeddings(
                project_root, config, model_name=model_name,
                loss_name=objective, branch="held_out",
                checkpoint_path=checkpoint,
            )
            metric_path = legacy.evaluate_neural(
                project_root, config, model_name=model_name,
                loss_name=objective, branch="held_out",
                checkpoint_path=checkpoint,
            )
            _rename_epoch_time_metric(metric_path)
            legacy.plot_neural_embeddings(
                project_root, config, model_name=model_name,
                loss_name=objective, branch="held_out",
                checkpoint_path=checkpoint,
            )
            legacy.plot_training_history(
                project_root, config, model_name=model_name,
                loss_name=objective, branch="held_out",
            )
            compute_trial_bootstrap_decoding(
                config, model_name=model_name, objective_name=objective_name,
                embedding_path=embedding_path, windows_path=inputs["windows"],
                split_path=inputs["split"],
                output_dir=root / "stage05_metrics" / "held_out",
                bootstrap_replicates=decoding_bootstrap_replicates,
            )
            embedding_paths.append((model_name, objective_name, embedding_path))
            best_step = int(json.loads(checkpoint.parent.joinpath("compute.json").read_text(encoding="utf-8"))["best_validation_step"])
            full_checkpoint = fit_validated_neural_model(
                project_root, config, model_name=model_name,
                objective_name=objective_name, schedule=schedule,
                branch="full_sample", full_sample_updates=best_step,
            )
            full_embedding_path = legacy.transform_neural_embeddings(
                project_root, config, model_name=model_name,
                loss_name=objective, branch="full_sample",
                checkpoint_path=full_checkpoint,
            )
            legacy.plot_neural_embeddings(
                project_root, config, model_name=model_name,
                loss_name=objective, branch="full_sample",
                checkpoint_path=full_checkpoint,
            )
            legacy.plot_training_history(
                project_root, config, model_name=model_name,
                loss_name=objective, branch="full_sample",
            )
            heldout_unit = f"{model_name}/{objective_name}"
            full_sample_unit = f"full_sample/{model_name}/{objective_name}@{best_step}"
            for unit in (heldout_unit, full_sample_unit):
                if unit not in completed:
                    completed.append(unit)
            fit_compute_path = checkpoint.parent / "compute.json"
            if fit_compute_path.is_file():
                fit_seconds.append(float(json.loads(fit_compute_path.read_text(encoding="utf-8")).get("training_seconds", 0.0)))
            remaining_fits = max(total_fit_units - len(completed), 0)
            estimated_remaining_minutes = float(np.mean(fit_seconds)) * remaining_fits / 60.0 if fit_seconds and remaining_fits else 0.0
            _append_progress(root / "progress.jsonl", {
                "run_id": config.run_label,
                "phase": "fit_and_downstream_outputs_complete",
                "fit": f"{model_name}/{objective_name}",
                "completed_fits": len(completed),
                "total_fits": total_fit_units,
                "full_sample_updates": best_step,
                "elapsed_block_minutes": (time.perf_counter() - block_start) / 60.0,
                "elapsed_total_minutes": (time.perf_counter() - suite_start) / 60.0,
                "estimated_remaining_training_minutes_from_measured_mean": estimated_remaining_minutes,
                "estimate_excludes_future_metrics_figures_and_rsa": True,
            })
            print(
                f"[{config.run_label}] completed {len(completed)}/{total_fit_units} fits; "
                f"block {(time.perf_counter() - block_start) / 60.0:.1f} min; "
                f"total {(time.perf_counter() - suite_start) / 60.0:.1f} min; "
                f"estimated remaining training {estimated_remaining_minutes:.1f} min",
                flush=True,
            )
            _write_run_manifest(
                project_root, config, schedule, status="running",
                completed=completed, parent_run_label=parent_run_label,
            )
            del checkpoint, embedding_path, full_checkpoint, full_embedding_path
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    geometry_path = compute_representation_geometry(
        project_root, config, embedding_paths=embedding_paths,
        windows_path=inputs["windows"], split_path=inputs["split"],
        output_dir=root / "stage05_metrics" / "held_out",
        bootstrap_replicates=rsa_bootstrap_replicates,
    )
    full_sample_metrics = evaluate_cached_real_metrics(
        project_root, config, branches=("full_sample",),
        rsa_bootstrap_replicates=rsa_bootstrap_replicates,
        decoding_bootstrap_replicates=decoding_bootstrap_replicates,
    )
    _append_progress(root / "progress.jsonl", {
        "run_id": config.run_label,
        "phase": "full_sample_descriptive_metrics_complete",
        "created_artifacts": sum(len(item["created"]) for item in full_sample_metrics["branches"].values()),
        "elapsed_total_minutes": (time.perf_counter() - suite_start) / 60.0,
    })
    result = {
        "run_id": config.run_label,
        "run_root": root,
        "reused_cache": inputs["cache_reuse"],
        "pca_embedding": pca_path,
        "pca_metrics": pca_metric,
        "neural_fits": completed,
        "geometry_metrics": geometry_path,
        "full_sample_metrics": full_sample_metrics,
    }
    _write_run_manifest(project_root, config, schedule, status="complete", completed=completed, parent_run_label=parent_run_label)
    return result


def run_training_diagnostics(
    project_root: str | Path,
    *,
    config: legacy.RealMonkeyConfig | None = None,
    diagnostic_dir: str | Path | None = None,
    overfit_steps: int = 200,
) -> Path:
    """Run fixed-batch overfit and deterministic easy-temporal diagnostics."""
    project_root = Path(project_root).resolve()
    config = config or canonical_config()
    legacy._seed_real_training(config.training_seed)
    inputs = reuse_natural_input_cache(project_root, config)
    dataset, values = legacy._load_real_window_dataset(inputs["windows"])
    split = json.loads(inputs["split"].read_text(encoding="utf-8"))
    train_trials = np.asarray(split["train"], dtype=np.int64)
    train_indices = np.flatnonzero(np.isin(values["trial_id"], train_trials) & values["lag_valid"].astype(bool)).tolist()
    sampler = legacy.CEBRATripletWindowDataset(
        dataset, train_indices, offset=config.cebra_time_offset,
        valid_mask=torch.as_tensor(values["lag_valid"].astype(bool)),
    )
    generator = torch.Generator().manual_seed(config.training_seed + 71)
    sampler.resample(generator=generator)
    anchor_rng = np.random.default_rng(config.training_seed + 72)
    sampled_anchor_positions = anchor_rng.choice(
        len(sampler), size=config.batch_size, replace=False,
    )
    sampled_anchor_indices = sampler.anchor_indices[torch.as_tensor(sampled_anchor_positions, dtype=torch.long)]
    anchor_trial_values = values["trial_id"][sampled_anchor_indices.numpy()]
    anchor_time_values = values["time_id"][sampled_anchor_indices.numpy()]
    anchor_trial_counts = {
        str(int(trial)): int(count)
        for trial, count in zip(*np.unique(anchor_trial_values, return_counts=True))
    }
    diagnostic_subset = Subset(sampler, sampled_anchor_positions.tolist())
    diagnostic_loader = DataLoader(
        diagnostic_subset, batch_size=config.batch_size, shuffle=False, drop_last=True,
    )
    batch = next(iter(diagnostic_loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = legacy._make_real_model(
        "transformer", int(values["X_windows"].shape[-1]), config, normalize=True,
    ).to(device)
    batch = _move_batch(batch, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    initial = _parameter_vector(model)
    model.eval()
    with torch.no_grad():
        initial_reference = F.normalize(model(batch["reference_x"]), dim=-1)
        initial_positive = F.normalize(model(batch["positive_x"]), dim=-1)
        initial_negative = F.normalize(model(batch["negative_x"]), dim=-1)
        initial_fixed_loss = float(cebra_infonce_loss(
            initial_reference, initial_positive, initial_negative,
            temperature=config.cebra_temperature,
        ).cpu())
        initial_positive_similarity = float((initial_reference * initial_positive).sum(dim=-1).mean().cpu())
        initial_negative_similarity = float((initial_reference @ initial_negative.T).mean().cpu())
    start_loss: float | None = None
    first_gradient: float | None = None
    loss_values: list[float] = []
    for step in range(overfit_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = _batch_loss(model, batch, "cebra_time", config)
        if step == 0:
            start_loss = float(loss.detach().cpu())
        loss.backward()
        if step == 0:
            first_gradient = float(np.sqrt(sum(
                float(parameter.grad.detach().float().norm().cpu()) ** 2
                for parameter in model.parameters() if parameter.grad is not None
            )))
        optimizer.step()
        loss_values.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        final_reference = F.normalize(model(batch["reference_x"]), dim=-1)
        final_positive = F.normalize(model(batch["positive_x"]), dim=-1)
        final_negative = F.normalize(model(batch["negative_x"]), dim=-1)
        final_fixed_loss = float(cebra_infonce_loss(
            final_reference, final_positive, final_negative,
            temperature=config.cebra_temperature,
        ).cpu())
        final_positive_similarity = float((final_reference * final_positive).sum(dim=-1).mean().cpu())
        final_negative_similarity = float((final_reference @ final_negative.T).mean().cpu())
    overfit_result = {
        "diagnostic": "fixed_batch_overfit_real_data",
        "model": "transformer",
        "objective": "time_contrastive_blocks",
        "batch_shapes": {key: list(value.shape) for key, value in batch.items()},
        "batch_size": int(batch["reference_x"].shape[0]),
        "negative_candidate_count": int(batch["negative_x"].shape[0]),
        "logits_shape_formula": "(batch, batch)",
        "optimizer_training_mode": "train",
        "transformer_dropout": config.transformer_dropout,
        "fixed_batch_anchor_trial_counts": anchor_trial_counts,
        "fixed_batch_anchor_examples_trial_time": [
            [int(trial), int(time_id)]
            for trial, time_id in zip(anchor_trial_values[:12], anchor_time_values[:12])
        ],
        "steps": overfit_steps,
        "first_gradient_l2_norm": first_gradient,
        "first_training_loss": start_loss,
        "initial_fixed_batch_eval_loss": initial_fixed_loss,
        "final_fixed_batch_eval_loss": final_fixed_loss,
        "initial_positive_cosine_mean": initial_positive_similarity,
        "initial_negative_cosine_mean": initial_negative_similarity,
        "final_positive_cosine_mean": final_positive_similarity,
        "final_negative_cosine_mean": final_negative_similarity,
        "final_loss_same_fixed_batch": loss_values[-1],
        "loss_history": loss_values,
        "parameter_delta_l2": float(torch.linalg.vector_norm(_parameter_vector(model) - initial)),
        "all_trainable_parameters_in_optimizer": True,
        "device": str(device),
    }

    toy = _run_easy_temporal_toy(config.training_seed)
    output_dir = Path(diagnostic_dir) if diagnostic_dir is not None else project_root / "outputs" / "audit_2026-09-23_monkey_training_thesis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"training_diagnostics_B{config.batch_size}_steps{overfit_steps}_seed{config.training_seed}_trainmode.json"
    )
    if output_path.exists():
        raise FileExistsError(f"diagnostic output already exists; use a new seed or diagnostic directory: {output_path}")
    _json_write(output_path, {
        "diagnostic_run_id": f"diagnostics_{config.training_seed}_B{config.batch_size}_steps{overfit_steps}",
        "source_run_id": config.run_label,
        "initialization_seed": config.training_seed,
        "real_monkey_one_batch_overfit": overfit_result,
        "deterministic_temporal_toy": toy,
        "historical_outputs_modified": False,
    })
    del model, optimizer, dataset, sampler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_path


def _run_easy_temporal_toy(seed: int) -> dict[str, Any]:
    """A tiny positive-control fit; it does not alter the synthetic reference."""
    from neurobridge.data.dataset import TemporalWindowDataset
    from neurobridge.models.temporal_cnn import TemporalCNNEncoder
    from neurobridge.sampling.cebra_time import CEBRATripletWindowDataset

    n_trials, trial_length, n_features = 8, 100, 4
    time = np.arange(trial_length, dtype=np.float32)
    raw = []
    for trial in range(n_trials):
        phase = trial * 0.13
        raw.append(np.column_stack([
            np.sin(0.08 * time + phase), np.cos(0.08 * time + phase),
            np.sin(0.16 * time - phase), np.cos(0.16 * time - phase),
        ]))
    series = np.asarray(raw, dtype=np.float32)
    return _train_easy_temporal_toy(series, seed, n_features)


def _train_easy_temporal_toy(series: np.ndarray, seed: int, n_features: int) -> dict[str, Any]:
    from neurobridge.data.dataset import TemporalWindowDataset
    from neurobridge.models.temporal_cnn import TemporalCNNEncoder
    from neurobridge.sampling.cebra_time import CEBRATripletWindowDataset

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # A center identity window keeps this control focused on temporal pairing.
    center_values = series.reshape(-1, n_features)
    n_trials, trial_length = series.shape[:2]
    centers = np.tile(np.arange(trial_length, dtype=np.int64), n_trials)
    trials = np.repeat(np.arange(n_trials, dtype=np.int64), trial_length)
    global_times = np.arange(len(centers), dtype=np.int64)
    windows = center_values[:, None, :]
    dataset = TemporalWindowDataset(windows, centers, global_times, trials)
    indices = np.arange(len(dataset)).tolist()
    sampler = CEBRATripletWindowDataset(dataset, indices, offset=5)
    generator = torch.Generator().manual_seed(seed)
    sampler.resample(generator=generator)
    loader = DataLoader(sampler, batch_size=128, shuffle=True, drop_last=True, generator=generator)
    model = TemporalCNNEncoder(n_features=n_features, embedding_dim=3, hidden_dim=16, kernel_size=1, n_layers=1, normalize=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    initial_loss = None
    final_loss = None
    iterator = iter(loader)
    for step in range(150):
        sampler.resample(generator=generator)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        reference = model(batch["reference_x"])
        positive = model(batch["positive_x"])
        negative = model(batch["negative_x"])
        loss = cebra_infonce_loss(reference, positive, negative, temperature=1.0)
        loss.backward()
        optimizer.step()
        if initial_loss is None:
            initial_loss = float(loss.detach())
        final_loss = float(loss.detach())
    return {
        "seed": seed,
        "n_trials": int(n_trials),
        "trial_length": int(trial_length),
        "input_shape": list(windows.shape),
        "positive_offset_bins": 5,
        "batch_size": 128,
        "optimizer_updates": 150,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "learned_temporal_signal": bool(final_loss is not None and initial_loss is not None and final_loss < initial_loss),
    }


__all__ = [
    "ValidationSchedule",
    "canonical_config",
    "reuse_natural_input_cache",
    "reuse_train_only_pca",
    "fit_validated_neural_model",
    "compute_representation_geometry",
    "compute_trial_bootstrap_decoding",
    "run_natural_monkey_suite",
    "run_training_diagnostics",
]
