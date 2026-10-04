"""Executable contracts and provenance for the staged research benchmark."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_provenance(root):
    root = Path(root)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    # Source hashes also identify uncommitted/untracked implementations.
    sources = {str(p.relative_to(root)): file_sha256(p)
               for p in sorted((root / "src" / "neurobridge").rglob("*.py"))}
    return {"git_commit": commit, "source_sha256": sources}


def check_cached_config(path, config):
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Missing cache configuration: {path}")
    saved = json.loads(path.read_text(encoding="utf-8"))
    mismatches = []
    # Scientific cache files use stable public labels for sampler settings.
    # Compare their values to the corresponding internal dataclass fields;
    # never expose implementation/vendor names in the saved artifacts.
    public_aliases = {
        "cebra_time_offset": "time_offset_bins",
        "cebra_temperature": "temporal_objective_temperature",
    }
    for key, value in asdict(config).items():
        if key in {"output_root", "run_label"}:
            continue
        # New optional seeds preserve the historical derivation when absent.
        saved_key = public_aliases.get(key, key)
        if saved.get(saved_key) != value:
            mismatches.append(key)
    if mismatches:
        raise ValueError(f"Cache configuration mismatch at {path}: {mismatches}; use a new run label")


def validate_split(split, trial_ids, expected_counts=None):
    names = ("train", "validation", "test")
    sets = []
    for name in names:
        ids = np.asarray(split[name])
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids):
            raise ValueError(f"{name} must contain integer trial IDs and be non-empty")
        values = set(ids.tolist())
        if len(values) != len(ids):
            raise ValueError(f"Duplicate trial IDs in {name}")
        if expected_counts is not None and len(values) != expected_counts[name]:
            raise ValueError(f"Unexpected {name} cardinality")
        sets.append(values)
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Trial partitions must be disjoint")
    if set.union(*sets) != set(np.asarray(trial_ids).tolist()):
        raise ValueError("Trial partitions must have complete coverage")
    masks = {name: np.isin(trial_ids, split[name]) for name in names}
    if not np.all(sum(mask.astype(int) for mask in masks.values()) == 1):
        raise ValueError("Every window must belong to exactly one partition")
    return masks


def validate_metadata(values, data, config, subject):
    trial = np.asarray(values["trial_id"])
    time = np.asarray(values["time_id"])
    if trial.ndim != 1 or time.shape != trial.shape or trial.dtype.kind not in "iu" or time.dtype.kind not in "iu":
        raise ValueError("trial_id/time_id must be aligned integer vectors")
    if np.any((trial < 0) | (trial >= config.n_trials)) or np.any((time < 0) | (time >= config.trial_length)):
        raise ValueError("trial/time IDs out of bounds")
    keys = trial * config.trial_length + time
    expected = (np.arange(config.n_trials)[:, None] * config.trial_length
                + np.arange(0, config.trial_length, config.stride)).ravel()
    if not np.array_equal(np.sort(keys), expected):
        raise ValueError("Window IDs must uniquely cover the configured centers")
    for key in ("global_time_id", "labels", "progress", "lag_valid"):
        if np.asarray(values[key]).shape != trial.shape:
            raise ValueError(f"Misaligned {key}")
    if not np.array_equal(values["global_time_id"], keys):
        raise ValueError("global_time_id disagrees with trial/time")
    if not np.array_equal(values["labels"], data["labels"][trial]):
        raise ValueError("Labels disagree with trial IDs")
    if not np.allclose(values["progress"], data["M"][trial, time, 2], rtol=1e-6, atol=1e-7):
        raise ValueError("Progress disagrees with trial/time")
    source_valid = np.asarray(data[f"valid_{subject}"], dtype=bool)
    if source_valid.shape != (config.n_trials, config.trial_length):
        raise ValueError("generative validity mask does not match trial/time dimensions")
    window_valid = np.zeros_like(source_valid)
    left_width = config.window_size // 2
    right_width = config.window_size - left_width
    for center in range(left_width, config.trial_length - right_width + 1):
        window_valid[:, center] = source_valid[
            :, center - left_width:center + right_width
        ].all(axis=1)
    if not np.array_equal(values["lag_valid"].astype(bool), window_valid[trial, time]):
        raise ValueError("lag_valid disagrees with strict window validity metadata")


def interior_mask(time_ids, config, subject):
    """Exclude windows touching padding or the lag-replicated prefix."""
    left = config.window_size // 2
    right = config.window_size - left - 1
    lower = config.lag_bins if subject == "B" else 0
    times = np.asarray(time_ids)
    return (times - left >= lower) & (times + right < config.trial_length)


def write_checkpoint_provenance(project_root, config, checkpoint, windows_path, split_path):
    import torch
    payload = {
        **source_provenance(project_root),
        "config": asdict(config),
        "data_seed": config.seed,
        "split_seed": config.seed + 101 if config.split_seed is None else config.split_seed,
        "training_seed_base": config.seed if config.training_seed is None else config.training_seed,
        "subject_seed_offset": "A=0, B=1",
        "preprocessing": "raw neural counts; PCA centers train windows only",
        "selection": "validation loss for held_out; training loss for full_sample",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_version": torch.__version__,
        "files": {str(p.name): file_sha256(p) for p in (checkpoint, windows_path, split_path)},
    }
    checkpoint.with_name("provenance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
