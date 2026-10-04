"""Immutable input preparation for the amended Real controlled-lag study.

The fitting adapter consumes only the generated train/validation bundle.
Test-only windows are materialized separately for embedding generation and
are never passed to an optimizer, validation loader, or model-selection code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from neurobridge.experiments import real_monkey
from neurobridge.sampling.f_windows import build_windows


SHIFTS_MS = (100, 160, 200)
RESIDUAL_LAGS_MS = tuple(range(-20, 21))
WINDOW_SIZE = 201
TRIAL_LENGTH = 600
POPULATION = "B_DISTAL"
RAW_SOURCE = "data/monkey_reaching_preload_smth_40/macaque_data.jl"
PARTITION_RELATIVE = "outputs/phase2a_hpo/partitions/real_somatotopic_v1.json"
SAFE_BUNDLE_RELATIVE = "outputs/phase2a_hpo/safe_inputs/real_somatotopic_v1/real_w201/B_DISTAL"


class LagProtocolError(RuntimeError):
    """Raised when an immutable parent or lag-window invariant is violated."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def frozen_split_sha256(split: Mapping[str, list[int]]) -> str:
    """Recreate the exact Windows-written frozen split bytes without saving them."""
    text = json.dumps(split, indent=2, default=str).replace("\n", "\r\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_verified_parents(project_root: Path, final_spec: Mapping[str, object]):
    """Load the exact frozen source/split and verify all sealed identities."""
    project = project_root.resolve()
    partition_path = project / PARTITION_RELATIVE
    source_path = project / RAW_SOURCE
    safe_root = project / SAFE_BUNDLE_RELATIVE
    manifest_path = safe_root / "manifest.json"
    windows_path = safe_root / "windows.npz"
    safe_split_path = safe_root / "split.json"
    if not all(path.is_file() for path in (
        partition_path, source_path, manifest_path, windows_path, safe_split_path,
    )):
        raise LagProtocolError("one or more frozen Real lag parents are missing")

    partition_sha = file_sha256(partition_path)
    if partition_sha != final_spec.get("somatotopic_partition_sha256"):
        raise LagProtocolError("canonical somatotopic partition hash changed")
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    indices = tuple(map(int, partition[POPULATION]["feature_indices"]))
    if len(indices) != 33 or len(set(indices)) != 33:
        raise LagProtocolError("B_DISTAL must contain exactly 33 unique frozen channels")

    safe_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (file_sha256(windows_path) != safe_manifest.get("safe_windows_sha256") or
            file_sha256(safe_split_path) != safe_manifest.get("safe_split_sha256")):
        raise LagProtocolError("frozen B train/validation bundle hash mismatch")
    if (safe_manifest.get("raw_source_sha256") != file_sha256(source_path) or
            safe_manifest.get("raw_source_sha256") != "bf6b9eab46d8daaa8f6a9e69c7df1dbd9e337c60ee38af520042d3047c244d13"):
        raise LagProtocolError("raw source differs from the source used for frozen fits")
    if (safe_manifest.get("partition_sha256") != partition_sha or
            safe_manifest.get("window_size") != WINDOW_SIZE or
            safe_manifest.get("trial_length_bins") != TRIAL_LENGTH or
            safe_manifest.get("population") != POPULATION):
        raise LagProtocolError("frozen B bundle does not match the Real lag design")

    config = real_monkey.RealMonkeyConfig(
        source_relative=RAW_SOURCE,
        setting_name="real_0",
        population_name=POPULATION,
        channel_split_seed=42,
        channel_indices=indices,
        imposed_shift_bins=0,
        window_size=WINDOW_SIZE,
        stride=1,
        padding="center",
        split_seed=42,
    )
    loaded_source, loaded = real_monkey._load_source(project, config)
    if (loaded_source.resolve() != source_path.resolve() or
            loaded["info"]["source_sha256"] != safe_manifest["raw_source_sha256"] or
            loaded["info"]["n_trials"] != 193 or
            loaded["info"]["trial_length_bins"] != TRIAL_LENGTH or
            loaded["info"]["n_neurons"] != len(indices)):
        raise LagProtocolError("raw B source shape or source identity changed")

    split = real_monkey._make_trial_split(loaded["arrays"]["target_by_trial"], config)
    original_split_hash = frozen_split_sha256(split)
    if original_split_hash != safe_manifest.get("original_split_sha256"):
        raise LagProtocolError("reconstructed split bytes do not match frozen parent hash")
    safe_split = json.loads(safe_split_path.read_text(encoding="utf-8"))
    if (split["train"] != safe_split["train"] or
            split["validation"] != safe_split["validation"] or
            split["train"] != safe_manifest["train_trial_ids"] or
            split["validation"] != safe_manifest["validation_trial_ids"]):
        raise LagProtocolError("reconstructed frozen train/validation IDs differ")
    if (len(split["train"],) != 134 or len(split["validation"]) != 20 or
            len(split["test"]) != 39 or
            set(split["train"]) | set(split["validation"]) | set(split["test"]) != set(range(193))):
        raise LagProtocolError("frozen Real split counts/coverage changed")

    with np.load(windows_path, allow_pickle=False) as source:
        values = {key: source[key] for key in source.files}
    count = len(values["trial_id"])
    if (values["X_windows"].shape != (count, WINDOW_SIZE, len(indices)) or
            count != (len(split["train"]) + len(split["validation"])) * TRIAL_LENGTH or
            np.any(np.isin(values["trial_id"], split["test"])) or
            set(np.unique(values["split"])) != {"train", "validation"}):
        raise LagProtocolError("frozen fit bundle includes test rows or has changed shape")
    parents = {
        "raw_source_sha256": file_sha256(source_path),
        "partition_sha256": partition_sha,
        "safe_bundle_manifest_sha256": file_sha256(manifest_path),
        "safe_windows_sha256": file_sha256(windows_path),
        "safe_split_sha256": file_sha256(safe_split_path),
        "original_split_sha256": original_split_hash,
    }
    return config, loaded["arrays"], split, safe_manifest, values, parents


def shift_safe_windows(values: Mapping[str, np.ndarray], shift_ms: int) -> dict[str, np.ndarray]:
    """Shift B window contents right within each trial, without circular wrap.

    Metadata/behavior stay at their original target time. The validity mask
    excludes windows crossing either a shifted invalid edge or trial padding.
    """
    if shift_ms not in SHIFTS_MS:
        raise LagProtocolError(f"unsupported imposed shift: {shift_ms}")
    ids = np.asarray(values["trial_id"], dtype=np.int64)
    times = np.asarray(values["time_id"], dtype=np.int64)
    original_valid = np.asarray(values["lag_valid"], dtype=bool)
    n = len(ids)
    if len(times) != n or len(original_valid) != n:
        raise LagProtocolError("window/time/validity cardinalities differ")
    if values["X_windows"].shape != (n, WINDOW_SIZE, 33):
        raise LagProtocolError("B safe windows must have shape N x 201 x 33")

    shifted = {key: np.array(value, copy=True) for key, value in values.items()
               if key not in {"X_windows", "lag_valid"}}
    shifted_x = np.zeros_like(values["X_windows"])
    shifted_valid = np.zeros(n, dtype=bool)
    lookup = {(int(trial), int(time)): int(row) for row, (trial, time) in
              enumerate(zip(ids, times))}
    if len(lookup) != n:
        raise LagProtocolError("duplicate trial/time rows in frozen safe windows")
    radius = WINDOW_SIZE // 2
    for dest, (trial, time) in enumerate(zip(ids, times)):
        source_time = int(time) - shift_ms
        source = lookup.get((int(trial), source_time))
        if source is None:
            continue
        shifted_x[dest] = values["X_windows"][source]
        shifted_valid[dest] = bool(original_valid[source]) and radius <= int(time) < TRIAL_LENGTH - radius
    shifted["X_windows"] = shifted_x
    shifted["lag_valid"] = shifted_valid
    return shifted


def _center_valid(valid_bins: np.ndarray, window_size: int) -> np.ndarray:
    n_trials, trial_length = valid_bins.shape
    radius = window_size // 2
    result = np.zeros((n_trials, trial_length), dtype=bool)
    for center in range(radius, trial_length - radius):
        result[:, center] = valid_bins[:, center - radius:center + radius + 1].all(axis=1)
    return result


def extract_test_windows(loaded: Mapping[str, np.ndarray], test_trial_ids: list[int],
                         shift_ms: int) -> dict[str, np.ndarray]:
    """Build test-only B windows; omit behavior targets and all fit metadata."""
    if shift_ms not in (0, *SHIFTS_MS):
        raise LagProtocolError(f"unsupported imposed shift: {shift_ms}")
    n_trials, trial_length = 193, TRIAL_LENGTH
    spikes = np.asarray(loaded["spikes"], dtype=np.float32).reshape(n_trials, trial_length, 33)
    targets = np.asarray(loaded["target_by_trial"], dtype=np.int64)
    position = np.asarray(loaded["position"], dtype=np.float32).reshape(n_trials, trial_length, -1)
    velocity = np.asarray(loaded["velocity"], dtype=np.float32).reshape(n_trials, trial_length, -1)
    if not test_trial_ids or any(t < 0 or t >= n_trials for t in test_trial_ids):
        raise LagProtocolError("test-only extractor received invalid test IDs")

    count = len(test_trial_ids) * trial_length
    all_windows = np.empty((count, WINDOW_SIZE, 33), dtype=np.float32)
    all_time = np.tile(np.arange(trial_length, dtype=np.int64), len(test_trial_ids))
    all_trial = np.repeat(np.asarray(test_trial_ids, dtype=np.int64), trial_length)
    all_valid = np.zeros(count, dtype=bool)
    radius = WINDOW_SIZE // 2
    for local, trial in enumerate(test_trial_ids):
        original_trial = spikes[trial]
        shifted_trial = np.zeros_like(original_trial)
        if shift_ms:
            shifted_trial[shift_ms:, :] = original_trial[:-shift_ms, :]
        else:
            shifted_trial[:] = original_trial
        shifted_valid_bins = np.ones((1, trial_length), dtype=bool)
        if shift_ms:
            shifted_valid_bins[0, :shift_ms] = False
        valid_centers = _center_valid(shifted_valid_bins, WINDOW_SIZE)[0]

        new_windows, new_time, _, _, _ = build_windows(
            shifted_trial, WINDOW_SIZE, 1,
            labels=np.asarray([targets[trial]], dtype=np.int64),
            trial_len=trial_length, time_mode="absolute", padding="center", pad_value=0.0,
        )
        old_windows, old_time, _, _, _ = build_windows(
            original_trial, WINDOW_SIZE, 1,
            labels=np.asarray([targets[trial]], dtype=np.int64),
            trial_len=trial_length, time_mode="absolute", padding="center", pad_value=0.0,
        )
        if not np.array_equal(new_time, old_time) or len(new_time) != trial_length:
            raise LagProtocolError("shifted test windows changed time coordinates")
        valid_times = np.flatnonzero(valid_centers)
        source_times = valid_times - shift_ms
        if (not len(valid_times) or source_times.min() < 0 or
                source_times.max() >= trial_length or
                not np.array_equal(new_windows[valid_times], old_windows[source_times])):
            raise LagProtocolError(f"shifted test windows are not exact no-wrap remaps: trial {trial}")
        start = local * trial_length
        stop = start + trial_length
        all_windows[start:stop] = new_windows
        all_valid[start:stop] = valid_centers

    return {
        "X_windows": all_windows,
        "trial_id": all_trial,
        "time_id": all_time,
        "global_time_id": all_trial * trial_length + all_time,
        "lag_valid": all_valid,
        "split": np.full(count, "test", dtype="U10"),
    }


def candidate_lags(shift_ms: int) -> tuple[int, ...]:
    if shift_ms == 0:
        return RESIDUAL_LAGS_MS
    if shift_ms not in SHIFTS_MS:
        raise LagProtocolError(f"unsupported lag condition: {shift_ms}")
    return tuple(shift_ms + offset for offset in RESIDUAL_LAGS_MS)
