"""Trusted extraction for the four amended Real temporal-window conditions.

Only this preparation module opens complete frozen Stage-1 data/split inputs.
It constructs centered windows from already-preprocessed spikes without
estimating any normalization or transform on the full recording.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neurobridge.experiments.phase2a_hpo import (
    ProtocolViolation, _canonical_json, channel_partition, file_sha256,
    guarded_output_root,
)
from neurobridge.sampling.f_windows import build_windows


WINDOWS = (21, 41, 121, 201)
PARENTS = {
    "TOTAL65": "outputs/runs/v2_corrected_2026-09-23_seed42_real_total_65",
    "A": "outputs/runs/clean_rebuild_2026-09-23_seed42_real_0_A",
    "B": "outputs/runs/clean_rebuild_2026-09-23_seed42_real_0_B",
}


def extract_real_window(project_root: Path, population: str, window_size: int) -> dict:
    project = project_root.resolve()
    if population not in PARENTS or window_size not in WINDOWS:
        raise ProtocolViolation("non-amended Real population/window")
    parent = project / PARENTS[population]
    data_path = parent / "stage01_data/data.npz"
    split_path = parent / "stage01_data/split.json"
    source_manifest_path = parent / "stage01_data/manifest.json"
    for path in (data_path, split_path, source_manifest_path):
        if not path.is_file():
            raise ProtocolViolation(f"frozen Real parent missing: {path}")
    original = json.loads(split_path.read_text(encoding="utf-8"))
    if set(original) != {"train", "validation", "test"}:
        raise ProtocolViolation("frozen Real split schema changed")
    train, validation, test = (list(map(int, original[name])) for name in ("train", "validation", "test"))
    if not train or not validation or any(set(a) & set(b) for a, b in ((train, validation), (train, test), (validation, test))):
        raise ProtocolViolation("frozen Real split has empty/overlapping partition")
    selected = np.asarray(sorted(train + validation), dtype=np.int64)
    with np.load(data_path, allow_pickle=False) as data:
        needed = {"spikes", "target_by_trial", "position", "velocity", "valid_bins"}
        if not needed <= set(data.files):
            raise ProtocolViolation("frozen Stage-1 Real cache lacks required arrays")
        labels_all = data["target_by_trial"]
        n_trials = len(labels_all)
        if n_trials != 193 or set(train + validation + test) != set(range(n_trials)):
            raise ProtocolViolation("frozen Real trial count/coverage changed")
        trial_length = data["spikes"].shape[0] // n_trials
        if trial_length != 600 or window_size >= trial_length:
            raise ProtocolViolation("frozen Real trial length/window changed")
        n_channels = data["spikes"].shape[1]
        expected_channels = {"TOTAL65": 65, "A": 32, "B": 33}
        if n_channels != expected_channels[population]:
            raise ProtocolViolation("frozen Real population channel count changed")
        spikes_selected = data["spikes"].reshape(n_trials, trial_length, n_channels)[selected]
        labels_selected = labels_all[selected]
        windows, time_id, _, trial_local, labels = build_windows(
            spikes_selected.reshape(len(selected) * trial_length, n_channels),
            window_size, 1, labels=labels_selected, trial_len=trial_length,
            time_mode="absolute", padding="center", pad_value=0.0,
        )
        trial_id = selected[trial_local]
        center = time_id.astype(np.int64)
        position = data["position"].reshape(n_trials, trial_length, -1)[trial_id, center]
        velocity = data["velocity"].reshape(n_trials, trial_length, -1)[trial_id, center]
        valid_bins = data["valid_bins"].reshape(n_trials, trial_length)
        radius = window_size // 2
        valid_centers = np.zeros((len(selected), trial_length), dtype=bool)
        for center_index in range(radius, trial_length - radius):
            valid_centers[:, center_index] = valid_bins[selected, center_index-radius:center_index+radius+1].all(axis=1)
        window_valid = valid_centers[trial_local, center]
        arrays = {
            "X_windows": windows.astype(np.float32, copy=False),
            "time_id": center,
            "global_time_id": (trial_id * trial_length + center).astype(np.int64),
            "trial_id": trial_id.astype(np.int64),
            "labels": np.asarray(labels, dtype=np.int64),
            "progress": center.astype(np.float32) / max(trial_length - 1, 1),
            "position": position.astype(np.float32, copy=False),
            "velocity": velocity.astype(np.float32, copy=False),
            "split": np.where(np.isin(trial_id, train), "train", "validation").astype("U10"),
            "lag_valid": window_valid,
        }
    if (len(arrays["trial_id"]) != len(selected) * trial_length or
            np.any(np.isin(arrays["trial_id"], test)) or
            set(np.unique(arrays["split"])) != {"train", "validation"}):
        raise ProtocolViolation("Real safe window extraction exposed test or lost rows")
    channels, partition_hash = channel_partition(project)
    if len(channels[population]) != n_channels:
        raise ProtocolViolation("channel partition disagrees with Stage-1 data")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    raw_sha = source_manifest.get("source", {}).get("source_sha256")
    if not raw_sha or len(raw_sha) != 64:
        raise ProtocolViolation("raw source hash absent from frozen parent")
    domain = f"real_w{window_size}"
    root = guarded_output_root(project) / "safe_inputs" / domain / population
    windows_path, safe_split_path, manifest_path = root / "windows.npz", root / "split.json", root / "manifest.json"
    parent_hashes = {"parent_data_sha256": file_sha256(data_path),
                     "original_split_sha256": file_sha256(split_path),
                     "raw_source_sha256": raw_sha,
                     "channel_partition_sha256": partition_hash,
                     "extractor_source_sha256": file_sha256(Path(__file__))}
    if any(path.exists() for path in (windows_path, safe_split_path, manifest_path)):
        if not all(path.is_file() for path in (windows_path, safe_split_path, manifest_path)):
            raise ProtocolViolation("partial immutable amended safe bundle")
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (any(cached.get(key) != value for key, value in parent_hashes.items())
                or cached.get("safe_windows_sha256") != file_sha256(windows_path)
                or cached.get("safe_split_sha256") != file_sha256(safe_split_path)
                or cached.get("train_trial_ids") != train
                or cached.get("validation_trial_ids") != validation):
            raise ProtocolViolation("amended safe bundle lineage/content changed")
        return cached
    root.mkdir(parents=True, exist_ok=False)
    with windows_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    with safe_split_path.open("x", encoding="utf-8") as handle:
        handle.write(_canonical_json({"train": train, "validation": validation}) + "\n")
    manifest = {
        "domain": domain, "population": population,
        "window_size": window_size, "stride": 1, "padding": "center",
        "trial_length_bins": trial_length,
        "window_center_convention": "one center at every original trial time; zero padded within each trial",
        "train_trial_ids": train, "validation_trial_ids": validation,
        "train_window_count": int(np.sum(arrays["split"] == "train")),
        "validation_window_count": int(np.sum(arrays["split"] == "validation")),
        "valid_train_window_count": int(np.sum((arrays["split"] == "train") & arrays["lag_valid"])),
        "valid_validation_window_count": int(np.sum((arrays["split"] == "validation") & arrays["lag_valid"])),
        "test_trial_intersection_empty": True,
        "all_windows_are_train_or_validation": True,
        "safe_windows_sha256": file_sha256(windows_path),
        "safe_split_sha256": file_sha256(safe_split_path),
        **parent_hashes,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(_canonical_json(manifest) + "\n")
    return manifest


def extract_all_real_windows(project_root: Path) -> dict[str, dict]:
    return {f"real_w{window}/{population}": extract_real_window(project_root, population, window)
            for window in WINDOWS for population in PARENTS}
