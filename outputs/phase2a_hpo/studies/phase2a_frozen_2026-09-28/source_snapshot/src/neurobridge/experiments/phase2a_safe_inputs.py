"""Trusted, one-time extraction of frozen V2 train/validation windows.

This module is the *only* Phase-2A component allowed to open complete V2
window caches and split metadata.  Fit/selection code must not import it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neurobridge.experiments.phase2a_hpo import (
    ProtocolViolation, _canonical_json, file_sha256,
    guarded_output_root,
)


PARENTS = {
    ("synthetic", "A"): (
        "outputs/runs/clean_rebuild_2026-09-23_seed42_synthetic/stage02_windows/windows_A.npz",
        "outputs/runs/clean_rebuild_2026-09-23_seed42_synthetic/stage02_windows/split.json",
        "outputs/runs/clean_rebuild_2026-09-23_seed42_synthetic/stage01_data/shared_data.npz",
    ),
    ("synthetic", "B"): (
        "outputs/runs/clean_rebuild_2026-09-23_seed42_synthetic/stage02_windows/windows_B.npz",
        "outputs/runs/clean_rebuild_2026-09-23_seed42_synthetic/stage02_windows/split.json",
        "outputs/runs/clean_rebuild_2026-09-23_seed42_synthetic/stage01_data/shared_data.npz",
    ),
    ("real", "TOTAL65"): (
        "outputs/runs/v2_corrected_2026-09-23_seed42_real_total_65/stage02_windows/windows.npz",
        "outputs/runs/v2_corrected_2026-09-23_seed42_real_total_65/stage01_data/split.json",
        "data/monkey_reaching_preload_smth_40/macaque_data.jl",
    ),
    ("real", "A"): (
        "outputs/runs/clean_rebuild_2026-09-23_seed42_real_0_A/stage02_windows/windows.npz",
        "outputs/runs/clean_rebuild_2026-09-23_seed42_real_0_A/stage01_data/split.json",
        "data/monkey_reaching_preload_smth_40/macaque_data.jl",
    ),
    ("real", "B"): (
        "outputs/runs/clean_rebuild_2026-09-23_seed42_real_0_B/stage02_windows/windows.npz",
        "outputs/runs/clean_rebuild_2026-09-23_seed42_real_0_B/stage01_data/split.json",
        "data/monkey_reaching_preload_smth_40/macaque_data.jl",
    ),
}


def extract_one(project_root: Path, domain: str, population: str) -> dict:
    """Subset *already preprocessed* windows; never refit a transform or metric."""
    project = project_root.resolve()
    if (domain, population) not in PARENTS:
        raise ProtocolViolation("unknown frozen input parent")
    root = guarded_output_root(project) / "safe_inputs" / domain / population
    parent_windows, parent_split, raw_source = (
        project / relative for relative in PARENTS[(domain, population)]
    )
    for parent in (parent_windows, parent_split, raw_source):
        if not parent.is_file():
            raise ProtocolViolation(f"frozen source missing: {parent}")
    original = json.loads(parent_split.read_text(encoding="utf-8"))
    if set(original) != {"train", "validation", "test"}:
        raise ProtocolViolation("frozen split schema changed")
    train = [int(x) for x in original["train"]]
    validation = [int(x) for x in original["validation"]]
    test = [int(x) for x in original["test"]]
    if not train or not validation or (set(train) & set(validation)) or (set(train) & set(test)) or (set(validation) & set(test)):
        raise ProtocolViolation("frozen split has empty/overlapping partitions")
    if domain == "synthetic" and (len(train), len(validation), len(test)) != (140, 20, 40):
        raise ProtocolViolation("synthetic frozen split counts changed")
    with np.load(parent_windows, allow_pickle=False) as source:
        required = {"X_windows", "trial_id", "time_id", "global_time_id", "labels", "lag_valid"}
        if not required <= set(source.files):
            raise ProtocolViolation("frozen windows lack required metadata")
        trial_ids = source["trial_id"]
        keep = np.isin(trial_ids, np.asarray(train + validation, dtype=np.int64))
        if not np.any(keep) or np.any(np.isin(trial_ids[keep], test)):
            raise ProtocolViolation("extractor selected test or no rows")
        count = len(trial_ids)
        arrays = {}
        for key in source.files:
            value = source[key]
            if value.ndim < 1 or len(value) != count:
                raise ProtocolViolation(f"non-rowwise source field: {key}")
            if key != "split":
                arrays[key] = value[keep]
        if not np.array_equal(arrays["trial_id"], trial_ids[keep]):
            raise ProtocolViolation("row order changed during extraction")
        source_split_for_rows = source["split"][keep] if domain == "real" else None
    arrays["split"] = np.where(np.isin(arrays["trial_id"], train), "train", "validation").astype("U10")
    if set(np.unique(arrays["split"])) != {"train", "validation"}:
        raise ProtocolViolation("both train and validation must have windows")
    if set(np.unique(arrays["trial_id"])) != set(train + validation):
        raise ProtocolViolation("safe windows do not cover the frozen train/validation trials")
    if domain == "real" and not np.array_equal(arrays["split"], source_split_for_rows):
        raise ProtocolViolation("real frozen per-window split differs from split metadata")
    # The complete source is not passed beyond this trusted extractor.
    windows_path = root / "windows.npz"
    split_path = root / "split.json"
    manifest_path = root / "manifest.json"
    if windows_path.exists() or split_path.exists() or manifest_path.exists():
        if not all(path.is_file() for path in (windows_path, split_path, manifest_path)):
            raise ProtocolViolation("partial immutable safe bundle")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (existing.get("parent_windows_sha256") != file_sha256(parent_windows)
                or existing.get("original_split_sha256") != file_sha256(parent_split)
                or existing.get("raw_source_sha256") != file_sha256(raw_source)
                or existing.get("extractor_source_sha256") != file_sha256(Path(__file__))
                or existing.get("safe_windows_sha256") != file_sha256(windows_path)
                or existing.get("safe_split_sha256") != file_sha256(split_path)
                or existing.get("train_trial_ids") != train
                or existing.get("validation_trial_ids") != validation):
            raise ProtocolViolation("safe bundle parent/content changed; no overwrite permitted")
        return existing
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids overwriting an existing scientific artifact.
    with windows_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    split_payload = {"train": train, "validation": validation}
    with split_path.open("x", encoding="utf-8") as handle:
        handle.write(_canonical_json(split_payload) + "\n")
    manifest = {
        "domain": domain, "population": population,
        "train_trial_ids": train, "validation_trial_ids": validation,
        "train_window_count": int(np.sum(arrays["split"] == "train")),
        "validation_window_count": int(np.sum(arrays["split"] == "validation")),
        "all_windows_are_train_or_validation": True,
        "test_trial_intersection_empty": bool(not set(arrays["trial_id"].tolist()) & set(test)),
        "parent_windows_sha256": file_sha256(parent_windows),
        "original_split_sha256": file_sha256(parent_split),
        "raw_source_sha256": file_sha256(raw_source),
        "extractor_source_sha256": file_sha256(Path(__file__)),
        "safe_windows_sha256": file_sha256(windows_path),
        "safe_split_sha256": file_sha256(split_path),
        "safe_windows_file": "windows.npz", "safe_split_file": "split.json",
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(_canonical_json(manifest) + "\n")
    return manifest


def extract_all(project_root: Path) -> dict[str, dict]:
    return {f"{domain}/{population}": extract_one(project_root, domain, population)
            for domain, population in PARENTS}
