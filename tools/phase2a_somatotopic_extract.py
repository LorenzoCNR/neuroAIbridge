"""Trusted Real train/validation-only extractor for the frozen somatotopic partition.

This is the sole new component that reads the monolithic V2 Stage-1 container
and complete split. It never emits test rows, statistics, diagnostics or fits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurobridge.experiments.phase2a_hpo import (  # noqa: E402
    ProtocolViolation, _canonical_json, file_sha256, guarded_output_root,
)
from neurobridge.experiments.phase2a_window_inputs import WINDOWS  # noqa: E402
from neurobridge.sampling.f_windows import build_windows  # noqa: E402

VERSION = "real_somatotopic_v1"
PARTITION_RELATIVE = Path("outputs/phase2a_hpo/partitions/real_somatotopic_v1.json")
PARENT_RELATIVE = Path("outputs/runs/v2_corrected_2026-09-23_seed42_real_total_65/stage01_data")


def partition(project: Path) -> tuple[dict, str]:
    path = project / PARTITION_RELATIVE
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["partition_version"] != VERSION or value["historical_random_partition_used_for_somatotopic_fits"]:
        raise ProtocolViolation("somatotopic partition version/history invalid")
    ids = value["feature_order_nlb_unit_ids"]
    a = value["A_PROXIMAL"]["feature_indices"]
    b = value["B_DISTAL"]["feature_indices"]
    if (len(ids) != 65 or len(a) != 32 or len(b) != 33 or set(a) & set(b) or
            set(a + b) != set(range(65)) or len(set(ids)) != 65):
        raise ProtocolViolation("somatotopic partition cardinality/coverage invalid")
    for name in ("A_PROXIMAL", "B_DISTAL"):
        if [ids[index] for index in value[name]["feature_indices"]] != value[name]["nlb_unit_ids"]:
            raise ProtocolViolation(f"frozen feature-to-NLB-ID mapping differs: {name}")
    if {ids[i] // 100 for i in a} & {ids[i] // 100 for i in b}:
        raise ProtocolViolation("physical electrode prefix split across populations")
    return value, file_sha256(path)


def _save_new_or_same(path: Path, value: object) -> None:
    data = (_canonical_json(value) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise ProtocolViolation(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def extract(project_root: Path, window_size: int, population: str) -> dict:
    project = project_root.resolve()
    if window_size not in WINDOWS or population not in {"A_PROXIMAL", "B_DISTAL"}:
        raise ProtocolViolation("not a frozen somatotopic condition")
    part, partition_sha = partition(project)
    indices = np.asarray(part[population]["feature_indices"], dtype=np.int64)
    parent = project / PARENT_RELATIVE
    data_path, split_path, parent_manifest_path = (parent / "data.npz", parent / "split.json", parent / "manifest.json")
    original = json.loads(split_path.read_text(encoding="utf-8"))
    if set(original) != {"train", "validation", "test"}:
        raise ProtocolViolation("complete split schema invalid")
    train, validation, test = [list(map(int, original[key])) for key in ("train", "validation", "test")]
    if (not train or not validation or any(set(a) & set(b) for a, b in
            ((train, validation), (train, test), (validation, test)))):
        raise ProtocolViolation("split empty or overlapping")
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest["config"]["channel_indices"] != list(range(65)):
        raise ProtocolViolation("TOTAL65 parent feature order changed")
    raw_sha = parent_manifest["source"]["source_sha256"]
    selected = np.asarray(sorted(train + validation), dtype=np.int64)
    with np.load(data_path, allow_pickle=False) as data:
        required = {"spikes", "target_by_trial", "position", "velocity", "valid_bins"}
        if not required <= set(data.files):
            raise ProtocolViolation("TOTAL65 Stage-1 parent missing arrays")
        n_trials = len(data["target_by_trial"])
        if (n_trials != 193 or set(train + validation + test) != set(range(n_trials)) or
                data["spikes"].shape != (n_trials * 600, 65)):
            raise ProtocolViolation("TOTAL65 source dimensions/split changed")
        spikes_selected = data["spikes"].reshape(n_trials, 600, 65)[selected][:, :, indices]
        labels_selected = data["target_by_trial"][selected]
        windows, time_id, _, trial_local, labels = build_windows(
            spikes_selected.reshape(len(selected) * 600, len(indices)), window_size, 1,
            labels=labels_selected, trial_len=600, time_mode="absolute",
            padding="center", pad_value=0.0,
        )
        trial_id = selected[trial_local]
        center = time_id.astype(np.int64)
        position = data["position"].reshape(n_trials, 600, -1)[trial_id, center]
        velocity = data["velocity"].reshape(n_trials, 600, -1)[trial_id, center]
        valid_bins = data["valid_bins"].reshape(n_trials, 600)
        radius = window_size // 2
        valid_centers = np.zeros((len(selected), 600), dtype=bool)
        for center_index in range(radius, 600 - radius):
            valid_centers[:, center_index] = valid_bins[selected, center_index-radius:center_index+radius+1].all(axis=1)
        lag_valid = valid_centers[trial_local, center]
        arrays = {
            "X_windows": windows.astype(np.float32, copy=False),
            "time_id": center, "global_time_id": (trial_id * 600 + center).astype(np.int64),
            "trial_id": trial_id.astype(np.int64), "labels": np.asarray(labels, dtype=np.int64),
            "progress": center.astype(np.float32) / 599,
            "position": position.astype(np.float32, copy=False),
            "velocity": velocity.astype(np.float32, copy=False),
            "split": np.where(np.isin(trial_id, train), "train", "validation").astype("U10"),
            "lag_valid": lag_valid,
        }
    if (len(arrays["trial_id"]) != len(selected) * 600 or
            np.any(np.isin(arrays["trial_id"], test)) or
            set(np.unique(arrays["split"])) != {"train", "validation"} or
            arrays["X_windows"].shape != (len(selected) * 600, window_size, len(indices))):
        raise ProtocolViolation("safe bundle cardinality/test firewall failed")
    total_root = guarded_output_root(project) / "safe_inputs" / f"real_w{window_size}" / "TOTAL65"
    total_manifest = json.loads((total_root / "manifest.json").read_text(encoding="utf-8"))
    if (total_manifest["train_trial_ids"] != train or total_manifest["validation_trial_ids"] != validation or
            total_manifest["original_split_sha256"] != file_sha256(split_path) or
            total_manifest["raw_source_sha256"] != raw_sha or
            total_manifest["window_size"] != window_size):
        raise ProtocolViolation("unchanged TOTAL65 safe bundle lineage differs")
    # Metadata checks do not decompress the large TOTAL65 neural tensor.
    with np.load(total_root / "windows.npz", allow_pickle=False) as existing:
        for field in ("time_id", "global_time_id", "trial_id", "labels", "progress",
                      "position", "velocity", "split", "lag_valid"):
            if not np.array_equal(arrays[field], existing[field]):
                raise ProtocolViolation(f"somatotopic vs TOTAL65 support differs: {field}")
    root = guarded_output_root(project) / "safe_inputs" / VERSION / f"real_w{window_size}" / population
    windows_path, safe_split_path, manifest_path = root / "windows.npz", root / "split.json", root / "manifest.json"
    parent_hashes = {
        "parent_data_sha256": file_sha256(data_path), "original_split_sha256": file_sha256(split_path),
        "raw_source_sha256": raw_sha, "partition_sha256": partition_sha,
        "partition_version": VERSION, "feature_order_sha256": hashlib.sha256(
            _canonical_json(part["feature_order_nlb_unit_ids"]).encode("utf-8")).hexdigest(),
        "extractor_source_sha256": file_sha256(Path(__file__)),
        "total65_safe_manifest_sha256": file_sha256(total_root / "manifest.json"),
        "total65_safe_windows_sha256": total_manifest["safe_windows_sha256"],
    }
    if any(path.exists() for path in (windows_path, safe_split_path, manifest_path)):
        if not all(path.is_file() for path in (windows_path, safe_split_path, manifest_path)):
            raise ProtocolViolation("partial immutable somatotopic bundle")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (any(existing.get(key) != value for key, value in parent_hashes.items()) or
                existing["safe_windows_sha256"] != file_sha256(windows_path) or
                existing["safe_split_sha256"] != file_sha256(safe_split_path)):
            raise ProtocolViolation("immutable somatotopic bundle drift")
        return existing
    root.mkdir(parents=True, exist_ok=False)
    with windows_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    _save_new_or_same(safe_split_path, {"train": train, "validation": validation})
    manifest = {
        "domain": f"real_w{window_size}", "population": population,
        "window_size": window_size, "stride": 1, "padding": "center", "trial_length_bins": 600,
        "channel_indices": indices.tolist(), "nlb_unit_ids": part[population]["nlb_unit_ids"],
        "train_trial_ids": train, "validation_trial_ids": validation,
        "train_window_count": int(np.sum(arrays["split"] == "train")),
        "validation_window_count": int(np.sum(arrays["split"] == "validation")),
        "valid_train_window_count": int(np.sum((arrays["split"] == "train") & lag_valid)),
        "valid_validation_window_count": int(np.sum((arrays["split"] == "validation") & lag_valid)),
        "test_trial_intersection_empty": True, "all_windows_are_train_or_validation": True,
        "safe_windows_sha256": file_sha256(windows_path), "safe_split_sha256": file_sha256(safe_split_path),
        **parent_hashes,
    }
    _save_new_or_same(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, choices=WINDOWS)
    parser.add_argument("--population", choices=("A_PROXIMAL", "B_DISTAL"))
    args = parser.parse_args()
    if (args.window is None) != (args.population is None):
        parser.error("specify both --window and --population, or neither")
    pairs = ([(args.window, args.population)] if args.window is not None else
             [(window, pop) for window in WINDOWS for pop in ("A_PROXIMAL", "B_DISTAL")])
    for window, pop in pairs:
        manifest = extract(ROOT, window, pop)
        print(f"{VERSION}/real_w{window}/{pop}: {manifest['train_window_count']} train, "
              f"{manifest['validation_window_count']} validation; sha={manifest['safe_windows_sha256']}", flush=True)


if __name__ == "__main__":
    main()
