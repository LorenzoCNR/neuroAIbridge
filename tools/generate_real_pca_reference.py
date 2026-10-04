"""Create the frozen-window Real PCA reference and its raw/unit plots.

The PCA fit is train-trial-only and separate from the neural model/metric
inventory. It projects the saved Real recording at all valid time points for
descriptive ``all`` and held-out views; no neural model is fitted or changed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import final_real_embeddings as real_export  # noqa: E402
import plot_final_thesis_embeddings as plotter  # noqa: E402

FINAL = ROOT / "outputs/final_thesis_v1/final_evaluation"
SPEC_PATH = ROOT / "outputs/final_thesis_v1/freeze/REAL_FINAL_EXPERIMENT_SPEC.json"
PARTITION_PATH = ROOT / "outputs/phase2a_hpo/partitions/real_somatotopic_v1.json"
CORE_INDEX = FINAL / "core_metrics/EMBEDDING_INDEX.csv"
ARTIFACT_ROOT = FINAL / "real_pca_reference"
PLOT_ROOT = plotter.OUTPUT
POPULATIONS = ("TOTAL65", "A_PROXIMAL", "B_DISTAL")
POPULATION_FOLDERS = {"TOTAL65": "T65", "A_PROXIMAL": "Aprox", "B_DISTAL": "Bdistal"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_index(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "dataset", "branch", "category", "representation", "population",
        "architecture", "objective", "seed", "fit_status", "support_rows",
        "file_type", "file",
    ]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _valid_pair_set(trial_id: np.ndarray, time_id: np.ndarray,
                    valid: np.ndarray) -> set[tuple[int, int]]:
    pairs = np.column_stack((trial_id[valid], time_id[valid]))
    if len(pairs) != len(np.unique(pairs, axis=0)):
        raise ValueError("duplicate trial/time pairs in Real PCA support")
    return set(map(tuple, pairs.astype(np.int64, copy=False).tolist()))


def _safe_population_references(spec: dict[str, Any], partition_sha: str,
                                source_hashes: dict[str, str]) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for population in POPULATIONS:
        slots = [slot for slot in spec["final_slots"] if slot.get("population") == population]
        if not slots:
            raise ValueError(f"frozen final specification has no {population} slot")
        slot = slots[0]
        reference_path = Path(slot["safe_input_reference"])
        if sha256(reference_path) != slot["safe_input_reference_sha256"]:
            raise ValueError(f"safe-input reference hash mismatch for {population}")
        reference = read_json(reference_path)
        if (reference.get("population") != population or
                reference.get("domain") != "real_w201" or
                reference.get("partition_sha256") != partition_sha):
            raise ValueError(f"safe-input identity mismatch for {population}")
        bundle = Path(reference["bundle_path"])
        bundle_manifest_path = bundle / "manifest.json"
        if sha256(bundle_manifest_path) != reference["bundle_manifest_sha256"]:
            raise ValueError(f"safe-input bundle manifest hash mismatch for {population}")
        bundle_manifest = read_json(bundle_manifest_path)
        if (bundle_manifest.get("original_split_sha256") != source_hashes["split_json_sha256"] or
                bundle_manifest.get("raw_source_sha256") != source_hashes["raw_source_sha256"]):
            raise ValueError(f"safe-input source/split ancestry mismatch for {population}")
        references[population] = {
            "reference_path": str(reference_path.relative_to(ROOT)),
            "reference_sha256": sha256(reference_path),
            "bundle_manifest_path": str(bundle_manifest_path.relative_to(ROOT)),
            "bundle_manifest_sha256": sha256(bundle_manifest_path),
        }
    return references


def _fit_population_pca(
    population: str,
    channel_indices: list[int],
    arrays: dict[str, np.ndarray],
    split: dict[str, list[int]],
    pca_seed: int,
    stage_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    window = int(real_export.WINDOW)
    trial_length = int(real_export.TRIAL_LENGTH)
    n_trials = len(arrays["target_by_trial"])
    spikes = arrays["spikes"].reshape(n_trials, trial_length, -1)
    valid_bins = arrays["valid_bins"].reshape(n_trials, trial_length)
    feature_count = window * len(channel_indices)
    n_fit_rows = len(split["train"]) * trial_length
    if n_fit_rows <= 3 or feature_count < 3:
        raise ValueError(f"insufficient PCA fit shape for {population}")

    # Match the established Real baseline: centered windows are flattened in
    # time-major/channel-minor order; PCA is fitted on train trials only.
    # A disk-backed matrix and copy=False reduce peak RAM without changing the
    # sklearn PCA model or its values.
    with tempfile.TemporaryDirectory(prefix=f"pca_{population}_", dir=stage_dir) as temp_dir:
        fit_path = Path(temp_dir) / "train_windows.float32"
        fit_matrix = np.memmap(
            fit_path, mode="w+", dtype=np.float32,
            shape=(n_fit_rows, feature_count), order="C",
        )
        cursor = 0
        for trial in split["train"]:
            trial_spikes = np.ascontiguousarray(spikes[int(trial)][:, channel_indices], dtype=np.float32)
            for _, windows in real_export.centered_batches(trial_spikes, batch_size=32):
                flattened = windows.reshape(len(windows), -1)
                fit_matrix[cursor:cursor + len(flattened)] = flattened
                cursor += len(flattened)
        if cursor != n_fit_rows:
            raise ValueError(f"PCA train-row count mismatch for {population}: {cursor} != {n_fit_rows}")
        fit_matrix.flush()
        pca = PCA(n_components=3, random_state=pca_seed, copy=False)
        pca.fit(fit_matrix)
        del fit_matrix

    n_rows = n_trials * trial_length
    raw = np.empty((n_rows, 3), dtype=np.float32)
    valid = np.zeros(n_rows, dtype=bool)
    trial_id = np.repeat(np.arange(n_trials, dtype=np.int64), trial_length)
    time_id = np.tile(np.arange(trial_length, dtype=np.int64), n_trials)
    split_name = np.empty(n_trials, dtype="U10")
    for name, trial_ids in split.items():
        split_name[np.asarray(trial_ids, dtype=np.int64)] = name
    row_split = np.repeat(split_name, trial_length)
    labels = np.repeat(np.asarray(arrays["target_by_trial"], dtype=np.int64), trial_length)

    for trial in range(n_trials):
        start = trial * trial_length
        trial_spikes = np.ascontiguousarray(spikes[trial][:, channel_indices], dtype=np.float32)
        valid[start:start + trial_length] = real_export.valid_centers(valid_bins[trial])
        for times, windows in real_export.centered_batches(trial_spikes, batch_size=32):
            transformed = pca.transform(windows.reshape(len(windows), -1))
            raw[start + times] = transformed.astype(np.float32, copy=False)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    unit = np.divide(raw, norms, out=np.zeros_like(raw), where=norms > 0)
    if not (np.isfinite(raw).all() and np.isfinite(unit).all()):
        raise ValueError(f"non-finite PCA embedding for {population}")
    if raw.shape != unit.shape or raw.shape[1] != 3:
        raise ValueError(f"PCA embedding is not N x 3 for {population}")
    metadata = {
        "embedding_raw": raw,
        "embedding_unit": unit,
        "trial_id": trial_id,
        "time_id": time_id,
        "global_time_id": trial_id * trial_length + time_id,
        "valid_mask": valid,
        "split": row_split,
        "labels": labels,
    }
    model_path = stage_dir / f"pca_{population}.joblib"
    embedding_path = stage_dir / f"embedding_{population}.npz"
    joblib.dump(pca, model_path)
    with embedding_path.open("xb") as stream:
        np.savez_compressed(stream, **metadata)
    info = {
        "population": population,
        "channel_indices": channel_indices,
        "n_channels": len(channel_indices),
        "n_components": 3,
        "window_size": window,
        "feature_shape": [window, len(channel_indices)],
        "flatten_order": "time-major/channel-minor (C order)",
        "fit_split": "train trials only",
        "fit_trial_count": len(split["train"]),
        "fit_window_count": n_fit_rows,
        "pca_random_state": pca_seed,
        "sklearn_effective_solver": getattr(pca, "_fit_svd_solver", "unknown"),
        "embedding_rows": n_rows,
        "valid_rows": int(valid.sum()),
        "pca_model": model_path.name,
        "pca_model_sha256": sha256(model_path),
        "embedding": embedding_path.name,
        "embedding_sha256": sha256(embedding_path),
    }
    return metadata, info


def _merge_plot_index(path: Path, additions: list[dict[str, object]], branch: str) -> tuple[list[dict[str, str]], str]:
    existing = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    new_rows = [dict(row) for row in additions if row["branch"] == branch]
    prior_files = {row["file"] for row in existing}
    if any(row["file"] in prior_files for row in new_rows):
        raise FileExistsError(f"Real PCA plot path already indexed: {path}")
    merged = existing + new_rows
    staging = path.with_name(path.stem + ".tmp.csv")
    if staging.exists():
        raise FileExistsError(f"temporary index path already exists: {staging}")
    fields = list(existing[0])
    with staging.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    old_hash = sha256(path)
    os.replace(staging, path)
    return merged, old_hash


def _refresh_branch_provenance(path: Path, index_path: Path,
                               index_rows: list[dict[str, str]],
                               pca_manifest_sha: str) -> None:
    payload = read_json(path)
    payload["plot_files"] = len(index_rows)
    payload["plot_index_sha256"] = sha256(index_path)
    payload["outputs_sha256"] = {
        row["file"]: sha256(PLOT_ROOT / row["file"])
        for row in index_rows
    }
    payload["real_pca_reference"] = {
        "manifest": str((ARTIFACT_ROOT / "manifest.json").relative_to(ROOT)),
        "manifest_sha256": pca_manifest_sha,
        "plot_entries_added_to_this_branch": 12,
    }
    payload["total_output_file_count_including_indexes_and_provenance"] = sum(
        1 for candidate in PLOT_ROOT.rglob("*") if candidate.is_file()
    )
    staging = path.with_name(path.stem + ".tmp.json")
    if staging.exists():
        raise FileExistsError(f"temporary provenance path already exists: {staging}")
    write_json_once(staging, payload)
    os.replace(staging, path)


def main() -> None:
    if ARTIFACT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite existing PCA reference: {ARTIFACT_ROOT}")
    for population_folder in POPULATION_FOLDERS.values():
        for branch in ("all", "test"):
            path = PLOT_ROOT / "real" / branch / population_folder / "pca"
            if path.exists():
                raise FileExistsError(f"refusing to overwrite existing Real PCA plots: {path}")
    for filename in ("PLOT_INDEX_REAL_PCA.csv", "PLOT_PROVENANCE_REAL_PCA.json"):
        if (PLOT_ROOT / filename).exists():
            raise FileExistsError(f"refusing to overwrite existing PCA plot manifest: {filename}")

    spec = read_json(SPEC_PATH)
    if int(spec["selected_real_window_size"]) != real_export.WINDOW:
        raise ValueError("Real PCA window does not match the frozen final window")
    arrays, split, source_hashes = real_export.stage1_arrays(ROOT, spec)
    partition = read_json(PARTITION_PATH)
    partition_sha = sha256(PARTITION_PATH)
    channel_map = {
        population: real_export.channel_indices(partition, population)
        for population in POPULATIONS
    }
    total = set(channel_map["TOTAL65"])
    proximal, distal = set(channel_map["A_PROXIMAL"]), set(channel_map["B_DISTAL"])
    if (len(total) != 65 or len(proximal) != 32 or len(distal) != 33 or
            proximal & distal or proximal | distal != total):
        raise ValueError("frozen 65 -> 32+33 channel partition invariant failed")
    references = _safe_population_references(spec, partition_sha, source_hashes)

    real_rows = [row for row in csv.DictReader(CORE_INDEX.open(encoding="utf-8", newline=""))
                 if row["dataset"] == "real"]
    if len(real_rows) != 72:
        raise ValueError(f"expected 72 frozen Real neural references, got {len(real_rows)}")
    common_all = plotter._validate_and_common_support(real_rows, "real", split_name=None)
    common_test = plotter._validate_and_common_support(real_rows, "real", split_name="test")

    stage_parent = ARTIFACT_ROOT.parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="real_pca_reference_", dir=stage_parent) as staging_name:
        stage_dir = Path(staging_name)
        pca_seed = int(read_json(ROOT / real_export.STAGE1 / "manifest.json")["fingerprint"]["config"]["split_seed"])
        pca_embeddings: dict[str, dict[str, np.ndarray]] = {}
        population_info = []
        for population in POPULATIONS:
            metadata, info = _fit_population_pca(
                population, channel_map[population], arrays, split,
                pca_seed, stage_dir,
            )
            pairs_all = _valid_pair_set(metadata["trial_id"], metadata["time_id"], metadata["valid_mask"])
            if pairs_all != common_all:
                raise ValueError(
                    f"PCA valid support differs from final neural support for {population}: "
                    f"pca={len(pairs_all)} vs common={len(common_all)}"
                )
            test_mask = metadata["valid_mask"] & (metadata["split"].astype(str) == "test")
            if _valid_pair_set(metadata["trial_id"], metadata["time_id"], test_mask) != common_test:
                raise ValueError(f"PCA held-out support differs from frozen Real test support: {population}")
            pca_embeddings[population] = metadata
            population_info.append(info)

        # Confirm behavior labels are carried through unchanged on shared support.
        reference_meta = plotter._metadata(real_rows[0])
        reference_labels = {
            (int(reference_meta["trial_id"][i]), int(reference_meta["time_id"][i])):
            int(reference_meta["labels"][i])
            for i in np.flatnonzero(reference_meta["valid_mask"])
        }
        for population, metadata in pca_embeddings.items():
            for i in np.flatnonzero(metadata["valid_mask"]):
                key = (int(metadata["trial_id"][i]), int(metadata["time_id"][i]))
                if key in common_all and int(metadata["labels"][i]) != reference_labels[key]:
                    raise ValueError(f"PCA behavior labels differ from frozen neural metadata: {population}/{key}")

        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "real",
            "architecture": "pca",
            "objective": "none",
            "purpose": "deterministic reference for embedding plots; not added to core metric inventory",
            "selected_window_size": real_export.WINDOW,
            "centered_window": True,
            "sample_rate_hz": 1000,
            "positive_offset": 10,
            "n_components": 3,
            "fit_trial_split": "train only",
            "fit_includes_zero_padded_edge_windows": True,
            "projection_support": "all centered rows; figures mask to frozen common valid support",
            "pca_random_state": pca_seed,
            "populations": population_info,
            "channel_partition_path": str(PARTITION_PATH.relative_to(ROOT)),
            "channel_partition_sha256": partition_sha,
            "channel_partition_counts": {"TOTAL65": len(total), "A_PROXIMAL": len(proximal), "B_DISTAL": len(distal)},
            "safe_input_references": references,
            "source_hashes": {
                **source_hashes,
                "final_spec_sha256": sha256(SPEC_PATH),
                "partition_sha256": partition_sha,
                "stage1_arrays_source_sha256": sha256(Path(real_export.__file__)),
                "real_pca_implementation_sha256": sha256(ROOT / "src/neurobridge/experiments/real_monkey.py"),
                "plot_adapter_sha256": sha256(Path(plotter.__file__)),
            },
            "support_rows": {"all": len(common_all), "held_out_test": len(common_test)},
            "neural_training_performed": False,
            "neural_optimizer_steps": 0,
            "core_metrics_modified": False,
        }
        write_json_once(stage_dir / "manifest.json", manifest)
        stage_dir.rename(ARTIFACT_ROOT)

    manifest_path = ARTIFACT_ROOT / "manifest.json"
    manifest_sha = sha256(manifest_path)
    manifest = read_json(manifest_path)
    index_rows: list[dict[str, object]] = []
    for population in POPULATIONS:
        metadata_values = pca_embeddings[population]
        metadata = {
            "trial_id": metadata_values["trial_id"],
            "time_id": metadata_values["time_id"],
            "valid_mask": metadata_values["valid_mask"],
            "split": metadata_values["split"],
            "labels": metadata_values["labels"],
        }
        population_folder = POPULATION_FOLDERS[population]
        folder_label = f"Real {population} | PCA reference"
        for branch, support, split_name, branch_folder, branch_label in (
            ("all_valid_rows_projection", common_all, None, "all", "all rows"),
            ("held_out_test", common_test, "test", "test", "held-out test"),
        ):
            mask = plotter._support_mask(metadata, support, split_name=split_name)
            artifact = ARTIFACT_ROOT / f"embedding_{population}.npz"
            with np.load(artifact, allow_pickle=False) as values:
                raw = values["embedding_raw"].copy()
                unit = values["embedding_unit"].copy()
            plotter._write_views(
                raw, unit, metadata, mask,
                folder=PLOT_ROOT / "real" / branch_folder / population_folder / "pca" / "none" / "reference",
                label=folder_label,
                stem=f"real_{population}_pca_reference",
                index_rows=index_rows,
                dataset="real", population=population,
                architecture="pca", objective="none", seed="deterministic",
                status="PCA_REFERENCE_TRAIN_FIT",
                branch=branch,
                branch_label=branch_label,
                category="pca_reference_embedding",
            )

    pca_index_path = PLOT_ROOT / "PLOT_INDEX_REAL_PCA.csv"
    write_index(pca_index_path, index_rows)
    held_index_path = PLOT_ROOT / "PLOT_INDEX.csv"
    all_index_path = PLOT_ROOT / "PLOT_INDEX_ALL.csv"
    held_rows, held_prior_sha = _merge_plot_index(held_index_path, index_rows, "held_out_test")
    all_rows, all_prior_sha = _merge_plot_index(all_index_path, index_rows, "all_valid_rows_projection")
    _refresh_branch_provenance(
        PLOT_ROOT / "PLOT_PROVENANCE.json", held_index_path, held_rows, manifest_sha
    )
    _refresh_branch_provenance(
        PLOT_ROOT / "PLOT_PROVENANCE_ALL.json", all_index_path, all_rows, manifest_sha
    )

    pca_plot_hashes = {
        row["file"]: sha256(PLOT_ROOT / str(row["file"])) for row in index_rows
    }
    extra_provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "issue": "Real PCA plots were absent from the final plot index; only Synthetic PCA references were indexed.",
        "scientific_resolution": (
            "legacy Real PCA artifacts used window_size=21 and were not substituted into the frozen "
            "window_size=201 final view; a train-only W201 PCA reference was fitted instead"
        ),
        "pca_artifact_manifest": str(manifest_path.relative_to(ROOT)),
        "pca_artifact_manifest_sha256": manifest_sha,
        "real_pca_plot_index": str(pca_index_path.relative_to(ROOT)),
        "real_pca_plot_index_sha256": sha256(pca_index_path),
        "parent_plot_index_sha256_before_append": {
            "held_out": held_prior_sha,
            "all": all_prior_sha,
        },
        "plot_index_sha256_after_append": {
            "held_out": sha256(held_index_path),
            "all": sha256(all_index_path),
        },
        "pca_plot_files": len(index_rows),
        "pca_plot_outputs_sha256": pca_plot_hashes,
        "training_scope": "PCA only; one train-trial fit per population; no neural refit",
        "test_metrics_computed": False,
        "core_metric_tables_modified": False,
    }
    write_json_once(PLOT_ROOT / "PLOT_PROVENANCE_REAL_PCA.json", extra_provenance)
    print(f"Created 3 Real train-fit PCA references and {len(index_rows)} plots")
    print(f"Held-out/common all support: {len(common_test)} / {len(common_all)}")
    print(f"PCA artifact manifest: {manifest_path}")
    print(f"Plot indexes updated: {held_index_path}, {all_index_path}")


if __name__ == "__main__":
    main()
