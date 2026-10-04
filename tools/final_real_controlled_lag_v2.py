"""Real controlled digital-lag study: frozen-encoder and retrained-B branches.

The frozen R0 outputs and all core metric outputs are read-only parents. New
inputs, B fits, embeddings, tables, and figures live under a separate v2 branch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurobridge.data.dataset import TemporalWindowDataset  # noqa: E402
from neurobridge.eval.representation import lagged_alignment_by_trial_time  # noqa: E402
from neurobridge.experiments import real_monkey  # noqa: E402
from neurobridge.experiments.final_real_lag_fit import (  # noqa: E402
    ARCHITECTURES, OBJECTIVES, SEEDS, fit_shifted_b_trial,
)
from neurobridge.experiments.final_real_lag_revision import (  # noqa: E402
    POPULATION, RESIDUAL_LAGS_MS, SHIFTS_MS, WINDOW_SIZE,
    LagProtocolError, canonical_json, candidate_lags, extract_test_windows,
    file_sha256, load_verified_parents, shift_safe_windows,
)
from neurobridge.experiments.phase2a_hpo import _canonical_json  # noqa: E402
from neurobridge.experiments.phase2a_safe_fit import _frozen_config  # noqa: E402
from neurobridge.train.loop import encode_windows  # noqa: E402


FINAL_ROOT = Path("outputs/final_thesis_v1")
EVAL_ROOT = FINAL_ROOT / "final_evaluation"
SPEC_PATH = FINAL_ROOT / "freeze/REAL_FINAL_EXPERIMENT_SPEC.json"
INDEX_PATH = EVAL_ROOT / "core_metrics/EMBEDDING_INDEX.csv"
BRANCH_ROOT = EVAL_ROOT / "controlled_lag_v2"
TRAINVAL_ROOT = BRANCH_ROOT / "inputs/train_validation"
TEST_ROOT = BRANCH_ROOT / "inputs/test_only"
EMBED_ROOT = BRANCH_ROOT / "embeddings"
FITS_ROOT = BRANCH_ROOT / "fits/refit"
CORE_SOURCE = "tools/final_thesis_core_metrics.py"
DOWNSTREAM_SOURCE = "tools/final_thesis_downstream_eval.py"
EXPECTED_SPLIT_SHA = "e1c115122c0424c124e0268bebd2a1d976a081564288538e785269ed9c5a666a"
EXPECTED_PARTITION_SHA = "027ca4b9a6ba53471b72cf93a11e9514968af4399df0dc347de4c44973b98025"


def _json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_immutable(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(value) + "\n")


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists() or path.exists():
        raise FileExistsError(f"refusing to overwrite existing immutable artifact: {path}")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)
    return file_sha256(path)


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _ensure_study_manifest(project: Path, spec: Mapping[str, Any], parents: Mapping[str, str]) -> dict[str, Any]:
    payload = {
        "study_id": "real_controlled_lag_v2",
        "decision_class": "NEW_PROTOCOL_DECISION",
        "imposed_B_shifts_ms": list(SHIFTS_MS),
        "canonical_window_ms": WINDOW_SIZE,
        "lag_search": {
            "R0_candidate_lags_ms": list(candidate_lags(0)),
            "shifted_candidate_lags_ms": {str(d): list(candidate_lags(d)) for d in SHIFTS_MS},
            "resolution_ms": 1,
            "local_residual_offsets_ms": list(RESIDUAL_LAGS_MS),
            "common_reference_support": "held-out trial/time rows valid for every lag in R0 and all three intervention curves, shared across both branches",
        },
        "branches": [
            "frozen_R0_B_encoder: reuse frozen B embedding rows with the exact no-wrap temporal input mapping",
            "retrained_B_encoder: random initialization and fresh B-only fit for each shift/architecture/objective/seed",
        ],
        "reused_anchor": "same frozen R0 A_PROXIMAL embedding/checkpoint per architecture/objective/seed",
        "populations": ["A_PROXIMAL", "B_DISTAL"],
        "architectures": list(ARCHITECTURES),
        "objectives": list(OBJECTIVES),
        "training_seeds": list(SEEDS),
        "new_B_fit_count": len(SHIFTS_MS) * len(ARCHITECTURES) * len(OBJECTIVES) * len(SEEDS),
        "hpo_or_test_based_selection": False,
        "frozen_real_spec_sha256": parents["frozen_spec_sha256"],
        "frozen_protocol_design_sha256": spec["canonical_design_sha256"],
        "frozen_split_sha256": parents["original_split_sha256"],
        "somatotopic_partition_sha256": parents["partition_sha256"],
        "raw_source_sha256": parents["raw_source_sha256"],
        "safe_trainval_bundle_sha256": parents["safe_windows_sha256"],
    }
    path = project / BRANCH_ROOT / "CONTROLLED_LAG_V2_SPEC.json"
    if path.exists():
        current = _json_read(path)
        if current != payload:
            raise LagProtocolError("existing controlled-lag v2 spec differs; preserving existing branch")
        return current
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_immutable(path, payload)
    return payload


def _extract_test_values(raw: Mapping[str, np.ndarray], split: Mapping[str, list[int]], shift: int) -> dict[str, np.ndarray]:
    return extract_test_windows(raw, list(split["test"]), shift)


def _bundle_paths(project: Path, shift: int, *, test: bool) -> tuple[Path, Path, Path]:
    parent = (project / TEST_ROOT if test else project / TRAINVAL_ROOT) / f"shift_{shift:03d}ms_B_DISTAL"
    if shift == 0 and test:
        parent = project / TEST_ROOT / "shift_000ms_B_DISTAL"
    return parent, parent / ("test_windows.npz" if test else "trainval_windows.npz"), parent / "manifest.json"


def _prepare_trainval_bundle(project: Path, shift: int, values: Mapping[str, np.ndarray],
                             split: Mapping[str, list[int]], safe_manifest: Mapping[str, Any],
                             parents: Mapping[str, str]) -> tuple[Path, dict[str, Any]]:
    root, windows_path, manifest_path = _bundle_paths(project, shift, test=False)
    split_path = root / "split.json"
    parent_payload = {
        "shift_ms": shift,
        "population": POPULATION,
        "window_size": WINDOW_SIZE,
        "stride": 1,
        "padding": "center",
        "trial_length_bins": 600,
        "train_trial_ids": split["train"],
        "validation_trial_ids": split["validation"],
        "test_trial_intersection_empty": True,
        "fit_accessible": True,
        "raw_source_sha256": parents["raw_source_sha256"],
        "original_split_sha256": parents["original_split_sha256"],
        "partition_sha256": parents["partition_sha256"],
        "parent_safe_manifest_sha256": parents["safe_bundle_manifest_sha256"],
        "parent_safe_windows_sha256": parents["safe_windows_sha256"],
        "extractor_source_sha256": file_sha256(ROOT / "src/neurobridge/experiments/final_real_lag_revision.py"),
    }
    if root.exists():
        if not all(p.is_file() for p in (windows_path, split_path, manifest_path)):
            raise LagProtocolError(f"partial shifted train/validation bundle; preserve it: {root}")
        current = _json_read(manifest_path)
        if (any(current.get(key) != value for key, value in parent_payload.items()) or
                current.get("shifted_windows_sha256") != file_sha256(windows_path) or
                current.get("safe_split_sha256") != file_sha256(split_path)):
            raise LagProtocolError(f"existing shifted train/validation bundle hash/parent mismatch: {root}")
        return root, current

    shifted = shift_safe_windows(values, shift)
    if (np.any(np.isin(shifted["trial_id"], split["test"])) or
            set(np.unique(shifted["split"])) != {"train", "validation"}):
        raise LagProtocolError("shifted training bundle violates held-out test firewall")
    root.mkdir(parents=True, exist_ok=False)
    windows_sha = _write_npz_atomic(windows_path, shifted)
    with split_path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json({"train": split["train"], "validation": split["validation"]}) + "\n")
    manifest = {
        **parent_payload,
        "train_window_count": int(np.sum(shifted["split"] == "train")),
        "validation_window_count": int(np.sum(shifted["split"] == "validation")),
        "valid_train_window_count": int(np.sum((shifted["split"] == "train") & shifted["lag_valid"])),
        "valid_validation_window_count": int(np.sum((shifted["split"] == "validation") & shifted["lag_valid"])),
        "safe_split_sha256": file_sha256(split_path),
        "shifted_windows_sha256": windows_sha,
    }
    _write_json_immutable(manifest_path, manifest)
    return root, manifest


def _prepare_test_bundle(project: Path, shift: int, raw: Mapping[str, np.ndarray],
                         split: Mapping[str, list[int]], parents: Mapping[str, str]) -> tuple[Path, dict[str, Any]]:
    root, windows_path, manifest_path = _bundle_paths(project, shift, test=True)
    parent_payload = {
        "shift_ms": shift,
        "population": POPULATION,
        "window_size": WINDOW_SIZE,
        "stride": 1,
        "padding": "center",
        "trial_length_bins": 600,
        "test_trial_ids": split["test"],
        "evaluation_only": True,
        "fit_accessible": False,
        "labels_or_behavior_in_bundle": False,
        "raw_source_sha256": parents["raw_source_sha256"],
        "original_split_sha256": parents["original_split_sha256"],
        "partition_sha256": parents["partition_sha256"],
        "extractor_source_sha256": file_sha256(ROOT / "src/neurobridge/experiments/final_real_lag_revision.py"),
    }
    if root.exists():
        if not all(p.is_file() for p in (windows_path, manifest_path)):
            raise LagProtocolError(f"partial test-only bundle; preserve it: {root}")
        current = _json_read(manifest_path)
        if (any(current.get(key) != value for key, value in parent_payload.items()) or
                current.get("test_windows_sha256") != file_sha256(windows_path)):
            raise LagProtocolError(f"existing test-only bundle hash/parent mismatch: {root}")
        return root, current

    arrays = _extract_test_values(raw, split, shift)
    if (set(np.unique(arrays["split"])) != {"test"} or
            set(np.unique(arrays["trial_id"])) != set(split["test"]) or
            any(key in arrays for key in ("labels", "position", "velocity", "progress"))):
        raise LagProtocolError("test-only bundle contains non-test rows or behavior variables")
    root.mkdir(parents=True, exist_ok=False)
    windows_sha = _write_npz_atomic(windows_path, arrays)
    manifest = {
        **parent_payload,
        "test_window_count": int(len(arrays["trial_id"])),
        "valid_test_window_count": int(arrays["lag_valid"].sum()),
        "test_windows_sha256": windows_sha,
    }
    _write_json_immutable(manifest_path, manifest)
    return root, manifest


def _prepare_inputs(project: Path, spec: Mapping[str, Any], config: Any,
                    raw: Mapping[str, np.ndarray], split: Mapping[str, list[int]],
                    safe_manifest: Mapping[str, Any], values: Mapping[str, np.ndarray],
                    parents: Mapping[str, str]) -> tuple[dict[int, Path], dict[int, Path], dict[int, dict], dict[int, dict]]:
    train_paths, test_paths, train_manifests, test_manifests = {}, {}, {}, {}
    for delay in SHIFTS_MS:
        train_paths[delay], train_manifests[delay] = _prepare_trainval_bundle(
            project, delay, values, split, safe_manifest, parents)
        test_paths[delay], test_manifests[delay] = _prepare_test_bundle(
            project, delay, raw, split, parents)
    test_paths[0], test_manifests[0] = _prepare_test_bundle(project, 0, raw, split, parents)
    return train_paths, test_paths, train_manifests, test_manifests


def _load_core_embedding_index(project: Path, split: Mapping[str, list[int]]) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    path = project / INDEX_PATH
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        if row.get("dataset") != "real" or row.get("population") not in {"A_PROXIMAL", "B_DISTAL"}:
            continue
        key = (row["population"], row["architecture"], row["objective"], int(row["seed"]))
        folder = Path(row["embedding_path"])
        manifest_path = Path(row["manifest_path"])
        if not manifest_path.is_file() or file_sha256(manifest_path) != row["manifest_sha256"]:
            raise LagProtocolError(f"frozen embedding manifest hash mismatch: {manifest_path}")
        manifest = _json_read(manifest_path)
        raw_path, unit_path, meta_path = (folder / name for name in (
            "embedding_raw.npz", "embedding_unit.npz", "evaluation_metadata.npz"))
        if any(not p.is_file() for p in (raw_path, unit_path, meta_path)):
            raise LagProtocolError(f"frozen embedding is incomplete: {folder}")
        for filename, artifact_path in ((raw_path.name, raw_path), (unit_path.name, unit_path), (meta_path.name, meta_path)):
            expected = manifest.get("artifact_sha256", {}).get(filename)
            if not expected or file_sha256(artifact_path) != expected:
                raise LagProtocolError(f"frozen embedding child hash mismatch: {artifact_path}")
        with np.load(raw_path, allow_pickle=False) as archive:
            raw = archive["embedding_raw"]
        with np.load(unit_path, allow_pickle=False) as archive:
            unit = archive["embedding_unit"]
        with np.load(meta_path, allow_pickle=False) as archive:
            metadata = {name: archive[name] for name in archive.files}
        n = len(metadata["trial_id"])
        if (raw.shape != (n, 3) or unit.shape != (n, 3) or
                any(len(value) != n for value in metadata.values()) or
                len(set(zip(metadata["trial_id"].tolist(), metadata["time_id"].tolist()))) != n):
            raise LagProtocolError(f"frozen embedding/metadata invariant failed: {row['trial_id']}")
        expected_test = set(split["test"])
        observed_test = set(np.unique(metadata["trial_id"][metadata["split"] == "test"]).tolist())
        if observed_test != expected_test:
            raise LagProtocolError("frozen embedding test trial IDs differ from exact reconstructed split")
        result[key] = {
            "embedding_raw": raw,
            "embedding_unit": unit,
            **metadata,
            "parent_trial_id": row["trial_id"],
            "parent_embedding_path": str(folder),
            "parent_manifest_sha256": file_sha256(manifest_path),
            "parent_raw_sha256": file_sha256(raw_path),
            "parent_unit_sha256": file_sha256(unit_path),
            "parent_metadata_sha256": file_sha256(meta_path),
            "fit_status": row["fit_status"],
            "near_collapse": row["near_collapse"],
        }
    expected = {
        (population, architecture, objective, seed)
        for population in ("A_PROXIMAL", "B_DISTAL")
        for architecture in ARCHITECTURES
        for objective in OBJECTIVES
        for seed in SEEDS
    }
    if set(result) != expected:
        raise LagProtocolError(f"expected {len(expected)} frozen A/B embedding slots; found {len(result)}")
    return result


def _load_base_fit_records(project: Path, spec: Mapping[str, Any]) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[tuple[str, str, int], Path]]:
    config_by_cell, checkpoint_by_cell = {}, {}
    for slot in spec["final_slots"]:
        if slot.get("population") != POPULATION:
            continue
        seed = int(slot["training_seed_root"])
        if seed not in SEEDS:
            raise LagProtocolError("frozen final slot has an unexpected training seed")
        trial_dir = (Path(slot["checkpoint_path"]).parent if slot.get("reuse_hpo_checkpoint") else
                     Path(slot["output_path"]))
        record_path = trial_dir / "trial_record.json"
        if not record_path.is_file():
            raise LagProtocolError(f"frozen fit record missing: {record_path}")
        record = _json_read(record_path)
        config = record.get("config")
        if (record.get("config_sha256") != slot["config_sha256"] or not isinstance(config, dict) or
                config.get("population") != POPULATION or config.get("architecture") != slot["architecture"] or
                config.get("objective") != slot["objective"] or
                config.get("training_seed_root") != seed or config.get("window_size") != WINDOW_SIZE or
                config.get("imposed_shift_bins") != 0):
            raise LagProtocolError("frozen B fit record differs from sealed Real spec")
        config_hash = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()
        if config_hash != slot["config_sha256"]:
            raise LagProtocolError("frozen B config bytes differ from sealed config hash")
        checkpoint = (Path(slot["checkpoint_path"]) if slot.get("reuse_hpo_checkpoint") else
                      trial_dir / "best_validation_checkpoint.pt")
        expected_checkpoint_sha = slot.get("checkpoint_sha256")
        if expected_checkpoint_sha is None:
            final_result_path = trial_dir / "result.json"
            if not final_result_path.is_file():
                raise LagProtocolError(f"frozen final B fit result missing: {final_result_path}")
            expected_checkpoint_sha = _json_read(final_result_path).get("artifact_sha256", {}).get("best_validation_checkpoint.pt")
        if not checkpoint.is_file() or file_sha256(checkpoint) != expected_checkpoint_sha:
            raise LagProtocolError("frozen B checkpoint hash mismatch")
        key = (slot["architecture"], slot["objective"], seed)
        config_by_cell[key] = {"config": config, "source_trial_id": slot["trial_id"],
                               "source_config_sha256": config_hash,
                               "source_record_sha256": file_sha256(record_path),
                               "source_checkpoint_sha256": file_sha256(checkpoint)}
        checkpoint_by_cell[key] = checkpoint
    expected = {(a, o, s) for a in ARCHITECTURES for o in OBJECTIVES for s in SEEDS}
    if set(config_by_cell) != expected:
        raise LagProtocolError("frozen B config/checkpoint slots are incomplete")
    return config_by_cell, checkpoint_by_cell


def _test_rows(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    mask = np.asarray(values["split"]) == "test"
    return {key: np.asarray(value)[mask] for key, value in values.items()}


def _core_test_rows(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    mask = (np.asarray(values["split"]) == "test") & np.asarray(values["valid_mask"], dtype=bool)
    return {
        "embedding_raw": values["embedding_raw"][mask],
        "embedding_unit": values["embedding_unit"][mask],
        "trial_id": np.asarray(values["trial_id"])[mask].astype(np.int64),
        "time_id": np.asarray(values["time_id"])[mask].astype(np.int64),
        "valid_mask": np.ones(int(mask.sum()), dtype=bool),
        "split": np.repeat("test", int(mask.sum())),
    }


def _remap_frozen_b(base: Mapping[str, np.ndarray], test_inputs: Mapping[str, np.ndarray], shift: int) -> dict[str, np.ndarray]:
    base_lookup = {
        (int(trial), int(time)): row
        for row, (trial, time) in enumerate(zip(base["trial_id"], base["time_id"]))
        if base["valid_mask"][row] and base["split"][row] == "test"
    }
    test = _test_rows(test_inputs)
    n = len(test["trial_id"])
    raw = np.zeros((n, 3), dtype=np.float32)
    unit = np.zeros((n, 3), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for dest, (trial, time_id) in enumerate(zip(test["trial_id"], test["time_id"])):
        if not test["lag_valid"][dest]:
            continue
        source = base_lookup.get((int(trial), int(time_id) - shift))
        if source is None:
            continue
        raw[dest] = base["embedding_raw"][source]
        unit[dest] = base["embedding_unit"][source]
        valid[dest] = True
    return {
        "embedding_raw": raw,
        "embedding_unit": unit,
        "trial_id": test["trial_id"].astype(np.int64),
        "time_id": test["time_id"].astype(np.int64),
        "global_time_id": test["global_time_id"].astype(np.int64),
        "valid_mask": valid,
        "split": np.repeat("test", n),
    }


def _save_embedding(project: Path, path: Path, arrays: Mapping[str, np.ndarray], identity: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = path.with_suffix(".manifest.json")
    if path.exists() or manifest_path.exists():
        if not path.is_file() or not manifest_path.is_file():
            raise LagProtocolError(f"partial immutable embedding; preserve it: {path}")
        current = _json_read(manifest_path)
        if (any(current.get(key) != value for key, value in identity.items()) or
                current.get("artifact_sha256") != file_sha256(path)):
            raise LagProtocolError(f"existing immutable embedding has different parents: {path}")
        return current
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = _write_npz_atomic(path, arrays)
    payload = {
        **identity,
        "artifact_path": str(path.resolve()),
        "artifact_sha256": digest,
        "embedding_shape": list(arrays["embedding_raw"].shape),
        "valid_count": int(np.asarray(arrays["valid_mask"]).sum()),
        "row_count": int(len(arrays["trial_id"])),
    }
    _write_json_immutable(manifest_path, payload)
    return payload


def _load_test_bundle(path: Path, shift: int) -> dict[str, np.ndarray]:
    # `path` is the absolute `shift_NNNms_B_DISTAL` folder supplied by caller.
    manifest = _json_read(path / "manifest.json")
    if (manifest.get("fit_accessible") is not False or manifest.get("evaluation_only") is not True or
            manifest.get("shift_ms") != shift or file_sha256(path / "test_windows.npz") != manifest.get("test_windows_sha256")):
        raise LagProtocolError("test-only bundle firewall/hash failed")
    with np.load(path / "test_windows.npz", allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    if (set(np.unique(values["split"])) != {"test"} or
            any(key in values for key in ("labels", "position", "velocity", "progress"))):
        raise LagProtocolError("test-only bundle includes behavior or non-test data")
    return values


def _infer_checkpoint(trial: Mapping[str, Any], checkpoint: Path,
                      test_bundle: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    row = trial["config"]
    base_config = _frozen_config(trial)
    x = np.asarray(test_bundle["X_windows"], dtype=np.float32)
    valid = np.asarray(test_bundle["lag_valid"], dtype=bool)
    dataset = TemporalWindowDataset(
        x,
        np.asarray(test_bundle["time_id"], dtype=np.int64),
        np.asarray(test_bundle["global_time_id"], dtype=np.int64),
        np.asarray(test_bundle["trial_id"], dtype=np.int64),
        extra_metadata={"lag_valid": valid.astype(np.int64)},
    )
    selected = np.flatnonzero(valid).tolist()
    if len(selected) < 3:
        raise LagProtocolError("test intervention has fewer than three valid windows")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_model = real_monkey._make_real_model(row["architecture"], x.shape[-1], base_config, normalize=False).to(device)
    unit_model = real_monkey._make_real_model(row["architecture"], x.shape[-1], base_config, normalize=True).to(device)
    raw_model.load_state_dict(checkpoint_data["state_dict"])
    unit_model.load_state_dict(checkpoint_data["state_dict"])
    loader = DataLoader(Subset(dataset, selected), batch_size=1024, shuffle=False)
    started = time.perf_counter()
    raw_tensor, _ = encode_windows(raw_model, loader, device=device)
    unit_tensor, _ = encode_windows(unit_model, loader, device=device)
    elapsed = time.perf_counter() - started
    n = len(test_bundle["trial_id"])
    raw = np.zeros((n, 3), dtype=np.float32)
    unit = np.zeros((n, 3), dtype=np.float32)
    raw[selected] = raw_tensor.numpy().astype(np.float32, copy=False)
    unit[selected] = unit_tensor.numpy().astype(np.float32, copy=False)
    embedding = {
        "embedding_raw": raw,
        "embedding_unit": unit,
        "trial_id": np.asarray(test_bundle["trial_id"], dtype=np.int64),
        "time_id": np.asarray(test_bundle["time_id"], dtype=np.int64),
        "global_time_id": np.asarray(test_bundle["global_time_id"], dtype=np.int64),
        "valid_mask": valid,
        "split": np.asarray(test_bundle["split"]),
    }
    if (raw.shape != unit.shape or raw.shape != (n, 3) or
            any(len(value) != n for value in embedding.values()) or
            not np.isfinite(raw).all() or not np.isfinite(unit).all()):
        raise LagProtocolError("new B held-out embedding invariant failed")
    details = {
        "inference_seconds": elapsed,
        "inference_valid_windows_per_second": len(selected) / max(elapsed, 1e-12),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_available": torch.cuda.is_available(),
        "pytorch": torch.__version__,
    }
    return embedding, details


def _make_shift_trial(base_config: Mapping[str, Any], shift: int, architecture: str,
                      objective: str, seed: int) -> dict[str, Any]:
    config = dict(base_config)
    config["imposed_shift_bins"] = shift
    config["population"] = POPULATION
    config["architecture"] = architecture
    config["objective"] = objective
    config["training_seed_root"] = seed
    config["training_seed_effective"] = seed
    config_sha = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()
    trial_id = (f"lagv2-refit-w201-B_DISTAL-{architecture}-{objective}-d{shift}-s{seed}-"
                f"{config_sha[:12]}")
    return {
        "trial_id": trial_id,
        "config": config,
        "config_sha256": config_sha,
        "output_path": str((ROOT / FITS_ROOT / f"d{shift}" / trial_id).resolve()),
    }


def _all_pair_rows(core: Mapping[tuple[str, str, str, int], dict[str, Any]]):
    for architecture in ARCHITECTURES:
        for objective in OBJECTIVES:
            for seed in SEEDS:
                a = core[("A_PROXIMAL", architecture, objective, seed)]
                b = core[("B_DISTAL", architecture, objective, seed)]
                yield architecture, objective, seed, a, b


def _coordinate_index(values: Mapping[str, np.ndarray], *, valid_only: bool = True,
                      test_only: bool = True) -> dict[tuple[int, int], int]:
    valid = np.asarray(values["valid_mask"], dtype=bool)
    if valid_only:
        keep = valid.copy()
    else:
        keep = np.ones(len(valid), dtype=bool)
    if test_only:
        keep &= np.asarray(values["split"]) == "test"
    coords = list(zip(np.asarray(values["trial_id"])[keep].astype(int),
                      np.asarray(values["time_id"])[keep].astype(int)))
    indices = np.flatnonzero(keep)
    if len(set(coords)) != len(coords):
        raise LagProtocolError("duplicate trial/time keys in lag support")
    return {coord: int(index) for coord, index in zip(coords, indices)}


def _common_reference(a: Mapping[str, np.ndarray], b0: Mapping[str, np.ndarray],
                      shifted_b: Mapping[tuple[str, int], Mapping[str, np.ndarray]]) -> list[tuple[int, int]]:
    a_map = _coordinate_index(a)
    b0_map = _coordinate_index(b0)
    maps = {(branch, delay): _coordinate_index(value)
            for (branch, delay), value in shifted_b.items()}
    lags_by_series = {("R0", 0): candidate_lags(0)}
    for branch in ("frozen_R0_encoder", "retrained_B_encoder"):
        for delay in SHIFTS_MS:
            lags_by_series[(branch, delay)] = candidate_lags(delay)
    result = []
    for trial, center in a_map:
        if any((trial, center + lag) not in b0_map for lag in lags_by_series[("R0", 0)]):
            continue
        if any((trial, center + lag) not in maps[(branch, delay)]
               for (branch, delay), lags in lags_by_series.items() if branch != "R0"
               for lag in lags):
            continue
        result.append((trial, center))
    result.sort()
    if len(result) < 3:
        raise LagProtocolError(f"common held-out support too small: {len(result)}")
    return result


def _score_curve(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray],
                 refs: list[tuple[int, int]], lags: tuple[int, ...]) -> dict[int, float]:
    a_map = _coordinate_index(a)
    b_map = _coordinate_index(b)
    idx_a = np.asarray([a_map[coordinate] for coordinate in refs], dtype=np.int64)
    b_coords = sorted(b_map)
    idx_b = np.asarray([b_map[coordinate] for coordinate in b_coords], dtype=np.int64)
    _, score, aligned = lagged_alignment_by_trial_time(
        a["embedding_unit"][idx_a], b["embedding_unit"][idx_b],
        np.asarray([coord[0] for coord in refs]), np.asarray([coord[1] for coord in refs]),
        np.asarray([coord[0] for coord in b_coords]), np.asarray([coord[1] for coord in b_coords]),
        lags, common_support=True,
    )
    if any(len(aligned[lag][0]) != len(refs) for lag in lags):
        raise LagProtocolError("lag scorer violated frozen equal-support requirement")
    return {int(lag): float(value) for lag, value in score.items()}


def _curve_summary(curve: Mapping[int, float]) -> dict[str, Any]:
    finite = [(int(lag), float(score)) for lag, score in curve.items() if np.isfinite(score)]
    if len(finite) < 3:
        raise LagProtocolError("lag curve has insufficient finite scores")
    finite.sort()
    best_lag, best_score = max(finite, key=lambda value: value[1])
    second = max((score for lag, score in finite if lag != best_lag), default=float("nan"))
    lags = [lag for lag, _ in finite]
    return {
        "lag_hat_ms": best_lag,
        "S_max": best_score,
        "peak_margin": float(best_score - second),
        "boundary_censored": best_lag in {min(lags), max(lags)},
        "lag_min_ms": min(lags),
        "lag_max_ms": max(lags),
    }


def _plot_results(project: Path, curves: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> list[str]:
    folder = project / BRANCH_ROOT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    colors = {100: "#0072B2", 160: "#E69F00", 200: "#009E73"}
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharey=True)
    for ax, (architecture, objective) in zip(axes.flat, [(a, o) for a in ARCHITECTURES for o in OBJECTIVES]):
        rows = [r for r in curves if r["architecture"] == architecture and r["objective"] == objective]
        for branch, label, linestyle in (("R0_baseline", "R0", ":"),
                                          ("frozen_R0_encoder", "frozen", "-"),
                                          ("retrained_B_encoder", "B retrained", "--")):
            relevant = [r for r in rows if r["branch"] == branch]
            groups = defaultdict(list)
            for row in relevant:
                groups[int(row["lag_ms"])].append(float(row["procrustes_r2"]))
            for delay in ([0] if branch == "R0_baseline" else SHIFTS_MS):
                x = sorted(lag for lag in groups if (delay == 0 and -20 <= lag <= 20) or
                           (delay and delay - 20 <= lag <= delay + 20))
                if not x:
                    continue
                mean = np.asarray([np.mean(groups[lag]) for lag in x])
                sd = np.asarray([np.std(groups[lag], ddof=1) if len(groups[lag]) > 1 else 0.0 for lag in x])
                color = "#555555" if delay == 0 else colors[delay]
                ax.plot(x, mean, color=color, linestyle=linestyle,
                        label=(label if delay == 0 else f"{label} +{delay}"), linewidth=1.4)
                if len(x) > 1:
                    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.10)
        ax.set_title(f"{architecture} / {objective}", fontsize=9)
        ax.set_xlabel("Candidate lag (ms; B(t + lag) vs A(t))")
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Procrustes R²")
    axes[1, 0].set_ylabel("Procrustes R²")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.895), ncol=7, fontsize=8)
    fig.suptitle("Real held-out temporal response to B-only digital shifts\nseed mean ± SD; R0 anchor and frozen/refitted B branches")
    fig.subplots_adjust(top=0.83, bottom=0.11, left=0.055, right=0.99, hspace=0.32, wspace=0.15)
    paths = []
    for extension, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        path = folder / f"REAL_CONTROLLED_LAG_CURVES.{extension}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path.resolve()))
    plt.close(fig)

    selected = [r for r in summaries if r["shift_ms"] in SHIFTS_MS and r["delta_identifiable"]]
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for branch, marker, color in (("frozen_R0_encoder", "o", "#0072B2"),
                                  ("retrained_B_encoder", "s", "#D55E00")):
        rs = [r for r in selected if r["branch"] == branch]
        ax.scatter([r["expected_shift_ms"] for r in rs], [r["delta_lag_hat_ms"] for r in rs],
                   marker=marker, color=color, alpha=0.75, label=branch)
    ax.plot([100, 200], [100, 200], color="black", linestyle="--", label="identity")
    ax.set(xlabel="Imposed B delay (ms)", ylabel="Δ estimated lag (ms)",
           title="Controlled temporal-shift response (interior peaks only)")
    intervention_rows = [r for r in summaries if r["shift_ms"] in SHIFTS_MS]
    baseline_rows = [r for r in summaries if r["branch"] == "R0_baseline"]
    identifiable_count = len(selected)
    unresolved_count = len(intervention_rows) - identifiable_count
    intervention_boundary_count = sum(bool(r["boundary_censored"]) for r in intervention_rows)
    baseline_boundary_count = sum(bool(r["boundary_censored"]) for r in baseline_rows)
    ax.text(
        0.98, 0.025,
        f"Identifiable deltas: {identifiable_count}/{len(intervention_rows)}\n"
        f"Unresolved/censored deltas: {unresolved_count}/{len(intervention_rows)}\n"
        f"Intervention boundary peaks: {intervention_boundary_count}/{len(intervention_rows)}\n"
        f"R0 boundary peaks: {baseline_boundary_count}/{len(baseline_rows)}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.75"},
    )
    ax.set_xlim(90, 210)
    ax.set_ylim(90, 210)
    ax.grid(alpha=0.25)
    ax.legend()
    for extension, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        path = folder / f"REAL_CONTROLLED_LAG_SHIFT_RECOVERY.{extension}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def _load_or_make_embedding(path: Path, identity: Mapping[str, Any], maker):
    manifest_path = path.with_suffix(".manifest.json")
    if path.is_file() and manifest_path.is_file():
        manifest = _json_read(manifest_path)
        if any(manifest.get(key) != value for key, value in identity.items()) or file_sha256(path) != manifest.get("artifact_sha256"):
            raise LagProtocolError(f"cached lag embedding provenance changed: {path}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        return arrays, manifest
    if path.exists() or manifest_path.exists():
        raise LagProtocolError(f"partial lag embedding; preserving it: {path}")
    arrays, extra = maker()
    manifest = _save_embedding(ROOT, path, arrays, {**identity, **extra})
    return arrays, manifest


def _qualify_trainval_mapping(values: Mapping[str, np.ndarray], raw: Mapping[str, np.ndarray],
                              split: Mapping[str, list[int]], delay: int) -> int:
    """Independently compare remapped safe rows to re-windowed raw B trials."""
    shifted = shift_safe_windows(values, delay)
    spikes = np.asarray(raw["spikes"], dtype=np.float32).reshape(193, 600, 33)
    checks = []
    for trial in (split["train"][0], split["train"][-1], split["validation"][0], split["validation"][-1]):
        original = spikes[trial]
        shifted_trial = np.zeros_like(original)
        shifted_trial[delay:] = original[:-delay]
        # Import the shared center-window builder directly to avoid any alternate preprocessing.
        from neurobridge.sampling.f_windows import build_windows
        shifted_windows, shifted_time, _, _, _ = build_windows(
            shifted_trial, WINDOW_SIZE, 1,
            labels=np.asarray([int(np.asarray(raw["target_by_trial"])[trial])]),
            trial_len=600, time_mode="absolute", padding="center", pad_value=0.0)
        original_windows, original_time, _, _, _ = build_windows(
            original, WINDOW_SIZE, 1,
            labels=np.asarray([int(np.asarray(raw["target_by_trial"])[trial])]),
            trial_len=600, time_mode="absolute", padding="center", pad_value=0.0)
        valid = np.arange(delay + 100, 500, dtype=np.int64)
        if (not np.array_equal(shifted_time, original_time) or
                not np.array_equal(shifted_windows[valid], original_windows[valid - delay])):
            raise LagProtocolError(f"train/validation raw-window equivalence failed for trial {trial}, shift {delay}")
        rows = np.flatnonzero((shifted["trial_id"] == trial) & shifted["lag_valid"])
        if len(rows) != len(valid) or not np.array_equal(shifted["X_windows"][rows], shifted_windows[valid]):
            raise LagProtocolError(f"safe train/validation shifted bundle mismatch for trial {trial}")
        checks.extend((int(trial), int(t)) for t in valid)
    return len(checks)


def _fit_slots(config_by_cell: Mapping[tuple[str, str, int], dict[str, Any]],
               train_paths: Mapping[int, Path], train_manifests: Mapping[int, dict[str, Any]],
               spec_sha: str, parents: Mapping[str, str], *, max_fits: int | None = None) -> list[dict[str, Any]]:
    fits = [(delay, arch, obj, seed) for delay in SHIFTS_MS
            for arch in ARCHITECTURES for obj in OBJECTIVES for seed in SEEDS]
    if max_fits is not None:
        fits = fits[:max_fits]
    completed = []
    result_rows = []
    for ordinal, (delay, arch, obj, seed) in enumerate(fits, start=1):
        base = config_by_cell[(arch, obj, seed)]["config"]
        trial = _make_shift_trial(base, delay, arch, obj, seed)
        fit_parents = {
            **parents,
            "frozen_spec_sha256": spec_sha,
            "r0_B_config_sha256": config_by_cell[(arch, obj, seed)]["source_config_sha256"],
            "r0_B_trial_record_sha256": config_by_cell[(arch, obj, seed)]["source_record_sha256"],
            "r0_B_checkpoint_sha256": config_by_cell[(arch, obj, seed)]["source_checkpoint_sha256"],
            "shifted_trainval_manifest_sha256": file_sha256(train_paths[delay] / "manifest.json"),
            "shifted_trainval_windows_sha256": train_manifests[delay]["shifted_windows_sha256"],
        }
        result = fit_shifted_b_trial(ROOT, trial, train_paths[delay], fit_parents)
        completed.append({"trial": trial, "result": result, "parents": fit_parents})
        result_rows.append({
            "ordinal": ordinal,
            "expected_fits": 72,
            "trial_id": trial["trial_id"],
            "shift_ms": delay,
            "architecture": arch,
            "objective": obj,
            "seed": seed,
            "status": result.get("status"),
            "validation_loss": result.get("validation_loss"),
            "selected_update": result.get("selected_update"),
            "stopping_update": result.get("stopping_update"),
            "training_seconds": result.get("training_seconds"),
            "device": result.get("device"),
            "gpu": result.get("gpu"),
            "near_collapse": result.get("geometry", {}).get("near_collapse"),
        })
        table = ROOT / BRANCH_ROOT / "REFIT_B_FIT_STATUS.csv"
        fields = list(result_rows[0])
        all_rows = []
        for result_path in sorted((ROOT / FITS_ROOT).glob("d*/lagv2-*/result.json")):
            record_path = result_path.parent / "trial_record.json"
            if not record_path.is_file():
                continue
            record = _json_read(record_path)
            r = _json_read(result_path)
            c = record["config"]
            all_rows.append({
                "ordinal": "", "expected_fits": 72, "trial_id": record["trial_id"],
                "shift_ms": c["imposed_shift_bins"], "architecture": c["architecture"],
                "objective": c["objective"], "seed": c["training_seed_root"],
                "status": r.get("status"), "validation_loss": r.get("validation_loss"),
                "selected_update": r.get("selected_update"), "stopping_update": r.get("stopping_update"),
                "training_seconds": r.get("training_seconds"), "device": r.get("device"),
                "gpu": r.get("gpu"), "near_collapse": r.get("geometry", {}).get("near_collapse"),
            })
        _write_csv_atomic(table, all_rows, fields)
        if ordinal % 15 == 0 or ordinal == len(fits):
            print(f"Refit B progress: {ordinal}/{len(fits)} this invocation; {len(all_rows)}/72 verified fit records.", flush=True)
    return completed


def _make_embeddings(project: Path, completed: list[dict[str, Any]], core: Mapping[tuple[str, str, str, int], dict[str, Any]],
                     test_paths: Mapping[int, Path], config_by_cell: Mapping[tuple[str, str, int], dict[str, Any]],
                     checkpoint_by_cell: Mapping[tuple[str, str, int], Path], *, do_refits: bool) -> dict[tuple[str, str, str, int, int], dict[str, np.ndarray]]:
    embeddings = {}
    # Frozen R0 B estimates are reused; exact input-window equivalence is qualified separately.
    for delay in SHIFTS_MS:
        test_values = _load_test_bundle(test_paths[delay], delay)
        for arch in ARCHITECTURES:
            for obj in OBJECTIVES:
                for seed in SEEDS:
                    key = ("B_DISTAL", arch, obj, seed)
                    base = core[key]
                    identity = {
                        "branch": "frozen_R0_encoder",
                        "shift_ms": delay,
                        "population": POPULATION,
                        "architecture": arch,
                        "objective": obj,
                        "seed": seed,
                        "source_trial_id": base["parent_trial_id"],
                        "source_embedding_manifest_sha256": base["parent_manifest_sha256"],
                        "source_embedding_raw_sha256": base["parent_raw_sha256"],
                        "source_embedding_unit_sha256": base["parent_unit_sha256"],
                        "test_input_manifest_sha256": file_sha256(test_paths[delay] / "manifest.json"),
                        "derivation": "no-wrap coordinate remap of frozen B R0 embeddings; exact to the centered-window input shift",
                    }
                    path = project / EMBED_ROOT / "frozen_R0_encoder" / f"shift_{delay:03d}ms" / f"{arch}_{obj}_s{seed}.npz"
                    maker = lambda b=base, tv=test_values, d=delay: (_remap_frozen_b(b, tv, d), {})
                    arrays, _ = _load_or_make_embedding(path, identity, maker)
                    embeddings[("frozen_R0_encoder", arch, obj, seed, delay)] = arrays

    if do_refits:
        for delay in SHIFTS_MS:
            for (arch, obj, seed), parent in config_by_cell.items():
                trial = _make_shift_trial(parent["config"], delay, arch, obj, seed)
                output = Path(trial["output_path"])
                result_path = output / "result.json"
                result = _json_read(result_path) if result_path.is_file() else None
                if result is None or result.get("status", "").startswith("FAILED"):
                    continue
                checkpoint = output / "best_validation_checkpoint.pt"
                if not checkpoint.is_file() or file_sha256(checkpoint) != result.get("artifact_sha256", {}).get("best_validation_checkpoint.pt"):
                    raise LagProtocolError(f"refit B checkpoint missing/hash mismatch: {trial['trial_id']}")
                test_values = _load_test_bundle(test_paths[delay], delay)
                identity = {
                    "branch": "retrained_B_encoder",
                    "shift_ms": delay,
                    "population": POPULATION,
                    "architecture": arch,
                    "objective": obj,
                    "seed": seed,
                    "fit_trial_id": trial["trial_id"],
                    "fit_result_sha256": file_sha256(result_path),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "test_input_manifest_sha256": file_sha256(test_paths[delay] / "manifest.json"),
                }
                path = project / EMBED_ROOT / "retrained_B_encoder" / f"shift_{delay:03d}ms" / f"{arch}_{obj}_s{seed}.npz"
                maker = lambda t=trial, cp=checkpoint, tv=test_values: _infer_checkpoint(t, cp, tv)
                arrays, _ = _load_or_make_embedding(path, identity, maker)
                embeddings[("retrained_B_encoder", arch, obj, seed, delay)] = arrays
    return embeddings


def _qualify_frozen_inference(project: Path, core: Mapping[tuple[str, str, str, int], dict[str, Any]],
                              base_configs: Mapping[tuple[str, str, int], dict[str, Any]],
                              base_checkpoints: Mapping[tuple[str, str, int], Path],
                              test_paths: Mapping[int, Path]) -> dict[str, Any]:
    evidence = {}
    key = ("cnn1d", "soft", 1201)
    config = base_configs[key]["config"]
    trial = {"config": config, "trial_id": base_configs[key]["source_trial_id"]}
    for delay in (0, *SHIFTS_MS):
        test_values = _load_test_bundle(test_paths[delay], delay)
        inferred, timing = _infer_checkpoint(trial, base_checkpoints[key], test_values)
        expected = _remap_frozen_b(core[("B_DISTAL", "cnn1d", "soft", 1201)], test_values, delay)
        mask = inferred["valid_mask"] & expected["valid_mask"]
        raw_error = float(np.max(np.abs(inferred["embedding_raw"][mask] - expected["embedding_raw"][mask])))
        unit_error = float(np.max(np.abs(inferred["embedding_unit"][mask] - expected["embedding_unit"][mask])))
        if (not mask.any() or raw_error > 2e-5 or unit_error > 2e-5 or
                not np.array_equal(inferred["trial_id"][mask], expected["trial_id"][mask]) or
                not np.array_equal(inferred["time_id"][mask], expected["time_id"][mask])):
            raise LagProtocolError(f"frozen encoder/raw-window equivalence failed for shift {delay} ms")
        evidence[str(delay)] = {
            "valid_embedding_rows_compared": int(mask.sum()),
            "raw_max_abs_error": raw_error,
            "unit_max_abs_error": unit_error,
            **timing,
        }
    return evidence


def _qualification_records_equivalent(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Compare immutable qualification facts, excluding repeat-run timing noise."""
    runtime_fields = {"inference_seconds", "inference_valid_windows_per_second"}
    error_fields = {"raw_max_abs_error", "unit_max_abs_error"}
    previous_base = {k: v for k, v in previous.items() if k != "frozen_encoder_input_equivalence"}
    current_base = {k: v for k, v in current.items() if k != "frozen_encoder_input_equivalence"}
    if previous_base != current_base:
        return False
    previous_eq = previous.get("frozen_encoder_input_equivalence")
    current_eq = current.get("frozen_encoder_input_equivalence")
    if not isinstance(previous_eq, Mapping) or not isinstance(current_eq, Mapping) or previous_eq.keys() != current_eq.keys():
        return False
    for delay in previous_eq:
        old, new = previous_eq[delay], current_eq[delay]
        if not isinstance(old, Mapping) or not isinstance(new, Mapping):
            return False
        old_stable = {k: v for k, v in old.items() if k not in runtime_fields | error_fields}
        new_stable = {k: v for k, v in new.items() if k not in runtime_fields | error_fields}
        if old_stable != new_stable:
            return False
        for field in error_fields:
            try:
                if abs(float(old[field])) > 2e-5 or abs(float(new[field])) > 2e-5:
                    return False
            except (KeyError, TypeError, ValueError):
                return False
    return True


def _make_curves(project: Path, core: Mapping[tuple[str, str, str, int], dict[str, Any]],
                 shifted_embeddings: Mapping[tuple[str, str, str, int, int], dict[str, np.ndarray]],
                 fit_results: Mapping[tuple[str, str, int, int], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curve_rows, summary_rows = [], []
    for arch in ARCHITECTURES:
        for objective in OBJECTIVES:
            for seed in SEEDS:
                a_all = core[("A_PROXIMAL", arch, objective, seed)]
                b0_all = core[("B_DISTAL", arch, objective, seed)]
                a = _core_test_rows(a_all)
                b0 = _core_test_rows(b0_all)
                b_branches = {}
                for branch in ("frozen_R0_encoder", "retrained_B_encoder"):
                    for delay in SHIFTS_MS:
                        value = shifted_embeddings.get((branch, arch, objective, seed, delay))
                        if value is not None:
                            b_branches[(branch, delay)] = value
                if len(b_branches) != 2 * len(SHIFTS_MS):
                    continue
                refs = _common_reference(a, b0, b_branches)
                sets = [("R0_baseline", 0, b0, candidate_lags(0), "ELIGIBLE")]
                for branch in ("frozen_R0_encoder", "retrained_B_encoder"):
                    for delay in SHIFTS_MS:
                        shifted = b_branches[(branch, delay)]
                        fit = fit_results.get((delay, arch, objective, seed,)) if branch == "retrained_B_encoder" else None
                        fit_status = (fit or {}).get("status", "FROZEN_R0_REUSED")
                        sets.append((branch, delay, shifted, candidate_lags(delay), fit_status))
                baseline = None
                for branch, delay, b, lags, fit_status in sets:
                    curve = _score_curve(a, b, refs, lags)
                    peak = _curve_summary(curve)
                    if branch == "R0_baseline":
                        baseline = peak
                    for lag, score in curve.items():
                        curve_rows.append({
                            "population_A": "A_PROXIMAL", "population_B": "B_DISTAL",
                            "architecture": arch, "objective": objective, "seed": seed,
                            "branch": branch, "shift_ms": delay, "lag_ms": lag,
                            "procrustes_r2": score, "common_reference_rows": len(refs),
                            "fit_status": fit_status,
                        })
                    if baseline is None and branch != "R0_baseline":
                        raise LagProtocolError("R0 baseline must be summarized before intervention curves")
                    delta = None if branch == "R0_baseline" else peak["lag_hat_ms"] - baseline["lag_hat_ms"]
                    identifiable = (branch != "R0_baseline" and not peak["boundary_censored"]
                                    and baseline is not None and not baseline["boundary_censored"])
                    summary_rows.append({
                        "population_A": "A_PROXIMAL", "population_B": "B_DISTAL",
                        "architecture": arch, "objective": objective, "seed": seed,
                        "branch": branch, "shift_ms": delay,
                        "expected_shift_ms": None if delay == 0 else delay,
                        **peak,
                        "delta_lag_hat_ms": delta,
                        "delta_identifiable": bool(identifiable),
                        "R0_lag_hat_ms": None if branch == "R0_baseline" else baseline["lag_hat_ms"],
                        "common_reference_rows": len(refs),
                        "fit_status": fit_status,
                        "near_collapse": ((fit or {}).get("geometry", {}).get("near_collapse")
                                          if branch == "retrained_B_encoder" else
                                          b0_all.get("near_collapse")),
                    })
    out = project / BRANCH_ROOT
    curve_fields = list(curve_rows[0]) if curve_rows else []
    summary_fields = list(summary_rows[0]) if summary_rows else []
    if curve_rows:
        _write_csv_atomic(out / "REAL_LAG_CURVES_R0_FROZEN_REFIT.csv", curve_rows, curve_fields)
        _write_csv_atomic(out / "REAL_LAG_SUMMARY_R0_FROZEN_REFIT.csv", summary_rows, summary_fields)
    return curve_rows, summary_rows


def _finite_stats(values: list[Any]) -> tuple[int, float | None, float | None, float | None, float | None]:
    finite = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            finite.append(numeric)
    if not finite:
        return 0, None, None, None, None
    sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return len(finite), float(np.mean(finite)), sd, float(np.min(finite)), float(np.max(finite))


def _write_descriptive_tables(project: Path, curve_rows: list[dict[str, Any]],
                              summary_rows: list[dict[str, Any]]) -> list[Path]:
    """Write seed-level descriptive aggregates; no inferential test or new metric."""
    root = project / BRANCH_ROOT
    fit_status_path = root / "REFIT_B_FIT_STATUS.csv"
    with fit_status_path.open("r", newline="", encoding="utf-8") as stream:
        fit_rows = list(csv.DictReader(stream))

    fit_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fit_rows:
        fit_groups[(str(row["shift_ms"]), str(row["architecture"]), str(row["objective"]))].append(row)
    fit_summary = []
    for (shift, architecture, objective), rows in sorted(fit_groups.items()):
        metrics = {
            "validation_loss": [row.get("validation_loss") for row in rows],
            "selected_update": [row.get("selected_update") for row in rows],
            "stopping_update": [row.get("stopping_update") for row in rows],
            "training_seconds": [row.get("training_seconds") for row in rows],
        }
        output: dict[str, Any] = {
            "population": "B_DISTAL", "shift_ms": int(shift),
            "architecture": architecture, "objective": objective,
            "fit_count": len(rows),
            "eligible_count": sum(row.get("status") == "ELIGIBLE" for row in rows),
            "near_collapse_count": sum(row.get("status") == "INELIGIBLE_NEAR_COLLAPSE" for row in rows),
            "failed_count": sum(str(row.get("status", "")).startswith("FAILED") for row in rows),
        }
        for metric, values in metrics.items():
            n, mean, sd, minimum, maximum = _finite_stats(values)
            output[f"{metric}_n"] = n
            output[f"{metric}_mean"] = mean
            output[f"{metric}_sd"] = sd
            output[f"{metric}_min"] = minimum
            output[f"{metric}_max"] = maximum
        fit_summary.append(output)

    baseline_boundary = {
        (str(row["architecture"]), str(row["objective"]), str(row["seed"])): bool(row["boundary_censored"])
        for row in summary_rows if row["branch"] == "R0_baseline"
    }
    lag_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        lag_groups[(str(row["architecture"]), str(row["objective"]),
                    str(row["branch"]), str(row["shift_ms"]))].append(row)
    lag_summary = []
    for (architecture, objective, branch, shift), rows in sorted(lag_groups.items()):
        identifiable_rows = [row for row in rows if row["delta_identifiable"]]
        metrics = {
            "lag_hat_ms": [row.get("lag_hat_ms") for row in rows],
            "S_max": [row.get("S_max") for row in rows],
            "peak_margin": [row.get("peak_margin") for row in rows],
            "delta_lag_hat_ms": [row.get("delta_lag_hat_ms") for row in identifiable_rows],
        }
        output = {
            "population_A": "A_PROXIMAL", "population_B": "B_DISTAL",
            "architecture": architecture, "objective": objective, "branch": branch,
            "shift_ms": int(shift), "seed_count": len(rows),
            "boundary_peak_count": sum(bool(row["boundary_censored"]) for row in rows),
            "identifiable_delta_count": len(identifiable_rows),
            "unresolved_delta_count": 0 if branch == "R0_baseline" else len(rows) - len(identifiable_rows),
            "near_collapse_count": sum(bool(row.get("near_collapse")) for row in rows),
            "r0_boundary_count": (
                sum(bool(row["boundary_censored"]) for row in rows) if branch == "R0_baseline" else
                sum(baseline_boundary.get((architecture, objective, str(row["seed"])), False) for row in rows)
            ),
        }
        for metric, values in metrics.items():
            n, mean, sd, minimum, maximum = _finite_stats(values)
            output[f"{metric}_n"] = n
            output[f"{metric}_mean"] = mean
            output[f"{metric}_sd"] = sd
            output[f"{metric}_min"] = minimum
            output[f"{metric}_max"] = maximum
        lag_summary.append(output)

    curve_groups: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for row in curve_rows:
        key = (str(row["architecture"]), str(row["objective"]), str(row["branch"]),
               str(row["shift_ms"]), str(row["lag_ms"]))
        curve_groups[key].append(float(row["procrustes_r2"]))
    curve_summary = []
    for (architecture, objective, branch, shift, lag), values in sorted(curve_groups.items()):
        n, mean, sd, minimum, maximum = _finite_stats(values)
        curve_summary.append({
            "architecture": architecture, "objective": objective, "branch": branch,
            "shift_ms": int(shift), "lag_ms": int(lag), "seed_count": n,
            "procrustes_r2_mean": mean, "procrustes_r2_sd": sd,
            "procrustes_r2_min": minimum, "procrustes_r2_max": maximum,
        })

    outputs = [
        (root / "REFIT_B_FIT_SUMMARY.csv", fit_summary),
        (root / "REAL_LAG_SEED_AGGREGATES.csv", lag_summary),
        (root / "REAL_LAG_CURVE_SEED_AGGREGATES.csv", curve_summary),
    ]
    for path, rows in outputs:
        if rows:
            _write_csv_atomic(path, rows, list(rows[0]))
    return [path for path, _ in outputs]


def _build_final_report(project: Path, spec: Mapping[str, Any], parents: Mapping[str, str],
                        fit_rows: list[dict[str, Any]], summaries: list[dict[str, Any]],
                        figure_paths: list[str], qualification: Mapping[str, Any],
                        postprocess_provenance_name: str | None = None) -> None:
    root = project / BRANCH_ROOT
    statuses = defaultdict(int)
    for row in fit_rows:
        statuses[row.get("status", "UNKNOWN")] += 1
    interior = [r for r in summaries if r["shift_ms"] in SHIFTS_MS and r["delta_identifiable"]]
    interventions = [r for r in summaries if r["shift_ms"] in SHIFTS_MS]
    baseline = [r for r in summaries if r["branch"] == "R0_baseline"]
    intervention_boundary = sum(bool(r["boundary_censored"]) for r in interventions)
    baseline_boundary = sum(bool(r["boundary_censored"]) for r in baseline)
    collapse_rows = []
    fit_status_path = root / "REFIT_B_FIT_STATUS.csv"
    if fit_status_path.is_file():
        with fit_status_path.open("r", newline="", encoding="utf-8") as stream:
            collapse_rows = [row for row in csv.DictReader(stream)
                             if row.get("status") == "INELIGIBLE_NEAR_COLLAPSE"]
    identifiable_text = [
        f"{r['architecture']}/{r['objective']}/seed {r['seed']}/{r['branch']}: shift {r['shift_ms']} ms, delta {r['delta_lag_hat_ms']} ms"
        for r in interior
    ]
    lines = [
        "# Real controlled-lag V2",
        "",
        "This is a separate, immutable controlled digital-intervention branch. It does not modify the frozen Real R0 fits or core metrics.",
        "",
        "## Protocol",
        "",
        f"- Canonical Real window: {spec['selected_real_window_size']} ms.",
        f"- B-only delays: {', '.join(map(str, SHIFTS_MS))} ms; no circular wrap; invalid centered windows excluded.",
        "- R0 scans -20..+20 ms; each intervention scans its predeclared local 41-point curve d-20..d+20 ms.",
        "- Curves use one held-out trial/time reference support shared by R0 and every shift in both branches.",
        "- Frozen branch reuses the saved B R0 embedding rows; refit branch trains only B from random initialization. A remains the paired frozen R0 anchor.",
        "- Frozen hyperparameters, optimizer, batch, early stopping, channel partition, split, objectives, architectures, and seeds are inherited unchanged from the sealed Real spec.",
        "- No test-based HPO, model choice, or hyperparameter decision. No biological/causal lag interpretation.",
        "",
        "## Completion",
        "",
        f"- Planned fresh B fits: {len(SHIFTS_MS)*len(ARCHITECTURES)*len(OBJECTIVES)*len(SEEDS)}.",
        f"- Fit statuses: {dict(statuses)}.",
        f"- Identifiable intervention deltas: {len(interior)}/{len(interventions)}; unresolved/censored deltas: {len(interventions) - len(interior)}/{len(interventions)}.",
        f"- Boundary peaks: R0 {baseline_boundary}/{len(baseline)}; interventions {intervention_boundary}/{len(interventions)}.",
        "- Identifiable cases (descriptive only): " + ("; ".join(identifiable_text) if identifiable_text else "none"),
        "- Near-collapse fits are retained and flagged: " + ("; ".join(
            f"{r.get('architecture')}/{r.get('objective')}/seed {r.get('seed')}/shift {r.get('shift_ms')} ms"
            for r in collapse_rows) if collapse_rows else "none"),
        "- Seed summaries use n=3 and are descriptive (mean/SD/range); no inferential p-values or population confidence intervals are claimed.",
        "- Per-seed lag summary: `REAL_LAG_SUMMARY_R0_FROZEN_REFIT.csv`; seed aggregates: `REAL_LAG_SEED_AGGREGATES.csv`.",
        "- Per-seed lag curves: `REAL_LAG_CURVES_R0_FROZEN_REFIT.csv`; curve mean/SD by seed: `REAL_LAG_CURVE_SEED_AGGREGATES.csv`.",
        "- Fit-level rows: `REFIT_B_FIT_STATUS.csv`; descriptive fit aggregate: `REFIT_B_FIT_SUMMARY.csv`.",
        "",
        "## Provenance",
        "",
        f"- Frozen spec SHA-256: `{parents['frozen_spec_sha256']}`.",
        f"- Raw source SHA-256: `{parents['raw_source_sha256']}`.",
        f"- Frozen split SHA-256: `{parents['original_split_sha256']}`.",
        f"- Somatotopic partition SHA-256: `{parents['partition_sha256']}`.",
        "- Frozen encoder equivalence qualification (max absolute errors):",
        "",
        "```json",
        json.dumps(qualification, indent=2, sort_keys=True),
        "```",
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{Path(path).name}`" for path in figure_paths)
    lines.extend([
        "",
        "Per-seed curves and estimates are in `REAL_LAG_CURVES_R0_FROZEN_REFIT.csv` and `REAL_LAG_SUMMARY_R0_FROZEN_REFIT.csv`; fresh-fit status is in `REFIT_B_FIT_STATUS.csv`.",
        f"Post-processing hashes and proof of zero training/metric recomputation: `{postprocess_provenance_name or 'REAL_CONTROLLED_LAG_V2_POSTPROCESS_PROVENANCE_*.json'}`.",
        "",
        "A boundary peak is censored/unresolved. `Delta lag` is compared to the imposed digital shift only when both the R0 and intervention peaks are interior. Most estimates are unresolved here; the four identifiable rows are concentrated in Transformer/behavior-contrastive seed 1201, so they are not broad evidence of robustness. These analyses measure response to a digital temporal intervention, not biological delay or causality.",
        "",
    ])
    path = root / "REAL_CONTROLLED_LAG_V2_REPORT.md"
    temp = path.with_suffix(".md.partial")
    temp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temp, path)


def _postprocess_cached_outputs(project: Path) -> None:
    """Regenerate descriptive tables/report/figures from frozen outputs only."""
    root = project / BRANCH_ROOT
    curve_path = root / "REAL_LAG_CURVES_R0_FROZEN_REFIT.csv"
    summary_path = root / "REAL_LAG_SUMMARY_R0_FROZEN_REFIT.csv"
    status_path = root / "REFIT_B_FIT_STATUS.csv"
    provenance_path = root / "REAL_CONTROLLED_LAG_V2_PROVENANCE_072.json"
    qualification_path = root / "QUALIFICATION.json"
    required = (curve_path, summary_path, status_path, provenance_path, qualification_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise LagProtocolError(f"postprocess-only requires completed frozen outputs: {missing}")

    with curve_path.open("r", newline="", encoding="utf-8") as stream:
        curve_rows = list(csv.DictReader(stream))
    with summary_path.open("r", newline="", encoding="utf-8") as stream:
        summary_rows = list(csv.DictReader(stream))
    with status_path.open("r", newline="", encoding="utf-8") as stream:
        fit_rows = list(csv.DictReader(stream))

    def number(value: Any, cast):
        return None if value in (None, "") else cast(value)

    def boolean(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "yes"}

    for row in curve_rows:
        for field in ("seed", "shift_ms", "lag_ms", "common_reference_rows"):
            row[field] = int(row[field])
        row["procrustes_r2"] = float(row["procrustes_r2"])
    for row in summary_rows:
        for field in ("seed", "shift_ms", "lag_hat_ms", "lag_min_ms", "lag_max_ms", "common_reference_rows"):
            row[field] = int(row[field])
        for field in ("expected_shift_ms", "S_max", "peak_margin", "delta_lag_hat_ms", "R0_lag_hat_ms"):
            row[field] = number(row[field], float if field in {"S_max", "peak_margin", "delta_lag_hat_ms", "R0_lag_hat_ms"} else int)
        for field in ("boundary_censored", "delta_identifiable", "near_collapse"):
            row[field] = boolean(row[field]) if row[field] not in (None, "") else None

    results = sorted((project / FITS_ROOT).glob("d*/lagv2-*/result.json"))
    if len(results) != 72 or len(fit_rows) != 72 or len(curve_rows) != 6888 or len(summary_rows) != 168:
        raise LagProtocolError(
            f"postprocess-only invariant mismatch: fit_results={len(results)}, fit_status={len(fit_rows)}, "
            f"curve_rows={len(curve_rows)}, summary_rows={len(summary_rows)}"
        )
    for result_path in results:
        record_path = result_path.parent / "trial_record.json"
        if not record_path.is_file():
            raise LagProtocolError(f"fit provenance is missing: {record_path}")
        record, result = _json_read(record_path), _json_read(result_path)
        if record.get("trial_id") != result.get("trial_id"):
            raise LagProtocolError(f"fit record/result ID mismatch: {result_path.parent.name}")
        for name, digest in result.get("artifact_sha256", {}).items():
            artifact = result_path.parent / name
            if not artifact.is_file() or file_sha256(artifact) != digest:
                raise LagProtocolError(f"fit artifact hash mismatch: {artifact}")

    embedding_counts = {}
    embedding_artifact_hashes = {}
    for branch in ("frozen_R0_encoder", "retrained_B_encoder"):
        paths = sorted((project / EMBED_ROOT / branch).glob("shift_*ms/*.npz"))
        if len(paths) != 72:
            raise LagProtocolError(f"expected 72 cached {branch} test embeddings, found {len(paths)}")
        for path in paths:
            manifest_path = path.with_suffix(".manifest.json")
            if not manifest_path.is_file():
                raise LagProtocolError(f"cached embedding manifest is missing: {manifest_path}")
            manifest = _json_read(manifest_path)
            artifact_digest = file_sha256(path)
            if (manifest.get("artifact_sha256") != artifact_digest or
                    manifest.get("branch") != branch or manifest.get("population") != "B_DISTAL"):
                raise LagProtocolError(f"cached embedding manifest identity/hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            n = len(arrays.get("trial_id", ()))
            required_arrays = ("embedding_raw", "embedding_unit", "trial_id", "time_id", "global_time_id", "valid_mask", "split")
            if (any(key not in arrays or len(arrays[key]) != n for key in required_arrays) or
                    arrays["embedding_raw"].shape != (n, 3) or arrays["embedding_unit"].shape != (n, 3) or
                    manifest.get("embedding_shape") != [n, 3] or manifest.get("row_count") != n or
                    set(np.unique(arrays["split"])) != {"test"} or
                    not np.isfinite(arrays["embedding_raw"]).all() or not np.isfinite(arrays["embedding_unit"]).all()):
                raise LagProtocolError(f"cached embedding metadata/cardinality/shape invalid: {path}")
            valid = np.asarray(arrays["valid_mask"], dtype=bool)
            if valid.any() and not np.allclose(np.linalg.norm(arrays["embedding_unit"][valid], axis=1), 1.0, atol=2e-4):
                raise LagProtocolError(f"cached unit embedding rows are not unit-normalized: {path}")
            embedding_artifact_hashes[str(path.relative_to(project))] = artifact_digest
            embedding_artifact_hashes[str(manifest_path.relative_to(project))] = file_sha256(manifest_path)
    embedding_counts = {branch: 72 for branch in ("frozen_R0_encoder", "retrained_B_encoder")}

    support_groups: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    curve_grid: dict[tuple[str, str, int, str, int], set[int]] = defaultdict(set)
    for row in curve_rows:
        support_groups[(row["architecture"], row["objective"], row["seed"])].add(row["common_reference_rows"])
        curve_grid[(row["architecture"], row["objective"], row["seed"], row["branch"], row["shift_ms"])].add(row["lag_ms"])
    if len(support_groups) != 24 or any(len(values) != 1 for values in support_groups.values()):
        raise LagProtocolError("curve rows do not retain one common held-out support per architecture/objective/seed")
    if len(curve_grid) != 168 or any(len(values) != 41 for values in curve_grid.values()):
        raise LagProtocolError("curve lag grids are incomplete or inconsistent")

    descriptive_paths = _write_descriptive_tables(project, curve_rows, summary_rows)
    figure_paths = _plot_results(project, curve_rows, summary_rows)
    provenance = _json_read(provenance_path)
    spec = _json_read(project / SPEC_PATH)
    qualification = _json_read(qualification_path)
    runner_digest = file_sha256(Path(__file__))
    snapshot_path = root / f"REAL_CONTROLLED_LAG_V2_POSTPROCESS_PROVENANCE_{runner_digest[:12]}.json"
    _build_final_report(project, spec, provenance, fit_rows, summary_rows, figure_paths, qualification,
                        postprocess_provenance_name=snapshot_path.name)

    input_paths = (curve_path, summary_path, status_path, provenance_path, qualification_path)
    output_paths = [*descriptive_paths, *[Path(path) for path in figure_paths], root / "REAL_CONTROLLED_LAG_V2_REPORT.md"]
    snapshot = {
        "postprocess_id": "controlled_lag_v2_descriptive_summary_and_figure_layout_fix_v2",
        "runner_source_sha256": runner_digest,
        "fit_records_verified": len(results),
        "training_performed": False,
        "raw_or_test_input_containers_loaded": False,
        "saved_test_embeddings_loaded_for_integrity_check": True,
        "metric_recomputation": False,
        "metric_definitions_changed": False,
        "core_metrics_modified": False,
        "embedding_archives_verified": embedding_counts,
        "embedding_artifact_manifest_hashes": embedding_artifact_hashes,
        "per_seed_metric_inputs": {str(path.relative_to(project)): file_sha256(path) for path in input_paths},
        "postprocess_outputs": {str(path.relative_to(project)): file_sha256(path) for path in output_paths},
    }
    if snapshot_path.exists():
        if _json_read(snapshot_path) != snapshot:
            raise LagProtocolError("postprocess provenance changed; preserving the existing snapshot")
    else:
        _write_json_immutable(snapshot_path, snapshot)
    print(
        "Postprocess-only complete: verified 72 fit records and 144 embeddings; "
        "training=0, raw/test inputs loaded=0, existing per-seed metrics unchanged.", flush=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="verify inputs/print plan without writing or fitting")
    parser.add_argument("--prepare-only", action="store_true", help="write immutable shifted trainval/test inputs; no model fitting")
    parser.add_argument("--qualify-only", action="store_true", help="prepare inputs and run source/split/window/frozen-embedding invariants; no model fitting")
    parser.add_argument("--run-all", action="store_true", help="run the frozen 72-fit B-only campaign")
    parser.add_argument("--postprocess-only", action="store_true", help="verify frozen outputs and regenerate descriptive tables/report/figures without loading data or fitting")
    parser.add_argument("--max-fits", type=int, default=None, help="bounded diagnostic subset of the campaign")
    parser.add_argument("--resume", action="store_true", help="reuse verified complete fits and artifacts under this branch")
    args = parser.parse_args()
    if sum((args.dry_run, args.prepare_only, args.qualify_only, args.run_all, args.postprocess_only)) != 1:
        parser.error("select exactly one of --dry-run, --prepare-only, --qualify-only, --run-all, --postprocess-only")
    if args.max_fits is not None and (args.max_fits < 1 or args.max_fits > 72):
        parser.error("--max-fits must be between 1 and 72")

    project = ROOT.resolve()
    if args.postprocess_only:
        _postprocess_cached_outputs(project)
        return
    spec_path = project / SPEC_PATH
    if not spec_path.is_file():
        raise SystemExit(f"missing frozen Real specification: {spec_path}")
    spec_sha = file_sha256(spec_path)
    spec = _json_read(spec_path)
    if (spec.get("selected_real_window_size") != WINDOW_SIZE or
            spec.get("final_training_seed_roots") != list(SEEDS) or
            spec.get("somatotopic_partition_sha256") != EXPECTED_PARTITION_SHA or
            spec.get("original_frozen_split_sha256") != EXPECTED_SPLIT_SHA):
        raise SystemExit("frozen Real spec window/split/partition/seed identity differs")
    config, raw, split, safe_manifest, values, parents = load_verified_parents(project, spec)
    parents = {**parents, "frozen_spec_sha256": spec_sha}
    if parents["original_split_sha256"] != EXPECTED_SPLIT_SHA or parents["partition_sha256"] != EXPECTED_PARTITION_SHA:
        raise SystemExit("frozen split or channel partition hash differs")
    core = _load_core_embedding_index(project, split)
    config_by_cell, checkpoint_by_cell = _load_base_fit_records(project, spec)
    if args.dry_run:
        print("No outputs written and no model fitting performed.")
        print(f"Frozen source and split verified: {parents['raw_source_sha256']} / {parents['original_split_sha256']}")
        print(f"Held-out trials: {len(split['test'])}; window=201; shifts={SHIFTS_MS}; seeds={SEEDS}")
        print(f"Planned fresh B fits: {len(SHIFTS_MS)*len(ARCHITECTURES)*len(OBJECTIVES)*len(SEEDS)}")
        print(f"Planned outputs: {(project / BRANCH_ROOT).resolve()}")
        print("A fits=0; HPO=0; core metrics modified=0; V2 paths writable=0.")
        return

    study_spec = _ensure_study_manifest(project, spec, parents)
    train_paths, test_paths, train_manifests, test_manifests = _prepare_inputs(
        project, spec, config, raw, split, safe_manifest, values, parents)
    if args.prepare_only:
        print(f"Prepared immutable shifted inputs for {SHIFTS_MS}; train/validation-only fit rows and isolated test-only input rows.")
        print(f"Branch: {(project / BRANCH_ROOT).resolve()}")
        return

    # Recheck test trial IDs using the frozen embeddings and qualify the raw-window mapping.
    split_rows = 0
    for delay in SHIFTS_MS:
        split_rows += _qualify_trainval_mapping(values, raw, split, delay)
    frozen_equivalence = _qualify_frozen_inference(project, core, config_by_cell, checkpoint_by_cell, test_paths)
    qualification = {
        "source_sha256_verified": parents["raw_source_sha256"],
        "frozen_split_sha256_verified": parents["original_split_sha256"],
        "partition_sha256_verified": parents["partition_sha256"],
        "rewindowed_trainval_trial_checks": split_rows,
        "test_trial_count": len(split["test"]),
        "test_rows_absent_from_fit_bundle": True,
        "local_lag_curves_ms": {str(d): [min(candidate_lags(d)), max(candidate_lags(d))] for d in SHIFTS_MS},
        "frozen_encoder_input_equivalence": frozen_equivalence,
    }
    qualify_path = project / BRANCH_ROOT / "QUALIFICATION.json"
    if qualify_path.exists():
        current = _json_read(qualify_path)
        if not _qualification_records_equivalent(current, qualification):
            raise LagProtocolError("existing qualification record differs; preserving it")
        # Keep the first immutable measurements; repeat-run timing is not identity.
        qualification = current
    else:
        _write_json_immutable(qualify_path, qualification)
    print("Source/split/window/frozen-embedding qualification PASSED.", flush=True)
    if args.qualify_only:
        print("Qualification only: no shifted B models trained.")
        return

    planned = 72 if args.max_fits is None else args.max_fits
    completed = _fit_slots(config_by_cell, train_paths, train_manifests, spec_sha, parents, max_fits=args.max_fits)
    fit_rows = []
    fit_results = {}
    for item in completed:
        trial = item["trial"]
        result = item["result"]
        c = trial["config"]
        fit_results[(c["imposed_shift_bins"], c["architecture"], c["objective"], c["training_seed_root"])] = result
        fit_rows.append({"trial_id": trial["trial_id"], "status": result.get("status"),
                         "validation_loss": result.get("validation_loss"),
                         "optimizer_updates": result.get("stopping_update"),
                         "training_seconds": result.get("training_seconds")})
        if result.get("status", "").startswith("FAILED"):
            print(f"Fit failed; recorded and continuing: {trial['trial_id']} ({result.get('status')})", flush=True)
    # Include verified prior rows on resume / bounded runs.
    for result_path in sorted((project / FITS_ROOT).glob("d*/lagv2-*/result.json")):
        record_path = result_path.parent / "trial_record.json"
        if not record_path.is_file():
            continue
        record = _json_read(record_path)
        result = _json_read(result_path)
        c = record["config"]
        fit_results[(c["imposed_shift_bins"], c["architecture"], c["objective"], c["training_seed_root"])] = result

    # Persist frozen-encoder remaps and test embeddings for each completed B refit.
    shifted_embeddings = _make_embeddings(project, completed, core, test_paths,
                                          config_by_cell, checkpoint_by_cell, do_refits=True)
    curve_rows, summary_rows = _make_curves(project, core, shifted_embeddings, fit_results)
    descriptive_paths = _write_descriptive_tables(project, curve_rows, summary_rows)
    figure_paths = _plot_results(project, curve_rows, summary_rows) if curve_rows else []

    provenance = {
        "study_spec_sha256": file_sha256(project / BRANCH_ROOT / "CONTROLLED_LAG_V2_SPEC.json"),
        "qualification_sha256": file_sha256(qualify_path),
        "frozen_spec_sha256": spec_sha,
        "raw_source_sha256": parents["raw_source_sha256"],
        "original_split_sha256": parents["original_split_sha256"],
        "partition_sha256": parents["partition_sha256"],
        "safe_windows_sha256": parents["safe_windows_sha256"],
        "core_metric_source_sha256": file_sha256(project / CORE_SOURCE),
        "downstream_source_sha256": file_sha256(project / DOWNSTREAM_SOURCE),
        "runner_source_sha256": file_sha256(Path(__file__)),
        "input_extractor_source_sha256": file_sha256(ROOT / "src/neurobridge/experiments/final_real_lag_revision.py"),
        "fit_adapter_source_sha256": file_sha256(ROOT / "src/neurobridge/experiments/final_real_lag_fit.py"),
        "training_core_source_sha256": file_sha256(project / "src/neurobridge/experiments/phase2a_safe_fit.py"),
        "shift_ms": list(SHIFTS_MS),
        "new_B_fit_plan": 72,
        "verified_fit_record_count": len(list((project / FITS_ROOT).glob("d*/lagv2-*/result.json"))),
        "complete_campaign_requested": args.run_all,
        "test_metric_used_for_model_selection": False,
        "core_metrics_modified": False,
    }
    prov_path = project / BRANCH_ROOT / "REAL_CONTROLLED_LAG_V2_PROVENANCE.json"
    if prov_path.exists():
        current = _json_read(prov_path)
        if current != provenance:
            # Final campaign status may grow during resume; keep immutable snapshots by completed count.
            prov_path = prov_path.with_name(f"REAL_CONTROLLED_LAG_V2_PROVENANCE_{provenance['verified_fit_record_count']:03d}.json")
            if not prov_path.exists():
                _write_json_immutable(prov_path, provenance)
    else:
        _write_json_immutable(prov_path, provenance)
    _build_final_report(project, spec, parents, fit_rows, summary_rows, figure_paths, qualification)
    count = len(list((project / FITS_ROOT).glob("d*/lagv2-*/result.json")))
    postprocess_inputs = [
        project / BRANCH_ROOT / "REFIT_B_FIT_STATUS.csv",
        project / BRANCH_ROOT / "REAL_LAG_CURVES_R0_FROZEN_REFIT.csv",
        project / BRANCH_ROOT / "REAL_LAG_SUMMARY_R0_FROZEN_REFIT.csv",
    ]
    postprocess_outputs = [*descriptive_paths, *[Path(path) for path in figure_paths],
                           project / BRANCH_ROOT / "REAL_CONTROLLED_LAG_V2_REPORT.md"]
    postprocess_record = {
        "postprocess_id": "controlled_lag_v2_descriptive_summary_and_figure_layout_fix_v1",
        "runner_source_sha256": file_sha256(Path(__file__)),
        "fit_records_verified": count,
        "training_performed": False,
        "core_metrics_modified": False,
        "per_seed_metric_inputs": {
            str(path.relative_to(project)): file_sha256(path) for path in postprocess_inputs
        },
        "postprocess_outputs": {
            str(path.relative_to(project)): file_sha256(path) for path in postprocess_outputs
        },
    }
    postprocess_path = project / BRANCH_ROOT / "REAL_CONTROLLED_LAG_V2_POSTPROCESS_PROVENANCE_001.json"
    if postprocess_path.exists():
        if _json_read(postprocess_path) != postprocess_record:
            raise LagProtocolError("existing postprocess provenance differs; preserving it")
    else:
        _write_json_immutable(postprocess_path, postprocess_record)
    print(f"Campaign invocation complete: {count}/72 immutable fit records; requested this pass={planned}.", flush=True)


if __name__ == "__main__":
    main()
