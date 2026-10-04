"""Train shifted B encoders from train/validation-only immutable bundles.

This adapter has no loader for the original source container, test bundle,
test indices, embeddings, or scientific evaluation outputs. It delegates the
frozen optimizer/validation/early-stopping semantics to Phase-2A's fit core.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from neurobridge.data.dataset import TemporalWindowDataset
from neurobridge.experiments.phase2a_hpo import (
    ProtocolViolation,
    _canonical_json,
    file_sha256,
)
from neurobridge.experiments.phase2a_safe_fit import _fit_new


FIT_ROOT = Path("outputs/final_thesis_v1/final_evaluation/controlled_lag_v2/fits/refit")
SHIFTS = (100, 160, 200)
ARCHITECTURES = ("cnn1d", "transformer")
OBJECTIVES = ("soft", "infonce", "time_contrastive_blocks", "behavior_contrastive_blocks")
SEEDS = (1101, 1201, 1301)


def _load_trainval_bundle(bundle_root: Path, shift_ms: int):
    root = bundle_root.resolve()
    manifest_path = root / "manifest.json"
    windows_path = root / "trainval_windows.npz"
    split_path = root / "split.json"
    if not all(path.is_file() for path in (manifest_path, windows_path, split_path)):
        raise ProtocolViolation("shifted train/validation bundle is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("shift_ms") != shift_ms or
            manifest.get("population") != "B_DISTAL" or
            manifest.get("test_trial_intersection_empty") is not True or
            manifest.get("fit_accessible") is not True or
            file_sha256(windows_path) != manifest.get("shifted_windows_sha256") or
            file_sha256(split_path) != manifest.get("safe_split_sha256")):
        raise ProtocolViolation("shifted train/validation bundle identity/hash/firewall failed")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if (set(split) != {"train", "validation"} or
            split["train"] != manifest.get("train_trial_ids") or
            split["validation"] != manifest.get("validation_trial_ids")):
        raise ProtocolViolation("fit split must contain only frozen train and validation IDs")
    with np.load(windows_path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    count = len(values.get("trial_id", ()))
    if (values.get("X_windows", np.empty((0,))).shape != (count, 201, 33) or
            any(len(value) != count for value in values.values()) or
            set(np.unique(values["split"])) != {"train", "validation"} or
            set(np.unique(values["trial_id"])) != set(split["train"] + split["validation"]) or
            np.any(~np.isin(values["trial_id"], split["train"] + split["validation"]))):
        raise ProtocolViolation("fit-accessible bundle contains unknown/test rows or malformed tensors")
    if (np.any(values["lag_valid"] & (values["time_id"] < shift_ms + 100)) or
            np.any(values["lag_valid"] & (values["time_id"] >= 500))):
        raise ProtocolViolation("shifted validity mask contains an invalid window center")
    dataset = TemporalWindowDataset(
        values["X_windows"], values["time_id"], values["global_time_id"],
        values["trial_id"], labels_windows=values["labels"],
        extra_metadata={
            "progress": values["progress"],
            "lag_valid": values["lag_valid"].astype(np.int64),
            "position": values["position"],
            "velocity": values["velocity"],
        },
    )
    return manifest, split, values, dataset, manifest_path, windows_path, split_path


def fit_shifted_b_trial(project_root: Path, trial: Mapping[str, Any],
                        bundle_root: Path, parent_provenance: Mapping[str, Any]) -> dict:
    """Fit/reuse exactly one immutable shifted-B fit using train/validation only."""
    project = project_root.resolve()
    row = trial.get("config")
    if not isinstance(row, dict):
        raise ProtocolViolation("shifted Real fit config is missing")
    shift = row.get("imposed_shift_bins")
    if (shift not in SHIFTS or row.get("population") != "B_DISTAL" or
            row.get("branch") != "held_out" or row.get("domain") != "real_w201" or
            row.get("window_size") != 201 or row.get("batch_size") != 1024 or
            row.get("positive_offset") != 10 or row.get("training_seed_root") not in SEEDS or
            row.get("training_seed_effective") != row.get("training_seed_root") or
            row.get("architecture") not in ARCHITECTURES or row.get("objective") not in OBJECTIVES):
        raise ProtocolViolation("shifted fit differs from frozen Real architecture/objective/training protocol")
    expected_id = str(trial.get("trial_id", ""))
    if not expected_id.startswith(f"lagv2-refit-w201-B_DISTAL-{row['architecture']}-{row['objective']}-d{shift}-s{row['training_seed_root']}-"):
        raise ProtocolViolation("shifted fit trial ID does not encode its frozen cell identity")
    config_sha = hashlib.sha256(_canonical_json(row).encode("utf-8")).hexdigest()
    if trial.get("config_sha256") != config_sha:
        raise ProtocolViolation("shifted Real fit config hash mismatch")
    output = Path(trial["output_path"]).resolve()
    expected_parent = (project / FIT_ROOT / f"d{shift}").resolve()
    if expected_parent not in output.parents:
        raise ProtocolViolation("shifted fit output escaped its immutable lag branch")
    if (project / "outputs/runs").resolve() in output.parents:
        raise ProtocolViolation("shifted fit cannot write into frozen V2 run directories")

    bundle_root = bundle_root.resolve()
    allowed_bundle_root = (project / "outputs/final_thesis_v1/final_evaluation/controlled_lag_v2/inputs/train_validation").resolve()
    if allowed_bundle_root not in bundle_root.parents:
        raise ProtocolViolation("fit adapter accepts only its immutable train/validation lag bundle")
    manifest, split, values, dataset, manifest_path, windows_path, split_path = _load_trainval_bundle(bundle_root, shift)
    record = {
        "trial_id": expected_id,
        "config": row,
        "config_sha256": config_sha,
        "parent_provenance": dict(parent_provenance),
        "shifted_bundle_path": str(bundle_root),
        "shifted_manifest_sha256": file_sha256(manifest_path),
        "shifted_windows_sha256": file_sha256(windows_path),
        "shifted_split_sha256": file_sha256(split_path),
        "training_core_source_sha256": file_sha256(Path(__file__).parents[0] / "phase2a_safe_fit.py"),
        "fit_adapter_source_sha256": file_sha256(Path(__file__)),
        "test_data_accessed_by_fit_adapter": False,
    }
    record_path = output / "trial_record.json"
    result_path = output / "result.json"
    if output.exists():
        if not record_path.is_file() or json.loads(record_path.read_text(encoding="utf-8")) != record:
            raise ProtocolViolation("existing shifted fit has different provenance; preserving it")
        if not result_path.is_file():
            raise ProtocolViolation("existing shifted fit is incomplete; preserving it without implicit retry")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for name, digest in result.get("artifact_sha256", {}).items():
            path = output / name
            if not path.is_file() or file_sha256(path) != digest:
                raise ProtocolViolation("existing shifted fit artifact hash mismatch")
        return result

    output.mkdir(parents=True, exist_ok=False)
    with record_path.open("x", encoding="utf-8") as stream:
        stream.write(_canonical_json(record) + "\n")
    try:
        result = _fit_new(output, trial, dataset, values, split, None)
    except Exception as exc:
        status = "FAILED_NUMERICAL" if isinstance(exc, FloatingPointError) else "FAILED_RUNTIME"
        result = {
            "trial_id": expected_id,
            "status": status,
            "error_class": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=12),
            "optimizer_updates": None,
            "artifact_sha256": {},
        }
    with result_path.open("x", encoding="utf-8") as stream:
        stream.write(_canonical_json(result) + "\n")
    return result
