"""Safe-bundle-only Real fit adapter for the versioned somatotopic study.

This module has no import of the trusted extractor and no path to a V2 parent
container or complete split. Optimization is delegated unchanged to the
frozen Phase-2A training core.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurobridge.data.dataset import TemporalWindowDataset  # noqa: E402
from neurobridge.experiments.phase2a_hpo import (  # noqa: E402
    ProtocolViolation, _canonical_json, file_sha256, guarded_output_root,
)
from neurobridge.experiments.phase2a_safe_fit import _fit_new  # noqa: E402
from neurobridge.experiments.phase2a_window_inputs import WINDOWS  # noqa: E402

VERSION = "real_somatotopic_v1"
POPULATIONS = ("TOTAL65", "A_PROXIMAL", "B_DISTAL")


def load_safe_real_bundle(project_root: Path, domain: str, population: str,
                          partition_sha256: str, design_sha256: str):
    """Only the existing/new train-validation-only HPO bundles are readable."""
    if domain not in {f"real_w{window}" for window in WINDOWS} or population not in POPULATIONS:
        raise ProtocolViolation("unknown somatotopic safe-bundle identity")
    hpo = guarded_output_root(project_root)
    alias = hpo / "safe_inputs" / VERSION / domain / population / "reference_manifest.json"
    reference = json.loads(alias.read_text(encoding="utf-8"))
    if (reference["domain"] != domain or reference["population"] != population or
            reference["partition_version"] != VERSION or
            reference["partition_sha256"] != partition_sha256 or
            reference["canonical_design_sha256"] != design_sha256):
        raise ProtocolViolation("safe reference identity/partition changed")
    root = (hpo / "safe_inputs" / domain / "TOTAL65" if population == "TOTAL65" else
            hpo / "safe_inputs" / VERSION / domain / population)
    if root.resolve() != Path(reference["bundle_path"]).resolve():
        raise ProtocolViolation("safe reference points elsewhere")
    if (hpo not in root.resolve().parents or (hpo / "safe_inputs") not in root.resolve().parents or
            "test" in root.parts):
        raise ProtocolViolation("safe bundle escaped HPO tree or names test")
    manifest_path, windows_path, split_path = root / "manifest.json", root / "windows.npz", root / "split.json"
    if not all(path.is_file() for path in (manifest_path, windows_path, split_path)):
        raise ProtocolViolation("safe bundle incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["domain"] != domain or manifest["population"] != population or
            file_sha256(manifest_path) != reference["bundle_manifest_sha256"] or
            file_sha256(windows_path) != manifest["safe_windows_sha256"] or
            file_sha256(split_path) != manifest["safe_split_sha256"]):
        raise ProtocolViolation("safe bundle identity/content hash invalid")
    if population != "TOTAL65" and (manifest["partition_version"] != VERSION or
                                    manifest["partition_sha256"] != partition_sha256):
        raise ProtocolViolation("somatotopic bundle partition mismatch")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if (set(split) != {"train", "validation"} or split["train"] != manifest["train_trial_ids"] or
            split["validation"] != manifest["validation_trial_ids"] or
            not manifest["test_trial_intersection_empty"] or
            not manifest["all_windows_are_train_or_validation"]):
        raise ProtocolViolation("safe train/validation split invalid")
    with np.load(windows_path, allow_pickle=False) as source:
        values = {key: source[key] for key in source.files}
    count = len(values["trial_id"])
    if (values["X_windows"].shape != (count, int(domain[6:]), 65 if population == "TOTAL65" else
                                      32 if population == "A_PROXIMAL" else 33) or
            any(len(values[key]) != count for key in values) or
            set(np.unique(values["split"])) != {"train", "validation"} or
            set(np.unique(values["trial_id"])) != set(split["train"] + split["validation"]) or
            not np.array_equal(values["split"], np.where(np.isin(values["trial_id"], split["train"]),
                                                          "train", "validation"))):
        raise ProtocolViolation("safe bundle shape/cardinality/firewall invalid")
    dataset = TemporalWindowDataset(
        values["X_windows"], values["time_id"], values["global_time_id"], values["trial_id"],
        labels_windows=values["labels"],
        extra_metadata={"progress": values["progress"], "lag_valid": values["lag_valid"].astype(np.int64),
                        "position": values["position"], "velocity": values["velocity"]},
    )
    return root, alias, manifest, split, dataset, values


def fit_real_trial(project_root: Path, trial: Mapping[str, Any], *, smoke_updates: int | None = None) -> dict:
    project = project_root.resolve()
    config = trial["config"]
    root, alias, safe_manifest, split, dataset, values = load_safe_real_bundle(
        project, config["domain"], config["population"],
        config["channel_partition_sha256"], config["canonical_design_sha256"],
    )
    output_root = guarded_output_root(project)
    target = (Path(trial["output_path"]) if smoke_updates is None else
              output_root / "smoke" / VERSION / f"{trial['trial_id']}-u{smoke_updates}")
    if output_root not in target.resolve().parents:
        raise ProtocolViolation("fit output escaped HPO root")
    record = {"trial_id": trial["trial_id"], "config": config,
              "config_sha256": trial["config_sha256"],
              "safe_bundle_path": str(root), "safe_windows_sha256": safe_manifest["safe_windows_sha256"],
              "safe_split_sha256": safe_manifest["safe_split_sha256"],
              "safe_manifest_sha256": file_sha256(root / "manifest.json"),
              "safe_reference_sha256": file_sha256(alias),
              "original_split_sha256": safe_manifest["original_split_sha256"],
              "raw_source_sha256": safe_manifest["raw_source_sha256"],
              "channel_partition_sha256": config["channel_partition_sha256"],
              "canonical_design_sha256": config["canonical_design_sha256"],
              "adapter_source_sha256": file_sha256(Path(__file__)),
              "training_core_source_sha256": file_sha256(
                  project / "src/neurobridge/experiments/phase2a_safe_fit.py"),
              "smoke_updates": smoke_updates}
    record_path, result_path = target / "trial_record.json", target / "result.json"
    if target.exists():
        if not record_path.is_file() or json.loads(record_path.read_text(encoding="utf-8")) != record:
            raise ProtocolViolation("existing trial provenance differs")
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            for name, digest in result.get("artifact_sha256", {}).items():
                if file_sha256(target / name) != digest:
                    raise ProtocolViolation("cached checkpoint/diagnostic hash changed")
            return result
        raise ProtocolViolation("partial immutable fit; no implicit overwrite or retry")
    target.mkdir(parents=True, exist_ok=False)
    with record_path.open("x", encoding="utf-8") as stream:
        stream.write(_canonical_json(record) + "\n")
    try:
        result = _fit_new(target, trial, dataset, values, split, smoke_updates)
    except Exception as exc:
        status = ("FAILED_NUMERICAL" if isinstance(exc, FloatingPointError) else
                  "FAILED_ARTIFACT" if isinstance(exc, ProtocolViolation) else "FAILED_RUNTIME")
        result = {"trial_id": trial["trial_id"], "status": status,
                  "error_class": type(exc).__name__, "error": str(exc),
                  "traceback": traceback.format_exc(limit=12), "optimizer_updates": None,
                  "artifact_sha256": {}}
    with result_path.open("x", encoding="utf-8") as stream:
        stream.write(_canonical_json(result) + "\n")
    return result
