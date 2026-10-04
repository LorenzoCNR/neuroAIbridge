"""Immutable final Real held-out fits from sealed train/validation-only bundles.

This adapter does not read complete source containers or test observations.
Optimization and validation remain in the frozen Phase-2A training core.
"""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurobridge.experiments.phase2a_hpo import (  # noqa: E402
    ProtocolViolation, _canonical_json, file_sha256,
)
from neurobridge.experiments.phase2a_safe_fit import _fit_new  # noqa: E402
from neurobridge.experiments.phase2a_window_inputs import WINDOWS  # noqa: E402
from phase2a_somatotopic_fit import (  # noqa: E402
    POPULATIONS, VERSION, load_safe_real_bundle,
)

FINAL_RELATIVE = Path("outputs/final_thesis_v1")
SPEC_RELATIVE = FINAL_RELATIVE / "freeze/REAL_FINAL_EXPERIMENT_SPEC.json"
REQUIRED_TRIAL_KEYS = frozenset({
    "trial_id", "config", "config_sha256", "output_path",
    "safe_input_reference", "safe_input_reference_sha256",
})
SUCCESS_ARTIFACTS = frozenset({
    "best_validation_checkpoint.pt", "stopping_checkpoint.pt",
    "training_history.csv", "validation_embedding.npz",
})


def _check_identity(project: Path, trial: Mapping[str, Any], spec_sha256: str,
                    smoke_updates: int | None) -> tuple[dict[str, Any], Path]:
    if not REQUIRED_TRIAL_KEYS <= trial.keys():
        raise ProtocolViolation("final Real fit trial contract is incomplete")
    spec_path = project / SPEC_RELATIVE
    if (not isinstance(spec_sha256, str) or len(spec_sha256) != 64 or
            not spec_path.is_file() or file_sha256(spec_path) != spec_sha256):
        raise ProtocolViolation("frozen final Real specification hash mismatch")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec.get("final_slots"), list):
        raise ProtocolViolation("frozen final Real specification has no trial slots")
    identifier = trial["trial_id"]
    if (not isinstance(identifier, str) or not identifier or
            any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for character in identifier)):
        raise ProtocolViolation("unsafe final Real trial ID")
    config = trial["config"]
    if not isinstance(config, dict):
        raise ProtocolViolation("final Real config is not a mapping")
    config_sha256 = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()
    if config_sha256 != trial["config_sha256"]:
        raise ProtocolViolation("final Real config hash mismatch")
    matches = [slot for slot in spec["final_slots"]
               if isinstance(slot, dict) and slot.get("trial_id") == identifier]
    if len(matches) != 1 or matches[0].get("reuse_hpo_checkpoint") is not False:
        raise ProtocolViolation("final Real trial is absent, duplicated, or HPO-reused in frozen specification")
    frozen = matches[0]
    for key in ("config_sha256", "output_path", "safe_input_reference",
                "safe_input_reference_sha256"):
        if trial[key] != frozen.get(key):
            raise ProtocolViolation(f"final Real trial {key} differs from frozen specification")
    for key in ("architecture", "objective", "population", "candidate_index",
                "training_seed_root"):
        if config.get(key) != frozen.get(key):
            raise ProtocolViolation(f"final Real {key} differs from frozen specification")
    if (config.get("window_size") != spec.get("selected_real_window_size") or
            config.get("domain") != f"real_w{spec.get('selected_real_window_size')}"):
        raise ProtocolViolation("final Real trial differs from selected frozen window")
    domain, population = config.get("domain"), config.get("population")
    if (domain not in {f"real_w{window}" for window in WINDOWS} or
            population not in POPULATIONS or config.get("branch") != "held_out" or
            config.get("window_size") != int(domain[6:]) or
            config.get("batch_size") != 1024 or config.get("embedding_dim") != 3 or
            config.get("positive_offset") != 10 or config.get("imposed_shift_bins") != 0):
        raise ProtocolViolation("final Real fit config violates the frozen held-out boundary")
    final_root = (project / FINAL_RELATIVE).resolve()
    expected = final_root / "held_out" / "trials" / identifier
    if Path(trial["output_path"]).resolve() != expected:
        raise ProtocolViolation("final Real fit output path escaped its immutable namespace")
    if smoke_updates is not None:
        if (isinstance(smoke_updates, bool) or not isinstance(smoke_updates, int) or
                smoke_updates < 1 or smoke_updates > config["schedule"]["max_updates"]):
            raise ProtocolViolation("invalid final Real smoke update count")
        expected = final_root / "smoke" / f"{identifier}-u{smoke_updates}"
    alias = (project / "outputs/phase2a_hpo/safe_inputs" / VERSION /
             domain / population / "reference_manifest.json").resolve()
    if (Path(trial["safe_input_reference"]).resolve() != alias or
            not alias.is_file() or
            file_sha256(alias) != trial["safe_input_reference_sha256"]):
        raise ProtocolViolation("final Real fit safe-input reference changed")
    return config, expected


def fit_final_real_trial(project_root: Path, trial: Mapping[str, Any],
                         spec_sha256: str, smoke_updates: int | None = None) -> dict:
    """Fit once or verify/reuse an immutable final held-out Real trial."""
    project = Path(project_root).resolve()
    config, target = _check_identity(project, trial, spec_sha256, smoke_updates)
    bundle, alias, safe_manifest, split, dataset, values = load_safe_real_bundle(
        project, config["domain"], config["population"],
        config["channel_partition_sha256"], config["canonical_design_sha256"],
    )
    if alias.resolve() != Path(trial["safe_input_reference"]).resolve():
        raise ProtocolViolation("loaded safe-input reference differs from final trial")
    record = {
        "trial_id": trial["trial_id"], "config": config,
        "config_sha256": trial["config_sha256"],
        "final_spec_path": str(project / SPEC_RELATIVE),
        "final_spec_sha256": spec_sha256,
        "safe_bundle_path": str(bundle),
        "safe_windows_sha256": safe_manifest["safe_windows_sha256"],
        "safe_split_sha256": safe_manifest["safe_split_sha256"],
        "safe_manifest_sha256": file_sha256(bundle / "manifest.json"),
        "safe_reference_sha256": file_sha256(alias),
        "original_split_sha256": safe_manifest["original_split_sha256"],
        "raw_source_sha256": safe_manifest["raw_source_sha256"],
        "channel_partition_sha256": config["channel_partition_sha256"],
        "canonical_design_sha256": config["canonical_design_sha256"],
        "adapter_source_sha256": file_sha256(Path(__file__)),
        "safe_loader_source_sha256": file_sha256(ROOT / "tools/phase2a_somatotopic_fit.py"),
        "training_core_source_sha256": file_sha256(
            ROOT / "src/neurobridge/experiments/phase2a_safe_fit.py"),
        "smoke_updates": smoke_updates,
    }
    record_path, result_path = target / "trial_record.json", target / "result.json"
    if target.exists():
        if not record_path.is_file() or json.loads(record_path.read_text(encoding="utf-8")) != record:
            raise ProtocolViolation("existing final Real trial provenance differs")
        if not result_path.is_file():
            raise ProtocolViolation("partial immutable final Real fit; no implicit overwrite")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("trial_id") != trial["trial_id"]:
            raise ProtocolViolation("cached final Real result ID changed")
        hashes = result.get("artifact_sha256")
        if not isinstance(hashes, dict):
            raise ProtocolViolation("cached final Real artifact manifest missing")
        if result.get("status") in {"ELIGIBLE", "INELIGIBLE_NEAR_COLLAPSE"} and not SUCCESS_ARTIFACTS <= hashes.keys():
            raise ProtocolViolation("cached final Real successful fit lacks artifacts")
        for name, digest in hashes.items():
            if (name not in SUCCESS_ARTIFACTS or not (target / name).is_file() or
                    file_sha256(target / name) != digest):
                raise ProtocolViolation("cached final Real child artifact hash changed")
        return result
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
                  "traceback": traceback.format_exc(limit=12),
                  "optimizer_updates": None, "artifact_sha256": {}}
    with result_path.open("x", encoding="utf-8") as stream:
        stream.write(_canonical_json(result) + "\n")
    return result
