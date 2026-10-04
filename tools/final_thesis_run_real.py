"""Resume-safe final Real held-out fits from the sealed one-window spec.

The selected seed-1101 HPO checkpoints are verified and reused. Only the
frozen seed-1201/1301 slots can fit, using train/validation-only bundles.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from neurobridge.experiments.phase2a_hpo import (  # noqa: E402
    _canonical_json, file_sha256,
)
from final_thesis_freeze_real import (  # noqa: E402
    FREEZE_RELATIVE, RULE_NAME, SPEC_NAME, WINDOWS_NAME, WINNERS_NAME,
)
from final_thesis_real_fit import fit_final_real_trial  # noqa: E402
from final_thesis_real_hpo import _checked_extension, _new_slot  # noqa: E402

RESULT_COLUMNS = (
    "trial_id", "window_size", "architecture", "objective", "population",
    "training_seed_root", "candidate_index", "reused_hpo_checkpoint", "status",
    "validation_loss", "selected_update", "stopping_update", "near_collapse",
    "training_seconds", "config_sha256", "checkpoint_sha256", "result_sha256",
    "final_spec_sha256", "error",
)


def _frozen_spec(project: Path) -> tuple[dict, str]:
    freeze = project / FREEZE_RELATIVE
    spec_path = freeze / SPEC_NAME
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    digest = file_sha256(spec_path)
    if (spec["selection_rule_sha256"] != file_sha256(freeze / RULE_NAME) or
            spec["hpo_winners_csv_sha256"] != file_sha256(freeze / WINNERS_NAME) or
            spec["window_selection_csv_sha256"] != file_sha256(freeze / WINDOWS_NAME) or
            spec["freeze_source_sha256"] != file_sha256(
                project / "tools/final_thesis_freeze_real.py") or
            spec["final_neural_model_instances"] != 72 or
            spec["reused_seed1101_hpo_checkpoints"] != 24 or
            spec["new_seed1201_1301_fits"] != 48 or
            spec["final_training_seed_roots"] != [1101, 1201, 1301] or
            spec["selected_real_window_size"] not in [21, 41, 121, 201] or
            not spec["no_test_or_scientific_metric_used_for_selection"] or
            spec["multiseed_hyperparameter_selection_robustness_performed"]):
        raise RuntimeError("final Real spec/provenance differs from write-once freeze")
    initial, extension, part, checked, completed = _checked_extension(project)
    if (spec["initial_summary_sha256"] != file_sha256(
            project / "outputs/phase2a_hpo/studies/phase2a_real_somatotopic_v1_2026-09-28/initial_summary.json") or
            spec["extension_summary_sha256"] != file_sha256(
                project / "outputs/phase2a_hpo/studies/final_thesis_real_hpo_v1/extension/summary.json") or
            spec["somatotopic_partition_sha256"] != initial["partition_sha256"] or
            spec["sealed_candidate_sha256"] != checked["seal"] or
            spec["canonical_design_sha256"] != initial["canonical_design_sha256"] or
            len(spec["final_slots"]) != 72 or extension["slot_count"] != 108 or
            completed["summary"]["failed"] != 0):
        raise RuntimeError("final Real parent HPO campaign changed")
    return spec, digest


def _trial(project: Path, spec: dict, slot: dict, part: dict, candidates: dict) -> dict:
    planned = _new_slot(project, part, candidates,
                        f"real_w{spec['selected_real_window_size']}",
                        slot["architecture"], slot["objective"],
                        slot["candidate_index"], slot["training_seed_root"],
                        slot["population"], "unused_finalist_path")
    if (planned["config_sha256"] != slot["config_sha256"] or
            planned["safe_input_reference"] != slot["safe_input_reference"] or
            planned["safe_input_reference_sha256"] != slot["safe_input_reference_sha256"]):
        raise RuntimeError("final Real reconstructed trial differs from sealed slot")
    return {"trial_id": slot["trial_id"], "config": planned["config"],
            "config_sha256": slot["config_sha256"],
            "output_path": slot["output_path"],
            "safe_input_reference": slot["safe_input_reference"],
            "safe_input_reference_sha256": slot["safe_input_reference_sha256"]}


def _reused_result(slot: dict) -> dict:
    checkpoint = Path(slot["checkpoint_path"])
    target = checkpoint.parent
    record_path, result_path = target / "trial_record.json", target / "result.json"
    if (not checkpoint.is_file() or file_sha256(checkpoint) != slot["checkpoint_sha256"] or
            not record_path.is_file() or not result_path.is_file()):
        raise RuntimeError("selected seed-1101 HPO checkpoint is missing or changed")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (record["trial_id"] != slot["trial_id"] or
            record["config_sha256"] != slot["config_sha256"] or
            result["trial_id"] != slot["trial_id"] or
            result["status"] != "ELIGIBLE" or
            result["selected_update"] != slot["selected_update"] or
            result["artifact_sha256"]["best_validation_checkpoint.pt"] !=
            slot["checkpoint_sha256"]):
        raise RuntimeError("selected seed-1101 HPO fit identity/eligibility changed")
    for name, digest in result["artifact_sha256"].items():
        if not (target / name).is_file() or file_sha256(target / name) != digest:
            raise RuntimeError(f"selected HPO child artifact changed: {target / name}")
    return result


def _result_row(slot: dict, result: dict, spec_sha: str) -> dict:
    root = (Path(slot["checkpoint_path"]).parent if slot["reuse_hpo_checkpoint"]
            else Path(slot["output_path"]))
    return {"trial_id": slot["trial_id"],
            "window_size": int(slot["safe_input_reference"].split("real_w")[-1].split(
                os.sep)[0]),
            "architecture": slot["architecture"], "objective": slot["objective"],
            "population": slot["population"], "training_seed_root": slot["training_seed_root"],
            "candidate_index": slot["candidate_index"],
            "reused_hpo_checkpoint": str(slot["reuse_hpo_checkpoint"]).lower(),
            "status": result["status"], "validation_loss": result.get("validation_loss"),
            "selected_update": result.get("selected_update"),
            "stopping_update": result.get("stopping_update"),
            "near_collapse": result.get("geometry", {}).get("near_collapse"),
            "training_seconds": result.get("training_seconds"),
            "config_sha256": slot["config_sha256"],
            "checkpoint_sha256": result.get("artifact_sha256", {}).get(
                "best_validation_checkpoint.pt"),
            "result_sha256": file_sha256(root / "result.json"),
            "final_spec_sha256": spec_sha, "error": result.get("error")}


def _rolling_table(path: Path, spec: dict, known: dict[str, dict], spec_sha: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for slot in spec["final_slots"]:
            if slot["trial_id"] in known:
                writer.writerow(_result_row(slot, known[slot["trial_id"]], spec_sha))
    os.replace(temporary, path)


def run(project: Path, smoke: bool = False, dry_run: bool = False) -> dict:
    spec, spec_sha = _frozen_spec(project)
    _, _, part, checked, _ = _checked_extension(project)
    slots = spec["final_slots"]
    new_slots = [slot for slot in slots if not slot["reuse_hpo_checkpoint"]]
    if len(new_slots) != 48 or len({slot["trial_id"] for slot in slots}) != 72:
        raise RuntimeError("final Real trial enumeration changed")
    for slot in slots:
        if slot["reuse_hpo_checkpoint"]:
            _reused_result(slot)
        else:
            _trial(project, spec, slot, part, checked["candidates"])
    if dry_run:
        return {"status": "FINAL_REAL_DRY_RUN", "selected_window_size":
                spec["selected_real_window_size"], "model_instances": len(slots),
                "reused_checkpoints": 24, "new_fits": len(new_slots),
                "spec_sha256": spec_sha, "no_test_loaded": True}
    if smoke:
        trial = _trial(project, spec, new_slots[0], part, checked["candidates"])
        result = fit_final_real_trial(project, trial, spec_sha, smoke_updates=1)
        return {"status": "FINAL_REAL_SMOKE", "trial_id": trial["trial_id"],
                "fit_status": result["status"], "optimizer_updates": result.get("stopping_update")}
    smoke_trial = new_slots[0]["trial_id"]
    smoke_path = project / "outputs/final_thesis_v1/smoke" / f"{smoke_trial}-u1/result.json"
    if not smoke_path.is_file():
        raise RuntimeError("final Real smoke fit must complete before full campaign")
    smoke_result = json.loads(smoke_path.read_text(encoding="utf-8"))
    if (smoke_result["trial_id"] != smoke_trial or
            smoke_result["status"] not in {"ELIGIBLE", "INELIGIBLE_NEAR_COLLAPSE"} or
            smoke_result.get("stopping_update") != 1 or
            any(not (smoke_path.parent / name).is_file() or
                file_sha256(smoke_path.parent / name) != digest
                for name, digest in smoke_result.get("artifact_sha256", {}).items()) or
            len(smoke_result.get("artifact_sha256", {})) != 4):
        raise RuntimeError("final Real smoke fit did not pass")
    known = {slot["trial_id"]: _reused_result(slot) for slot in slots
             if slot["reuse_hpo_checkpoint"]}
    for slot in new_slots:
        if Path(slot["output_path"]).exists():
            trial = _trial(project, spec, slot, part, checked["candidates"])
            known[slot["trial_id"]] = fit_final_real_trial(project, trial, spec_sha)
    table = project / "outputs/final_thesis_v1/held_out/FINAL_MODEL_FITS.csv"
    _rolling_table(table, spec, known, spec_sha)
    consecutive_technical_failures = 0
    for number, slot in enumerate(new_slots, 1):
        trial = _trial(project, spec, slot, part, checked["candidates"])
        result = fit_final_real_trial(project, trial, spec_sha)
        known[slot["trial_id"]] = result
        _rolling_table(table, spec, known, spec_sha)
        print(f"Final Real new fit {number}/{len(new_slots)} {slot['trial_id']}: "
              f"{result['status']} validation={result.get('validation_loss')}", flush=True)
        consecutive_technical_failures = (consecutive_technical_failures + 1
                                          if result["status"].startswith("FAILED") else 0)
        if consecutive_technical_failures >= 2:
            raise RuntimeError("two consecutive technical fit failures; stop this campaign")
    summary = {"status": "FINAL_REAL_HELD_OUT_FITS_COMPLETE",
               "selected_window_size": spec["selected_real_window_size"],
               "model_instances": 72, "reused_hpo_checkpoints": 24,
               "new_fits": 48,
               "eligible": sum(result["status"] == "ELIGIBLE" for result in known.values()),
               "near_collapse_ineligible": sum(result["status"] == "INELIGIBLE_NEAR_COLLAPSE"
                                                for result in known.values()),
               "failed": sum(result["status"].startswith("FAILED") for result in known.values()),
               "spec_sha256": spec_sha, "result_table_sha256": file_sha256(table),
               "test_or_scientific_metrics_used_for_selection": False}
    path = project / "outputs/final_thesis_v1/held_out/summary.json"
    content = (_canonical_json(summary) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError("immutable final Real fit summary differs")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    result = run(ROOT, smoke=args.smoke, dry_run=args.dry_run)
    print(_canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
