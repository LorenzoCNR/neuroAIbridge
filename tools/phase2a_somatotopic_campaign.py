"""Versioned Real somatotopic four-window Phase-2A initial campaign."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurobridge.experiments.phase2a_hpo import (  # noqa: E402
    ARCHITECTURES, INITIAL_CANDIDATES, OBJECTIVES, PROTOCOL_RELATIVE,
    _canonical_json, file_sha256, guarded_output_root, load_sealed_candidates,
    ranked_search_candidates, resolved_config, source_hashes,
)
from neurobridge.experiments.phase2a_window_inputs import WINDOWS  # noqa: E402
from phase2a_somatotopic_extract import PARTITION_RELATIVE, VERSION, partition  # noqa: E402
from phase2a_somatotopic_fit import POPULATIONS, fit_real_trial, load_safe_real_bundle  # noqa: E402

STUDY = "phase2a_real_somatotopic_v1_2026-09-28"
DESIGN_RELATIVE = Path("outputs/phase2a_hpo/partitions/REAL_CANONICAL_DESIGN_v1.json")
SOURCE_FILES = ("tools/phase2a_somatotopic_extract.py", "tools/phase2a_somatotopic_fit.py",
                "tools/phase2a_somatotopic_campaign.py", "tests/test_phase2a_somatotopic.py")


def save_new_or_same(path: Path, value: object) -> None:
    data = (_canonical_json(value) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _checked_design(project: Path) -> tuple[dict, str, dict, str]:
    part, partition_sha = partition(project)
    design_path = project / DESIGN_RELATIVE
    design = json.loads(design_path.read_text(encoding="utf-8"))
    design_sha = file_sha256(design_path)
    if (design["design_version"] != VERSION or design["partition_sha256"] != partition_sha or
            design["real_window_sizes_bins"] != list(WINDOWS) or
            design["positive_offset_bins"] != 10 or design["real_sampling_interval_ms"] != 1 or
            design["window_padding"] != "center"):
        raise RuntimeError("canonical Real design/partition mismatch")
    return part, partition_sha, design, design_sha


def safe_references(project: Path) -> dict[str, dict]:
    hpo = guarded_output_root(project)
    part, partition_sha, _, design_sha = _checked_design(project)
    refs = {}
    for window in WINDOWS:
        domain = f"real_w{window}"
        for population in POPULATIONS:
            bundle = (hpo / "safe_inputs" / domain / "TOTAL65" if population == "TOTAL65" else
                      hpo / "safe_inputs" / VERSION / domain / population)
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest["domain"] != domain or manifest["population"] != population or
                    manifest["window_size"] != window or
                    file_sha256(bundle / "windows.npz") != manifest["safe_windows_sha256"] or
                    file_sha256(bundle / "split.json") != manifest["safe_split_sha256"]):
                raise RuntimeError(f"Real safe bundle invalid: {bundle}")
            indices = list(range(65)) if population == "TOTAL65" else part[population]["feature_indices"]
            if population != "TOTAL65" and (manifest["channel_indices"] != indices or
                                            manifest["partition_sha256"] != partition_sha):
                raise RuntimeError("somatotopic bundle indices/partition mismatch")
            reference = {"domain": domain, "population": population,
                         "partition_version": VERSION, "partition_sha256": partition_sha,
                         "canonical_design_sha256": design_sha, "channel_indices": indices,
                         "nlb_unit_ids": part["feature_order_nlb_unit_ids"] if population == "TOTAL65" else
                                         part[population]["nlb_unit_ids"],
                         "bundle_path": str(bundle), "bundle_manifest_sha256": file_sha256(manifest_path),
                         "safe_windows_sha256": manifest["safe_windows_sha256"],
                         "safe_split_sha256": manifest["safe_split_sha256"],
                         "original_split_sha256": manifest["original_split_sha256"],
                         "raw_source_sha256": manifest["raw_source_sha256"],
                         "train_trial_ids": manifest["train_trial_ids"],
                         "validation_trial_ids": manifest["validation_trial_ids"],
                         "test_rows_in_bundle": False}
            alias = hpo / "safe_inputs" / VERSION / domain / population / "reference_manifest.json"
            save_new_or_same(alias, reference)
            refs[f"{domain}/{population}"] = reference
    return refs


def plan(project_root: Path) -> dict:
    project = project_root.resolve()
    hpo = guarded_output_root(project)
    part, partition_sha, _, design_sha = _checked_design(project)
    candidates, candidate_sha = load_sealed_candidates(project / PROTOCOL_RELATIVE)
    references = safe_references(project)
    slots = []
    for window in WINDOWS:
        domain = f"real_w{window}"
        for objective in OBJECTIVES:
            for architecture in ARCHITECTURES:
                for index in INITIAL_CANDIDATES:
                    for population in POPULATIONS:
                        channels = list(range(65)) if population == "TOTAL65" else part[population]["feature_indices"]
                        alias_population = "TOTAL65" if population == "TOTAL65" else "A" if population == "A_PROXIMAL" else "B"
                        config = resolved_config("real", alias_population, architecture,
                                                 candidates[objective][index], 1101, channels)
                        config.update({"domain": domain, "population": population, "window_size": window,
                                       "channel_partition_version": VERSION,
                                       "channel_partition_sha256": partition_sha,
                                       "canonical_design_sha256": design_sha,
                                       "real_sampling_interval_ms": 1,
                                       "channel_partition_seed_applied": False})
                        digest = _hash(config)
                        objective_code = {"soft": "s", "infonce": "i", "time_contrastive_blocks": "t",
                                          "behavior_contrastive_blocks": "b"}[objective]
                        architecture_code = "c" if architecture == "cnn1d" else "t"
                        population_code = {"TOTAL65": "T", "A_PROXIMAL": "A", "B_DISTAL": "B"}[population]
                        identifier = (f"p2a-rs-w{window}-{population_code}{architecture_code}{objective_code}"
                                      f"-c{index}-s1101-{digest[:16]}")
                        target = hpo / "studies" / STUDY / "trials" / identifier
                        if len(str((target / "best_validation_checkpoint.pt").resolve())) >= 250:
                            raise RuntimeError("Real checkpoint path too long")
                        slots.append({"trial_id": identifier, "candidate_id": config["candidate_id"],
                                      "config": config, "config_sha256": digest,
                                      "output_path": str(target),
                                      "safe_input_reference": str(hpo / "safe_inputs" / VERSION / domain /
                                                                  population / "reference_manifest.json")})
    if (len(slots) != 384 or len({item["trial_id"] for item in slots}) != 384 or
            {item["config"]["candidate_index"] for item in slots} != set(INITIAL_CANDIDATES)):
        raise RuntimeError("384-slot frozen Real initial plan invalid")
    artifact = {"study_id": STUDY, "slot_count": len(slots), "partition_version": VERSION,
                "partition_sha256": partition_sha, "canonical_design_sha256": design_sha,
                "candidate_sha256": candidate_sha,
                "historical_random_partition_sha256": file_sha256(project / part["historical_random_partition"]),
                "historical_random_partition_used": False,
                "safe_reference_hashes": {key: file_sha256(hpo / "safe_inputs" / VERSION /
                                                            key / "reference_manifest.json") for key in references},
                "slots": slots}
    save_new_or_same(hpo / "studies" / STUDY / "initial_plan.json", artifact)
    return artifact


def gate(project_root: Path) -> dict:
    project = project_root.resolve()
    hpo = guarded_output_root(project)
    study_root = hpo / "studies" / STUDY
    campaign = plan(project)
    sources = source_hashes(project)
    for relative in SOURCE_FILES:
        sources[relative] = file_sha256(project / relative)
    sources[PARTITION_RELATIVE.as_posix()] = file_sha256(project / PARTITION_RELATIVE)
    sources[DESIGN_RELATIVE.as_posix()] = file_sha256(project / DESIGN_RELATIVE)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True, check=False)
    snapshot = {"source_file_sha256": sources, "git_head": git.stdout.strip() if git.returncode == 0 else "UNAVAILABLE",
                "parent_initial_study_plan_sha256": file_sha256(hpo / "studies" /
                                                                 "phase2a_real_windows_2026-09-28" / "initial_plan.json")}
    save_new_or_same(study_root / "source_snapshot_manifest.json", snapshot)
    for window in WINDOWS:
        domain = f"real_w{window}"
        for population in POPULATIONS:
            root, alias, manifest, split, dataset, values = load_safe_real_bundle(
                project, domain, population, campaign["partition_sha256"],
                campaign["canonical_design_sha256"])
            if (manifest["train_window_count"] + manifest["validation_window_count"] != len(values["trial_id"]) or
                    not len(dataset) or split["train"] != manifest["train_trial_ids"] or
                    split["validation"] != manifest["validation_trial_ids"]):
                raise RuntimeError("safe Real bundle cardinality invalid")
            del root, alias, manifest, split, dataset, values
            gc.collect()
    smoke_slot = next(slot for slot in campaign["slots"] if
                      slot["config"]["domain"] == "real_w21" and
                      slot["config"]["population"] == "A_PROXIMAL" and
                      slot["config"]["architecture"] == "cnn1d" and
                      slot["config"]["objective"] == "soft" and
                      slot["config"]["candidate_index"] == 0)
    smoke = fit_real_trial(project, smoke_slot, smoke_updates=1)
    if smoke["status"] not in {"ELIGIBLE", "INELIGIBLE_NEAR_COLLAPSE"} or smoke["stopping_update"] != 1:
        raise RuntimeError(f"somatotopic one-update smoke failed: {smoke.get('error')}")
    report = {"gate": "GO", "study_id": STUDY, "slot_count": 384,
              "partition_sha256": campaign["partition_sha256"],
              "canonical_design_sha256": campaign["canonical_design_sha256"],
              "candidate_sha256": campaign["candidate_sha256"],
              "source_snapshot_sha256": file_sha256(study_root / "source_snapshot_manifest.json"),
              "safe_reference_hashes": campaign["safe_reference_hashes"],
              "smoke_status": smoke["status"], "smoke_updates": 1,
              "test_or_scientific_outcome_loaded": False, "v2_write": False}
    save_new_or_same(study_root / "go_gate.json", report)
    return report


def _selection_row(slot: dict, result: dict) -> dict:
    config = slot["config"]
    return {"domain": "real", "population": {"TOTAL65": "TOTAL65", "A_PROXIMAL": "A", "B_DISTAL": "B"}[config["population"]],
            "architecture": config["architecture"], "objective": config["objective"],
            "candidate_index": config["candidate_index"], "training_seed_root": 1101,
            "status": result["status"], "validation_loss": result.get("validation_loss"),
            "selected_update": result.get("selected_update"), "stopping_update": result.get("stopping_update")}


def execute(project_root: Path) -> dict:
    project = project_root.resolve()
    go = gate(project)
    hpo = guarded_output_root(project)
    study_root = hpo / "studies" / STUDY
    slots = json.loads((study_root / "initial_plan.json").read_text(encoding="utf-8"))["slots"]
    results = []
    cells = {}
    table = study_root / "real_initial_results.csv"
    columns = ("trial_id", "domain", "population", "architecture", "objective", "candidate_index",
               "training_seed_root", "training_seed_effective", "learning_rate", "weight_decay",
               "active_temperature", "status", "validation_loss", "selected_update", "stopping_update",
               "near_collapse", "distance_q50", "fraction_distance_lt_0_01", "covariance_trace",
               "training_seconds", "config_sha256", "checkpoint_sha256", "result_sha256", "error")
    for number, slot in enumerate(slots, 1):
        result = fit_real_trial(project, slot)
        results.append(result)
        config = slot["config"]
        print(f"Real somatotopic {number}/384 {slot['trial_id']}: {result['status']} "
              f"validation={result.get('validation_loss')} updates={result.get('stopping_update')}", flush=True)
        temp = table.with_suffix(".csv.tmp")
        with temp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for current, outcome in zip(slots[:number], results):
                row = current["config"]
                writer.writerow({**{field: row.get(field) for field in columns},
                                 "trial_id": current["trial_id"], "config_sha256": current["config_sha256"],
                                 "status": outcome["status"], "validation_loss": outcome.get("validation_loss"),
                                 "selected_update": outcome.get("selected_update"),
                                 "stopping_update": outcome.get("stopping_update"),
                                 "training_seconds": outcome.get("training_seconds"),
                                 "checkpoint_sha256": outcome.get("artifact_sha256", {}).get("best_validation_checkpoint.pt"),
                                 "result_sha256": file_sha256(Path(current["output_path"]) / "result.json"),
                                 "error": outcome.get("error"),
                                 **{field: outcome.get("geometry", {}).get(field) for field in
                                    ("near_collapse", "distance_q50", "fraction_distance_lt_0_01", "covariance_trace")}})
        os.replace(temp, table)
        cell = (config["domain"], config["architecture"], config["objective"])
        cell_slots = [candidate for candidate in slots if
                      (candidate["config"]["domain"], candidate["config"]["architecture"],
                       candidate["config"]["objective"]) == cell]
        if slot["trial_id"] != cell_slots[-1]["trial_id"]:
            continue
        records = [_selection_row(current, outcome) for current, outcome in zip(slots[:number], results)
                   if (current["config"]["domain"], current["config"]["architecture"],
                       current["config"]["objective"]) == cell]
        ranking = ranked_search_candidates(records, "real", cell[1], cell[2])
        key = "/".join(cell)
        cells[key] = {"domain_window": cell[0], "architecture": cell[1], "objective": cell[2],
                      "candidate_population_results": records, "eligible_ranking": ranking,
                      "current_finalists": [index for index, _ in ranking[:2]],
                      "extension_status": "PENDING_PAIRED_ARCHITECTURE"}
        save_new_or_same(study_root / "cells" / f"{key.replace('/', '_')}.json", cells[key])
        mate = "/".join((cell[0], "transformer" if cell[1] == "cnn1d" else "cnn1d", cell[2]))
        if mate in cells:
            extension = len(cells[key]["eligible_ranking"]) < 2 or len(cells[mate]["eligible_ranking"]) < 2
            paired = {"domain_window": cell[0], "objective": cell[2],
                      "extension_status": "REQUIRED_FOR_EXTENSION" if extension else "NO_EXTENSION",
                      "architecture_rankings": {arch: cells[f"{cell[0]}/{arch}/{cell[2]}"]["eligible_ranking"]
                                                for arch in ARCHITECTURES}}
            save_new_or_same(study_root / "paired_extension" / f"{cell[0]}_{cell[2]}.json", paired)
    summary = {"status": "REAL_INITIAL_CAMPAIGN_COMPLETE", "slot_count": 384,
               "completed_valid_or_ineligible": sum(r["status"] in {"ELIGIBLE", "INELIGIBLE_NEAR_COLLAPSE"} for r in results),
               "failed": sum(r["status"].startswith("FAILED") for r in results),
               "eligible": sum(r["status"] == "ELIGIBLE" for r in results),
               "near_collapse_ineligible": sum(r["status"] == "INELIGIBLE_NEAR_COLLAPSE" for r in results),
               "gate_sha256": file_sha256(study_root / "go_gate.json"),
               "result_table_sha256": file_sha256(table), "cells": cells,
               "no_test_or_scientific_outcome_used": True}
    save_new_or_same(study_root / "initial_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--gate-only", action="store_true")
    mode.add_argument("--run-initial", action="store_true")
    args = parser.parse_args()
    result = plan(ROOT) if args.plan else gate(ROOT) if args.gate_only else execute(ROOT)
    print(_canonical_json({key: value for key, value in result.items() if key not in {"slots", "cells"}}), flush=True)


if __name__ == "__main__":
    main()
