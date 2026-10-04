"""Complete frozen Real Phase-2A HPO without opening test observations.

The initial 384 fits are immutable parents. This runner only fits the paired
candidate-4/5 extensions and the already-defined finalist robustness seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurobridge.experiments.phase2a_hpo import (  # noqa: E402
    ARCHITECTURES, EXTENSION_CANDIDATES, FINALIST_SEEDS, OBJECTIVES,
    PROTOCOL_RELATIVE, SEARCH_SEED, _canonical_json, extension_required,
    file_sha256, finalist_indices, guarded_output_root, load_sealed_candidates,
    ranked_search_candidates, resolved_config, select_finalist,
)
from neurobridge.experiments.phase2a_window_inputs import WINDOWS  # noqa: E402
from phase2a_somatotopic_campaign import STUDY as INITIAL_STUDY  # noqa: E402
from phase2a_somatotopic_extract import partition  # noqa: E402
from phase2a_somatotopic_fit import POPULATIONS, fit_real_trial  # noqa: E402

STUDY = "final_thesis_real_hpo_v1"
PARTITION_RELATIVE = Path("outputs/phase2a_hpo/partitions/real_somatotopic_v1.json")
DESIGN_RELATIVE = Path("outputs/phase2a_hpo/partitions/REAL_CANONICAL_DESIGN_v1.json")
OBJECTIVE_CODE = {"soft": "s", "infonce": "i", "time_contrastive_blocks": "t",
                  "behavior_contrastive_blocks": "b"}
POPULATION_CODE = {"TOTAL65": "T", "A_PROXIMAL": "A", "B_DISTAL": "B"}
POPULATION_ALIAS = {"TOTAL65": "TOTAL65", "A_PROXIMAL": "A", "B_DISTAL": "B"}
RESULT_COLUMNS = ("trial_id", "domain", "population", "architecture", "objective",
                  "candidate_index", "training_seed_root", "training_seed_effective",
                  "learning_rate", "weight_decay", "active_temperature", "status",
                  "validation_loss", "selected_update", "stopping_update",
                  "near_collapse", "distance_q50", "fraction_distance_lt_0_01",
                  "covariance_trace", "training_seconds", "config_sha256",
                  "checkpoint_sha256", "trial_record_sha256", "result_sha256", "error")


def _root(project: Path) -> Path:
    return guarded_output_root(project) / "studies" / STUDY


def _initial_root(project: Path) -> Path:
    return guarded_output_root(project) / "studies" / INITIAL_STUDY


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _save_new_or_same(path: Path, value: object) -> None:
    content = (_canonical_json(value) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def _verified_result(slot: dict) -> dict:
    target = Path(slot["output_path"])
    record_path, result_path = target / "trial_record.json", target / "result.json"
    if not record_path.is_file() or not result_path.is_file():
        raise RuntimeError(f"missing immutable HPO fit: {target}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (record["trial_id"] != slot["trial_id"] or result["trial_id"] != slot["trial_id"] or
            record["config_sha256"] != slot["config_sha256"] or
            record["config"] != slot["config"]):
        raise RuntimeError(f"fit identity/config mismatch: {target}")
    for name, digest in result.get("artifact_sha256", {}).items():
        if not (target / name).is_file() or file_sha256(target / name) != digest:
            raise RuntimeError(f"fit child artifact hash mismatch: {target / name}")
    return result


def _selection_row(slot: dict, result: dict) -> dict:
    config = slot["config"]
    return {"domain": "real", "population": POPULATION_ALIAS[config["population"]],
            "architecture": config["architecture"], "objective": config["objective"],
            "candidate_index": config["candidate_index"],
            "training_seed_root": config["training_seed_root"], "status": result["status"],
            "validation_loss": result.get("validation_loss"),
            "selected_update": result.get("selected_update"),
            "stopping_update": result.get("stopping_update")}


def _source_snapshot(project: Path) -> dict:
    initial = _initial_root(project)
    previous = json.loads((initial / "source_snapshot_manifest.json").read_text(encoding="utf-8"))
    for relative, digest in previous["source_file_sha256"].items():
        if file_sha256(project / relative) != digest:
            raise RuntimeError(f"initial frozen source drift: {relative}")
    source_files = dict(previous["source_file_sha256"])
    source_files["tools/final_thesis_real_hpo.py"] = file_sha256(Path(__file__))
    source_files["tests/test_final_thesis_real_hpo.py"] = file_sha256(
        project / "tests/test_final_thesis_real_hpo.py")
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project,
                              capture_output=True, text=True, check=False)
    return {"source_file_sha256": source_files,
            "git_head": git_head.stdout.strip() if git_head.returncode == 0 else "UNAVAILABLE",
            "parent_source_snapshot_sha256": file_sha256(initial / "source_snapshot_manifest.json")}


def _checked_parent(project: Path) -> tuple[dict, dict, dict, dict]:
    initial = _initial_root(project)
    plan = json.loads((initial / "initial_plan.json").read_text(encoding="utf-8"))
    summary = json.loads((initial / "initial_summary.json").read_text(encoding="utf-8"))
    if (plan["slot_count"] != 384 or len(plan["slots"]) != 384 or
            summary["status"] != "REAL_INITIAL_CAMPAIGN_COMPLETE" or
            summary["completed_valid_or_ineligible"] != 384 or summary["failed"] != 0 or
            summary["eligible"] != 310 or summary["near_collapse_ineligible"] != 74 or
            file_sha256(initial / "real_initial_results.csv") != summary["result_table_sha256"]):
        raise RuntimeError("Real initial parent is not the verified 384-fit campaign")
    part, partition_sha = partition(project)
    design = json.loads((project / DESIGN_RELATIVE).read_text(encoding="utf-8"))
    if (plan["partition_sha256"] != partition_sha or
            plan["canonical_design_sha256"] != file_sha256(project / DESIGN_RELATIVE) or
            design["partition_sha256"] != partition_sha or
            design["real_window_sizes_bins"] != list(WINDOWS)):
        raise RuntimeError("canonical partition/window design changed")
    candidates, seal = load_sealed_candidates(project / PROTOCOL_RELATIVE)
    if plan["candidate_sha256"] != seal:
        raise RuntimeError("sealed candidate hash differs from initial campaign")
    _source_snapshot(project)
    ledgers = {f"real_w{window}": [] for window in WINDOWS}
    with (initial / "real_initial_results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 384 or len({row["trial_id"] for row in rows}) != 384:
        raise RuntimeError("Real initial result table cardinality/identity changed")
    for slot, row in zip(plan["slots"], rows):
        result = _verified_result(slot)
        config = slot["config"]
        window = config["domain"]
        if (row["trial_id"] != slot["trial_id"] or row["status"] != result["status"] or
                window not in ledgers or config["channel_partition_sha256"] != partition_sha or
                config["canonical_design_sha256"] != plan["canonical_design_sha256"]):
            raise RuntimeError("Real initial row/config/result mismatch")
        ledgers[window].append(_selection_row(slot, result))
    if any(len(ledger) != 96 for ledger in ledgers.values()):
        raise RuntimeError("Real window ledgers are not isolated 96-fit cohorts")
    for window, ledger in ledgers.items():
        for objective in OBJECTIVES:
            decision = json.loads((initial / "paired_extension" / f"{window}_{objective}.json").read_text())
            expected = "REQUIRED_FOR_EXTENSION" if extension_required(ledger, "real", objective) else "NO_EXTENSION"
            if decision["extension_status"] != expected:
                raise RuntimeError("frozen paired extension decision changed")
    return plan, summary, part, {"candidates": candidates, "seal": seal, "ledgers": ledgers}


def _new_slot(project: Path, part: dict, candidates: dict, window: str,
              architecture: str, objective: str, index: int, seed: int,
              population: str, stage: str) -> dict:
    channels = (list(range(65)) if population == "TOTAL65" else part[population]["feature_indices"])
    config = resolved_config("real", POPULATION_ALIAS[population], architecture,
                             candidates[objective][index], seed, channels)
    config.update({"domain": window, "population": population,
                   "window_size": int(window.removeprefix("real_w")),
                   "channel_partition_version": "real_somatotopic_v1",
                   "channel_partition_sha256": file_sha256(project / PARTITION_RELATIVE),
                   "canonical_design_sha256": file_sha256(project / DESIGN_RELATIVE),
                   "real_sampling_interval_ms": 1, "channel_partition_seed_applied": False})
    digest = _sha(config)
    architecture_code = "c" if architecture == "cnn1d" else "t"
    identifier = (f"ft-rh-w{config['window_size']}-{POPULATION_CODE[population]}"
                  f"{architecture_code}{OBJECTIVE_CODE[objective]}-c{index}-s{seed}-{digest[:12]}")
    target = _root(project) / stage / "trials" / identifier
    if len(str((target / "best_validation_checkpoint.pt").resolve())) >= 250:
        raise RuntimeError("final-thesis Real checkpoint path too long")
    reference = (guarded_output_root(project) / "safe_inputs" / "real_somatotopic_v1" /
                 window / population / "reference_manifest.json")
    if not reference.is_file():
        raise RuntimeError(f"missing canonical safe-input reference: {reference}")
    return {"trial_id": identifier, "candidate_id": config["candidate_id"],
            "config": config, "config_sha256": digest, "output_path": str(target),
            "safe_input_reference": str(reference),
            "safe_input_reference_sha256": file_sha256(reference)}


def build_extension_plan(project: Path) -> dict:
    parent, _, part, checked = _checked_parent(project)
    slots = []
    triggers = {}
    for window in (f"real_w{size}" for size in WINDOWS):
        ledger = checked["ledgers"][window]
        for objective in OBJECTIVES:
            triggered = extension_required(ledger, "real", objective)
            triggers[f"{window}/{objective}"] = triggered
            if not triggered:
                continue
            for architecture in ARCHITECTURES:
                for index in EXTENSION_CANDIDATES:
                    for population in POPULATIONS:
                        slots.append(_new_slot(project, part, checked["candidates"], window,
                                               architecture, objective, index, SEARCH_SEED,
                                               population, "extension"))
    if (len(slots) != 108 or sum(triggers.values()) != 9 or
            len({slot["trial_id"] for slot in slots}) != 108 or
            {slot["config"]["candidate_index"] for slot in slots} != set(EXTENSION_CANDIDATES)):
        raise RuntimeError("frozen paired Real extension plan is not 108 unique fits")
    return {"study_id": STUDY, "stage": "paired_extension", "slot_count": len(slots),
            "parent_initial_plan_sha256": file_sha256(_initial_root(project) / "initial_plan.json"),
            "parent_initial_summary_sha256": file_sha256(_initial_root(project) / "initial_summary.json"),
            "parent_initial_result_table_sha256": file_sha256(
                _initial_root(project) / "real_initial_results.csv"),
            "parent_gate_sha256": file_sha256(_initial_root(project) / "go_gate.json"),
            "partition_sha256": parent["partition_sha256"],
            "canonical_design_sha256": parent["canonical_design_sha256"],
            "sealed_candidate_sha256": checked["seal"],
            "protocol_sha256": file_sha256(project / PROTOCOL_RELATIVE),
            "source_snapshot": _source_snapshot(project), "paired_triggers": triggers,
            "slots": slots, "test_or_scientific_outcomes_used": False, "v2_write": False}


def _seal_extension_plan(project: Path) -> dict:
    plan = build_extension_plan(project)
    _save_new_or_same(_root(project) / "extension" / "plan.json", plan)
    return plan


def _rolling_table(path: Path, outcomes: list[tuple[dict, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for slot, result in outcomes:
            config = slot["config"]
            target = Path(slot["output_path"])
            writer.writerow({**{key: config.get(key) for key in RESULT_COLUMNS},
                             "trial_id": slot["trial_id"], "status": result["status"],
                             "validation_loss": result.get("validation_loss"),
                             "selected_update": result.get("selected_update"),
                             "stopping_update": result.get("stopping_update"),
                             "training_seconds": result.get("training_seconds"),
                             "near_collapse": result.get("geometry", {}).get("near_collapse"),
                             "distance_q50": result.get("geometry", {}).get("distance_q50"),
                             "fraction_distance_lt_0_01": result.get("geometry", {}).get(
                                 "fraction_distance_lt_0_01"),
                             "covariance_trace": result.get("geometry", {}).get("covariance_trace"),
                             "config_sha256": slot["config_sha256"],
                             "checkpoint_sha256": result.get("artifact_sha256", {}).get(
                                 "best_validation_checkpoint.pt"),
                             "trial_record_sha256": file_sha256(target / "trial_record.json"),
                             "result_sha256": file_sha256(target / "result.json"),
                             "error": result.get("error")})
    os.replace(temporary, path)


def _extension_records(project: Path, plan: dict) -> list[tuple[dict, dict]]:
    return [(slot, _verified_result(slot)) for slot in plan["slots"]]


def run_extension(project: Path) -> dict:
    plan = _seal_extension_plan(project)
    outcomes = []
    table = _root(project) / "extension" / "results.csv"
    for number, slot in enumerate(plan["slots"], 1):
        result = fit_real_trial(project, slot)
        outcomes.append((slot, result))
        _rolling_table(table, outcomes)
        print(f"Real extension {number}/{len(plan['slots'])} {slot['trial_id']}: "
              f"{result['status']} validation={result.get('validation_loss')}", flush=True)
    summary = {"status": "REAL_EXTENSION_COMPLETE", "slot_count": len(outcomes),
               "eligible": sum(result["status"] == "ELIGIBLE" for _, result in outcomes),
               "near_collapse_ineligible": sum(result["status"] == "INELIGIBLE_NEAR_COLLAPSE"
                                                for _, result in outcomes),
               "failed": sum(result["status"].startswith("FAILED") for _, result in outcomes),
               "plan_sha256": file_sha256(_root(project) / "extension" / "plan.json"),
               "result_table_sha256": file_sha256(table),
               "test_or_scientific_outcomes_used": False}
    _save_new_or_same(_root(project) / "extension" / "summary.json", summary)
    return summary


def _checked_extension(project: Path) -> tuple[dict, dict, dict, dict, dict]:
    initial, _, part, checked = _checked_parent(project)
    plan = _seal_extension_plan(project)
    path = _root(project) / "extension"
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if (summary["status"] != "REAL_EXTENSION_COMPLETE" or summary["slot_count"] != 108 or
            summary["plan_sha256"] != file_sha256(path / "plan.json") or
            summary["result_table_sha256"] != file_sha256(path / "results.csv")):
        raise RuntimeError("Real extension is not complete/immutable")
    results = _extension_records(project, plan)
    if (sum(result["status"] == "ELIGIBLE" for _, result in results) != summary["eligible"] or
            sum(result["status"] == "INELIGIBLE_NEAR_COLLAPSE" for _, result in results) !=
            summary["near_collapse_ineligible"]):
        raise RuntimeError("Real extension status counts differ")
    ledgers = {window: list(rows) for window, rows in checked["ledgers"].items()}
    for slot, result in results:
        ledgers[slot["config"]["domain"]].append(_selection_row(slot, result))
    return initial, plan, part, checked, {"summary": summary, "results": results, "ledgers": ledgers}


def build_finalist_plan(project: Path) -> dict:
    initial, extension, part, checked, completed = _checked_extension(project)
    slots = []
    cells = {}
    for window in (f"real_w{size}" for size in WINDOWS):
        ledger = completed["ledgers"][window]
        if any(row not in ledger for row in checked["ledgers"][window]):
            raise RuntimeError("window-specific initial ledger missing")
        for objective in OBJECTIVES:
            extended = extension["paired_triggers"][f"{window}/{objective}"]
            for architecture in ARCHITECTURES:
                finalists = finalist_indices(ledger, "real", architecture, objective, extended)
                ranking = ranked_search_candidates(
                    ledger, "real", architecture, objective,
                    (0, 1, 2, 3, 4, 5) if extended else (0, 1, 2, 3))
                key = f"{window}/{architecture}/{objective}"
                cells[key] = {"window": window, "architecture": architecture,
                              "objective": objective, "extended": extended,
                              "eligible_search_ranking": ranking, "finalists": list(finalists)}
                for index in finalists:
                    for seed in FINALIST_SEEDS:
                        for population in POPULATIONS:
                            slots.append(_new_slot(project, part, checked["candidates"], window,
                                                   architecture, objective, index, seed,
                                                   population, "finalists"))
    if len(cells) != 32 or len({slot["trial_id"] for slot in slots}) != len(slots):
        raise RuntimeError("Real finalist cell/fit identity invalid")
    artifact = {"study_id": STUDY, "stage": "finalist_robustness", "slot_count": len(slots),
                "parent_initial_plan_sha256": file_sha256(_initial_root(project) / "initial_plan.json"),
                "parent_extension_plan_sha256": file_sha256(_root(project) / "extension" / "plan.json"),
                "parent_extension_summary_sha256": file_sha256(_root(project) / "extension" / "summary.json"),
                "partition_sha256": initial["partition_sha256"],
                "sealed_candidate_sha256": checked["seal"],
                "protocol_sha256": file_sha256(project / PROTOCOL_RELATIVE),
                "source_snapshot": _source_snapshot(project), "cells": cells,
                "slots": slots, "test_or_scientific_outcomes_used": False, "v2_write": False}
    return artifact


def _seal_finalist_plan(project: Path) -> dict:
    plan = build_finalist_plan(project)
    _save_new_or_same(_root(project) / "finalists" / "plan.json", plan)
    return plan


def run_finalists(project: Path) -> dict:
    plan = _seal_finalist_plan(project)
    outcomes = []
    table = _root(project) / "finalists" / "results.csv"
    for number, slot in enumerate(plan["slots"], 1):
        result = fit_real_trial(project, slot)
        outcomes.append((slot, result))
        _rolling_table(table, outcomes)
        print(f"Real finalist {number}/{len(plan['slots'])} {slot['trial_id']}: "
              f"{result['status']} validation={result.get('validation_loss')}", flush=True)
    _, extension_plan, _, checked, completed = _checked_extension(project)
    ledgers = {window: list(rows) for window, rows in completed["ledgers"].items()}
    initial_slots = json.loads((_initial_root(project) / "initial_plan.json").read_text())["slots"]
    all_trials = [(slot, _verified_result(slot)) for slot in initial_slots]
    all_trials.extend(completed["results"])
    all_trials.extend(outcomes)
    for slot, result in outcomes:
        ledgers[slot["config"]["domain"]].append(_selection_row(slot, result))
    winners = []
    for key, cell in plan["cells"].items():
        window, architecture, objective = key.split("/")
        finalists = cell["finalists"]
        selected = select_finalist(ledgers[window], "real", architecture, objective, finalists)
        details = []
        for index in finalists:
            matching = [row for row in ledgers[window] if
                        row["architecture"] == architecture and row["objective"] == objective and
                        row["candidate_index"] == index]
            eligible = (len(matching) == 9 and all(row["status"] == "ELIGIBLE" for row in matching))
            details.append({"candidate_index": index,
                            "candidate_id": checked["candidates"][objective][index].candidate_id,
                            "hyperparameters": checked["candidates"][objective][index].numeric(),
                            "eligible_all_seeds_populations": eligible,
                            "mean_validation_loss": (sum(float(row["validation_loss"]) for row in matching) / 9
                                                     if eligible else None),
                            "seed_population_results": matching})
        checkpoints = []
        if selected is not None:
            for slot, result in all_trials:
                config = slot["config"]
                if (config["domain"], config["architecture"], config["objective"],
                        config["candidate_index"]) != (window, architecture, objective, selected):
                    continue
                checkpoints.append({"trial_id": slot["trial_id"],
                                    "population": config["population"],
                                    "training_seed_root": config["training_seed_root"],
                                    "config_sha256": slot["config_sha256"],
                                    "selected_update": result.get("selected_update"),
                                    "stopping_update": result.get("stopping_update"),
                                    "checkpoint_path": str(Path(slot["output_path"]) /
                                                           "best_validation_checkpoint.pt"),
                                    "checkpoint_sha256": result["artifact_sha256"][
                                        "best_validation_checkpoint.pt"]})
            if len(checkpoints) != 9:
                raise RuntimeError(f"winner lacks nine frozen HPO checkpoints: {key}")
        winner = {"window": window, "architecture": architecture, "objective": objective,
                  "frozen_finalists": finalists, "finalist_details": details,
                  "selected_candidate_index": selected,
                  "status": "WINNER" if selected is not None else "UNSTABLE_PHASE2A",
                  "selected_checkpoints": checkpoints,
                  "selection_uses_test_or_scientific_outcomes": False}
        _save_new_or_same(_root(project) / "winners" / f"{window}_{architecture}_{objective}.json", winner)
        winners.append(winner)
    completion = {"status": "REAL_PHASE2A_HPO_COMPLETE", "initial_fits_reused": 384,
                  "extension_fits": len(extension_plan["slots"]),
                  "finalist_fits": len(outcomes),
                  "extension_failed": completed["summary"]["failed"],
                  "finalist_failed": sum(result["status"].startswith("FAILED") for _, result in outcomes),
                  "finalist_ineligible_near_collapse": sum(
                      result["status"] == "INELIGIBLE_NEAR_COLLAPSE" for _, result in outcomes),
                  "winner_count": sum(winner["selected_candidate_index"] is not None for winner in winners),
                  "unstable_count": sum(winner["selected_candidate_index"] is None for winner in winners),
                  "winners": [{"window": winner["window"], "architecture": winner["architecture"],
                               "objective": winner["objective"],
                               "candidate_index": winner["selected_candidate_index"],
                               "status": winner["status"]} for winner in winners],
                  "initial_summary_sha256": file_sha256(_initial_root(project) / "initial_summary.json"),
                  "extension_summary_sha256": file_sha256(_root(project) / "extension" / "summary.json"),
                  "finalist_plan_sha256": file_sha256(_root(project) / "finalists" / "plan.json"),
                  "finalist_result_table_sha256": file_sha256(table),
                  "test_or_scientific_outcomes_used": False}
    _save_new_or_same(_root(project) / "completion_summary.json", completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-extension", action="store_true")
    mode.add_argument("--run-extension", action="store_true")
    mode.add_argument("--plan-finalists", action="store_true")
    mode.add_argument("--run-finalists", action="store_true")
    args = parser.parse_args()
    if args.plan_extension:
        result = _seal_extension_plan(ROOT)
    elif args.run_extension:
        result = run_extension(ROOT)
    elif args.plan_finalists:
        result = _seal_finalist_plan(ROOT)
    else:
        result = run_finalists(ROOT)
    print(_canonical_json({key: value for key, value in result.items()
                           if key not in {"slots", "cells", "source_snapshot"}}), flush=True)


if __name__ == "__main__":
    main()
