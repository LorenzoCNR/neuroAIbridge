"""Freeze validation-only Real HPO winners and the one-window final study.

No test container, scientific metric or model fitting is accessible here.
The selection rule is a separate write-once artifact created before this
program computes the cross-window winner.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from neurobridge.experiments.phase2a_hpo import (  # noqa: E402
    ARCHITECTURES, OBJECTIVES, PROTOCOL_RELATIVE, SEARCH_SEED, _canonical_json,
    file_sha256, ranked_search_candidates,
)
from final_thesis_real_hpo import (  # noqa: E402
    POPULATIONS, _checked_extension, _initial_root, _new_slot, _root,
    _verified_result,
)

FREEZE_RELATIVE = Path("outputs/final_thesis_v1/freeze")
RULE_NAME = "REAL_SELECTION_RULE_v1.json"
WINNERS_NAME = "REAL_HPO_FINAL_WINNERS_SEED1101.csv"
WINDOWS_NAME = "REAL_WINDOW_SELECTION.csv"
SPEC_NAME = "REAL_FINAL_EXPERIMENT_SPEC.json"
FINAL_SEEDS = (1101, 1201, 1301)
WINDOWS = (21, 41, 121, 201)
WINNER_FIELDS = (
    "window_size", "architecture", "objective", "candidate_index", "candidate_id",
    "learning_rate", "weight_decay", "active_temperature", "validation_mean_3pop",
    "validation_total65", "validation_a_proximal", "validation_b_distal",
    "status_total65", "status_a_proximal", "status_b_distal", "eligible_candidate_count",
    "paired_extension_triggered", "trial_total65", "trial_a_proximal", "trial_b_distal",
    "checkpoint_sha256_total65", "checkpoint_sha256_a_proximal",
    "checkpoint_sha256_b_distal", "hpo_selection_seed", "multiseed_hpo_selection_checked",
)
WINDOW_FIELDS = (
    "window_size", "architecture", "objective", "candidate_index",
    "winner_validation_mean_3pop", "rank_within_architecture_objective",
    "rank_sum_8_cells", "rank_mean_8_cells", "rank_median_8_cells",
    "selected_canonical_window", "selection_scope", "valid_validation_centers",
)


def _save_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable freeze artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _csv_bytes(fields: tuple[str, ...], rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _checked_rule(project: Path, initial: dict, extension: dict, checked: dict) -> tuple[dict, str]:
    path = project / FREEZE_RELATIVE / RULE_NAME
    rule = json.loads(path.read_text(encoding="utf-8"))
    source = rule["source"]
    expected = {
        "initial_summary_sha256": file_sha256(_initial_root(project) / "initial_summary.json"),
        "extension_plan_sha256": file_sha256(_root(project) / "extension" / "plan.json"),
        "extension_result_table_sha256": file_sha256(_root(project) / "extension" / "results.csv"),
        "candidate_seal_sha256": checked["seal"],
        "somatotopic_partition_sha256": initial["partition_sha256"],
    }
    if (any(source[key] != value for key, value in expected.items()) or
            rule["decision_class"] != "NEW_PROTOCOL_DECISION" or
            rule["hpo_training_seed"] != SEARCH_SEED or
            rule["final_training_seed_roots"] != list(FINAL_SEEDS) or
            rule["window_selection"]["windows_bins"] != list(WINDOWS) or
            rule["window_selection"]["test_or_scientific_metrics_allowed"] or
            rule["finalist_hpo_fits_authorized"] or
            rule["hyperparameter_selection_multiseed_robustness_performed"] or
            extension["slot_count"] != 108):
        raise RuntimeError("write-once Real selection rule or parent hashes differ")
    return rule, file_sha256(path)


def _all_trials(project: Path, initial: dict, extension: dict, extension_results: list) -> dict:
    combined = [(slot, _verified_result(slot)) for slot in initial["slots"]]
    combined.extend(extension_results)
    if len(combined) != 492 or len({slot["trial_id"] for slot, _ in combined}) != 492:
        raise RuntimeError("expected exactly 492 unique completed Real HPO fits")
    found = {}
    for slot, result in combined:
        config = slot["config"]
        key = (config["window_size"], config["architecture"], config["objective"],
               config["candidate_index"], config["population"])
        if key in found or config["training_seed_root"] != SEARCH_SEED:
            raise RuntimeError(f"duplicate or non-search-seed HPO fit: {key}")
        found[key] = slot, result
    return found


def _winner_rows(project: Path, initial: dict, extension: dict, part: dict,
                 checked: dict, completed: dict) -> tuple[list[dict], dict]:
    trials = _all_trials(project, initial, extension, completed["results"])
    winners = []
    winner_slots = {}
    for window in WINDOWS:
        domain = f"real_w{window}"
        ledger = completed["ledgers"][domain]
        for architecture in ARCHITECTURES:
            for objective in OBJECTIVES:
                extended = extension["paired_triggers"][f"{domain}/{objective}"]
                indices = (0, 1, 2, 3, 4, 5) if extended else (0, 1, 2, 3)
                ranking = ranked_search_candidates(ledger, "real", architecture,
                                                   objective, indices)
                if not ranking:
                    raise RuntimeError(f"no all-population eligible candidate: {domain}/{architecture}/{objective}")
                candidate_index, mean_loss = ranking[0]
                candidate = checked["candidates"][objective][candidate_index]
                picked = {population: trials[(window, architecture, objective,
                                              candidate_index, population)]
                          for population in POPULATIONS}
                losses = {population: float(result["validation_loss"])
                          for population, (_, result) in picked.items()}
                if (any(result["status"] != "ELIGIBLE" for _, result in picked.values()) or
                        abs(sum(losses.values()) / 3 - mean_loss) > 1e-12):
                    raise RuntimeError("winner eligibility or three-population score differs")
                row = {
                    "window_size": window, "architecture": architecture, "objective": objective,
                    "candidate_index": candidate_index, "candidate_id": candidate.candidate_id,
                    "learning_rate": candidate.lr, "weight_decay": candidate.weight_decay,
                    "active_temperature": candidate.temperature,
                    "validation_mean_3pop": mean_loss,
                    "validation_total65": losses["TOTAL65"],
                    "validation_a_proximal": losses["A_PROXIMAL"],
                    "validation_b_distal": losses["B_DISTAL"],
                    "status_total65": picked["TOTAL65"][1]["status"],
                    "status_a_proximal": picked["A_PROXIMAL"][1]["status"],
                    "status_b_distal": picked["B_DISTAL"][1]["status"],
                    "eligible_candidate_count": len(ranking),
                    "paired_extension_triggered": str(extended).lower(),
                    "trial_total65": picked["TOTAL65"][0]["trial_id"],
                    "trial_a_proximal": picked["A_PROXIMAL"][0]["trial_id"],
                    "trial_b_distal": picked["B_DISTAL"][0]["trial_id"],
                    "checkpoint_sha256_total65": picked["TOTAL65"][1]["artifact_sha256"][
                        "best_validation_checkpoint.pt"],
                    "checkpoint_sha256_a_proximal": picked["A_PROXIMAL"][1]["artifact_sha256"][
                        "best_validation_checkpoint.pt"],
                    "checkpoint_sha256_b_distal": picked["B_DISTAL"][1]["artifact_sha256"][
                        "best_validation_checkpoint.pt"],
                    "hpo_selection_seed": SEARCH_SEED,
                    "multiseed_hpo_selection_checked": "false",
                }
                winners.append(row)
                winner_slots[(window, architecture, objective)] = picked
    if len(winners) != 32:
        raise RuntimeError("Real HPO freeze must contain 32 window-specific winners")
    return winners, winner_slots


def _window_rows(winners: list[dict], support: dict[int, int]) -> tuple[list[dict], int, dict]:
    ranks = {window: [] for window in WINDOWS}
    rank_by_cell = {}
    for architecture in ARCHITECTURES:
        for objective in OBJECTIVES:
            group = [row for row in winners if row["architecture"] == architecture and
                     row["objective"] == objective]
            if len(group) != 4:
                raise RuntimeError("window-ranking cell lacks four windows")
            ordered = sorted(group, key=lambda row: (row["validation_mean_3pop"],
                                                    row["window_size"]))
            for rank, row in enumerate(ordered, 1):
                window = row["window_size"]
                ranks[window].append(rank)
                rank_by_cell[(window, architecture, objective)] = rank
    aggregates = {window: {"rank_sum": sum(values), "rank_mean": sum(values) / 8,
                           "rank_median": statistics.median(values), "ranks": values}
                  for window, values in ranks.items()}
    if any(len(item["ranks"]) != 8 for item in aggregates.values()):
        raise RuntimeError("window rank aggregation requires eight cells")
    selected = min(WINDOWS, key=lambda window: (aggregates[window]["rank_sum"],
                                                aggregates[window]["rank_median"], window))
    rows = []
    for winner in winners:
        window = winner["window_size"]
        rows.append({"window_size": window, "architecture": winner["architecture"],
                     "objective": winner["objective"],
                     "candidate_index": winner["candidate_index"],
                     "winner_validation_mean_3pop": winner["validation_mean_3pop"],
                     "rank_within_architecture_objective": rank_by_cell[
                         (window, winner["architecture"], winner["objective"])],
                     "rank_sum_8_cells": aggregates[window]["rank_sum"],
                     "rank_mean_8_cells": aggregates[window]["rank_mean"],
                     "rank_median_8_cells": aggregates[window]["rank_median"],
                     "selected_canonical_window": str(window == selected).lower(),
                     "selection_scope": "validation_only_seed1101",
                     "valid_validation_centers": support[window]})
    return rows, selected, aggregates


def _final_slots(project: Path, window: int, part: dict, checked: dict,
                 winner_slots: dict) -> tuple[list[dict], list[dict]]:
    result = []
    configurations = []
    for architecture in ARCHITECTURES:
        for objective in OBJECTIVES:
            picked = winner_slots[(window, architecture, objective)]
            index = picked["TOTAL65"][0]["config"]["candidate_index"]
            candidate = checked["candidates"][objective][index]
            configurations.append({"architecture": architecture, "objective": objective,
                                   "candidate_index": index, "candidate_id": candidate.candidate_id,
                                   "learning_rate": candidate.lr,
                                   "weight_decay": candidate.weight_decay,
                                   "active_temperature": candidate.temperature})
            for population in POPULATIONS:
                for seed in FINAL_SEEDS:
                    planned = _new_slot(project, part, checked["candidates"],
                                        f"real_w{window}", architecture, objective,
                                        index, seed, population, "unused_finalist_path")
                    config = planned["config"]
                    digest = planned["config_sha256"]
                    if seed == SEARCH_SEED:
                        source_slot, source_result = picked[population]
                        if source_slot["config_sha256"] != digest:
                            raise RuntimeError("seed-1101 HPO checkpoint config cannot be reused")
                        result.append({"architecture": architecture, "objective": objective,
                                       "population": population, "training_seed_root": seed,
                                       "candidate_index": index,
                                       "config_sha256": digest,
                                       "trial_id": source_slot["trial_id"],
                                       "reuse_hpo_checkpoint": True,
                                       "checkpoint_path": str(Path(source_slot["output_path"]) /
                                                              "best_validation_checkpoint.pt"),
                                       "checkpoint_sha256": source_result["artifact_sha256"][
                                           "best_validation_checkpoint.pt"],
                                       "selected_update": source_result["selected_update"],
                                       "safe_input_reference": planned["safe_input_reference"],
                                       "safe_input_reference_sha256": planned[
                                           "safe_input_reference_sha256"]})
                    else:
                        ident = (f"ft-final-w{window}-{population}-{architecture}-"
                                 f"{objective}-c{index}-s{seed}-{digest[:12]}")
                        target = project / "outputs/final_thesis_v1/held_out/trials" / ident
                        if len(str((target / "best_validation_checkpoint.pt").resolve())) >= 250:
                            raise RuntimeError("final Real checkpoint path too long")
                        result.append({"architecture": architecture, "objective": objective,
                                       "population": population, "training_seed_root": seed,
                                       "candidate_index": index,
                                       "config_sha256": digest, "trial_id": ident,
                                       "reuse_hpo_checkpoint": False,
                                       "output_path": str(target),
                                       "safe_input_reference": planned["safe_input_reference"],
                                       "safe_input_reference_sha256": planned[
                                           "safe_input_reference_sha256"]})
    if (len(result) != 72 or len(configurations) != 8 or
            sum(slot["reuse_hpo_checkpoint"] for slot in result) != 24 or
            len({slot["trial_id"] for slot in result}) != 72):
        raise RuntimeError("final Real plan must be 72 instances with 24 HPO checkpoints reused")
    return result, configurations


def build_freeze(project: Path) -> tuple[bytes, bytes, dict]:
    initial, extension, part, checked, completed = _checked_extension(project)
    rule, rule_sha = _checked_rule(project, initial, extension, checked)
    winners, winner_slots = _winner_rows(project, initial, extension, part,
                                         checked, completed)
    support = {}
    source_split_hashes = set()
    for window in WINDOWS:
        manifest = json.loads((project / "outputs/phase2a_hpo/safe_inputs" /
                               f"real_w{window}/TOTAL65/manifest.json").read_text(encoding="utf-8"))
        support[window] = manifest["valid_validation_window_count"]
        source_split_hashes.add(manifest["original_split_sha256"])
    if support != {21: 11600, 41: 11200, 121: 9600, 201: 8000} or len(source_split_hashes) != 1:
        raise RuntimeError("validation support or original split changed")
    window_rows, selected, aggregates = _window_rows(winners, support)
    winner_bytes = _csv_bytes(WINNER_FIELDS, winners)
    window_bytes = _csv_bytes(WINDOW_FIELDS, window_rows)
    slots, configurations = _final_slots(project, selected, part, checked, winner_slots)
    safe_references = {population: {
        "path": str(project / "outputs/phase2a_hpo/safe_inputs/real_somatotopic_v1" /
                    f"real_w{selected}" / population / "reference_manifest.json"),
        "sha256": file_sha256(project / "outputs/phase2a_hpo/safe_inputs/real_somatotopic_v1" /
                              f"real_w{selected}" / population / "reference_manifest.json")}
        for population in POPULATIONS}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project,
                          capture_output=True, text=True, check=False)
    spec = {
        "spec_version": "real_final_one_window_seed1101_hpo_v1",
        "decision_class": "NEW_PROTOCOL_DECISION",
        "hpo_status": "CLOSED_AFTER_492_FITS",
        "initial_fit_count_reused_for_selection": 384,
        "extension_fit_count_reused_for_selection": 108,
        "finalist_hpo_fit_count": 0,
        "multiseed_hyperparameter_selection_robustness_performed": False,
        "selection_rule_path": str(project / FREEZE_RELATIVE / RULE_NAME),
        "selection_rule_sha256": rule_sha,
        "hpo_winners_csv_sha256": hashlib.sha256(winner_bytes).hexdigest(),
        "window_selection_csv_sha256": hashlib.sha256(window_bytes).hexdigest(),
        "selected_real_window_size": selected,
        "all_explored_real_windows": list(WINDOWS),
        "window_rank_summary": {str(window): aggregates[window] for window in WINDOWS},
        "window_selection_interpretation": rule["interpretation_limit"],
        "populations": list(POPULATIONS),
        "architecture_objective_configurations": configurations,
        "final_training_seed_roots": list(FINAL_SEEDS),
        "final_neural_model_instances": 72,
        "reused_seed1101_hpo_checkpoints": 24,
        "new_seed1201_1301_fits": 48,
        "held_out_is_primary_scientific_evidence": True,
        "full_sample_refits_deferred_descriptive_only": True,
        "no_test_or_scientific_metric_used_for_selection": True,
        "somatotopic_partition_path": str(project / "outputs/phase2a_hpo/partitions/real_somatotopic_v1.json"),
        "somatotopic_partition_sha256": initial["partition_sha256"],
        "canonical_design_sha256": initial["canonical_design_sha256"],
        "sealed_candidate_sha256": checked["seal"],
        "original_frozen_split_sha256": next(iter(source_split_hashes)),
        "safe_input_references": safe_references,
        "initial_summary_sha256": file_sha256(_initial_root(project) / "initial_summary.json"),
        "extension_summary_sha256": file_sha256(_root(project) / "extension/summary.json"),
        "protocol_sha256": file_sha256(project / PROTOCOL_RELATIVE),
        "freeze_source_sha256": file_sha256(Path(__file__)),
        "git_head_before_final_training": head.stdout.strip() if head.returncode == 0 else "UNAVAILABLE",
        "final_slots": slots,
    }
    return winner_bytes, window_bytes, spec


def main() -> None:
    winner_bytes, window_bytes, spec = build_freeze(ROOT)
    output = ROOT / FREEZE_RELATIVE
    _save_once(output / WINNERS_NAME, winner_bytes)
    _save_once(output / WINDOWS_NAME, window_bytes)
    _save_once(output / SPEC_NAME, (_canonical_json(spec) + "\n").encode("utf-8"))
    print(_canonical_json({"selected_real_window_size": spec["selected_real_window_size"],
                           "final_neural_model_instances": spec["final_neural_model_instances"],
                           "reused_seed1101_hpo_checkpoints": spec[
                               "reused_seed1101_hpo_checkpoints"],
                           "new_seed1201_1301_fits": spec["new_seed1201_1301_fits"],
                           "window_rank_summary": spec["window_rank_summary"]}), flush=True)


if __name__ == "__main__":
    main()
