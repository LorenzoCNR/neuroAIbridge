"""Execute frozen Synthetic Phase-2A finalists using safe validation-only inputs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurobridge.experiments.phase2a_hpo import (  # noqa: E402
    ARCHITECTURES, FINALIST_SEEDS, OBJECTIVES, PROTOCOL_RELATIVE,
    SEALED_CANDIDATE_SHA256, _canonical_json, file_sha256, guarded_output_root,
    load_sealed_candidates, resolved_config, select_finalist,
)
from neurobridge.experiments.phase2a_safe_fit import fit_trial  # noqa: E402
from neurobridge.experiments.phase2a_window_campaign import STUDY_ID  # noqa: E402

FINALISTS = {
    ("cnn1d", "soft"): (2, 3), ("transformer", "soft"): (3, 2),
    ("cnn1d", "infonce"): (3, 0), ("transformer", "infonce"): (3, 2),
    ("cnn1d", "time_contrastive_blocks"): (2, 1),
    ("transformer", "time_contrastive_blocks"): (2, 1),
    ("cnn1d", "behavior_contrastive_blocks"): (0, 2),
    ("transformer", "behavior_contrastive_blocks"): (2, 3),
}


def save_new_or_same(path: Path, value: object) -> None:
    data = (_canonical_json(value) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def verified_result(root: Path, trial_id: str, config_sha: str) -> tuple[dict, dict]:
    record = json.loads((root / "trial_record.json").read_text(encoding="utf-8"))
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    if record["trial_id"] != trial_id or result["trial_id"] != trial_id or record["config_sha256"] != config_sha:
        raise RuntimeError(f"initial trial provenance mismatch: {root}")
    if result["status"] not in {"ELIGIBLE", "INELIGIBLE_NEAR_COLLAPSE"}:
        raise RuntimeError(f"initial finalist is not valid: {root}")
    for name, digest in result["artifact_sha256"].items():
        if file_sha256(root / name) != digest:
            raise RuntimeError(f"initial artifact hash mismatch: {root / name}")
    return record, result


def selection_row(config: dict, result: dict) -> dict:
    return {"domain": "synthetic", "population": config["population"],
            "architecture": config["architecture"], "objective": config["objective"],
            "candidate_index": config["candidate_index"],
            "training_seed_root": config["training_seed_root"], "status": result["status"],
            "validation_loss": result.get("validation_loss"),
            "selected_update": result.get("selected_update"),
            "stopping_update": result.get("stopping_update")}


def main() -> None:
    project = ROOT.resolve()
    output = guarded_output_root(project) / "studies" / STUDY_ID
    initial = json.loads((output / "synthetic_only_summary.json").read_text(encoding="utf-8"))
    if initial["status"] != "SYNTHETIC_INITIAL_COMPLETE" or initial["valid_completed"] != 64:
        raise RuntimeError("Synthetic initial study incomplete")
    candidates, seal = load_sealed_candidates(project / PROTOCOL_RELATIVE)
    if seal != SEALED_CANDIDATE_SHA256:
        raise RuntimeError("candidate seal mismatch")
    snapshot = json.loads((output / "source_snapshot" / "manifest.json").read_text(encoding="utf-8"))
    for relative, digest in snapshot["source_file_sha256"].items():
        if file_sha256(project / relative) != digest:
            raise RuntimeError(f"frozen initial source drift: {relative}")
    rows = list(csv.DictReader((output / "rolling_fit_results.csv").open(newline="", encoding="utf-8")))
    if len(rows) != 64 or {r["domain"] for r in rows} != {"synthetic"}:
        raise RuntimeError("initial table must contain exactly 64 Synthetic rows")
    expected_finalists = {f"synthetic/{arch}/{obj}": list(pair) for (arch, obj), pair in FINALISTS.items()}
    if {key: value["current_finalists"] for key, value in initial["cells"].items()} != expected_finalists:
        raise RuntimeError("finalist list differs from frozen initial rankings")
    if any(cell["extension_status"] != "NO_EXTENSION" for cell in initial["cells"].values()):
        raise RuntimeError("extension required; finalist plan would be invalid")
    initial_records = []
    for row in rows:
        root = (guarded_output_root(project) / "studies" / "phase2a_frozen_2026-09-28" / "trials" / row["trial_id"]
                if row["source"] == "reused_verified_old_synthetic" else output / "trials" / row["trial_id"])
        record = json.loads((root / "trial_record.json").read_text(encoding="utf-8"))
        record, result = verified_result(root, row["trial_id"], record["config_sha256"])
        if (record["config"]["architecture"] != row["architecture"] or
                record["config"]["objective"] != row["objective"] or
                record["config"]["candidate_index"] != int(row["candidate_index"]) or
                record["config"]["population"] != row["population"]):
            raise RuntimeError("initial CSV vs immutable record mismatch")
        initial_records.append({"record": record, "result": result, "path": str(root),
                                "record_sha256": file_sha256(root / "trial_record.json"),
                                "result_sha256": file_sha256(root / "result.json")})
    plan = []
    destination = output / "synthetic_finalists"
    for objective in OBJECTIVES:
        for architecture in ARCHITECTURES:
            for index in FINALISTS[(architecture, objective)]:
                for seed in FINALIST_SEEDS:
                    for population in ("A", "B"):
                        config = resolved_config("synthetic", population, architecture,
                                                 candidates[objective][index], seed)
                        digest = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()
                        objective_code = {"soft": "s", "infonce": "i", "time_contrastive_blocks": "t",
                                          "behavior_contrastive_blocks": "b"}[objective]
                        architecture_code = "c" if architecture == "cnn1d" else "t"
                        identifier = (f"p2a-syn-fin-{architecture_code}{objective_code}-{population}"
                                      f"-c{index}-s{seed}-{digest[:16]}")
                        plan.append({"trial_id": identifier, "candidate_id": config["candidate_id"],
                                     "config": config, "config_sha256": digest,
                                     "output_path": str(destination / "trials" / identifier),
                                     "safe_input_manifest_planned": str(guarded_output_root(project) / "safe_inputs" /
                                                                        "synthetic" / population / "manifest.json")})
    if len(plan) != 64 or len({p["trial_id"] for p in plan}) != 64:
        raise RuntimeError("expected 64 unique finalist fits")
    manifest = {"parent_initial_summary_sha256": file_sha256(output / "synthetic_only_summary.json"),
                "parent_initial_table_sha256": file_sha256(output / "rolling_fit_results.csv"),
                "sealed_candidate_sha256": seal,
                "protocol_sha256": file_sha256(project / PROTOCOL_RELATIVE),
                "runner_sha256": file_sha256(Path(__file__)),
                "safe_bundle_manifest_sha256": {
                    pop: file_sha256(guarded_output_root(project) / "safe_inputs" / "synthetic" / pop / "manifest.json")
                    for pop in ("A", "B")},
                "initial_result_provenance": initial_records, "plan": plan,
                "test_artifacts_used": False, "v2_write": False}
    save_new_or_same(destination / "finalist_plan.json", manifest)
    result_rows = []
    for number, trial in enumerate(plan, 1):
        result = fit_trial(project, trial)
        result_rows.append((trial, result))
        print(f"Synthetic finalist {number}/64 {trial['trial_id']}: {result['status']} "
              f"validation={result.get('validation_loss')}", flush=True)
        table = destination / "finalist_results.csv"
        temp = table.with_suffix(".csv.tmp")
        columns = ("trial_id", "architecture", "objective", "candidate_index", "population",
                   "training_seed_root", "training_seed_effective", "learning_rate", "weight_decay",
                   "active_temperature", "status", "validation_loss", "selected_update", "stopping_update",
                   "near_collapse", "distance_q50", "fraction_distance_lt_0_01", "covariance_trace",
                   "config_sha256", "checkpoint_sha256", "trial_record_sha256", "result_sha256")
        with temp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for current, outcome in result_rows:
                config = current["config"]
                root = Path(current["output_path"])
                writer.writerow({**{field: config.get(field) for field in columns},
                                 "trial_id": current["trial_id"], "config_sha256": current["config_sha256"],
                                 "status": outcome["status"], "validation_loss": outcome.get("validation_loss"),
                                 "selected_update": outcome.get("selected_update"),
                                 "stopping_update": outcome.get("stopping_update"),
                                 "checkpoint_sha256": outcome.get("artifact_sha256", {}).get("best_validation_checkpoint.pt"),
                                 "trial_record_sha256": file_sha256(root / "trial_record.json"),
                                 "result_sha256": file_sha256(root / "result.json"),
                                 **{field: outcome.get("geometry", {}).get(field) for field in
                                    ("near_collapse", "distance_q50", "fraction_distance_lt_0_01", "covariance_trace")}})
        os.replace(temp, table)
    selection = []
    for item in initial_records:
        selection.append(selection_row(item["record"]["config"], item["result"]))
    for trial, result in result_rows:
        selection.append(selection_row(trial["config"], result))
    winners = []
    for objective in OBJECTIVES:
        for architecture in ARCHITECTURES:
            finalists = FINALISTS[(architecture, objective)]
            winner = select_finalist(selection, "synthetic", architecture, objective, finalists)
            detail = []
            for index in finalists:
                matching = [r for r in selection if r["architecture"] == architecture and
                            r["objective"] == objective and r["candidate_index"] == index]
                eligible = len(matching) == 6 and all(r["status"] == "ELIGIBLE" for r in matching)
                detail.append({"candidate_index": index, "candidate_id": candidates[objective][index].candidate_id,
                               "hyperparameters": candidates[objective][index].numeric(),
                               "eligible_all_seeds_populations": eligible,
                               "mean_validation_loss": (sum(r["validation_loss"] for r in matching) / 6
                                                        if eligible else None),
                               "seed_population_results": matching})
            selected_checkpoints = []
            if winner is not None:
                for item in initial_records:
                    config = item["record"]["config"]
                    if (config["architecture"], config["objective"], config["candidate_index"]) == (architecture, objective, winner):
                        selected_checkpoints.append({"seed": 1101, "population": config["population"],
                                                     "checkpoint_path": str(Path(item["path"]) / "best_validation_checkpoint.pt"),
                                                     "checkpoint_sha256": item["result"]["artifact_sha256"]["best_validation_checkpoint.pt"],
                                                     "config_sha256": item["record"]["config_sha256"]})
                for trial, result in result_rows:
                    config = trial["config"]
                    if (config["architecture"], config["objective"], config["candidate_index"]) == (architecture, objective, winner):
                        selected_checkpoints.append({"seed": config["training_seed_root"], "population": config["population"],
                                                     "checkpoint_path": str(Path(trial["output_path"]) / "best_validation_checkpoint.pt"),
                                                     "checkpoint_sha256": result["artifact_sha256"]["best_validation_checkpoint.pt"],
                                                     "config_sha256": trial["config_sha256"]})
            winner_record = {"architecture": architecture, "objective": objective,
                             "frozen_finalists": list(finalists), "finalist_details": detail,
                             "selected_candidate_index": winner,
                             "status": "WINNER" if winner is not None else "UNSTABLE_PHASE2A",
                             "selected_checkpoints": selected_checkpoints,
                             "selection_uses_test_or_scientific_outcomes": False}
            save_new_or_same(destination / "winners" / f"{architecture}_{objective}.json", winner_record)
            winners.append(winner_record)
    completion = {"status": "SYNTHETIC_PHASE2A_HPO_COMPLETE", "initial_fits_reused": 64,
                  "new_finalist_fits": 64, "finalist_failed": sum(r["status"].startswith("FAILED") for _, r in result_rows),
                  "finalist_ineligible_near_collapse": sum(r["status"] == "INELIGIBLE_NEAR_COLLAPSE" for _, r in result_rows),
                  "winner_count": sum(w["selected_candidate_index"] is not None for w in winners),
                  "winners": [{"architecture": w["architecture"], "objective": w["objective"],
                               "candidate_index": w["selected_candidate_index"], "status": w["status"]} for w in winners],
                  "finalist_plan_sha256": file_sha256(destination / "finalist_plan.json"),
                  "finalist_table_sha256": file_sha256(destination / "finalist_results.csv"),
                  "test_or_scientific_outcomes_used": False, "real_fit_executed_by_this_runner": False}
    save_new_or_same(destination / "completion_summary.json", completion)
    print(_canonical_json(completion), flush=True)


if __name__ == "__main__":
    main()
