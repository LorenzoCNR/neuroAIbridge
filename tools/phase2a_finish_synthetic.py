"""Finish only the frozen Synthetic Phase-2A initial slots; never enter Real."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurobridge.experiments.phase2a_hpo import (  # noqa: E402
    _canonical_json, file_sha256, guarded_output_root, ranked_search_candidates,
)
from neurobridge.experiments.phase2a_safe_fit import fit_trial  # noqa: E402
from neurobridge.experiments.phase2a_window_campaign import (  # noqa: E402
    STUDY_ID, _cell_key, _rolling_tables, _selection_records, _verify_reuse,
)


def _save_new_or_same(path: Path, payload: object) -> None:
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)


def main() -> None:
    project = ROOT.resolve()
    output = guarded_output_root(project) / "studies" / STUDY_ID
    gate = json.loads((output / "go_gate.json").read_text(encoding="utf-8"))
    if gate["gate"] != "GO" or gate["slot_count"] != 448:
        raise RuntimeError("frozen gate not GO")
    frozen = json.loads((output / "initial_plan.json").read_text(encoding="utf-8"))
    slots = [dict(s) for s in frozen["slots"] if s["config"]["domain"] == "synthetic"]
    if len(slots) != 64 or any(s["config"]["domain"] != "synthetic" for s in slots):
        raise RuntimeError("Synthetic-only plan invariant failed")
    snapshot = json.loads((output / "source_snapshot" / "manifest.json").read_text(encoding="utf-8"))
    for relative, digest in snapshot["source_file_sha256"].items():
        if file_sha256(project / relative) != digest:
            raise RuntimeError(f"frozen source drift: {relative}")
    prior_retry_records = []
    prior_retry_path = output / "synthetic_only_retry_manifest.json"
    if prior_retry_path.is_file():
        prior_retry_records = json.loads(prior_retry_path.read_text(encoding="utf-8"))
    retry_records = []
    path_records = []
    for slot in slots:
        if slot["source"] == "reused_verified_old_synthetic":
            continue
        target = Path(slot["output_path"])
        previous = next((r for r in prior_retry_records if r["planned_trial_id"] == slot["trial_id"]), None)
        if previous is not None:
            if file_sha256(target / "trial_record.json") != previous["preserved_trial_record_sha256"]:
                raise RuntimeError(f"prior partial changed: {target}")
            old_id = slot["trial_id"]
            slot["trial_id"] = previous["retry_trial_id"]
            slot["output_path"] = str(output / "trials" / slot["trial_id"])
            slot["source"] = "new_synthetic_fit_after_user_pause"
            retry_records.append(previous)
            continue
        result_path = target / "result.json"
        failed = False
        if result_path.is_file():
            prior_result = json.loads(result_path.read_text(encoding="utf-8"))
            failed = prior_result["status"].startswith("FAILED")
            if not failed:
                continue
        partial = target.exists() and not result_path.is_file()
        checkpoint_length = len(str((target / "best_validation_checkpoint.pt").resolve()))
        if not (failed or partial or checkpoint_length >= 250):
            continue
        prior_record_hash = None
        if failed or partial:
            record_path = target / "trial_record.json"
            if not record_path.is_file():
                raise RuntimeError(f"unexplained existing trial: {target}")
            preserved = json.loads(record_path.read_text(encoding="utf-8"))
            if preserved["config_sha256"] != slot["config_sha256"] or preserved["config"] != slot["config"]:
                raise RuntimeError(f"existing trial provenance mismatch: {target}")
            prior_record_hash = file_sha256(record_path)
        old_id = slot["trial_id"]
        config = slot["config"]
        objective_code = {"soft": "s", "infonce": "i", "time_contrastive_blocks": "t", "behavior_contrastive_blocks": "b"}[config["objective"]]
        architecture_code = "c" if config["architecture"] == "cnn1d" else "t"
        short_id = (f"p2a-syn-{architecture_code}{objective_code}-{config['population']}"
                    f"-c{config['candidate_index']}-s1101-{slot['config_sha256'][:16]}")
        slot["trial_id"] = short_id
        slot["output_path"] = str(output / "trials" / short_id)
        slot["source"] = ("new_synthetic_fit_retry_after_path_failure" if failed else
                          "new_synthetic_fit_retry_after_user_pause" if partial else
                          "new_synthetic_fit_short_path")
        path_records.append({"planned_trial_id": old_id, "actual_trial_id": short_id,
                             "original_output_path": str(target), "actual_output_path": slot["output_path"],
                             "original_checkpoint_path_length": checkpoint_length,
                             "prior_state": "failed" if failed else "interrupted" if partial else "not_started",
                             "prior_trial_record_sha256": prior_record_hash,
                             "prior_result_sha256": file_sha256(result_path) if failed else None,
                             "config_sha256": slot["config_sha256"]})
    _save_new_or_same(output / "synthetic_only_retry_manifest.json", retry_records)
    _save_new_or_same(output / "synthetic_only_short_path_manifest.json", path_records)
    results = []
    cells = {}
    for number, slot in enumerate(slots, 1):
        result = (_verify_reuse(project, slot) if slot["source"] == "reused_verified_old_synthetic"
                  else fit_trial(project, slot))
        results.append(result)
        print(f"Synthetic {number}/64 {slot['trial_id']}: {result['status']} "
              f"validation={result.get('validation_loss')} updates={result.get('stopping_update')}", flush=True)
        cell = _cell_key(slot)
        cell_slots = [s for s in slots if _cell_key(s) == cell]
        if slot["trial_id"] != cell_slots[-1]["trial_id"]:
            continue
        records = _selection_records(slots[:number], results, cell)
        ranked = ranked_search_candidates(records, "synthetic", cell[1], cell[2])
        key = "/".join(cell)
        cells[key] = {"domain_window": cell[0], "architecture": cell[1], "objective": cell[2],
                      "eligible_ranked_candidates": ranked,
                      "current_finalists": [index for index, _ in ranked[:2]],
                      "candidate_population_results": records,
                      "extension_status": "PENDING_PAIRED_ARCHITECTURE"}
        mate = "/".join((cell[0], "transformer" if cell[1] == "cnn1d" else "cnn1d", cell[2]))
        if mate in cells:
            extension = (len(cells[key]["eligible_ranked_candidates"]) < 2 or
                         len(cells[mate]["eligible_ranked_candidates"]) < 2)
            for current in (key, mate):
                cells[current]["extension_status"] = "REQUIRED_FOR_EXTENSION" if extension else "NO_EXTENSION"
        if number > 16:
            _rolling_tables(output, slots[:number], results, cells)
            _save_new_or_same(output / "cells" / f"{key.replace('/', '_')}.json", cells[key])
    summary = {"status": "SYNTHETIC_INITIAL_COMPLETE", "synthetic_slots": len(slots),
               "valid_completed": sum(r["status"] in {"ELIGIBLE", "INELIGIBLE_NEAR_COLLAPSE"} for r in results),
               "failed": sum(r["status"].startswith("FAILED") for r in results),
               "reused_old": sum(s["source"] == "reused_verified_old_synthetic" for s in slots),
               "retried_after_user_pause": len(retry_records), "real_fits_executed_by_this_script": 0,
               "cells": cells}
    _save_new_or_same(output / "synthetic_only_summary.json", summary)
    print(_canonical_json(summary), flush=True)


if __name__ == "__main__":
    main()
