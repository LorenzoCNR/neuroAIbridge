"""Versioned 448-slot Phase-2A campaign with rolling validation-only reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from neurobridge.experiments.phase2a_hpo import (
    ARCHITECTURES, INITIAL_CANDIDATES, OBJECTIVES, OUTPUT_RELATIVE,
    PROTOCOL_RELATIVE, ProtocolViolation, _canonical_json, channel_partition,
    file_sha256, guarded_output_root, load_sealed_candidates,
    ranked_search_candidates, resolved_config, source_hashes,
)
from neurobridge.experiments.phase2a_safe_fit import fit_trial
from neurobridge.experiments.phase2a_window_inputs import WINDOWS


STUDY_ID = "phase2a_real_windows_2026-09-28"
AMENDMENT_RELATIVE = Path("outputs/phase2a_hpo/PHASE2A_REAL_WINDOW_AMENDMENT.md")
OLD_STUDY = "phase2a_frozen_2026-09-28"


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_new_or_same(path: Path, payload: Any) -> None:
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise ProtocolViolation(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(encoded)


def plan(project_root: Path) -> dict:
    project = project_root.resolve()
    output = guarded_output_root(project)
    candidates, candidate_sha = load_sealed_candidates(project / PROTOCOL_RELATIVE)
    channels, partition_sha = channel_partition(project)
    old_plan = json.loads((output / "dry_run/initial_plan.json").read_text(encoding="utf-8"))
    if old_plan["candidate_sha256"] != candidate_sha:
        raise ProtocolViolation("old Synthetic plan candidate seal changed")
    old_synthetic = [entry for entry in old_plan["initial_trials"] if entry["config"]["domain"] == "synthetic"]
    if len(old_synthetic) != 64:
        raise ProtocolViolation("old Synthetic plan does not contain 64 slots")
    amendment_sha = file_sha256(project / AMENDMENT_RELATIVE)
    slots = []
    for entry in old_synthetic:
        row = dict(entry)
        old_result = Path(entry["output_path"]) / "result.json"
        if old_result.is_file():
            row["source"] = "reused_verified_old_synthetic"
            row["old_trial_id"] = entry["trial_id"]
        else:
            row["source"] = "new_synthetic_fit"
            row["old_trial_id"] = entry["trial_id"]
            row["trial_id"] = f"{STUDY_ID}-{entry['trial_id'].split('-', 3)[-1]}"
            row["output_path"] = str(output / "studies" / STUDY_ID / "trials" / row["trial_id"])
        slots.append(row)
    for window in WINDOWS:
        domain = f"real_w{window}"
        for objective in OBJECTIVES:
            for architecture in ARCHITECTURES:
                for index in INITIAL_CANDIDATES:
                    for population in ("TOTAL65", "A", "B"):
                        config = resolved_config("real", population, architecture,
                                                 candidates[objective][index], 1101, channels[population])
                        config["domain"] = domain
                        config["window_size"] = window
                        config_sha = _sha(config)
                        identifier = (f"{STUDY_ID}-{domain}-{population}-{architecture}-{objective}"
                                      f"-c{index}-s1101-{config_sha[:16]}")
                        slots.append({"trial_id": identifier, "candidate_id": config["candidate_id"],
                                      "config": config, "config_sha256": config_sha,
                                      "output_path": str(output / "studies" / STUDY_ID / "trials" / identifier),
                                      "safe_input_manifest_planned": str(output / "safe_inputs" / domain / population / "manifest.json"),
                                      "source": "new_real_window_fit"})
    if len(slots) != 448 or len({row["trial_id"] for row in slots}) != 448:
        raise ProtocolViolation("amended initial plan must have 448 unique slots")
    if sum(row["source"] == "reused_verified_old_synthetic" for row in slots) != 10:
        raise ProtocolViolation("expected 10 completed Synthetic fits for verified reuse")
    if sum(row["config"]["domain"] == "synthetic" for row in slots) != 64:
        raise ProtocolViolation("Synthetic count changed")
    for window in WINDOWS:
        if sum(row["config"]["domain"] == f"real_w{window}" for row in slots) != 96:
            raise ProtocolViolation("Real window fit count changed")
    artifact = {"study_id": STUDY_ID, "amendment_sha256": amendment_sha,
                "candidate_sha256": candidate_sha, "channel_partition_sha256": partition_sha,
                "slot_count": 448, "reused_completed_synthetic_slots": 10,
                "new_fits_remaining": 438, "slots": slots}
    _write_new_or_same(output / "studies" / STUDY_ID / "initial_plan.json", artifact)
    return artifact


def _verify_reuse(project: Path, slot: dict) -> dict:
    old_root = Path(slot["output_path"])
    if guarded_output_root(project) not in old_root.resolve().parents:
        raise ProtocolViolation("reused trial is outside HPO root")
    record = json.loads((old_root / "trial_record.json").read_text(encoding="utf-8"))
    result = json.loads((old_root / "result.json").read_text(encoding="utf-8"))
    old_snapshot = json.loads((guarded_output_root(project) / "studies" / OLD_STUDY /
                               "source_snapshot/manifest.json").read_text(encoding="utf-8"))
    expected_adapter = old_snapshot["source_file_sha256"]["src/neurobridge/experiments/phase2a_safe_fit.py"]
    if (record["trial_id"] != slot["trial_id"] or record["config_sha256"] != slot["config_sha256"]
            or record["config"] != slot["config"] or record["adapter_source_sha256"] != expected_adapter
            or result["trial_id"] != slot["trial_id"]
            or result["status"] not in {"ELIGIBLE", "INELIGIBLE_NEAR_COLLAPSE"}):
        raise ProtocolViolation("old Synthetic run identity/source/config invalid for reuse")
    for name, digest in result["artifact_sha256"].items():
        if file_sha256(old_root / name) != digest:
            raise ProtocolViolation("old Synthetic checkpoint/diagnostic hash changed")
    return result


def _snapshot(project: Path, output: Path) -> dict:
    source = source_hashes(project)
    for relative in ("tools/phase2a_window_campaign.py", "tests/test_phase2a_window_campaign.py",
                     AMENDMENT_RELATIVE.as_posix()):
        source[relative] = file_sha256(project / relative)
    target = output / "source_snapshot"
    manifest_path = target / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source_file_sha256"] != source:
            raise ProtocolViolation("amended study source changed after snapshot")
        for relative, digest in source.items():
            if file_sha256(target / relative) != digest:
                raise ProtocolViolation("amended source snapshot corrupted")
        return manifest
    if target.exists() and any(target.iterdir()):
        raise ProtocolViolation("partial amended source snapshot")
    for relative, digest in source.items():
        copy = target / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / relative, copy)
        if file_sha256(copy) != digest:
            raise ProtocolViolation("amended source copy mismatch")
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=project, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"
    manifest = {"source_file_sha256": source, "git_head": git("rev-parse", "HEAD"),
                "git_status": git("status", "--short"),
                "amendment_sha256": file_sha256(project / AMENDMENT_RELATIVE)}
    _write_new_or_same(manifest_path, manifest)
    return manifest


def gate(project_root: Path) -> tuple[dict, dict]:
    project = project_root.resolve()
    campaign = plan(project)
    output = guarded_output_root(project) / "studies" / STUDY_ID
    reused = [_verify_reuse(project, row) for row in campaign["slots"]
              if row["source"] == "reused_verified_old_synthetic"]
    safe_hashes = {}
    safe_output = guarded_output_root(project)
    for domain in ("synthetic", *(f"real_w{window}" for window in WINDOWS)):
        for population in (("A", "B") if domain == "synthetic" else ("TOTAL65", "A", "B")):
            safe_root = safe_output / "safe_inputs" / domain / population
            manifest = json.loads((safe_root / "manifest.json").read_text(encoding="utf-8"))
            split = json.loads((safe_root / "split.json").read_text(encoding="utf-8"))
            if (set(split) != {"train", "validation"} or
                    split["train"] != manifest["train_trial_ids"] or
                    split["validation"] != manifest["validation_trial_ids"] or
                    file_sha256(safe_root / "windows.npz") != manifest["safe_windows_sha256"] or
                    file_sha256(safe_root / "split.json") != manifest["safe_split_sha256"]):
                raise ProtocolViolation("amended safe bundle hash/split mismatch")
            with np.load(safe_root / "windows.npz", allow_pickle=False) as windows:
                trial_ids, split_rows = windows["trial_id"], windows["split"]
            if (not manifest["test_trial_intersection_empty"] or
                    not manifest["all_windows_are_train_or_validation"] or
                    len(trial_ids) != manifest["train_window_count"] + manifest["validation_window_count"] or
                    set(np.unique(trial_ids)) != set(split["train"] + split["validation"]) or
                    set(np.unique(split_rows)) != {"train", "validation"} or
                    not manifest.get("original_split_sha256") or not manifest.get("raw_source_sha256")):
                raise ProtocolViolation("amended safe-input manifest incomplete")
            if domain.startswith("real_w") and manifest["window_size"] != int(domain[6:]):
                raise ProtocolViolation("Real window domain/manifest mismatch")
            safe_hashes[f"{domain}/{population}"] = {
                "manifest_sha256": file_sha256(safe_root / "manifest.json"),
                "windows_sha256": manifest["safe_windows_sha256"],
                "split_sha256": manifest["safe_split_sha256"],
            }
    snapshot = _snapshot(project, output)
    # Smoke both the V2-continuity condition and the maximum new window.
    smoke = {}
    for domain in ("real_w21", "real_w201"):
        row = next(row for row in campaign["slots"] if row["config"]["domain"] == domain
                   and row["config"]["population"] == "TOTAL65"
                   and row["config"]["architecture"] == "cnn1d"
                   and row["config"]["objective"] == "soft"
                   and row["config"]["candidate_index"] == 0)
        result = fit_trial(project, row, smoke_updates=1)
        if result["status"] not in {"ELIGIBLE", "INELIGIBLE_NEAR_COLLAPSE"} or result["stopping_update"] != 1:
            raise ProtocolViolation(f"amended Real safe-fit smoke failed: {domain}: {result.get('error')}")
        smoke[domain] = {"status": result["status"], "validation_loss": result["validation_loss"],
                         "updates": result["stopping_update"]}
    record = {"gate": "GO", "slot_count": 448, "reused_synthetic": len(reused),
              "new_fits": 438, "amendment_sha256": campaign["amendment_sha256"],
              "candidate_sha256": campaign["candidate_sha256"],
              "snapshot_sha256": file_sha256(output / "source_snapshot/manifest.json"),
              "safe_input_hashes": safe_hashes, "smoke": smoke,
              "test_access_by_fit_or_selection": False, "v2_write": False}
    _write_new_or_same(output / "go_gate.json", record)
    return campaign, record


def _cell_key(slot: dict) -> tuple[str, str, str]:
    config = slot["config"]
    return config["domain"], config["architecture"], config["objective"]


def _selection_records(slots: list[dict], results: list[dict], cell: tuple[str, str, str]) -> list[dict]:
    domain, architecture, objective = cell
    normalized_domain = "synthetic" if domain == "synthetic" else "real"
    records = []
    for slot, result in zip(slots, results):
        config = slot["config"]
        if _cell_key(slot) != cell:
            continue
        records.append({"domain": normalized_domain, "population": config["population"],
                        "architecture": architecture, "objective": objective,
                        "candidate_index": config["candidate_index"],
                        "training_seed_root": 1101, "status": result["status"],
                        "validation_loss": result.get("validation_loss"),
                        "selected_update": result.get("selected_update"),
                        "stopping_update": result.get("stopping_update")})
    return records


def _rolling_tables(output: Path, slots: list[dict], results: list[dict], completed_cells: dict) -> None:
    """The rolling files alone are intentionally replaceable derived views."""
    table = output / "rolling_fit_results.csv"
    temp = table.with_suffix(".csv.tmp")
    fields = ("trial_id", "domain", "population", "architecture", "objective", "candidate_index",
              "training_seed_effective", "learning_rate", "weight_decay", "active_temperature",
              "status", "validation_loss", "selected_update", "stopping_update", "near_collapse",
              "distance_q50", "fraction_distance_lt_0_01", "covariance_trace",
              "training_seconds", "device", "error", "source")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for slot, result in zip(slots, results):
            config = slot["config"]
            geo = result.get("geometry", {})
            writer.writerow({**{key: config.get(key) for key in fields},
                             "trial_id": slot["trial_id"], "source": slot["source"],
                             **{key: result.get(key) for key in ("status", "validation_loss", "selected_update",
                                                                       "stopping_update", "training_seconds", "device", "error")},
                             **{key: geo.get(key) for key in ("near_collapse", "distance_q50",
                                                                   "fraction_distance_lt_0_01", "covariance_trace")}})
    os.replace(temp, table)
    _write_new_or_same(output / "cell_snapshots" / f"count_{len(results):03d}.json", completed_cells)


def execute(project_root: Path) -> dict:
    project = project_root.resolve()
    campaign, gate_record = gate(project)
    slots = campaign["slots"]
    output = guarded_output_root(project) / "studies" / STUDY_ID
    report_path = guarded_output_root(project) / "PHASE2A_INITIAL_HPO_REPORT.md"
    results = []
    cells: dict[str, dict] = {}
    for number, slot in enumerate(slots, start=1):
        result = (_verify_reuse(project, slot) if slot["source"] == "reused_verified_old_synthetic"
                  else fit_trial(project, slot))
        results.append(result)
        print(f"amended Phase-2A {number}/448 {slot['trial_id']}: {result['status']} "
              f"best={result.get('validation_loss')} updates={result.get('stopping_update')}", flush=True)
        cell = _cell_key(slot)
        expected_population_count = 2 if cell[0] == "synthetic" else 3
        cell_slots = [s for s in slots if _cell_key(s) == cell]
        if len(cell_slots) != 4 * expected_population_count:
            raise ProtocolViolation("cell plan count invalid")
        if slot["trial_id"] != cell_slots[-1]["trial_id"]:
            continue
        records = _selection_records(slots[:number], results, cell)
        ranked = ranked_search_candidates(records, "synthetic" if cell[0] == "synthetic" else "real",
                                          cell[1], cell[2])
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
        _rolling_tables(output, slots[:number], results, cells)
        _write_new_or_same(output / "cells" / f"{key.replace('/', '_')}.json", cells[key])
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n## Rolling cell: {key} ({number}/448 slots)\n\n")
            handle.write("Validation-only candidate/population results and ranking are in "
                         f"`studies/{STUDY_ID}/cells/{key.replace('/', '_')}.json`; ")
            handle.write(f"eligible ranking: {ranked}; extension: {cells[key]['extension_status']}. ")
            handle.write("No test or scientific outcome used for selection.\n")
    summary = {"gate": gate_record, "slots": 448, "reused_completed_synthetic": 10,
               "attempted_new_fits": 438, "completed": sum(r["status"] in {"ELIGIBLE", "INELIGIBLE_NEAR_COLLAPSE"} for r in results),
               "failed": sum(r["status"].startswith("FAILED") for r in results),
               "cells": cells, "rolling_table": str(output / "rolling_fit_results.csv")}
    _write_new_or_same(output / "initial_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--run-initial", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    if sum((args.plan, args.gate_only, args.run_initial)) != 1:
        parser.error("choose exactly one action")
    if args.plan:
        result = plan(args.project_root)
        print(_canonical_json({key: value for key, value in result.items() if key != "slots"}))
    elif args.gate_only:
        _, result = gate(args.project_root)
        print(_canonical_json(result))
    else:
        result = execute(args.project_root)
        print(_canonical_json(result))


if __name__ == "__main__":
    main()
