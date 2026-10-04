"""Gate and execute the 160 frozen initial Phase-2A fits, without test access."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from neurobridge.experiments.phase2a_hpo import (
    ARCHITECTURES, DOMAINS, INITIAL_CANDIDATES, OBJECTIVES, OUTPUT_RELATIVE,
    PROTOCOL_RELATIVE, STUDY_ID, ProtocolViolation, _canonical_json,
    channel_partition, extension_required, file_sha256, finalist_indices,
    guarded_output_root, load_sealed_candidates, plan_initial,
    ranked_search_candidates, source_hashes,
)
from neurobridge.experiments.phase2a_safe_fit import fit_trial, load_safe_bundle


def _json_new_or_same(path: Path, payload: Any) -> None:
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise ProtocolViolation(f"immutable artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(encoded)


def _snapshot(project: Path, output: Path) -> dict:
    """Copy the exact current sources, including dirty/untracked Python files."""
    sources = source_hashes(project)
    sources["src/neurobridge/experiments/phase2a_initial.py"] = file_sha256(Path(__file__))
    for name in ("tools/phase2a_initial.py", "tests/test_phase2a_hpo.py", "tests/test_phase2a_safe_execution.py"):
        sources[name] = file_sha256(project / name)
    destination = output / "studies" / STUDY_ID / "source_snapshot"
    manifest = destination / "manifest.json"
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous["source_file_sha256"] != sources:
            raise ProtocolViolation("source changed after immutable study snapshot; new study ID required")
        for name, digest in sources.items():
            if file_sha256(destination / name) != digest:
                raise ProtocolViolation("source snapshot content changed")
        return previous
    if destination.exists() and any(destination.iterdir()):
        raise ProtocolViolation("partial source snapshot; preserve and use a new study ID")
    for name, digest in sources.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / name, target)
        if file_sha256(target) != digest:
            raise ProtocolViolation(f"source copy hash mismatch: {name}")
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=project, text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else f"UNAVAILABLE: {result.stderr.strip()}"
    payload = {"study_id": STUDY_ID, "source_file_sha256": sources,
               "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
               "protocol_sha256": sources[PROTOCOL_RELATIVE.as_posix()]}
    _json_new_or_same(manifest, payload)
    return payload


def gate(project_root: Path) -> tuple[list[dict], dict]:
    """No complete V2 container is opened here: only sealed plan/safe bundles."""
    project = project_root.resolve()
    output = guarded_output_root(project)
    candidates, checksum = load_sealed_candidates(project / PROTOCOL_RELATIVE)
    channels, partition_hash = channel_partition(project)
    expected = plan_initial(project, candidates, channels)
    frozen = json.loads((output / "dry_run" / "initial_plan.json").read_text(encoding="utf-8"))
    if (frozen["candidate_sha256"] != checksum or
            [(row["trial_id"], row["config_sha256"], row["config"]) for row in frozen["initial_trials"]] !=
            [(row["trial_id"], row["config_sha256"], row["config"]) for row in expected]):
        raise ProtocolViolation("sealed dry-run plan differs from current frozen trial enumeration")
    if len(expected) != 160 or {row["config"]["candidate_index"] for row in expected} != set(INITIAL_CANDIDATES):
        raise ProtocolViolation("initial-only budget changed")
    if any(output not in Path(row["output_path"]).resolve().parents for row in expected):
        raise ProtocolViolation("initial plan can write outside HPO root")
    manifests = {}
    for domain, populations in DOMAINS.items():
        for population in populations:
            safe_root, manifest, split, _, values = load_safe_bundle(project, domain, population)
            if (not manifest["test_trial_intersection_empty"] or not manifest["all_windows_are_train_or_validation"]
                    or len(values["trial_id"]) != manifest["train_window_count"] + manifest["validation_window_count"]
                    or not manifest.get("original_split_sha256") or not manifest.get("raw_source_sha256")
                    or not manifest.get("extractor_source_sha256")
                    or manifest["extractor_source_sha256"] != file_sha256(project / "src/neurobridge/experiments/phase2a_safe_inputs.py")):
                raise ProtocolViolation("safe bundle manifest/provenance incomplete")
            manifests[f"{domain}/{population}"] = {
                "manifest_sha256": file_sha256(safe_root / "manifest.json"),
                "windows_sha256": manifest["safe_windows_sha256"],
                "split_sha256": manifest["safe_split_sha256"],
                "parent_source_sha256": manifest["raw_source_sha256"],
                "original_split_sha256": manifest["original_split_sha256"],
                "train_window_count": manifest["train_window_count"],
                "validation_window_count": manifest["validation_window_count"],
            }
    snapshot = _snapshot(project, output)
    smoke_specs = (("synthetic", "A", "soft"), ("real", "TOTAL65", "time_contrastive_blocks"))
    smoke_results = {}
    for domain, population, objective in smoke_specs:
        row = next(row for row in expected if row["config"]["domain"] == domain
                   and row["config"]["population"] == population
                   and row["config"]["objective"] == objective
                   and row["config"]["architecture"] == "cnn1d"
                   and row["config"]["candidate_index"] == 0)
        result = fit_trial(project, row, smoke_updates=1)
        if result["status"] not in {"ELIGIBLE", "INELIGIBLE_NEAR_COLLAPSE"} or result["stopping_update"] != 1:
            raise ProtocolViolation(f"safe one-update smoke failed: {domain}/{population}/{objective}")
        smoke_results[f"{domain}/{population}/{objective}"] = {
            "status": result["status"], "validation_loss": result["validation_loss"],
            "optimizer_updates": result["stopping_update"],
        }
    gate_record = {"gate": "GO", "candidate_sha256": checksum,
                   "channel_partition_sha256": partition_hash,
                   "source_snapshot_manifest_sha256": file_sha256(output / "studies" / STUDY_ID / "source_snapshot" / "manifest.json"),
                   "safe_inputs": manifests, "smoke": smoke_results,
                   "initial_fit_count": len(expected), "test_path_used_by_fit": False,
                   "v2_output_writable": False}
    _json_new_or_same(output / "studies" / STUDY_ID / "go_gate.json", gate_record)
    return expected, gate_record


def _selection_rows(plan: list[dict], results: list[dict]) -> list[dict]:
    rows = []
    for trial, result in zip(plan, results):
        config = trial["config"]
        rows.append({
            "domain": config["domain"], "population": config["population"],
            "architecture": config["architecture"], "objective": config["objective"],
            "candidate_index": config["candidate_index"],
            "training_seed_root": config["training_seed_root"],
            "status": result["status"], "validation_loss": result.get("validation_loss"),
            "selected_update": result.get("selected_update"),
            "stopping_update": result.get("stopping_update"),
        })
    return rows


def execute_initial(project_root: Path) -> dict:
    project = project_root.resolve()
    plan, gate_record = gate(project)
    results = []
    for number, trial in enumerate(plan, start=1):
        result = fit_trial(project, trial)
        results.append(result)
        print(f"Phase-2A {number}/160 {trial['trial_id']}: {result['status']} "
              f"best={result.get('validation_loss')} updates={result.get('stopping_update')}", flush=True)
    rows = _selection_rows(plan, results)
    output = guarded_output_root(project) / "studies" / STUDY_ID
    table = output / "initial_fit_results.csv"
    if table.exists():
        raise ProtocolViolation("initial result table already exists; no overwrite")
    columns = ["trial_id", "domain", "population", "architecture", "objective", "candidate_index",
               "training_seed_root", "training_seed_effective", "learning_rate", "weight_decay",
               "active_temperature", "status", "validation_loss", "selected_update", "stopping_update",
               "distance_q50", "fraction_distance_lt_0_01", "covariance_trace", "near_collapse",
               "training_seconds", "trainable_parameters", "peak_gpu_memory_allocated", "device", "error"]
    with table.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for trial, result in zip(plan, results):
            config = trial["config"]
            geometry = result.get("geometry", {})
            writer.writerow({**{key: config.get(key) for key in columns},
                             "trial_id": trial["trial_id"], "status": result["status"],
                             **{key: result.get(key) for key in ("validation_loss", "selected_update", "stopping_update",
                                                                      "training_seconds", "trainable_parameters", "peak_gpu_memory_allocated", "device", "error")},
                             **{key: geometry.get(key) for key in ("distance_q50", "fraction_distance_lt_0_01",
                                                                       "covariance_trace", "near_collapse")}})
    cells = []
    for domain in DOMAINS:
        for objective in OBJECTIVES:
            extended = extension_required(rows, domain, objective)
            for architecture in ARCHITECTURES:
                ranking = ranked_search_candidates(rows, domain, architecture, objective)
                cells.append({"domain": domain, "objective": objective, "architecture": architecture,
                              "eligible_candidates": [index for index, _ in ranking],
                              "ranked_validation_means": ranking,
                              "extension": "REQUIRED_FOR_EXTENSION" if extended else "NO_EXTENSION"})
    summary = {"gate": gate_record, "attempted_fits": len(results),
               "completed_fits": sum(result["status"] in {"ELIGIBLE", "INELIGIBLE_NEAR_COLLAPSE"} for result in results),
               "failed_fits": sum(result["status"].startswith("FAILED") for result in results),
               "cells": cells, "results_csv": str(table), "results_csv_sha256": file_sha256(table)}
    _json_new_or_same(output / "initial_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--run-initial", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    if args.gate_only == args.run_initial:
        parser.error("select exactly one of --gate-only or --run-initial")
    if args.gate_only:
        _, report = gate(args.project_root)
    else:
        report = execute_initial(args.project_root)
    print(_canonical_json(report))


if __name__ == "__main__":
    main()
