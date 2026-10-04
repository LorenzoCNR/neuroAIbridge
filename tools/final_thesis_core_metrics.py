"""Frozen, checkpoint-only final core evaluation for NeuroBridge.

This driver has two explicit stages. ``--audit`` verifies parent lineage and
freezes evaluation support before loading any test features/targets. ``--run``
then reuses those immutable manifests, exports embeddings from frozen
checkpoints only, evaluates the existing held-out split, and writes a new
core-metrics branch. It never calls a training routine.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from neurobridge.eval.representation import (  # noqa: E402
    distance_geometry_correlation, evaluate_latent_recovery, lagged_alignment_by_trial_time,
    linear_cka, procrustes_r2,
)
from neurobridge.experiments import staged_shared_latent as synth  # noqa: E402
from neurobridge.experiments.phase2a_safe_fit import _frozen_config  # noqa: E402
from neurobridge.experiments.real_monkey import _participation_ratio  # noqa: E402
import final_real_embeddings as real_export  # noqa: E402

REAL_ROOT = Path("outputs/final_thesis_v1")
REAL_SPEC = REAL_ROOT / "freeze/REAL_FINAL_EXPERIMENT_SPEC.json"
REAL_FITS = REAL_ROOT / "held_out/FINAL_MODEL_FITS.csv"
REAL_SUMMARY = REAL_ROOT / "held_out/summary.json"
PARTITION = Path("outputs/phase2a_hpo/partitions/real_somatotopic_v1.json")
REAL_STAGE1 = Path("outputs/runs/v2_corrected_2026-09-23_seed42_real_total_65/stage01_data")
SYN_RUN = Path("outputs/runs/clean_rebuild_2026-09-23_seed42_synthetic")
SYN_STUDY = Path("outputs/phase2a_hpo/studies/phase2a_real_windows_2026-09-28/synthetic_finalists")
SYN_INITIAL = Path("outputs/phase2a_hpo/studies/phase2a_real_windows_2026-09-28")
CORE_REL = REAL_ROOT / "final_evaluation/core_metrics"
SUPPORT_NAME = "EVALUATION_SUPPORT_MANIFEST.json"
AUDIT_NAME = "AUDIT_MANIFEST.json"
ARCHS = ("cnn1d", "transformer")
OBJECTIVES = ("soft", "infonce", "time_contrastive_blocks", "behavior_contrastive_blocks")
SEEDS = (1101, 1201, 1301)
POPS_REAL = ("TOTAL65", "A_PROXIMAL", "B_DISTAL")
POPS_SYN = ("A", "B")
LAGS = tuple(range(-20, 21))
PAIR_COUNT = 100_000
PAIR_SAMPLE_SEED = 2026
PROBE_C_GRID = (0.1, 1.0, 10.0)
RIDGE_ALPHA_GRID = (0.1, 1.0, 10.0)
SYN_WINNERS = {
    ("cnn1d", "soft"): 2,
    ("cnn1d", "infonce"): 3,
    ("cnn1d", "time_contrastive_blocks"): 2,
    ("cnn1d", "behavior_contrastive_blocks"): 2,
    ("transformer", "soft"): 3,
    ("transformer", "infonce"): 3,
    ("transformer", "time_contrastive_blocks"): 2,
    ("transformer", "behavior_contrastive_blocks"): 2,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(blob).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _verify_real_status_consistency(statuses: list[str], near_collapse_flags: list[bool]) -> None:
    require(len(statuses) == len(near_collapse_flags) == 72,
            "REAL status inventory is not 72 fit/result records")
    require(statuses.count("ELIGIBLE") == 69 and
            statuses.count("INELIGIBLE_NEAR_COLLAPSE") == 3,
            "REAL per-slot status counts differ from frozen 69/3 summary")
    require(all((status == "INELIGIBLE_NEAR_COLLAPSE") == flag
                for status, flag in zip(statuses, near_collapse_flags)),
            "REAL near-collapse status disagrees with the result geometry flag")


def _verify_real_sources(project: Path) -> dict[str, Any]:
    spec_path, fits_path, summary_path = project / REAL_SPEC, project / REAL_FITS, project / REAL_SUMMARY
    spec, summary = read_json(spec_path), read_json(summary_path)
    fit_rows = list(csv.DictReader(fits_path.open(encoding="utf-8", newline="")))
    require(spec["selected_real_window_size"] == 201 and len(spec["final_slots"]) == 72,
            "frozen REAL final spec is not the 72-slot W201 design")
    require(spec["final_training_seed_roots"] == list(SEEDS), "REAL frozen seed set differs")
    require(summary["model_instances"] == 72 and summary["eligible"] == 69 and
            summary["near_collapse_ineligible"] == 3 and summary["failed"] == 0,
            "REAL completed fit summary differs from prompt")
    require(len(fit_rows) == 72 and len({r["trial_id"] for r in fit_rows}) == 72,
            "REAL final fit table has missing/duplicate slots")
    require(len({r["training_seed_root"] for r in fit_rows}) == 3,
            "REAL final table does not contain three training seeds")
    table = {r["trial_id"]: r for r in fit_rows}
    expected_cells = {(arch, objective, pop, seed) for arch in ARCHS for objective in OBJECTIVES
                      for pop in POPS_REAL for seed in SEEDS}
    observed_cells = {(r["architecture"], r["objective"], r["population"], int(r["training_seed_root"]))
                      for r in fit_rows}
    require(observed_cells == expected_cells,
            "REAL final fit grid differs from frozen architecture/objective/population/seed factorial")
    artifact_records: dict[str, Any] = {}
    observed_statuses: list[str] = []
    observed_collapse_flags: list[bool] = []
    for slot in spec["final_slots"]:
        row = table.get(slot["trial_id"])
        require(row is not None, f"REAL final slot absent from fit CSV: {slot['trial_id']}")
        require(int(row["window_size"]) == 201 and int(row["training_seed_root"]) == slot["training_seed_root"],
                f"REAL fit row differs from frozen slot {slot['trial_id']}")
        trial_root = (Path(slot["checkpoint_path"]).parent if slot.get("reuse_hpo_checkpoint")
                      else Path(slot["output_path"]))
        checkpoint = trial_root / "best_validation_checkpoint.pt"
        result_path, record_path = trial_root / "result.json", trial_root / "trial_record.json"
        result, record = read_json(result_path), read_json(record_path)
        require(result["trial_id"] == record["trial_id"] == slot["trial_id"],
                f"REAL checkpoint lineage trial ID differs: {slot['trial_id']}")
        require(record["config_sha256"] == slot["config_sha256"] == row["config_sha256"],
                f"REAL fit config hash differs across frozen spec/record/table: {slot['trial_id']}")
        require(result["status"] in {"ELIGIBLE", "INELIGIBLE_NEAR_COLLAPSE"},
                f"REAL slot failed rather than being a retained model: {slot['trial_id']}")
        geometry_flag = result.get("geometry", {}).get("near_collapse")
        require(isinstance(geometry_flag, bool),
                f"REAL near-collapse geometry flag is missing/not boolean: {slot['trial_id']}")
        observed_statuses.append(str(result["status"]))
        observed_collapse_flags.append(geometry_flag)
        expected = result["artifact_sha256"]
        for name, digest in expected.items():
            path = trial_root / name
            require(path.is_file() and sha256_file(path) == digest,
                    f"REAL artifact hash mismatch: {path}")
        checkpoint_sha = sha256_file(checkpoint)
        require(checkpoint_sha == row["checkpoint_sha256"] == expected[checkpoint.name],
                f"REAL checkpoint hash differs from final table: {slot['trial_id']}")
        require(sha256_file(result_path) == row["result_sha256"],
                f"REAL result hash differs from final table: {slot['trial_id']}")
        require(result["status"] == row["status"], f"REAL status mismatch: {slot['trial_id']}")
        artifact_records[slot["trial_id"]] = {
            "architecture": slot["architecture"], "objective": slot["objective"],
            "population": slot["population"], "seed": int(slot["training_seed_root"]),
            "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
            "trial_record_sha256": sha256_file(record_path),
            "result_sha256": sha256_file(result_path),
            "artifacts": expected, "fit_status": result["status"],
            "near_collapse": geometry_flag,
        }
    _verify_real_status_consistency(observed_statuses, observed_collapse_flags)
    return {"spec_sha256": sha256_file(spec_path), "fit_table_sha256": sha256_file(fits_path),
            "summary_sha256": sha256_file(summary_path), "summary": summary,
            "slots": artifact_records, "spec": spec, "fit_rows": fit_rows}


def _verify_synthetic_sources(project: Path) -> dict[str, Any]:
    winners_dir = project / SYN_STUDY / "winners"
    completion_path = project / SYN_STUDY / "completion_summary.json"
    plan_path, result_path = project / SYN_STUDY / "finalist_plan.json", project / SYN_STUDY / "finalist_results.csv"
    completion = read_json(completion_path)
    require(completion["status"] == "SYNTHETIC_PHASE2A_HPO_COMPLETE" and
            completion["winner_count"] == 8 and completion["finalist_failed"] == 0,
            "Synthetic Phase-2A completion manifest differs")
    finalist_rows = list(csv.DictReader(result_path.open(encoding="utf-8", newline="")))
    synthetic_config_path = project / SYN_RUN / "stage01_data/config.json"
    synthetic_config = read_json(synthetic_config_path)
    require(int(synthetic_config["lag_bins"]) == 10,
            "Synthetic frozen displacement is not the required +10 bins")
    winner_records, checkpoint_records = {}, {}
    expected_files = {f"{arch}_{obj}.json" for arch, obj in SYN_WINNERS}
    actual_files = {p.name for p in winners_dir.glob("*.json")}
    require(actual_files == expected_files, "Synthetic winner manifest set differs")
    for (arch, objective), candidate in SYN_WINNERS.items():
        path = winners_dir / f"{arch}_{objective}.json"
        manifest = read_json(path)
        require(manifest["status"] == "WINNER" and
                manifest["selected_candidate_index"] == candidate and
                not manifest["selection_uses_test_or_scientific_outcomes"],
                f"Synthetic frozen winner differs: {path.name}")
        selected = manifest["selected_checkpoints"]
        require(len(selected) == 6, f"Synthetic winner does not identify six final checkpoints: {path.name}")
        selected_cells: set[tuple[str, int]] = set()
        for item in selected:
            cp = Path(item["checkpoint_path"])
            root = cp.parent
            record_path, result_json = root / "trial_record.json", root / "result.json"
            record, result = read_json(record_path), read_json(result_json)
            cfg = record["config"]
            require(sha256_file(cp) == item["checkpoint_sha256"] ==
                    result["artifact_sha256"][cp.name], f"Synthetic checkpoint hash mismatch: {cp}")
            require(record["config_sha256"] == item["config_sha256"] and
                    cfg["architecture"] == arch and cfg["objective"] == objective and
                    cfg["candidate_index"] == candidate and cfg["population"] == item["population"] and
                    int(cfg["training_seed_root"]) == int(item["seed"]),
                    f"Synthetic selected checkpoint identity mismatch: {cp}")
            selected_cells.add((str(item["population"]), int(item["seed"])))
            require(result["status"] == "ELIGIBLE" and result["trial_id"] == record["trial_id"],
                    f"Synthetic frozen winner is not eligible: {cp}")
            for name, digest in result["artifact_sha256"].items():
                path_art = root / name
                require(path_art.is_file() and sha256_file(path_art) == digest,
                        f"Synthetic artifact hash mismatch: {path_art}")
            checkpoint_records[record["trial_id"]] = {
                "architecture": arch, "objective": objective, "population": item["population"],
                "seed": int(item["seed"]), "candidate_index": candidate,
                "checkpoint": str(cp), "checkpoint_sha256": sha256_file(cp),
                "config_sha256": record["config_sha256"], "result_sha256": sha256_file(result_json),
                "trial_record_sha256": sha256_file(record_path), "artifacts": result["artifact_sha256"],
                "fit_status": result["status"], "near_collapse": False,
            }
        require(selected_cells == {(pop, seed) for pop in POPS_SYN for seed in SEEDS},
                f"Synthetic winner checkpoint seed/population grid incomplete: {path.name}")
        winner_records[path.name] = {"sha256": sha256_file(path), "selected_candidate_index": candidate}
    require(len(checkpoint_records) == 48, "Synthetic final winner checkpoint count is not 48")
    near_collapse = [r for r in finalist_rows if r.get("status") == "INELIGIBLE_NEAR_COLLAPSE"]
    return {"completion_sha256": sha256_file(completion_path),
            "plan_sha256": sha256_file(plan_path), "results_sha256": sha256_file(result_path),
            "winner_manifests": winner_records, "slots": checkpoint_records,
            "nonwinner_near_collapse": near_collapse, "finalist_fit_count": len(finalist_rows)}


def _coordinates_hash(coords: list[list[int]]) -> str:
    return canonical_hash([[int(a), int(b)] for a, b in coords])


def _audit_manifest(project: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    real = _verify_real_sources(project)
    synthetic = _verify_synthetic_sources(project)
    real_stage1 = project / REAL_STAGE1
    real_manifest = read_json(real_stage1 / "manifest.json")
    real_reference_paths: dict[str, Path] = {}
    real_bundle_paths: dict[str, Path] = {}
    for slot in real["spec"]["final_slots"]:
        pop = slot["population"]
        ref_path = Path(slot["safe_input_reference"])
        if not ref_path.is_absolute():
            ref_path = project / ref_path
        require(sha256_file(ref_path) == slot["safe_input_reference_sha256"],
                f"REAL safe-input reference hash mismatch: {ref_path}")
        if pop in real_reference_paths:
            require(real_reference_paths[pop] == ref_path,
                    f"REAL slots use inconsistent safe-input references for {pop}")
        real_reference_paths[pop] = ref_path
    for pop, ref_path in real_reference_paths.items():
        ref = read_json(ref_path)
        bundle = Path(ref["bundle_path"])
        require(bundle.is_absolute() and bundle.is_dir(), f"REAL safe bundle is missing: {bundle}")
        manifest_path = bundle / "manifest.json"
        windows_path = bundle / "windows.npz"
        split_path = bundle / "split.json"
        require(sha256_file(manifest_path) == ref["bundle_manifest_sha256"],
                f"REAL safe-bundle manifest hash mismatch: {manifest_path}")
        manifest = read_json(manifest_path)
        require(sha256_file(windows_path) == ref["safe_windows_sha256"] == manifest["safe_windows_sha256"],
                f"REAL safe-bundle windows hash mismatch: {windows_path}")
        require(sha256_file(split_path) == ref["safe_split_sha256"] == manifest["safe_split_sha256"],
                f"REAL safe-bundle split hash mismatch: {split_path}")
        require(ref["population"] == manifest["population"] == pop and
                ref["domain"] == manifest["domain"] == "real_w201" and
                ref["original_split_sha256"] == manifest["original_split_sha256"],
                f"REAL safe-bundle identity differs from its reference: {pop}")
        real_bundle_paths[pop] = bundle
    part_path = project / PARTITION
    part = read_json(part_path)
    require(sha256_file(part_path) == real["spec"]["somatotopic_partition_sha256"],
            "REAL canonical channel-partition SHA mismatch")
    ids = part["feature_order_nlb_unit_ids"]
    ia, ib = part["A_PROXIMAL"]["feature_indices"], part["B_DISTAL"]["feature_indices"]
    require(len(ids) == 65 and len(ia) == 32 and len(ib) == 33 and not set(ia) & set(ib)
            and set(ia + ib) == set(range(65)), "REAL canonical 32+33 split invariant fails")
    real_paths = {
        "real_final_spec": project / REAL_SPEC,
        "real_final_fit_table": project / REAL_FITS,
        "real_final_summary": project / REAL_SUMMARY,
        "real_partition": part_path,
        "real_stage1_data": real_stage1 / "data.npz",
        "real_stage1_split": real_stage1 / "split.json",
        "real_stage1_manifest": real_stage1 / "manifest.json",
        "real_source_snapshot": project / "outputs/phase2a_hpo/studies/phase2a_real_somatotopic_v1_2026-09-28/source_snapshot_manifest.json",
        "real_freeze_source": project / "tools/final_thesis_freeze_real.py",
        "real_model_source": project / "src/neurobridge/experiments/real_monkey.py",
        "real_architecture_source": project / "src/neurobridge/models/temporal_cnn.py",
        "real_exporter_source": project / "tools/final_real_embeddings.py",
    }
    for pop in POPS_REAL:
        bundle = real_bundle_paths[pop]
        real_paths.update({f"real_{pop}_safe_reference": real_reference_paths[pop],
                           f"real_{pop}_safe_manifest": bundle / "manifest.json",
                           f"real_{pop}_safe_windows": bundle / "windows.npz",
                           f"real_{pop}_safe_split": bundle / "split.json"})
    for trial_id, record in real["slots"].items():
        for kind in ("checkpoint",):
            real_paths[f"real_slot_{trial_id}_{kind}"] = Path(record[kind])
        root = Path(record["checkpoint"]).parent
        for name in record["artifacts"]:
            real_paths[f"real_slot_{trial_id}_{name}"] = root / name
        real_paths[f"real_slot_{trial_id}_trial_record"] = root / "trial_record.json"
        real_paths[f"real_slot_{trial_id}_result"] = root / "result.json"
    real_hashes = {name: {"path": str(path), "sha256": sha256_file(path)}
                   for name, path in real_paths.items()}

    syn_stage1 = project / SYN_RUN / "stage01_data"
    syn_stage2 = project / SYN_RUN / "stage02_windows"
    pca_run = project / SYN_RUN
    pca_dir = pca_run / "stage04_embeddings/held_out"
    syn_paths = {
        "synthetic_data": syn_stage1 / "shared_data.npz",
        "synthetic_config": syn_stage1 / "config.json",
        "synthetic_split": syn_stage2 / "split.json",
        "synthetic_windows_A": syn_stage2 / "windows_A.npz",
        "synthetic_windows_B": syn_stage2 / "windows_B.npz",
        "synthetic_finalist_plan": project / SYN_STUDY / "finalist_plan.json",
        "synthetic_finalist_results": project / SYN_STUDY / "finalist_results.csv",
        "synthetic_completion": project / SYN_STUDY / "completion_summary.json",
        "synthetic_HPO_completion": project / SYN_INITIAL / "synthetic_only_summary.json",
        "synthetic_model_source": project / "src/neurobridge/experiments/staged_shared_latent.py",
        "synthetic_safe_fit_source": project / "src/neurobridge/experiments/phase2a_safe_fit.py",
        "representation_metric_source": project / "src/neurobridge/eval/representation.py",
    }
    for pop in POPS_SYN:
        syn_paths[f"synthetic_pca_{pop}_embedding"] = pca_dir / f"{pop}_pca_none.npz"
        syn_paths[f"synthetic_pca_{pop}_model"] = pca_run / f"stage03_models/held_out/{pop}_pca_none/pca.joblib"
    for name in synthetic["winner_manifests"]:
        syn_paths[f"synthetic_winner_{name}"] = project / SYN_STUDY / "winners" / name
    for trial_id, record in synthetic["slots"].items():
        cp = Path(record["checkpoint"])
        syn_paths[f"synthetic_slot_{trial_id}_checkpoint"] = cp
        for name in record["artifacts"]:
            syn_paths[f"synthetic_slot_{trial_id}_{name}"] = cp.parent / name
        syn_paths[f"synthetic_slot_{trial_id}_trial_record"] = cp.parent / "trial_record.json"
        syn_paths[f"synthetic_slot_{trial_id}_result"] = cp.parent / "result.json"
    for key, path in syn_paths.items():
        require(path.is_file(), f"missing Synthetic parent artifact: {path}")
    syn_hashes = {name: {"path": str(path), "sha256": sha256_file(path)}
                  for name, path in syn_paths.items()}

    # Verify that the frozen PCA baseline is already defined on the exact
    # canonical Synthetic cache/split. No PCA is refit in this task.
    pca_inputs = {}
    for pop in POPS_SYN:
        emb_path = pca_dir / f"{pop}_pca_none.npz"
        pca_model = pca_run / f"stage03_models/held_out/{pop}_pca_none/pca.joblib"
        pca_inputs[pop] = {"embedding_path": str(emb_path), "embedding_sha256": sha256_file(emb_path),
                           "pca_model_path": str(pca_model), "pca_model_sha256": sha256_file(pca_model)}

    audit = {
        "audit_version": "neurobridge_final_downstream_v1",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "no_training_or_hpo": True,
        "real": {k: v for k, v in real.items() if k not in {"spec", "fit_rows"}},
        "synthetic": synthetic,
        "source_artifact_hashes": {**real_hashes, **syn_hashes},
        "evaluation_driver_sha256": sha256_file(Path(__file__).resolve()),
        "canonical_pca_baseline": pca_inputs,
        "software": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                     "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                     "matplotlib": matplotlib.__version__},
        "near_collapse_policy": "retain every frozen final instance; no substitution; flag status in all outputs",
    }

    # Freeze exact held-out coordinates from split + validity metadata only;
    # no neural activity, behavior values, embeddings, or metric outcomes are
    # loaded while creating this support manifest.
    syn_split = read_json(syn_stage2 / "split.json")
    syn_support: dict[str, Any] = {}
    syn_windows: dict[str, dict[str, np.ndarray]] = {}
    for pop in POPS_SYN:
        with np.load(syn_stage2 / f"windows_{pop}.npz", allow_pickle=False) as npz:
            meta = {key: npz[key] for key in ("trial_id", "time_id", "global_time_id", "lag_valid")}
        syn_windows[pop] = meta
        trials = set(map(int, syn_split["test"]))
        base = np.isin(meta["trial_id"], list(trials))
        valid = meta["lag_valid"].astype(bool)
        keep = base if pop == "A" else base & valid
        coords = np.column_stack((meta["trial_id"][keep], meta["time_id"][keep])).astype(int).tolist()
        syn_support[pop] = {"test_trial_ids": sorted(trials), "rows_before_filtering": int(base.sum()),
                            "rows_after_frozen_validity": int(keep.sum()),
                            "evaluation_index_sha256": _coordinates_hash(coords),
                            "coordinates_trial_time": coords,
                            "support_rule": "Stage-5 frozen convention: A uses all held-out centers; B excludes frozen lag-invalid centers"}
    a, b = syn_windows["A"], syn_windows["B"]
    a_valid = np.isin(a["trial_id"], syn_split["test"]) & a["lag_valid"].astype(bool)
    b_valid = np.isin(b["trial_id"], syn_split["test"]) & b["lag_valid"].astype(bool)
    b_coords = set(zip(b["trial_id"][b_valid].tolist(), b["time_id"][b_valid].tolist()))
    paired = [(int(t), int(tm)) for t, tm, k in zip(a["trial_id"], a["time_id"], a_valid)
              if k and (int(t), int(tm)) in b_coords]
    lag_support = [(t, tm) for t, tm in paired
                   if all((t, tm + lag) in b_coords for lag in LAGS)]
    syn_support["matched_A_B"] = {"rows": len(paired), "evaluation_index_sha256": _coordinates_hash(paired),
                                  "coordinates_trial_time": [list(x) for x in paired]}
    syn_support["lag_common_support"] = {"rows_per_lag": len(lag_support),
                                         "candidate_lags_bins": list(LAGS),
                                         "evaluation_index_sha256": _coordinates_hash(lag_support),
                                         "coordinates_trial_time": [list(x) for x in lag_support],
                                         "support_rule": "intersection of rows valid for every lag in the existing frozen grid"}
    real_split_path = real_stage1 / "split.json"
    real_split = read_json(real_split_path)
    real_test_ids = sorted(map(int, real_split["test"]))
    require(sha256_file(real_split_path) == real["spec"]["original_frozen_split_sha256"],
            "REAL original test split hash differs from frozen spec")
    real_data_path = real_stage1 / "data.npz"
    with np.load(real_data_path, allow_pickle=False) as npz:
        valid_bins = np.asarray(npz["valid_bins"], dtype=bool).reshape(193, 600)
    radius = 100
    real_center_valid = np.zeros((193, 600), dtype=bool)
    for center in range(radius, 600 - radius):
        real_center_valid[:, center] = np.all(valid_bins[:, center-radius:center+radius+1], axis=1)
    real_coords_by_split: dict[str, list[list[int]]] = {}
    for name in ("train", "validation", "test"):
        ids = sorted(map(int, real_split[name]))
        coords = [[trial, time] for trial in ids for time in range(600) if real_center_valid[trial, time]]
        real_coords_by_split[name] = coords
    real_support = {
        "window_size": 201, "half_window": 100, "time_center_range_inclusive": [100, 499],
        "heldout_trial_ids": real_test_ids,
        "heldout_trial_count": len(real_test_ids),
        "rows_before_filtering": len(real_test_ids) * 600,
        "rows_after_filtering": len(real_coords_by_split["test"]),
        "evaluation_index_sha256": _coordinates_hash(real_coords_by_split["test"]),
        "coordinates_trial_time": real_coords_by_split["test"],
        "per_split_rows_after_filtering": {k: len(v) for k, v in real_coords_by_split.items()},
        "per_split_index_sha256": {k: _coordinates_hash(v) for k, v in real_coords_by_split.items()},
        "support_rule": "test trials only; centered W201 windows; require all 201 source bins valid; same support for TOTAL65/A_PROXIMAL/B_DISTAL",
    }
    require(real_support["rows_after_filtering"] == len(real_test_ids) * 400,
            "REAL W201 primary support is not 400 centers per held-out trial")
    support = {
        "support_version": "neurobridge_final_evaluation_support_v1",
        "final_real_spec_sha256": real["spec_sha256"],
        "synthetic_split_sha256": sha256_file(syn_stage2 / "split.json"),
        "synthetic": syn_support,
        "real": real_support,
        "real_window_size": 201, "embedding_scope": "all_centered_rows_with_valid_mask",
        "test_opening_authorized": True,
        "test_support_created_from_split_and_validity_metadata_only": True,
        "support_sha256": "computed-after-serialization",
    }
    support["support_sha256"] = canonical_hash({k: v for k, v in support.items() if k != "support_sha256"})
    return audit, support


def _write_bytes_once(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == encoded, f"refusing to overwrite different immutable manifest: {path}")
        return
    staging = path.parent / f".tmp-{uuid.uuid4().hex[:10]}"
    with staging.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    require(not path.exists(), f"immutable output appeared during publication: {path}")
    staging.rename(path)


def _new_embedding_staging_directory(target: Path) -> Path:
    """Return a short same-parent staging path, independent of the trial ID.

    Final artifact paths retain their descriptive frozen IDs. Keeping temporary
    names short prevents Windows MAX_PATH failures when a trial ID is long,
    while same-parent rename preserves atomic publication.
    """
    return target.parent / f".staging-{uuid.uuid4().hex[:10]}"


def _tag_index_dataset(record: dict[str, Any], dataset: str) -> dict[str, Any]:
    normalized = dataset.lower()
    existing = record.get("dataset")
    require(existing is None or str(existing).lower() == normalized,
            f"embedding index dataset conflicts with export branch: {record.get('trial_id')}")
    return {**record, "dataset": normalized}


def write_manifest_once(path: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    encoded = text.encode("utf-8")
    _write_bytes_once(path, encoded)
    return hashlib.sha256(encoded).hexdigest()


def _paired_preflight_archive_names(output: Path) -> tuple[str, ...]:
    """Return only intact, paired archived audit/support manifests."""
    names = []
    if not output.is_dir():
        return ()
    for audit_path in sorted(output.glob("AUDIT_MANIFEST_PRE_*.json")):
        suffix = audit_path.name.removeprefix("AUDIT_MANIFEST_")
        support_path = output / f"EVALUATION_SUPPORT_MANIFEST_{suffix}"
        if not support_path.is_file():
            continue
        audit_sha = sha256_file(audit_path)
        archived_support = json.loads(support_path.read_text(encoding="utf-8"))
        require(archived_support.get("audit_manifest_sha256") == audit_sha,
                f"archived support/audit parent mismatch: {support_path.name}")
        actual_support_sha = canonical_hash({key: value for key, value in archived_support.items()
                                             if key != "support_sha256"})
        require(archived_support.get("support_sha256") == actual_support_sha,
                f"archived support self-hash mismatch: {support_path.name}")
        names.extend((audit_path.name, support_path.name))
    return tuple(sorted(names))


def _preserved_complete_embedding_files(output: Path, archived_names: tuple[str, ...]) -> set[str]:
    """Allow only complete, hash-valid frozen embedding directories at preflight.

    This is metadata/hash verification only; it never opens NPZ feature arrays.
    It permits code-only audit revisions to retain reusable embeddings while
    still rejecting unknown, partial, or unparented files.
    """
    root = output / "embeddings"
    if not root.is_dir():
        return set()
    support_parents: set[str] = set()
    support_paths = [output / name for name in archived_names
                     if name.startswith("EVALUATION_SUPPORT_MANIFEST_")]
    canonical_support = output / SUPPORT_NAME
    if canonical_support.is_file():
        support_paths.append(canonical_support)
    for support_path in support_paths:
        support = json.loads(support_path.read_text(encoding="utf-8"))
        support_parents.add(sha256_file(support_path))
        support_parents.add(str(support["support_sha256"]))

    complete_files: set[str] = set()
    artifact_dirs: set[Path] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(".staging-" in part for part in path.relative_to(root).parts):
            continue
        artifact_dirs.add(path.parent)
    for folder in sorted(artifact_dirs):
        manifest_path = folder / "manifest.json"
        require(manifest_path.is_file(), f"unmanifested embedding files cannot be preflighted: {folder}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset") is not None:
            dataset = str(manifest["dataset"]).lower()
        elif ("parents" in manifest and
              set(manifest.get("artifact_sha256", {})) ==
              {"embedding_raw.npz", "embedding_unit.npz", "evaluation_metadata.npz"}):
            dataset = "real"
        else:
            dataset = ""
        if dataset == "synthetic":
            payload_names = {"embedding.npz"}
            parent = manifest.get("support_parent_sha256")
            digests = {"embedding.npz": manifest.get("embedding_sha256")}
        elif dataset == "real":
            payload_names = {"embedding_raw.npz", "embedding_unit.npz", "evaluation_metadata.npz"}
            parent = manifest.get("parents", {}).get("evaluation_support_sha256")
            digests = manifest.get("artifact_sha256", {})
        else:
            raise RuntimeError(f"unknown embedding dataset in preflight manifest: {manifest_path}")
        require(parent in support_parents, f"embedding has no preserved support parent: {manifest_path}")
        actual_names = {p.name for p in folder.iterdir() if p.is_file()}
        require(actual_names == payload_names | {"manifest.json"},
                f"embedding directory is incomplete or contains unknown files: {folder}")
        require(set(digests) == payload_names,
                f"embedding payload digest inventory differs: {manifest_path}")
        for name in payload_names:
            artifact = folder / name
            require(sha256_file(artifact) == digests[name],
                    f"embedding payload hash mismatch at preflight: {artifact}")
            complete_files.add(artifact.relative_to(output).as_posix())
        complete_files.add(manifest_path.relative_to(output).as_posix())
    return complete_files


def _verify_preflight_embedding_index(path: Path) -> None:
    index = pd.read_csv(path)
    required = {"dataset", "trial_id", "architecture", "objective", "population", "seed",
                "embedding_path", "manifest_path", "embedding_sha256", "manifest_sha256"}
    require(required <= set(index.columns), f"embedding index lacks required identity/hash columns: {path}")
    require(len(index) == 122 and index["trial_id"].notna().all(),
            f"preflight embedding index is not the frozen 122-row inventory: {path}")
    counts = index["dataset"].astype(str).str.lower().value_counts().to_dict()
    require(counts == {"synthetic": 50, "real": 72},
            f"preflight embedding index dataset inventory differs: {counts}")


def run_audit(project: Path) -> dict[str, Any]:
    output = project / CORE_REL
    archived_names = _paired_preflight_archive_names(output)
    if output.exists() and not (output / AUDIT_NAME).is_file():
        existing_files = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
        complete_embedding_files = _preserved_complete_embedding_files(output, archived_names)
        allowed_preflight_files = {
            "AUDIT_ISSUES_AND_FIXES.md", "EMBEDDING_INDEX_PRE_DATASET_FIX.csv",
            *archived_names, *complete_embedding_files,
        }
        if (output / "EMBEDDING_INDEX.csv").is_file():
            _verify_preflight_embedding_index(output / "EMBEDDING_INDEX.csv")
            allowed_preflight_files.add("EMBEDDING_INDEX.csv")
        preserved_staging_files = {
            rel for rel in existing_files
            if "embeddings" in Path(rel).parts and (
                any(".staging-" in part for part in Path(rel).parts) or
                Path(rel).name in {"embedding.npz", "manifest.json"}
            )
        }
        require(existing_files <= (allowed_preflight_files | preserved_staging_files),
                f"core metrics directory contains unmanifested outputs; preserve it: {output}")
    fresh, support = _audit_manifest(project)
    fresh["superseded_preflight_manifests"] = {
        name: sha256_file(output / name) for name in archived_names
    }
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / AUDIT_NAME
    if audit_path.exists():
        audit = read_json(audit_path)
        require(audit.get("source_artifact_hashes") == fresh["source_artifact_hashes"] and
                audit.get("evaluation_driver_sha256") == fresh["evaluation_driver_sha256"],
                "existing audit source/code hashes differ; preserving existing branch")
    else:
        audit = fresh
    audit_sha = write_manifest_once(audit_path, audit)
    support["audit_manifest_sha256"] = audit_sha
    support["support_sha256"] = "computed-after-serialization"
    support["support_sha256"] = canonical_hash({k: v for k, v in support.items() if k != "support_sha256"})
    support_sha = write_manifest_once(output / SUPPORT_NAME, support)
    return {"audit_manifest": str(output / AUDIT_NAME), "audit_sha256": audit_sha,
            "support_manifest": str(output / SUPPORT_NAME), "support_sha256": support_sha,
            "real_slots_verified": len(audit["real"]["slots"]),
            "synthetic_slots_verified": len(audit["synthetic"]["slots"]),
            "test_feature_values_loaded": False, "training_performed": False}


def verify_frozen_manifests(project: Path) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    root = project / CORE_REL
    audit_path, support_path = root / AUDIT_NAME, root / SUPPORT_NAME
    require(audit_path.is_file() and support_path.is_file(), "run --audit before core evaluation")
    audit, support = read_json(audit_path), read_json(support_path)
    support_core = {k: v for k, v in support.items() if k != "support_sha256"}
    require(canonical_hash(support_core) == support["support_sha256"], "support-manifest checksum mismatch")
    require(support["audit_manifest_sha256"] == sha256_file(audit_path), "audit-manifest parent hash mismatch")
    # Re-run source checks and ensure every parent hash remains byte-identical.
    fresh_audit, fresh_support = _audit_manifest(project)
    require(fresh_audit["source_artifact_hashes"] == audit["source_artifact_hashes"],
            "one or more frozen input/source artifacts changed after audit")
    require(fresh_audit["evaluation_driver_sha256"] == audit["evaluation_driver_sha256"],
            "evaluation driver changed after audit")
    expected_support = {**fresh_support, "audit_manifest_sha256": sha256_file(audit_path)}
    expected_support["support_sha256"] = canonical_hash(
        {k: v for k, v in expected_support.items() if k != "support_sha256"})
    require(expected_support == support,
            "saved evaluation coordinates/support differ from frozen source split and masks")
    return root, support_path, audit, support


def _save_csv_once(path: Path, frame: pd.DataFrame) -> None:
    encoded = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if path.exists():
        require(path.read_bytes() == encoded, f"refusing to overwrite different core output: {path}")
    else:
        _write_bytes_once(path, encoded)


def _load_synthetic_records(project: Path) -> list[dict[str, Any]]:
    records = []
    for winner_path in sorted((project / SYN_STUDY / "winners").glob("*.json")):
        manifest = read_json(winner_path)
        expected_candidate = SYN_WINNERS[(manifest["architecture"], manifest["objective"])]
        require(int(manifest["selected_candidate_index"]) == expected_candidate,
                f"Synthetic winner candidate differs from frozen selection map: {winner_path.name}")
        for selected in manifest["selected_checkpoints"]:
            cp = Path(selected["checkpoint_path"])
            records.append({**selected, "architecture": manifest["architecture"],
                            "objective": manifest["objective"], "trial_id": read_json(cp.parent / "trial_record.json")["trial_id"],
                            "candidate_index": expected_candidate,
                            "checkpoint_path": str(cp)})
    return records


def _attach_frozen_fit_status(item: dict[str, Any], frozen_slots: dict[str, Any]) -> dict[str, Any]:
    require(item["trial_id"] in frozen_slots,
            f"Synthetic trial is absent from frozen audit status table: {item['trial_id']}")
    status = frozen_slots[item["trial_id"]]
    return {**item, "fit_status": status["fit_status"],
            "near_collapse": bool(status["near_collapse"])}


def _neural_synthetic_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items
            if item["dataset"] == "Synthetic" and item["architecture"] != "pca"]


def _support_definition_hash(support: dict[str, Any]) -> str:
    return canonical_hash({key: value for key, value in support.items()
                           if key not in {"audit_manifest_sha256", "support_sha256"}})


def _compatible_support_parents(output: Path, support: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Return canonical support hashes and file hashes for exact archived equivalents."""
    support_hashes = {str(support["support_sha256"])}
    support_file_hashes = {sha256_file(output / SUPPORT_NAME)}
    target_definition = _support_definition_hash(support)
    for support_path in sorted(output.glob("EVALUATION_SUPPORT_MANIFEST_PRE_*.json")):
        suffix = support_path.name.removeprefix("EVALUATION_SUPPORT_MANIFEST_")
        audit_path = output / f"AUDIT_MANIFEST_{suffix}"
        if not audit_path.is_file():
            continue
        archived = read_json(support_path)
        archived_audit_sha = sha256_file(audit_path)
        require(archived.get("audit_manifest_sha256") == archived_audit_sha,
                f"archived support/audit parent mismatch: {support_path.name}")
        stored_sha = archived.get("support_sha256")
        actual_sha = canonical_hash({key: value for key, value in archived.items()
                                     if key != "support_sha256"})
        require(stored_sha == actual_sha,
                f"archived support self-hash mismatch: {support_path.name}")
        if _support_definition_hash(archived) == target_definition:
            support_hashes.add(str(stored_sha))
            support_file_hashes.add(sha256_file(support_path))
    return support_hashes, support_file_hashes


def export_synthetic(project: Path, output: Path, support: dict[str, Any]) -> list[dict[str, Any]]:
    windows_root = project / SYN_RUN / "stage02_windows"
    split = read_json(windows_root / "split.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    written = []
    compatible_support_hashes, _ = _compatible_support_parents(output, support)
    for item in _load_synthetic_records(project):
        record_path = Path(item["checkpoint_path"]).parent / "trial_record.json"
        fit_record = read_json(record_path)
        config = fit_record["config"]
        model_cfg = _frozen_config({"config": config, "trial_id": item["trial_id"]})
        window_path = windows_root / f"windows_{item['population']}.npz"
        with np.load(window_path, allow_pickle=False) as source:
            windows = {key: source[key] for key in source.files}
        folder = output / "embeddings/synthetic" / item["trial_id"]
        source_window_sha = sha256_file(window_path)
        source_split_sha = sha256_file(windows_root / "split.json")
        checkpoint_sha = sha256_file(Path(item["checkpoint_path"]))
        if folder.exists():
            embedding_path, manifest_path = folder / "embedding.npz", folder / "manifest.json"
            require(embedding_path.is_file() and manifest_path.is_file(),
                    f"partial Synthetic embedding artifact preserved: {folder}")
            prior = read_json(manifest_path)
            expected = {"dataset": "synthetic", "trial_id": item["trial_id"],
                        "architecture": item["architecture"], "objective": item["objective"],
                        "population": item["population"], "seed": int(item["seed"]),
                        "candidate_index": int(item["candidate_index"]),
                        "checkpoint_sha256": checkpoint_sha, "config_sha256": item["config_sha256"],
                        "source_windows_sha256": source_window_sha,
                        "source_split_sha256": source_split_sha,
                        "embedding_shape": [len(windows["trial_id"]), 3],
                        "raw_unit_relation": "unit = raw / max(L2_norm, 1e-12)",
                        "no_fit_or_optimizer_steps": True}
            require(all(prior.get(key) == value for key, value in expected.items()) and
                    prior.get("support_parent_sha256") in compatible_support_hashes and
                    sha256_file(embedding_path) == prior.get("embedding_sha256"),
                    f"existing Synthetic embedding has different lineage/content: {folder}")
            _synthetic_embedding({"embedding_path": str(embedding_path), "trial_id": item["trial_id"]})
            written.append(_tag_index_dataset(
                {**item, "embedding_path": str(embedding_path),
                 "manifest_path": str(manifest_path),
                 "embedding_sha256": sha256_file(embedding_path),
                 "manifest_sha256": sha256_file(manifest_path),
                 "fit_status": "ELIGIBLE", "near_collapse": False}, "synthetic"))
            print(f"Synthetic embedding reused {len(written)}/48 {item['trial_id']}", flush=True)
            continue
        model = synth._make_model(item["architecture"], windows["X_windows"].shape[-1], model_cfg,
                                  normalize=False).to(device).eval()
        checkpoint = torch.load(item["checkpoint_path"], map_location="cpu", weights_only=True)
        require(checkpoint["trial_id"] == item["trial_id"], "Synthetic checkpoint trial ID mismatch")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        raw_parts = []
        with torch.inference_mode():
            for start in range(0, len(windows["X_windows"]), 1024):
                x = torch.as_tensor(windows["X_windows"][start:start+1024], dtype=torch.float32, device=device)
                raw_parts.append(model(x).detach().cpu().numpy().astype(np.float32))
        raw = np.concatenate(raw_parts)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        unit = raw / np.maximum(norms, 1e-12)
        require(raw.shape == unit.shape == (len(windows["trial_id"]), 3) and
                np.isfinite(raw).all() and np.isfinite(unit).all(), "Synthetic N x 3 embedding invariant fails")
        trial_id = windows["trial_id"].astype(np.int64)
        split_name = np.empty(len(trial_id), dtype="U10")
        for name, ids in split.items():
            split_name[np.isin(trial_id, ids)] = name
        payload = {"embedding_raw": raw, "embedding_unit": unit, "trial_id": trial_id,
                   "time_id": windows["time_id"].astype(np.int64),
                   "global_time_id": windows["global_time_id"].astype(np.int64),
                   "valid_mask": windows["lag_valid"].astype(bool), "split": split_name,
                   "labels": windows["labels"].astype(np.int64), "progress": windows["progress"].astype(np.float32)}
        folder.parent.mkdir(parents=True, exist_ok=True)
        staging = _new_embedding_staging_directory(folder)
        staging.mkdir(exist_ok=False)
        staged_embedding, staged_manifest = staging / "embedding.npz", staging / "manifest.json"
        with staged_embedding.open("xb") as handle:
            np.savez_compressed(handle, **payload)
        manifest = {"dataset": "synthetic", "trial_id": item["trial_id"],
                    "architecture": item["architecture"], "objective": item["objective"],
                    "population": item["population"], "seed": int(item["seed"]),
                    "candidate_index": int(item["candidate_index"]),
                    "checkpoint_sha256": checkpoint_sha,
                    "config_sha256": item["config_sha256"],
                    "source_windows_sha256": source_window_sha,
                    "source_split_sha256": source_split_sha,
                    "embedding_sha256": sha256_file(staged_embedding),
                    "embedding_shape": [len(raw), 3], "raw_unit_relation": "unit = raw / max(L2_norm, 1e-12)",
                    "device": str(device), "no_fit_or_optimizer_steps": True,
                    "support_parent_sha256": support["support_sha256"]}
        write_manifest_once(staged_manifest, manifest)
        require(not folder.exists(), f"Synthetic embedding target appeared during staging: {folder}")
        staging.rename(folder)
        embedding_path, manifest_path = folder / "embedding.npz", folder / "manifest.json"
        written.append(_tag_index_dataset(
            {**item, "embedding_path": str(embedding_path), "manifest_path": str(manifest_path),
             "embedding_sha256": sha256_file(embedding_path),
             "manifest_sha256": sha256_file(manifest_path), "fit_status": "ELIGIBLE",
             "near_collapse": False}, "synthetic"))
        print(f"Synthetic embedding {len(written)}/48 {item['trial_id']}", flush=True)
    # Reuse the already existing canonical train-fitted PCA baseline if its
    # row metadata exactly matches this frozen cache.
    pca_dir = project / SYN_RUN / "stage04_embeddings/held_out"
    for pop in POPS_SYN:
        source_path = pca_dir / f"{pop}_pca_none.npz"
        model_path = project / SYN_RUN / f"stage03_models/held_out/{pop}_pca_none/pca.joblib"
        with np.load(source_path, allow_pickle=False) as source:
            pca = {key: source[key] for key in source.files}
        with np.load(windows_root / f"windows_{pop}.npz", allow_pickle=False) as source:
            for key in ("trial_id", "time_id", "global_time_id", "labels", "progress", "lag_valid"):
                require(np.array_equal(pca[key], source[key]), f"canonical PCA metadata mismatch: {pop}/{key}")
        folder = output / "embeddings/synthetic" / f"pca_none_{pop}"
        if folder.exists():
            target_path, pca_manifest_path = folder / "embedding.npz", folder / "manifest.json"
            require(target_path.is_file() and pca_manifest_path.is_file(),
                    f"partial PCA embedding artifact preserved: {folder}")
            prior = read_json(pca_manifest_path)
            require(prior["source_embedding_sha256"] == sha256_file(source_path) and
                    prior["pca_checkpoint_sha256"] == sha256_file(model_path) and
                    prior["support_parent_sha256"] in compatible_support_hashes and
                    sha256_file(target_path) == prior["embedding_sha256"],
                    f"existing PCA embedding lineage/content mismatch: {folder}")
            _synthetic_embedding({"embedding_path": str(target_path), "trial_id": f"pca_none_{pop}"})
            written.append({"dataset": "synthetic", "architecture": "pca", "objective": "none",
                            "population": pop, "seed": "deterministic", "trial_id": f"pca_none_{pop}",
                            "embedding_path": str(target_path), "manifest_path": str(pca_manifest_path),
                            "embedding_sha256": sha256_file(target_path),
                            "manifest_sha256": sha256_file(pca_manifest_path),
                            "fit_status": "DETERMINISTIC_REFERENCE", "near_collapse": False})
            continue
        folder.parent.mkdir(parents=True, exist_ok=True)
        staging = _new_embedding_staging_directory(folder)
        staging.mkdir(exist_ok=False)
        target_path = staging / "embedding.npz"
        payload = {**pca, "valid_mask": pca["lag_valid"].astype(bool)}
        split_name = np.empty(len(pca["trial_id"]), dtype="U10")
        for name, ids in split.items():
            split_name[np.isin(pca["trial_id"], ids)] = name
        payload["split"] = split_name
        with target_path.open("xb") as handle:
            np.savez_compressed(handle, **payload)
        manifest = {"dataset": "synthetic", "trial_id": f"pca_none_{pop}",
                    "architecture": "pca", "objective": "none", "population": pop,
                    "seed": "deterministic", "source_embedding_sha256": sha256_file(source_path),
                    "pca_checkpoint_sha256": sha256_file(model_path),
                    "embedding_sha256": sha256_file(target_path), "embedding_shape": list(pca["embedding_raw"].shape),
                    "fitted_in_this_task": False, "support_parent_sha256": support["support_sha256"]}
        pca_manifest_path = staging / "manifest.json"
        write_manifest_once(pca_manifest_path, manifest)
        require(not folder.exists(), f"PCA embedding target appeared during staging: {folder}")
        staging.rename(folder)
        target_path = folder / "embedding.npz"
        pca_manifest_path = folder / "manifest.json"
        written.append({"dataset": "synthetic", "architecture": "pca", "objective": "none",
                        "population": pop, "seed": "deterministic", "trial_id": f"pca_none_{pop}",
                        "embedding_path": str(target_path), "manifest_path": str(folder / "manifest.json"),
                        "embedding_sha256": sha256_file(target_path),
                        "manifest_sha256": sha256_file(pca_manifest_path), "fit_status": "DETERMINISTIC_REFERENCE",
                        "near_collapse": False})
    return written


def export_real(project: Path, output: Path, support_path: Path) -> list[dict[str, Any]]:
    spec, partition, _ = real_export.frozen_inputs(project)
    arrays, split, parents = real_export.stage1_arrays(project, spec)
    support = read_json(support_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    records = []
    _, compatible_support_file_hashes = _compatible_support_parents(output, support)
    for i, slot in enumerate(spec["final_slots"], 1):
        out = output / "embeddings" / slot["trial_id"]
        if out.exists():
            _verify_reusable_real_export(project, slot, partition, parents, support_path,
                                         out, compatible_support_file_hashes, real_export,
                                         len(arrays["target_by_trial"]) * real_export.TRIAL_LENGTH)
        else:
            out = real_export.export_slot(project, slot, spec, partition, arrays, split, parents,
                                          support_path, batch_size=256, device=device,
                                          output_root=output / "embeddings")
        manifest = read_json(out / "manifest.json")
        records.append({"dataset": "real", "trial_id": slot["trial_id"],
                        "architecture": slot["architecture"], "objective": slot["objective"],
                        "population": slot["population"], "seed": int(slot["training_seed_root"]),
                        "embedding_path": str(out), "manifest_path": str(out / "manifest.json"),
                        "embedding_sha256": manifest["artifact_sha256"]["embedding_raw.npz"],
                        "manifest_sha256": sha256_file(out / "manifest.json"),
                        "fit_status": manifest["fit_status"],
                        "near_collapse": bool(manifest["near_collapse"])})
        print(f"Real inference export {i}/72 {slot['trial_id']}", flush=True)
    return records


def _real_export_parent_matches(prior_parent: dict[str, Any], expected_parent: dict[str, Any],
                                compatible_support_file_hashes: set[str]) -> bool:
    prior_support = prior_parent.get("evaluation_support_sha256")
    if prior_support not in compatible_support_file_hashes:
        return False
    expected = dict(expected_parent)
    expected["evaluation_support_sha256"] = prior_support
    return prior_parent == expected


def _verify_reusable_real_export(project: Path, slot: dict[str, Any], partition: dict[str, Any],
                                 parent_hashes: dict[str, Any], support_path: Path,
                                 output_dir: Path, compatible_support_file_hashes: set[str],
                                 real_export: Any, expected_rows: int) -> None:
    checkpoint, record, result, _ = real_export.verified_checkpoint(project, slot, partition)
    indices = real_export.channel_indices(partition, slot["population"])
    expected_parent = {
        "trial_id": slot["trial_id"], "population": slot["population"],
        "architecture": slot["architecture"], "objective": slot["objective"],
        "seed": slot["training_seed_root"], "window_size": real_export.WINDOW,
        "channel_indices": indices,
        "nlb_unit_ids": [partition["feature_order_nlb_unit_ids"][i] for i in indices],
        "partition_sha256": sha256_file(project / real_export.PARTITION),
        "checkpoint_sha256": sha256_file(checkpoint), "checkpoint_path": str(checkpoint),
        "config_sha256": record["config_sha256"],
        "trial_record_sha256": sha256_file(checkpoint.parent / "trial_record.json"),
        "fit_result_sha256": sha256_file(checkpoint.parent / "result.json"),
        "final_spec_sha256": sha256_file(project / real_export.SPEC),
        "evaluation_support_sha256": sha256_file(support_path),
        "exporter_source_sha256": sha256_file(Path(real_export.__file__)),
        **parent_hashes,
    }
    manifest_path = output_dir / "manifest.json"
    require(manifest_path.is_file(), f"partial Real export preserved: {output_dir}")
    manifest = read_json(manifest_path)
    require(_real_export_parent_matches(manifest.get("parents", {}), expected_parent,
                                        compatible_support_file_hashes),
            f"existing Real export lineage differs from frozen slot/support: {output_dir}")
    require(manifest.get("rows") == expected_rows and manifest.get("embedding_shape") == [expected_rows, 3] and
            manifest.get("checkpoint_selected_update") == result["selected_update"] and
            manifest.get("fit_status") == result["status"] and
            manifest.get("near_collapse") == bool(result.get("geometry", {}).get("near_collapse", False)),
            f"existing Real export shape/fit status differs from frozen result: {output_dir}")
    expected_names = {"embedding_raw.npz", "embedding_unit.npz", "evaluation_metadata.npz"}
    require(set(manifest.get("artifact_sha256", {})) == expected_names,
            f"existing Real export payload inventory differs: {output_dir}")
    for name, digest in manifest["artifact_sha256"].items():
        artifact = output_dir / name
        require(artifact.is_file() and sha256_file(artifact) == digest,
                f"existing Real export payload hash differs: {artifact}")


def _real_embedding(item: dict[str, Any]) -> dict[str, np.ndarray]:
    folder = Path(item["embedding_path"])
    out = {}
    for filename, keys in (("embedding_raw.npz", ("embedding_raw",)),
                           ("embedding_unit.npz", ("embedding_unit",)),
                           ("evaluation_metadata.npz", ("trial_id", "time_id", "global_time_id", "valid_mask", "split", "target", "progress", "position", "velocity"))):
        with np.load(folder / filename, allow_pickle=False) as npz:
            out.update({key: npz[key] for key in keys})
    n = len(out["embedding_raw"])
    require(out["embedding_raw"].shape == out["embedding_unit"].shape == (n, 3),
            f"Real embedding N x 3 invariant fails: {item['trial_id']}")
    for key in ("trial_id", "time_id", "global_time_id", "valid_mask", "split", "target", "progress",
                "position", "velocity"):
        require(len(out[key]) == n, f"Real metadata cardinality mismatch {key}: {item['trial_id']}")
    require(np.isfinite(out["embedding_raw"]).all() and np.isfinite(out["embedding_unit"]).all(),
            f"Real embeddings contain non-finite values: {item['trial_id']}")
    return out


def _synthetic_embedding(item: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(item["embedding_path"], allow_pickle=False) as npz:
        out = {key: npz[key] for key in npz.files}
    n = len(out["embedding_raw"])
    require(out["embedding_raw"].shape == out["embedding_unit"].shape == (n, 3),
            f"Synthetic embedding N x 3 invariant fails: {item['trial_id']}")
    for key in ("trial_id", "time_id", "global_time_id", "valid_mask", "split", "labels", "progress"):
        require(len(out[key]) == n, f"Synthetic metadata cardinality mismatch {key}: {item['trial_id']}")
    return out


def _metric_row(base: dict[str, Any], metric: str, value: float, **extra: Any) -> dict[str, Any]:
    return {**base, "metric": metric, "value": float(value) if np.isfinite(value) else np.nan, **extra}


def _probe_metrics(embedding: np.ndarray, labels: np.ndarray,
                   masks: dict[str, np.ndarray], seed: int,
                   progress: np.ndarray | None = None,
                   behavior: dict[str, np.ndarray] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train, validation, test = (masks[name] for name in ("train", "validation", "test"))
    require(len(np.unique(labels[train])) == 8, "multinomial probe train split must contain eight classes")
    candidates = []
    for c_value in PROBE_C_GRID:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, solver="lbfgs",
                                                                    max_iter=1000, random_state=seed))
        model.fit(embedding[train], labels[train])
        score = balanced_accuracy_score(labels[validation], model.predict(embedding[validation]))
        candidates.append((float(score), model, c_value))
    _, classifier, selected_c = max(candidates, key=lambda x: x[0])
    prediction = classifier.predict(embedding[test])
    rows = [_metric_row({}, "direction_balanced_accuracy" if behavior is not None else "condition_balanced_accuracy",
                        balanced_accuracy_score(labels[test], prediction), selected_C=selected_c,
                        validation_metric="balanced_accuracy", selection_split="validation")]
    selected_alpha = None
    if progress is not None:
        regression = []
        for alpha in RIDGE_ALPHA_GRID:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(embedding[train], progress[train])
            score = r2_score(progress[validation], model.predict(embedding[validation]))
            regression.append((float(score), model, alpha))
        _, progress_model, selected_alpha = max(regression, key=lambda x: x[0])
        progress_pred = progress_model.predict(embedding[test])
        rows.extend([_metric_row({}, "progress_r2", r2_score(progress[test], progress_pred),
                                 selected_alpha=selected_alpha, selection_split="validation"),
                     _metric_row({}, "progress_mae", mean_absolute_error(progress[test], progress_pred),
                                 selected_alpha=selected_alpha, selection_split="validation")])
    selected_behavior = {}
    if behavior is not None:
        for name, target in behavior.items():
            reg_candidates = []
            for alpha in RIDGE_ALPHA_GRID:
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(embedding[train], target[train])
                pred = model.predict(embedding[validation])
                score = r2_score(target[validation], pred, multioutput="variance_weighted")
                reg_candidates.append((float(score), model, alpha))
            _, model, alpha = max(reg_candidates, key=lambda x: x[0])
            pred = model.predict(embedding[test])
            rows.extend([_metric_row({}, f"{name}_r2", r2_score(target[test], pred, multioutput="variance_weighted"),
                                     selected_alpha=alpha, selection_split="validation"),
                         _metric_row({}, f"{name}_mae", mean_absolute_error(target[test], pred),
                                     selected_alpha=alpha, selection_split="validation")])
            selected_behavior[name] = alpha
    return rows, {"selected_C": selected_c, "condition_grid": list(PROBE_C_GRID),
                  "selected_progress_alpha": selected_alpha,
                  "behavior_alphas": selected_behavior, "selection_split": "validation",
                  "final_evaluation_split": "test"}


def _sample_pair_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic pair sample shared across runs/representations of size n."""
    n_pairs = n * (n - 1) // 2
    if n_pairs <= PAIR_COUNT:
        return np.triu_indices(n, k=1)
    rng = np.random.default_rng(PAIR_SAMPLE_SEED)
    left = rng.integers(0, n, size=PAIR_COUNT)
    right = rng.integers(0, n - 1, size=PAIR_COUNT)
    right += right >= left
    return left, right


def _diagnostics(embedding: np.ndarray) -> dict[str, float]:
    x = np.asarray(embedding, dtype=np.float64)
    n, d = x.shape
    centered = x - x.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(n - 1, 1)
    eig = np.maximum(np.linalg.eigvalsh(cov)[::-1], 0.0)
    norms = np.linalg.norm(x, axis=1)
    left, right = _sample_pair_indices(n)
    delta = x[left] - x[right]
    euclid = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    a, b = x[left], x[right]
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = np.divide(np.einsum("ij,ij->i", a, b), denom,
                       out=np.full(len(denom), np.nan), where=denom > 1e-12)
    result = {"participation_ratio": _participation_ratio(x)}
    result.update({f"covariance_eigenvalue_{j+1}": float(eig[j]) if j < d else 0.0 for j in range(d)})
    result.update({f"coordinate_variance_{j+1}": float(cov[j, j]) for j in range(d)})
    for label, values in (("embedding_norm", norms), ("euclidean_distance", euclid),
                          ("cosine_similarity", cosine[np.isfinite(cosine)])):
        result[f"{label}_mean"] = float(np.mean(values)) if len(values) else np.nan
        result[f"{label}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        for q, name in ((0.05, "q05"), (0.5, "median"), (0.95, "q95")):
            result[f"{label}_{name}"] = float(np.quantile(values, q)) if len(values) else np.nan
    result["pairwise_samples"] = float(len(euclid))
    result["cosine_undefined_pairs"] = float(np.sum(~np.isfinite(cosine)))
    return result


def _summarize_seeds(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    neural = frame[frame["seed"].astype(str).isin([str(s) for s in SEEDS])]
    keys = ["dataset", "architecture", "objective", "population", "representation", "category", "reference", "metric", "lag_bins"]
    result = []
    for key, group in neural.groupby(keys, dropna=False, sort=True):
        values = pd.to_numeric(group["value"], errors="coerce").dropna().to_numpy(dtype=float)
        near = sorted({int(seed) for seed, flag in zip(group["seed"], group["near_collapse"])
                       if str(flag).lower() == "true"})
        result.append(dict(zip(keys, key)) | {
            "n_expected_seeds": 3, "n_valid_seeds": len(values),
            "mean": float(np.mean(values)) if len(values) else np.nan,
            "median": float(np.median(values)) if len(values) else np.nan,
            "sd_sample": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "min": float(np.min(values)) if len(values) else np.nan,
            "max": float(np.max(values)) if len(values) else np.nan,
            "near_collapse_seeds": ";".join(map(str, near)),
        })
    return pd.DataFrame(result)


def _style_save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_core(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    expected = {"synthetic_geometry_fidelity", "synthetic_accessibility", "synthetic_lag_recovery",
                "synthetic_participation_ratio", "real_participation_ratio_collapse",
                "real_ab_consistency", "real_direction_accessibility"}
    if fig_dir.exists():
        manifest_path = fig_dir / "figure_manifest.json"
        require(manifest_path.is_file(),
                f"unmanifested presentation figures exist; preserve them: {fig_dir}")
        manifest = read_json(manifest_path)
        expected_hashes = manifest.get("figure_sha256", {})
        require(manifest.get("figure_stems") == sorted(expected) and
                set(expected_hashes) == {f"{stem}.{ext}" for stem in expected for ext in ("png", "pdf")},
                "presentation figure manifest inventory differs")
        for name, digest in expected_hashes.items():
            path = fig_dir / name
            require(path.is_file() and sha256_file(path) == digest,
                    f"presentation figure hash mismatch: {path}")
        require(manifest.get("metrics_parent_sha256") == sha256_file(fig_dir.parent / "CORE_METRICS_LONG.csv"),
                "presentation figures were rendered from different core metrics")
        return
    render_dir = fig_dir.with_name(f"{fig_dir.name}.staging-{uuid.uuid4().hex[:10]}")
    render_dir.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(rows)
    syn_geo = frame[(frame.dataset == "Synthetic") & (frame.category == "geometry") &
                    (frame.reference == "Z") & frame.metric.isin(["procrustes_r2", "rsa_spearman", "rsa_pearson", "linear_cka"])]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for ax, metric in zip(axes.flat, ["procrustes_r2", "rsa_spearman", "rsa_pearson", "linear_cka"]):
        part = syn_geo[syn_geo.metric == metric]
        groups = sorted({(r.architecture, r.objective, r.population, r.representation) for r in part.itertuples()})
        positions = np.arange(len(groups))
        for j, group in enumerate(groups):
            sub = part[(part.architecture == group[0]) & (part.objective == group[1]) &
                       (part.population == group[2]) & (part.representation == group[3])]
            vals = sub.value.to_numpy(float)
            ax.scatter(np.full(len(vals), j) + np.linspace(-0.08, 0.08, len(vals)), vals,
                       s=18, alpha=0.55, color=({"A": "#2166ac", "B": "#b2182b"}[group[2]] if group[2] in {"A", "B"} else "#555555"),
                       marker=("o" if group[3] == "raw" else "s"))
            if len(vals):
                ax.plot(j, np.mean(vals), marker="D", color="black", markersize=4)
        labels = [f"{a}\n{obj[:4]}-{pop}-{rep[0]}" for a, obj, pop, rep in groups]
        ax.set_xticks(positions, labels, rotation=70, ha="right", fontsize=7)
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Synthetic held-out geometry vs matched latent Z (seed points; diamonds = mean)")
    _style_save(fig, render_dir / "synthetic_geometry_fidelity")

    syn_acc = frame[(frame.dataset == "Synthetic") & (frame.category == "accessibility") &
                    frame.metric.isin(["condition_balanced_accuracy", "progress_r2"])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for ax, metric in zip(axes, ("condition_balanced_accuracy", "progress_r2")):
        part = syn_acc[syn_acc.metric == metric]
        groups = sorted({(r.architecture, r.objective, r.population, r.representation) for r in part.itertuples()})
        for j, group in enumerate(groups):
            sub = part[(part.architecture == group[0]) & (part.objective == group[1]) &
                       (part.population == group[2]) & (part.representation == group[3])]
            vals = sub.value.to_numpy(float)
            color = "#2166ac" if group[2] == "A" else "#b2182b"
            ax.scatter(np.full(len(vals), j), vals, color=color, alpha=.6,
                       marker="o" if group[3] == "raw" else "s", s=20)
            if len(vals): ax.plot(j, np.mean(vals), "D", color="black", ms=4)
        ax.set_xticks(range(len(groups)), [f"{a}\n{o[:4]}-{p}-{r[0]}" for a,o,p,r in groups], rotation=70, ha="right", fontsize=7)
        ax.set_title(metric.replace("_", " ")); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Synthetic held-out accessibility (individual seeds; black diamonds = mean)")
    _style_save(fig, render_dir / "synthetic_accessibility")

    lag_path = frame[(frame.dataset == "Synthetic") & (frame.category == "temporal_fidelity")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True, sharey=True, constrained_layout=True)
    for (arch, objective), ax in zip([(a,o) for a in ARCHS for o in OBJECTIVES], axes.flat):
        for rep, color in (("raw", "#2166ac"), ("unit", "#b2182b")):
            subset = lag_path[(lag_path.architecture == arch) & (lag_path.objective == objective) &
                              (lag_path.representation == rep) & (lag_path.metric == "s_lag_r2")]
            curves = []
            for seed in SEEDS:
                cur = subset[subset.seed.astype(str) == str(seed)].sort_values("lag_bins")
                if len(cur):
                    y = cur.value.to_numpy(float); curves.append(y)
                    ax.plot(cur.lag_bins, y, color=color, alpha=.20, linewidth=.75)
            if curves:
                arr = np.stack(curves)
                ax.plot(sorted(subset.lag_bins.unique()), np.nanmean(arr, axis=0), color=color, linewidth=1.8, label=rep)
        ax.axvline(10, color="black", linestyle="--", linewidth=1, label="true +10" if ax is axes.flat[0] else None)
        ax.set_title(f"{arch} / {objective}", fontsize=8); ax.grid(alpha=.2)
    axes[-1, 0].set_xlabel("Lag (bins)"); axes[0, 0].set_ylabel("Procrustes R²")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Synthetic held-out A(t) vs B(t + lag); thin lines=seeds, thick=mean")
    _style_save(fig, render_dir / "synthetic_lag_recovery")

    pr = frame[(frame.category == "representation_diagnostics") & (frame.metric == "participation_ratio")]
    for dataset, name in (("Synthetic", "synthetic_participation_ratio"), ("Real", "real_participation_ratio_collapse")):
        part = pr[pr.dataset == dataset]
        fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
        groups = sorted({(r.architecture, r.objective, r.population, r.representation) for r in part.itertuples()})
        for j, group in enumerate(groups):
            sub = part[(part.architecture == group[0]) & (part.objective == group[1]) &
                       (part.population == group[2]) & (part.representation == group[3])]
            for k, row in enumerate(sub.itertuples()):
                marker = "x" if str(row.near_collapse).lower() == "true" else ("o" if group[3] == "raw" else "s")
                ax.scatter(j + (k - 1) * .045, row.value, color="#b2182b" if "B" in group[2] else "#2166ac",
                           marker=marker, s=28, alpha=.8)
            vals = sub.value.to_numpy(float)
            if len(vals): ax.plot(j, np.nanmean(vals), "D", color="black", ms=4)
        ax.set_xticks(range(len(groups)), [f"{a}\n{o[:4]}-{p}-{r[0]}" for a,o,p,r in groups], rotation=75, ha="right", fontsize=7)
        ax.set_ylabel("Participation ratio"); ax.set_title(f"{dataset}: individual seed points; × = near-collapse; diamond = mean")
        ax.grid(axis="y", alpha=.2)
        _style_save(fig, render_dir / name)

    real_geo = frame[(frame.dataset == "Real") & (frame.population == "A_PROXIMAL_vs_B_DISTAL") &
                     (frame.category == "cross_population_consistency") &
                     frame.metric.isin(["procrustes_r2", "rsa_spearman", "rsa_pearson", "linear_cka"])]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.5), constrained_layout=True)
    for ax, metric in zip(axes, ("procrustes_r2", "rsa_spearman", "rsa_pearson", "linear_cka")):
        p = real_geo[real_geo.metric == metric]
        groups = sorted({(r.architecture, r.objective, r.representation) for r in p.itertuples()})
        for j,g in enumerate(groups):
            sub=p[(p.architecture==g[0])&(p.objective==g[1])&(p.representation==g[2])]
            vals=sub.value.to_numpy(float)
            for k, row in enumerate(sub.itertuples()):
                collapsed = str(row.near_collapse).lower() == "true"
                ax.scatter(j + (k - 1) * .045, row.value,
                           marker="x" if collapsed else ("o" if g[2] == "raw" else "s"),
                           alpha=.8, s=28, color="#b2182b" if g[2] == "unit" else "#2166ac")
            if len(vals): ax.plot(j,np.mean(vals),"D",color="black",ms=4)
        ax.set_xticks(range(len(groups)),[f"{a}\n{o[:4]}-{r[0]}" for a,o,r in groups],rotation=70,ha="right",fontsize=7)
        ax.set_title(metric.replace("_"," ")); ax.grid(axis="y",alpha=.2)
    fig.suptitle("REAL held-out A_PROXIMAL vs B_DISTAL consistency; × status also retained in CSV")
    _style_save(fig, render_dir / "real_ab_consistency")

    real_ba = frame[(frame.dataset=="Real")&(frame.category=="accessibility")&(frame.metric=="direction_balanced_accuracy")]
    fig, ax = plt.subplots(figsize=(14,5.5),constrained_layout=True)
    groups=sorted({(r.architecture,r.objective,r.population,r.representation) for r in real_ba.itertuples()})
    for j,g in enumerate(groups):
        sub=real_ba[(real_ba.architecture==g[0])&(real_ba.objective==g[1])&(real_ba.population==g[2])&(real_ba.representation==g[3])]
        for row in sub.itertuples():
            marker="x" if str(row.near_collapse).lower()=="true" else ("o" if g[3]=="raw" else "s")
            ax.scatter(j,row.value,marker=marker,s=27,color="#b2182b" if g[2]=="B_DISTAL" else ("#4d9221" if g[2]=="TOTAL65" else "#2166ac"),alpha=.8)
        if len(sub): ax.plot(j,sub.value.mean(),"D",color="black",ms=4)
    ax.set_xticks(range(len(groups)),[f"{a}\n{o[:4]}-{p}-{r[0]}" for a,o,p,r in groups],rotation=75,ha="right",fontsize=7)
    ax.set_ylabel("Held-out balanced accuracy"); ax.set_title("REAL eight-direction decoding; × = near-collapse; diamond = mean")
    ax.grid(axis="y",alpha=.2); _style_save(fig,render_dir/"real_direction_accessibility")


    png_stems = {p.stem for p in render_dir.glob("*.png")}
    pdf_stems = {p.stem for p in render_dir.glob("*.pdf")}
    require(png_stems == expected and pdf_stems == expected,
            f"incomplete core figure set: png={png_stems}, pdf={pdf_stems}")
    figure_files = sorted(list(render_dir.glob("*.png")) + list(render_dir.glob("*.pdf")))
    figure_manifest = {"figure_stems": sorted(expected),
                       "metrics_parent_sha256": sha256_file(fig_dir.parent / "CORE_METRICS_LONG.csv"),
                       "figure_sha256": {path.name: sha256_file(path) for path in figure_files}}
    write_manifest_once(render_dir / "figure_manifest.json", figure_manifest)
    require(not fig_dir.exists(), f"final figure directory appeared during rendering: {fig_dir}")
    render_dir.rename(fig_dir)


def _test_mask(item: dict[str, Any], data: dict[str, np.ndarray], population: str) -> np.ndarray:
    keep = (data["split"].astype(str) == "test") & data["valid_mask"].astype(bool)
    # The canonical Synthetic Stage-5 convention keeps all A held-out centers,
    # while B excludes only its frozen lag-invalid centers. Real is restricted
    # to fully observed W201 centers for every population/split.
    if item["dataset"] == "Synthetic" and population == "A":
        keep = data["split"].astype(str) == "test"
    return keep


def _split_masks(data: dict[str, np.ndarray], dataset: str, population: str) -> dict[str, np.ndarray]:
    masks = {name: (data["split"].astype(str) == name) & data["valid_mask"].astype(bool)
             for name in ("train", "validation", "test")}
    # Frozen Synthetic Stage-5 fits/evaluates A on all centers; B excludes
    # the first lag-invalid centers. Real is always restricted to fully valid
    # centered W201 windows.
    if dataset == "Synthetic" and population == "A":
        masks = {name: data["split"].astype(str) == name for name in masks}
    return masks


def _synthetic_latents(project: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = project / SYN_RUN / "stage01_data/shared_data.npz"
    cfg = read_json(project / SYN_RUN / "stage01_data/config.json")
    with np.load(source, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("M", "eta", "Z_A", "Z_B")}
    require(data["Z_A"].shape == data["Z_B"].shape == data["M"].shape == data["eta"].shape,
            "Synthetic truth array shape mismatch")
    require(data["Z_A"].ndim == 3 and data["Z_A"].shape[-1] == 3,
            "Synthetic latent truth is not trial x time x 3")
    with np.load(source, allow_pickle=False) as archive:
        z_shared = archive["Z_shared"]
    lag = int(cfg["lag_bins"])
    require(np.allclose(z_shared, data["M"] + data["eta"], rtol=1e-6, atol=1e-7),
            "Synthetic ground-truth identity Z=M+eta fails")
    require(np.array_equal(data["Z_A"], z_shared) and
            np.array_equal(data["Z_B"], synth.apply_temporal_lag(z_shared, lag)),
            "Synthetic A/B latent references differ from frozen shared process/+10 shift")
    return data, cfg


def _metric_geometry(embedding: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return evaluate_latent_recovery(embedding, target, max_pairs=PAIR_COUNT)


def _append_core_row(rows: list[dict[str, Any]], base: dict[str, Any],
                     category: str, metric: str, value: float, **extra: Any) -> None:
    rows.append(_metric_row({**base, "category": category}, metric, value, **extra))


def _evaluate_core(project: Path, output: Path, support: dict[str, Any],
                   items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    synthetic_items = _neural_synthetic_items(items)
    real_items = [item for item in items if item["dataset"] == "Real"]
    syn_data, syn_cfg = _synthetic_latents(project)
    syn_split = read_json(project / SYN_RUN / "stage02_windows/split.json")
    syn_by_key: dict[tuple[str, str, int, str], tuple[dict[str, Any], dict[str, np.ndarray]]] = {}

    for item in synthetic_items:
        meta = _synthetic_embedding(item)
        base = {key: item[key] for key in ("dataset", "architecture", "objective", "population", "seed",
                                           "trial_id", "fit_status", "near_collapse")}
        population = item["population"]
        keep_test = _test_mask(item, meta, population)
        expected_coords = support["synthetic"][population]["coordinates_trial_time"]
        observed_coords = np.column_stack((meta["trial_id"][keep_test], meta["time_id"][keep_test])).astype(int).tolist()
        require(observed_coords == expected_coords, f"Synthetic held-out support mismatch: {item['trial_id']}")
        masks = _split_masks(meta, "Synthetic", population)
        for rep in ("raw", "unit"):
            z = meta[f"embedding_{rep}"]
            for name, mask in masks.items():
                require(mask.any(), f"empty Synthetic {name} support for {item['trial_id']}")
            trial = meta["trial_id"][keep_test].astype(int)
            time = meta["time_id"][keep_test].astype(int)
            references = {"Z": syn_data[f"Z_{population}"]}
            if population == "A":
                references.update({"M": syn_data["M"], "eta": syn_data["eta"]})
            else:
                references.update({"M": synth.apply_temporal_lag(syn_data["M"], int(syn_cfg["lag_bins"])),
                                   "eta": synth.apply_temporal_lag(syn_data["eta"], int(syn_cfg["lag_bins"]))})
            for reference, array in references.items():
                target = array[trial, time]
                geom = _metric_geometry(z[keep_test], target)
                for metric, value in geom.items():
                    _append_core_row(rows, {**base, "representation": rep, "reference": reference},
                                     "geometry", metric, value,
                                     reference_role="primary" if reference == "Z" else "complementary")
            diagnostics = _diagnostics(z[keep_test])
            for metric, value in diagnostics.items():
                _append_core_row(rows, {**base, "representation": rep, "reference": "none"},
                                 "representation_diagnostics", metric, value,
                                 diagnostic_split="test", pair_sampling_seed=PAIR_SAMPLE_SEED)
            probe_rows, selected = _probe_metrics(z, meta["labels"], masks=masks, seed=int(item["seed"]),
                                                   progress=meta["progress"])
            for probe in probe_rows:
                metric = probe["metric"]
                category = "accessibility" if metric == "condition_balanced_accuracy" else "accessibility"
                _append_core_row(rows, {**base, "representation": rep, "reference": "none"}, category,
                                 metric, probe["value"], selected_C=probe.get("selected_C"),
                                 selected_alpha=probe.get("selected_alpha"),
                                 selection_split="validation", final_evaluation_split="test")
            selection_rows.append({**base, "representation": rep, **selected})
            syn_by_key[(item["architecture"], item["objective"], int(item["seed"]), population)] = (item, meta)

    # Existing canonical PCA is a deterministic Synthetic reference only.
    for item in [x for x in items if x["dataset"] == "Synthetic" and x["architecture"] == "pca"]:
        meta = _synthetic_embedding(item)
        base = {key: item[key] for key in ("dataset", "architecture", "objective", "population", "seed",
                                           "trial_id", "fit_status", "near_collapse")}
        pop = item["population"]
        keep_test = _test_mask(item, meta, pop)
        masks = _split_masks(meta, "Synthetic", pop)
        trial, time = meta["trial_id"][keep_test].astype(int), meta["time_id"][keep_test].astype(int)
        refs = {"Z": syn_data[f"Z_{pop}"]}
        refs["M"] = syn_data["M"] if pop == "A" else synth.apply_temporal_lag(syn_data["M"], int(syn_cfg["lag_bins"]))
        refs["eta"] = syn_data["eta"] if pop == "A" else synth.apply_temporal_lag(syn_data["eta"], int(syn_cfg["lag_bins"]))
        for rep in ("raw", "unit"):
            z = meta[f"embedding_{rep}"]
            for reference, array in refs.items():
                for metric, value in _metric_geometry(z[keep_test], array[trial, time]).items():
                    _append_core_row(rows, {**base, "representation": rep, "reference": reference},
                                     "geometry", metric, value,
                                     reference_role="primary" if reference == "Z" else "complementary")
            for metric, value in _diagnostics(z[keep_test]).items():
                _append_core_row(rows, {**base, "representation": rep, "reference": "none"},
                                 "representation_diagnostics", metric, value, diagnostic_split="test")
            probe_rows, selected = _probe_metrics(z, meta["labels"], masks=masks, seed=0, progress=meta["progress"])
            for probe in probe_rows:
                _append_core_row(rows, {**base, "representation": rep, "reference": "none"},
                                 "accessibility", probe["metric"], probe["value"],
                                 selected_C=probe.get("selected_C"), selected_alpha=probe.get("selected_alpha"),
                                 selection_split="validation", final_evaluation_split="test")
            selection_rows.append({**base, "representation": rep, **selected})

    # Synthetic A/B temporal fidelity, with the frozen lag-independent support.
    for arch in ARCHS:
        for objective in OBJECTIVES:
            for seed in SEEDS:
                a_item, a = syn_by_key[(arch, objective, seed, "A")]
                b_item, b = syn_by_key[(arch, objective, seed, "B")]
                amask = (a["split"].astype(str) == "test")
                bmask = (b["split"].astype(str) == "test") & b["valid_mask"].astype(bool)
                lag_support = support["synthetic"]["lag_common_support"]["coordinates_trial_time"]
                trial_ids = np.asarray([c[0] for c in lag_support], dtype=np.int64)
                time_ids = np.asarray([c[1] for c in lag_support], dtype=np.int64)
                alook = {(int(t), int(tm)): i for i, (t, tm) in enumerate(zip(a["trial_id"], a["time_id"]))}
                blook = {(int(t), int(tm)): i for i, (t, tm) in enumerate(zip(b["trial_id"], b["time_id"]))}
                ai = np.asarray([alook[(int(t), int(tm))] for t, tm in zip(trial_ids, time_ids)])
                for rep in ("raw", "unit"):
                    ref = a[f"embedding_{rep}"][ai]
                    _, scores, aligned = lagged_alignment_by_trial_time(
                        ref, b[f"embedding_{rep}"][bmask], trial_ids, time_ids,
                        b["trial_id"][bmask], b["time_id"][bmask], LAGS, common_support=True)
                    counts = {int(lag): len(aligned[int(lag)][0]) for lag in LAGS}
                    require(len(set(counts.values())) == 1 and counts[LAGS[0]] == len(lag_support),
                            f"Synthetic lag support varies by lag: {arch}/{objective}/{seed}/{rep}")
                    best = max(scores, key=scores.get)
                    vals = np.asarray([scores[int(lag)] for lag in LAGS])
                    ordered = np.sort(vals[np.isfinite(vals)])
                    margin = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else np.nan
                    ties = int(np.sum(np.isclose(vals, np.nanmax(vals), rtol=0, atol=1e-12)))
                    base = {"dataset": "Synthetic", "architecture": arch, "objective": objective,
                            "population": "A_vs_B", "seed": seed, "representation": rep,
                            "trial_id": f"{a_item['trial_id']}|{b_item['trial_id']}",
                            "fit_status": "ELIGIBLE", "near_collapse": False,
                            "reference": "B(t+lag)_vs_A(t)", "category": "temporal_fidelity"}
                    for lag in LAGS:
                        rows.append({**base, "metric": "s_lag_r2", "value": float(scores[int(lag)]),
                                     "lag_bins": int(lag), "n_comparisons": counts[int(lag)]})
                    for metric, value in (("estimated_lag_bins", best), ("signed_error_bins", best - 10),
                                          ("absolute_error_bins", abs(best - 10)),
                                          ("s_true_plus10", scores[10]), ("s_max", scores[best]),
                                          ("peak_margin", margin), ("boundary_peak", best in {LAGS[0], LAGS[-1]}),
                                          ("peak_identifiable_unique_interior", best not in {LAGS[0], LAGS[-1]} and ties == 1)):
                        rows.append({**base, "metric": metric, "value": value,
                                     "estimated_lag_bins": int(best), "lag_bins": np.nan,
                                     "n_comparisons": counts[LAGS[0]], "peak_ties_within_1e-12": ties})
                    if 10 in LAGS:
                        bi_true = np.asarray([blook[(int(t), int(tm) + 10)]
                                              for t, tm in zip(trial_ids, time_ids)], dtype=np.int64)
                        matched_a, matched_b = a[f"embedding_{rep}"][ai], b[f"embedding_{rep}"][bi_true]
                        corr_metrics = {
                            "procrustes_r2": procrustes_r2(matched_a, matched_b),
                            "rsa_spearman": distance_geometry_correlation(matched_a, matched_b,
                                                                           method="spearman", max_pairs=PAIR_COUNT),
                            "rsa_pearson": distance_geometry_correlation(matched_a, matched_b,
                                                                          method="pearson", max_pairs=PAIR_COUNT),
                            "linear_cka": linear_cka(matched_a, matched_b),
                        }
                        corr_base = {**base, "population": "A_vs_B", "reference": "known_true_shift_plus10"}
                        for metric, value in corr_metrics.items():
                            _append_core_row(rows, corr_base, "cross_population_correspondence", metric, value,
                                             matched_rows=len(lag_support), evaluation_split="test")

    # Real test support is exactly the same trial/time rows in all three populations.
    real_by_key: dict[tuple[str, str, int, str], tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    for item in real_items:
        meta = _real_embedding(item)
        base = {key: item[key] for key in ("dataset", "architecture", "objective", "population", "seed",
                                           "trial_id", "fit_status", "near_collapse")}
        keep_test = _test_mask(item, meta, item["population"])
        coords = np.column_stack((meta["trial_id"][keep_test], meta["time_id"][keep_test])).astype(int).tolist()
        require(coords == support["real"]["coordinates_trial_time"],
                f"Real held-out support mismatch: {item['trial_id']}")
        masks = _split_masks(meta, "Real", item["population"])
        for rep in ("raw", "unit"):
            z = meta[f"embedding_{rep}"]
            for metric, value in _diagnostics(z[keep_test]).items():
                _append_core_row(rows, {**base, "representation": rep, "reference": "none"},
                                 "representation_diagnostics", metric, value,
                                 diagnostic_split="test", pair_sampling_seed=PAIR_SAMPLE_SEED)
            behavior = {"position": meta["position"], "velocity": meta["velocity"]}
            probe_rows, selected = _probe_metrics(z, meta["target"].astype(int), masks=masks,
                                                   seed=int(item["seed"]), progress=meta["progress"],
                                                   behavior=behavior)
            for probe in probe_rows:
                _append_core_row(rows, {**base, "representation": rep, "reference": "none"},
                                 "accessibility", probe["metric"], probe["value"],
                                 selected_C=probe.get("selected_C"), selected_alpha=probe.get("selected_alpha"),
                                 selection_split="validation", final_evaluation_split="test",
                                 evaluation_scope="held_out_generalization")
            selection_rows.append({**base, "representation": rep, **selected})
            real_by_key[(item["architecture"], item["objective"], int(item["seed"]), item["population"])] = (item, meta)

    for arch in ARCHS:
        for objective in OBJECTIVES:
            for seed in SEEDS:
                a_item, a = real_by_key[(arch, objective, seed, "A_PROXIMAL")]
                b_item, b = real_by_key[(arch, objective, seed, "B_DISTAL")]
                ai = np.flatnonzero(_test_mask(a_item, a, "A_PROXIMAL"))
                bi = np.flatnonzero(_test_mask(b_item, b, "B_DISTAL"))
                ac = list(zip(a["trial_id"][ai].astype(int), a["time_id"][ai].astype(int)))
                bc = list(zip(b["trial_id"][bi].astype(int), b["time_id"][bi].astype(int)))
                require(ac == bc, f"Real A/B trial/time correspondence differs: {arch}/{objective}/{seed}")
                for rep in ("raw", "unit"):
                    ea, eb = a[f"embedding_{rep}"][ai], b[f"embedding_{rep}"][bi]
                    metrics = {"procrustes_r2": procrustes_r2(ea, eb),
                               "rsa_spearman": distance_geometry_correlation(ea, eb, method="spearman", max_pairs=PAIR_COUNT),
                               "rsa_pearson": distance_geometry_correlation(ea, eb, method="pearson", max_pairs=PAIR_COUNT),
                               "linear_cka": linear_cka(ea, eb)}
                    status_a = str(a_item["fit_status"])
                    status_b = str(b_item["fit_status"])
                    base = {"dataset": "Real", "architecture": arch, "objective": objective,
                            "population": "A_PROXIMAL_vs_B_DISTAL", "seed": seed,
                            "trial_id": f"{a_item['trial_id']}|{b_item['trial_id']}",
                            "fit_status": f"{status_a}|{status_b}",
                            "near_collapse": bool(a_item["near_collapse"] or b_item["near_collapse"]),
                            "representation": rep, "reference": "matched_A_B"}
                    for metric, value in metrics.items():
                        _append_core_row(rows, base, "cross_population_consistency", metric, value,
                                         evaluation_split="test", matched_rows=len(ac))

    # Record all and only real final near-collapse cases; Synthetic's excluded
    # finalist collapse is listed separately as a non-selected HPO diagnostic.
    for slot_id, record in read_json(output / AUDIT_NAME)["real"]["slots"].items():
        if record["fit_status"] == "INELIGIBLE_NEAR_COLLAPSE":
            trial_root = Path(record["checkpoint"]).parent
            trial_record = read_json(trial_root / "trial_record.json")
            cfg = trial_record["config"]
            rows.append({"dataset": "Real", "architecture": cfg["architecture"],
                         "objective": cfg["objective"], "population": cfg["population"],
                         "seed": int(cfg["training_seed_root"]), "trial_id": slot_id,
                         "fit_status": record["fit_status"], "near_collapse": True,
                         "category": "near_collapse_inventory", "metric": "status", "value": 1.0,
                         "representation": "not_applicable", "reference": "not_applicable"})
    finalist_path = project / SYN_STUDY / "finalist_results.csv"
    for record in csv.DictReader(finalist_path.open(encoding="utf-8", newline="")):
        if record.get("status") == "INELIGIBLE_NEAR_COLLAPSE":
            rows.append({"dataset": "Synthetic", "architecture": record.get("architecture"),
                         "objective": record.get("objective"), "population": record.get("population"),
                         "seed": record.get("training_seed_root"), "trial_id": record.get("trial_id"),
                         "fit_status": record.get("status"), "near_collapse": True,
                         "selected_final_checkpoint": False,
                         "category": "near_collapse_inventory", "metric": "status", "value": 1.0,
                         "representation": "not_applicable", "reference": "not_applicable"})
    return rows, selection_rows


def _summary_tables(rows: list[dict[str, Any]], output: Path) -> dict[str, pd.DataFrame]:
    frame = pd.DataFrame(rows)
    tables = {
        "synthetic_geometry": frame[(frame.dataset == "Synthetic") & (frame.category == "geometry")],
        "synthetic_accessibility": frame[(frame.dataset == "Synthetic") & (frame.category == "accessibility")],
        "synthetic_lag": frame[(frame.dataset == "Synthetic") & (frame.category == "temporal_fidelity")],
        "synthetic_ab_correspondence": frame[(frame.dataset == "Synthetic") &
                                              (frame.category == "cross_population_correspondence")],
        "real_ab_consistency": frame[(frame.dataset == "Real") & (frame.category == "cross_population_consistency")],
        "real_accessibility": frame[(frame.dataset == "Real") & (frame.category == "accessibility")],
        "dimension_diagnostics": frame[frame.category == "representation_diagnostics"],
        "near_collapse_cases": frame[frame.category == "near_collapse_inventory"].drop_duplicates(
            subset=["dataset", "trial_id"]),
    }
    summary_by_name = {"synthetic_geometry": "SYNTHETIC_GEOMETRY_SUMMARY.csv",
                       "synthetic_accessibility": "SYNTHETIC_ACCESSIBILITY_SUMMARY.csv",
                       "synthetic_lag": "SYNTHETIC_LAG_SUMMARY.csv",
                       "synthetic_ab_correspondence": "SYNTHETIC_AB_CORRESPONDENCE_SUMMARY.csv",
                       "real_ab_consistency": "REAL_AB_CONSISTENCY_SUMMARY.csv",
                       "real_accessibility": "REAL_ACCESSIBILITY_SUMMARY.csv",
                       "dimension_diagnostics": "DIMENSION_DIAGNOSTICS_SUMMARY.csv",
                       "near_collapse_cases": "NEAR_COLLAPSE_CASES.csv"}
    for name, path in summary_by_name.items():
        _save_csv_once(output / path, tables[name])
    tables["cross_seed"] = _summarize_seeds(rows)
    _save_csv_once(output / "CROSS_SEED_DESCRIPTIVE_SUMMARY.csv", tables["cross_seed"])
    return tables


def _write_provenance(project: Path, output: Path, support_path: Path,
                      rows: list[dict[str, Any]], tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    files = [output / name for name in (
        "CORE_METRICS_LONG.csv", "SYNTHETIC_GEOMETRY_SUMMARY.csv", "SYNTHETIC_ACCESSIBILITY_SUMMARY.csv",
        "SYNTHETIC_LAG_SUMMARY.csv", "REAL_AB_CONSISTENCY_SUMMARY.csv", "REAL_ACCESSIBILITY_SUMMARY.csv",
        "SYNTHETIC_AB_CORRESPONDENCE_SUMMARY.csv", "DIMENSION_DIAGNOSTICS_SUMMARY.csv", "NEAR_COLLAPSE_CASES.csv", "CROSS_SEED_DESCRIPTIVE_SUMMARY.csv",
        "EMBEDDING_INDEX.csv", "DOWNSTREAM_PROBE_SELECTION.csv")]
    files.append(output / "AUDIT_ISSUES_AND_FIXES.md")
    figure_dir = output / "figures"
    files.extend(sorted(figure_dir.glob("*.png")) + sorted(figure_dir.glob("*.pdf")))
    files.append(figure_dir / "figure_manifest.json")
    embedding_root = output / "embeddings"
    incomplete_staging = {}
    for path in sorted(embedding_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(embedding_root)
        if any(".staging-" in part for part in relative.parts):
            incomplete_staging[str(path.relative_to(project))] = sha256_file(path)
        elif path.name == "manifest.json" or path.suffix == ".npz":
            files.append(path)
    for path in sorted(output.rglob(".tmp-*")):
        if path.is_file():
            incomplete_staging[str(path.relative_to(project))] = sha256_file(path)
    archived_preflights = {
        path.name: sha256_file(path)
        for path in sorted(output.glob("*_PRE_*.json")) if path.is_file()
    }
    audit_path = output / AUDIT_NAME
    payload = {
        "provenance_version": "neurobridge_final_core_metrics_v1",
        "audit_manifest_sha256": sha256_file(audit_path),
        "evaluation_support_manifest_sha256": sha256_file(support_path),
        "source_hashes": read_json(audit_path)["source_artifact_hashes"],
        "evaluation_driver_sha256": sha256_file(Path(__file__).resolve()),
        "embedding_exporter_sha256": sha256_file(Path(real_export.__file__).resolve()),
        "no_training_or_hpo": True,
        "checkpoint_inference_only": True,
        "split_usage": "TRAIN for probe fitting; VALIDATION for downstream hyperparameter selection; TEST once for all held-out metrics",
        "raw_and_unit_embeddings": True,
        "metrics_row_count": len(rows),
        "summary_counts": {key: len(value) for key, value in tables.items()},
        "probe_grids": {"multinomial_logistic_C": list(PROBE_C_GRID), "ridge_alpha": list(RIDGE_ALPHA_GRID)},
        "lag_grid_bins": list(LAGS),
        "pairwise_rsa_sample_count_max": PAIR_COUNT,
        "pairwise_diagnostic_sample_seed": PAIR_SAMPLE_SEED,
        "real_accessibility_targets": {
            "direction_balanced_accuracy": "trial-level movement direction",
            "progress_r2_mae": "temporal progress coordinate; reported separately from behavior/kinematics",
            "position_r2_mae": "continuous kinematic position",
            "velocity_r2_mae": "continuous kinematic velocity",
        },
        "synthetic_truth": "Z is primary; M and eta are complementary; B components are shifted by the frozen lag",
        "real_interpretation": "cross-population consistency only; no biological latent ground truth",
        "near_collapse_rule": "all 72 Real slots retained; 3 ineligible near-collapse models flagged; no replacement",
        "scientific_test_metrics_used_for_selection": False,
        "output_artifacts": {str(path.relative_to(project)): sha256_file(path) for path in files if path.is_file()},
        "preserved_incomplete_embedding_staging": incomplete_staging,
        "preserved_preflight_manifests": archived_preflights,
    }
    write_manifest_once(output / "PROVENANCE.json", payload)
    return payload


def run_core(project: Path) -> dict[str, Any]:
    output, support_path, audit, support = verify_frozen_manifests(project)
    existing_provenance = output / "PROVENANCE.json"
    if existing_provenance.exists():
        payload = read_json(existing_provenance)
        require(payload["audit_manifest_sha256"] == sha256_file(output / AUDIT_NAME) and
                payload["evaluation_support_manifest_sha256"] == sha256_file(support_path),
                "completed core provenance parent hash mismatch; preserve existing outputs")
        for rel, digest in payload["output_artifacts"].items():
            artifact = project / rel
            require(artifact.is_file() and sha256_file(artifact) == digest,
                    f"completed core output hash mismatch: {artifact}")
        print(json.dumps({"status": "ALREADY_COMPLETE_REUSED", "core_provenance_sha256": sha256_file(existing_provenance),
                          "training_performed": False}, indent=2))
        return payload
    item_records: list[dict[str, Any]] = []
    for item in _load_synthetic_records(project):
        item_records.append({**_attach_frozen_fit_status(item, audit["synthetic"]["slots"]),
                             "dataset": "Synthetic"})
    real_spec = read_json(project / REAL_SPEC)
    # Build Real records only from the frozen spec and verified fit table.
    for slot in real_spec["final_slots"]:
        record = audit["real"]["slots"][slot["trial_id"]]
        item_records.append({"dataset": "Real", "trial_id": slot["trial_id"],
                             "architecture": slot["architecture"], "objective": slot["objective"],
                             "population": slot["population"], "seed": int(slot["training_seed_root"]),
                             "fit_status": record["fit_status"], "near_collapse": record["near_collapse"]})
    embedding_index = output / "EMBEDDING_INDEX.csv"
    if not embedding_index.exists():
        syn_written = export_synthetic(project, output, support)
        real_written = export_real(project, output, support_path)
        written = syn_written + real_written
        _save_csv_once(embedding_index, pd.DataFrame(written))
    else:
        index = pd.read_csv(embedding_index).to_dict("records")
        written = index
        require(len(index) == 122, "embedding index does not contain 48 Synthetic + 2 PCA + 72 Real references")
    _verify_embedding_index(project, output, audit, support, written)
    # Resolve frozen indexed embeddings by identity, without changing the run set.
    by_id = {(str(x["trial_id"]), str(x["architecture"]), str(x["objective"]), str(x["population"]), str(x["seed"])): x
             for x in written}
    for item in item_records:
        key = (str(item["trial_id"]), str(item["architecture"]), str(item["objective"]),
               str(item["population"]), str(item["seed"]))
        require(key in by_id, f"frozen embedding absent from index: {key}")
        item.update({k: by_id[key][k] for k in ("embedding_path", "manifest_path", "embedding_sha256")})
    pca_items = [x for x in written if x["architecture"] == "pca"]
    item_records.extend([{**x, "dataset": "Synthetic"} for x in pca_items])
    metrics_path = output / "CORE_METRICS_LONG.csv"
    selection_path = output / "DOWNSTREAM_PROBE_SELECTION.csv"
    if metrics_path.is_file() and selection_path.is_file():
        rows = pd.read_csv(metrics_path).to_dict("records")
        selection = pd.read_csv(selection_path).to_dict("records")
    else:
        # If interrupted between the two immutable publications, regenerate the
        # deterministic pair; an existing peer must match byte-for-byte.
        rows, selection = _evaluate_core(project, output, support, item_records)
        _save_csv_once(metrics_path, pd.DataFrame(rows))
        _save_csv_once(selection_path, pd.DataFrame(selection))
    tables = _summary_tables(rows, output)
    _plot_core(rows, output / "figures")
    # _plot_core is deterministic; verify generated figures before final provenance.
    for figure in list((output / "figures").glob("*.png")) + list((output / "figures").glob("*.pdf")):
        require(figure.stat().st_size > 1000, f"empty/invalid core figure: {figure}")
    provenance = _write_provenance(project, output, support_path, rows, tables)
    print(json.dumps({"core_metrics_rows": len(rows), "embedding_index_rows": len(written),
                      "real_final_instances": 72, "real_near_collapse": 3,
                      "synthetic_final_neural_instances": 48, "training_performed": False,
                      "audit_manifest_sha256": sha256_file(output / AUDIT_NAME),
                      "core_provenance_sha256": sha256_file(output / "PROVENANCE.json")}, indent=2))
    return provenance


def _verify_embedding_index(project: Path, output: Path, audit: dict[str, Any],
                            support: dict[str, Any], records: list[dict[str, Any]]) -> None:
    require(len(records) == 122, "embedding inventory differs from frozen 48 Synthetic + 2 PCA + 72 Real set")
    keys = []
    synthetic_slots = audit["synthetic"]["slots"]
    real_slots = audit["real"]["slots"]
    support_path = output / SUPPORT_NAME
    compatible_support_hashes, compatible_support_file_hashes = _compatible_support_parents(output, support)
    for item in records:
        dataset = str(item["dataset"]).lower()
        trial_id = str(item["trial_id"])
        arch, objective, pop, seed = (str(item[k]) for k in ("architecture", "objective", "population", "seed"))
        keys.append((dataset, trial_id, arch, objective, pop, seed))
        path = Path(str(item["embedding_path"]))
        manifest_path = Path(str(item["manifest_path"]))
        if not path.is_absolute(): path = project / path
        if not manifest_path.is_absolute(): manifest_path = project / manifest_path
        require(path.exists() and manifest_path.is_file(), f"indexed embedding or manifest missing: {path}")
        embed_root = (output / "embeddings").resolve()
        require(path.resolve().is_relative_to(embed_root) and manifest_path.resolve().is_relative_to(embed_root),
                f"indexed artifact path escapes the immutable core embedding tree: {path}")
        require(sha256_file(manifest_path) == str(item["manifest_sha256"]),
                f"indexed embedding manifest hash mismatch: {manifest_path}")
        manifest = read_json(manifest_path)
        manifest_trial_id = (manifest.get("parents", {}).get("trial_id")
                             if dataset == "real" else manifest.get("trial_id"))
        require(str(manifest_trial_id) == trial_id,
                f"embedding manifest trial ID mismatch: {trial_id}")
        if dataset == "real":
            require(trial_id in real_slots, f"unfrozen Real embedding in index: {trial_id}")
            parent = real_slots[trial_id]
            parents = manifest["parents"]
            require(parents["checkpoint_sha256"] == parent["checkpoint_sha256"] and
                    parents["trial_id"] == trial_id and parents["population"] == pop and
                    parents["architecture"] == arch and parents["objective"] == objective and
                    str(parents["seed"]) == seed and
                    parents["evaluation_support_sha256"] in compatible_support_file_hashes,
                    f"Real embedding parent lineage differs from frozen checkpoint/support: {trial_id}")
            for name, digest in manifest["artifact_sha256"].items():
                artifact = manifest_path.parent / name
                require(artifact.is_file() and sha256_file(artifact) == digest,
                        f"Real embedding artifact hash mismatch: {artifact}")
            raw_path = manifest_path.parent / "embedding_raw.npz"
            require(sha256_file(raw_path) == str(item["embedding_sha256"]),
                    f"Real index embedding hash mismatch: {trial_id}")
            data = _real_embedding({**item, "embedding_path": str(manifest_path.parent)})
            require(data["embedding_raw"].shape == (manifest["rows"], 3),
                    f"Real manifest embedding shape differs: {trial_id}")
        elif arch == "pca":
            require(dataset == "synthetic" and pop in audit["canonical_pca_baseline"],
                    f"unexpected PCA embedding in index: {trial_id}")
            source = audit["canonical_pca_baseline"][pop]
            require(manifest["source_embedding_sha256"] == source["embedding_sha256"] and
                    manifest["pca_checkpoint_sha256"] == source["pca_model_sha256"] and
                    manifest["support_parent_sha256"] in compatible_support_hashes,
                    f"PCA source/support lineage mismatch: {trial_id}")
            require(sha256_file(path) == manifest["embedding_sha256"] == str(item["embedding_sha256"]),
                    f"PCA embedding hash mismatch: {trial_id}")
            data = _synthetic_embedding({**item, "embedding_path": str(path)})
        elif dataset == "synthetic":
            require(trial_id in synthetic_slots, f"unfrozen Synthetic embedding in index: {trial_id}")
            parent = synthetic_slots[trial_id]
            stored_collapse = item.get("near_collapse")
            if isinstance(stored_collapse, str):
                stored_collapse = stored_collapse.strip().lower() in {"true", "1"}
            require(str(item.get("fit_status")) == str(parent["fit_status"]) and
                    bool(stored_collapse) == bool(parent["near_collapse"]),
                    f"Synthetic index status differs from frozen slot audit: {trial_id}")
            require(manifest["checkpoint_sha256"] == parent["checkpoint_sha256"] and
                    manifest["config_sha256"] == parent["config_sha256"] and
                    manifest["source_windows_sha256"] == audit["source_artifact_hashes"][f"synthetic_windows_{pop}"]["sha256"] and
                    manifest["source_split_sha256"] == audit["source_artifact_hashes"]["synthetic_split"]["sha256"] and
                    manifest["support_parent_sha256"] in compatible_support_hashes and
                    parent["architecture"] == arch and parent["objective"] == objective and
                    parent["population"] == pop and str(parent["seed"]) == seed,
                    f"Synthetic embedding parent lineage differs from frozen checkpoint/support: {trial_id}")
            require(sha256_file(path) == manifest["embedding_sha256"] == str(item["embedding_sha256"]),
                    f"Synthetic embedding hash mismatch: {trial_id}")
            data = _synthetic_embedding({**item, "embedding_path": str(path)})
        else:
            raise RuntimeError(f"unexpected dataset in embedding inventory: {dataset}")
        raw = data["embedding_raw"]
        unit = data["embedding_unit"]
        expected_unit = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
        require(np.allclose(unit, expected_unit, rtol=2e-5, atol=2e-6),
                f"raw/unit embedding normalization invariant fails: {trial_id}")
    require(len(set(keys)) == 122, "embedding inventory contains duplicate identities")
    expected = {("synthetic", tid, rec["architecture"], rec["objective"], rec["population"], str(rec["seed"]))
                for tid, rec in synthetic_slots.items()}
    expected |= {("real", tid, rec["architecture"], rec["objective"], rec["population"], str(rec["seed"]))
                 for tid, rec in real_slots.items()}
    expected |= {("synthetic", f"pca_none_{pop}", "pca", "none", pop, "deterministic") for pop in POPS_SYN}
    require(set(keys) == expected, "embedding inventory identities differ from frozen slot grid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit", action="store_true", help="verify parents and freeze exact evaluation support")
    mode.add_argument("--run", action="store_true", help="export from frozen checkpoints and evaluate core metrics")
    parser.add_argument("--project", type=Path, default=ROOT)
    args = parser.parse_args()
    project = args.project.resolve()
    if args.audit:
        print(json.dumps(run_audit(project), indent=2))
    else:
        run_core(project)


if __name__ == "__main__":
    main()
