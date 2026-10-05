"""Recover the final label-null merge from completed V2 checkpoints only.

The original 244 permutation slots are immutable. This script fixes only the
missing-seed key normalization for the deterministic PCA reference and never
loads embeddings or refits a probe.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import final_gap_closure_completion as comp


OUT = comp.ROOT / "label_shuffle_holm_3000"
CHECKPOINTS = OUT / "_checkpoints"
OLD_REPLICATES = comp.NULLS / "NULL_LABEL_SHUFFLE_REPLICATES.csv"
OLD_SUMMARY = comp.NULLS / "NULL_LABEL_SHUFFLE.csv"
ISSUE_ID = "GC2-01"


def main() -> None:
    if (OUT / "PROVENANCE.json").exists():
        raise FileExistsError(f"final merge already sealed: {OUT / 'PROVENANCE.json'}")
    campaign_path = CHECKPOINTS / "CAMPAIGN.json"
    campaign_wrapped = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign_hash = campaign_wrapped["campaign_sha256"]
    campaign = {k: v for k, v in campaign_wrapped.items() if k != "campaign_sha256"}
    if comp.canonical_sha(campaign) != campaign_hash:
        raise ValueError("campaign manifest hash mismatch")

    parent_paths = {
        "core_provenance_sha256": comp.CORE / "PROVENANCE.json",
        "core_metrics_sha256": comp.CORE / "CORE_METRICS_LONG.csv",
        "embedding_index_sha256": comp.CORE / "EMBEDDING_INDEX.csv",
        "support_sha256": comp.CORE / "EVALUATION_SUPPORT_MANIFEST.json",
        "original_label_summary_sha256": OLD_SUMMARY,
        "original_label_replicates_sha256": OLD_REPLICATES,
        "driver_sha256": comp.PROJECT / "tools/final_gap_closure_completion.py",
        "downstream_eval_source_sha256": comp.PROJECT / "tools/final_thesis_downstream_eval.py",
    }
    for name, path in parent_paths.items():
        if comp.sha(path) != campaign["parent_hashes"][name]:
            raise ValueError(f"campaign parent hash mismatch: {name}")

    with OLD_REPLICATES.open(newline="", encoding="utf-8") as f:
        original = list(csv.DictReader(f))
    with OLD_SUMMARY.open(newline="", encoding="utf-8") as f:
        original_summary = list(csv.DictReader(f))
    if len(original) != 244_000 or len(original_summary) != 244:
        raise ValueError("unexpected frozen 1000-permutation table dimensions")
    if {int(row["permutation_seed"]) for row in original} != set(range(730_000, 731_000)):
        raise ValueError("frozen label-null seed range mismatch")

    total_reps = int(campaign["settings"]["total_replicates"])
    extension_reps = int(campaign["settings"]["additional_replicates"])
    seed_start, seed_end = campaign["settings"]["extension_seed_range"]
    if (total_reps != 3000 or extension_reps != 2000 or
            seed_start != 731000 or seed_end != 732999):
        raise ValueError("campaign settings differ from frozen 3000-replicate extension")
    if len(campaign["slot_ids"]) != 244:
        raise ValueError("campaign slot count mismatch")

    extension_by_key: dict[tuple[str, ...], np.ndarray] = {}
    extension_rows: list[dict[str, object]] = []
    seen_slots: set[str] = set()
    for slot_id in campaign["slot_ids"]:
        slot_hash = hashlib.sha256(slot_id.encode("utf-8")).hexdigest()
        checkpoint_path = CHECKPOINTS / f"{slot_hash}.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("campaign_sha256") != campaign_hash:
            raise ValueError(f"checkpoint campaign hash mismatch: {checkpoint_path.name}")
        values = np.asarray(checkpoint.get("values", []), dtype=float)
        if values.shape != (extension_reps,) or not np.all(np.isfinite(values)):
            raise ValueError(f"incomplete/non-finite extension checkpoint: {checkpoint_path.name}")
        identity = json.loads(slot_id)
        dataset = identity["dataset"]
        metric = ("condition_balanced_accuracy" if dataset == "Synthetic"
                  else "direction_balanced_accuracy")
        # Frozen PCA rows encode their absent training seed as ""; normalize
        # JSON null to the same canonical value (GC2-01).
        training_seed = "" if identity["training_seed"] is None else str(identity["training_seed"])
        key = (dataset, identity["architecture"], identity["objective"],
               identity["population"], training_seed, identity["trial_id"],
               identity["representation"], metric)
        if key in extension_by_key:
            raise ValueError(f"duplicate extension key: {key}")
        extension_by_key[key] = values
        seen_slots.add(slot_hash)
        extension_rows.extend({
            "dataset": key[0], "architecture": key[1], "objective": key[2],
            "population": key[3], "training_seed": key[4], "trial_id": key[5],
            "representation": key[6], "metric": key[7], "replicate": 1000 + i,
            "permutation_seed": int(seed_start) + i, "null_balanced_accuracy": float(value),
        } for i, value in enumerate(values))
    if len(seen_slots) != 244 or len(extension_rows) != 488_000:
        raise ValueError("extension checkpoint/row total mismatch")
    if len(list(CHECKPOINTS.glob("*.json"))) != 245:
        raise ValueError("checkpoint directory contains missing or unexpected JSON files")

    original_by_key: dict[tuple[str, ...], list[float]] = {}
    for row in original:
        key = (row["dataset"], row["architecture"], row["objective"], row["population"],
               row["training_seed"], row["trial_id"], row["representation"], row["metric"])
        original_by_key.setdefault(key, []).append(float(row["null_balanced_accuracy"]))
    if set(original_by_key) != set(extension_by_key):
        raise ValueError("frozen and extension null slot identities do not match")
    if any(len(values) != 1000 for values in original_by_key.values()):
        raise ValueError("one or more frozen slots do not contain exactly 1000 null values")

    combined_summary: list[dict[str, object]] = []
    for row in original_summary:
        key = (row["dataset"], row["architecture"], row["objective"], row["population"],
               row["training_seed"], row["trial_id"], row["representation"], row["metric"])
        if key not in extension_by_key:
            raise KeyError(f"extension slot missing for frozen summary key: {key}")
        values = np.r_[np.asarray(original_by_key[key], dtype=float), extension_by_key[key]]
        observed = float(row["observed"])
        combined_summary.append({
            **row,
            "null_mean": float(values.mean()),
            "null_sd": float(values.std(ddof=1)),
            "empirical_p": float((1 + np.sum(values >= observed)) / (total_reps + 1)),
            "percentile_within_null": float(100 * (np.sum(values < observed) +
                .5 * np.sum(values == observed)) / len(values)),
            "observed_minus_null_mean": observed - float(values.mean()),
            "replicates": total_reps,
            "permutation_seed_start": 730000,
            "permutation_seed_end": int(seed_end),
            "unit_of_permutation": "complete trial label within split",
        })
    if len(combined_summary) != 244:
        raise ValueError("combined label-null summary row count mismatch")

    holm_rows: list[dict[str, object]] = []
    holm_summary: list[dict[str, object]] = []
    for dataset in ("Synthetic", "Real"):
        family = [row for row in combined_summary if row["dataset"] == dataset]
        family.sort(key=lambda row: float(row["empirical_p"]))
        previous = 0.0
        for rank, row in enumerate(family):
            adjusted = max(previous, min(1.0, (len(family) - rank) * float(row["empirical_p"])))
            previous = adjusted
            holm_rows.append({
                "dataset": dataset, "architecture": row["architecture"],
                "objective": row["objective"], "population": row["population"],
                "training_seed": row["training_seed"], "trial_id": row["trial_id"],
                "representation": row["representation"], "metric": row["metric"],
                "observed": row["observed"], "empirical_p": row["empirical_p"],
                "holm_p_family": adjusted, "family_size": len(family),
            })
        holm_summary.append({
            "family": f"{dataset}_label_balanced_accuracy", "n_tests": len(family),
            "raw_p_lt_0_05": sum(float(row["empirical_p"]) < .05 for row in family),
            "holm_p_lt_0_05": sum(float(row["holm_p_family"]) < .05 for row in holm_rows
                                   if row["dataset"] == dataset),
            "total_permutations": total_reps,
            "minimum_attainable_p": 1 / (total_reps + 1),
            "minimum_attainable_holm_p": min(1.0, len(family) / (total_reps + 1)),
        })

    outputs = {
        "NULL_LABEL_SHUFFLE_EXTENSION_REPLICATES_2000.csv": comp.csv_bytes(extension_rows),
        "NULL_LABEL_SHUFFLE_SUMMARY_3000.csv": comp.csv_bytes(combined_summary),
        "HOLM_LABEL_FAMILY_RESULTS_3000.csv": comp.csv_bytes(holm_rows),
        "HOLM_LABEL_FAMILY_SUMMARY_3000.csv": comp.csv_bytes(holm_summary),
        "PERMUTATION_SEEDS.csv": comp.csv_bytes(
            [{"stage": "original_reused", "replicate": i, "seed": 730000 + i}
             for i in range(1000)] +
            [{"stage": "extension", "replicate": 1000 + i, "seed": int(seed_start) + i}
             for i in range(extension_reps)]),
    }
    for name, data in outputs.items():
        comp.write_once(OUT / name, data)

    helper_path = Path(__file__).resolve()
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE",
        "parent_hashes": {**campaign["parent_hashes"],
                          "campaign_manifest_sha256": comp.sha(campaign_path)},
        "settings": campaign["settings"],
        "no_model_selection": True,
        "no_encoder_training": True,
        "aggregation_recovery": {
            "issue_id": ISSUE_ID,
            "original_campaign_driver_sha256": campaign["parent_hashes"]["driver_sha256"],
            "merge_recovery_source_sha256": comp.sha(helper_path),
            "seed_key_normalization": "JSON null -> empty string to match frozen PCA seed field",
            "reused_extension_checkpoints": len(seen_slots),
            "recomputed_permutation_slots": 0,
        },
        "artifacts": {str(path.relative_to(comp.PROJECT)): comp.sha(path)
                      for path in [OUT / name for name in outputs]},
    }
    comp.write_json_once(OUT / "PROVENANCE.json", provenance)
    print(json.dumps({"status": "COMPLETE", "extension_replicate_rows": len(extension_rows),
                      "summary_rows": len(combined_summary), "holm_summary": holm_summary,
                      "provenance": str(OUT / "PROVENANCE.json")}, indent=2))


if __name__ == "__main__":
    main()
