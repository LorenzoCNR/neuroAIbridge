"""Aggregate existing real-data Stage-5 metrics for presentation.

This is a reporting-only utility: it reads completed run manifests and cached
Stage-5 CSVs, does not fit models, and does not compute new metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SEEDS = (42, 123, 456)
POPULATIONS = ("total_65", "A", "B")
BRANCHES = ("held_out", "full_sample")
ARCHITECTURES = ("cnn1d", "transformer")
OBJECTIVES = (
    "soft",
    "infonce",
    "time_contrastive_blocks",
    "behavior_contrastive_blocks",
)
POINT_FILES = (
    "pca_none.csv",
    "representation_geometry.csv",
    *(f"{architecture}_{objective}.csv" for architecture in ARCHITECTURES for objective in OBJECTIVES),
)
BOOTSTRAP_FILES = tuple(
    f"{architecture}_{objective}_trial_bootstrap.csv"
    for architecture in ARCHITECTURES
    for objective in OBJECTIVES
)
OUTPUT_FIELDS = (
    "run_id",
    "dataset",
    "setting",
    "population",
    "branch",
    "architecture",
    "objective",
    "training_seed",
    "channel_split_seed",
    "reference",
    "metric",
    "value",
    "category",
    "evaluation_scope",
    "evaluation_split",
    "representation",
    "perturbation",
    "perturbation_level",
    "representation_fit_scope",
    "ci_low",
    "ci_high",
    "bootstrap_replicates",
    "n_evaluation_trials",
    "selected_hyperparameter",
    "selection_split",
    "source_artifact",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _real_run_manifests(project_root: Path) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    runs_root = project_root / "outputs" / "runs"
    expected = {(seed, population) for seed in SEEDS for population in POPULATIONS}
    found: dict[tuple[int, str], tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    for manifest_path in sorted(runs_root.glob("*/run_manifest.json")):
        manifest = _read_json(manifest_path)
        config = manifest.get("configuration", {})
        if config.get("setting_name") != "real_0":
            continue
        population = config.get("population_name")
        seed = config.get("training_seed", manifest.get("training_seed"))
        if population not in POPULATIONS or seed not in SEEDS:
            continue
        key = (int(seed), population)
        if key in found:
            raise RuntimeError(f"duplicate completed real_0 run for seed/population {key}: {manifest_path}")
        if manifest.get("status") != "complete":
            raise RuntimeError(f"real_0 run is not complete: {manifest_path}")
        if config.get("channel_split_seed", manifest.get("channel_split_seed")) != 42:
            raise RuntimeError(f"unexpected channel-partition seed in {manifest_path}")
        if config.get("max_iterations") != 4000:
            raise RuntimeError(f"unexpected max_iterations in {manifest_path}")
        found[key] = (manifest_path.parent, manifest, config)

    missing = expected - set(found)
    extra = set(found) - expected
    if missing or extra:
        raise RuntimeError(f"real run coverage mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    return [found[key] for key in sorted(found)]


def _parse_number(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _point_records(
    run_root: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
    branch: str,
    project_root: Path,
) -> list[dict[str, str]]:
    stage = run_root / "stage05_metrics" / branch
    expected_files = set(POINT_FILES) | set(BOOTSTRAP_FILES)
    present_files = {path.name for path in stage.glob("*.csv")}
    missing_files = expected_files - present_files
    if missing_files:
        raise FileNotFoundError(f"{run_root.name}/{branch} is missing Stage-5 files: {sorted(missing_files)}")

    records: list[dict[str, str]] = []
    record_indexes: dict[tuple[str, str, str, str, str, str], list[int]] = {}
    for filename in POINT_FILES:
        source = stage / filename
        for metric_row in _read_csv(source):
            artifact_branch = metric_row.get("branch")
            if artifact_branch not in (None, "", branch):
                raise ValueError(f"branch mismatch in {source}")
            evaluation_scope = metric_row.get("evaluation_scope") or (
                "held_out_representation_generalization"
                if branch == "held_out"
                else "descriptive_in_sample_representation"
            )
            evaluation_split = metric_row.get("evaluation_split", "")
            if not evaluation_split:
                if branch == "held_out":
                    evaluation_split = "test"
                elif filename == "representation_geometry.csv":
                    evaluation_split = "all_trials"
            n_evaluation_trials = metric_row.get("n_evaluation_trials") or metric_row.get("n_test_trials", "")
            value = metric_row.get("value", "")
            parsed_value = _parse_number(value)
            if parsed_value is None or not math.isfinite(parsed_value):
                raise ValueError(f"non-finite or non-numeric metric value in {source}: {value!r}")
            architecture = metric_row.get("model", "")
            objective = metric_row.get("objective") or metric_row.get("loss", "")
            representation = metric_row.get("representation", "")
            metric = metric_row.get("metric", "")
            record = {
                "run_id": manifest.get("run_id", run_root.name),
                "dataset": "real_monkey",
                "setting": config.get("setting_name", ""),
                "population": config.get("population_name", ""),
                "branch": branch,
                "architecture": architecture,
                "objective": objective,
                "training_seed": str(config.get("training_seed", manifest.get("training_seed", ""))),
                "channel_split_seed": str(config.get("channel_split_seed", manifest.get("channel_split_seed", ""))),
                "reference": metric_row.get("reference_geometry", metric_row.get("reference", "")),
                "metric": metric,
                "value": value,
                "category": metric_row.get("category", ""),
                "evaluation_scope": evaluation_scope,
                "evaluation_split": evaluation_split,
                "representation": representation,
                "perturbation": metric_row.get("perturbation", ""),
                "perturbation_level": metric_row.get("perturbation_level", ""),
                "representation_fit_scope": metric_row.get("representation_fit_scope") or (
                    "training_trials_only" if branch == "held_out" else "all_trials"
                ),
                "ci_low": metric_row.get("trial_bootstrap_ci_low", ""),
                "ci_high": metric_row.get("trial_bootstrap_ci_high", ""),
                "bootstrap_replicates": metric_row.get("bootstrap_replicates", ""),
                "n_evaluation_trials": n_evaluation_trials,
                "selected_hyperparameter": "",
                "selection_split": "",
                "source_artifact": source.relative_to(project_root).as_posix(),
            }
            index = len(records)
            records.append(record)
            key = (architecture, objective, representation, metric, evaluation_split, value)
            record_indexes.setdefault(key, []).append(index)

    for filename in BOOTSTRAP_FILES:
        source = stage / filename
        for bootstrap_row in _read_csv(source):
            key = (
                bootstrap_row.get("model", ""),
                bootstrap_row.get("objective", ""),
                bootstrap_row.get("representation", ""),
                bootstrap_row.get("metric", ""),
                bootstrap_row.get("evaluation_split", ""),
                bootstrap_row.get("value", ""),
            )
            candidates = record_indexes.get(key, [])
            candidates = [
                index
                for index in candidates
                if records[index]["category"] in {"downstream_task_decoding", "downstream_behavior_decoding"}
            ]
            if len(candidates) != 1:
                raise ValueError(f"bootstrap metric does not uniquely match a point estimate in {source}: {key}")
            record = records[candidates[0]]
            record["ci_low"] = bootstrap_row.get("trial_bootstrap_ci_low", "")
            record["ci_high"] = bootstrap_row.get("trial_bootstrap_ci_high", "")
            record["bootstrap_replicates"] = bootstrap_row.get("bootstrap_replicates", "")
            record["selected_hyperparameter"] = bootstrap_row.get("selected_hyperparameter", "")
            record["selection_split"] = bootstrap_row.get("selection_split", "")

    return records


def build_real_branch_report(project_root: str | Path) -> tuple[Path, Path, int, list[str]]:
    project_root = Path(project_root).resolve()
    runs = _real_run_manifests(project_root)
    records: list[dict[str, str]] = []
    run_ids: list[str] = []
    branch_csv_counts: dict[str, int] = {}
    for run_root, manifest, config in runs:
        run_ids.append(manifest.get("run_id", run_root.name))
        for branch in BRANCHES:
            stage = run_root / "stage05_metrics" / branch
            branch_csv_counts[f"{run_root.name}/{branch}"] = len(list(stage.glob("*.csv")))
            records.extend(_point_records(run_root, manifest, config, branch, project_root))

    table_path = project_root / "outputs" / "presentation_ready" / "tables" / "REAL_BRANCH_METRICS.csv"
    summary_path = project_root / "outputs" / "presentation_ready" / "summary" / "RESULTS_FOR_PRESENTATION.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    records.sort(
        key=lambda row: (
            int(row["training_seed"]),
            {"total_65": 0, "A": 1, "B": 2}.get(row["population"], 9),
            row["branch"],
            row["architecture"],
            row["objective"],
            row["metric"],
            row["reference"],
            row["representation"],
            row["perturbation"],
            row["perturbation_level"],
        )
    )
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    expected_csv_count = len(POINT_FILES) + len(BOOTSTRAP_FILES)
    if set(branch_csv_counts.values()) != {expected_csv_count}:
        raise RuntimeError(f"unexpected Stage-5 CSV counts: {branch_csv_counts}")
    run_links = "\n".join(
        f"- `{run_id}` - [manifest](../../runs/{run_id}/run_manifest.json)"
        for run_id in run_ids
    )
    geometry_figure_links = "\n".join(
        f"- `{run_id}` - [full-sample behavioral-geometry RSA](../../runs/{run_id}/stage06_figures/full_sample/behavioral_geometry_rsa_spearman.png)"
        for run_id in run_ids
    )
    branch_coverage = len(runs) * len(BRANCHES)
    summary = f"""# Real-data results: branch-aware report

## Coverage and protocol

- Completed baseline real-data runs: **{len(runs)}** (training seeds 42, 123, 456; populations total_65, A, B).
- Stage-5 branches: **held_out** and **full_sample** for every run ({branch_coverage} run/branch combinations).
- Each run/branch has **{expected_csv_count} Stage-5 CSV artifacts**: point metrics, representation geometry, and the existing trial-bootstrap summaries.
- The real-data frozen schedule is `max_iterations=4000`, validation interval 150, minimum 750 updates, patience 5, and relative minimum delta 0.001; channel-partition seed is 42. Model/objective names, measured values, raw/unit representations, scope/split metadata, and available bootstrap intervals are kept run-wise in [REAL_BRANCH_METRICS.csv](../tables/REAL_BRANCH_METRICS.csv).
- No model was retrained for this reporting repair; the table was assembled from completed manifests and cached Stage-5 outputs.

## How to read the branches

- **held_out** is the primary evidence for held-out accessibility/generalization.
- **full_sample** is explicitly marked `descriptive_in_sample_representation`: its representation was fit using all trials. Its decoder rows are descriptive and must not be presented as an independent generalization estimate.
- Full-sample CKA and window-level participation-ratio diagnostics use all windows; their held-out versions use test windows. Trial-prototype geometry/RSA and its prototype-level participation-ratio diagnostic use 193 trials in `full_sample` and the recorded test trials in `held_out`. These are distinct records; the table preserves their categories, references, and scopes rather than pooling them.
- `raw` and `unit` are separate rows. RSA rows retain their behavioral reference geometry and existing bootstrap intervals. Decoder robustness rows retain their original category and perturbation fields.

## Metric contents

This is an aggregation of existing metrics only: task/behavior decoding, post-hoc embedding-noise robustness, linear CKA to input windows, participation ratio, and behavioral-geometry RSA. No metric is introduced, recomputed with a new definition, or averaged across seeds in this table.

Controlled-lag R10 artifacts remain in their separate run folders and are not
mixed into this R0 baseline table.

## Run manifests

{run_links}

## Full-sample figures

The evaluation stage generated the branch-matched RSA figure for every real
run; these are linked from their immutable run folders:

{geometry_figure_links}

Embedding/trajectory and raw-versus-unit figures remain alongside these under
each run's `stage06_figures/<branch>/` directory.

## Existing synthetic table

The pre-existing [MAIN_RESULTS_TABLE.csv](../tables/MAIN_RESULTS_TABLE.csv) is preserved unchanged; it contains the synthetic seed-42 aggregation. This real-data table is separate so its branch scopes and evaluation provenance remain explicit.
"""
    summary_path.write_text(summary, encoding="utf-8")
    return table_path, summary_path, len(records), run_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    table, summary, rows, run_ids = build_real_branch_report(args.project_root)
    print(f"runs={len(run_ids)} rows={rows}")
    print(f"table={table}")
    print(f"summary={summary}")


if __name__ == "__main__":
    main()
