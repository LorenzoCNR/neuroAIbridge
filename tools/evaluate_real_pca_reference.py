"""Evaluate the frozen Real W201 PCA reference without refitting encoders.

The PCA models/embeddings are parents created by generate_real_pca_reference.py.
This downstream branch mirrors the existing core Real probe and A/B metric
definitions, while keeping PCA results outside the immutable neural core table.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import final_thesis_core_metrics as core  # noqa: E402

PCA_ROOT = ROOT / "outputs/final_thesis_v1/final_evaluation/real_pca_reference"
METRICS_ROOT = PCA_ROOT / "metrics"
CORE_ROOT = ROOT / "outputs/final_thesis_v1/final_evaluation/core_metrics"
CORE_INDEX = CORE_ROOT / "EMBEDDING_INDEX.csv"
SUPPORT_PATH = CORE_ROOT / "EVALUATION_SUPPORT_MANIFEST.json"
SPEC_PATH = ROOT / "outputs/final_thesis_v1/freeze/REAL_FINAL_EXPERIMENT_SPEC.json"
PARTITION_PATH = ROOT / "outputs/phase2a_hpo/partitions/real_somatotopic_v1.json"
POPULATIONS = ("TOTAL65", "A_PROXIMAL", "B_DISTAL")
PRIMARY_METRICS = {
    "direction_balanced_accuracy", "progress_r2", "position_r2", "velocity_r2",
    "procrustes_r2", "rsa_spearman", "rsa_pearson", "linear_cka",
}
CSV_FIELDS = (
    "dataset", "architecture", "objective", "population", "seed", "trial_id",
    "fit_status", "near_collapse", "representation", "category", "reference",
    "metric", "value", "evaluation_split", "evaluation_scope", "fit_scope",
    "n_rows", "n_trials", "selection_split", "selected_C", "selected_alpha",
    "matched_rows", "pair_sampling_seed",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_bytes(rows: list[dict[str, Any]], fields: tuple[str, ...] = CSV_FIELDS) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        safe = {}
        for key, value in row.items():
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
            elif isinstance(value, (np.bool_,)):
                value = bool(value)
            if isinstance(value, float) and not np.isfinite(value):
                value = ""
            safe[key] = value
        writer.writerow(safe)
    return output.getvalue().encode("utf-8")


def _coordinate_lookup(values: dict[str, np.ndarray], label: str) -> dict[tuple[int, int], int]:
    trial = np.asarray(values["trial_id"], dtype=np.int64)
    time = np.asarray(values["time_id"], dtype=np.int64)
    if trial.shape != time.shape or trial.ndim != 1:
        raise ValueError(f"invalid trial/time metadata for {label}")
    lookup = {(int(t), int(tm)): i for i, (t, tm) in enumerate(zip(trial, time))}
    if len(lookup) != len(trial):
        raise ValueError(f"duplicate trial/time coordinates for {label}")
    return lookup


def _load_parent_context() -> tuple[dict, dict, dict, list[dict[str, str]], dict[str, dict[str, Any]]]:
    pca_manifest_path = PCA_ROOT / "manifest.json"
    pca_manifest = _read_json(pca_manifest_path)
    spec = _read_json(SPEC_PATH)
    partition = _read_json(PARTITION_PATH)
    support = _read_json(SUPPORT_PATH)
    if pca_manifest.get("architecture") != "pca" or pca_manifest.get("dataset") != "real":
        raise ValueError("PCA artifact manifest is not the Real reference")
    if (int(pca_manifest["selected_window_size"]) != int(spec["selected_real_window_size"]) or
            int(pca_manifest["selected_window_size"]) != int(support["real"]["window_size"])):
        raise ValueError("PCA reference window differs from frozen Real specification/support")
    if (pca_manifest.get("channel_partition_sha256") != sha256(PARTITION_PATH) or
            pca_manifest.get("source_hashes", {}).get("final_spec_sha256") != sha256(SPEC_PATH)):
        raise ValueError("PCA parent partition/final-spec hash mismatch")

    with CORE_INDEX.open(encoding="utf-8", newline="") as stream:
        index = list(csv.DictReader(stream))
    real_rows = [row for row in index if row.get("dataset", "").lower() == "real"]
    if len(real_rows) != 72:
        raise ValueError(f"expected 72 frozen Real neural reference rows, found {len(real_rows)}")

    reference_metadata: dict[str, dict[str, Any]] = {}
    reference_sources: dict[str, dict[str, Any]] = {}
    for population in POPULATIONS:
        candidates = [row for row in real_rows if row["population"] == population]
        if not candidates:
            raise ValueError(f"no frozen Real metadata parent for {population}")
        item = candidates[0]
        manifest_path = Path(item["manifest_path"])
        if not manifest_path.is_file() or sha256(manifest_path) != item["manifest_sha256"]:
            raise ValueError(f"neural metadata parent hash mismatch for {population}")
        reference_metadata[population] = core._real_embedding(item)
        reference_sources[population] = {
            "trial_id": item["trial_id"],
            "manifest_sha256": item["manifest_sha256"],
            "embedding_sha256": item["embedding_sha256"],
        }
    return pca_manifest, spec, partition, real_rows, {
        "support": support,
        "reference_metadata": reference_metadata,
        "reference_sources": reference_sources,
        "pca_manifest_sha256": sha256(pca_manifest_path),
    }


def _load_pca_data(population: str, manifest: dict[str, Any], reference: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    records = [row for row in manifest["populations"] if row["population"] == population]
    if len(records) != 1:
        raise ValueError(f"PCA manifest population entry is missing/duplicated: {population}")
    record = records[0]
    path = PCA_ROOT / record["embedding"]
    if not path.is_file() or sha256(path) != record["embedding_sha256"]:
        raise ValueError(f"PCA embedding parent hash mismatch: {population}")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    raw, unit = data["embedding_raw"], data["embedding_unit"]
    n = len(raw)
    if raw.shape != (n, 3) or unit.shape != (n, 3) or not np.isfinite(raw).all() or not np.isfinite(unit).all():
        raise ValueError(f"Real PCA embedding must be finite N x 3: {population}")
    required = ("trial_id", "time_id", "global_time_id", "valid_mask", "split", "labels")
    if any(key not in data or len(data[key]) != n for key in required):
        raise ValueError(f"Real PCA metadata cardinality mismatch: {population}")
    norm = np.linalg.norm(raw, axis=1, keepdims=True)
    expected_unit = np.divide(raw, norm, out=np.zeros_like(raw), where=norm > 0)
    if not np.allclose(unit, expected_unit, rtol=2e-5, atol=2e-6):
        raise ValueError(f"raw/unit normalization invariant failed: {population}")

    pca_lookup = _coordinate_lookup(data, f"PCA/{population}")
    ref_lookup = _coordinate_lookup(reference, f"frozen core/{population}")
    if set(pca_lookup) != set(ref_lookup):
        raise ValueError(f"PCA/core metadata coordinate sets differ: {population}")
    ref_indices = np.asarray([ref_lookup[key] for key in pca_lookup], dtype=np.int64)
    if not np.array_equal(data["split"].astype(str), reference["split"][ref_indices].astype(str)):
        raise ValueError(f"PCA/core split labels differ: {population}")
    if not np.array_equal(data["labels"].astype(int), reference["target"][ref_indices].astype(int)):
        raise ValueError(f"PCA direction labels differ from frozen metadata: {population}")
    for key in ("target", "progress", "position", "velocity"):
        data[key] = np.asarray(reference[key])[ref_indices]
    return data


def _base(population: str, representation: str) -> dict[str, Any]:
    return {
        "dataset": "Real", "architecture": "pca", "objective": "none",
        "population": population, "seed": "deterministic",
        "trial_id": f"real_pca_{population}", "fit_status": "PCA_REFERENCE_TRAIN_FIT",
        "near_collapse": "not_applicable", "representation": representation,
        "reference": "none", "fit_scope": "PCA fit on frozen train trials; all rows projected",
    }


def _add_metric(rows: list[dict[str, Any]], base: dict[str, Any], category: str,
                metric: str, value: float, **extra: Any) -> None:
    rows.append({**base, "category": category, "metric": metric,
                 "value": float(value) if np.isfinite(value) else "", **extra})


def evaluate() -> dict[str, Any]:
    if not PCA_ROOT.is_dir():
        raise FileNotFoundError(f"frozen PCA reference missing: {PCA_ROOT}")
    if METRICS_ROOT.exists():
        prov_path = METRICS_ROOT / "PROVENANCE.json"
        if not prov_path.is_file():
            raise FileExistsError(f"partial PCA metric branch preserved: {METRICS_ROOT}")
        previous = _read_json(prov_path)
        for name, digest in previous.get("artifacts_sha256", {}).items():
            path = METRICS_ROOT / name
            if not path.is_file() or sha256(path) != digest:
                raise ValueError(f"existing PCA metric artifact hash mismatch: {path}")
        print(json.dumps({"status": "REUSED", "metrics_root": str(METRICS_ROOT),
                          "artifacts": len(previous.get("artifacts_sha256", {}))}))
        return previous

    pca_manifest, spec, partition, real_rows, context = _load_parent_context()
    support = context["support"]
    metadata_by_population = context["reference_metadata"]
    data_by_population = {
        population: _load_pca_data(population, pca_manifest, metadata_by_population[population])
        for population in POPULATIONS
    }

    test_coords = support["real"]["coordinates_trial_time"]
    for population, data in data_by_population.items():
        mask = core._test_mask({"dataset": "Real"}, data, population)
        observed = np.column_stack((data["trial_id"][mask], data["time_id"][mask])).astype(int).tolist()
        if observed != test_coords:
            raise ValueError(f"PCA held-out support differs from frozen support: {population}")
    if not (len(test_coords) == int(pca_manifest["support_rows"]["held_out_test"]) == 15_600):
        raise ValueError("unexpected held-out PCA support count")

    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for population, data in data_by_population.items():
        masks = core._split_masks(data, "Real", population)
        test_mask = core._test_mask({"dataset": "Real"}, data, population)
        all_mask = data["valid_mask"].astype(bool)
        for representation in ("raw", "unit"):
            embedding = data[f"embedding_{representation}"]
            base = _base(population, representation)
            probe_rows, selected = core._probe_metrics(
                embedding, data["target"].astype(int), masks=masks,
                seed=0, progress=data["progress"],
                behavior={"position": data["position"], "velocity": data["velocity"]},
            )
            for metric_row in probe_rows:
                _add_metric(
                    rows, base, "accessibility", metric_row["metric"], metric_row["value"],
                    evaluation_split="test", evaluation_scope="held_out_generalization",
                    n_rows=int(test_mask.sum()), n_trials=int(np.unique(data["trial_id"][test_mask]).size),
                    selection_split="validation", selected_C=metric_row.get("selected_C"),
                    selected_alpha=metric_row.get("selected_alpha"),
                )
            selection_rows.append({
                **{key: base[key] for key in ("dataset", "architecture", "objective", "population", "seed", "representation")},
                **selected,
                "fit_scope": "PCA train trials; downstream probe train split",
                "final_evaluation_split": "test",
            })

            for label, mask, split_name, scope in (
                ("held_out", test_mask, "test", "held_out_generalization"),
                ("all_valid", all_mask, "all_valid_descriptive", "full_sample_descriptive"),
            ):
                for metric, value in core._diagnostics(embedding[mask]).items():
                    _add_metric(
                        rows, base, "representation_diagnostics", metric, value,
                        evaluation_split=split_name, evaluation_scope=scope,
                        diagnostic_split=label, n_rows=int(mask.sum()),
                        n_trials=int(np.unique(data["trial_id"][mask]).size),
                        pair_sampling_seed=core.PAIR_SAMPLE_SEED,
                    )

    for representation in ("raw", "unit"):
        for scope_name, predicate, split_label, scope in (
            ("held_out", lambda x: core._test_mask({"dataset": "Real"}, x, "A_PROXIMAL"),
             "test", "held_out_generalization"),
            ("all_valid", lambda x: x["valid_mask"].astype(bool),
             "all_valid_descriptive", "full_sample_descriptive"),
        ):
            a, b = data_by_population["A_PROXIMAL"], data_by_population["B_DISTAL"]
            ai, bi = np.flatnonzero(predicate(a)), np.flatnonzero(predicate(b))
            acoords = list(zip(a["trial_id"][ai].astype(int), a["time_id"][ai].astype(int)))
            bcoords = list(zip(b["trial_id"][bi].astype(int), b["time_id"][bi].astype(int)))
            if acoords != bcoords:
                raise ValueError(f"PCA A/B {scope_name} support mismatch")
            ea, eb = a[f"embedding_{representation}"][ai], b[f"embedding_{representation}"][bi]
            metrics = {
                "procrustes_r2": core.procrustes_r2(ea, eb),
                "rsa_spearman": core.distance_geometry_correlation(
                    ea, eb, method="spearman", max_pairs=core.PAIR_COUNT),
                "rsa_pearson": core.distance_geometry_correlation(
                    ea, eb, method="pearson", max_pairs=core.PAIR_COUNT),
                "linear_cka": core.linear_cka(ea, eb),
            }
            pair_base = {
                "dataset": "Real", "architecture": "pca", "objective": "none",
                "population": "A_PROXIMAL_vs_B_DISTAL", "seed": "deterministic",
                "trial_id": "real_pca_A_PROXIMAL|real_pca_B_DISTAL",
                "fit_status": "PCA_REFERENCE_TRAIN_FIT|PCA_REFERENCE_TRAIN_FIT",
                "near_collapse": "not_applicable", "representation": representation,
                "reference": "matched_A_B", "fit_scope": "independent PCA fits on frozen train trials",
            }
            for metric, value in metrics.items():
                _add_metric(
                    rows, pair_base, "cross_population_consistency", metric, value,
                    evaluation_split=split_label, evaluation_scope=scope,
                    n_rows=len(ai), n_trials=int(np.unique(a["trial_id"][ai]).size),
                    matched_rows=len(ai), pair_sampling_seed=core.PAIR_SAMPLE_SEED,
                )

    primary = [
        row for row in rows
        if row["metric"] in PRIMARY_METRICS and row.get("evaluation_split") == "test"
    ]
    rows.sort(key=lambda r: (str(r.get("category")), str(r.get("population")),
                             str(r.get("representation")), str(r.get("evaluation_split")),
                             str(r.get("metric"))))
    primary.sort(key=lambda r: (str(r.get("category")), str(r.get("population")),
                                str(r.get("representation")), str(r.get("metric"))))
    selection_rows.sort(key=lambda r: (r["population"], r["representation"]))

    parent_hashes = {
        "pca_manifest_sha256": context["pca_manifest_sha256"],
        "core_embedding_index_sha256": sha256(CORE_INDEX),
        "evaluation_support_sha256": sha256(SUPPORT_PATH),
        "final_real_spec_sha256": sha256(SPEC_PATH),
        "channel_partition_sha256": sha256(PARTITION_PATH),
        "pca_embeddings": {
            row["population"]: row["embedding_sha256"] for row in pca_manifest["populations"]
        },
        "core_metadata_parents": context["reference_sources"],
        "core_metric_implementation_sha256": sha256(Path(core.__file__)),
    }
    artifacts = {
        "REAL_PCA_METRICS_LONG.csv": _csv_bytes(rows),
        "REAL_PCA_PRIMARY_HELD_OUT.csv": _csv_bytes(primary),
        "REAL_PCA_PROBE_SELECTION.csv": _csv_bytes(selection_rows, (
            "dataset", "architecture", "objective", "population", "seed", "representation",
            "selected_C", "condition_grid", "selected_progress_alpha", "behavior_alphas",
            "selection_split", "fit_scope", "final_evaluation_split",
        )),
    }
    report = (
        "# Real-data PCA reference results\n\n"
        "This is a separate downstream evaluation of the frozen W201 PCA reference. "
        "The PCA basis was fit on training trials only; the saved PCA embeddings were reused. "
        "No neural encoder was retrained and the frozen core metric tables were not modified.\n\n"
        "## Evaluation\n\n"
        "- Populations: TOTAL65, A_PROXIMAL (32 channels), B_DISTAL (33 channels).\n"
        "- Raw and unit-normalized embeddings are evaluated separately.\n"
        "- Held-out accessibility probes are fit on train, selected on validation, and scored on test, "
        "using the same probe grids and metric functions as the frozen core evaluator.\n"
        "- Held-out A/B consistency uses matched trial/time points and the frozen common test support.\n"
        "- All-valid diagnostics and A/B consistency are descriptive only; they are not generalization evidence.\n"
        "- No latent-ground-truth recovery is available for real data. PCA was not added to the controlled-lag "
        "model grid, whose frozen protocol covers the neural encoders.\n\n"
        "## Files\n\n"
        "- `REAL_PCA_PRIMARY_HELD_OUT.csv`: compact held-out accessibility and A/B consistency results.\n"
        "- `REAL_PCA_METRICS_LONG.csv`: all held-out and all-valid descriptive metrics.\n"
        "- `REAL_PCA_PROBE_SELECTION.csv`: validation-selected linear probe parameters.\n"
        "- `PROVENANCE.json`: source hashes, definitions, supports, and output hashes.\n\n"
        "Embedding figures already exist in `outputs/final_thesis_v1/final_evaluation/embedding_process_plots/real/` "
        "and `outputs/presentation_ready/figures/real/seed42/total_65/`; this branch adds metrics, not replacement plots.\n"
    ).encode("utf-8")
    artifacts["REAL_PCA_RESULTS.md"] = report
    METRICS_ROOT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="real_pca_metrics_", dir=METRICS_ROOT.parent) as temp_name:
        stage = Path(temp_name)
        for name, content in artifacts.items():
            with (stage / name).open("xb") as stream:
                stream.write(content)
        provenance = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "branch": "real_pca_reference/metrics",
            "scientific_scope": "downstream metrics on frozen train-fit W201 PCA embeddings",
            "protocol": {
                "populations": list(POPULATIONS), "window_size": int(pca_manifest["selected_window_size"]),
                "representations": ["raw", "unit"], "pca_refit": False,
                "neural_training": False, "neural_optimizer_steps": 0,
                "probe_C_grid": list(core.PROBE_C_GRID),
                "ridge_alpha_grid": list(core.RIDGE_ALPHA_GRID),
                "probe_selection_split": "validation", "probe_evaluation_split": "test",
                "pairwise_metric_max_pairs": core.PAIR_COUNT,
                "pair_sampling_seed": core.PAIR_SAMPLE_SEED,
                "held_out_support_rows": len(test_coords),
                "all_valid_support_rows": int(pca_manifest["support_rows"]["all"]),
                "all_valid_metrics_are_descriptive": True,
                "core_metric_tables_modified": False,
                "test_used_for_model_or_hyperparameter_selection": False,
            },
            "parent_hashes": parent_hashes,
            "source_script_sha256": sha256(Path(__file__)),
            "metric_implementation_sha256": sha256(Path(core.__file__)),
            "artifacts_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()},
            "row_counts": {"long_metrics": len(rows), "primary_held_out": len(primary),
                           "probe_selection": len(selection_rows)},
        }
        with (stage / "PROVENANCE.json").open("xb") as stream:
            stream.write((json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        stage.rename(METRICS_ROOT)
    print(json.dumps({
        "status": "CREATED", "metrics_root": str(METRICS_ROOT),
        "row_counts": {"long_metrics": len(rows), "primary_held_out": len(primary),
                       "probe_selection": len(selection_rows)},
        "primary_held_out": primary,
        "core_metrics_modified": False, "neural_training": False,
    }, indent=2, default=str))
    return provenance


if __name__ == "__main__":
    evaluate()
