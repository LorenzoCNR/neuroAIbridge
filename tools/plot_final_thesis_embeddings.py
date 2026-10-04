"""Regenerate final held-out embedding plots from frozen artifacts only.

This is a plotting-only adapter around the project's established manifold
plotters. It does not train models, alter embeddings, or calculate metrics.
"""

from __future__ import annotations

import contextlib
import argparse
import csv
import hashlib
import io
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neurobridge.viz import (
    plot_direction_averaged_embedding,
    plot_direction_averaged_embedding_raw,
)


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "outputs/final_thesis_v1/final_evaluation/core_metrics"
INDEX = CORE / "EMBEDDING_INDEX.csv"
OUTPUT = ROOT / "outputs/final_thesis_v1/final_evaluation/embedding_process_plots"
PLOTLY_JS = OUTPUT / "plotly.min.js"
SYNTHETIC_RUN = ROOT / "outputs/runs/clean_rebuild_2026-09-23_seed42_synthetic"
SYNTHETIC_SAFE = ROOT / "outputs/phase2a_hpo/safe_inputs/synthetic"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(row: dict[str, str]) -> tuple[Path, Path | None, Path]:
    embedding_path = Path(row["embedding_path"])
    manifest_path = Path(row["manifest_path"])
    if row["dataset"] == "synthetic":
        return embedding_path, None, manifest_path
    return (
        embedding_path / "embedding_raw.npz",
        embedding_path / "embedding_unit.npz",
        manifest_path,
    )


def _metadata(row: dict[str, str]) -> dict[str, np.ndarray]:
    embedding_path = Path(row["embedding_path"])
    if row["dataset"] == "synthetic":
        with np.load(embedding_path, allow_pickle=False) as values:
            return {
                key: values[key].copy()
                for key in ("trial_id", "time_id", "valid_mask", "split", "labels")
            }
    with np.load(embedding_path / "evaluation_metadata.npz", allow_pickle=False) as values:
        return {
            "trial_id": values["trial_id"].copy(),
            "time_id": values["time_id"].copy(),
            "valid_mask": values["valid_mask"].astype(bool, copy=True),
            "split": values["split"].copy(),
            "labels": values["target"].copy(),
        }


def _validate_and_common_support(
    rows: list[dict[str, str]], dataset: str, split_name: str | None = "test"
) -> set[tuple[int, int]]:
    common: set[tuple[int, int]] | None = None
    for row in rows:
        metadata = _metadata(row)
        n = len(metadata["trial_id"])
        if not all(len(values) == n for values in metadata.values()):
            raise ValueError(f"metadata cardinality mismatch: {row['trial_id']}")
        valid = metadata["valid_mask"].astype(bool)
        split = metadata["split"].astype(str)
        if split_name is None:
            if not set(np.unique(split)) <= {"train", "validation", "test"}:
                raise ValueError(f"unknown split label in {row['trial_id']}")
        else:
            valid &= split == split_name
        pairs = np.column_stack((metadata["trial_id"][valid], metadata["time_id"][valid]))
        if len(pairs) == 0 or len(np.unique(pairs, axis=0)) != len(pairs):
            raise ValueError(f"empty or duplicate valid trial/time support: {row['trial_id']}")
        current = set(map(tuple, pairs.astype(np.int64, copy=False).tolist()))
        common = current if common is None else common.intersection(current)
    if not common:
        raise ValueError(f"no common held-out support for {dataset}")
    return common


def _check_hashes(row: dict[str, str]) -> None:
    embedding_path, unit_path, manifest_path = _paths(row)
    if sha256(manifest_path) != row["manifest_sha256"]:
        raise ValueError(f"manifest hash mismatch: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if row["dataset"] == "synthetic":
        if sha256(embedding_path) != row["embedding_sha256"]:
            raise ValueError(f"synthetic embedding hash mismatch: {embedding_path}")
    else:
        for name, path in (
            ("embedding_raw.npz", embedding_path),
            ("embedding_unit.npz", unit_path),
            ("evaluation_metadata.npz", Path(row["embedding_path"]) / "evaluation_metadata.npz"),
        ):
            if sha256(path) != manifest["artifact_sha256"][name]:
                raise ValueError(f"real artifact hash mismatch: {path}")


def _load_representation(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    embedding_path, unit_path, _ = _paths(row)
    if row["dataset"] == "synthetic":
        with np.load(embedding_path, allow_pickle=False) as values:
            raw, unit = values["embedding_raw"].copy(), values["embedding_unit"].copy()
    else:
        with np.load(embedding_path, allow_pickle=False) as values:
            raw = values["embedding_raw"].copy()
        with np.load(unit_path, allow_pickle=False) as values:
            unit = values["embedding_unit"].copy()
    if raw.ndim != 2 or unit.ndim != 2 or raw.shape != unit.shape or raw.shape[1] != 3:
        raise ValueError(f"raw/unit embeddings must match N x 3: {row['trial_id']}")
    if not (np.isfinite(raw).all() and np.isfinite(unit).all()):
        raise ValueError(f"non-finite embedding value: {row['trial_id']}")
    return raw, unit


def _support_mask(
    metadata: dict[str, np.ndarray], common: set[tuple[int, int]], split_name: str | None = "test"
) -> np.ndarray:
    base = metadata["valid_mask"].astype(bool)
    if split_name is not None:
        base &= metadata["split"].astype(str) == split_name
    out = np.zeros(len(base), dtype=bool)
    indices = np.flatnonzero(base)
    pairs = zip(metadata["trial_id"][indices].tolist(), metadata["time_id"][indices].tolist())
    out[indices] = np.fromiter((pair in common for pair in pairs), dtype=bool, count=len(indices))
    return out


def _plot_2d_legacy_style(
    embedding: np.ndarray,
    labels: np.ndarray,
    trial_id: np.ndarray,
    time_id: np.ndarray,
    *,
    destination: Path,
    title: str,
) -> None:
    """The static scatter/trajectory panel used by the existing staged plots."""
    cmap = plt.get_cmap("hsv")
    norm = plt.Normalize(vmin=0, vmax=7)
    fig, axis = plt.subplots(figsize=(7, 5))
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.12, top=0.88)
    zero_based = labels.astype(int) - 1
    axis.scatter(
        embedding[:, 0], embedding[:, 1], c=zero_based,
        s=1, alpha=0.08, cmap=cmap, norm=norm,
    )
    axis.set_title(title)
    axis.set_xlabel("Embedding 1")
    axis.set_ylabel("Embedding 2")
    axis.set_aspect("equal", adjustable="box")
    for condition in range(8):
        condition_mask = zero_based == condition
        times = np.unique(time_id[condition_mask])
        trajectory = np.asarray([
            embedding[condition_mask & (time_id == time)].mean(axis=0)
            for time in times
        ])
        if len(trajectory) < 2:
            continue
        axis.plot(
            trajectory[:, 0], trajectory[:, 1],
            color=cmap(condition / 8.0), linewidth=1.5, alpha=0.9,
        )
        axis.scatter(*trajectory[0, :2], color="maroon", marker="o", s=22, zorder=3)
        axis.scatter(*trajectory[-1, :2], color="black", marker="x", s=28, zorder=3)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _write_views(
    raw: np.ndarray,
    unit: np.ndarray,
    metadata: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    folder: Path,
    label: str,
    stem: str,
    index_rows: list[dict[str, object]],
    dataset: str,
    population: str,
    architecture: str,
    objective: str,
    seed: str,
    status: str,
    branch: str = "held_out_test",
    branch_label: str = "held-out test",
    category: str = "learned_embedding",
    reuse_png: bool = False,
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    labels = metadata["labels"].astype(int) + 1
    trial_id, time_id = metadata["trial_id"], metadata["time_id"]
    selected_labels = labels[mask]
    selected_trials, selected_times = trial_id[mask], time_id[mask]
    if len(selected_labels) != int(mask.sum()) or len(selected_labels) == 0:
        raise ValueError(f"invalid plotting support for {stem}")
    for view, embedding in (("unit_sphere", unit), ("raw", raw)):
        view_folder = folder / view
        view_folder.mkdir(parents=True, exist_ok=True)
        png = view_folder / ("embedding.png" if view == "unit_sphere" else "embedding_raw.png")
        html = view_folder / (
            "trajectory_sphere.html" if view == "unit_sphere"
            else "trajectory_raw.html"
        )
        if not reuse_png or not png.is_file():
            _plot_2d_legacy_style(
                embedding[mask], selected_labels, selected_trials, selected_times,
                destination=png,
                title=f"{label} ({'unit' if view == 'unit_sphere' else 'raw'}, {branch_label})",
            )
        plotter = (
            plot_direction_averaged_embedding
            if view == "unit_sphere"
            else plot_direction_averaged_embedding_raw
        )
        with contextlib.redirect_stdout(io.StringIO()):
            fig = plotter(
                embedding,
                labels,
                original_label_order=np.sort(np.unique(selected_labels)),
                c_s="maroon",
                output_folder=view_folder,
                name=html.name,
                quiescent_label=None,
                show=False,
                trial_id=trial_id,
                time_id=time_id,
                valid_mask=mask,
            )
        local_js = Path(os.path.relpath(PLOTLY_JS, html.parent)).as_posix()
        fig.write_html(html, include_plotlyjs=local_js, full_html=True, auto_open=False)
        for path, file_type in ((png, "png_2d_legacy_style"), (html, "html_3d")):
            index_rows.append({
                "dataset": dataset,
                "branch": branch,
                "category": category,
                "representation": view,
                "population": population,
                "architecture": architecture,
                "objective": objective,
                "seed": seed,
                "fit_status": status,
                "support_rows": int(mask.sum()),
                "file_type": file_type,
                "file": path.relative_to(OUTPUT).as_posix(),
            })


def _plot_original_synthetic_process(
    common: set[tuple[int, int]], index_rows: list[dict[str, object]], *,
    split_name: str | None = "test", branch_folder: str = "held_out_test",
    branch: str = "held_out_test",
) -> dict[str, str]:
    data_path = SYNTHETIC_RUN / "stage01_data/shared_data.npz"
    windows_a_path = SYNTHETIC_RUN / "stage02_windows/windows_A.npz"
    windows_b_path = SYNTHETIC_RUN / "stage02_windows/windows_B.npz"
    split_path = SYNTHETIC_RUN / "stage02_windows/split.json"
    expected_a = json.loads((SYNTHETIC_SAFE / "A/manifest.json").read_text(encoding="utf-8"))
    expected_b = json.loads((SYNTHETIC_SAFE / "B/manifest.json").read_text(encoding="utf-8"))
    expected_hashes = {
        "shared_data.npz": expected_a["raw_source_sha256"],
        "windows_A.npz": expected_a["parent_windows_sha256"],
        "windows_B.npz": expected_b["parent_windows_sha256"],
        "split.json": expected_a["original_split_sha256"],
    }
    actual_hashes = {
        "shared_data.npz": sha256(data_path),
        "windows_A.npz": sha256(windows_a_path),
        "windows_B.npz": sha256(windows_b_path),
        "split.json": sha256(split_path),
    }
    if actual_hashes != expected_hashes:
        raise ValueError(f"Synthetic original-process source hashes differ: {actual_hashes}")

    with np.load(data_path, allow_pickle=False) as data:
        z_shared = data["Z_shared"].copy()
        source_labels = data["labels"].copy()
    with np.load(windows_a_path, allow_pickle=False) as a, np.load(windows_b_path, allow_pickle=False) as b:
        for key in ("trial_id", "time_id", "labels", "lag_valid"):
            if a[key].shape != b[key].shape:
                raise ValueError(f"Synthetic A/B source metadata mismatch: {key}")
        for key in ("trial_id", "time_id", "labels"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f"Synthetic A/B source coordinates differ: {key}")
        trial_id = a["trial_id"].astype(int, copy=True)
        time_id = a["time_id"].astype(int, copy=True)
        labels = a["labels"].astype(int, copy=True)
        source_valid = a["lag_valid"].astype(bool) & b["lag_valid"].astype(bool)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split_name is not None:
        source_valid &= np.isin(trial_id, np.asarray(split[split_name], dtype=int))
    source_pairs = set(map(tuple, np.column_stack((trial_id[source_valid], time_id[source_valid])).tolist()))
    if source_pairs != common:
        raise ValueError(
            "Synthetic embedding support does not match the frozen shared-latent A/B test support"
        )
    if not np.array_equal(labels, source_labels[trial_id]):
        raise ValueError("Synthetic labels do not match the frozen source process")
    z_rows = z_shared[trial_id, time_id]
    if z_rows.shape != (len(trial_id), 3) or not np.isfinite(z_rows[source_valid]).all():
        raise ValueError("Invalid original Z_shared values on plotting support")

    folder = OUTPUT / "synthetic" / branch_folder / "original_process/Z_shared_pre_shift"
    folder.mkdir(parents=True, exist_ok=True)
    original_labels = labels + 1
    for view, plotter, suffix in (
        ("unit_sphere", plot_direction_averaged_embedding, "sphere"),
        ("raw", plot_direction_averaged_embedding_raw, "trajectories_raw"),
    ):
        name = f"trajectory_{suffix}.html"
        with contextlib.redirect_stdout(io.StringIO()):
            fig = plotter(
                z_rows,
                original_labels,
                original_label_order=np.sort(np.unique(original_labels[source_valid])),
                c_s="maroon",
                output_folder=folder,
                name=name,
                quiescent_label=None,
                show=False,
                trial_id=trial_id,
                time_id=time_id,
                valid_mask=source_valid,
            )
        local_js = Path(os.path.relpath(PLOTLY_JS, folder)).as_posix()
        fig.write_html(folder / name, include_plotlyjs=local_js, full_html=True, auto_open=False)
        index_rows.append({
            "dataset": "synthetic",
            "branch": branch,
            "category": "original_latent_process_Z_only",
            "representation": view,
            "population": "shared_latent",
            "architecture": "ground_truth",
            "objective": "not_applicable",
            "seed": "generator_42",
            "fit_status": "GROUND_TRUTH_REFERENCE",
            "support_rows": int(source_valid.sum()),
            "file_type": "html_3d",
            "file": (folder / name).relative_to(OUTPUT).as_posix(),
        })
    return actual_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse existing PNGs in this plotting output folder and rebuild HTML/index/provenance.",
    )
    parser.add_argument(
        "--all-sample-only",
        action="store_true",
        help=("Create a separate all-valid-rows descriptive projection from the already saved "
              "embeddings; do not alter the held-out plots or their index/provenance."),
    )
    args = parser.parse_args()
    if args.all_sample_only:
        if not INDEX.is_file():
            raise FileNotFoundError(INDEX)
        rows = list(csv.DictReader(INDEX.open(encoding="utf-8", newline="")))
        rows = [row for row in rows if row["dataset"] in {"synthetic", "real"}]
        if not rows:
            raise ValueError("No final embeddings are indexed")
        for row in rows:
            _check_hashes(row)
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row["dataset"]].append(row)
        common_support = {
            dataset: _validate_and_common_support(dataset_rows, dataset, split_name=None)
            for dataset, dataset_rows in grouped.items()
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        if not PLOTLY_JS.is_file():
            from plotly.offline import get_plotlyjs

            PLOTLY_JS.write_text(get_plotlyjs(), encoding="utf-8")
        index_rows: list[dict[str, object]] = []
        for number, row in enumerate(rows, start=1):
            metadata = _metadata(row)
            mask = _support_mask(metadata, common_support[row["dataset"]], split_name=None)
            raw, unit = _load_representation(row)
            if len(raw) != len(mask) or len(metadata["labels"]) != len(mask):
                raise ValueError(f"Embedding/metadata cardinality mismatch: {row['trial_id']}")
            pop = row["population"]
            arch, objective, seed = row["architecture"], row["objective"], row["seed"]
            population_folder = {
                "A": "A", "B": "B", "TOTAL65": "T65",
                "A_PROXIMAL": "Aprox", "B_DISTAL": "Bdistal",
            }.get(pop, pop)
            architecture_folder = {"cnn1d": "cnn", "transformer": "tr", "pca": "pca"}.get(arch, arch)
            objective_folder = {
                "soft": "soft", "infonce": "infonce",
                "time_contrastive_blocks": "time",
                "behavior_contrastive_blocks": "behavior", "none": "none",
            }.get(objective, objective)
            folder = (
                OUTPUT / row["dataset"] / "all" / population_folder
                / architecture_folder / objective_folder / f"s{seed}"
            )
            _write_views(
                raw, unit, metadata, mask,
                folder=folder,
                label=f"{row['dataset'].capitalize()} {pop} | {arch} / {objective} / s{seed}",
                stem=f"{pop}_{arch}_{objective}_seed{seed}",
                index_rows=index_rows,
                dataset=row["dataset"], population=pop, architecture=arch,
                objective=objective, seed=seed, status=row["fit_status"],
                branch="all_valid_rows_projection",
                branch_label="all rows",
            )
            if number % 10 == 0 or number == len(rows):
                print(f"Plotted all valid rows {number}/{len(rows)} frozen embeddings", flush=True)

        if "synthetic" not in common_support:
            raise ValueError("Synthetic embeddings are required for the original Z plot")
        latent_source_hashes = _plot_original_synthetic_process(
            common_support["synthetic"], index_rows, split_name=None,
            branch_folder="all", branch="all_valid_rows_projection",
        )
        index_path = OUTPUT / "PLOT_INDEX_ALL.csv"
        columns = [
            "dataset", "branch", "category", "representation", "population",
            "architecture", "objective", "seed", "fit_status", "support_rows",
            "file_type", "file",
        ]
        with index_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(index_rows)
        output_files = [
            path for dataset in ("synthetic", "real")
            for path in (OUTPUT / dataset / "all").rglob("*") if path.is_file()
        ]
        source_hashes = {
            "embedding_index_sha256": sha256(INDEX),
            "manifold_plots_source_sha256": sha256(ROOT / "src/neurobridge/viz/manifold_plots.py"),
            "staged_shared_latent_source_sha256": sha256(ROOT / "src/neurobridge/experiments/staged_shared_latent.py"),
            "real_monkey_plot_source_sha256": sha256(ROOT / "src/neurobridge/experiments/real_monkey.py"),
            "synthetic_original_process_sources": latent_source_hashes,
        }
        provenance = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "output_root": str(OUTPUT.relative_to(ROOT)),
            "branch": "all_valid_rows_projection",
            "input_embedding_rows": len(rows),
            "plots_per_embedding": 4,
            "plot_files": len(index_rows),
            "plot_index": str(index_path.relative_to(ROOT)),
            "support_policy": (
                "valid trial/time pairs common across all indexed models/populations; "
                "includes train, validation, and test rows"
            ),
            "interpretation": (
                "descriptive projection of all valid rows using the already frozen "
                "held-out-trained checkpoints; this is not a full-sample refit"
            ),
            "synthetic_original_reference": "Z_shared before imposed A/B shift; M and eta are not plotted",
            "real_original_latent": "not applicable; real data have no known latent ground truth",
            "plot_functions": [
                "neurobridge.viz.manifold_plots.plot_direction_averaged_embedding",
                "neurobridge.viz.manifold_plots.plot_direction_averaged_embedding_raw",
                "legacy 2D scatter/trajectory layout from staged_shared_latent.stage_plot and real_monkey._plot_shared_embedding_views",
            ],
            "common_all_valid_support_pairs": {
                dataset: len(pairs) for dataset, pairs in common_support.items()
            },
            "original_Z_common_support_pairs": len(common_support["synthetic"]),
            "training_performed": False,
            "optimizer_steps": 0,
            "embeddings_modified": False,
            "metrics_modified": False,
            "source_hashes": source_hashes,
            "outputs_sha256": {
                path.relative_to(OUTPUT).as_posix(): sha256(path)
                for path in output_files
            },
            "plot_index_sha256": sha256(index_path),
        }
        provenance_path = OUTPUT / "PLOT_PROVENANCE_ALL.json"
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print(f"Completed all-sample projection: {len(index_rows)} plot files")
        print(f"Output: {OUTPUT / 'synthetic' / 'all'} and {OUTPUT / 'real' / 'all'}")
        print(f"Support pairs: {provenance['common_all_valid_support_pairs']}")
        return
    if OUTPUT.exists() and not args.resume_existing:
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {OUTPUT}"
        )
    if args.resume_existing and not any(OUTPUT.rglob("*.png")):
        raise FileNotFoundError("--resume-existing requires previously generated PNGs")
    if not INDEX.is_file():
        raise FileNotFoundError(INDEX)
    rows = list(csv.DictReader(INDEX.open(encoding="utf-8", newline="")))
    rows = [row for row in rows if row["dataset"] in {"synthetic", "real"}]
    if not rows:
        raise ValueError("No final embeddings are indexed")
    for row in rows:
        _check_hashes(row)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    common_support = {
        dataset: _validate_and_common_support(dataset_rows, dataset)
        for dataset, dataset_rows in grouped.items()
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not PLOTLY_JS.is_file():
        from plotly.offline import get_plotlyjs

        PLOTLY_JS.write_text(get_plotlyjs(), encoding="utf-8")
    index_rows: list[dict[str, object]] = []
    for number, row in enumerate(rows, start=1):
        metadata = _metadata(row)
        mask = _support_mask(metadata, common_support[row["dataset"]])
        raw, unit = _load_representation(row)
        if len(raw) != len(mask) or len(metadata["labels"]) != len(mask):
            raise ValueError(f"Embedding/metadata cardinality mismatch: {row['trial_id']}")
        pop = row["population"]
        arch, objective, seed = row["architecture"], row["objective"], row["seed"]
        population_folder = {
            "A": "A", "B": "B", "TOTAL65": "T65",
            "A_PROXIMAL": "Aprox", "B_DISTAL": "Bdistal",
        }.get(pop, pop)
        architecture_folder = {"cnn1d": "cnn", "transformer": "tr", "pca": "pca"}.get(arch, arch)
        objective_folder = {
            "soft": "soft", "infonce": "infonce",
            "time_contrastive_blocks": "time",
            "behavior_contrastive_blocks": "behavior", "none": "none",
        }.get(objective, objective)
        folder = (
            OUTPUT / row["dataset"] / "test" / population_folder
            / architecture_folder / objective_folder / f"s{seed}"
        )
        stem = f"{pop}_{arch}_{objective}_seed{seed}"
        _write_views(
            raw, unit, metadata, mask,
            folder=folder,
            label=f"{row['dataset'].capitalize()} {pop} | {arch} / {objective} / seed {seed}",
            stem=stem,
            index_rows=index_rows,
            dataset=row["dataset"],
            population=pop,
            architecture=arch,
            objective=objective,
            seed=seed,
            status=row["fit_status"],
            reuse_png=args.resume_existing,
        )
        if number % 10 == 0 or number == len(rows):
            print(f"Plotted {number}/{len(rows)} frozen embeddings")

    latent_source_hashes = None
    if "synthetic" not in common_support:
        raise ValueError("Synthetic embeddings are required for the original Z plot")
    latent_source_hashes = _plot_original_synthetic_process(
        common_support["synthetic"], index_rows
    )

    index_path = OUTPUT / "PLOT_INDEX.csv"
    columns = [
        "dataset", "branch", "category", "representation", "population",
        "architecture", "objective", "seed", "fit_status", "support_rows",
        "file_type", "file",
    ]
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(index_rows)

    source_hashes = {
        "embedding_index_sha256": sha256(INDEX),
        "manifold_plots_source_sha256": sha256(ROOT / "src/neurobridge/viz/manifold_plots.py"),
        "staged_shared_latent_source_sha256": sha256(ROOT / "src/neurobridge/experiments/staged_shared_latent.py"),
        "real_monkey_plot_source_sha256": sha256(ROOT / "src/neurobridge/experiments/real_monkey.py"),
        "synthetic_original_process_sources": latent_source_hashes,
    }
    output_files = [path for path in OUTPUT.rglob("*") if path.is_file()]
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(OUTPUT.relative_to(ROOT)),
        "input_embedding_rows": len(rows),
        "plots_per_embedding": 4,
        "plot_files": sum(1 for row in index_rows),
        "plot_index": str(index_path.relative_to(ROOT)),
        "support_policy": "valid held-out test trial/time pairs common across every indexed model/population within each dataset",
        "synthetic_original_reference": "Z_shared before imposed A/B shift; M and eta are not plotted",
        "real_original_latent": "not applicable; real data have no known latent ground truth",
        "plot_functions": [
            "neurobridge.viz.manifold_plots.plot_direction_averaged_embedding",
            "neurobridge.viz.manifold_plots.plot_direction_averaged_embedding_raw",
            "legacy 2D scatter/trajectory layout from staged_shared_latent.stage_plot and real_monkey._plot_shared_embedding_views",
        ],
        "common_test_support_pairs": {
            dataset: len(pairs) for dataset, pairs in common_support.items()
        },
        "original_Z_common_support_pairs": len(common_support["synthetic"]),
        "training_performed": False,
        "optimizer_steps": 0,
        "embeddings_modified": False,
        "metrics_modified": False,
        "plotly_javascript": "one shared local plotly.min.js file; not duplicated inside each HTML",
        "source_hashes": source_hashes,
        "outputs_sha256": {
            path.relative_to(OUTPUT).as_posix(): sha256(path)
            for path in output_files
        },
        "total_output_file_count_including_index_and_provenance": len(output_files) + 1,
    }
    provenance_path = OUTPUT / "PLOT_PROVENANCE.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Completed: {len(index_rows)} plot files from {len(rows)} frozen embeddings")
    print(f"Output: {OUTPUT}")
    print(f"Support pairs: {provenance['common_test_support_pairs']}")


if __name__ == "__main__":
    main()
