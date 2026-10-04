"""Controlled B-only digital-delay evaluation using frozen REAL-0 models.

This module never fits a model. It builds a separately cached +10-bin B input,
transforms it with the corresponding REAL-0 B checkpoint, and compares lag
curves on one strict trial/time support shared by REAL-0 and REAL-10.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from neurobridge.eval.representation import lagged_alignment_by_trial_time
from neurobridge.experiments import real_monkey as legacy
from neurobridge.experiments.lag_shuffle import LAGS
from neurobridge.experiments.real_monkey_validated import (
    MODEL_NAMES,
    OBJECTIVE_NAMES,
    _hash,
    _json_write,
)
from neurobridge.train.loop import encode_windows


_CHECK_FIELDS = (
    "trial_id", "time_id", "global_time_id", "target", "position",
    "velocity", "progress", "split",
)


def _config_hash(config: legacy.RealMonkeyConfig) -> str:
    payload = json.dumps(legacy._public_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_matching_metadata(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], label: str) -> None:
    for key in _CHECK_FIELDS:
        if key not in reference or key not in candidate:
            raise ValueError(f"{label} is missing aligned metadata field {key!r}")
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"{label} metadata differ at {key!r}; refusing lag comparison")


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def _validate_embedding(values: dict[str, np.ndarray], valid: np.ndarray, label: str) -> None:
    n = len(values.get("embedding_unit", ()))
    if values.get("embedding_raw", np.empty((0,))).shape != (n, 3):
        raise ValueError(f"{label}: raw embedding must be N x 3")
    if values.get("embedding_unit", np.empty((0,))).shape != (n, 3):
        raise ValueError(f"{label}: unit embedding must be N x 3")
    for key in _CHECK_FIELDS:
        if key not in values or len(values[key]) != n:
            raise ValueError(f"{label}: {key} cardinality differs from embedding rows")
    if len(valid) != n:
        raise ValueError(f"{label}: validity mask cardinality differs from embedding rows")


def _transform_frozen_checkpoint(
    project_root: Path,
    shifted_config: legacy.RealMonkeyConfig,
    *,
    model_name: str,
    objective_name: str,
    branch: str,
    checkpoint: Path,
    windows_path: Path,
) -> Path:
    """Transform shifted windows with an R0 checkpoint; cache by parent hashes."""
    root = legacy._run_root(project_root, shifted_config)
    output_dir = root / "stage04_embeddings" / branch
    output = output_dir / f"{model_name}_{objective_name}.npz"
    metadata_path = output_dir / f"{model_name}_{objective_name}_metadata.json"
    parents = {"checkpoint_sha256": _hash(checkpoint), "windows_sha256": _hash(windows_path)}
    expected = {
        "config": legacy._public_config(shifted_config),
        "config_sha256": _config_hash(shifted_config),
        "setting": "real_10",
        "population": "B",
        "architecture": model_name,
        "objective": objective_name,
        "training_seed": shifted_config.training_seed,
        "channel_split_seed": shifted_config.channel_split_seed,
        "branch": branch,
        "imposed_shift_bins": 10,
        "fitted": False,
        "parent_artifacts": parents,
    }
    if output.exists() or metadata_path.exists():
        if not output.is_file() or not metadata_path.is_file():
            raise FileExistsError(f"partial controlled-lag transform exists; preserve it: {output_dir}")
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(cached.get(key) == value for key, value in expected.items()) and cached.get("artifact_sha256", {}).get(output.name) == _hash(output):
            return output
        raise FileExistsError(f"controlled-lag cache has different parents/configuration; preserve it: {output}")

    _, values = legacy._load_real_window_dataset(windows_path)
    checkpoint_values = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        checkpoint_values.get("model_name") != model_name
        or checkpoint_values.get("objective") != objective_name
        or checkpoint_values.get("branch") != branch
    ):
        raise ValueError(f"R0 checkpoint identity mismatch: {checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_model = legacy._make_real_model(model_name, int(values["X_windows"].shape[-1]), shifted_config, normalize=False)
    unit_model = legacy._make_real_model(model_name, int(values["X_windows"].shape[-1]), shifted_config, normalize=True)
    raw_model.load_state_dict(checkpoint_values["state_dict"])
    unit_model.load_state_dict(checkpoint_values["state_dict"])
    loader = DataLoader(
        legacy._load_real_window_dataset(windows_path)[0],
        batch_size=max(shifted_config.batch_size, 512),
        shuffle=False,
    )
    start = time.perf_counter()
    raw, _ = encode_windows(raw_model, loader, device=device)
    unit, _ = encode_windows(unit_model, loader, device=device)
    encoding_seconds = time.perf_counter() - start
    arrays = {
        "embedding_raw": raw.numpy().astype(np.float32),
        "embedding_unit": unit.numpy().astype(np.float32),
        "trial_id": values["trial_id"],
        "time_id": values["time_id"],
        "global_time_id": values["global_time_id"],
        "target": values["labels"],
        "position": values["position"],
        "velocity": values["velocity"],
        "progress": values["progress"],
        "split": values["split"],
        "valid_mask": values["lag_valid"].astype(bool),
    }
    _validate_embedding(arrays, arrays["valid_mask"], str(output))
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    _json_write(metadata_path, {
        **expected,
        "checkpoint_path": str(checkpoint.relative_to(project_root)),
        "windows_path": str(windows_path.relative_to(project_root)),
        "embedding_shape": list(arrays["embedding_raw"].shape),
        "artifact_sha256": {output.name: _hash(output)},
        "parent_artifact_ids": parents,
        "encoding_seconds": encoding_seconds,
        "inference_windows_per_second": len(arrays["embedding_raw"]) / max(encoding_seconds, 1e-12),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "normalization": "raw and row-wise unit embeddings transformed from the same frozen R0 checkpoint",
    })
    return output


def _coordinate_lookup(values: dict[str, np.ndarray], valid: np.ndarray, branch: str) -> dict[tuple[int, int], int]:
    split = values["split"]
    keep = np.asarray(valid, dtype=bool)
    if branch == "held_out":
        keep &= split == "test"
    return {
        (int(values["trial_id"][index]), int(values["time_id"][index])): int(index)
        for index in np.flatnonzero(keep)
    }


def _lag_curves_on_joint_support(
    a: dict[str, np.ndarray],
    b_r0: dict[str, np.ndarray],
    b_r10: dict[str, np.ndarray],
    valid_a: np.ndarray,
    valid_b_r0: np.ndarray,
    valid_b_r10: np.ndarray,
    *,
    branch: str,
) -> tuple[dict[int, float], dict[int, float], int]:
    """Compare R0/R10 on identical reference rows and equal pairs per lag."""
    a_lookup = _coordinate_lookup(a, valid_a, branch)
    b0_lookup = _coordinate_lookup(b_r0, valid_b_r0, branch)
    b10_lookup = _coordinate_lookup(b_r10, valid_b_r10, branch)
    common_reference = [
        coordinate for coordinate in a_lookup
        if all(
            (coordinate[0], coordinate[1] + lag) in b0_lookup
            and (coordinate[0], coordinate[1] + lag) in b10_lookup
            for lag in LAGS
        )
    ]
    if len(common_reference) < 3:
        raise ValueError(f"only {len(common_reference)} joint valid reference rows for lag scan")
    def score_curve(b: dict[str, np.ndarray], lookup: dict[tuple[int, int], int]) -> dict[int, float]:
        idx_a = [a_lookup[coordinate] for coordinate in common_reference]
        b_coordinates = list(lookup)
        idx_b = [lookup[coordinate] for coordinate in b_coordinates]
        _, scores, pairs = lagged_alignment_by_trial_time(
            a["embedding_unit"][idx_a],
            b["embedding_unit"][idx_b],
            np.asarray([c[0] for c in common_reference]),
            np.asarray([c[1] for c in common_reference]),
            np.asarray([c[0] for c in b_coordinates]),
            np.asarray([c[1] for c in b_coordinates]),
            LAGS,
            common_support=True,
        )
        if any(len(pairs[int(lag)][0]) != len(common_reference) for lag in LAGS):
            raise RuntimeError("strict common support produced unequal lag comparison counts")
        return {int(lag): float(score) for lag, score in scores.items()}

    return score_curve(b_r0, b0_lookup), score_curve(b_r10, b10_lookup), len(common_reference)


def _curve_summary(scores: dict[int, float]) -> dict[str, Any]:
    ordered = sorted((lag, score) for lag, score in scores.items() if np.isfinite(score))
    if not ordered:
        raise ValueError("lag curve has no finite scores")
    best_lag, best_score = max(ordered, key=lambda item: item[1])
    alternatives = [(lag, score) for lag, score in ordered if lag != best_lag]
    second_lag, second_score = max(alternatives, key=lambda item: item[1])
    by_lag = dict(ordered)
    adjacent = [by_lag[lag] for lag in (best_lag - 1, best_lag + 1) if lag in by_lag]
    floor = min(score for _, score in ordered)
    half_prominence = floor + 0.5 * (best_score - floor)
    above = {lag for lag, score in ordered if score >= half_prominence}
    left = right = best_lag
    while left - 1 in above:
        left -= 1
    while right + 1 in above:
        right += 1
    return {
        "best_lag_bins": int(best_lag),
        "best_procrustes_r2": float(best_score),
        "second_best_lag_bins": int(second_lag),
        "second_best_procrustes_r2": float(second_score),
        "peak_margin_r2": float(best_score - second_score),
        "local_sharpness_r2": float(best_score - max(adjacent)) if adjacent else float("nan"),
        "half_prominence_width_bins": int(right - left + 1),
        "boundary_peak": bool(best_lag in {min(LAGS), max(LAGS)}),
    }


def run_controlled_lag_evaluation(
    project_root: str | Path,
    *,
    seed: int,
    config_a: legacy.RealMonkeyConfig,
    config_b: legacy.RealMonkeyConfig,
    run_label: str,
) -> dict[str, Path]:
    """Evaluate unshifted R0 and B-only +10-bin R10 using frozen R0 models."""
    project_root = Path(project_root).resolve()
    if config_a.setting_name != "real_0" or config_b.setting_name != "real_0":
        raise ValueError("controlled lag requires REAL-0 source configs")
    if config_a.training_seed != seed or config_b.training_seed != seed:
        raise ValueError("R0 configs must match the requested training seed")
    if config_a.channel_indices is None or config_b.channel_indices is None:
        raise ValueError("controlled pseudo-population lag requires explicit A/B channel indices")
    if set(config_a.channel_indices).intersection(config_b.channel_indices):
        raise ValueError("A/B channel partition overlaps")

    shifted_config = replace(
        config_b,
        run_label=run_label,
        setting_name="real_10",
        population_name="B",
        imposed_shift_bins=10,
    )
    shifted_windows = legacy.stage_windows(project_root, shifted_config)
    shifted_data = np.load(shifted_windows, allow_pickle=False)
    shifted_valid = shifted_data["lag_valid"].astype(bool)
    eval_root = legacy._run_root(project_root, shifted_config)
    metrics_root = eval_root / "stage05_metrics"
    figure_root = eval_root / "stage06_figures"
    metrics_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    curves_path = metrics_root / "lag_curves_r0_vs_r10.csv"
    summary_path = metrics_root / "lag_recovery_summary.csv"
    provenance_path = metrics_root / "lag_recovery_provenance.json"
    rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    plotted: dict[str, list[tuple[str, dict[int, float], dict[int, float]]]] = {
        "held_out": [], "full_sample": [],
    }

    shifted_split = json.loads((eval_root / "stage01_data" / "split.json").read_text(encoding="utf-8"))
    r0_a_split_path = legacy._run_root(project_root, config_a) / "stage01_data" / "split.json"
    r0_b_split_path = legacy._run_root(project_root, config_b) / "stage01_data" / "split.json"
    split_a = json.loads(r0_a_split_path.read_text(encoding="utf-8"))
    split_b = json.loads(r0_b_split_path.read_text(encoding="utf-8"))
    if split_a != split_b or split_a != shifted_split:
        raise ValueError("R0 A, R0 B, and R10 B must use the exact same trial split")

    parent_artifacts: dict[str, str] = {
        "raw_source_sha256": _hash(project_root / config_b.source_relative),
        "r0_a_split_sha256": _hash(r0_a_split_path),
        "r0_b_split_sha256": _hash(r0_b_split_path),
        "r10_b_windows_sha256": _hash(shifted_windows),
    }
    for model_name in MODEL_NAMES:
        for objective_name in OBJECTIVE_NAMES:
            panel_label = f"{model_name}/{objective_name}"
            for branch in ("held_out", "full_sample"):
                root_a = legacy._run_root(project_root, config_a)
                root_b = legacy._run_root(project_root, config_b)
                embedding_a_path = root_a / "stage04_embeddings" / branch / f"{model_name}_{objective_name}.npz"
                embedding_b_path = root_b / "stage04_embeddings" / branch / f"{model_name}_{objective_name}.npz"
                checkpoint_b = root_b / "stage03_models" / branch / f"{model_name}_{objective_name}" / "model.pt"
                for path in (embedding_a_path, embedding_b_path, checkpoint_b):
                    if not path.is_file():
                        raise FileNotFoundError(f"required REAL-0 artifact is missing: {path}")
                shifted_embedding_path = _transform_frozen_checkpoint(
                    project_root, shifted_config,
                    model_name=model_name,
                    objective_name=objective_name,
                    branch=branch,
                    checkpoint=checkpoint_b,
                    windows_path=shifted_windows,
                )
                a, b0, b10 = map(_read_npz, (embedding_a_path, embedding_b_path, shifted_embedding_path))
                _require_matching_metadata(a, b0, "REAL-0 A/B")
                for field in _CHECK_FIELDS:
                    if not np.array_equal(b0[field], b10[field]):
                        raise ValueError(f"REAL-0 B and REAL-10 B metadata differ at {field!r}")
                windows_a = np.load(legacy._run_root(project_root, config_a) / "stage02_windows" / "windows.npz", allow_pickle=False)
                windows_b = np.load(legacy._run_root(project_root, config_b) / "stage02_windows" / "windows.npz", allow_pickle=False)
                valid_a = windows_a["lag_valid"].astype(bool)
                valid_b0 = windows_b["lag_valid"].astype(bool)
                _validate_embedding(a, valid_a, str(embedding_a_path))
                _validate_embedding(b0, valid_b0, str(embedding_b_path))
                _validate_embedding(b10, shifted_valid, str(shifted_embedding_path))
                curve_r0, curve_r10, n_common = _lag_curves_on_joint_support(
                    a, b0, b10, valid_a, valid_b0, shifted_valid, branch=branch,
                )
                summary_r0 = _curve_summary(curve_r0)
                summary_r10 = _curve_summary(curve_r10)
                for condition, curve in (("REAL-0", curve_r0), ("REAL-10_B_shifted", curve_r10)):
                    for lag, score in curve.items():
                        curve_rows.append({
                            "training_seed": seed, "setting": condition,
                            "architecture": model_name, "objective": objective_name,
                            "branch": branch, "lag_bins": lag,
                            "procrustes_r2": score, "n_comparisons": n_common,
                        })
                rows.append({
                    "training_seed": seed, "architecture": model_name,
                    "objective": objective_name, "branch": branch,
                    "n_comparisons_per_lag": n_common,
                    "l_hat_r0_bins": summary_r0["best_lag_bins"],
                    "l_hat_r10_bins": summary_r10["best_lag_bins"],
                    "delta_l_hat_bins": summary_r10["best_lag_bins"] - summary_r0["best_lag_bins"],
                    "target_delta_bins": 10,
                    "delta_absolute_error_bins": abs(summary_r10["best_lag_bins"] - summary_r0["best_lag_bins"] - 10),
                    **{f"r0_{key}": value for key, value in summary_r0.items()},
                    **{f"r10_{key}": value for key, value in summary_r10.items()},
                })
                plotted[branch].append((panel_label, curve_r0, curve_r10))
                parent_artifacts[f"{branch}/{model_name}/{objective_name}/r0_a_embedding_sha256"] = _hash(embedding_a_path)
                parent_artifacts[f"{branch}/{model_name}/{objective_name}/r0_b_embedding_sha256"] = _hash(embedding_b_path)
                parent_artifacts[f"{branch}/{model_name}/{objective_name}/r0_b_checkpoint_sha256"] = _hash(checkpoint_b)
                parent_artifacts[f"{branch}/{model_name}/{objective_name}/r10_b_embedding_sha256"] = _hash(shifted_embedding_path)

    figure_targets = {
        branch: figure_root / f"lag_curves_{branch}.png"
        for branch in ("held_out", "full_sample")
    }
    existing_outputs = [curves_path, summary_path, provenance_path, *figure_targets.values()]
    if any(path.exists() for path in existing_outputs):
        if not all(path.is_file() for path in existing_outputs):
            raise FileExistsError(f"partial controlled-lag summary output exists; preserve it: {eval_root}")
        cached = json.loads(provenance_path.read_text(encoding="utf-8"))
        cached_artifacts = cached.get("artifacts", {})
        parents_match = (
            cached.get("training_seed") == seed
            and cached.get("parent_artifacts") == parent_artifacts
            and cached.get("candidate_lags") == list(LAGS)
        )
        files_match = all(
            cached_artifacts.get(path.name) == _hash(path)
            for path in (curves_path, summary_path, *figure_targets.values())
        )
        if parents_match and files_match:
            return {
                "run_root": eval_root, "curves": curves_path,
                "summary": summary_path, "provenance": provenance_path,
                **{f"figure_{key}": value for key, value in figure_targets.items()},
            }
        raise FileExistsError(f"controlled-lag outputs have different parents/config; preserve them and use a new run ID: {eval_root}")

    with curves_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure_paths: dict[str, Path] = {}
    for branch, panels in plotted.items():
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True, constrained_layout=True)
        for axis, (title, r0, r10) in zip(axes.flat, panels):
            axis.plot(list(r0), list(r0.values()), label="REAL-0", color="#2878B5", linewidth=2)
            axis.plot(list(r10), list(r10.values()), label="REAL-10 (B +10 bins)", color="#E87500", linewidth=2)
            axis.axvline(0, color="0.5", linewidth=0.8, linestyle=":")
            axis.axvline(10, color="black", linewidth=1, linestyle="--")
            axis.set_title(title.replace("/", " / "))
            axis.grid(alpha=0.2)
        axes[0, 0].set_ylabel("Procrustes R²")
        axes[1, 0].set_ylabel("Procrustes R²")
        for axis in axes[1, :]:
            axis.set_xlabel("Candidate lag (bins), positive = compare A(t) with B(t + lag)")
        axes[0, 0].legend(frameon=False, fontsize=9)
        fig.suptitle(f"Controlled lag recovery — seed {seed} — {branch}")
        target = figure_targets[branch]
        fig.savefig(target, dpi=220)
        plt.close(fig)
        figure_paths[branch] = target
    _json_write(provenance_path, {
        "run_id": run_label,
        "setting": "REAL-10_B_only_digital_shift",
        "population": "A/B pseudo-populations; A unchanged, B shifted +10 bins",
        "training_seed": seed,
        "channel_split_seed": config_b.channel_split_seed,
        "channel_indices": {"A": list(config_a.channel_indices), "B": list(config_b.channel_indices)},
        "split_trial_ids": split_a,
        "candidate_lags": list(LAGS),
        "support": "same reference trial/time rows for REAL-0 and REAL-10; valid full windows; equal comparison count at every candidate lag",
        "metric": "existing lagged Procrustes R2 on unit embeddings; positive lag compares A(t) with B(t+lag)",
        "summary_definitions": {
            "peak_margin_r2": "best curve R2 minus second-best candidate R2",
            "local_sharpness_r2": "best curve R2 minus the larger immediate-neighbor R2",
            "half_prominence_width_bins": "contiguous bins around the peak at or above half the peak prominence over curve minimum",
            "boundary_peak": "best candidate is either endpoint of the frozen [-20,20] scan",
        },
        "fitting_performed": False,
        "reuse_policy": "all REAL-0 A/B checkpoints reused; A embedding reused unchanged; only B input is shifted and transformed",
        "parent_artifacts": parent_artifacts,
        "artifacts": {
            curves_path.name: _hash(curves_path),
            summary_path.name: _hash(summary_path),
            **{path.name: _hash(path) for path in figure_paths.values()},
        },
    })
    return {"run_root": eval_root, "curves": curves_path, "summary": summary_path, "provenance": provenance_path, **{f"figure_{key}": value for key, value in figure_paths.items()}}
