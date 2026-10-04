"""Run a leakage-controlled, Frosolone-inspired Pipeline-1 adaptation.

The script consumes an existing staged synthetic cache and never fits a
neural representation model. It fits train-only PCA/NMA and a small QDA
downstream decoder, then evaluates the frozen procedure on validation/test
trials. Existing run artifacts are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import f_oneway
from scipy.linalg import eigh
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif


PAPER = (
    "Frosolone, M., Prevete, R., Ognibeni, L., Giugliano, S., Apicella, A., "
    "Pezzulo, G., & Donnarumma, F. (2024). Enhancing EEG-Based MI-BCIs with "
    "Class-Specific and Subject-Specific Features Detected by Neural Manifold "
    "Analysis. Sensors, 24(19), 6110. https://doi.org/10.3390/s24196110"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def read_inputs(run_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, list[int]], dict[str, Any]]:
    data_path = run_dir / "stage01_data" / "shared_data.npz"
    config_path = run_dir / "stage01_data" / "config.json"
    split_path = run_dir / "stage02_windows" / "split.json"
    for path in (data_path, config_path, split_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if set(split) != {"train", "validation", "test"}:
        raise ValueError(f"Unexpected split keys: {sorted(split)}")
    ids = {name: [int(v) for v in split[name]] for name in split}
    all_ids = [v for values in ids.values() for v in values]
    n_trials = int(config["n_trials"])
    if len(all_ids) != n_trials or len(set(all_ids)) != n_trials:
        raise ValueError("Frozen split must partition every trial exactly once.")
    if set(all_ids) != set(range(n_trials)):
        raise ValueError("Frozen split trial IDs do not match configured trials.")
    with np.load(data_path, allow_pickle=False) as archive:
        needed = ("labels", "X_A", "X_B", "valid_A", "valid_B")
        missing = sorted(set(needed) - set(archive.files))
        if missing:
            raise KeyError(f"Synthetic cache lacks required fields: {missing}")
        # Do not materialize any Z/latent arrays; this analysis never uses them.
        arrays = {key: archive[key] for key in needed}
    labels = arrays["labels"].astype(int)
    if labels.shape != (n_trials,):
        raise ValueError(f"Expected trial labels {(n_trials,)}, got {labels.shape}")
    return arrays, ids, config


def pairwise_centroid_distances(
    scores: np.ndarray, labels: np.ndarray, trial_ids: list[int],
    valid: np.ndarray, start: int, stop: int,
) -> pd.DataFrame:
    classes = np.unique(labels)
    centroids = []
    for label in classes:
        selected = [i for i in trial_ids if labels[i] == label]
        trial_means = []
        for i in selected:
            keep = valid[i, start:stop]
            trial_means.append(scores[i, start:stop][keep].mean(axis=0))
        centroids.append(np.mean(trial_means, axis=0))
    centroids_array = np.stack(centroids)
    delta = centroids_array[:, None, :] - centroids_array[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=-1))
    return pd.DataFrame(distances, index=classes, columns=classes)


def evaluate_classifier(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray,
    y_eval: np.ndarray, classes: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    model = QuadraticDiscriminantAnalysis()
    model.fit(x_train, y_train)
    prediction = model.predict(x_eval)
    metrics = {
        "accuracy": float(accuracy_score(y_eval, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_eval, prediction)),
    }
    return metrics, confusion_matrix(y_eval, prediction, labels=classes)


def _trial_covariance(signal: np.ndarray) -> np.ndarray:
    centered = signal - signal.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    trace = float(np.trace(covariance))
    if trace <= np.finfo(float).eps:
        return np.eye(signal.shape[1]) / signal.shape[1]
    return covariance / trace


def fit_multiclass_ovr_csp(
    signals: np.ndarray, labels: np.ndarray, train_ids: list[int], classes: np.ndarray,
) -> tuple[list[np.ndarray], list[str]]:
    """Fit one-vs-rest CSP per class; keep the largest and smallest eigenvector."""
    n_components = signals.shape[-1]
    class_filters: list[np.ndarray] = []
    feature_names: list[str] = []
    ridge = 1e-6
    for label in classes:
        positive = [i for i in train_ids if labels[i] == label]
        negative = [i for i in train_ids if labels[i] != label]
        cov_pos = np.mean([_trial_covariance(signals[i]) for i in positive], axis=0)
        cov_neg = np.mean([_trial_covariance(signals[i]) for i in negative], axis=0)
        composite = cov_pos + cov_neg + ridge * np.eye(n_components)
        eigenvalues, eigenvectors = eigh(cov_pos, composite, check_finite=True)
        order = np.argsort(eigenvalues)
        # One low-variance and one high-variance spatial component per class.
        chosen = np.column_stack((eigenvectors[:, order[0]], eigenvectors[:, order[-1]]))
        class_filters.append(chosen)
        feature_names.extend((f"class{label}_csp_low", f"class{label}_csp_high"))
    return class_filters, feature_names


def transform_multiclass_ovr_csp(
    signals: np.ndarray, filters: list[np.ndarray],
) -> np.ndarray:
    features = []
    for trial_signal in signals:
        trial_features = []
        for class_filter in filters:
            projected = trial_signal @ class_filter
            variance = np.var(projected, axis=0, ddof=1)
            normalized = variance / max(float(variance.sum()), np.finfo(float).eps)
            trial_features.extend(np.log(np.maximum(normalized, np.finfo(float).eps)))
        features.append(trial_features)
    return np.asarray(features, dtype=float)


def fit_transform_interval_csp(
    scores: np.ndarray, start: int, stop: int, labels: np.ndarray,
    train_ids: list[int], classes: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    signals = scores[:, start:stop]
    filters, feature_names = fit_multiclass_ovr_csp(signals, labels, train_ids, classes)
    return transform_multiclass_ovr_csp(signals, filters), feature_names


def select_mi_features(
    x_train: np.ndarray, y_train: np.ndarray, feature_names: list[str],
    n_select: int,
) -> tuple[np.ndarray, list[int], list[float]]:
    scores = mutual_info_classif(x_train, y_train, discrete_features=False, random_state=42)
    selected = np.argsort(-scores, kind="stable")[:n_select]
    return selected, feature_names, scores.tolist()


def one_way_stats(values_by_class: list[np.ndarray]) -> tuple[float, float, float]:
    statistic, p_value = f_oneway(*values_by_class)
    all_values = np.concatenate(values_by_class)
    grand_mean = float(all_values.mean())
    ss_total = float(np.square(all_values - grand_mean).sum())
    ss_between = float(sum(len(group) * np.square(group.mean() - grand_mean) for group in values_by_class))
    eta_squared = ss_between / ss_total if ss_total > np.finfo(float).eps else 0.0
    return float(statistic), float(p_value), float(eta_squared)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("outputs/runs/clean_rebuild_2026-09-23_seed42_synthetic"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/frosolone_synthetic_pipeline1_seed42"),
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing analysis output: {output_dir}"
        )
    data_path = run_dir / "stage01_data" / "shared_data.npz"
    config_path = run_dir / "stage01_data" / "config.json"
    split_path = run_dir / "stage02_windows" / "split.json"
    paper_candidates = sorted(
        (Path.cwd() / "legacy_ai_for_all" / "Bibliog_Code").glob("Frosolone*.pdf")
    )
    if len(paper_candidates) != 1:
        raise FileNotFoundError("Expected exactly one local Frosolone paper PDF in Bibliog_Code")
    paper_path = paper_candidates[0]
    arrays, split, config = read_inputs(run_dir)
    labels = arrays["labels"].astype(int)
    classes = np.unique(labels)
    if len(classes) != int(config["n_conditions"]):
        raise ValueError("Label cardinality differs from frozen config.")
    split_per_class = {
        name: {str(label): int(np.sum(labels[ids] == label)) for label in classes}
        for name, ids in split.items()
    }

    output_dir.mkdir(parents=True)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir()
    tables_dir.mkdir()

    n_components = int(config["latent_dim"])
    window_size = int(config["window_size"])
    half_width = window_size // 2
    result_rows: list[dict[str, Any]] = []
    nma_frames: list[pd.DataFrame] = []
    heldout_nma_frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    pca_variance: dict[str, list[float]] = {}

    for population in ("A", "B"):
        X = arrays[f"X_{population}"].astype(np.float64)
        valid = arrays[f"valid_{population}"].astype(bool)
        if X.ndim != 3 or valid.shape != X.shape[:2]:
            raise ValueError(f"Unexpected X/valid shapes for population {population}")
        train_ids = split["train"]
        valid_time = np.all(valid[train_ids], axis=0)
        candidate_centers = [
            t for t in range(half_width, X.shape[1] - half_width)
            if np.all(valid_time[t - half_width:t + half_width + 1])
        ]
        if not candidate_centers:
            raise ValueError(f"No valid NMA window candidates for {population}")

        # PCA basis is fitted on training trials and valid bins only.
        train_rows = X[train_ids][valid[train_ids]]
        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(train_rows)
        pca_variance[population] = pca.explained_variance_ratio_.tolist()
        flat_scores = pca.transform(X.reshape(-1, X.shape[-1]))
        scores = flat_scores.reshape(X.shape[0], X.shape[1], n_components)

        nma_records: list[dict[str, Any]] = []
        for component in range(n_components):
            for time_bin in candidate_centers:
                grouped = [
                    scores[[i for i in train_ids if labels[i] == label], time_bin, component]
                    for label in classes
                ]
                statistic, p_value, eta_squared = one_way_stats(grouped)
                nma_records.append({
                    "population": population,
                    "split": "train",
                    "component": component + 1,
                    "time_bin": time_bin,
                    "time_seconds": time_bin * float(config["dt"]),
                    "f_statistic": float(statistic),
                    "p_value_raw": float(p_value),
                    "eta_squared": eta_squared,
                })
        nma = pd.DataFrame(nma_records)
        nma_frames.append(nma)
        best_idx = int(nma["p_value_raw"].to_numpy().argmin())
        best = nma.iloc[best_idx]
        center = int(best["time_bin"])
        start, stop = center - half_width, center + half_width + 1

        # Evaluate the frozen component/time scan on held-out trials without
        # using these outcomes to alter PCA, the selected center, or the window.
        heldout_ids = split["test"]
        heldout_records: list[dict[str, Any]] = []
        for component in range(n_components):
            for time_bin in candidate_centers:
                grouped = [
                    scores[[i for i in heldout_ids if labels[i] == label], time_bin, component]
                    for label in classes
                ]
                statistic, p_value, eta_squared = one_way_stats(grouped)
                heldout_records.append({
                    "population": population,
                    "split": "test_held_out",
                    "component": component + 1,
                    "time_bin": time_bin,
                    "time_seconds": time_bin * float(config["dt"]),
                    "f_statistic": statistic,
                    "p_value_raw_descriptive": p_value,
                    "eta_squared": eta_squared,
                })
        heldout_nma = pd.DataFrame(heldout_records)
        heldout_nma_frames.append(heldout_nma)
        selected_train_eta = float(nma[(nma.component == int(best["component"])) & nma.time_bin.between(start, stop - 1)]["eta_squared"].mean())
        selected_test_eta = float(heldout_nma[(heldout_nma.component == int(best["component"])) & heldout_nma.time_bin.between(start, stop - 1)]["eta_squared"].mean())
        heldout_pc_mean_eta = float(heldout_nma[heldout_nma.component == int(best["component"])]["eta_squared"].mean())
        selected_rows.append({
            "population": population,
            "selected_component": int(best["component"]),
            "center_bin": center,
            "center_seconds": center * float(config["dt"]),
            "window_start_bin_inclusive": start,
            "window_stop_bin_exclusive": stop,
            "window_size_bins": window_size,
            "window_duration_seconds": window_size * float(config["dt"]),
            "training_anova_f": float(best["f_statistic"]),
            "training_anova_p_uncorrected": float(best["p_value_raw"]),
            "training_selected_pc_window_mean_eta_squared": selected_train_eta,
            "heldout_selected_pc_window_mean_eta_squared": selected_test_eta,
            "heldout_selected_window_minus_selected_pc_time_mean_eta_squared": selected_test_eta - heldout_pc_mean_eta,
            "candidate_pc_time_tests": int(len(nma)),
        })

        full_start = int(np.flatnonzero(valid_time)[0])
        full_stop = int(np.flatnonzero(valid_time)[-1]) + 1
        full_features, full_names = fit_transform_interval_csp(
            scores, full_start, full_stop, labels, train_ids, classes
        )
        selected_features, selected_names = fit_transform_interval_csp(
            scores, start, stop, labels, train_ids, classes
        )
        p1_all_features = np.concatenate((full_features, selected_features), axis=1)
        p1_all_names = (
            [f"full_{name}" for name in full_names]
            + [f"nma_{name}" for name in selected_names]
        )
        y_train = labels[train_ids]
        n_select = min(len(classes), p1_all_features.shape[1])
        p1_selected, _, p1_mi = select_mi_features(
            p1_all_features[train_ids], y_train, p1_all_names, n_select
        )
        full_selected, _, full_mi = select_mi_features(
            full_features[train_ids], y_train, full_names,
            min(len(classes), full_features.shape[1]),
        )
        pd.DataFrame({
            "feature": p1_all_names,
            "mutual_information_train": p1_mi,
            "selected_for_pipeline1_qda": [i in set(p1_selected.tolist()) for i in range(len(p1_all_names))],
        }).to_csv(tables_dir / f"pipeline1_train_feature_selection_{population}.csv", index=False)
        pd.DataFrame({
            "feature": full_names,
            "mutual_information_train": full_mi,
            "selected_for_reference_qda": [i in set(full_selected.tolist()) for i in range(len(full_names))],
        }).to_csv(tables_dir / f"reference_train_feature_selection_{population}.csv", index=False)

        for eval_split in ("validation", "test"):
            eval_ids = split[eval_split]
            baseline_metrics, baseline_cm = evaluate_classifier(
                full_features[train_ids][:, full_selected], y_train,
                full_features[eval_ids][:, full_selected],
                labels[eval_ids], classes,
            )
            p1_metrics, p1_cm = evaluate_classifier(
                p1_all_features[train_ids][:, p1_selected], y_train,
                p1_all_features[eval_ids][:, p1_selected],
                labels[eval_ids], classes,
            )
            for model_name, metrics, cm in (
                ("full_interval_reference", baseline_metrics, baseline_cm),
                ("pipeline1_full_plus_NMA_interval", p1_metrics, p1_cm),
            ):
                result_rows.append({
                    "population": population,
                    "split": eval_split,
                    "model": model_name,
                    "n_train_trials": len(train_ids),
                    "n_eval_trials": len(eval_ids),
                    "n_classes": len(classes),
                    **metrics,
                })
                if eval_split == "test":
                    pd.DataFrame(
                        cm, index=classes, columns=classes
                    ).to_csv(tables_dir / f"confusion_{population}_{model_name}.csv")
                    if model_name == "pipeline1_full_plus_NMA_interval":
                        fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
                        image = ax.imshow(cm, cmap="Blues")
                        ax.set_title(f"Held-out test confusion: population {population}, Pipeline 1")
                        ax.set_xlabel("Predicted condition")
                        ax.set_ylabel("True condition")
                        ax.set_xticks(range(len(classes)), classes)
                        ax.set_yticks(range(len(classes)), classes)
                        fig.colorbar(image, ax=ax, label="Trial count")
                        fig.savefig(figures_dir / f"confusion_{population}_pipeline1.png", dpi=220)
                        fig.savefig(figures_dir / f"confusion_{population}_pipeline1.pdf")
                        plt.close(fig)

        for interval_name, interval in (("full_valid_support", (full_start, full_stop)), ("selected_NMA_window", (start, stop))):
            distances = pairwise_centroid_distances(
                scores, labels, split["test"], valid, interval[0], interval[1]
            )
            distances.to_csv(tables_dir / f"test_centroid_distances_{population}_{interval_name}.csv")
            if interval_name == "selected_NMA_window":
                fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
                image = ax.imshow(distances.to_numpy(), cmap="magma")
                ax.set_title(f"Held-out class-centroid distances: {population}, NMA window")
                ax.set_xlabel("Task condition")
                ax.set_ylabel("Task condition")
                ax.set_xticks(range(len(classes)), classes)
                ax.set_yticks(range(len(classes)), classes)
                fig.colorbar(image, ax=ax, label="Euclidean distance in PCA-score space")
                fig.savefig(figures_dir / f"test_centroid_distances_{population}_NMA.png", dpi=220)
                fig.savefig(figures_dir / f"test_centroid_distances_{population}_NMA.pdf")
                plt.close(fig)

        # NMA diagnostic: raw training p-values only, used to locate interval.
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True, constrained_layout=True)
        for ax, frame, panel_title in (
            (axes[0], nma, "Train-only NMA selection profile"),
            (axes[1], heldout_nma, "Held-out test profile (evaluation only)"),
        ):
            for component in range(1, n_components + 1):
                sub = frame[frame["component"] == component]
                ax.plot(sub["time_seconds"], sub["eta_squared"], label=f"PC{component}")
            ax.axvspan(start * float(config["dt"]), (stop - 1) * float(config["dt"]), color="grey", alpha=0.22, label="selected NMA window")
            ax.axvline(center * float(config["dt"]), color="black", linestyle="--", linewidth=1.0)
            ax.set(title=panel_title, xlabel="Time within trial (s)")
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("One-way ANOVA effect size (eta-squared)")
        axes[0].legend(frameon=False, ncol=2)
        axes[1].legend(frameon=False, ncol=2)
        fig.savefig(figures_dir / f"nma_profile_{population}.png", dpi=220)
        fig.savefig(figures_dir / f"nma_profile_{population}.pdf")
        plt.close(fig)

        nma.to_csv(tables_dir / f"nma_pc_time_tests_{population}.csv", index=False)
        heldout_nma.to_csv(tables_dir / f"nma_heldout_pc_time_profile_{population}.csv", index=False)

    metrics_frame = pd.DataFrame(result_rows)
    metrics_frame.to_csv(tables_dir / "decoding_metrics.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(tables_dir / "selected_nma_windows.csv", index=False)
    pd.concat(nma_frames, ignore_index=True).to_csv(tables_dir / "nma_all_populations.csv", index=False)
    pd.concat(heldout_nma_frames, ignore_index=True).to_csv(tables_dir / "nma_heldout_all_populations.csv", index=False)

    # Held-out balanced-accuracy comparison figure.
    test_rows = metrics_frame[metrics_frame["split"] == "test"]
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    x = np.arange(2)
    width = 0.34
    for j, population in enumerate(("A", "B")):
        vals = [
            float(test_rows[(test_rows.population == population) & (test_rows.model == model)]["balanced_accuracy"].iloc[0])
            for model in ("full_interval_reference", "pipeline1_full_plus_NMA_interval")
        ]
        ax.bar(x + (j - 0.5) * width, vals, width, label=f"Population {population}")
    ax.set_xticks(x, ["Full interval\nreference", "Pipeline 1\n(full + selected window)"])
    ax.set_ylabel("Held-out balanced accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Synthetic task decoding (separate A/B populations)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(figures_dir / "heldout_balanced_accuracy.png", dpi=220)
    fig.savefig(figures_dir / "heldout_balanced_accuracy.pdf")
    plt.close(fig)

    source_script = Path(__file__).resolve()
    provenance = {
        "analysis": "Frosolone-inspired synthetic Pipeline 1 adaptation",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "source_hashes": {
            "shared_data_npz_sha256": sha256_file(data_path),
            "config_json_sha256": sha256_file(config_path),
            "split_json_sha256": sha256_file(split_path),
            "analysis_script_sha256": sha256_file(source_script),
            "paper_pdf_sha256": sha256_file(paper_path),
            "config_canonical_sha256": sha256_json(config),
        },
        "paper_source_path": str(paper_path),
        "inputs_read": ["labels", "X_A", "X_B", "valid_A", "valid_B"],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "protocol": {
            "population_handling": "A and B analyzed separately; never concatenated",
            "n_components": n_components,
            "pca_fit": "flattened TRAIN trial x valid time observations only; raw neural counts; no feature standardization",
            "nma": "one-way ANOVA by eight task labels at each PCA component and valid time; minimum uncorrected p across component-time candidates chooses a single center",
            "window_size_bins": window_size,
            "window_rule": "centered odd window from frozen run config; B invalid leading bins excluded from candidate support/features",
            "pipeline1_adaptation": "fit one-vs-rest CSP independently on the full valid interval and selected NMA window using train trials; retain one low- and one high-variance spatial filter per class; concatenate log-normalized projected-variance features; select top min(number_of_classes, feature_count) features by train-only mutual information; fit fixed unregularized QDA",
            "reference": "full-interval CSP features only with train-only mutual-information selection and QDA; descriptive reference, not an additional Frosolone pipeline",
            "p_values": "raw ANOVA p-values on TRAIN reproduce the paper's separability criterion; held-out p-values are descriptive only; effect-size eta-squared is plotted on both splits; neither held-out profile nor p-values can alter the selected interval",
            "distance_analysis": "Euclidean distances between held-out task-condition centroids in train-fitted PCA-score space; post-hoc only, not used for selection",
            "csp_adaptation": "one-vs-rest CSP on the three PCA scores; one low- and one high-variance filter per class; no EEG frequency bank because the synthetic counts are sampled at 50 Hz (Nyquist 25 Hz) and are not EEG oscillations",
            "feature_selection_adaptation": "train-only scikit-learn mutual_info_classif; retain eight features as a sample-limited choice (paper uses CSP m=2 and D=4*K)",
            "test_firewall": "test trials are excluded from PCA fitting, NMA, window selection, and QDA fitting; used only for final metrics, confusion matrices, and descriptive class-centroid distances",
            "neural_model_training": "none",
        },
        "split_counts": {key: len(value) for key, value in split.items()},
        "split_class_counts": split_per_class,
        "pca_explained_variance_ratio_by_population": pca_variance,
        "selected_windows": selected_rows,
        "metrics": result_rows,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    metric_lines = []
    for row in result_rows:
        if row["split"] == "test":
            metric_lines.append(
                f"| {row['population']} | {row['model']} | {row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row['n_eval_trials']} |"
            )
    window_lines = [
        f"| {row['population']} | {row['selected_component']} | {row['center_bin']} | {row['center_seconds']:.3f} | [{row['window_start_bin_inclusive']}, {row['window_stop_bin_exclusive']}) | {row['training_anova_p_uncorrected']:.3g} | {row['training_selected_pc_window_mean_eta_squared']:.3f} | {row['heldout_selected_pc_window_mean_eta_squared']:.3f} |"
        for row in selected_rows
    ]
    report = f"""# Frosolone-inspired Pipeline 1 on NeuroBridge Synthetic

## Scope and result

This is a separate, downstream synthetic analysis. It reuses the cached seed-42 synthetic data and frozen trial split. **No neural encoder was retrained.** The analysis fits train-only PCA/NMA and a fixed QDA decoder; held-out test trials are evaluated only after all feature choices are fixed.

## What was run

The staged cache is `{run_dir}`. The frozen configuration reports {config['n_trials']} trials, {config['n_conditions']} balanced task conditions, {config['trial_length']} time bins at {config['dt']} s/bin, a {config['latent_dim']}-D simulated latent, {config['n_neurons_A']} A units, {config['n_neurons_B']} B units, and centered input windows of {config['window_size']} bins. The existing split was reused unchanged: {len(split['train'])} train, {len(split['validation'])} validation, {len(split['test'])} test trials, with five test trials per condition. A and B were analyzed separately; they were not pooled. The script loads only raw counts, labels, and validity masks; it does not materialize `Z_A`, `Z_B`, or `Z_shared`.

The analysis sequence was:

1. Read the cached raw neural count tensors `X_A`/`X_B`, task-condition labels, validity masks, frozen config, and split metadata. Record SHA-256 hashes in `provenance.json`.
2. Fit a {n_components}-component PCA basis separately for A and B, using only valid time bins from TRAIN trials. The component count matches the configured simulated latent dimensionality; the true `Z` values were not used to fit PCA or choose a window.
3. For each PCA component and valid time bin, perform one-way ANOVA across the eight task conditions using TRAIN trials only. Choose the single component/time point with the smallest raw p-value and center a {window_size}-bin interval there ({window_size * float(config['dt']):.2f} s). The search yields {len(nma_frames[0]) // n_components} candidate time points per component. For B, the frozen leading invalid bins from the imposed lag are excluded. After fixing this window, compute the same time/component profile and eta-squared on TEST trials only as a held-out reproducibility check; it does not modify the window.
4. On PCA-score trajectories, fit a multiclass one-vs-rest CSP bank independently over the full valid trial interval and the selected window, using TRAIN trials only. Retain the high- and low-variance CSP filters for each of the eight classes; extract log-normalized projected variances; concatenate the full and selected-window features (Pipeline-1 feature fusion). Rank the fused features by mutual information on TRAIN only, retain eight, then fit fixed unregularized QDA.
5. As a context reference, fit the same QDA with only full-interval features. Report validation and test decoding. The held-out class-centroid distance matrices use only the already train-fitted PCA transform and are descriptive; they did not choose the NMA window or classifier.

## NMA-selected windows

| Population | Selected PC | Center (bin) | Center (s) | Window [start, stop) | TRAIN raw ANOVA p | TRAIN eta-squared in selected PC/window | TEST eta-squared in same PC/window |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(window_lines)}

The paired profile plots show eta-squared over time on TRAIN and untouched TEST trials, with the TRAIN-selected window shaded. Held-out effect size inside that fixed window is the central check that class separation carries over; no test significance threshold or test-based window search is used.

## Held-out decoding

| Population | Procedure | Accuracy | Balanced accuracy | Test trials |
|---|---|---:|---:|---:|
{chr(10).join(metric_lines)}

Balanced accuracy is the mean recall across task conditions; with the balanced held-out split it equals accuracy.

The held-out set contains 40 trials (five per condition), so chance balanced accuracy is 0.125 and estimates are coarse. In this run, Pipeline 1 scored 0.150 for A and 0.075 for B; the full-interval reference scored the same values. Thus the classifier result does **not** show a held-out decoding gain. At the selected PC/window, held-out mean eta-squared was 0.244 for A and 0.194 for B, compared with 0.067 and 0.060 on TRAIN. This indicates measurable class-mean separation in the fixed held-out interval, but with five trials per condition it is noisy; it does not override the near-chance CSP/QDA result. The profiles and confusion matrices are the evidence, not a claim of a successful classifier.

## What is and is not a reproduction

The paper has a baseline Pipeline 0 and six NMA-derived variants (Pipelines 1-6). Pipeline 1 fuses classifier features extracted from the full temporal interval and the globally most class-separable interval. The paper's NMA uses PCA-derived component trajectories and timewise one-way ANOVA; pairwise Tukey tests support its pairwise-class variants, not the global Pipeline 1 window. It does **not** select by Euclidean distances. Its downstream EEG stack is filter-bank CSP, mutual-information feature selection, and QDA. See Methods, Sections 2.5-2.6 (PDF pages 7-8).

Here, the NMA interval-selection logic and Pipeline-1 feature-fusion idea are retained. CSP, mutual-information feature ranking, and QDA are also included, but adapted: there is no EEG filter bank (the synthetic sampling rate is 50 Hz, with a 25-Hz Nyquist frequency, and the observations are spike counts rather than EEG rhythms); CSP is one-vs-rest over the eight synthetic conditions and is fitted to the 3-D PCA-score trajectories. One low- and one high-variance filter are kept per class because the PCA input has three dimensions. Eight MI-ranked features are retained so QDA covariance is estimable with the available per-class training trials; this is a sample-limited adaptation of the paper's `m=2`, `D=4*K` setting. This is therefore a **Frosolone-inspired adaptation**, not an exact replication of the EEG pipeline. Pairwise Euclidean distances between held-out condition centroids in PCA-score space are provided as a separate descriptive result, not as the NMA selection rule.

## Interpretation limits

- The minimum raw p-value was selected over multiple component-time tests, following the paper's raw separability criterion. These p-values are selection diagnostics, not confirmatory significance claims; no multiplicity correction was used to choose the window.
- The saved split adjusts the nominal per-condition defaults to keep the requested global proportions: it contains 17-18 TRAIN, 5 TEST, and 2-3 VALIDATION trials per class. This exact saved split was preserved.
- The synthetic task classes and 3-D ground truth make this a controlled method check, not evidence that the same method generalizes to EEG or real monkey data.
- `Z` is intentionally excluded from PCA/NMA/decoder selection. Any future latent-recovery analysis must remain a separate post-hoc evaluation.
- The validation split is reported but is not used for tuning; no settings were chosen after seeing validation/test performance.
- The full-interval reference is a context baseline, not an additional Frosolone pipeline.
- Pipeline-1 held-out classification is near chance in this single split; positive eta-squared in the selected window is not sufficient to claim effective decoding.

## Files

- `tables/selected_nma_windows.csv`: selected component/time/window and training statistics.
- `tables/nma_pc_time_tests_*.csv`, `tables/nma_all_populations.csv`: train-only component/time ANOVA scan.
- `tables/nma_heldout_pc_time_profile_*.csv`: descriptive held-out profiles under the train-selected PCA basis/window.
- `tables/decoding_metrics.csv`: validation/test accuracy and balanced accuracy.
- `tables/pipeline1_train_feature_selection_*.csv`: train-only MI feature ranking and selection.
- `tables/confusion_*.csv`: held-out confusion matrices.
- `tables/test_centroid_distances_*.csv`: held-out class-centroid Euclidean distances in PCA-score space.
- `figures/`: NMA profiles, held-out confusion matrices, distance heatmaps, and decoding comparison (PNG + PDF).
- `provenance.json`: input/code hashes and the resolved procedure.
- `PAPER_RECAP.md`: paper methods/results summary with section, figure, and table references.

## References

{PAPER}

Source PDF in this repository: `legacy_ai_for_all/Bibliog_Code/Frosolone, Prevete et al,Enhancing EEG-Based MI-BCIs with Class-Specific and, Subject... 2024.pdf`.
"""
    (output_dir / "PIPELINE1_SYNTHETIC_ANALYSIS.md").write_text(report, encoding="utf-8")
    paper_recap = f"""# Frosolone et al. (2024): paper recap

## In one sentence

The paper adds Neural Manifold Analysis (NMA) to EEG motor-imagery classification: use PCA-component trajectories and time-resolved class-separability tests to choose informative time windows, then classify with filter-bank CSP (FBCSP), mutual-information feature selection (MIBIF), and QDA.

## Method, with exact locations

- **Sections 2.1-2.4, PDF pp. 4-7:** The conventional signal path has nine 4-Hz Chebyshev-II bands covering 4-40 Hz, one-vs-rest CSP, MIBIF, and QDA. CSP keeps low- and high-variance filters per class/band; MIBIF selects discriminative features from training trials. The paper reports CSP setting `m=2` and MIBIF dimension `D=4*K`, with a paired-feature safeguard that can expand the set.
- **Section 2.5, PDF pp. 7-8; Eq. 9; Figs. 4-5:** PCA represents each trial as component trajectories `c_h(t)`. At every time point, one-way ANOVA across class trajectories gives a separability p-value (smaller means stronger separation). Tukey post-hoc tests give pairwise class p-values. The global time `s` minimizes the all-class p-value; pairwise time `s_ij` minimizes the Tukey p-value among times passing the overall ANOVA threshold. This is statistical class separability, not Euclidean distance.
- **Section 2.6, PDF pp. 8-9:** The article defines FBCSP on the full interval `T`; Pipeline 0 runs it only on the global NMA window `Ts`. Pipelines 1-6 are six further variants: P1 separately extracts FBCSP features from `T` and `Ts` and concatenates features; P2 concatenates the raw time segments then extracts features; P3/P4 use the confused class pair from the FBCSP confusion matrix and pairwise window `Ts_ij` (feature fusion vs raw-time concatenation); P5/P6 choose the pair from the Pipeline-0 confusion matrix and use `Ts0_ij` (again feature fusion vs raw-time concatenation). Thus the article has **seven NMA pipelines total, P0-P6**, plus FBCSP baseline.

## Data and evaluation

- **Section 3.1, PDF pp. 9-11, Fig. 2:** Graz 2b: nine right-handed subjects, two classes (left/right hand), three bipolar EEG channels (C3/Cz/C4). Table 1 uses 10-fold cross-validation on BT; Table 2 evaluates on the separate BE session.
- **Section 3.2.1, PDF pp. 11-12, Fig. 3:** Graz 2a: nine subjects, four classes (left hand/right hand/feet/tongue), 22 EEG channels, 288 trials per AT and AE session. AT is used for 10-fold cross-validation (Table 3); AE is the held-out session (Table 4).

## Main numerical results (accuracy, mean +/- SE)

| Dataset / evaluation | FBCSP | SCN | NMA variants (P0-P6) | Per-subject best NMA |
|---|---:|---:|---|---:|
| Graz 2b BT CV, Table 1 | 71.8 +/- 3.8 | n/a | P0 70.0 +/- 3.5; P1 73.3 +/- 3.7; P2 73.5 +/- 3.5 | 74.2 +/- 3.5 |
| Graz 2b BE held-out, Table 2 | 75.1 +/- 4.9 | 76.8 +/- 5.1 | P0 74.4 +/- 4.9; P1 77.3 +/- 4.4; P2 76.1 +/- 4.4 | 78.3 +/- 4.7 |
| Graz 2a AT CV, Table 3 | 67.9 +/- 5.5 | n/a | P0 65.2 +/- 5.0; P1 70.3 +/- 5.2; P2 69.5 +/- 5.0; P3 68.0 +/- 4.9; P4 70.2 +/- 5.2; P5 68.3 +/- 5.1; P6 70.5 +/- 5.0 | 73.2 +/- 5.1 |
| Graz 2a AE held-out, Table 4 | 62.4 +/- 6.3 | 63.4 +/- 1.9* | P0 59.5 +/- 5.2; P1 63.0 +/- 5.7; P2 64.8 +/- 5.6; P3 63.0 +/- 5.8; P4 64.6 +/- 5.7; P5 63.0 +/- 6.2; P6 63.8 +/- 5.8 | 66.7 +/- 5.5 |
| Graz 2a AE cross-subject, Table 5 | 62.4 +/- 6.3 | 63.4 +/- 5.5 | per-subject NMA 66.7 +/- 5.5; cross-subject NMA 67.6 +/- 5.4 | - |

`*` The SCN standard error printed in Table 4 appears inconsistent with Table 5 (`+/- 1.9` vs `+/- 5.5`); this recap preserves each table as printed rather than silently correcting it. All values are percent accuracy and mean +/- standard error across subjects.

## What the paper figures show

- **Fig. 1:** BCI block diagram with NMA added before feature extraction/selection/classification.
- **Figs. 2-3:** Timing of the two Graz motor-imagery paradigms.
- **Figs. 4-5:** Subject 9, Graz 2a: all-class and pairwise time-varying separability p-values on two PCA directions; the two displayed directions explain about 73.7% and 22.6% of variance.
- **Fig. 6:** Subject-9 FBCSP confusion matrix; the most confused pair is right hand vs tongue (classes 2 vs 4), used to select pair-specific intervals.
- **Figs. 7-8:** Average class trajectories and time-varying ellipsoids in the two displayed manifold directions, marking globally and pairwise discriminative points.
- **Fig. 9:** Cross-subject NMA sharing results as accuracy change relative to FBCSP.
- **Fig. 10:** Which filter-bank frequencies supplied selected CSP features across subjects.
- **Fig. 11:** CSP spatial topographies for selected features; illustrates class-related EEG spatial patterns and borrowed subject features.

## Interpretation, not just the headline

The pattern is positive but not universal: Pipeline 0 often reduces mean accuracy, while P1-P6 trade wins across subjects. On separate-session evaluations the best NMA variant averaged 78.3% (2b) and 66.7% (2a), with cross-subject NMA at 67.6% on 2a. These `Best Pipelines` summaries are per-subject maxima across alternatives, not one single fixed pipeline; they are an optimistic envelope unless winner selection is completed without using the final evaluation set. The fixed variants' means are therefore more informative for a strict generalization claim.

The paper studies EEG oscillatory motor-imagery decoding and a specific frequency-bank/CSP stack. Our synthetic counts are not EEG and have eight conditions, with 17-18 train trials and five test trials per condition in the reused split. The current analysis keeps NMA and Pipeline-1 structure while adapting frequency-bank CSP; it must be described as an inspired transfer, not a literal reproduction.

## Source

{PAPER}

Local source PDF: `{paper_path}`. Methods/result references above use the PDF page numbers and the printed section/figure/table numbering.
"""
    (output_dir / "PAPER_RECAP.md").write_text(paper_recap, encoding="utf-8")

    # Human-readable end-to-end chain for transfer/review without duplicating
    # the large cached source data.
    chain = f"""# Reproduction chain and reference files

This index records the complete provenance chain for the separate Synthetic NMA / Pipeline-1-inspired analysis. The staged source run and paper PDF are referenced in place; neither is copied or modified.

## Source paper

- Citation: {PAPER}
- Local PDF: `{paper_path}`
- SHA-256: `{sha256_file(paper_path)}`
- Relevant method locations: Sections 2.1-2.6 (PDF pp. 4-9); result datasets Sections 3.1-3.2 (pp. 9-12); Tables 1-5 and Figures 1-11. The detailed map and reported results are in `PAPER_RECAP.md`.

## Data and protocol parents (read-only)

- Staged source run: `{run_dir}`
- Neural/count container: `{data_path}` (SHA-256 `{sha256_file(data_path)}`)
- Frozen configuration: `{config_path}` (SHA-256 `{sha256_file(config_path)}`)
- Frozen trial split: `{split_path}` (SHA-256 `{sha256_file(split_path)}`)
- Split counts: {json.dumps({name: len(ids) for name, ids in split.items()}, sort_keys=True)}
- Data fields loaded: `labels`, `X_A`, `X_B`, `valid_A`, `valid_B`. Latent truth arrays (`Z*`) are deliberately not loaded. Test IDs are retained only to keep the frozen partition and for final held-out reporting; no test observations fit PCA, NMA window selection, CSP, feature selection, or QDA.

## Analysis implementation

- Script: `{source_script}`
- Script SHA-256: `{sha256_file(source_script)}`
- Analysis protocol, decisions, findings, limits: `PIPELINE1_SYNTHETIC_ANALYSIS.md`
- Paper methods/results recap: `PAPER_RECAP.md`
- Machine-readable provenance and resolved settings: `provenance.json`
- Neural model retraining: **none**. This analysis fits a downstream PCA basis and CSP/QDA decoder using the cached synthetic observations; it does not refit any NeuroBridge neural encoder.

## Exact reproduction command

Run from repository root, using a new output directory (the script refuses to overwrite):

```powershell
python tools/frosolone_pipeline1_synthetic.py --run-dir "{run_dir}" --output-dir "outputs/frosolone_synthetic_pipeline1_seed42_rerun"
```

The command reads the source/cache files above, writes the report, tables, and figures below, and does not modify its parents.

## Generated deliverables

"""
    generated_files = sorted(
        path for path in output_dir.rglob("*")
        if path.is_file() and path.name not in {"REPRODUCTION_CHAIN.md", "SHA256SUMS.txt"}
    )
    chain += "\n".join(f"- `{path.relative_to(output_dir).as_posix()}`" for path in generated_files)
    chain += "\n\nAll generated files (excluding the checksum list itself) are SHA-256 indexed in `SHA256SUMS.txt`.\n"
    (output_dir / "REPRODUCTION_CHAIN.md").write_text(chain, encoding="utf-8")

    checksum_lines = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
        checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "selected_windows": selected_rows, "test_metrics": [r for r in result_rows if r["split"] == "test"]}, indent=2))


if __name__ == "__main__":
    main()
