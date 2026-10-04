"""Static figures generated solely from a completed frozen-audit report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


_MODEL_ORDER = {"pca": 0, "cnn1d": 1, "transformer": 2}
_LOSS_ORDER = {
    "PCA": 0,
    "Soft structured contrastive": 1,
    "Supervised InfoNCE": 2,
    "Time Contrastive Blocks": 3,
    "Behavior Contrastive Blocks": 4,
}
_COLORS = {"raw": "#4C566A", "unit": "#2A9D8F"}


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _method_order(rows):
    keys = {(row["model"], row["loss"]) for row in rows}
    return sorted(
        keys,
        key=lambda key: (_MODEL_ORDER.get(key[0], 99), _LOSS_ORDER.get(key[1], 99), key),
    )


def _method_label(key):
    model, loss = key
    if model == "pca":
        return "PCA"
    architecture = {"cnn1d": "CNN1D", "transformer": "Transformer"}.get(model, model)
    short_loss = {
        "Soft structured contrastive": "Soft",
        "Supervised InfoNCE": "InfoNCE",
        "Time Contrastive Blocks": "CEBRA-time",
        "Behavior Contrastive Blocks": "CEBRA-behavior",
    }.get(loss, loss)
    return f"{architecture}\n{short_loss}"


def _value(rows, *, model, loss, subject, representation, metric, category):
    matched = [
        row for row in rows
        if row["model"] == model
        and row["loss"] == loss
        and row["subject"] == subject
        and row["representation"] == representation
        and row["metric"] == metric
        and row["category"] == category
    ]
    if len(matched) != 1:
        raise ValueError(
            "Expected exactly one plotted metric for "
            f"{model}/{loss}/{subject}/{representation}/{metric}, found {len(matched)}"
        )
    return float(matched[0]["value"])


def _scope_title(manifest):
    branch = manifest["branch"]
    if branch == "held_out":
        return "Held-out representation generalization"
    if branch == "full_sample":
        return "Full sample — descriptive in-sample (not generalization)"
    raise ValueError(f"Unsupported audit branch: {branch}")


def _grouped_metric_figure(rows, methods, metric_names, metric_labels, category, scope, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.arange(len(methods), dtype=float)
    figure, axes = plt.subplots(
        nrows=len(metric_names),
        ncols=2,
        figsize=(18, max(7, 3.4 * len(metric_names))),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(metric_names), 2)
    for row_index, (metric, metric_label) in enumerate(zip(metric_names, metric_labels)):
        for column_index, subject in enumerate(("A", "B")):
            axis = axes[row_index, column_index]
            for representation, offset in (("raw", -0.19), ("unit", 0.19)):
                values = [
                    _value(
                        rows,
                        model=model,
                        loss=loss,
                        subject=subject,
                        representation=representation,
                        metric=metric,
                        category=category,
                    )
                    for model, loss in methods
                ]
                axis.bar(
                    positions + offset,
                    values,
                    width=0.36,
                    color=_COLORS[representation],
                    label=representation if row_index == 0 and column_index == 0 else None,
                )
            axis.set_title(f"Subject {subject}")
            axis.set_ylabel(metric_label)
            axis.axhline(0.0, color="#A0A0A0", linewidth=0.7, zorder=0)
            if metric in {"linear_cka", "condition_balanced_accuracy"}:
                axis.set_ylim(0.0, 1.0)
            elif metric.startswith("rsa_"):
                axis.set_ylim(-1.0, 1.0)
            if row_index == len(metric_names) - 1:
                axis.set_xticks(positions)
                axis.set_xticklabels(
                    [_method_label(key) for key in methods],
                    rotation=20,
                    ha="right",
                    fontsize=8,
                )
            else:
                axis.set_xticks(positions, [])
    axes[0, 0].legend(title="Embedding")
    figure.suptitle(scope)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _lag_figure(profiles, methods, manifest, scope, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, constrained_layout=True)
    expected_lag = int(manifest["config"]["lag_bins"])
    for axis, (model, loss) in zip(axes.flat, methods):
        for representation in ("raw", "unit"):
            points = sorted(
                (
                    int(row["lag"]),
                    float(row["score"]),
                )
                for row in profiles
                if row["model"] == model
                and row["loss"] == loss
                and row["representation"] == representation
            )
            if not points:
                raise ValueError(f"Missing lag profile for {model}/{loss}/{representation}")
            lag, score = zip(*points)
            axis.plot(lag, score, marker="o", markersize=2.5, linewidth=1.5,
                      color=_COLORS[representation], label=representation)
        axis.axvline(expected_lag, color="#E76F51", linewidth=1.2, linestyle="--", label="true lag")
        axis.set_title(_method_label((model, loss)))
        axis.set_xlabel("Candidate lag (bins): A(t) vs B(t + lag)")
        axis.set_ylabel("Procrustes R²")
        axis.axhline(0.0, color="#A0A0A0", linewidth=0.7, zorder=0)
    for axis in axes.flat[len(methods):]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle(f"Lag recovery — {scope}")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def generate_frozen_figures(report):
    """Create non-overwriting summary figures from one frozen-audit directory.

    The function reads only ``metrics.csv``, ``lag_profiles.csv``, and the
    manifest written by :func:`evaluate_frozen_run`; it never reloads models or
    regenerates embeddings.
    """
    report = Path(report).resolve()
    metrics_path, profiles_path, manifest_path = (
        report / "metrics.csv",
        report / "lag_profiles.csv",
        report / "manifest.json",
    )
    if not all(path.exists() for path in (metrics_path, profiles_path, manifest_path)):
        raise FileNotFoundError("A completed frozen audit requires metrics.csv, lag_profiles.csv, and manifest.json")
    figures = report / "figures"
    if figures.exists() and any(figures.iterdir()):
        raise FileExistsError("Refusing to overwrite existing frozen-audit figures")

    rows = _read_csv(metrics_path)
    profiles = _read_csv(profiles_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recovery = [row for row in rows if row["category"] == "latent_recovery"]
    if not recovery or any(row.get("reference") != "Z" for row in recovery):
        raise ValueError("Frozen-audit recovery figures require Z-only latent-recovery rows")
    methods = _method_order(recovery)
    if len(methods) != 9:
        raise ValueError(f"Expected nine configured methods, found {len(methods)}")
    scope = _scope_title(manifest)
    figures.mkdir(parents=True, exist_ok=True)

    outputs = {
        "z_recovery_raw_vs_unit.png": figures / "z_recovery_raw_vs_unit.png",
        "task_decoding_raw_vs_unit.png": figures / "task_decoding_raw_vs_unit.png",
        "lag_profiles_raw_vs_unit.png": figures / "lag_profiles_raw_vs_unit.png",
    }
    _grouped_metric_figure(
        rows,
        methods,
        ("procrustes_r2", "rsa_spearman", "rsa_pearson", "linear_cka"),
        ("Procrustes R²", "RSA Spearman", "RSA Pearson", "Linear CKA"),
        "latent_recovery",
        f"Z recovery — {scope}",
        outputs["z_recovery_raw_vs_unit.png"],
    )
    _grouped_metric_figure(
        rows,
        methods,
        ("condition_balanced_accuracy", "progress_r2", "progress_mae"),
        ("Condition balanced accuracy", "Progress R²", "Progress MAE"),
        "downstream_task_decoding",
        f"Task decoding — {scope}",
        outputs["task_decoding_raw_vs_unit.png"],
    )
    _lag_figure(profiles, methods, manifest, scope, outputs["lag_profiles_raw_vs_unit.png"])
    (figures / "plot_manifest.json").write_text(
        json.dumps(
            {
                "source_report": str(report),
                "branch": manifest["branch"],
                "evaluation_scope": manifest["evaluation_scope"],
                "latent_target": "Z only",
                "figures": {name: str(path.name) for name, path in outputs.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return outputs
