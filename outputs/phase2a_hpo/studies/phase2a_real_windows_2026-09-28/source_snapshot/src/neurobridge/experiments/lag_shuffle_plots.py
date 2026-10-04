"""Plot observed lag profiles against the trial-pairing shuffle null."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from neurobridge.experiments.frozen_report_plots import _method_label, _method_order


def plot_lag_shuffle(report):
    report = Path(report).resolve()
    summary_path, profiles_path, manifest_path = (
        report / "summary.csv", report / "profiles.csv", report / "manifest.json"
    )
    if not all(path.exists() for path in (summary_path, profiles_path, manifest_path)):
        raise FileNotFoundError("A completed lag-shuffle audit is required")
    figures = report / "figures"
    if figures.exists() and any(figures.iterdir()):
        raise FileExistsError("Refusing to overwrite existing lag-shuffle figures")
    with profiles_path.open(newline="", encoding="utf-8") as handle:
        profiles = list(csv.DictReader(handle))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    methods = _method_order(profiles)
    if len(methods) != 9:
        raise ValueError(f"Expected nine configured methods, found {len(methods)}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    colors = {"raw": "#4C566A", "unit": "#2A9D8F"}
    figure, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, constrained_layout=True)
    expected_lag = int(manifest["config"]["lag_bins"])
    for axis, method in zip(axes.flat, methods):
        model, loss = method
        for representation in ("raw", "unit"):
            selected = [
                row for row in profiles
                if row["model"] == model
                and row["loss"] == loss
                and row["representation"] == representation
            ]
            observed = sorted((int(row["lag"]), float(row["score"])) for row in selected if row["permutation"] == "0")
            null = [row for row in selected if row["permutation"] != "0"]
            lags = sorted({int(row["lag"]) for row in null})
            mean, lower, upper = [], [], []
            for lag in lags:
                values = [float(row["score"]) for row in null if int(row["lag"]) == lag]
                mean.append(float(np.mean(values)))
                lower.append(float(np.quantile(values, 0.025)))
                upper.append(float(np.quantile(values, 0.975)))
            observed_lags, observed_scores = zip(*observed)
            axis.fill_between(lags, lower, upper, color=colors[representation], alpha=0.16)
            axis.plot(observed_lags, observed_scores, color=colors[representation], linewidth=1.8,
                      label=f"{representation} observed")
            axis.plot(lags, mean, color=colors[representation], linestyle="--", linewidth=1.0,
                      label=f"{representation} shuffle mean")
        axis.axvline(expected_lag, color="#E76F51", linestyle="--", linewidth=1.1)
        axis.set_title(_method_label(method))
        axis.set_xlabel("Candidate lag (bins)")
        axis.set_ylabel("Procrustes R²")
        axis.axhline(0.0, color="#A0A0A0", linewidth=0.7)
    axes.flat[0].legend(fontsize=7)
    branch = manifest["branch"]
    scope = "full-sample descriptive" if branch == "full_sample" else "held-out"
    figure.suptitle(f"Lag shuffle null — {scope}; shaded = 95% permutation envelope")
    output = figures / "lag_shuffle_null.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    (figures / "plot_manifest.json").write_text(
        json.dumps({"source_report": str(report), "branch": branch, "figure": output.name}, indent=2),
        encoding="utf-8",
    )
    return output
