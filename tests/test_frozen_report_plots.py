import csv
import json

import pytest

from neurobridge.experiments.frozen_report_plots import generate_frozen_figures


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def test_frozen_figures_are_z_only_and_never_overwrite(tmp_path):
    report = tmp_path / "audit"
    report.mkdir()
    methods = [("pca", "PCA")]
    losses = [
        ("Soft structured contrastive", "soft"),
        ("Supervised InfoNCE", "infonce"),
        ("Time Contrastive Blocks", "cebra_time"),
        ("Behavior Contrastive Blocks", "cebra_behavior"),
    ]
    methods.extend((model, loss) for model in ("cnn1d", "transformer") for loss, _ in losses)
    rows, profiles = [], []
    for method_index, (model, loss) in enumerate(methods):
        for subject in ("A", "B"):
            for representation in ("raw", "unit"):
                for metric in ("procrustes_r2", "rsa_spearman", "rsa_pearson", "linear_cka"):
                    rows.append({
                        "model": model, "loss": loss, "subject": subject,
                        "representation": representation, "category": "latent_recovery",
                        "reference": "Z", "metric": metric, "value": 0.5,
                    })
                for metric in ("condition_balanced_accuracy", "progress_r2", "progress_mae"):
                    rows.append({
                        "model": model, "loss": loss, "subject": subject,
                        "representation": representation, "category": "downstream_task_decoding",
                        "reference": "", "metric": metric, "value": 0.5,
                    })
        for representation in ("raw", "unit"):
            for lag in (-1, 0, 1):
                profiles.append({
                    "model": model, "loss": loss, "representation": representation,
                    "lag": lag, "score": 0.1 * method_index,
                })
    _write_csv(report / "metrics.csv", rows)
    _write_csv(report / "lag_profiles.csv", profiles)
    (report / "manifest.json").write_text(
        json.dumps({"branch": "full_sample", "evaluation_scope": "descriptive_in_sample_representation", "config": {"lag_bins": 10}}),
        encoding="utf-8",
    )
    # An empty output directory is allowed; a populated one is not.
    (report / "figures").mkdir()

    outputs = generate_frozen_figures(report)
    assert set(outputs) == {
        "z_recovery_raw_vs_unit.png",
        "task_decoding_raw_vs_unit.png",
        "lag_profiles_raw_vs_unit.png",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.values())
    assert (report / "figures" / "plot_manifest.json").exists()
    with pytest.raises(FileExistsError, match="overwrite"):
        generate_frozen_figures(report)
