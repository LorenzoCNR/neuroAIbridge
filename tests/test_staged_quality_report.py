import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurobridge.experiments.staged_shared_latent import (
    SharedLatentStageConfig,
    stage_quality_report,
)


class TestStagedQualityReport(unittest.TestCase):
    def test_report_uses_cached_artifacts_and_separates_categories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = SharedLatentStageConfig(output_root="outputs", run_label="cached")
            run = root / "outputs" / "cached"
            embedding_dir = run / "stage04_embeddings" / "held_out"
            metric_dir = run / "stage05_metrics" / "held_out"
            model_dir = run / "stage03_models" / "held_out" / "A_pca_none"
            window_dir = run / "stage02_windows"
            for directory in (embedding_dir, metric_dir, model_dir, window_dir):
                directory.mkdir(parents=True, exist_ok=True)

            split = {
                "train": [0, 1, 2],
                "validation": [3, 4, 5],
                "test": [6, 7, 8],
            }
            (window_dir / "split.json").write_text(json.dumps(split), encoding="utf-8")
            self.assertTrue(all(split[name] for name in ("train", "validation", "test")))
            self.assertFalse(set(split["train"]) & set(split["validation"]))
            self.assertFalse(set(split["train"]) & set(split["test"]))
            self.assertFalse(set(split["validation"]) & set(split["test"]))
            trial_id = np.repeat(np.arange(9), 8)
            labels = np.repeat(np.tile(np.arange(3), 3), 8)
            progress = np.tile(np.linspace(0.0, 1.0, 8), 9)
            embedding = np.column_stack((labels, progress, progress**2)).astype(float)
            np.savez_compressed(
                embedding_dir / "A_pca_none.npz",
                embedding_raw=embedding,
                embedding_unit=embedding,
                trial_id=trial_id,
                time_id=np.tile(np.arange(8), 9),
                global_time_id=np.arange(72),
                labels=labels,
                progress=progress,
                lag_valid=np.ones(72, dtype=bool),
            )
            (model_dir / "compute.json").write_text(
                json.dumps({"fit_seconds": 0.5, "n_parameters": 0}), encoding="utf-8"
            )
            (model_dir / "pca.joblib").write_bytes(b"cached-model")
            with (metric_dir / "pca_none.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["branch", "subject", "model", "loss", "reference", "metric", "value"])
                writer.writerow(["held_out", "A", "pca", "none", "Z", "procrustes_r2", 0.7])
                writer.writerow(["held_out", "A_B", "pca", "none", "", "estimated_lag_bins", 8])

            output = stage_quality_report(root, config, noise_scales=(0.1,))
            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            categories = {row["category"] for row in rows}
            self.assertEqual(
                categories,
                {
                    "latent_process_recovery",
                    "downstream_task_decoding",
                    "temporal_lag_structure",
                    "posthoc_embedding_noise_robustness",
                    "computational_representational_efficiency",
                },
            )
            lag_error = next(row for row in rows if row["metric"] == "absolute_lag_error_bins")
            self.assertEqual(float(lag_error["value"]), 2.0)
            manifest = json.loads(output.with_name("manifest.json").read_text(encoding="utf-8"))
            self.assertIn("not neural-input or encoder robustness", " ".join(manifest["limitations"]))
            selection = manifest["decoder_selections"][0]
            self.assertEqual(selection["n_classes"], 3)
            self.assertEqual(selection["condition_solver"], "lbfgs")
            self.assertEqual(selection["condition_formulation"], "multinomial_softmax")
            self.assertEqual(selection["selection_split"], "VALIDATION")
            self.assertEqual(selection["final_evaluation_split"], "TEST")
            self.assertIn(selection["selected_condition_c"], selection["condition_c_grid"])
            self.assertIn(selection["selected_progress_alpha"], selection["progress_alpha_grid"])


if __name__ == "__main__":
    unittest.main()
