"""Mocked smoke/resume checks for the final Real campaign runner.

No optimizer or model fit is executed by this test module.
"""

from __future__ import annotations

import csv
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from neurobridge.experiments.phase2a_hpo import _canonical_json, file_sha256  # noqa: E402
import final_thesis_run_real as runner  # noqa: E402


class FinalRealRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name).resolve()
        self.spec_sha256 = "a" * 64
        self.slots = []
        for index in range(72):
            reused = index < 24
            identifier = f"real-final-{index:02d}"
            folder = self.project / ("hpo_reused" if reused else
                                     "outputs/final_thesis_v1/held_out/trials") / identifier
            slot = {
                "trial_id": identifier, "reuse_hpo_checkpoint": reused,
                "checkpoint_path": str(folder / "best_validation_checkpoint.pt") if reused else None,
                "output_path": str(folder) if not reused else None,
                "safe_input_reference": str(self.project / "safe_inputs/real_w21/A_PROXIMAL/"
                                            "reference_manifest.json"),
                "architecture": "cnn1d", "objective": "soft", "population": "A_PROXIMAL",
                "training_seed_root": 1101 if reused else 1201,
                "candidate_index": 2, "config_sha256": "b" * 64,
            }
            self.slots.append(slot)
            if reused:
                folder.mkdir(parents=True)
                (folder / "result.json").write_text('{"reused":true}\n', encoding="utf-8")
        self.spec = {"selected_real_window_size": 21, "final_slots": self.slots}
        smoke_slot = self.slots[24]
        smoke_dir = (self.project / "outputs/final_thesis_v1/smoke" /
                     f"{smoke_slot['trial_id']}-u1")
        smoke_dir.mkdir(parents=True)
        artifact_names = (
            "best_validation_checkpoint.pt", "stopping_checkpoint.pt",
            "training_history.csv", "validation_embedding.npz",
        )
        for name in artifact_names:
            (smoke_dir / name).write_bytes(name.encode("utf-8"))
        smoke_result = {
            "trial_id": smoke_slot["trial_id"], "status": "ELIGIBLE",
            "stopping_update": 1,
            "artifact_sha256": {name: file_sha256(smoke_dir / name)
                                for name in artifact_names},
        }
        (smoke_dir / "result.json").write_text(
            _canonical_json(smoke_result) + "\n", encoding="utf-8")

    @staticmethod
    def _result(identifier: str) -> dict:
        return {"trial_id": identifier, "status": "ELIGIBLE",
                "validation_loss": 1.0, "selected_update": 1,
                "stopping_update": 1, "artifact_sha256": {}, "geometry": {}}

    def _fit_without_training(self, _project, trial, _spec_sha):
        target = Path(trial["output_path"])
        result_path = target / "result.json"
        if result_path.is_file():
            return json.loads(result_path.read_text(encoding="utf-8"))
        self.new_fit_ids.append(trial["trial_id"])
        target.mkdir(parents=True)
        result = self._result(trial["trial_id"])
        result_path.write_text(_canonical_json(result) + "\n", encoding="utf-8")
        return result

    def _run_mocked(self):
        self.new_fit_ids = []
        with patch.object(runner, "_frozen_spec", return_value=(self.spec, self.spec_sha256)), patch.object(
            runner, "_checked_extension", return_value=(None, None, None, {"candidates": {}}, None)), patch.object(
            runner, "_trial", side_effect=lambda _project, _spec, slot, *_args: slot), patch.object(
            runner, "_reused_result", side_effect=lambda slot: self._result(slot["trial_id"])), patch.object(
            runner, "fit_final_real_trial", side_effect=self._fit_without_training):
            with contextlib.redirect_stdout(io.StringIO()):
                return runner.run(self.project)

    def test_successful_one_update_smoke_allows_campaign(self) -> None:
        # The smoke record deliberately has no `optimizer_updates` key; the
        # frozen core reports its actual count as `stopping_update`.
        summary = self._run_mocked()
        self.assertEqual(summary["status"], "FINAL_REAL_HELD_OUT_FITS_COMPLETE")
        self.assertEqual(len(self.new_fit_ids), 48)
        table = self.project / "outputs/final_thesis_v1/held_out/FINAL_MODEL_FITS.csv"
        with table.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 72)

    def test_restart_reconciles_existing_records_before_first_table_write(self) -> None:
        for slot in self.slots[24:26]:
            target = Path(slot["output_path"])
            target.mkdir(parents=True)
            (target / "result.json").write_text(
                _canonical_json(self._result(slot["trial_id"])) + "\n", encoding="utf-8")
        original_table = runner._rolling_table
        first_known_count = []

        def capture_table(path, spec, known, spec_sha):
            if not first_known_count:
                first_known_count.append(len(known))
            return original_table(path, spec, known, spec_sha)

        with patch.object(runner, "_rolling_table", side_effect=capture_table):
            summary = self._run_mocked()
        self.assertEqual(first_known_count, [26])
        self.assertEqual(len(self.new_fit_ids), 46)
        self.assertEqual(summary["model_instances"], 72)
        table = self.project / "outputs/final_thesis_v1/held_out/FINAL_MODEL_FITS.csv"
        with table.open(newline="", encoding="utf-8") as stream:
            ids = {row["trial_id"] for row in csv.DictReader(stream)}
        self.assertEqual(ids, {slot["trial_id"] for slot in self.slots})


if __name__ == "__main__":
    unittest.main()
