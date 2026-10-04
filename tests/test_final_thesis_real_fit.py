"""Boundary tests for the isolated final Real train/validation fit adapter.

The frozen optimization core is mocked: these tests perform no model fitting.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from neurobridge.experiments.phase2a_hpo import (  # noqa: E402
    ProtocolViolation, _canonical_json, file_sha256,
)
from final_thesis_real_fit import (  # noqa: E402
    SUCCESS_ARTIFACTS, fit_final_real_trial,
)


class FinalRealFitBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name).resolve()
        self.spec_path = (self.project / "outputs/final_thesis_v1/freeze/"
                          "REAL_FINAL_EXPERIMENT_SPEC.json")
        self.spec_path.parent.mkdir(parents=True)
        self.alias = (self.project / "outputs/phase2a_hpo/safe_inputs/real_somatotopic_v1/"
                      "real_w21/A_PROXIMAL/reference_manifest.json")
        self.alias.parent.mkdir(parents=True)
        self.alias.write_text('{"safe":true}\n', encoding="utf-8")
        self.bundle = self.project / "safe_bundle"
        self.bundle.mkdir()
        (self.bundle / "manifest.json").write_text('{}\n', encoding="utf-8")
        self.config = {
            "domain": "real_w21", "population": "A_PROXIMAL", "branch": "held_out",
            "window_size": 21, "batch_size": 1024, "embedding_dim": 3,
            "positive_offset": 10, "imposed_shift_bins": 0,
            "architecture": "cnn1d", "objective": "soft",
            "candidate_index": 2, "training_seed_root": 1201,
            "channel_partition_sha256": "a" * 64,
            "canonical_design_sha256": "b" * 64,
            "schedule": {"max_updates": 4000},
        }
        self.trial = {
            "trial_id": "final-real-check",
            "config": self.config,
            "config_sha256": hashlib.sha256(
                _canonical_json(self.config).encode("utf-8")).hexdigest(),
            "output_path": str(self.project / "outputs/final_thesis_v1/held_out/trials/final-real-check"),
            "safe_input_reference": str(self.alias),
            "safe_input_reference_sha256": file_sha256(self.alias),
        }
        self.frozen_slot = {
            **{key: self.trial[key] for key in (
                "trial_id", "config_sha256", "output_path", "safe_input_reference",
                "safe_input_reference_sha256")},
            **{key: self.config[key] for key in (
                "architecture", "objective", "population", "candidate_index",
                "training_seed_root")},
            "reuse_hpo_checkpoint": False,
        }
        self._write_spec({"selected_real_window_size": 21,
                          "final_slots": [self.frozen_slot]})
        self.safe_manifest = {
            "safe_windows_sha256": "c" * 64,
            "safe_split_sha256": "d" * 64,
            "original_split_sha256": "e" * 64,
            "raw_source_sha256": "f" * 64,
        }

    def _write_spec(self, value) -> None:
        self.spec_path.write_text(_canonical_json(value) + "\n", encoding="utf-8")
        self.spec_sha256 = file_sha256(self.spec_path)

    def _safe_bundle(self, *args):
        return self.bundle, self.alias, self.safe_manifest, {}, None, {}

    def test_rejects_wrong_spec_before_safe_bundle_or_fit(self) -> None:
        with patch("final_thesis_real_fit.load_safe_real_bundle") as loader, patch(
            "final_thesis_real_fit._fit_new") as trainer:
            with self.assertRaises(ProtocolViolation):
                fit_final_real_trial(self.project, self.trial, "0" * 64)
            loader.assert_not_called()
            trainer.assert_not_called()

    def test_rejects_v2_output_and_changed_safe_reference(self) -> None:
        outside = dict(self.trial, output_path=str(self.project / "outputs/runs/old/model"))
        with patch("final_thesis_real_fit.load_safe_real_bundle") as loader:
            with self.assertRaises(ProtocolViolation):
                fit_final_real_trial(self.project, outside, self.spec_sha256)
            loader.assert_not_called()
        changed = dict(self.trial, safe_input_reference_sha256="0" * 64)
        with patch("final_thesis_real_fit.load_safe_real_bundle") as loader:
            with self.assertRaises(ProtocolViolation):
                fit_final_real_trial(self.project, changed, self.spec_sha256)
            loader.assert_not_called()

    def test_rejects_changed_config_and_non_held_out_branch(self) -> None:
        changed = dict(self.trial, config={**self.config, "window_size": 41})
        with patch("final_thesis_real_fit.load_safe_real_bundle") as loader:
            with self.assertRaises(ProtocolViolation):
                fit_final_real_trial(self.project, changed, self.spec_sha256)
            loader.assert_not_called()
        config = {**self.config, "branch": "full_sample"}
        changed = dict(self.trial, config=config, config_sha256=hashlib.sha256(
            _canonical_json(config).encode("utf-8")).hexdigest())
        with patch("final_thesis_real_fit.load_safe_real_bundle") as loader:
            with self.assertRaises(ProtocolViolation):
                fit_final_real_trial(self.project, changed, self.spec_sha256)
            loader.assert_not_called()

    def test_rejects_out_of_spec_and_reused_slot_before_bundle_load(self) -> None:
        unauthorized_config = {**self.config, "training_seed_root": 1301}
        unauthorized = dict(self.trial, config=unauthorized_config,
                            config_sha256=hashlib.sha256(
                                _canonical_json(unauthorized_config).encode("utf-8")).hexdigest())
        with patch("final_thesis_real_fit.load_safe_real_bundle") as loader:
            with self.assertRaises(ProtocolViolation):
                fit_final_real_trial(self.project, unauthorized, self.spec_sha256)
            loader.assert_not_called()
        self._write_spec({"selected_real_window_size": 21,
                          "final_slots": [{**self.frozen_slot, "reuse_hpo_checkpoint": True}]})
        with patch("final_thesis_real_fit.load_safe_real_bundle") as loader:
            with self.assertRaises(ProtocolViolation):
                fit_final_real_trial(self.project, self.trial, self.spec_sha256)
            loader.assert_not_called()

    def test_immutable_result_reuse_and_child_hash_validation(self) -> None:
        def fake_fit(target, *_args):
            for name in SUCCESS_ARTIFACTS:
                (target / name).write_bytes(name.encode("utf-8"))
            return {"trial_id": self.trial["trial_id"], "status": "ELIGIBLE",
                    "artifact_sha256": {name: file_sha256(target / name)
                                        for name in SUCCESS_ARTIFACTS}}

        with patch("final_thesis_real_fit.load_safe_real_bundle", side_effect=self._safe_bundle), patch(
            "final_thesis_real_fit._fit_new", side_effect=fake_fit) as trainer:
            first = fit_final_real_trial(self.project, self.trial, self.spec_sha256)
            second = fit_final_real_trial(self.project, self.trial, self.spec_sha256)
            self.assertEqual(first, second)
            self.assertEqual(trainer.call_count, 1)
            record_path = Path(self.trial["output_path"]) / "trial_record.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["final_spec_sha256"], self.spec_sha256)
            self.assertEqual(record["safe_reference_sha256"], file_sha256(self.alias))
            self.assertEqual(record["original_split_sha256"], "e" * 64)
            self.assertEqual(record["channel_partition_sha256"], "a" * 64)
            damaged = Path(self.trial["output_path"]) / "validation_embedding.npz"
            damaged.write_bytes(b"changed")
            with self.assertRaises(ProtocolViolation):
                fit_final_real_trial(self.project, self.trial, self.spec_sha256)
            self.assertEqual(trainer.call_count, 1)

    def test_smoke_is_isolated_and_performs_no_real_training(self) -> None:
        with patch("final_thesis_real_fit.load_safe_real_bundle", side_effect=self._safe_bundle), patch(
            "final_thesis_real_fit._fit_new", return_value={
                "trial_id": self.trial["trial_id"], "status": "FAILED_RUNTIME", "artifact_sha256": {},
            }) as trainer:
            fit_final_real_trial(self.project, self.trial, self.spec_sha256, smoke_updates=1)
            self.assertEqual(trainer.call_count, 1)
            self.assertTrue((self.project / "outputs/final_thesis_v1/smoke/"
                             "final-real-check-u1/result.json").is_file())
            self.assertFalse(Path(self.trial["output_path"]).exists())


if __name__ == "__main__":
    unittest.main()
