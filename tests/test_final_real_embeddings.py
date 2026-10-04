"""Generated-data tests for the frozen Real embedding export boundary.

These tests never read the monkey recording, trial split, or a neural checkpoint.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from neurobridge.sampling.f_windows import build_windows  # noqa: E402
import final_real_embeddings as exporter  # noqa: E402
from final_real_embeddings import (  # noqa: E402
    ExportError, WINDOW, TRIAL_LENGTH, authorize_real_export, centered_batches,
    channel_indices, encode_batch, export_slot, valid_centers,
)


class FinalRealEmbeddingTests(unittest.TestCase):
    def test_centered_w201_batches_equal_trusted_window_builder(self) -> None:
        rng = np.random.default_rng(19)
        spikes = rng.normal(size=(TRIAL_LENGTH, 4)).astype(np.float32)
        labels = np.asarray([3], dtype=np.int64)
        trusted, time_id, _, trial_id, _ = build_windows(
            spikes, WINDOW, 1, labels=labels, trial_len=TRIAL_LENGTH,
            time_mode="absolute", padding="center", pad_value=0.0,
        )
        batches = list(centered_batches(spikes, batch_size=37))
        actual_time = np.concatenate([times for times, _ in batches])
        actual_windows = np.concatenate([windows for _, windows in batches])
        np.testing.assert_array_equal(actual_time, time_id)
        np.testing.assert_array_equal(trial_id, np.zeros(TRIAL_LENGTH, dtype=np.int64))
        np.testing.assert_array_equal(actual_windows, trusted)
        self.assertEqual(actual_windows.shape, (600, 201, 4))
        validity = valid_centers(np.ones(TRIAL_LENGTH, dtype=bool))
        self.assertEqual(int(validity.sum()), 400)
        self.assertFalse(validity[99])
        self.assertTrue(validity[100])
        self.assertTrue(validity[499])
        self.assertFalse(validity[500])

    def test_channel_indices_follow_canonical_feature_order(self) -> None:
        perm = np.random.default_rng(42).permutation(65)
        a = perm[:32].tolist()
        b = perm[32:].tolist()
        part = {"A_PROXIMAL": {"feature_indices": a},
                "B_DISTAL": {"feature_indices": b}}
        self.assertEqual(channel_indices(part, "TOTAL65"), list(range(65)))
        self.assertEqual(channel_indices(part, "A_PROXIMAL"), a)
        self.assertEqual(channel_indices(part, "B_DISTAL"), b)
        self.assertEqual(set(a) | set(b), set(range(65)))
        self.assertFalse(set(a) & set(b))
        with self.assertRaises(ExportError):
            channel_indices(part, "unknown")

    def test_raw_unit_embedding_shape_and_normalization_invariants(self) -> None:
        class ThreeOutput(torch.nn.Module):
            def forward(self, x):
                return x[:, WINDOW // 2, :3] + 0.25

        rng = np.random.default_rng(7)
        windows = rng.normal(size=(9, WINDOW, 4)).astype(np.float32)
        raw, unit = encode_batch(ThreeOutput(), windows, torch.device("cpu"))
        self.assertEqual(raw.shape, (9, 3))
        self.assertEqual(unit.shape, (9, 3))
        self.assertEqual(raw.dtype, np.float32)
        self.assertEqual(unit.dtype, np.float32)
        self.assertTrue(np.isfinite(raw).all() and np.isfinite(unit).all())
        np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(unit, raw / np.linalg.norm(raw, axis=1)[:, None], atol=1e-6)

        class WrongDimension(torch.nn.Module):
            def forward(self, x):
                return x[:, WINDOW // 2, :2]

        with self.assertRaises(ExportError):
            encode_batch(WrongDimension(), windows, torch.device("cpu"))

    def test_export_authorization_rejects_absent_or_wrong_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary).resolve()
            spec = project / "outputs/final_thesis_v1/freeze/REAL_FINAL_EXPERIMENT_SPEC.json"
            spec.parent.mkdir(parents=True)
            spec.write_text('{"frozen":true}\n', encoding="utf-8")
            manifest = project / "outputs/final_thesis_v1/final_evaluation/core_metrics/EVALUATION_SUPPORT_MANIFEST.json"
            with self.assertRaises((FileNotFoundError, ExportError)):
                authorize_real_export(project, spec, manifest)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                '{"final_spec_sha256":"wrong","window_size":201,'
                '"embedding_scope":"all_centered_rows_with_valid_mask",'
                '"test_opening_authorized":true}\n', encoding="utf-8")
            with self.assertRaises(ExportError):
                authorize_real_export(project, spec, manifest)

            audit = manifest.parent / "AUDIT_MANIFEST.json"
            audit.write_text('{"source_artifact_hashes":{}}\n', encoding="utf-8")
            policy = {
                "final_spec_sha256": exporter.sha256(spec),
                "window_size": WINDOW,
                "embedding_scope": "all_centered_rows_with_valid_mask",
                "test_opening_authorized": True,
                "audit_manifest_sha256": exporter.sha256(audit),
            }
            manifest.write_text(exporter.json_bytes(policy).decode("utf-8"), encoding="utf-8")
            self.assertEqual(authorize_real_export(project, spec, manifest), policy)

            audit.write_text('{"source_artifact_hashes":{"changed":"yes"}}\n', encoding="utf-8")
            with self.assertRaises(ExportError):
                authorize_real_export(project, spec, manifest)

    def test_atomic_publish_status_and_immutable_export_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary).resolve()
            spec_path = project / exporter.SPEC
            part_path = project / exporter.PARTITION
            policy_path = project / "outputs/final_thesis_v1/freeze/EVALUATION_SUPPORT.json"
            for path in (spec_path, part_path, policy_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}\n', encoding="utf-8")
            fit_root = project / "mock_fit"
            fit_root.mkdir()
            checkpoint = fit_root / "best_validation_checkpoint.pt"
            checkpoint.write_bytes(b"generated checkpoint marker only")
            (fit_root / "trial_record.json").write_text('{}\n', encoding="utf-8")
            (fit_root / "result.json").write_text('{}\n', encoding="utf-8")
            permutation = np.random.default_rng(3).permutation(65)
            indices = permutation[:32].tolist()
            part = {"feature_order_nlb_unit_ids": list(range(65)),
                    "A_PROXIMAL": {"feature_indices": indices},
                    "B_DISTAL": {"feature_indices": permutation[32:].tolist()}}
            slot = {"trial_id": "generated-fit-1", "population": "A_PROXIMAL",
                    "architecture": "cnn1d", "objective": "soft", "training_seed_root": 1201}
            result = {"status": "INELIGIBLE_NEAR_COLLAPSE", "selected_update": 150,
                      "geometry": {"near_collapse": True}}
            config = {"architecture": "cnn1d"}
            record = {"config_sha256": "a" * 64}
            arrays = {
                "spikes": np.random.default_rng(4).normal(size=(TRIAL_LENGTH, 65)).astype(np.float32),
                "valid_bins": np.ones(TRIAL_LENGTH, dtype=bool),
                "target_by_trial": np.asarray([3], dtype=np.int64),
                "position": np.zeros((TRIAL_LENGTH, 2), dtype=np.float32),
                "velocity": np.zeros((TRIAL_LENGTH, 2), dtype=np.float32),
            }
            split = {"train": [0], "validation": [], "test": []}
            output = project / exporter.FINAL_ROOT / "embeddings" / slot["trial_id"]
            published_from_staging = []
            original_rename = Path.rename

            def inspect_rename(source, target):
                if Path(target) != output:
                    return original_rename(source, target)
                self.assertFalse(output.exists())
                self.assertIn(".staging-", source.name)
                self.assertEqual(Path(target), output)
                self.assertEqual({path.name for path in source.iterdir()}, {
                    "embedding_raw.npz", "embedding_unit.npz",
                    "evaluation_metadata.npz", "manifest.json"})
                published_from_staging.append(source)
                return original_rename(source, target)

            def generated_encode(_model, windows, _device):
                base = windows[:, WINDOW // 2, :3].astype(np.float32) + 0.5
                unit = base / np.linalg.norm(base, axis=1, keepdims=True)
                return base, unit

            with patch.object(exporter, "verified_checkpoint", return_value=(
                checkpoint, record, result, config)), patch.object(
                exporter, "load_encoder", return_value=object()), patch.object(
                exporter, "encode_batch", side_effect=generated_encode), patch.object(
                Path, "rename", autospec=True, side_effect=inspect_rename):
                actual = export_slot(project, slot, {}, part, arrays, split, {}, policy_path,
                                     batch_size=131, device=torch.device("cpu"))
            self.assertEqual(actual, output)
            self.assertEqual(len(published_from_staging), 1)
            self.assertFalse(published_from_staging[0].exists())
            manifest = exporter.read_json(output / "manifest.json")
            self.assertEqual(manifest["fit_status"], "INELIGIBLE_NEAR_COLLAPSE")
            self.assertTrue(manifest["near_collapse"])
            self.assertEqual(manifest["checkpoint_selected_update"], 150)
            with np.load(output / "embedding_raw.npz", allow_pickle=False) as raw, np.load(
                output / "embedding_unit.npz", allow_pickle=False) as unit, np.load(
                output / "evaluation_metadata.npz", allow_pickle=False) as metadata:
                self.assertEqual(raw["embedding_raw"].shape, (TRIAL_LENGTH, 3))
                self.assertEqual(unit["embedding_unit"].shape, (TRIAL_LENGTH, 3))
                self.assertEqual(len(metadata["trial_id"]), TRIAL_LENGTH)
                self.assertEqual(int(metadata["valid_mask"].sum()), 400)
            hashes_before = dict(manifest["artifact_sha256"])
            with patch.object(exporter, "verified_checkpoint", return_value=(
                checkpoint, record, result, config)), patch.object(
                exporter, "load_encoder", side_effect=AssertionError("immutable export re-encoded")):
                reused = export_slot(project, slot, {}, part, arrays, split, {}, policy_path,
                                     batch_size=13, device=torch.device("cpu"))
            self.assertEqual(reused, output)
            self.assertEqual(exporter.read_json(output / "manifest.json")["artifact_sha256"], hashes_before)
            self.assertEqual(list(output.parent.glob(f"{slot['trial_id']}.staging-*")), [])


if __name__ == "__main__":
    unittest.main()
