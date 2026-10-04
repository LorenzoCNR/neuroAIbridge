"""Checks for the approved seed-1101 Real freeze without fitting models."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from final_thesis_freeze_real import (  # noqa: E402
    ARCHITECTURES, OBJECTIVES, WINDOWS, _window_rows, build_freeze,
)


class RealFreezeTests(unittest.TestCase):
    def test_equal_cell_scores_get_unique_shorter_first_ranks(self) -> None:
        rows = [{"window_size": window, "architecture": architecture,
                 "objective": objective, "candidate_index": 0,
                 "validation_mean_3pop": 1.0}
                for window in WINDOWS for architecture in ARCHITECTURES
                for objective in OBJECTIVES]
        table, chosen, aggregates = _window_rows(rows, dict.fromkeys(WINDOWS, 100))
        self.assertEqual(chosen, 21)
        self.assertEqual(len(table), 32)
        self.assertEqual([aggregates[window]["rank_mean"] for window in WINDOWS],
                         [1.0, 2.0, 3.0, 4.0])

    def test_real_freeze_is_72_instances_with_24_immutable_reuses(self) -> None:
        winner_csv, window_csv, spec = build_freeze(ROOT)
        self.assertEqual(len(winner_csv.decode("utf-8").splitlines()), 33)
        self.assertEqual(len(window_csv.decode("utf-8").splitlines()), 33)
        self.assertIn(spec["selected_real_window_size"], WINDOWS)
        self.assertEqual(spec["final_neural_model_instances"], 72)
        self.assertEqual(spec["reused_seed1101_hpo_checkpoints"], 24)
        self.assertEqual(spec["new_seed1201_1301_fits"], 48)
        self.assertEqual(spec["final_training_seed_roots"], [1101, 1201, 1301])
        slots = spec["final_slots"]
        self.assertEqual(len(slots), 72)
        self.assertEqual(len({slot["trial_id"] for slot in slots}), 72)
        self.assertEqual(Counter(slot["training_seed_root"] for slot in slots),
                         {1101: 24, 1201: 24, 1301: 24})
        self.assertEqual(sum(slot["reuse_hpo_checkpoint"] for slot in slots), 24)
        self.assertTrue(spec["no_test_or_scientific_metric_used_for_selection"])
        self.assertFalse(spec["multiseed_hyperparameter_selection_robustness_performed"])


if __name__ == "__main__":
    unittest.main()
