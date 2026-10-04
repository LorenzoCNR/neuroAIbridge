"""Read-only gates for the frozen Real HPO continuation."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from neurobridge.experiments.phase2a_hpo import (  # noqa: E402
    ProtocolViolation, SEARCH_SEED, validate_outcome,
)
from final_thesis_real_hpo import (  # noqa: E402
    POPULATIONS, _checked_parent, _new_slot, _root, build_extension_plan,
)


class FinalThesisRealHpoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parent, _, cls.partition, cls.checked = _checked_parent(ROOT)
        cls.plan = build_extension_plan(ROOT)

    def test_initial_window_ledgers_are_strictly_separate(self) -> None:
        self.assertEqual(set(self.checked["ledgers"]),
                         {"real_w21", "real_w41", "real_w121", "real_w201"})
        for rows in self.checked["ledgers"].values():
            self.assertEqual(len(rows), 96)
            self.assertEqual({row["training_seed_root"] for row in rows}, {SEARCH_SEED})
            for row in rows:
                validate_outcome(row)

    def test_extension_is_paired_and_population_complete(self) -> None:
        self.assertEqual(self.plan["slot_count"], 108)
        self.assertEqual(len(self.plan["slots"]), 108)
        self.assertEqual(sum(self.plan["paired_triggers"].values()), 9)
        counts = Counter((s["config"]["domain"], s["config"]["objective"],
                          s["config"]["architecture"], s["config"]["candidate_index"])
                         for s in self.plan["slots"])
        for key, triggered in self.plan["paired_triggers"].items():
            window, objective = key.split("/")
            for architecture in ("cnn1d", "transformer"):
                for candidate in (4, 5):
                    self.assertEqual(counts[(window, objective, architecture, candidate)],
                                     len(POPULATIONS) if triggered else 0)
        self.assertEqual({s["config"]["training_seed_root"] for s in self.plan["slots"]},
                         {SEARCH_SEED})
        self.assertFalse(self.plan["test_or_scientific_outcomes_used"])
        self.assertFalse(self.plan["v2_write"])

    def test_all_paths_are_new_hpo_namespace_and_safe_inputs(self) -> None:
        namespace = _root(ROOT).resolve()
        for slot in self.plan["slots"]:
            self.assertIn(namespace, Path(slot["output_path"]).resolve().parents)
            self.assertIn("safe_inputs", Path(slot["safe_input_reference"]).parts)
            self.assertNotIn("test", Path(slot["safe_input_reference"]).parts)

    def test_baseline_config_matches_initial_candidate_zero(self) -> None:
        original = next(slot for slot in self.parent["slots"] if
                        slot["config"]["domain"] == "real_w21" and
                        slot["config"]["population"] == "TOTAL65" and
                        slot["config"]["architecture"] == "cnn1d" and
                        slot["config"]["objective"] == "soft" and
                        slot["config"]["candidate_index"] == 0)
        replacement = _new_slot(ROOT, self.partition, self.checked["candidates"],
                                "real_w21", "cnn1d", "soft", 0, SEARCH_SEED,
                                "TOTAL65", "extension")
        self.assertEqual(replacement["config"], original["config"])
        self.assertEqual(replacement["config_sha256"], original["config_sha256"])

    def test_selection_rejects_scientific_test_fields(self) -> None:
        row = dict(self.checked["ledgers"]["real_w21"][0])
        row["test_decoding"] = 1.0
        with self.assertRaises(ProtocolViolation):
            validate_outcome(row)


if __name__ == "__main__":
    unittest.main()
