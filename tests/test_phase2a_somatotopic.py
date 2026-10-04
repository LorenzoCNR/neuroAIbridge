"""Qualification tests for the versioned Real somatotopic HPO boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from neurobridge.experiments.phase2a_hpo import file_sha256  # noqa: E402
from phase2a_somatotopic_campaign import _checked_design, plan, safe_references  # noqa: E402
from phase2a_somatotopic_extract import VERSION, partition  # noqa: E402
from phase2a_somatotopic_fit import load_safe_real_bundle  # noqa: E402


def test_partition_exact_cover_units_and_electrodes():
    value, digest = partition(ROOT)
    ids = value["feature_order_nlb_unit_ids"]
    a = value["A_PROXIMAL"]["feature_indices"]
    b = value["B_DISTAL"]["feature_indices"]
    assert len(a) == 32 and len(b) == 33
    assert not set(a) & set(b)
    assert set(a + b) == set(range(65))
    assert [ids[i] for i in a] == value["A_PROXIMAL"]["nlb_unit_ids"]
    assert [ids[i] for i in b] == value["B_DISTAL"]["nlb_unit_ids"]
    assert not {ids[i] // 100 for i in a} & {ids[i] // 100 for i in b}
    assert digest == file_sha256(ROOT / "outputs/phase2a_hpo/partitions/real_somatotopic_v1.json")


def test_canonical_design_and_historical_random_preservation():
    part, digest, design, design_sha = _checked_design(ROOT)
    assert part["partition_version"] == VERSION
    assert design["partition_sha256"] == digest
    assert design["real_window_sizes_bins"] == [21, 41, 121, 201]
    assert design["positive_offset_bins"] == 10
    assert len(design_sha) == 64
    assert (ROOT / part["historical_random_partition"]).is_file()


def test_all_somatotopic_bundles_preserve_train_validation_firewall():
    _, partition_sha, _, design_sha = _checked_design(ROOT)
    safe_references(ROOT)
    hpo = ROOT / "outputs/phase2a_hpo"
    for window in (21, 41, 121, 201):
        total = json.loads((hpo / "safe_inputs" / f"real_w{window}" / "TOTAL65" / "manifest.json").read_text())
        for population, size in (("A_PROXIMAL", 32), ("B_DISTAL", 33)):
            root = hpo / "safe_inputs" / VERSION / f"real_w{window}" / population
            manifest = json.loads((root / "manifest.json").read_text())
            split = json.loads((root / "split.json").read_text())
            assert manifest["partition_sha256"] == partition_sha
            assert len(manifest["channel_indices"]) == size
            assert manifest["train_trial_ids"] == total["train_trial_ids"] == split["train"]
            assert manifest["validation_trial_ids"] == total["validation_trial_ids"] == split["validation"]
            assert manifest["safe_windows_sha256"] == file_sha256(root / "windows.npz")
            assert manifest["safe_split_sha256"] == file_sha256(root / "split.json")
            assert manifest["test_trial_intersection_empty"]
            assert manifest["all_windows_are_train_or_validation"]
            assert manifest["train_window_count"] == total["train_window_count"]
            assert manifest["validation_window_count"] == total["validation_window_count"]
    root, alias, manifest, split, dataset, values = load_safe_real_bundle(
        ROOT, "real_w21", "A_PROXIMAL", partition_sha, design_sha)
    assert root.is_dir() and alias.is_file()
    assert len(dataset) == len(values["trial_id"]) == 92400
    assert set(np.unique(values["split"])) == {"train", "validation"}
    assert set(np.unique(values["trial_id"])) == set(split["train"] + split["validation"])
    assert manifest["test_trial_intersection_empty"]


def test_384_slot_plan_pairs_candidates_and_keeps_test_firewall():
    campaign = plan(ROOT)
    slots = campaign["slots"]
    assert len(slots) == 384 and not campaign["historical_random_partition_used"]
    _, partition_sha, _, design_sha = _checked_design(ROOT)
    for window in (21, 41, 121, 201):
        subset = [s for s in slots if s["config"]["domain"] == f"real_w{window}"]
        assert len(subset) == 96
        for objective in ("soft", "infonce", "time_contrastive_blocks", "behavior_contrastive_blocks"):
            for candidate in (0, 1, 2, 3):
                matching = [s for s in subset if s["config"]["objective"] == objective and
                            s["config"]["candidate_index"] == candidate]
                assert len(matching) == 6
                assert len({tuple(s["config"]["exact_hyperparameter_strings"].values()) for s in matching}) == 1
                assert {s["config"]["population"] for s in matching} == {"TOTAL65", "A_PROXIMAL", "B_DISTAL"}
                assert {s["config"]["training_seed_root"] for s in matching} == {1101}
                assert {s["config"]["channel_partition_sha256"] for s in matching} == {partition_sha}
                assert {s["config"]["canonical_design_sha256"] for s in matching} == {design_sha}
                assert all(s["config"]["positive_offset"] == 10 for s in matching)
                assert all("test" not in Path(s["safe_input_reference"]).parts for s in matching)
                assert all("outputs/runs" not in s["output_path"].replace("\\", "/") for s in matching)


def test_fit_adapter_cannot_name_v2_parent_or_complete_split():
    source = (ROOT / "tools/phase2a_somatotopic_fit.py").read_text(encoding="utf-8")
    assert "PARENT_RELATIVE" not in source
    assert "stage01_data" not in source
    assert "outputs/runs" not in source
    assert "phase2a_somatotopic_extract" not in source
