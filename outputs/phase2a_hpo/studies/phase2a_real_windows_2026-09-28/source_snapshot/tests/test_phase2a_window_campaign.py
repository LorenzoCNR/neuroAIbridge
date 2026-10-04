"""Fit-free checks for the versioned 21/41/121/201 Real campaign."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurobridge.experiments import phase2a_hpo as hpo  # noqa: E402
from neurobridge.experiments import phase2a_window_campaign as campaign  # noqa: E402
from neurobridge.experiments.phase2a_window_inputs import WINDOWS, PARENTS  # noqa: E402
from neurobridge.sampling.f_windows import build_windows  # noqa: E402


def test_amended_plan_count_reuse_and_sealed_pairing() -> None:
    data = campaign.plan(ROOT)
    slots = data["slots"]
    assert data["slot_count"] == 448
    assert data["reused_completed_synthetic_slots"] == 10
    assert data["new_fits_remaining"] == 438
    assert {row["config"]["candidate_index"] for row in slots} == {0, 1, 2, 3}
    for domain, count in (("synthetic", 64), *( (f"real_w{w}", 96) for w in WINDOWS)):
        subset = [row for row in slots if row["config"]["domain"] == domain]
        assert len(subset) == count
        assert {row["config"]["window_size"] for row in subset} == ({21} if domain == "synthetic" else {int(domain[6:])})
    candidates, checksum = hpo.load_sealed_candidates(ROOT / hpo.PROTOCOL_RELATIVE)
    assert checksum == data["candidate_sha256"]
    for objective in hpo.OBJECTIVES:
        for index in range(4):
            related = [row["config"] for row in slots if row["config"]["objective"] == objective
                       and row["config"]["candidate_index"] == index]
            assert len({tuple(config["exact_hyperparameter_strings"][key]
                              for key in ("lr", "weight_decay", "temperature")) for config in related}) == 1
            assert related[0]["exact_hyperparameter_strings"]["lr"] == candidates[objective][index].lr
    assert all(row["config"]["training_seed_effective"] == 1102 for row in slots
               if row["config"]["domain"] == "synthetic" and row["config"]["population"] == "B")
    assert all(row["config"]["training_seed_effective"] == 1101 for row in slots
               if row["config"]["domain"].startswith("real_w"))
    for row in slots:
        assert hpo.guarded_output_root(ROOT) in Path(row["output_path"]).resolve().parents


def test_reused_old_synthetic_hashes_and_interrupted_attempt() -> None:
    slots = campaign.plan(ROOT)["slots"]
    reused = [row for row in slots if row["source"] == "reused_verified_old_synthetic"]
    assert len(reused) == 10
    assert all(campaign._verify_reuse(ROOT, row)["status"] in
               {"ELIGIBLE", "INELIGIBLE_NEAR_COLLAPSE"} for row in reused)
    old_partial = (ROOT / hpo.OUTPUT_RELATIVE / "studies" / campaign.OLD_STUDY / "trials" /
                   "phase2a_frozen_2026-09-28-synthetic-A-transformer-soft-c1-s1101-65c9d2abb09c9730")
    assert (old_partial / "trial_record.json").is_file()
    assert not (old_partial / "result.json").exists()
    assert all(row["trial_id"] != old_partial.name for row in slots if row["source"] != "reused_verified_old_synthetic")


@pytest.mark.parametrize("window", WINDOWS)
def test_centered_window_constructor_never_crosses_trial(window: int) -> None:
    length = 600
    x = np.concatenate((np.full((length, 2), 1.0), np.full((length, 2), 2.0)))
    windows, time_id, global_id, trial_id, labels = build_windows(
        x, window, 1, labels=np.array([0, 1]), trial_len=length,
        time_mode="absolute", padding="center", pad_value=0.0,
    )
    radius = window // 2
    assert windows.shape == (2 * length, window, 2)
    assert np.all(windows[0, :radius] == 0.0)
    assert np.all(windows[length, :radius] == 0.0)
    assert np.all(windows[length - 1, radius + 1:] == 0.0)
    assert np.all(windows[length, radius:] == 2.0)
    assert np.all(windows[length - 1, :radius + 1] == 1.0)
    assert np.array_equal(trial_id[[0, length - 1, length, 2 * length - 1]], [0, 0, 1, 1])
    assert np.array_equal(time_id[[0, length - 1, length, 2 * length - 1]], [0, length - 1, 0, length - 1])


def test_all_real_safe_manifests_are_train_validation_only() -> None:
    channels, partition_sha = hpo.channel_partition(ROOT)
    frozen_split_pairs = []
    for window in WINDOWS:
        for population in PARENTS:
            root = ROOT / hpo.OUTPUT_RELATIVE / "safe_inputs" / f"real_w{window}" / population
            manifest = json.loads((root / "manifest.json").read_text())
            split = json.loads((root / "split.json").read_text())
            assert set(split) == {"train", "validation"}
            assert manifest["window_size"] == window
            assert manifest["channel_partition_sha256"] == partition_sha
            assert manifest["test_trial_intersection_empty"] is True
            assert manifest["all_windows_are_train_or_validation"] is True
            assert manifest["train_window_count"] == 80400
            assert manifest["validation_window_count"] == 12000
            assert manifest["valid_train_window_count"] == len(split["train"]) * (600 - window + 1)
            assert manifest["valid_validation_window_count"] == len(split["validation"]) * (600 - window + 1)
            assert manifest["safe_windows_sha256"] == hpo.file_sha256(root / "windows.npz")
            frozen_split_pairs.append((split["train"], split["validation"]))
    assert all(pair == frozen_split_pairs[0] for pair in frozen_split_pairs)


def test_real_w21_safe_values_identical_to_previous_v2_continuity_bundle() -> None:
    # This trusted test may inspect complete input-derived safe bundles; no test result is opened.
    for population in PARENTS:
        previous = ROOT / hpo.OUTPUT_RELATIVE / "safe_inputs" / "real" / population / "windows.npz"
        amended = ROOT / hpo.OUTPUT_RELATIVE / "safe_inputs" / "real_w21" / population / "windows.npz"
        with np.load(previous, allow_pickle=False) as a, np.load(amended, allow_pickle=False) as b:
            assert set(a.files) == set(b.files)
            assert all(np.array_equal(a[key], b[key]) for key in a.files)
