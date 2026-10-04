"""Trusted-extraction equivalence and fit-only firewall tests; no model fit."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurobridge.experiments import phase2a_hpo as hpo  # noqa: E402
from neurobridge.experiments import phase2a_safe_fit as fit  # noqa: E402
from neurobridge.experiments.phase2a_safe_inputs import PARENTS, extract_one  # noqa: E402


@pytest.mark.parametrize("domain,population", list(PARENTS))
def test_safe_bundle_equals_frozen_train_validation_rows(domain: str, population: str) -> None:
    """This test alone is in the trusted boundary and may open full V2 inputs."""
    manifest = extract_one(ROOT, domain, population)
    parent_windows, parent_split, _ = (ROOT / p for p in PARENTS[(domain, population)])
    original_split = json.loads(parent_split.read_text(encoding="utf-8"))
    assert manifest["train_trial_ids"] == original_split["train"]
    assert manifest["validation_trial_ids"] == original_split["validation"]
    assert manifest["original_split_sha256"] == hpo.file_sha256(parent_split)
    safe = ROOT / hpo.OUTPUT_RELATIVE / "safe_inputs" / domain / population / "windows.npz"
    with np.load(parent_windows, allow_pickle=False) as original, np.load(safe, allow_pickle=False) as filtered:
        keep = np.isin(original["trial_id"], original_split["train"] + original_split["validation"])
        assert not np.isin(filtered["trial_id"], original_split["test"]).any()
        assert len(filtered["trial_id"]) == int(keep.sum())
        for key in original.files:
            if key != "split":
                assert np.array_equal(filtered[key], original[key][keep]), key
        assert set(filtered["split"].tolist()) == {"train", "validation"}
    assert manifest["test_trial_intersection_empty"] is True


def test_fit_loader_rejects_test_metadata_and_bad_hash(tmp_path: Path) -> None:
    source = ROOT / hpo.OUTPUT_RELATIVE / "safe_inputs/synthetic/A"
    target = tmp_path / hpo.OUTPUT_RELATIVE / "safe_inputs/synthetic/A"
    target.mkdir(parents=True)
    for name in ("windows.npz", "split.json", "manifest.json"):
        shutil.copyfile(source / name, target / name)
    fit.load_safe_bundle(tmp_path, "synthetic", "A")
    bad_split = json.loads((target / "split.json").read_text())
    bad_split["test"] = [999]
    (target / "split.json").write_text(json.dumps(bad_split))
    with pytest.raises(hpo.ProtocolViolation, match="content hash mismatch"):
        fit.load_safe_bundle(tmp_path, "synthetic", "A")
    (target / "split.json").write_bytes((source / "split.json").read_bytes())
    bad_manifest = json.loads((target / "manifest.json").read_text())
    bad_manifest["validation_trial_ids"] = bad_manifest["validation_trial_ids"][:-1]
    (target / "manifest.json").write_text(json.dumps(bad_manifest))
    with pytest.raises(hpo.ProtocolViolation, match="safe split may contain only"):
        fit.load_safe_bundle(tmp_path, "synthetic", "A")


def test_checkpoint_and_patience_are_separate() -> None:
    schedule = hpo.SCHEDULES["real"]
    state = (float("inf"), None, None, 0)
    # Best can be selected before the 750-update patience gate.
    state = fit.early_stop_transition(*state, 6.2, 150, schedule)[:4]
    assert state[:2] == (6.2, 150) and state[2] is None
    state = fit.early_stop_transition(*state, 6.0, 750, schedule)[:4]
    assert state[:3] == (6.0, 750, 6.0)
    # A small raw improvement updates best but not meaningful reference.
    state = fit.early_stop_transition(*state, 5.999, 900, schedule)[:4]
    assert state == (5.999, 900, 6.0, 1)


def test_failure_accounting_resume_and_invalidation_without_fit(tmp_path: Path, monkeypatch) -> None:
    source = ROOT / hpo.OUTPUT_RELATIVE / "safe_inputs/synthetic/A"
    target = tmp_path / hpo.OUTPUT_RELATIVE / "safe_inputs/synthetic/A"
    target.mkdir(parents=True)
    for name in ("windows.npz", "split.json", "manifest.json"):
        shutil.copyfile(source / name, target / name)
    plan = json.loads((ROOT / hpo.OUTPUT_RELATIVE / "dry_run/initial_plan.json").read_text())
    trial = copy.deepcopy(next(row for row in plan["initial_trials"] if row["config"]["domain"] == "synthetic"
                               and row["config"]["population"] == "A"))
    trial["output_path"] = str(tmp_path / hpo.OUTPUT_RELATIVE / "studies/test/trials" / trial["trial_id"])
    def fail_without_training(*args, **kwargs):
        raise RuntimeError("synthetic pre-fit test failure")
    monkeypatch.setattr(fit, "_fit_new", fail_without_training)
    first = fit.fit_trial(tmp_path, trial)
    assert first["status"] == "FAILED_RUNTIME"
    assert fit.fit_trial(tmp_path, trial) == first
    altered = copy.deepcopy(trial)
    altered["config_sha256"] = "0" * 64
    with pytest.raises(hpo.ProtocolViolation, match="immutable provenance"):
        fit.fit_trial(tmp_path, altered)
    with pytest.raises(hpo.ProtocolViolation, match="output escaped"):
        altered["output_path"] = str(tmp_path / "outputs/runs/forbidden")
        fit.fit_trial(tmp_path, altered)


def test_sealed_plan_seed_and_population_pairing_still_hold() -> None:
    candidates, checksum = hpo.load_sealed_candidates(ROOT / hpo.PROTOCOL_RELATIVE)
    assert checksum == hpo.SEALED_CANDIDATE_SHA256
    plan = json.loads((ROOT / hpo.OUTPUT_RELATIVE / "dry_run/initial_plan.json").read_text())["initial_trials"]
    assert len(plan) == 160 and {x["config"]["candidate_index"] for x in plan} == {0, 1, 2, 3}
    for objective in hpo.OBJECTIVES:
        for index in range(4):
            triples = {tuple(x["config"]["exact_hyperparameter_strings"][key]
                             for key in ("lr", "weight_decay", "temperature")) for x in plan
                       if x["config"]["objective"] == objective and x["config"]["candidate_index"] == index}
            assert len(triples) == 1
            assert candidates[objective][index].numeric() == {
                "learning_rate": float(next(iter(triples))[0]),
                "weight_decay": float(next(iter(triples))[1]),
                "active_temperature": float(next(iter(triples))[2]),
            }
    assert {x["config"]["training_seed_effective"] for x in plan
            if x["config"]["domain"] == "synthetic" and x["config"]["population"] == "B"} == {1102}
    assert {x["config"]["training_seed_effective"] for x in plan
            if x["config"]["domain"] == "real"} == {1101}


def test_frozen_parent_data_protocol_and_real_channel_mapping() -> None:
    channels, _ = hpo.channel_partition(ROOT)
    real_train_validation = []
    for (domain, population), (window_relative, split_relative, _) in PARENTS.items():
        windows = ROOT / window_relative
        metadata_name = "metadata.json" if domain == "real" else "config.json"
        metadata = json.loads((windows.parent / metadata_name).read_text())
        config = metadata["config"] if domain == "real" else metadata
        assert config["window_size"] == 21 and config["stride"] == 1
        assert config["split_seed"] == 42
        if domain == "real":
            assert config["setting_name"] == "real_0"
            assert config["imposed_shift_bins"] == 0
            assert config["channel_split_seed"] == 42
            assert config["population_name"] == ("total_65" if population == "TOTAL65" else population)
            assert config["channel_indices"] in (list(channels[population]), None if population == "TOTAL65" else [])
            frozen_split = json.loads((ROOT / split_relative).read_text())
            real_train_validation.append((frozen_split["train"], frozen_split["validation"]))
        else:
            assert config["seed"] == 42
    assert real_train_validation[0] == real_train_validation[1] == real_train_validation[2]
