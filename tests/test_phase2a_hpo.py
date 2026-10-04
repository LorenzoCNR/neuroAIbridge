"""Fit-free tests of the frozen Phase-2A HPO boundary."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurobridge.experiments import phase2a_hpo as hpo  # noqa: E402


@pytest.fixture
def miniature_project(tmp_path: Path) -> Path:
    for relative in (hpo.PROTOCOL_RELATIVE, hpo.CHANNEL_PARTITION_RELATIVE,
                     Path("src/neurobridge/experiments/phase2a_hpo.py"),
                     Path("tools/phase2a_hpo.py")):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    return tmp_path


def test_sealed_candidates_checksum_and_baseline(miniature_project: Path) -> None:
    path = miniature_project / hpo.PROTOCOL_RELATIVE
    table, checksum = hpo.load_sealed_candidates(path)
    assert checksum == hpo.SEALED_CANDIDATE_SHA256
    assert list(table) == list(hpo.OBJECTIVES)
    assert all([candidate.index for candidate in table[objective]] == list(range(6))
               for objective in hpo.OBJECTIVES)
    for objective in hpo.OBJECTIVES:
        baseline = table[objective][0]
        assert baseline.lr == "0.001" and baseline.weight_decay == "0.0001"
        assert baseline.temperature == ("0.1" if objective in {"soft", "infonce"} else "1.0")
    altered = path.read_text(encoding="utf-8").replace("0.0010717221240559393", "0.0010717221240559394")
    path.write_text(altered, encoding="utf-8")
    with pytest.raises(hpo.ProtocolViolation, match="checksum"):
        hpo.load_sealed_candidates(path)


def test_plan_count_pairing_population_sharing_and_seeds(miniature_project: Path) -> None:
    table, _ = hpo.load_sealed_candidates(miniature_project / hpo.PROTOCOL_RELATIVE)
    channels, _ = hpo.channel_partition(miniature_project)
    plan = hpo.plan_initial(miniature_project, table, channels)
    assert len(plan) == 160
    assert len({trial["trial_id"] for trial in plan}) == 160
    assert {trial["config"]["candidate_index"] for trial in plan} == {0, 1, 2, 3}
    assert not {4, 5} & {trial["config"]["candidate_index"] for trial in plan}
    for objective in hpo.OBJECTIVES:
        for index in range(4):
            related = [trial for trial in plan if trial["config"]["objective"] == objective
                       and trial["config"]["candidate_index"] == index]
            assert len(related) == 10  # 2 architectures x (2 synthetic + 3 real populations)
            assert len({tuple(sorted(trial["config"]["exact_hyperparameter_strings"].items()))
                        for trial in related}) == 1
    synthetic = [trial["config"] for trial in plan if trial["config"]["domain"] == "synthetic"]
    real = [trial["config"] for trial in plan if trial["config"]["domain"] == "real"]
    assert {c["training_seed_effective"] for c in synthetic if c["population"] == "A"} == {1101}
    assert {c["training_seed_effective"] for c in synthetic if c["population"] == "B"} == {1102}
    assert {c["training_seed_effective"] for c in real} == {1101}
    assert {c["dataset_generator_seed"] for c in synthetic} == {42}
    assert {c["split_seed"] for c in synthetic + real} == {42}
    assert {c["channel_partition_seed"] for c in real} == {42}
    assert {len(c["channel_indices"]) for c in real if c["population"] == "TOTAL65"} == {65}
    assert {len(c["channel_indices"]) for c in real if c["population"] == "A"} == {32}
    assert {len(c["channel_indices"]) for c in real if c["population"] == "B"} == {33}
    assert {tuple(c["schedule"].items()) for c in synthetic} != {tuple(c["schedule"].items()) for c in real}
    assert all("test" not in Path(trial["safe_input_manifest_planned"]).parts for trial in plan)
    assert all(Path(trial["output_path"]).is_relative_to(miniature_project / hpo.OUTPUT_RELATIVE)
               for trial in plan)


def _outcomes(domain: str, objective: str, architecture: str, indices: tuple[int, ...],
              seeds: tuple[int, ...] = (1101,)) -> list[dict[str, object]]:
    rows = []
    for index in indices:
        for seed in seeds:
            for population in hpo.DOMAINS[domain]:
                rows.append({"domain": domain, "population": population,
                             "architecture": architecture, "objective": objective,
                             "candidate_index": index, "training_seed_root": seed,
                             "status": "ELIGIBLE", "validation_loss": 1.0 + index * 0.01,
                             "selected_update": 150, "stopping_update": 1500})
    return rows


def test_extension_and_finalist_rules() -> None:
    rows = _outcomes("real", "soft", "cnn1d", (0, 1))
    rows += _outcomes("real", "soft", "transformer", (0,))
    assert hpo.extension_required(rows, "real", "soft")
    rows += _outcomes("real", "soft", "transformer", (1,))
    assert not hpo.extension_required(rows, "real", "soft")
    assert hpo.finalist_indices(rows, "real", "cnn1d", "soft", False) == (0, 1)
    assert hpo.finalist_indices(rows, "real", "cnn1d", "soft", True) == (0, 1)
    assert hpo.select_finalist(rows, "real", "cnn1d", "soft", (0, 1)) is None  # extra seeds absent
    rows += _outcomes("real", "soft", "cnn1d", (0, 1), (1201, 1301))
    assert hpo.select_finalist(rows, "real", "cnn1d", "soft", (0, 1)) == 0
    failed = copy.deepcopy(rows)
    next(row for row in failed if row["architecture"] == "cnn1d" and
         row["candidate_index"] == 0 and row["training_seed_root"] == 1301)["status"] = "INELIGIBLE_NEAR_COLLAPSE"
    assert hpo.select_finalist(failed, "real", "cnn1d", "soft", (0, 1)) == 1


def test_validation_geometry_three_way_and_firewall() -> None:
    rng = np.random.default_rng(7)
    concentrated = np.tile(np.array([[1.0, 0.0, 0.0]]), (600, 1))
    concentrated += rng.normal(scale=1e-4, size=concentrated.shape)
    concentrated /= np.linalg.norm(concentrated, axis=1, keepdims=True)
    metadata = np.arange(600)
    valid = np.ones(600, dtype=bool)
    splits = np.repeat("validation", 600)
    collapsed = hpo.validation_geometry(concentrated, metadata, metadata, valid, splits)
    assert collapsed["near_collapse"] is True
    assert collapsed["pair_sample_rows"] == 512
    dispersed = rng.normal(size=(600, 3))
    dispersed /= np.linalg.norm(dispersed, axis=1, keepdims=True)
    assert hpo.validation_geometry(dispersed, metadata, metadata, valid, splits)["near_collapse"] is False
    with pytest.raises(hpo.ProtocolViolation, match="non-validation"):
        hpo.validation_geometry(concentrated, metadata, metadata, valid, np.repeat("test", 600))
    row = _outcomes("real", "soft", "cnn1d", (0,))[0]
    row["test_decoding"] = 0.9
    with pytest.raises(hpo.ProtocolViolation, match="forbidden"):
        hpo.validate_outcome(row)


def test_safe_input_contract_and_v2_write_protection(miniature_project: Path) -> None:
    safe_root = miniature_project / hpo.OUTPUT_RELATIVE / "safe_inputs"
    manifest = {"domain": "synthetic", "population": "A", "train_trial_ids": [1, 2],
                "validation_trial_ids": [3], "windows_path": str(safe_root / "synthetic/A/windows.npz"),
                "windows_sha256": "a" * 64, "split_sha256": "b" * 64,
                "raw_source_sha256": "c" * 64, "dataset_sha256": "d" * 64}
    hpo.check_safe_input_manifest(manifest, safe_root)
    with pytest.raises(hpo.ProtocolViolation, match="forbidden"):
        hpo.check_safe_input_manifest({**manifest, "test_trial_ids": [4]}, safe_root)
    with pytest.raises(hpo.ProtocolViolation, match="train/validation trial overlap"):
        hpo.check_safe_input_manifest({**manifest, "validation_trial_ids": [2]}, safe_root)
    with pytest.raises(hpo.ProtocolViolation, match="safe window data"):
        hpo.check_safe_input_manifest({**manifest, "windows_path": str(miniature_project / "outputs/runs/x.npz")}, safe_root)
    with pytest.raises(hpo.ProtocolViolation, match="restricted"):
        hpo.guarded_output_root(miniature_project, miniature_project / "outputs/runs")
    root = hpo.guarded_output_root(miniature_project)
    assert hpo.immutable_manifest(root / "trial/manifest.json", {"config": 1}, root) == "created"
    assert hpo.immutable_manifest(root / "trial/manifest.json", {"config": 1}, root) == "reused"
    with pytest.raises(hpo.ProtocolViolation, match="differs"):
        hpo.immutable_manifest(root / "trial/manifest.json", {"config": 2}, root)
    with pytest.raises(hpo.ProtocolViolation, match="outside"):
        hpo.immutable_manifest(miniature_project / "outputs/runs/model.json", {"config": 1}, root)


def test_dry_run_integration_no_fit_or_test_access(miniature_project: Path) -> None:
    result = hpo.dry_run(miniature_project)
    assert result["initial_trial_count"] == 160
    assert result["hard_firewall"] == {
        "test_indices_loaded": False, "test_data_loaded": False, "test_artifacts_referenced": False,
        "stage05_or_scientific_outcomes_loaded": False, "v2_output_writable": False,
        "fit_hooks_called": False, "optimizer_steps": 0,
    }
    assert result["fit_readiness"].startswith("NO_GO")
    assert len(result["reserved_candidate_ids"]) == 8
    assert (miniature_project / hpo.OUTPUT_RELATIVE / "dry_run/initial_plan.json").is_file()
    assert hpo.dry_run(miniature_project)["source_tree_sha256"] == result["source_tree_sha256"]
    assert not list((miniature_project / "outputs/runs").glob("**/model.pt"))
