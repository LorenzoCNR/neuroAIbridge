import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from final_thesis_uncertainty import (
    _parse_mapping_field,
    _stream_slot_probes,
    _build_seed_variability,
    _CellCheckpoint,
    _write_partial_seed_branch,
    _summary,
    bootstrap_row_indices,
    trial_bootstrap_draws,
)


def test_trial_bootstrap_resamples_complete_trials_without_class_stratification():
    trial = np.repeat(np.arange(8), 3)
    labels = np.repeat(np.repeat([0, 1], 4), 3)
    mask = np.ones(len(trial), dtype=bool)
    data = {"trial_id": trial}
    draws, seeds = trial_bootstrap_draws(data, labels, mask, n_replicates=100, seed_start=8100)
    assert seeds == list(range(8100, 8200))
    class_counts = []
    for draw in draws:
        assert len(draw) == 8
        assert set(draw).issubset(set(range(8)))
        class_counts.append(sum(labels[np.flatnonzero(trial == tid)[0]] == 1 for tid in draw))
    assert len(set(class_counts)) > 1


def test_bootstrap_row_indices_duplicate_whole_trial_blocks_only():
    trial = np.repeat(np.arange(3), 4)
    time = np.tile(np.arange(4), 3)
    data = {"trial_id": trial, "time_id": time}
    sampled = bootstrap_row_indices(data, np.ones(len(trial), dtype=bool), np.array([2, 0, 2]))
    assert len(sampled) == 12
    assert trial[sampled].tolist() == [2] * 4 + [0] * 4 + [2] * 4
    assert time[sampled].tolist() == list(range(4)) * 3


def test_probe_selection_mapping_accepts_frozen_python_repr_and_json():
    assert _parse_mapping_field("{'position': 1.0, 'velocity': 10.0}") == {
        "position": 1.0,
        "velocity": 10.0,
    }
    assert _parse_mapping_field('{"position": 1.0}') == {"position": 1.0}


def test_streamed_probe_qualification_loads_each_slot_once_for_raw_and_unit():
    item = {"trial_id": "slot-a"}
    loaded_ids = []
    fit_calls = []

    def load(slot):
        loaded_ids.append(slot["trial_id"])
        return {"payload": np.array([len(loaded_ids)])}

    def fit(_ctx, item, data, rep):
        fit_calls.append((item["trial_id"], rep, id(data)))
        return {"rep": rep}

    predictions = _stream_slot_probes({}, item, load_arrays=load, fit_predictions=fit)
    assert loaded_ids == ["slot-a"]
    assert len(fit_calls) == 2
    assert fit_calls[0][2] == fit_calls[1][2]
    assert set(predictions) == {"raw", "unit"}


def test_summary_records_undefined_bootstrap_instead_of_aborting_branch():
    result = _summary({"metric": "constant_geometry"}, 0.0, [np.nan, np.nan])
    assert result["status"] == "NO_FINITE_REPLICATES"
    assert result["finite_replicates"] == 0
    assert np.isnan(result["bootstrap_se"])


def test_seed_summary_keeps_near_collapse_value_and_marks_hpo_seed_role():
    rows = []
    for seed, value, collapse, status in [
        (1101, 0.7, "False", "ELIGIBLE"),
        (1201, 0.0, "True", "INELIGIBLE_NEAR_COLLAPSE"),
        (1301, 0.8, "False", "ELIGIBLE"),
    ]:
        rows.append({
            "dataset": "Real", "category": "accessibility",
            "metric": "direction_balanced_accuracy", "reference": "none",
            "population": "TOTAL65", "architecture": "cnn1d", "objective": "soft",
            "representation": "raw", "lag_bins": "", "seed": str(seed),
            "value": str(value), "near_collapse": collapse, "fit_status": status,
        })
    summary = _build_seed_variability({"metrics": rows})[0]
    assert summary["seed_1201"] == 0.0
    assert summary["seed_1201_near_collapse"] == "True"
    assert summary["seed_1101_is_independent_replication"] is False
    assert "HPO-selected" in summary["seed_1101_role"]
    assert summary["n_independent_final_seeds"] == 2


def test_cell_checkpoint_commits_atomically_and_resumes_without_overwrite(tmp_path):
    settings = {"replicates": 1000, "source": "abc", "parent": "xyz"}
    key = {"stage": "single_population", "trial_id": "trial-1", "representation": "raw"}
    first = _CellCheckpoint(tmp_path / "work", settings)
    assert not first.has(key)
    first.save(key, [{"metric": "score", "value": 0.5}])
    resumed = _CellCheckpoint(tmp_path / "work", settings)
    assert resumed.has(key)
    assert resumed.all_rows() == [{"metric": "score", "value": 0.5}]
    with pytest.raises(FileExistsError):
        resumed.save(key, [{"metric": "score", "value": 0.6}])
    with pytest.raises(ValueError):
        _CellCheckpoint(tmp_path / "work", {**settings, "replicates": 50})


def test_partial_seed_branch_is_labeled_pending_and_write_once(tmp_path):
    output = tmp_path / "uncertainty"
    rows = [{"dataset": "Synthetic", "seed_1101": 0.5,
             "seed_1101_role": "HPO-selected checkpoint; not independent"}]
    parent_hashes = {"core": "abc123"}
    status = _write_partial_seed_branch(output, rows, parent_hashes)
    assert status["status"] == "PARTIAL_BOOTSTRAP_PENDING"
    assert status["bootstrap_status"] == "PENDING"
    assert status["training_seed_variability_status"] == "COMPLETE"
    assert (output / "TRAINING_SEED_VARIABILITY.csv").is_file()
    assert (output / "UNCERTAINTY_STATUS.json").is_file()
    assert not (output / "UNCERTAINTY_PROVENANCE.json").exists()
    assert _write_partial_seed_branch(output, rows, parent_hashes) == status
    with pytest.raises(ValueError):
        _write_partial_seed_branch(output, [{"dataset": "Real", "seed_1101": 0.1}], parent_hashes)
