import numpy as np

from neurobridge.experiments.final_real_lag_revision import (
    SHIFTS_MS,
    candidate_lags,
    extract_test_windows,
    shift_safe_windows,
)
from neurobridge.experiments.final_real_lag_fit import _load_trainval_bundle
from neurobridge.experiments.phase2a_hpo import file_sha256
from tools.final_real_controlled_lag_v2 import _qualification_records_equivalent
import json


def _window_fixture(trials=(3,), window=201):
    trial_id = np.repeat(np.asarray(trials, dtype=np.int64), 600)
    time_id = np.tile(np.arange(600, dtype=np.int64), len(trials))
    x = np.broadcast_to(
        (trial_id * 1000 + time_id).astype(np.float32)[:, None, None],
        (len(trial_id), window, 33),
    ).copy()
    valid = np.tile(np.asarray([100 <= t < 500 for t in range(600)]), len(trials))
    split = np.full(len(trial_id), "train", dtype="U10")
    return {
        "X_windows": x,
        "time_id": time_id,
        "global_time_id": trial_id * 600 + time_id,
        "trial_id": trial_id,
        "labels": (trial_id % 8).astype(np.int64),
        "progress": (time_id / 599).astype(np.float32),
        "position": np.zeros((len(trial_id), 2), dtype=np.float32),
        "velocity": np.zeros((len(trial_id), 2), dtype=np.float32),
        "split": split,
        "lag_valid": valid,
    }


def test_shifted_safe_windows_are_no_wrap_and_keep_behavior_time_fixed():
    base = _window_fixture()
    shifted = shift_safe_windows(base, 200)
    src = 100
    dst = 300
    np.testing.assert_array_equal(shifted["X_windows"][dst], base["X_windows"][src])
    assert shifted["time_id"][dst] == 300
    assert shifted["progress"][dst] == base["progress"][dst]
    assert shifted["labels"][dst] == base["labels"][dst]
    assert not shifted["lag_valid"][299]
    assert shifted["lag_valid"][dst]
    assert shifted["lag_valid"].sum() == 200
    assert not shifted["X_windows"][100].any()


def test_test_extractor_materializes_test_only_rows_and_exact_shift():
    spikes = np.arange(193 * 600 * 33, dtype=np.float32).reshape(193 * 600, 33)
    position = np.zeros((193 * 600, 2), dtype=np.float32)
    velocity = np.zeros((193 * 600, 2), dtype=np.float32)
    targets = np.arange(193, dtype=np.int64) % 8
    values = {
        "spikes": spikes,
        "position": position,
        "velocity": velocity,
        "target_by_trial": targets,
    }
    shifted = extract_test_windows(values, [7, 21], 160)
    assert set(np.unique(shifted["split"])) == {"test"}
    assert set(np.unique(shifted["trial_id"])) == {7, 21}
    assert shifted["X_windows"].shape == (1200, 201, 33)
    assert shifted["lag_valid"].sum() == 2 * 240
    assert np.all(shifted["global_time_id"] == shifted["trial_id"] * 600 + shifted["time_id"])


def test_intervention_values_and_local_lag_grids_are_frozen():
    assert SHIFTS_MS == (100, 160, 200)
    assert candidate_lags(0) == tuple(range(-20, 21))
    assert candidate_lags(100) == tuple(range(80, 121))
    assert candidate_lags(160) == tuple(range(140, 181))
    assert candidate_lags(200) == tuple(range(180, 221))


def test_qualification_resume_ignores_timing_but_keeps_scientific_invariants():
    base = {
        "source_sha256_verified": "source",
        "test_rows_absent_from_fit_bundle": True,
        "frozen_encoder_input_equivalence": {
            "100": {
                "valid_embedding_rows_compared": 11700,
                "raw_max_abs_error": 1e-7,
                "unit_max_abs_error": 2e-6,
                "inference_seconds": 0.4,
                "inference_valid_windows_per_second": 29000.0,
                "device": "cuda",
            }
        },
    }
    repeated = {
        **base,
        "frozen_encoder_input_equivalence": {
            "100": {
                **base["frozen_encoder_input_equivalence"]["100"],
                "inference_seconds": 0.7,
                "inference_valid_windows_per_second": 16500.0,
            }
        },
    }
    assert _qualification_records_equivalent(base, repeated)
    changed_source = {**repeated, "source_sha256_verified": "other"}
    assert not _qualification_records_equivalent(base, changed_source)
    bad_error = {
        **repeated,
        "frozen_encoder_input_equivalence": {
            "100": {**repeated["frozen_encoder_input_equivalence"]["100"], "unit_max_abs_error": 3e-5}
        },
    }
    assert not _qualification_records_equivalent(base, bad_error)


def test_fit_adapter_reads_trusted_trainval_windows_filename(tmp_path):
    count = 2
    values = {
        "X_windows": np.zeros((count, 201, 33), dtype=np.float32),
        "time_id": np.asarray([200, 200], dtype=np.int64),
        "global_time_id": np.asarray([800, 1400], dtype=np.int64),
        "trial_id": np.asarray([1, 2], dtype=np.int64),
        "labels": np.asarray([0, 1], dtype=np.int64),
        "progress": np.asarray([0.2, 0.2], dtype=np.float32),
        "position": np.zeros((count, 2), dtype=np.float32),
        "velocity": np.zeros((count, 2), dtype=np.float32),
        "split": np.asarray(["train", "validation"], dtype="U10"),
        "lag_valid": np.ones(count, dtype=bool),
    }
    windows_path = tmp_path / "trainval_windows.npz"
    split_path = tmp_path / "split.json"
    manifest_path = tmp_path / "manifest.json"
    np.savez_compressed(windows_path, **values)
    split_path.write_text(json.dumps({"train": [1], "validation": [2]}), encoding="utf-8")
    manifest_path.write_text(json.dumps({
        "shift_ms": 100,
        "population": "B_DISTAL",
        "test_trial_intersection_empty": True,
        "fit_accessible": True,
        "shifted_windows_sha256": file_sha256(windows_path),
        "safe_split_sha256": file_sha256(split_path),
        "train_trial_ids": [1],
        "validation_trial_ids": [2],
    }), encoding="utf-8")
    bundle = _load_trainval_bundle(tmp_path, 100)
    assert bundle[5].name == "trainval_windows.npz"
