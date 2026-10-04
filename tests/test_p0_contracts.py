"""P0 scientific contracts: semantic correspondence rather than shape alone."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from neurobridge.experiments.audit_contracts import interior_mask, validate_metadata, validate_split
from neurobridge.experiments.staged_shared_latent import (
    SharedLatentStageConfig, _balanced_trial_split, _window_subject,
    _load_window_dataset, _stage_dir, _append_decoder_metrics, _public_config,
)
from neurobridge.train.loop import encode_windows


def test_partition_coverage_and_invalid_partitions():
    config = SharedLatentStageConfig()
    split = _balanced_trial_split(config)
    trial = np.repeat(np.arange(config.n_trials), config.trial_length)
    masks = validate_split(split, trial, config.split_counts)
    assert np.all(sum(m.astype(int) for m in masks.values()) == 1)
    for mutation in ("duplicate", "missing", "overlap"):
        bad = {k: list(v) for k, v in split.items()}
        if mutation == "duplicate":
            bad["train"].append(bad["train"][0])
        elif mutation == "missing":
            bad["test"].pop()
        else:
            bad["test"][0] = bad["train"][0]
        with pytest.raises(ValueError):
            validate_split(bad, trial)


def test_semantic_ids_window_reload_shuffle_and_mask():
    config = SimpleNamespace(n_trials=3, trial_length=8, window_size=3, stride=1)
    ids = np.arange(24).reshape(3, 8)
    X = np.stack((100 + ids, 200 + ids), axis=-1)
    M = np.stack((ids, ids + 1, ids + 2), axis=-1)
    data = {"labels": np.array([7, 8, 9]), "M": M, "valid_A": ids % 3 != 0}
    values = _window_subject(X, data["labels"], M[..., 2], data["valid_A"], config)
    validate_metadata(values, data, config, "A")
    for row, (trial, time) in enumerate(zip(values["trial_id"], values["time_id"])):
        for offset in range(3):
            source_time = time + offset - 1
            expected = X[trial, source_time] if 0 <= source_time < 8 else np.zeros(2)
            np.testing.assert_array_equal(values["X_windows"][row, offset], expected)
            for neuron in range(2):
                assert values["X_windows"].reshape(24, -1)[row, offset * 2 + neuron] == expected[neuron]
    class Center(torch.nn.Module):
        def forward(self, x):
            assert not self.training
            assert not torch.is_grad_enabled()
            return x[:, 1]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "windows.npz"
        np.savez(path, **values)
        dataset = _load_window_dataset(path)
        encoded, metadata = encode_windows(Center(), DataLoader(
            dataset, batch_size=5, shuffle=True, generator=torch.Generator().manual_seed(9)))
    trial, time = metadata["trial_id"].numpy(), metadata["time_id"].numpy().astype(int)
    np.testing.assert_array_equal(encoded.numpy(), X[trial, time])
    np.testing.assert_array_equal(metadata["label"], data["labels"][trial])
    np.testing.assert_array_equal(metadata["progress"], M[trial, time, 2])
    mask = metadata["lag_valid"].numpy().astype(bool)
    np.testing.assert_array_equal(encoded.numpy()[mask], X[trial[mask], time[mask]])
    bad = dict(values, labels=np.roll(values["labels"], 1))
    with pytest.raises(ValueError, match="Labels"):
        validate_metadata(bad, data, config, "A")


def test_strict_padding_support():
    config = SharedLatentStageConfig()
    time = np.arange(200)
    np.testing.assert_array_equal(time[interior_mask(time, config, "A")], np.arange(10, 190))
    np.testing.assert_array_equal(time[interior_mask(time, config, "B")], np.arange(20, 190))
    assert not interior_mask(np.array([10, 19, 190]), config, "B").any()


def test_reference_stage_entry_rejected_before_writes(tmp_path):
    config = SharedLatentStageConfig(run_label="output_2026-08-27_comparison_2000")
    with pytest.raises(ValueError, match="immutable"):
        _stage_dir(tmp_path, config, "stage01_data")
    assert list(tmp_path.iterdir()) == []


def test_public_config_aliases_keep_synthetic_cache_resumable(tmp_path):
    config = SharedLatentStageConfig(run_label="cache_alias_test")
    run_root = tmp_path / config.output_root / config.run_label
    stage01 = run_root / "stage01_data"
    stage01.mkdir(parents=True)
    path = stage01 / "config.json"
    saved = _public_config(config)
    path.write_text(json.dumps(saved), encoding="utf-8")

    assert _stage_dir(tmp_path, config, "stage02_windows") == run_root / "stage02_windows"

    saved["time_offset_bins"] += 1
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="cebra_time_offset"):
        _stage_dir(tmp_path, config, "stage03_models")


def test_probe_selection_independent_of_test_targets():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(120, 3))
    labels = np.argmax(X, axis=1)
    progress = X[:, 0] + 0.2 * rng.normal(size=120)
    arguments = {"base": {}, "noise_scales": (), "random_state": 0}
    for name, section in (("train", slice(0, 60)), ("validation", slice(60, 90)), ("test", slice(90, 120))):
        arguments[name + "_embedding"] = X[section]
        arguments[name + "_labels"] = labels[section]
        arguments[name + "_progress"] = progress[section]
    rows = []
    original = _append_decoder_metrics(rows, **arguments)
    arguments["test_labels"] = (arguments["test_labels"] + 1) % 3
    arguments["test_progress"] = arguments["test_progress"] + 100
    changed_rows = []
    changed = _append_decoder_metrics(changed_rows, **arguments)
    assert original == changed
    assert rows != changed_rows
