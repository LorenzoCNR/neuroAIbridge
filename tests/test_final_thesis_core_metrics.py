from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.final_thesis_core_metrics import (
    _new_embedding_staging_directory,
    _paired_preflight_archive_names,
    _preserved_complete_embedding_files,
    _verify_preflight_embedding_index,
    _tag_index_dataset,
    _probe_metrics,
    _real_export_parent_matches,
    _sample_pair_indices,
    _split_masks,
    _compatible_support_parents,
    _attach_frozen_fit_status,
    _neural_synthetic_items,
    _summarize_seeds,
    _test_mask,
    _verify_real_status_consistency,
    SUPPORT_NAME,
    _write_bytes_once,
    canonical_hash,
    write_manifest_once,
)


def test_test_support_keeps_frozen_synthetic_a_convention_and_masks_b():
    item = {"dataset": "Synthetic"}
    data = {
        "split": np.asarray(["train", "test", "test", "validation"]),
        "valid_mask": np.asarray([True, False, True, True]),
    }
    assert _test_mask(item, data, "A").tolist() == [False, True, True, False]
    assert _test_mask(item, data, "B").tolist() == [False, False, True, False]


def test_real_primary_test_support_requires_fully_valid_window_centers():
    item = {"dataset": "Real"}
    data = {
        "split": np.asarray(["test", "test", "train", "validation"]),
        "valid_mask": np.asarray([True, False, True, True]),
    }
    assert _test_mask(item, data, "TOTAL65").tolist() == [True, False, False, False]


def test_synthetic_probe_masks_use_frozen_population_validity_convention():
    data = {
        "split": np.asarray(["train", "train", "validation", "test"]),
        "valid_mask": np.asarray([False, True, False, True]),
    }
    a = _split_masks(data, "Synthetic", "A")
    b = _split_masks(data, "Synthetic", "B")
    assert a["train"].tolist() == [True, True, False, False]
    assert a["validation"].tolist() == [False, False, True, False]
    assert b["train"].tolist() == [False, True, False, False]
    assert b["validation"].tolist() == [False, False, False, False]


def test_probe_hyperparameter_selection_does_not_read_test_labels():
    rng = np.random.default_rng(481)
    labels = np.tile(np.arange(8), 12)
    embedding = rng.normal(size=(len(labels), 3)) + labels[:, None] * np.asarray([0.2, -0.1, 0.15])
    masks = {
        "train": np.arange(len(labels)) < 48,
        "validation": (np.arange(len(labels)) >= 48) & (np.arange(len(labels)) < 72),
        "test": np.arange(len(labels)) >= 72,
    }
    _, first = _probe_metrics(embedding, labels, masks, seed=7, progress=None)
    changed = labels.copy()
    changed[masks["test"]] = np.roll(changed[masks["test"]], 3)
    second_rows, second = _probe_metrics(embedding, changed, masks, seed=7, progress=None)
    assert first["selected_C"] == second["selected_C"]
    assert all(row.get("selection_split") == "validation" for row in second_rows)


def test_manifest_write_once_is_idempotent_and_refuses_replacement(tmp_path):
    path = tmp_path / "manifest.json"
    first = {"source_hash": canonical_hash([1, 2, 3]), "immutable": True}
    first_hash = write_manifest_once(path, first)
    assert write_manifest_once(path, first) == first_hash
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        write_manifest_once(path, {"source_hash": "different", "immutable": True})
    assert not list(tmp_path.glob(".tmp-*"))


def test_atomic_write_once_publishes_complete_immutable_file(tmp_path):
    path = tmp_path / "payload.csv"
    _write_bytes_once(path, b"a,b\n1,2\n")
    assert path.read_bytes() == b"a,b\n1,2\n"
    assert not list(tmp_path.glob(".tmp-*"))
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        _write_bytes_once(path, b"other\n")


def test_embedding_staging_directory_is_short_even_for_long_trial_ids(tmp_path):
    target = tmp_path / ("frozen-descriptive-trial-id-" + "x" * 100)
    staging = _new_embedding_staging_directory(target)
    assert staging.parent == target.parent
    assert staging.name.startswith(".staging-")
    assert len(str(staging / "embedding.npz")) < len(str(target / "embedding.npz"))
    assert len(staging.name) < 32


def test_embedding_index_dataset_is_explicit_and_consistent():
    tagged = _tag_index_dataset({"trial_id": "syn-1", "seed": 1101}, "Synthetic")
    assert tagged == {"trial_id": "syn-1", "seed": 1101, "dataset": "synthetic"}
    with pytest.raises(RuntimeError, match="dataset conflicts"):
        _tag_index_dataset({"trial_id": "x", "dataset": "real"}, "synthetic")


def test_preflight_embedding_index_accepts_only_frozen_inventory(tmp_path):
    path = tmp_path / "EMBEDDING_INDEX.csv"
    columns = ["dataset", "trial_id", "architecture", "objective", "population", "seed",
               "embedding_path", "manifest_path", "embedding_sha256", "manifest_sha256"]
    records = ([{"dataset": "synthetic", "trial_id": f"s{i}", **{c: "x" for c in columns[2:]}}
                for i in range(50)] +
               [{"dataset": "real", "trial_id": f"r{i}", **{c: "x" for c in columns[2:]}}
                for i in range(72)])
    pd.DataFrame(records, columns=columns).to_csv(path, index=False)
    _verify_preflight_embedding_index(path)
    pd.DataFrame(records[:-1], columns=columns).to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="122-row inventory"):
        _verify_preflight_embedding_index(path)


def test_real_parent_reuse_allows_only_an_archived_equivalent_support_hash():
    expected = {"trial_id": "frozen", "checkpoint_sha256": "checkpoint-new",
                "evaluation_support_sha256": "current-support"}
    prior = {**expected, "evaluation_support_sha256": "archived-compatible-support"}
    assert _real_export_parent_matches(prior, expected, {"archived-compatible-support"})
    assert not _real_export_parent_matches(prior, expected, {"unrelated-support"})
    prior["checkpoint_sha256"] = "different-checkpoint"
    assert not _real_export_parent_matches(prior, expected, {"archived-compatible-support"})


def test_synthetic_fit_status_comes_from_frozen_audit_slot():
    item = {"trial_id": "syn-1", "architecture": "cnn1d"}
    enriched = _attach_frozen_fit_status(
        item, {"syn-1": {"fit_status": "ELIGIBLE", "near_collapse": False}})
    assert enriched["fit_status"] == "ELIGIBLE"
    assert enriched["near_collapse"] is False
    with pytest.raises(RuntimeError, match="absent from frozen audit"):
        _attach_frozen_fit_status(item, {})


def test_deterministic_pca_is_excluded_from_synthetic_neural_loop():
    items = [{"dataset": "Synthetic", "architecture": "cnn1d", "seed": 1101},
             {"dataset": "Synthetic", "architecture": "pca", "seed": "deterministic"},
             {"dataset": "Real", "architecture": "cnn1d", "seed": 1101}]
    assert _neural_synthetic_items(items) == [items[0]]


def test_preflight_archive_discovery_requires_intact_manifest_pairs(tmp_path):
    audit = tmp_path / "AUDIT_MANIFEST_PRE_FIX.json"
    support = tmp_path / "EVALUATION_SUPPORT_MANIFEST_PRE_FIX.json"
    audit.write_text('{"version":1}\n', encoding="utf-8")
    payload = {"audit_manifest_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
               "coordinates": [[1, 2]]}
    payload["support_sha256"] = canonical_hash(payload)
    support.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    assert _paired_preflight_archive_names(tmp_path) == (
        "AUDIT_MANIFEST_PRE_FIX.json", "EVALUATION_SUPPORT_MANIFEST_PRE_FIX.json")
    support.write_text('{"corrupt":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="archived support/audit parent mismatch"):
        _paired_preflight_archive_names(tmp_path)


def test_preflight_preserves_only_hash_valid_embedding_payloads(tmp_path):
    output = tmp_path
    support_payload = {"coordinates": [[1, 10]], "support_sha256": "placeholder"}
    support_payload["support_sha256"] = canonical_hash(
        {key: value for key, value in support_payload.items() if key != "support_sha256"})
    (output / "EVALUATION_SUPPORT_MANIFEST.json").write_text(
        json.dumps(support_payload), encoding="utf-8")
    folder = output / "embeddings" / "real-slot"
    folder.mkdir(parents=True)
    artifact_bytes = {name: (name + "-payload").encode() for name in (
        "embedding_raw.npz", "embedding_unit.npz", "evaluation_metadata.npz")}
    for name, content in artifact_bytes.items():
        (folder / name).write_bytes(content)
    manifest = {"parents": {"evaluation_support_sha256": "support-file-hash"},
                "artifact_sha256": {name: hashlib.sha256(content).hexdigest()
                                    for name, content in artifact_bytes.items()}}
    # The exporter records the support file digest as the parent.
    support_digest = hashlib.sha256((output / "EVALUATION_SUPPORT_MANIFEST.json").read_bytes()).hexdigest()
    manifest["parents"]["evaluation_support_sha256"] = support_digest
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    preserved = _preserved_complete_embedding_files(output, ())
    assert len(preserved) == 4
    (folder / "embedding_raw.npz").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="payload hash mismatch"):
        _preserved_complete_embedding_files(output, ())


def test_support_parent_reuse_requires_identical_definition_and_valid_archive(tmp_path):
    output = tmp_path
    audit_old = output / "AUDIT_MANIFEST_PRE_SHORT_TEMP_FIX.json"
    support_old = output / "EVALUATION_SUPPORT_MANIFEST_PRE_SHORT_TEMP_FIX.json"
    audit_old.write_text('{"source_artifact_hashes":{"frozen":"same"}}\n', encoding="utf-8")
    archived = {
        "support_version": "test-v1", "coordinates": [[1, 100], [1, 101]],
        "audit_manifest_sha256": hashlib.sha256(audit_old.read_bytes()).hexdigest(),
    }
    archived["support_sha256"] = canonical_hash(archived)
    support_old.write_text(json.dumps(archived, sort_keys=True), encoding="utf-8")
    current = {
        "support_version": "test-v1", "coordinates": [[1, 100], [1, 101]],
        "audit_manifest_sha256": "new-audit-hash",
    }
    current["support_sha256"] = canonical_hash(current)
    (output / SUPPORT_NAME).write_text(json.dumps(current, sort_keys=True), encoding="utf-8")
    support_hashes, file_hashes = _compatible_support_parents(output, current)
    assert archived["support_sha256"] in support_hashes
    assert hashlib.sha256(support_old.read_bytes()).hexdigest() in file_hashes

    current_changed = {**current, "coordinates": [[1, 100], [1, 102]]}
    current_changed["support_sha256"] = canonical_hash({k: v for k, v in current_changed.items()
                                                          if k != "support_sha256"})
    (output / SUPPORT_NAME).write_text(json.dumps(current_changed, sort_keys=True), encoding="utf-8")
    support_hashes_changed, _ = _compatible_support_parents(output, current_changed)
    assert archived["support_sha256"] not in support_hashes_changed


def test_cross_seed_summary_keeps_lag_bins_separate():
    rows = []
    for lag in (-1, 0):
        for seed in (1101, 1201, 1301):
            rows.append({
                "dataset": "Synthetic", "architecture": "cnn1d", "objective": "soft",
                "population": "A_vs_B", "representation": "raw", "category": "temporal_fidelity",
                "reference": "B(t+lag)_vs_A(t)", "metric": "s_lag_r2", "lag_bins": lag,
                "seed": seed, "value": float(lag + seed / 10000), "near_collapse": False,
            })
    summary = _summarize_seeds(rows)
    assert len(summary) == 2
    assert set(summary["lag_bins"]) == {-1, 0}
    assert summary["n_valid_seeds"].tolist() == [3, 3]


def test_pair_diagnostic_sample_is_shared_and_never_self_paired():
    left_a, right_a = _sample_pair_indices(1000)
    left_b, right_b = _sample_pair_indices(1000)
    np.testing.assert_array_equal(left_a, left_b)
    np.testing.assert_array_equal(right_a, right_b)
    assert len(left_a) == 100_000
    assert np.all(left_a != right_a)


def test_real_status_and_geometry_collapse_flags_are_reconciled():
    statuses = ["ELIGIBLE"] * 69 + ["INELIGIBLE_NEAR_COLLAPSE"] * 3
    flags = [False] * 69 + [True] * 3
    _verify_real_status_consistency(statuses, flags)
    flags[-1] = False
    with pytest.raises(RuntimeError, match="geometry flag"):
        _verify_real_status_consistency(statuses, flags)


def test_real_progress_probe_is_reported_and_test_progress_cannot_select_alpha():
    rng = np.random.default_rng(71)
    embedding = rng.normal(size=(180, 3))
    progress = np.linspace(0, 1, len(embedding))
    labels = np.tile(np.arange(8), 180 // 8 + 1)[:180]
    masks = {
        "train": np.arange(180) < 90,
        "validation": (np.arange(180) >= 90) & (np.arange(180) < 135),
        "test": np.arange(180) >= 135,
    }
    rows_a, selection_a = _probe_metrics(embedding, labels, masks, seed=9, progress=progress)
    changed = progress.copy()
    changed[masks["test"]] = np.roll(changed[masks["test"]], 7)
    rows_b, selection_b = _probe_metrics(embedding, labels, masks, seed=9, progress=changed)
    progress_rows = {row["metric"]: row for row in rows_a if row["metric"].startswith("progress_")}
    assert set(progress_rows) == {"progress_r2", "progress_mae"}
    assert selection_a["selected_progress_alpha"] == selection_b["selected_progress_alpha"]
    assert all(progress_rows[name]["selection_split"] == "validation" for name in progress_rows)
