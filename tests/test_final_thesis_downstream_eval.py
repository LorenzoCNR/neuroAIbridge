import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import final_thesis_core_metrics as core
from final_thesis_downstream_eval import (
    _null_slot_key,
    _prepare_null_checkpoint_campaign,
    _run_checkpointed_null_slots,
    _trial_label_permutation,
    paired_vectorized_metrics,
)
from neurobridge.eval.representation import distance_geometry_correlation
from neurobridge.experiments.lag_shuffle import shuffled_trial_ids


def test_trial_label_shuffle_keeps_labels_constant_within_trial_and_split_counts():
    trial = np.repeat(np.arange(12), 5)
    split = np.repeat(np.repeat(["train", "validation", "test"], 4), 5)
    labels = np.repeat(np.tile([0, 0, 1, 1], 3), 5)
    data = {"trial_id": trial, "split": split}
    shuffled = _trial_label_permutation(data, labels, seed=44)
    for tid in np.unique(trial):
        assert len(np.unique(shuffled[trial == tid])) == 1
        assert len(np.unique(split[trial == tid])) == 1
    for name in ("train", "validation", "test"):
        before = [labels[trial == tid][0] for tid in np.unique(trial[split == name])]
        after = [shuffled[trial == tid][0] for tid in np.unique(trial[split == name])]
        assert sorted(before) == sorted(after)


def test_vectorized_pair_metrics_match_frozen_metric_definitions():
    rng = np.random.default_rng(25)
    # Float32 exercises the same serialized embedding dtype that exposed the
    # prior float32-vs-float64 RSA rank discrepancy.
    x = rng.normal(size=(700, 3)).astype(np.float32)
    y = (x @ rng.normal(size=(3, 3)).astype(np.float32) +
         rng.normal(scale=.15, size=(700, 3)).astype(np.float32)).astype(np.float32)
    actual = paired_vectorized_metrics(x, y)
    assert np.isclose(actual["procrustes_r2"], core.procrustes_r2(x, y), rtol=1e-12, atol=1e-12)
    assert np.isclose(actual["linear_cka"], core.linear_cka(x, y), rtol=1e-12, atol=1e-12)
    assert np.isclose(actual["rsa_spearman"], distance_geometry_correlation(x, y, method="spearman"),
                      rtol=1e-12, atol=1e-12)
    assert np.isclose(actual["rsa_pearson"], distance_geometry_correlation(x, y, method="pearson"),
                      rtol=1e-12, atol=1e-12)


def test_null_cell_checkpoints_resume_without_recomputing_completed_slots(tmp_path):
    def payload(trial_id):
        return {"item": {"dataset": "Synthetic", "architecture": "cnn1d",
                          "objective": "soft", "population": "A", "seed": 1101,
                          "trial_id": trial_id},
                "representation": "raw", "permutation_seeds": [730000, 730001],
                "observed": .5}

    payloads = [payload("trial-a"), payload("trial-b")]
    parent_hashes = {"core_provenance": "parent-hash", "evaluation_support": "support-hash",
                     "analysis_driver_source": "source-hash"}
    settings = {"permutations": 2, "label_seed_range": [730000, 730001],
                "training_seeds": [1101, 1201, 1301]}
    root = tmp_path / "null_controls"
    checkpoint_root, campaign_hash, manifest_path = _prepare_null_checkpoint_campaign(
        root, parent_hashes=parent_hashes, settings=settings,
        label_slot_keys=[_null_slot_key("label_shuffle", p) for p in payloads],
        pair_slot_keys=[])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["parent_hashes"] == parent_hashes
    assert manifest["settings"]["permutations"] == 2
    assert manifest["settings"]["label_seed_range"] == [730000, 730001]
    assert manifest["settings"]["training_seeds"] == [1101, 1201, 1301]
    assert len(manifest["expected_label_slots"]) == 2
    calls = []

    def interrupted_worker(task):
        trial_id = task["item"]["trial_id"]
        calls.append(trial_id)
        if trial_id == "trial-b":
            raise RuntimeError("simulated interruption after first committed slot")
        return {"item": task["item"], "representation": task["representation"],
                "metric": "condition_balanced_accuracy", "observed": .5,
                "null_values": [.25, .75], "null_mean": .5, "null_sd": .35,
                "empirical_p": .67, "percentile": 50., "effect": 0., "replicates": 2}

    with pytest.raises(RuntimeError, match="simulated interruption"):
        _run_checkpointed_null_slots(payloads, stage="label_shuffle", worker=interrupted_worker,
            checkpoint_root=checkpoint_root, campaign_hash=campaign_hash, permutations=2,
            workers=1, progress_label="fixture")
    assert calls == ["trial-a", "trial-b"]

    calls.clear()

    def resumed_worker(task):
        calls.append(task["item"]["trial_id"])
        return {"item": task["item"], "representation": task["representation"],
                "metric": "condition_balanced_accuracy", "observed": .5,
                "null_values": [.25, .75], "null_mean": .5, "null_sd": .35,
                "empirical_p": .67, "percentile": 50., "effect": 0., "replicates": 2}

    results, reused, files = _run_checkpointed_null_slots(payloads, stage="label_shuffle",
        worker=resumed_worker, checkpoint_root=checkpoint_root, campaign_hash=campaign_hash,
        permutations=2, workers=1, progress_label="fixture-resume")
    assert calls == ["trial-b"]
    assert reused == 1
    assert len(results) == 2
    assert len(files) == 2


def test_pair_null_checkpoints_validate_all_representations_metrics_and_reuse(tmp_path):
    def item(population):
        return {"dataset": "Synthetic", "architecture": "cnn1d", "objective": "soft",
                "population": population, "seed": 1101, "trial_id": f"trial-{population}"}

    task = {"item_a": item("A"), "item_b": item("B"), "permutations": 2,
            "seed_start": 810000}
    key = _null_slot_key("trial_pairing", task)
    root = tmp_path / "null_controls"
    checkpoint_root, campaign_hash, _ = _prepare_null_checkpoint_campaign(
        root, parent_hashes={"core": "parent", "support": "support", "source": "source"},
        settings={"permutations": 2, "pair_seed_range": [810000, 810001]},
        label_slot_keys=[], pair_slot_keys=[key])
    metric_names = ("procrustes_r2", "rsa_spearman", "rsa_pearson", "linear_cka")
    calls = []

    def worker(payload):
        calls.append(payload["seed_start"])
        return {"item_a": payload["item_a"], "item_b": payload["item_b"],
                "permutations": 2, "seed_start": payload["seed_start"], "n_matched_rows": 6,
                "null_values": {rep: {metric: [.1, .2] for metric in metric_names}
                                for rep in ("raw", "unit")}}

    results, reused, files = _run_checkpointed_null_slots([task], stage="trial_pairing",
        worker=worker, checkpoint_root=checkpoint_root, campaign_hash=campaign_hash,
        permutations=2, workers=1, progress_label="pair-fixture")
    assert calls == [810000]
    assert reused == 0 and len(results) == 1 and len(files) == 1

    calls.clear()
    resumed, reused, _ = _run_checkpointed_null_slots([task], stage="trial_pairing",
        worker=worker, checkpoint_root=checkpoint_root, campaign_hash=campaign_hash,
        permutations=2, workers=1, progress_label="pair-fixture-resume")
    assert calls == []
    assert reused == 1 and len(resumed) == 1


def test_null_checkpoint_rejects_changed_campaign_fingerprint(tmp_path):
    task = {"item": {"dataset": "Synthetic", "architecture": "cnn1d",
                     "objective": "soft", "population": "A", "seed": 1101,
                     "trial_id": "trial-a"}, "representation": "raw"}
    root = tmp_path / "null_controls"
    checkpoint_root, campaign_hash, _ = _prepare_null_checkpoint_campaign(
        root, parent_hashes={"core": "old"}, settings={"permutations": 2},
        label_slot_keys=[_null_slot_key("label_shuffle", task)], pair_slot_keys=[])
    result = {"item": task["item"], "representation": "raw", "replicates": 2,
              "null_values": [.2, .3]}
    from final_thesis_downstream_eval import _save_null_checkpoint
    _save_null_checkpoint(checkpoint_root, stage="label_shuffle",
        slot_key=_null_slot_key("label_shuffle", task), campaign_hash=campaign_hash,
        result=result)
    with pytest.raises(ValueError, match="provenance mismatch"):
        _run_checkpointed_null_slots([task], stage="label_shuffle", worker=lambda _: {},
            checkpoint_root=checkpoint_root, campaign_hash="different-campaign",
            permutations=2, workers=1, progress_label="fixture-invalid")


def test_trial_pair_null_derangement_never_pairs_same_trial():
    ids = np.arange(20)
    mapped = shuffled_trial_ids(ids, np.random.default_rng(17))
    assert set(mapped.tolist()) == set(ids.tolist())
    assert np.all(mapped != ids)
