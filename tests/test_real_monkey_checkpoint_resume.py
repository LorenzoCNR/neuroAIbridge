import json

import pytest
import torch
from neurobridge.experiments.presentation_rebuild import (
    _complete_run,
    _load_or_create_manifest,
)

from neurobridge.experiments.real_monkey import (
    RealMonkeyConfig,
    _public_config,
    _resolve_neural_checkpoint,
)


def test_validated_checkpoint_can_be_reused_without_legacy_refit(tmp_path):
    config = RealMonkeyConfig(run_label="checkpoint_resume_test")
    checkpoint = (
        tmp_path
        / config.output_root
        / config.run_label
        / "stage03_models"
        / "held_out"
        / "cnn1d_soft"
        / "model.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    (checkpoint.parent / "config.json").write_text(
        json.dumps({
            "config": _public_config(config),
            "schedule": {"max_steps": 4000},
            "branch": "held_out",
            "model": "cnn1d",
            "objective": "soft",
        }),
        encoding="utf-8",
    )
    torch.save({
        "model_name": "cnn1d",
        "objective": "soft",
        "branch": "held_out",
        "state_dict": {},
    }, checkpoint)

    resolved = _resolve_neural_checkpoint(
        tmp_path,
        config,
        model_name="cnn1d",
        loss_name="soft",
        branch="held_out",
        checkpoint_path=checkpoint,
    )
    assert resolved == checkpoint.resolve()

    cached = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    cached["config"]["batch_size"] = config.batch_size + 1
    (checkpoint.parent / "config.json").write_text(json.dumps(cached), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint configuration"):
        _resolve_neural_checkpoint(
            tmp_path,
            config,
            model_name="cnn1d",
            loss_name="soft",
            branch="held_out",
            checkpoint_path=checkpoint,
        )


def test_top_level_manifest_merges_phases_and_preserves_completed_runs(tmp_path):
    manifest_path, state = _load_or_create_manifest(tmp_path, "synthetic", None)
    _complete_run(manifest_path, state, "synthetic", "synthetic_run_42")
    partition = tmp_path / "channel_partition.json"
    partition.write_text('{"channel_split_seed": 42}', encoding="utf-8")

    resumed_path, resumed = _load_or_create_manifest(tmp_path, "real", partition)
    assert resumed_path == manifest_path
    assert resumed["completed_runs"] == ["synthetic_run_42"]
    assert resumed["protocol"]["channel_partition_sha256"] is not None

    other_partition = tmp_path / "other_channel_partition.json"
    other_partition.write_text('{"channel_split_seed": 123}', encoding="utf-8")
    with pytest.raises(FileExistsError, match="different channel partition"):
        _load_or_create_manifest(tmp_path, "real_10", other_partition)
