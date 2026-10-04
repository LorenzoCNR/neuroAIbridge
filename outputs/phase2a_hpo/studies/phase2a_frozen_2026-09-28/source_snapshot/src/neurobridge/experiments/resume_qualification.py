"""Infrastructure-only end-to-end cache/resume qualification for the monkey stages.

This writes a separate short-run qualification under ``outputs/qualification``;
its one-update checkpoints are never included in scientific result tables.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from neurobridge.experiments import real_monkey as legacy
from neurobridge.experiments.presentation_rebuild import _channel_partition
from neurobridge.experiments.real_monkey_validated import (
    ValidationSchedule,
    _write_run_manifest,
    canonical_config,
    fit_validated_neural_model,
    run_natural_monkey_suite,
)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_resume_qualification(project_root: str | Path) -> Path:
    project_root = Path(project_root).resolve()
    run_id = "resume_qualification_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    config = canonical_config(
        run_label=run_id,
        max_steps=1,
        setting_name="real_0",
        population_name="A",
        channel_indices=tuple(_channel_partition(project_root)[0]["A_channel_indices"]),
        channel_split_seed=42,
        imposed_shift_bins=0,
        output_root="outputs/qualification",
    )
    # This is an infrastructure smoke run, not a change to the frozen V2
    # scientific schedule; its isolated run ID prevents result mixing.
    schedule = ValidationSchedule(
        max_steps=1,
        validation_interval=1,
        min_steps=0,
        patience=1,
        relative_min_delta=0.001,
    )
    report_path = legacy._run_root(project_root, config) / "resume_invalidation_report.json"
    events: list[dict[str, object]] = []
    try:
        windows_path = legacy.stage_windows(project_root, config)
        data_path = legacy._run_root(project_root, config) / "stage01_data" / "data.npz"
        split_path = legacy._run_root(project_root, config) / "stage01_data" / "split.json"
        data_hash_before = _sha256(data_path)
        windows_hash_before = _sha256(windows_path)
        events.append({"event": "prepared_raw_split_windows", "data_sha256": data_hash_before, "windows_sha256": windows_hash_before})

        checkpoint = fit_validated_neural_model(
            project_root, config, model_name="cnn1d", objective_name="soft",
            schedule=schedule,
        )
        checkpoint_hash = _sha256(checkpoint)
        events.append({"event": "short_qualification_fit_complete", "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash})

        # Simulate process death immediately after the fit stage.
        events.append({"event": "simulated_interruption", "after_stage": "fit"})
        with patch.object(legacy, "_make_real_model", side_effect=AssertionError("cache hit unexpectedly rebuilt the fit")):
            resumed_checkpoint = fit_validated_neural_model(
                project_root, config, model_name="cnn1d", objective_name="soft",
                schedule=schedule,
            )
        if _sha256(resumed_checkpoint) != checkpoint_hash:
            raise AssertionError("restart changed the completed checkpoint")

        embedding_path = legacy.transform_neural_embeddings(
            project_root, config, model_name="cnn1d", loss_name="soft",
            branch="held_out", checkpoint_path=resumed_checkpoint,
        )
        embedding_hash = _sha256(embedding_path)
        embedding_meta_path = embedding_path.with_name("cnn1d_soft_metadata.json")
        embedding_meta = json.loads(embedding_meta_path.read_text(encoding="utf-8"))
        if embedding_meta.get("checkpoint_sha256") != checkpoint_hash:
            raise AssertionError("embedding provenance does not identify the fitted checkpoint")
        events.append({"event": "embedding_created_after_fit_resume", "embedding": str(embedding_path), "embedding_sha256": embedding_hash})

        # Simulate process death after the embedding stage, then forbid both
        # model construction and encoding to prove the embedding is a cache hit.
        events.append({"event": "simulated_interruption", "after_stage": "embedding"})
        with (
            patch.object(legacy, "_make_real_model", side_effect=AssertionError("embedding cache miss rebuilt a model")),
            patch.object(legacy, "encode_windows", side_effect=AssertionError("embedding cache miss re-encoded windows")),
        ):
            restarted_checkpoint = fit_validated_neural_model(
                project_root, config, model_name="cnn1d", objective_name="soft",
                schedule=schedule,
            )
            restarted_embedding = legacy.transform_neural_embeddings(
                project_root, config, model_name="cnn1d", loss_name="soft",
                branch="held_out", checkpoint_path=restarted_checkpoint,
            )
        if _sha256(restarted_checkpoint) != checkpoint_hash or _sha256(restarted_embedding) != embedding_hash:
            raise AssertionError("restart failed to reuse checkpoint and embedding")
        events.append({"event": "restart_reused_checkpoint_and_embedding", "passed": True})

        # A fit-only parameter change must retain raw/window artifacts and
        # refuse the old fit cache rather than silently reuse or overwrite it.
        changed_config = replace(config, learning_rate=config.learning_rate * 1.5)
        changed_windows = legacy.stage_windows(project_root, changed_config)
        if _sha256(data_path) != data_hash_before or _sha256(changed_windows) != windows_hash_before:
            raise AssertionError("learning-rate change invalidated an upstream data/window cache")
        try:
            fit_validated_neural_model(
                project_root, changed_config, model_name="cnn1d", objective_name="soft",
                schedule=schedule,
            )
        except FileExistsError:
            fit_invalidated = True
        else:
            fit_invalidated = False
        if not fit_invalidated:
            raise AssertionError("learning-rate change did not invalidate the dependent fit cache")
        events.append({
            "event": "training_parameter_invalidation",
            "parameter": "learning_rate",
            "upstream_data_reused": True,
            "upstream_windows_reused": True,
            "dependent_fit_rejected_as_stale": True,
        })

        # A plot-only DPI change must not touch the fitted checkpoint or the
        # raw/unit embedding. It explicitly rewrites only this qualification's
        # figure bundle.
        figure_path = legacy.plot_neural_embeddings(
            project_root, config, model_name="cnn1d", loss_name="soft",
            branch="held_out", checkpoint_path=checkpoint, dpi=120, force=True,
        )
        figure_manifest = figure_path.parent / "cnn1d_soft_figures.provenance.json"
        figure_provenance = json.loads(figure_manifest.read_text(encoding="utf-8"))
        if figure_provenance["plotting_config"].get("dpi") != 120:
            raise AssertionError("plot-only setting is not recorded in figure provenance")
        if _sha256(checkpoint) != checkpoint_hash or _sha256(embedding_path) != embedding_hash:
            raise AssertionError("plot-only change invalidated a checkpoint or embedding")
        # Restore canonical rendering before asking the complete suite to
        # resume; no scientific cache is changed by this plot-only operation.
        legacy.plot_neural_embeddings(
            project_root, config, model_name="cnn1d", loss_name="soft",
            branch="held_out", checkpoint_path=checkpoint, dpi=180, force=True,
        )
        events.append({
            "event": "plot_only_invalidation",
            "parameter": "dpi",
            "checkpoint_unchanged": True,
            "embedding_unchanged": True,
            "figure_provenance": str(figure_manifest),
        })

        # Qualify branch isolation: the full-sample fit is independent and
        # uses the selected held-out update count (one in this smoke run).
        best_step = int(json.loads(checkpoint.parent.joinpath("compute.json").read_text(encoding="utf-8"))["best_validation_step"])
        full_checkpoint = fit_validated_neural_model(
            project_root, config, model_name="cnn1d", objective_name="soft",
            schedule=schedule, branch="full_sample", full_sample_updates=best_step,
        )
        full_embedding = legacy.transform_neural_embeddings(
            project_root, config, model_name="cnn1d", loss_name="soft",
            branch="full_sample", checkpoint_path=full_checkpoint,
        )
        if full_checkpoint.parent == checkpoint.parent or full_embedding.parent == embedding_path.parent:
            raise AssertionError("held-out and full-sample artifacts share a cache directory")

        # Seed the manifest with a completed held-out unit, then resume the
        # suite. Cache-hit guards prove neither held-out nor full-sample is fit
        # a second time; the suite must preserve the prior unit and add the
        # full-sample unit without counting branches together.
        completed_before = ["cnn1d/soft"]
        _write_run_manifest(
            project_root, config, schedule, status="interrupted",
            completed=completed_before, parent_run_label="qualification_only",
        )
        with patch.object(legacy, "_make_real_model", side_effect=AssertionError("suite restart unexpectedly retrained")):
            suite_result = run_natural_monkey_suite(
                project_root, config, schedule=schedule,
                models=("cnn1d",), objectives=("soft",),
                rsa_bootstrap_replicates=2, decoding_bootstrap_replicates=2,
            )
        run_manifest_path = Path(suite_result["run_root"]) / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        expected_completed = {"cnn1d/soft", f"full_sample/cnn1d/soft@{best_step}"}
        if not expected_completed.issubset(set(run_manifest.get("completed_fits", []))):
            raise AssertionError("run manifest lost a branch during restart")
        if len(run_manifest.get("completed_fits", [])) != 2:
            raise AssertionError("run manifest duplicated or collapsed fit branches")
        events.append({
            "event": "manifest_restart_and_branch_isolation",
            "status": run_manifest["status"],
            "completed_fits": run_manifest["completed_fits"],
            "held_out_checkpoint": str(checkpoint),
            "full_sample_checkpoint": str(full_checkpoint),
            "passed": True,
        })

        report = {
            "status": "PASS",
            "run_id": config.run_label,
            "classification": "infrastructure qualification only; not part of V2 scientific results",
            "setting": config.setting_name,
            "population": config.population_name,
            "architecture": "cnn1d",
            "objective": "soft",
            "training_seed": config.training_seed,
            "channel_split_seed": config.channel_split_seed,
            "channel_indices": list(config.channel_indices),
            "qualification_schedule": {
                "max_steps": schedule.max_steps,
                "validation_interval": schedule.validation_interval,
                "min_steps": schedule.min_steps,
                "patience": schedule.patience,
                "relative_min_delta": schedule.relative_min_delta,
            },
            "split_trial_ids": json.loads(split_path.read_text(encoding="utf-8")),
            "config": legacy._public_config(config),
            "events": events,
            "artifact_hashes": {
                "data": data_hash_before,
                "windows": windows_hash_before,
                "held_out_checkpoint": checkpoint_hash,
                "held_out_embedding": embedding_hash,
                "full_sample_checkpoint": _sha256(full_checkpoint),
                "full_sample_embedding": _sha256(full_embedding),
            },
            "parent_artifacts": {
                "raw_source_sha256": _sha256(project_root / config.source_relative),
                "split_sha256": _sha256(split_path),
            },
            "run_manifest": str(run_manifest_path),
            "elapsed_seconds": time.time(),
        }
        _write_json(report_path, report)
        return report_path
    except Exception as error:
        _write_json(report_path, {
            "status": "FAIL",
            "run_id": config.run_label,
            "classification": "infrastructure qualification only",
            "error": repr(error),
            "events": events,
        })
        raise


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    report_path = run_resume_qualification(root)
    print(f"Resume/invalidation qualification passed: {report_path}", flush=True)


if __name__ == "__main__":
    main()
