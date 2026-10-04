"""Resumable seed-42 entry point for the clean presentation rebuild.

Each domain remains in its existing staged runner. Select ``synthetic``,
``real_0`` or ``real_10`` to resume only that domain/setting; each lower stage
continues to validate and reuse its own cached artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from neurobridge.experiments import real_monkey as legacy_real
from neurobridge.experiments.real_monkey_lag_eval import run_controlled_lag_evaluation
from neurobridge.experiments.real_monkey_validated import (
    DEFAULT_SCHEDULE,
    MODEL_NAMES,
    OBJECTIVE_NAMES,
    canonical_config,
    evaluate_cached_real_metrics,
    make_random_channel_split,
    run_natural_monkey_suite,
)
from neurobridge.experiments.staged_shared_latent import (
    SharedLatentStageConfig,
    stage_generate,
    run_staged_benchmark,
)


SEED = 42
RUN_STAMP = "clean_rebuild_2026-09-23_seed42"
TRAINING_SEEDS = (42, 123, 456)
DATA_SEED = 42
CHANNEL_SPLIT_SEED = 42
CORRECTED_RUN_STAMP = "v2_corrected_2026-09-23"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _channel_partition(project_root: Path) -> tuple[dict[str, Any], Path]:
    path = project_root / "outputs" / "runs" / "channel_split_seed_42.json"
    expected = make_random_channel_split(n_channels=65, seed=SEED)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != expected:
            raise FileExistsError(
                f"saved canonical channel split differs from seed 42; preserve it and use a new output root: {path}"
            )
    else:
        _write_json(path, expected)
    return expected, path


def _load_or_create_manifest(
    project_root: Path,
    phase: str,
    partition_path: Path | None,
) -> tuple[Path, dict[str, Any]]:
    path = project_root / "outputs" / "runs" / f"{RUN_STAMP}_progress.json"
    partition_hash = _sha256(partition_path) if partition_path is not None else None
    expected_protocol = {
        "seed": SEED,
        "synthetic_run_label": f"{RUN_STAMP}_synthetic",
        "real_schedule": {
            "max_steps": DEFAULT_SCHEDULE.max_steps,
            "validation_interval": DEFAULT_SCHEDULE.validation_interval,
            "min_steps": DEFAULT_SCHEDULE.min_steps,
            "patience": DEFAULT_SCHEDULE.patience,
            "relative_min_delta": DEFAULT_SCHEDULE.relative_min_delta,
        },
        "channel_partition_sha256": partition_hash,
    }
    state: dict[str, Any] = {"protocol": expected_protocol, "phase_status": {}, "completed_runs": []}
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        saved_protocol = state.get("protocol", {})
        invariant_keys = set(expected_protocol) - {"channel_partition_sha256"}
        if any(saved_protocol.get(key) != expected_protocol.get(key) for key in invariant_keys):
            raise FileExistsError(
                f"rebuild progress manifest has a different protocol; use a new run stamp: {path}"
            )
        saved_partition_hash = saved_protocol.get("channel_partition_sha256")
        if (
            saved_partition_hash is not None
            and partition_hash is not None
            and saved_partition_hash != partition_hash
        ):
            raise FileExistsError(
                f"rebuild progress manifest has a different channel partition; use a new run stamp: {path}"
            )
        if saved_partition_hash is None and partition_hash is not None:
            saved_protocol["channel_partition_sha256"] = partition_hash
        state["protocol"] = saved_protocol
        state.setdefault("phase_status", {})
        state.setdefault("completed_runs", [])
    state["phase_status"][phase] = "running"
    _write_json(path, state)
    return path, state


def _complete_run(state_path: Path, state: dict[str, Any], phase: str, run_id: str) -> None:
    if run_id not in state["completed_runs"]:
        state["completed_runs"].append(run_id)
    state["phase_status"][phase] = "running"
    _write_json(state_path, state)


def run_seed42_rebuild(
    project_root: str | Path,
    *,
    phase: str = "all",
    force: bool = False,
) -> dict[str, Any]:
    """Run/re-enter the selected seed-42 domain without discarding stage caches."""
    project_root = Path(project_root).resolve()
    valid_phases = {"all", "synthetic", "real", "real_0", "real_10"}
    if phase not in valid_phases:
        raise ValueError(f"phase must be one of {sorted(valid_phases)}")

    wants_synthetic = phase in {"all", "synthetic"}
    wants_real = phase in {"all", "real", "real_0", "real_10"}
    partition: dict[str, Any] | None = None
    partition_path: Path | None = None
    if wants_real:
        partition, partition_path = _channel_partition(project_root)

    manifest_path, state = _load_or_create_manifest(project_root, phase, partition_path)
    results: dict[str, Any] = {"progress_manifest": manifest_path}

    if wants_synthetic:
        run_id = f"{RUN_STAMP}_synthetic"
        print(f"[{phase}] synthetic stages: data -> windows -> fit -> embeddings -> metrics -> figures", flush=True)
        synthetic_config = SharedLatentStageConfig(
            run_label=run_id,
            seed=SEED,
            split_seed=SEED,
            training_seed=SEED,
            output_root="outputs/runs",
            batch_size=1024,
            max_iterations=4000,
            validation_interval=400,
            min_iterations=800,
            relative_min_delta=0.001,
            early_stopping_patience=3,
        )
        results["synthetic"] = run_staged_benchmark(
            project_root, synthetic_config, force=force,
        )
        _complete_run(manifest_path, state, phase, run_id)

    if wants_real:
        assert partition is not None
        selected_settings = (
            ("real_0", "real_10") if phase in {"all", "real"}
            else (phase,)
        )
        real_results: dict[str, Any] = {}
        for setting_name in selected_settings:
            shift = 0 if setting_name == "real_0" else 10
            setting_results: dict[str, Any] = {}
            for population_name, channel_key in (("A", "A_channel_indices"), ("B", "B_channel_indices")):
                run_id = f"{RUN_STAMP}_{setting_name}_{population_name}"
                config = canonical_config(
                    run_label=run_id,
                    max_steps=DEFAULT_SCHEDULE.max_steps,
                    setting_name=setting_name,
                    population_name=population_name,
                    channel_indices=tuple(partition[channel_key]),
                    channel_split_seed=SEED,
                    imposed_shift_bins=shift,
                    output_root="outputs/runs",
                )
                print(
                    f"[{phase}] {setting_name}/{population_name}: raw -> split -> windows -> PCA/neural fits -> embeddings -> metrics/figures",
                    flush=True,
                )
                setting_results[population_name] = run_natural_monkey_suite(
                    project_root,
                    config,
                    schedule=DEFAULT_SCHEDULE,
                )
                _complete_run(manifest_path, state, phase, run_id)
            real_results[setting_name] = setting_results
        results["real"] = real_results

    state["phase_status"][phase] = "complete"
    _write_json(manifest_path, state)
    return results


def _content_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _corrected_manifest(project_root: Path, partition_path: Path) -> tuple[Path, dict[str, Any]]:
    path = project_root / "outputs" / "runs" / f"{CORRECTED_RUN_STAMP}_progress.json"
    protocol = {
        "training_seeds": list(TRAINING_SEEDS),
        "synthetic_data_seed": DATA_SEED,
        "synthetic_split_seed": DATA_SEED,
        "channel_split_seed": CHANNEL_SPLIT_SEED,
        "channel_partition_sha256": _sha256(partition_path),
        "real_schedule": {
            "max_steps": DEFAULT_SCHEDULE.max_steps,
            "validation_interval": DEFAULT_SCHEDULE.validation_interval,
            "min_steps": DEFAULT_SCHEDULE.min_steps,
            "patience": DEFAULT_SCHEDULE.patience,
            "relative_min_delta": DEFAULT_SCHEDULE.relative_min_delta,
        },
        "controlled_lag": {"shift_bins": 10, "shift_population": "B_only", "reuse_r0_checkpoints": True},
    }
    state: dict[str, Any] = {"protocol": protocol, "phase_status": {}, "runs": {}}
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("protocol") != protocol:
            raise FileExistsError(f"corrected V2 manifest has a different frozen protocol; use a new run stamp: {path}")
        state.setdefault("phase_status", {})
        state.setdefault("runs", {})
    return path, state


def _record_corrected_run(
    manifest_path: Path,
    state: dict[str, Any],
    phase: str,
    run_id: str,
    run_root: Path,
    *,
    reused: bool = False,
) -> None:
    state.setdefault("runs", {})[run_id] = {
        "status": "complete",
        "phase": phase,
        "run_root": str(run_root.relative_to(manifest_path.parents[2])),
        "reused_existing_artifacts": reused,
    }
    state.setdefault("phase_status", {})[phase] = "running"
    _write_json(manifest_path, state)


def _synthetic_config(training_seed: int) -> SharedLatentStageConfig:
    run_label = (
        f"{RUN_STAMP}_synthetic" if training_seed == 42
        else f"{CORRECTED_RUN_STAMP}_seed{training_seed}_synthetic"
    )
    return SharedLatentStageConfig(
        run_label=run_label,
        seed=DATA_SEED,
        split_seed=DATA_SEED,
        training_seed=training_seed,
        output_root="outputs/runs",
        batch_size=1024,
        max_iterations=4000,
        validation_interval=400,
        min_iterations=800,
        relative_min_delta=0.001,
        early_stopping_patience=3,
    )


def _real_config(
    partition: dict[str, Any],
    *,
    training_seed: int,
    population: str,
) -> Any:
    if population == "total_65":
        channel_indices = tuple(range(65))
        run_id = f"{CORRECTED_RUN_STAMP}_seed{training_seed}_real_total_65"
    elif population in {"A", "B"}:
        channel_indices = tuple(partition[f"{population}_channel_indices"])
        run_id = (
            f"{RUN_STAMP}_real_0_{population}" if training_seed == 42
            else f"{CORRECTED_RUN_STAMP}_seed{training_seed}_real_0_{population}"
        )
    else:
        raise ValueError(f"unsupported real population: {population}")
    return canonical_config(
        run_label=run_id,
        max_steps=DEFAULT_SCHEDULE.max_steps,
        training_seed=training_seed,
        setting_name="real_0",
        population_name=population,
        channel_indices=channel_indices,
        channel_split_seed=CHANNEL_SPLIT_SEED,
        imposed_shift_bins=0,
        output_root="outputs/runs",
    )


def _validate_complete_real_run(project_root: Path, config: Any) -> None:
    """Read-only qualification gate before reusing an already completed run."""
    root = legacy_real._run_root(project_root, config)
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"completed REAL-0 run manifest missing: {manifest_path}")
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    if saved.get("status") != "complete" or saved.get("configuration") != legacy_real._public_config(config):
        raise FileExistsError(f"existing REAL-0 artifacts do not match the requested config: {root}")
    if saved.get("validation_schedule") != {
        "max_steps": DEFAULT_SCHEDULE.max_steps,
        "validation_interval": DEFAULT_SCHEDULE.validation_interval,
        "min_steps": DEFAULT_SCHEDULE.min_steps,
        "patience": DEFAULT_SCHEDULE.patience,
        "relative_min_delta": DEFAULT_SCHEDULE.relative_min_delta,
    }:
        raise FileExistsError(f"existing REAL-0 schedule differs from the frozen schedule: {root}")
    required = [
        root / "stage04_embeddings" / branch / f"{model}_{objective}.npz"
        for branch in ("held_out", "full_sample")
        for model in MODEL_NAMES
        for objective in OBJECTIVE_NAMES
    ]
    required.extend(
        root / "stage03_models" / branch / f"{model}_{objective}" / "model.pt"
        for branch in ("held_out", "full_sample")
        for model in MODEL_NAMES
        for objective in OBJECTIVE_NAMES
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"completed REAL-0 run is missing {len(missing)} required artifacts; first: {missing[0]}")


def _ensure_synthetic_training_seed(
    project_root: Path,
    training_seed: int,
    manifest_path: Path,
    state: dict[str, Any],
) -> Path:
    config = _synthetic_config(training_seed)
    run_root = project_root / config.output_root / config.run_label
    run_id = config.run_label
    if training_seed == 42:
        reference = run_root / "stage01_data" / "shared_data.npz"
        if not reference.is_file():
            raise FileNotFoundError(f"seed-42 synthetic reference data are missing: {reference}")
        _record_corrected_run(manifest_path, state, "synthetic", run_id, run_root, reused=True)
        return run_root

    print(
        f"[corrected V2] synthetic training seed {training_seed}: reusing the seed-42 latent/data/split; only training randomness varies",
        flush=True,
    )
    generated = stage_generate(project_root, config)
    reference_path = project_root / "outputs" / "runs" / f"{RUN_STAMP}_synthetic" / "stage01_data" / "shared_data.npz"
    if not reference_path.is_file():
        raise FileNotFoundError(f"seed-42 synthetic reference data are missing: {reference_path}")
    with np.load(reference_path, allow_pickle=False) as reference, np.load(generated, allow_pickle=False) as candidate:
        for key in ("Z_shared", "labels", "X_A", "X_B", "M", "eta"):
            if not np.array_equal(reference[key], candidate[key]):
                raise RuntimeError(f"synthetic training-seed {training_seed} changed fixed ground-truth/data array {key}")
        latent_hash = _content_sha256(reference["Z_shared"])
        data_hash = _content_sha256(reference["X_A"]) + _content_sha256(reference["X_B"])
    if run_id not in state.get("runs", {}):
        run_staged_benchmark(project_root, config)
    state.setdefault("runs", {}).setdefault(run_id, {}).update({
        "status": "complete",
        "phase": "synthetic",
        "run_root": str(run_root.relative_to(manifest_path.parents[2])),
        "fixed_latent_sha256": latent_hash,
        "fixed_spike_data_sha256": hashlib.sha256(data_hash.encode("ascii")).hexdigest(),
        "varied_parameter": "training_seed",
    })
    state.setdefault("phase_status", {})["synthetic"] = "running"
    _write_json(manifest_path, state)
    return run_root


def _ensure_real_population(
    project_root: Path,
    partition: dict[str, Any],
    *,
    training_seed: int,
    population: str,
    manifest_path: Path,
    state: dict[str, Any],
) -> Any:
    config = _real_config(partition, training_seed=training_seed, population=population)
    root = legacy_real._run_root(project_root, config)
    manifest_path_on_disk = root / "run_manifest.json"
    if population in {"A", "B"} and training_seed == 42:
        _validate_complete_real_run(project_root, config)
        evaluate_cached_real_metrics(project_root, config, branches=("full_sample",))
        _record_corrected_run(manifest_path, state, "real_ab_r0", config.run_label, root, reused=True)
        return config
    if manifest_path_on_disk.is_file():
        saved = json.loads(manifest_path_on_disk.read_text(encoding="utf-8"))
        if saved.get("status") == "complete":
            _validate_complete_real_run(project_root, config)
            evaluate_cached_real_metrics(project_root, config, branches=("full_sample",))
            _record_corrected_run(manifest_path, state, f"real_{population}", config.run_label, root, reused=True)
            return config
    print(
        f"[corrected V2] fitting REAL-0/{population}, seed={training_seed}: raw -> split -> windows -> models -> embeddings -> metrics/figures",
        flush=True,
    )
    run_natural_monkey_suite(project_root, config, schedule=DEFAULT_SCHEDULE)
    _record_corrected_run(manifest_path, state, f"real_{population}", config.run_label, root)
    return config


def run_corrected_v2_rebuild(
    project_root: str | Path,
    *,
    phase: str = "all",
) -> dict[str, Any]:
    """Resume corrected full-65/A-B/controlled-lag work in separate run roots."""
    project_root = Path(project_root).resolve()
    valid_phases = {"all", "synthetic", "real_total_65", "real_ab_r0", "controlled_lag"}
    if phase not in valid_phases:
        raise ValueError(f"phase must be one of {sorted(valid_phases)}")
    partition, partition_path = _channel_partition(project_root)
    manifest_path, state = _corrected_manifest(project_root, partition_path)
    state.setdefault("phase_status", {})[phase] = "running"
    _write_json(manifest_path, state)
    results: dict[str, Any] = {"progress_manifest": manifest_path}

    def run_total(seed: int) -> Any:
        return _ensure_real_population(
            project_root, partition, training_seed=seed, population="total_65",
            manifest_path=manifest_path, state=state,
        )

    def run_ab(seed: int) -> tuple[Any, Any]:
        return tuple(
            _ensure_real_population(
                project_root, partition, training_seed=seed, population=population,
                manifest_path=manifest_path, state=state,
            )
            for population in ("A", "B")
        )

    def run_lag(seed: int) -> Any:
        config_a, config_b = run_ab(seed)
        run_label = f"{CORRECTED_RUN_STAMP}_seed{seed}_real_10_B_reuse_R0"
        print(
            f"[corrected V2] controlled lag seed {seed}: shift B by +10 bins; reuse R0 checkpoints; no fitting",
            flush=True,
        )
        output = run_controlled_lag_evaluation(
            project_root, seed=seed, config_a=config_a, config_b=config_b,
            run_label=run_label,
        )
        _record_corrected_run(manifest_path, state, "controlled_lag", run_label, output["run_root"])
        return output

    if phase == "all":
        run_total(42)
        run_ab(42)
        results["controlled_lag_seed42"] = run_lag(42)
        for seed in (123, 456):
            _ensure_synthetic_training_seed(project_root, seed, manifest_path, state)
        for seed in (123, 456):
            run_total(seed)
            run_ab(seed)
            results[f"controlled_lag_seed{seed}"] = run_lag(seed)
    elif phase == "synthetic":
        for seed in TRAINING_SEEDS:
            _ensure_synthetic_training_seed(project_root, seed, manifest_path, state)
    elif phase == "real_total_65":
        for seed in TRAINING_SEEDS:
            run_total(seed)
    elif phase == "real_ab_r0":
        for seed in TRAINING_SEEDS:
            run_ab(seed)
    elif phase == "controlled_lag":
        for seed in TRAINING_SEEDS:
            results[f"controlled_lag_seed{seed}"] = run_lag(seed)

    state.setdefault("phase_status", {})[phase] = "complete"
    _write_json(manifest_path, state)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "synthetic", "real_total_65", "real_ab_r0", "controlled_lag"),
        default="all",
        help="resume only the selected corrected V2 phase",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    args = parser.parse_args()
    run_corrected_v2_rebuild(args.project_root, phase=args.phase)


if __name__ == "__main__":
    main()
