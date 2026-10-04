"""Phase-2A fit adapter: the only data handle is an immutable safe bundle.

This module deliberately does not import the trusted extractor, open a V2
window cache, or deserialize a V2 split.  It reuses frozen model/objective
primitives while owning its train/validation-only control flow and outputs.
"""

from __future__ import annotations

import csv
import json
import math
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from neurobridge.experiments import real_monkey as real
from neurobridge.experiments import real_monkey_validated as rv
from neurobridge.experiments import staged_shared_latent as synth
from neurobridge.experiments.phase2a_hpo import (
    ProtocolViolation, _canonical_json, file_sha256, guarded_output_root,
    validation_geometry,
)
from neurobridge.losses.infonce import cebra_infonce_loss, soft_contrastive_loss, supervised_infonce_loss
from neurobridge.sampling.cebra_behavior import CEBRASupervisedWindowDataset
from neurobridge.sampling.cebra_time import CEBRATripletWindowDataset
from neurobridge.train.loop import train_steps, train_triplet_steps


_OBJECTIVE_INTERNAL = {
    "soft": "soft", "infonce": "infonce",
    "time_contrastive_blocks": "cebra_time",
    "behavior_contrastive_blocks": "cebra_behavior",
}


def load_safe_bundle(project_root: Path, domain: str, population: str):
    """Reject any path or split outside the dedicated HPO safe-input tree."""
    root = guarded_output_root(project_root) / "safe_inputs" / domain / population
    if domain not in {"synthetic", "real"} or population not in (
        ("A", "B") if domain == "synthetic" else ("TOTAL65", "A", "B")
    ):
        raise ProtocolViolation("unknown safe bundle identity")
    manifest_path, windows_path, split_path = (
        root / "manifest.json", root / "windows.npz", root / "split.json"
    )
    if not all(path.is_file() and root.resolve() in path.resolve().parents
               for path in (manifest_path, windows_path, split_path)):
        raise ProtocolViolation("missing/escaped HPO safe bundle")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("domain"), manifest.get("population")) != (domain, population):
        raise ProtocolViolation("safe bundle identity mismatch")
    if (file_sha256(windows_path) != manifest.get("safe_windows_sha256")
            or file_sha256(split_path) != manifest.get("safe_split_sha256")):
        raise ProtocolViolation("safe bundle content hash mismatch")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if set(split) != {"train", "validation"} or split["train"] != manifest["train_trial_ids"] or split["validation"] != manifest["validation_trial_ids"]:
        raise ProtocolViolation("safe split may contain only frozen train/validation IDs")
    with np.load(windows_path, allow_pickle=False) as arrays:
        if "split" not in arrays.files:
            raise ProtocolViolation("safe windows lack per-row split")
        trial_ids, split_rows = arrays["trial_id"], arrays["split"]
        if len(trial_ids) != len(split_rows) or set(np.unique(split_rows)) != {"train", "validation"}:
            raise ProtocolViolation("safe windows contain unknown split rows")
        if set(np.unique(trial_ids)) != set(split["train"] + split["validation"]):
            raise ProtocolViolation("safe windows include outside trial IDs")
        if not np.array_equal(split_rows, np.where(np.isin(trial_ids, split["train"]), "train", "validation")):
            raise ProtocolViolation("safe row partition differs from safe split")
    if domain == "real":
        dataset, values = real._load_real_window_dataset(windows_path)
    else:
        dataset = synth._load_window_dataset(windows_path)
        with np.load(windows_path, allow_pickle=False) as arrays:
            values = {key: arrays[key] for key in arrays.files}
    return root, manifest, split, dataset, values


def _frozen_config(trial: Mapping[str, Any], max_updates: int | None = None):
    row = trial["config"]
    domain = row["domain"]
    active = row["active_temperature"]
    schedule = row["schedule"]
    cap = max_updates if max_updates is not None else schedule["max_updates"]
    if domain == "synthetic":
        return replace(
            synth.SharedLatentStageConfig(), seed=42, split_seed=42,
            training_seed=row["training_seed_root"],
            learning_rate=row["learning_rate"], weight_decay=row["weight_decay"],
            embedding_temperature=active if row["objective"] in {"soft", "infonce"} else 0.1,
            cebra_temperature=active if row["objective"] not in {"soft", "infonce"} else 1.0,
            max_iterations=cap,
            validation_interval=(1 if max_updates is not None else schedule["validation_interval"]),
            min_iterations=(1 if max_updates is not None else schedule["min_updates"]),
            early_stopping_patience=schedule["patience"],
            relative_min_delta=schedule["relative_min_delta"],
        )
    return replace(
        real.RealMonkeyConfig(), source_relative="__HPO_PARENT_ACCESS_FORBIDDEN__",
        output_root="outputs/phase2a_hpo", run_label=trial["trial_id"],
        setting_name="real_0", population_name=row["population"],
        channel_indices=tuple(row["channel_indices"]), channel_split_seed=42,
        imposed_shift_bins=0, split_seed=42, batch_size=1024,
        training_seed=row["training_seed_effective"],
        learning_rate=row["learning_rate"], weight_decay=row["weight_decay"],
        embedding_temperature=active if row["objective"] in {"soft", "infonce"} else 0.1,
        cebra_temperature=active if row["objective"] not in {"soft", "infonce"} else 1.0,
        max_iterations=cap,
    )


def _loaders(domain: str, population: str, objective: str, dataset: Any,
             values: Mapping[str, np.ndarray], split: Mapping[str, list[int]],
             config: Any, device_seed: int):
    valid = np.asarray(values["lag_valid"], dtype=bool)
    trial_id = np.asarray(values["trial_id"])
    # Frozen Synthetic V2 excludes lag-invalid rows only for B; Real R0
    # excludes invalid windows for every population.
    eligible = valid if domain == "real" or (domain == "synthetic" and population == "B") else np.ones(len(valid), dtype=bool)
    train = np.flatnonzero(np.isin(trial_id, split["train"]) & eligible).tolist()
    validation = np.flatnonzero(np.isin(trial_id, split["validation"]) & eligible).tolist()
    generator = torch.Generator().manual_seed(device_seed)
    internal = _OBJECTIVE_INTERNAL[objective]
    triplet = None
    if internal in {"cebra_time", "cebra_behavior"}:
        klass = CEBRATripletWindowDataset if internal == "cebra_time" else CEBRASupervisedWindowDataset
        kwargs = {"offset": config.cebra_time_offset} if internal == "cebra_time" else {}
        if domain == "synthetic" and population == "B":
            kwargs["valid_mask"] = valid
        if domain == "real" and internal == "cebra_time":
            kwargs["valid_mask"] = torch.as_tensor(valid)
        triplet = klass(dataset, train, **kwargs)
        validation_triplet = klass(dataset, validation, **kwargs)
        train_loader = DataLoader(triplet, batch_size=1024, shuffle=True, drop_last=True, generator=generator)
        if domain == "real":
            order = np.random.default_rng(device_seed + 10000).permutation(len(validation_triplet))
            validation_loader = DataLoader(Subset(validation_triplet, order.tolist()), batch_size=1024, shuffle=False, drop_last=True)
        else:
            validation_loader = DataLoader(validation_triplet, batch_size=1024, shuffle=False, drop_last=True)
    else:
        train_loader = DataLoader(Subset(dataset, train), batch_size=1024, shuffle=True, drop_last=True, generator=generator)
        if domain == "real":
            validation_loader, _ = rv._make_validation_loader(dataset, validation, batch_size=1024, seed=device_seed + 10000)
        else:
            validation_loader = DataLoader(Subset(dataset, validation), batch_size=1024, shuffle=False, drop_last=True)
    if not len(train_loader) or not len(validation_loader):
        raise ProtocolViolation("safe fit lacks a full B=1024 train or validation batch")
    return train_loader, validation_loader, triplet, generator, train, validation


def _save_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(_canonical_json(payload) + "\n")


def early_stop_transition(best_loss: float, best_update: int | None,
                          reference: float | None, stale: int, loss: float,
                          update: int, schedule: Mapping[str, Any]) -> tuple[float, int | None, float | None, int, bool]:
    """Frozen strict-best and separate meaningful-improvement patience rule."""
    selected = loss < best_loss
    if selected:
        best_loss, best_update = loss, update
    if update >= schedule["min_updates"]:
        if reference is None:
            reference, stale = loss, 0
        elif (reference - loss) / max(abs(reference), 1e-12) >= schedule["relative_min_delta"]:
            reference, stale = loss, 0
        else:
            stale += 1
    return best_loss, best_update, reference, stale, selected


def fit_trial(project_root: Path, trial: Mapping[str, Any], *, smoke_updates: int | None = None) -> dict:
    """Run one isolated trial; existing outcomes are reused, never overwritten."""
    project = project_root.resolve()
    row = trial["config"]
    domain, population, objective = row["domain"], row["population"], row["objective"]
    output_root = guarded_output_root(project)
    trial_root = ((output_root / "smoke" / f"{trial['trial_id']}-u{smoke_updates}-{file_sha256(Path(__file__))[:12]}")
                  if smoke_updates is not None else Path(trial["output_path"]))
    if output_root not in trial_root.resolve().parents:
        raise ProtocolViolation("fit output escaped HPO root")
    safe_root, safe_manifest, split, dataset, values = load_safe_bundle(project, domain, population)
    safe_parent_hash = safe_manifest["safe_windows_sha256"]
    record = {
        "trial_id": trial["trial_id"], "config_sha256": trial["config_sha256"],
        "config": row, "safe_windows_sha256": safe_parent_hash,
        "safe_split_sha256": safe_manifest["safe_split_sha256"],
        "raw_source_sha256": safe_manifest["raw_source_sha256"],
        "original_split_sha256": safe_manifest["original_split_sha256"],
        "parent_windows_sha256": safe_manifest["parent_windows_sha256"],
        "safe_manifest_sha256": file_sha256(safe_root / "manifest.json"),
        "adapter_source_sha256": file_sha256(Path(__file__)),
        "smoke_updates": smoke_updates,
    }
    result_path = trial_root / "result.json"
    record_path = trial_root / "trial_record.json"
    if trial_root.exists():
        if not record_path.is_file() or json.loads(record_path.read_text(encoding="utf-8")) != record:
            raise ProtocolViolation("existing trial has missing/different immutable provenance")
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            for filename, digest in result.get("artifact_sha256", {}).items():
                if file_sha256(trial_root / filename) != digest:
                    raise ProtocolViolation("cached trial artifact hash changed")
            return result
        raise ProtocolViolation("interrupted fit preserved; no implicit retry or overwrite")
    trial_root.mkdir(parents=True, exist_ok=False)
    _save_json_new(record_path, record)
    try:
        result = _fit_new(trial_root, trial, dataset, values, split, smoke_updates)
    except Exception as exc:
        status = "FAILED_NUMERICAL" if isinstance(exc, FloatingPointError) else "FAILED_ARTIFACT" if isinstance(exc, ProtocolViolation) else "FAILED_RUNTIME"
        result = {"trial_id": trial["trial_id"], "status": status,
                  "error_class": type(exc).__name__, "error": str(exc),
                  "traceback": traceback.format_exc(limit=12), "optimizer_updates": None,
                  "artifact_sha256": {}}
    _save_json_new(result_path, result)
    return result


def _fit_new(trial_root: Path, trial: Mapping[str, Any], dataset: Any,
             values: Mapping[str, np.ndarray], split: Mapping[str, list[int]],
             smoke_updates: int | None) -> dict:
    row = trial["config"]
    domain, population, objective = row["domain"], row["population"], row["objective"]
    internal = _OBJECTIVE_INTERNAL[objective]
    effective_seed = row["training_seed_effective"]
    if domain == "synthetic":
        synth._seed_everything(effective_seed)
    else:
        real._seed_real_training(effective_seed)
    config = _frozen_config(trial, smoke_updates)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (synth._make_model(row["architecture"], values["X_windows"].shape[-1], config, normalize=True)
             if domain == "synthetic" else
             real._make_real_model(row["architecture"], values["X_windows"].shape[-1], config, normalize=True)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=row["learning_rate"], weight_decay=row["weight_decay"])
    train_loader, validation_loader, triplet, generator, train_indices, validation_indices = _loaders(
        domain, population, objective, dataset, values, split, config, effective_seed
    )
    schedule = row["schedule"]
    cap = smoke_updates if smoke_updates is not None else schedule["max_updates"]
    interval = 1 if smoke_updates is not None else schedule["validation_interval"]
    min_updates = 1 if smoke_updates is not None else schedule["min_updates"]
    patience = schedule["patience"]
    history: list[dict[str, Any]] = []
    best_state = None
    best_loss = float("inf")
    best_update = None
    reference = None
    stale = 0
    updates = 0
    stop_reason = "max_updates_reached"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    if domain == "real":
        iterator = iter(train_loader)
        interval_values: list[tuple[float, int]] = []
    while updates < cap:
        if domain == "synthetic":
            block_steps = min(interval - updates % interval, cap - updates)
            if triplet is not None:
                train_loss = train_triplet_steps(
                    model, train_loader, optimizer,
                    lambda reference_z, positive_z, negative_z: cebra_infonce_loss(
                        reference_z, positive_z, negative_z, temperature=config.cebra_temperature),
                    steps=block_steps, device=device,
                    resample=lambda: triplet.resample(generator=generator),
                )
            elif internal == "soft":
                train_loss = train_steps(
                    model, train_loader, optimizer,
                    lambda embedding, similarity: soft_contrastive_loss(
                        embedding, similarity, temperature=config.embedding_temperature),
                    steps=block_steps, device=device,
                    similarity_builder=lambda batch: synth._soft_metadata_similarity(batch, config),
                )
            else:
                train_loss = train_steps(
                    model, train_loader, optimizer,
                    lambda embedding, labels: supervised_infonce_loss(
                        embedding, labels, temperature=config.embedding_temperature),
                    steps=block_steps, device=device,
                )
            updates += block_steps
            validation_loss = (
                synth._evaluate_triplet_loss(model, validation_loader, config.cebra_temperature, device)
                if triplet is not None else
                synth._evaluate_training_loss(model, validation_loader, internal, config, device)
            )
        else:
            if triplet is not None:
                triplet.resample(generator=generator)
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                if triplet is not None:
                    triplet.resample(generator=generator)
                raw_batch = next(iterator)
            batch = rv._move_batch(raw_batch, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, batch_size, _ = rv._batch_loss(model, batch, internal, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite train loss at update {updates + 1}")
            loss.backward()
            optimizer.step()
            updates += 1
            interval_values.append((float(loss.detach().cpu()), batch_size))
            if updates % interval and updates < cap:
                continue
            denominator = sum(n for _, n in interval_values)
            train_loss = sum(value * n for value, n in interval_values) / denominator
            interval_values.clear()
            validation_loss = rv._evaluate_fixed_validation(model, validation_loader, internal, config, device)
        if not math.isfinite(validation_loss):
            raise FloatingPointError(f"nonfinite validation loss at update {updates}")
        best_loss, best_update, reference, stale, selected = early_stop_transition(
            best_loss, best_update, reference, stale, validation_loss, updates,
            {**schedule, "min_updates": min_updates},
        )
        if selected:
            best_state = rv._state_cpu(model)
        history.append({"optimizer_step": updates, "train_loss": float(train_loss),
                        "validation_loss": float(validation_loss), "stale_checks": stale,
                        "meaningful_reference": reference})
        if updates >= min_updates and stale >= patience and updates < cap:
            stop_reason = "patience_exhausted"
            break
    if best_state is None:
        raise ProtocolViolation("no best-validation checkpoint")
    stop_state = rv._state_cpu(model)
    elapsed = time.perf_counter() - started
    checkpoint = {"state_dict": best_state, "model_name": row["architecture"],
                  "objective": objective, "selected_validation_step": best_update,
                  "trial_id": trial["trial_id"]}
    torch.save(checkpoint, trial_root / "best_validation_checkpoint.pt")
    torch.save({"state_dict": stop_state, "optimizer_steps": updates,
                "stop_reason": stop_reason, "trial_id": trial["trial_id"]},
               trial_root / "stopping_checkpoint.pt")
    with (trial_root / "training_history.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    model.load_state_dict(best_state)
    valid = np.asarray(values["lag_valid"], dtype=bool)
    diagnostic_indices = [i for i in validation_indices if valid[i]]
    embedding_parts = []
    model.eval()
    inference_start = time.perf_counter()
    with torch.no_grad():
        for batch in DataLoader(Subset(dataset, diagnostic_indices), batch_size=1024, shuffle=False):
            embedding_parts.append(model(batch["x"].to(device)).cpu().numpy())
    inference_seconds = time.perf_counter() - inference_start
    embedding = np.concatenate(embedding_parts, axis=0)
    trial_ids = np.asarray(values["trial_id"])[diagnostic_indices]
    time_ids = np.asarray(values["time_id"])[diagnostic_indices]
    geometry = validation_geometry(embedding, trial_ids, time_ids,
                                   np.ones(len(embedding), dtype=bool),
                                   np.repeat("validation", len(embedding)))
    with (trial_root / "validation_embedding.npz").open("xb") as handle:
        np.savez_compressed(handle, embedding=embedding, trial_id=trial_ids,
                            time_id=time_ids, valid_mask=np.ones(len(embedding), dtype=bool),
                            split=np.repeat("validation", len(embedding)))
    status = "INELIGIBLE_NEAR_COLLAPSE" if geometry["near_collapse"] else "ELIGIBLE"
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    artifacts = {name: file_sha256(trial_root / name) for name in (
        "best_validation_checkpoint.pt", "stopping_checkpoint.pt",
        "training_history.csv", "validation_embedding.npz")}
    return {
        "trial_id": trial["trial_id"], "status": status, "validation_loss": float(best_loss),
        "selected_update": best_update, "stopping_update": updates,
        "validation_checks": len(history), "stop_reason": stop_reason,
        "geometry": geometry, "training_seconds": elapsed,
        "seconds_per_100_updates": elapsed / updates * 100,
        "updates_per_second": updates / max(elapsed, 1e-12),
        "train_windows_per_second": updates * 1024 / max(elapsed, 1e-12),
        "inference_seconds": inference_seconds,
        "inference_windows_per_second": len(embedding) / max(inference_seconds, 1e-12),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_gpu_memory_allocated": peak_allocated,
        "peak_gpu_memory_reserved": peak_reserved,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "pytorch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "artifact_sha256": artifacts,
    }
