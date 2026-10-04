"""Checkpoint-only, streaming Real embedding export for the frozen thesis slots.

No fitting, validation-loss calculation, or scientific evaluation occurs here.
The real-data path is deliberately gated on a separate, frozen evaluation-support
manifest. ``--smoke-mock`` exercises one existing checkpoint without opening any
observation or split container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurobridge.experiments import real_monkey as real  # noqa: E402

FINAL_ROOT = Path("outputs/final_thesis_v1")
SPEC = FINAL_ROOT / "freeze/REAL_FINAL_EXPERIMENT_SPEC.json"
PARTITION = Path("outputs/phase2a_hpo/partitions/real_somatotopic_v1.json")
STAGE1 = Path("outputs/runs/v2_corrected_2026-09-23_seed42_real_total_65/stage01_data")
SNAPSHOT = Path("outputs/phase2a_hpo/studies/phase2a_real_somatotopic_v1_2026-09-28/source_snapshot_manifest.json")
SOURCE_FILES = (
    "src/neurobridge/experiments/real_monkey.py",
    "src/neurobridge/models/temporal_cnn.py",
)
POPULATIONS = ("TOTAL65", "A_PROXIMAL", "B_DISTAL")
WINDOW = 201
TRIAL_LENGTH = 600


class ExportError(RuntimeError):
    """An immutable parent, frozen contract, or output invariant failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExportError(message)


def frozen_inputs(project: Path) -> tuple[dict, dict, dict]:
    spec_path = project / SPEC
    part_path = project / PARTITION
    spec = read_json(spec_path)
    part = read_json(part_path)
    snapshot = read_json(project / SNAPSHOT)
    require(spec["selected_real_window_size"] == WINDOW, "final Real window is not 201")
    require(spec["final_neural_model_instances"] == 72 and len(spec["final_slots"]) == 72,
            "expected exactly 72 frozen neural slots")
    require(spec["somatotopic_partition_sha256"] == sha256(part_path), "partition hash drift")
    require(part["partition_version"] == "real_somatotopic_v1", "partition version drift")
    ids = part["feature_order_nlb_unit_ids"]
    require(len(ids) == 65 and len(set(ids)) == 65, "65-unit mapping invalid")
    a = part["A_PROXIMAL"]["feature_indices"]
    b = part["B_DISTAL"]["feature_indices"]
    require(len(a) == 32 and len(b) == 33 and not set(a) & set(b)
            and set(a + b) == set(range(65)), "32+33 partition invalid")
    for population in ("A_PROXIMAL", "B_DISTAL"):
        indices = part[population]["feature_indices"]
        require([ids[i] for i in indices] == part[population]["nlb_unit_ids"],
                f"{population} feature-to-unit mapping invalid")
    for relative in SOURCE_FILES:
        require(sha256(project / relative) == snapshot["source_file_sha256"][relative],
                f"model source differs from frozen HPO source: {relative}")
    return spec, part, snapshot


def channel_indices(part: dict, population: str) -> list[int]:
    require(population in POPULATIONS, f"unknown population: {population}")
    return list(range(65)) if population == "TOTAL65" else list(part[population]["feature_indices"])


def verified_checkpoint(project: Path, slot: dict, part: dict) -> tuple[Path, dict, dict, dict]:
    trial_root = (Path(slot["checkpoint_path"]).parent if slot.get("reuse_hpo_checkpoint")
                  else Path(slot["output_path"]))
    checkpoint = trial_root / "best_validation_checkpoint.pt"
    record_path = trial_root / "trial_record.json"
    result_path = trial_root / "result.json"
    require(all(path.is_file() for path in (checkpoint, record_path, result_path)),
            f"frozen slot lacks checkpoint/record/result: {slot['trial_id']}")
    record = read_json(record_path)
    result = read_json(result_path)
    config = record["config"]
    require(record["trial_id"] == result["trial_id"] == slot["trial_id"], "trial ID mismatch")
    require(record["config_sha256"] == slot["config_sha256"], "config hash mismatch")
    require(config["architecture"] == slot["architecture"] and
            config["objective"] == slot["objective"] and
            config["population"] == slot["population"] and
            config["training_seed_root"] == slot["training_seed_root"] and
            config["window_size"] == WINDOW, "slot/config fields differ")
    require(config["channel_indices"] == channel_indices(part, slot["population"]),
            "checkpoint was not trained on canonical channel order")
    require(config["channel_partition_sha256"] == sha256(project / PARTITION),
            "checkpoint partition hash differs")
    reference = Path(slot["safe_input_reference"])
    require(sha256(reference) == slot["safe_input_reference_sha256"],
            "train/validation bundle reference drift")
    ref = read_json(reference)
    indices = channel_indices(part, slot["population"])
    require(ref["domain"] == "real_w201" and ref["population"] == slot["population"] and
            ref["partition_version"] == "real_somatotopic_v1" and
            ref["partition_sha256"] == sha256(project / PARTITION) and
            ref["channel_indices"] == indices and
            ref["nlb_unit_ids"] == [part["feature_order_nlb_unit_ids"][i] for i in indices],
            "safe-bundle reference feature order differs from canonical partition")
    bundle_manifest = Path(ref["bundle_path"]) / "manifest.json"
    require(sha256(bundle_manifest) == ref["bundle_manifest_sha256"],
            "safe-bundle manifest differs from frozen reference")
    bundle_root = bundle_manifest.parent
    bundle_values = read_json(bundle_manifest)
    safe_windows = bundle_root / "windows.npz"
    safe_split = bundle_root / "split.json"
    require(sha256(safe_windows) == ref["safe_windows_sha256"] ==
            bundle_values["safe_windows_sha256"],
            "safe-bundle windows differ from frozen reference/manifest")
    require(sha256(safe_split) == ref["safe_split_sha256"] ==
            bundle_values["safe_split_sha256"],
            "safe-bundle split differs from frozen reference/manifest")
    require(bundle_values["population"] == slot["population"] and
            bundle_values["domain"] == "real_w201" and
            bundle_values["original_split_sha256"] == ref["original_split_sha256"],
            "safe-bundle content identity differs from its frozen reference")
    require(result["status"] in {"ELIGIBLE", "INELIGIBLE_NEAR_COLLAPSE"},
            "failed fit has no usable frozen encoder")
    checkpoint_sha = sha256(checkpoint)
    require(result["artifact_sha256"][checkpoint.name] == checkpoint_sha,
            "checkpoint differs from fit-result hash")
    if slot.get("checkpoint_sha256"):
        require(slot["checkpoint_sha256"] == checkpoint_sha, "spec checkpoint hash differs")
    return checkpoint, record, result, config


def load_encoder(checkpoint: Path, config: dict, n_features: int,
                 expected_trial_id: str, device: torch.device) -> torch.nn.Module:
    settings = config["architecture_settings"]
    base = real.RealMonkeyConfig()
    if config["architecture"] == "cnn1d":
        require(settings == {"hidden_dim": base.hidden_dim, "cnn_layers": base.cnn_layers},
                "CNN architecture differs from frozen configuration")
    else:
        require(settings == {"transformer_dim": base.transformer_dim,
                             "transformer_heads": base.transformer_heads,
                             "transformer_layers": base.transformer_layers,
                             "transformer_dropout": base.transformer_dropout},
                "Transformer architecture differs from frozen configuration")
    require(config["embedding_dim"] == 3, "embedding dimension must be 3")
    model_config = replace(base, window_size=WINDOW, embedding_dim=3)
    model = real._make_real_model(config["architecture"], n_features, model_config,
                                  normalize=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(payload["trial_id"] == expected_trial_id, "checkpoint trial ID differs")
    require(payload["model_name"] == config["architecture"] and
            payload["objective"] == config["objective"], "checkpoint model/objective mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


def encode_batch(model: torch.nn.Module, windows: np.ndarray,
                 device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    require(windows.ndim == 3 and windows.shape[1] == WINDOW, "input window shape differs")
    with torch.inference_mode():
        raw = model(torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32)).to(device))
        unit = F.normalize(raw, p=2, dim=-1)
    raw_np = raw.cpu().numpy().astype(np.float32, copy=False)
    unit_np = unit.cpu().numpy().astype(np.float32, copy=False)
    require(raw_np.shape == unit_np.shape == (len(windows), 3), "N×3 encoder invariant failed")
    require(np.isfinite(raw_np).all() and np.isfinite(unit_np).all(), "nonfinite embedding")
    return raw_np, unit_np


def centered_batches(spikes_trial: np.ndarray, batch_size: int):
    """Yield every trial center in order, with the frozen zero-padding rule."""
    require(spikes_trial.shape[0] == TRIAL_LENGTH and batch_size > 0, "trial/batch shape invalid")
    radius = WINDOW // 2
    padded = np.pad(spikes_trial, ((radius, radius), (0, 0)), mode="constant")
    for start in range(0, TRIAL_LENGTH, batch_size):
        times = np.arange(start, min(start + batch_size, TRIAL_LENGTH), dtype=np.int64)
        windows = np.stack([padded[t:t + WINDOW] for t in times]).astype(np.float32, copy=False)
        yield times, windows


def valid_centers(valid_bins_trial: np.ndarray) -> np.ndarray:
    require(valid_bins_trial.shape == (TRIAL_LENGTH,), "valid-bin length differs")
    radius = WINDOW // 2
    valid = np.zeros(TRIAL_LENGTH, dtype=bool)
    for center in range(radius, TRIAL_LENGTH - radius):
        valid[center] = bool(np.all(valid_bins_trial[center-radius:center+radius+1]))
    return valid


def stage1_arrays(project: Path, spec: dict) -> tuple[dict, dict, dict]:
    parent = project / STAGE1
    data_path, split_path, manifest_path = parent / "data.npz", parent / "split.json", parent / "manifest.json"
    split = read_json(split_path)
    manifest = read_json(manifest_path)
    total_slot = next((slot for slot in spec["final_slots"] if slot["population"] == "TOTAL65"), None)
    require(total_slot is not None, "frozen Real spec has no TOTAL65 reference slot")
    reference_path = Path(total_slot["safe_input_reference"])
    if not reference_path.is_absolute():
        reference_path = project / reference_path
    require(sha256(reference_path) == total_slot["safe_input_reference_sha256"],
            "TOTAL65 safe-input reference hash differs")
    reference = read_json(reference_path)
    bundle_manifest_path = Path(reference["bundle_path"]) / "manifest.json"
    require(sha256(bundle_manifest_path) == reference["bundle_manifest_sha256"],
            "TOTAL65 safe-bundle manifest hash differs from reference")
    safe_manifest = read_json(bundle_manifest_path)
    require(set(split) == {"train", "validation", "test"}, "frozen split schema differs")
    require(sha256(split_path) == spec["original_frozen_split_sha256"], "split hash differs")
    require(manifest["config"]["channel_indices"] == list(range(65)),
            "Stage1 parent is not TOTAL65 in original feature order")
    require(sha256(data_path) == safe_manifest["parent_data_sha256"] and
            sha256(split_path) == safe_manifest["original_split_sha256"] and
            manifest["source"]["source_sha256"] == safe_manifest["raw_source_sha256"],
            "Stage1 source differs from frozen HPO lineage")
    with np.load(data_path, allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in
                  ("spikes", "target_by_trial", "position", "velocity", "valid_bins")}
    n_trials = len(arrays["target_by_trial"])
    require(n_trials == 193 and arrays["spikes"].shape == (n_trials * TRIAL_LENGTH, 65),
            "preprocessed Stage1 data shape differs")
    require(set().union(*(set(split[key]) for key in split)) == set(range(n_trials)) and
            all(not (set(split[x]) & set(split[y])) for x, y in
                (("train", "validation"), ("train", "test"), ("validation", "test"))),
            "frozen trial split does not partition Stage1")
    return arrays, split, {"data_npz_sha256": sha256(data_path),
                           "split_json_sha256": sha256(split_path),
                           "manifest_json_sha256": sha256(manifest_path),
                           "raw_source_sha256": manifest["source"]["source_sha256"]}


def authorize_real_export(project: Path, spec_path: Path, evaluation_manifest: Path) -> dict:
    """The explicit support freeze is required before opening any test rows."""
    policy = read_json(evaluation_manifest)
    canonical_policy_path = (project / FINAL_ROOT / "final_evaluation" / "core_metrics" /
                             "EVALUATION_SUPPORT_MANIFEST.json").resolve()
    require(evaluation_manifest.resolve() == canonical_policy_path,
            "Real export requires the core branch's canonical support manifest")
    audit_path = canonical_policy_path.parent / "AUDIT_MANIFEST.json"
    require(audit_path.is_file() and policy.get("audit_manifest_sha256") == sha256(audit_path),
            "support policy is not descended from the current verified source audit")
    require(policy.get("final_spec_sha256") == sha256(spec_path) and
            policy.get("window_size") == WINDOW and
            policy.get("embedding_scope") == "all_centered_rows_with_valid_mask" and
            policy.get("test_opening_authorized") is True,
            "evaluation support/test-opening policy not frozen for this specification")
    require((project / FINAL_ROOT) in evaluation_manifest.resolve().parents,
            "evaluation authorization must be within final-thesis namespace")
    return policy


def export_slot(project: Path, slot: dict, spec: dict, part: dict, arrays: dict,
                split: dict, parent_hashes: dict, policy_path: Path,
                *, batch_size: int, device: torch.device,
                output_root: Path | None = None) -> Path:
    checkpoint, record, result, config = verified_checkpoint(project, slot, part)
    canonical_output = project / FINAL_ROOT / "embeddings"
    if output_root is None:
        output_root = canonical_output
    else:
        output_root = output_root.resolve()
        required = (project / FINAL_ROOT / "final_evaluation" / "core_metrics" / "embeddings").resolve()
        require(output_root == required,
                "custom output root is reserved for the new immutable core-metrics branch")
    out = output_root / slot["trial_id"]
    meta_path = out / "manifest.json"
    parent = {
        "trial_id": slot["trial_id"], "population": slot["population"],
        "architecture": slot["architecture"], "objective": slot["objective"],
        "seed": slot["training_seed_root"], "window_size": WINDOW,
        "channel_indices": channel_indices(part, slot["population"]),
        "nlb_unit_ids": [part["feature_order_nlb_unit_ids"][i]
                         for i in channel_indices(part, slot["population"])],
        "partition_sha256": sha256(project / PARTITION),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_path": str(checkpoint), "config_sha256": record["config_sha256"],
        "trial_record_sha256": sha256(checkpoint.parent / "trial_record.json"),
        "fit_result_sha256": sha256(checkpoint.parent / "result.json"),
        "final_spec_sha256": sha256(project / SPEC),
        "evaluation_support_sha256": sha256(policy_path),
        "exporter_source_sha256": sha256(Path(__file__)), **parent_hashes,
    }
    files = {"raw": out / "embedding_raw.npz", "unit": out / "embedding_unit.npz",
             "metadata": out / "evaluation_metadata.npz"}
    if out.exists():
        require(meta_path.is_file(), f"partial export preserved: {out}")
        prior = read_json(meta_path)
        require(prior["parents"] == parent, f"existing export lineage differs: {out}")
        require(all(path.is_file() and sha256(path) == prior["artifact_sha256"][path.name]
                    for path in files.values()), f"existing export content differs: {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name(f"{out.name}.staging-{uuid.uuid4().hex[:12]}")
    staging.mkdir(exist_ok=False)
    staged_files = {key: staging / path.name for key, path in files.items()}
    n_trials = len(arrays["target_by_trial"])
    n_rows = n_trials * TRIAL_LENGTH
    indices = channel_indices(part, slot["population"])
    model = load_encoder(checkpoint, config, len(indices), slot["trial_id"], device)
    raw = np.empty((n_rows, 3), dtype=np.float32)
    unit = np.empty_like(raw)
    mask = np.empty(n_rows, dtype=bool)
    spikes = arrays["spikes"].reshape(n_trials, TRIAL_LENGTH, 65)
    valid_bins = arrays["valid_bins"].reshape(n_trials, TRIAL_LENGTH)
    for trial in range(n_trials):
        selected = spikes[trial][:, indices]
        mask[trial * TRIAL_LENGTH:(trial + 1) * TRIAL_LENGTH] = valid_centers(valid_bins[trial])
        for times, windows in centered_batches(selected, batch_size):
            a, b = encode_batch(model, windows, device)
            rows = trial * TRIAL_LENGTH + times
            raw[rows], unit[rows] = a, b
    trial_id = np.repeat(np.arange(n_trials, dtype=np.int64), TRIAL_LENGTH)
    time_id = np.tile(np.arange(TRIAL_LENGTH, dtype=np.int64), n_trials)
    split_by_trial = np.empty(n_trials, dtype="U10")
    for name, ids in split.items():
        split_by_trial[np.asarray(ids, dtype=np.int64)] = name
    require(raw.shape == unit.shape == (len(trial_id), 3) and len(mask) == len(time_id),
            "embedding/metadata cardinality differs")
    np.savez_compressed(staged_files["raw"], embedding_raw=raw)
    np.savez_compressed(staged_files["unit"], embedding_unit=unit)
    np.savez_compressed(staged_files["metadata"],
                        trial_id=trial_id, time_id=time_id, global_time_id=trial_id * TRIAL_LENGTH + time_id,
                        valid_mask=mask, split=split_by_trial[trial_id],
                        target=np.asarray(arrays["target_by_trial"])[trial_id],
                        progress=time_id.astype(np.float32) / (TRIAL_LENGTH - 1),
                        position=np.asarray(arrays["position"]), velocity=np.asarray(arrays["velocity"]))
    manifest = {"parents": parent, "rows": n_rows, "embedding_shape": [n_rows, 3],
                "normalization": "unit = torch.nn.functional.normalize(raw, p=2, dim=-1)",
                "checkpoint_selected_update": result["selected_update"],
                "fit_status": result["status"],
                "near_collapse": bool(result.get("geometry", {}).get("near_collapse", False)),
                "artifact_sha256": {path.name: sha256(path) for path in staged_files.values()}}
    with (staging / "manifest.json").open("xb") as stream:
        stream.write(json_bytes(manifest))
    require(not out.exists(), f"final export appeared during staging: {out}")
    staging.rename(out)
    return out


def smoke_mock(project: Path) -> None:
    """One frozen encoder on generated data only: no source/split/test loading."""
    spec, part, _ = frozen_inputs(project)
    slot = next(slot for slot in spec["final_slots"] if slot["training_seed_root"] == 1101 and
                slot["population"] == "TOTAL65" and slot["architecture"] == "cnn1d")
    checkpoint, _, _, config = verified_checkpoint(project, slot, part)
    device = torch.device("cpu")
    model = load_encoder(checkpoint, config, 65, slot["trial_id"], device)
    dummy = np.zeros((TRIAL_LENGTH, 65), dtype=np.float32)
    batches = centered_batches(dummy, 7)
    times, windows = next(batches)
    raw, unit = encode_batch(model, windows, device)
    require(len(times) == 7 and raw.shape == unit.shape == (7, 3), "mock batch failed")
    require(valid_centers(np.ones(TRIAL_LENGTH, dtype=bool)).sum() == 400,
            "centered validity mask differs from frozen W201 rule")
    for population in ("A_PROXIMAL", "B_DISTAL"):
        peer = next(item for item in spec["final_slots"] if item["training_seed_root"] == 1101 and
                    item["population"] == population and item["architecture"] == "cnn1d")
        verified_checkpoint(project, peer, part)
    print("MOCK_ONLY_PASS: one saved seed1101 checkpoint, W201 streaming batch 7, raw/unit N×3; canonical A/B parent records checked; no observations/splits opened")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-mock", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--output-root", type=Path,
                        help="Optional immutable final-evaluation embedding root")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--trial-id")
    args = parser.parse_args()
    project = ROOT.resolve()
    if args.smoke_mock:
        smoke_mock(project)
        return
    require(args.evaluation_manifest is not None, "--run requires frozen evaluation-support manifest")
    policy_path = args.evaluation_manifest.resolve()
    authorize_real_export(project, project / SPEC, policy_path)
    spec, part, _ = frozen_inputs(project)
    arrays, split, parent_hashes = stage1_arrays(project, spec)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slots = [slot for slot in spec["final_slots"] if args.trial_id in (None, slot["trial_id"])]
    require(slots, "no frozen slot matches --trial-id")
    for i, slot in enumerate(slots, 1):
        out = export_slot(project, slot, spec, part, arrays, split, parent_hashes,
                          policy_path, batch_size=args.batch_size, device=device,
                          output_root=args.output_root)
        print(f"{i}/{len(slots)} {slot['trial_id']} -> {out}", flush=True)


if __name__ == "__main__":
    main()
