"""Read-only evaluation of frozen checkpoints; writes a separate report directory."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import joblib
import numpy as np
import torch

from neurobridge.eval.representation import evaluate_latent_recovery, lagged_alignment_by_trial_time
from neurobridge.experiments.audit_contracts import (
    check_cached_config, file_sha256, interior_mask, source_provenance,
    validate_metadata, validate_split,
)
from neurobridge.experiments.staged_shared_latent import (
    SharedLatentStageConfig, _append_decoder_metrics, _make_model,
)


LOSS_LABELS = {
    "cebra_time": "Time Contrastive Blocks",
    "cebra_behavior": "Behavior Contrastive Blocks",
    "soft": "Soft structured contrastive",
    "infonce": "Supervised InfoNCE",
    "none": "PCA",
}


def _evaluation_scope(branch):
    """Describe whether representation training excluded evaluation trials."""
    if branch == "held_out":
        return "held_out_representation_generalization"
    if branch == "full_sample":
        return "descriptive_in_sample_representation"
    raise ValueError("branch must be 'held_out' or 'full_sample'")


def _evaluation_mask(masks, valid, branch):
    """Select the fixed support for one evaluation branch."""
    _evaluation_scope(branch)
    valid = np.asarray(valid, dtype=bool)
    if branch == "held_out":
        return np.asarray(masks["test"], dtype=bool) & valid
    return valid


def read_config(run):
    payload = json.loads((Path(run) / "stage01_data/config.json").read_text(encoding="utf-8"))
    return SharedLatentStageConfig(**payload)


def verify_frozen_checkpoint(path, values, windows, config, model_config, checkpoint, *, diagnostics=None):
    """Check deterministic, spread-out rows against the actual saved checkpoint."""
    count = len(values["trial_id"])
    for key in ("embedding_raw", "embedding_unit"):
        if values[key].shape != (count, config.latent_dim) or not np.all(np.isfinite(values[key])):
            raise ValueError(f"Invalid {key}: {path}")
    unit = values["embedding_raw"] / np.maximum(np.linalg.norm(values["embedding_raw"], axis=1, keepdims=True), 1e-8)
    if not np.allclose(unit, values["embedding_unit"], atol=2e-5, rtol=2e-4):
        raise ValueError(f"Raw/unit mismatch: {path}")
    for key in ("trial_id", "time_id", "global_time_id", "labels", "progress", "lag_valid"):
        if not np.array_equal(values[key], windows[key]):
            raise ValueError(f"Frozen/window metadata mismatch for {key}: {path}")
    indices = np.unique(np.linspace(0, count - 1, min(64, count), dtype=int))
    x = windows["X_windows"][indices]
    model_name = model_config["model"]
    if model_name == "pca":
        raw = joblib.load(checkpoint).transform(x.reshape(len(x), -1))
    else:
        model = _make_model(model_name, x.shape[-1], config, normalize=False)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
        model.eval()
        with torch.no_grad():
            raw = model(torch.as_tensor(x, dtype=torch.float32)).numpy()
    cached = values["embedding_raw"][indices]
    reproduced = bool(np.allclose(raw, cached, atol=2e-4, rtol=2e-4))
    if diagnostics is not None:
        diagnostics.update({
            "checkpoint_reproduced_within_tolerance": reproduced,
            "sampled_rows_checked": len(indices),
            "atol": 2e-4, "rtol": 2e-4,
            "max_absolute_difference": float(np.max(np.abs(raw - cached))),
            "relative_frobenius_difference": float(np.linalg.norm(raw - cached) / max(np.linalg.norm(cached), 1e-12)),
        })
    if not reproduced and diagnostics is None:
        raise ValueError(f"Checkpoint does not reproduce sampled frozen rows: {path}")
    return len(indices)


def evaluate_frozen_run(
    project_root,
    run,
    output,
    *,
    branch="held_out",
    record_checkpoint_mismatches=False,
):
    """Audit all inputs first, then evaluate BOTH representations without selection.

    Historical training provenance cannot be inferred from present-day hashes.
    The report distinguishes this limitation from verified artifact consistency.
    """
    run, output = Path(run).resolve(), Path(output).resolve()
    evaluation_scope = _evaluation_scope(branch)
    if output == run or run in output.parents:
        raise ValueError("Use a separate output directory outside the source run")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Report directory must be new or empty")
    config = read_config(run)
    check_cached_config(run / "stage02_windows/config.json", config)
    split_path = run / "stage02_windows/split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validate_split(split, np.arange(config.n_trials), config.split_counts)
    data_path = run / "stage01_data/shared_data.npz"
    # Load only the generative targets, not full spike/rate arrays.
    with np.load(data_path, allow_pickle=False) as source:
        data = {key: source[key] for key in ("M", "eta", "Z_A", "Z_B", "labels", "valid_A", "valid_B")}
    files = {str(p): file_sha256(p) for p in (
        data_path, split_path, run / "stage01_data/config.json", run / "stage02_windows/config.json",
    )}
    artifacts = {}
    provenance = []
    for subject in ("A", "B"):
        window_path = run / f"stage02_windows/windows_{subject}.npz"
        files[str(window_path)] = file_sha256(window_path)
        with np.load(window_path, allow_pickle=False) as source:
            windows = {key: source[key] for key in source.files}
        validate_metadata(windows, data, config, subject)
        # Check full center values against source spikes without rebuilding windows.
        with np.load(data_path, allow_pickle=False) as source:
            spikes = source[f"X_{subject}"]
        if not np.array_equal(windows["X_windows"][:, config.window_size // 2],
                              spikes[windows["trial_id"], windows["time_id"]]):
            raise ValueError("Window center values disagree with the source spike matrix")
        del spikes
        for path in sorted((run / "stage04_embeddings" / branch).glob(f"{subject}_*.npz")):
            model_dir = run / "stage03_models" / branch / path.stem
            check_cached_config(model_dir / "config.json", config)
            mc = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
            if mc["subject"] != subject or mc["branch"] != branch or path.stem != f"{subject}_{mc['model']}_{mc['loss']}":
                raise ValueError(f"Model identity mismatch: {model_dir}")
            checkpoint = model_dir / ("pca.joblib" if mc["model"] == "pca" else "model.pt")
            with np.load(path, allow_pickle=False) as source:
                values = {key: source[key] for key in source.files}
            validate_metadata(values, data, config, subject)
            diagnostics = {} if record_checkpoint_mismatches else None
            sampled = verify_frozen_checkpoint(path, values, windows, config, mc, checkpoint, diagnostics=diagnostics)
            for artifact in (path, checkpoint, model_dir / "config.json"):
                files[str(artifact)] = file_sha256(artifact)
            key = (mc["model"], mc["loss"])
            artifacts.setdefault(key, {})[subject] = values
            provenance.append({
                "artifact": str(path), "checkpoint": str(checkpoint),
                "sampled_rows_checked": sampled,
                **(diagnostics if diagnostics is not None else {"checkpoint_reproduced_within_tolerance": True}),
                "historical_training_provenance": "present" if (model_dir / "provenance.json").exists() else "UNVERIFIED: legacy checkpoint has no training source/input hashes",
            })
        del windows
    if not artifacts or any(set(subjects) != {"A", "B"} for subjects in artifacts.values()):
        raise ValueError("Every evaluated method requires frozen checkpoints for A and B")

    # Fixed protocol is recorded BEFORE examining any metrics.
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": asdict(config), "evaluation_source": source_provenance(project_root),
        "checkpoint_policy": "record discrepancies; evaluate saved embeddings" if record_checkpoint_mismatches else "strict reproduction required",
        "checkpoint_reproduction_passed": all(p["checkpoint_reproduced_within_tolerance"] for p in provenance),
        "runtime": {"torch": torch.__version__, "numpy": np.__version__},
        "source_files_sha256": files, "checkpoint_checks": provenance,
        "branch": branch,
        "evaluation_scope": evaluation_scope,
        "representations": ["raw", "unit"], "selection": "none; both representations reported",
        "latent_target": "Z only",
        "geometry_support": (
            "test windows with all input bins valid; same rows and RSA seed=0 for raw/unit"
            if branch == "held_out"
            else "all full-sample windows with all input bins valid; same rows and RSA seed=0 for raw/unit"
        ),
        "rsa_max_pairs": 100000, "lag_candidates": list(range(-20, 21)),
        "lag_support": "common reference trial/time coordinates across ALL candidate lags; fully valid windows",
        "lag_sign": "positive lag compares A(trial,t) with B(trial,t+lag)",
        "decoder_selection": "train-fit scaling; hyperparameters from validation only",
        "limitations": [
            "Historical source versions cannot be certified retrospectively.",
            "If checkpoint_reproduction_passed is false, scores describe saved embeddings and do not certify current checkpoint reproduction.",
            "Checkpoint reproduction samples 64 rows per artifact, not every row.",
            "Legacy training included some windows touching the lag prefix; evaluation cannot undo this.",
            "Lag is estimated and alignment fitted/evaluated on the same observations: descriptive lag recovery, not an out-of-sample alignment score.",
            "For full_sample, encoder training included every trial partition; its results are descriptive and not representation generalization.",
            "One data/training seed cannot establish ranking stability or universal recovery.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows, selections, profiles = [], [], []
    for (model_name, loss_name), subjects in artifacts.items():
        label = LOSS_LABELS.get(loss_name, loss_name)
        subject_masks = {}
        for representation in ("raw", "unit"):
            embedding_key = f"embedding_{representation}"
            for subject, values in subjects.items():
                masks = validate_split(split, values["trial_id"], config.split_counts)
                valid = interior_mask(values["time_id"], config, subject)
                masks = {name: mask & valid for name, mask in masks.items()}
                subject_masks[subject] = (masks, valid)
                base = {"model": model_name, "loss": label, "subject": subject,
                        "representation": representation, "branch": branch,
                        "evaluation_scope": evaluation_scope}
                evaluation = _evaluation_mask(masks, valid, branch)
                trial, time = values["trial_id"][evaluation], values["time_id"][evaluation]
                scores = evaluate_latent_recovery(
                    values[embedding_key][evaluation], data[f"Z_{subject}"][trial, time]
                )
                rows.extend({**base, "category": "latent_recovery", "reference": "Z",
                             "metric": metric, "value": value, "n_samples": int(evaluation.sum())}
                            for metric, value in scores.items())
                selections.append(_append_decoder_metrics(
                    rows, base=base,
                    **{f"{name}_embedding": values[embedding_key][masks[name]] for name in masks},
                    **{f"{name}_labels": values["labels"][masks[name]] for name in masks},
                    **{f"{name}_progress": values["progress"][masks[name]] for name in masks},
                    noise_scales=(), random_state=config.seed,
                ))
            a, b = subjects["A"], subjects["B"]
            a_masks, a_valid = subject_masks["A"]
            b_masks, b_valid = subject_masks["B"]
            am = _evaluation_mask(a_masks, a_valid, branch)
            bm = _evaluation_mask(b_masks, b_valid, branch)
            best, scores, pairs = lagged_alignment_by_trial_time(
                a[embedding_key][am], b[embedding_key][bm], a["trial_id"][am], a["time_id"][am],
                b["trial_id"][bm], b["time_id"][bm], range(-20, 21), common_support=True,
            )
            base = {"model": model_name, "loss": label, "subject": "A_B", "representation": representation,
                    "branch": branch, "evaluation_scope": evaluation_scope, "category": "lag"}
            rows.extend([{**base, "metric": "estimated_lag_bins", "value": best},
                         {**base, "metric": "absolute_lag_error_bins", "value": abs(best - config.lag_bins)}])
            profiles.extend({**base, "lag": lag, "score": score, "n_pairs": len(pairs[lag][0])}
                            for lag, score in scores.items())
    for name, records in (("metrics.csv", rows), ("lag_profiles.csv", profiles)):
        fields = list(dict.fromkeys(key for row in records for key in row))
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    (output / "decoder_selections.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")
    return output / "metrics.csv"


def repeated_seed_plan(config):
    """Preparation only: no fitting, data generation or model selection."""
    return {
        "status": "PLANNED; no runs launched",
        "design": "fixed data seed and split; paired training seeds across methods",
        "split_seed": config.seed + 101 if config.split_seed is None else config.split_seed,
        "training_seeds": [101, 202, 303, 404, 505],
        "analysis": "Report every seed, mean, SD and paired method differences; confidence intervals resample seed-level differences, never overlapping windows. Five seeds give limited precision.",
        "scope": "Optimization variability conditional on this dataset/split; split/data robustness needs a separate campaign.",
        "configs": [asdict(replace(config, training_seed=seed,
                                  split_seed=config.seed + 101 if config.split_seed is None else config.split_seed,
                                  run_label=f"audit_validated_train_seed_{seed}"))
                    for seed in (101, 202, 303, 404, 505)],
    }
