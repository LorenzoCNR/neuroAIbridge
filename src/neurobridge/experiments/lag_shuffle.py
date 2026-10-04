"""Permutation null for cross-population lag recovery from frozen embeddings."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from neurobridge.eval.representation import lagged_alignment_by_trial_time
from neurobridge.experiments.audit_contracts import interior_mask, validate_split
from neurobridge.experiments.frozen_sensitivity import (
    LOSS_LABELS,
    _evaluation_mask,
    read_config,
)


LAGS = tuple(range(-20, 21))


def shuffled_trial_ids(trial_ids, rng):
    """Return a derangement of trial IDs, preserving every ID exactly once."""
    trial_ids = np.asarray(trial_ids)
    unique = np.unique(trial_ids)
    if len(unique) < 2:
        raise ValueError("At least two trials are required for a shuffle null")
    shuffled = rng.permutation(unique)
    while np.any(shuffled == unique):
        shuffled = rng.permutation(unique)
    mapping = dict(zip(unique.tolist(), shuffled.tolist()))
    return np.asarray([mapping[int(value)] for value in trial_ids], dtype=trial_ids.dtype)


def _artifact_values(run, branch):
    artifacts = {}
    for subject in ("A", "B"):
        for path in sorted((run / "stage04_embeddings" / branch).glob(f"{subject}_*.npz")):
            stem = path.stem.split("_", 1)
            if len(stem) != 2:
                raise ValueError(f"Unexpected embedding artifact name: {path}")
            method_name = stem[1]
            model_name = next((name for name in ("cnn1d", "transformer", "pca") if method_name.startswith(f"{name}_")), None)
            if model_name is None:
                raise ValueError(f"Unexpected method name: {path}")
            loss_name = method_name[len(model_name) + 1:]
            with np.load(path, allow_pickle=False) as source:
                values = {key: source[key] for key in source.files}
            artifacts.setdefault((model_name, loss_name), {})[subject] = values
    if len(artifacts) != 9 or any(set(subjects) != {"A", "B"} for subjects in artifacts.values()):
        raise ValueError("Lag shuffle requires all nine frozen methods for both subjects")
    return artifacts


def _scope_masks(values, split, config, subject, branch):
    masks = validate_split(split, values["trial_id"], config.split_counts)
    valid = interior_mask(values["time_id"], config, subject)
    return _evaluation_mask(masks, valid, branch)


def _alignment(embedding_a, embedding_b, trial_a, time_a, trial_b, time_b):
    return lagged_alignment_by_trial_time(
        embedding_a,
        embedding_b,
        trial_a,
        time_a,
        trial_b,
        time_b,
        LAGS,
        common_support=True,
    )


def evaluate_lag_shuffle(
    project_root,
    run,
    output,
    *,
    branch="held_out",
    n_permutations=20,
    seed=0,
):
    """Evaluate observed lag against a trial-pairing permutation null.

    The embeddings, within-trial time coordinates, and candidate lags remain
    unchanged.  Only B's trial IDs are deranged, so a null peak cannot be
    attributed to the known A/B trial correspondence.
    """
    if branch not in {"held_out", "full_sample"}:
        raise ValueError("branch must be 'held_out' or 'full_sample'")
    if int(n_permutations) < 2:
        raise ValueError("n_permutations must be at least 2")
    run, output = Path(run).resolve(), Path(output).resolve()
    if output == run or run in output.parents:
        raise ValueError("Use a separate output directory outside the source run")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Lag-shuffle report directory must be new or empty")

    config = read_config(run)
    split = json.loads((run / "stage02_windows/split.json").read_text(encoding="utf-8"))
    artifacts = _artifact_values(run, branch)
    rows, profiles = [], []
    rng = np.random.default_rng(seed)
    for method_index, ((model_name, loss_name), subjects) in enumerate(sorted(artifacts.items())):
        a, b = subjects["A"], subjects["B"]
        a_mask = _scope_masks(a, split, config, "A", branch)
        b_mask = _scope_masks(b, split, config, "B", branch)
        a_trial, a_time = a["trial_id"][a_mask], a["time_id"][a_mask]
        b_trial, b_time = b["trial_id"][b_mask], b["time_id"][b_mask]
        label = LOSS_LABELS.get(loss_name, loss_name)
        for representation in ("raw", "unit"):
            key = f"embedding_{representation}"
            observed_best, observed_scores, observed_pairs = _alignment(
                a[key][a_mask], b[key][b_mask], a_trial, a_time, b_trial, b_time
            )
            observed_peak = float(observed_scores[observed_best])
            null_best, null_peak, null_true = [], [], []
            for permutation in range(1, int(n_permutations) + 1):
                shuffled_b_trial = shuffled_trial_ids(b_trial, rng)
                best, scores, pairs = _alignment(
                    a[key][a_mask], b[key][b_mask], a_trial, a_time,
                    shuffled_b_trial, b_time,
                )
                null_best.append(int(best))
                null_peak.append(float(scores[best]))
                null_true.append(float(scores[config.lag_bins]))
                profiles.extend({
                    "branch": branch,
                    "model": model_name,
                    "loss": label,
                    "representation": representation,
                    "permutation": permutation,
                    "lag": int(lag),
                    "score": float(score),
                    "n_pairs": len(pairs[lag][0]),
                } for lag, score in sorted(scores.items()))
            profiles.extend({
                "branch": branch,
                "model": model_name,
                "loss": label,
                "representation": representation,
                "permutation": 0,
                "lag": int(lag),
                "score": float(score),
                "n_pairs": len(observed_pairs[lag][0]),
            } for lag, score in sorted(observed_scores.items()))
            null_peak_array = np.asarray(null_peak, dtype=float)
            p_value = (1.0 + float(np.sum(null_peak_array >= observed_peak))) / (len(null_peak_array) + 1.0)
            rows.append({
                "branch": branch,
                "model": model_name,
                "loss": label,
                "representation": representation,
                "observed_lag_bins": int(observed_best),
                "true_lag_bins": int(config.lag_bins),
                "absolute_lag_error_bins": abs(int(observed_best) - int(config.lag_bins)),
                "observed_peak_score": observed_peak,
                "observed_true_lag_score": float(observed_scores[config.lag_bins]),
                "null_permutations": int(n_permutations),
                "null_peak_mean": float(null_peak_array.mean()),
                "null_peak_sd": float(null_peak_array.std(ddof=1)),
                "null_peak_q025": float(np.quantile(null_peak_array, 0.025)),
                "null_peak_q975": float(np.quantile(null_peak_array, 0.975)),
                "null_true_lag_mean": float(np.mean(null_true)),
                "null_peak_p_value": p_value,
                "null_best_lag_mean": float(np.mean(null_best)),
                "null_best_lag_sd": float(np.std(null_best, ddof=1)),
            })

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": asdict(config),
        "branch": branch,
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "null": "deranged B trial IDs; within-trial time and embedding values preserved",
        "candidate_lags": list(LAGS),
        "support": "strict interior windows; common trial/time support for every candidate lag",
        "interpretation": "null for cross-population trial-pairing lag, not a causal test",
        "outputs": ["summary.csv", "profiles.csv"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, records in (("summary.csv", rows), ("profiles.csv", profiles)):
        fields = list(dict.fromkeys(key for row in records for key in row))
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    return output / "summary.csv"
