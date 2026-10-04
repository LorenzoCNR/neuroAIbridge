"""Frozen downstream null, uncertainty, and controlled-lag analyses.

This driver reads only the immutable final-evaluation embeddings/manifests and
their already frozen source inputs. It never fits or updates an encoder. Each
analysis is published to its own write-once branch.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import final_thesis_core_metrics as core  # noqa: E402
from neurobridge.eval.representation import (  # noqa: E402
    _sampled_pair_distances,
    distance_geometry_correlation,
    lagged_alignment_by_trial_time,
    linear_cka,
    procrustes_r2,
)
from neurobridge.experiments.lag_shuffle import LAGS, shuffled_trial_ids  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
FINAL = PROJECT / "outputs" / "final_thesis_v1"
EVAL = FINAL / "final_evaluation"
CORE = EVAL / "core_metrics"
SEEDS = (1101, 1201, 1301)
PAIR_COUNT = core.PAIR_COUNT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"write-once artifact already exists: {path}")
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(payload, encoding="utf-8", newline="")
    temp.replace(path)


def csv_write_once(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"write-once artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def csv_read(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json_write_once(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish one immutable JSON checkpoint on the same volume."""
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == serialized:
            return
        raise FileExistsError(f"write-once checkpoint differs from existing artifact: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            if path.read_text(encoding="utf-8") == serialized:
                return
            raise FileExistsError(f"write-once checkpoint differs from existing artifact: {path}")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _null_slot_key(stage: str, payload: dict[str, Any]) -> str:
    if stage == "label_shuffle":
        item = payload["item"]
        identity = {"dataset": item["dataset"], "architecture": item["architecture"],
                    "objective": item["objective"], "population": item["population"],
                    "trial_id": item["trial_id"], "training_seed": item["seed"],
                    "representation": payload["representation"]}
    elif stage == "trial_pairing":
        item_a, item_b = payload["item_a"], payload["item_b"]
        identity = {"dataset": item_a["dataset"], "architecture": item_a["architecture"],
                    "objective": item_a["objective"], "training_seed": item_a["seed"],
                    "population_a": item_a["population"], "population_b": item_b["population"]}
    else:
        raise ValueError(f"unknown null checkpoint stage: {stage}")
    return _canonical_json(identity)


def _null_checkpoint_path(checkpoint_root: Path, stage: str, slot_key: str) -> Path:
    slot_hash = hashlib.sha256(slot_key.encode("utf-8")).hexdigest()
    return checkpoint_root / stage / f"{slot_hash}.json"


def _validate_null_checkpoint_result(stage: str, payload: dict[str, Any],
                                     result: dict[str, Any], permutations: int) -> None:
    if stage == "label_shuffle":
        item = payload["item"]
        if (result.get("item", {}).get("trial_id") != item["trial_id"] or
                result.get("representation") != payload["representation"] or
                int(result.get("replicates", -1)) != permutations or
                len(result.get("null_values", [])) != permutations):
            raise ValueError("label-shuffle checkpoint is incomplete or mismatched")
    elif stage == "trial_pairing":
        item_a, item_b = payload["item_a"], payload["item_b"]
        values = result.get("null_values", {})
        if (result.get("item_a", {}).get("trial_id") != item_a["trial_id"] or
                result.get("item_b", {}).get("trial_id") != item_b["trial_id"] or
                int(result.get("permutations", -1)) != permutations):
            raise ValueError("trial-pairing checkpoint is incomplete or mismatched")
        expected_metrics = {"procrustes_r2", "rsa_spearman", "rsa_pearson", "linear_cka"}
        if set(values) != {"raw", "unit"} or any(
                set(values[rep]) != expected_metrics or
                any(len(values[rep][metric]) != permutations for metric in expected_metrics)
                for rep in ("raw", "unit")):
            raise ValueError("trial-pairing checkpoint does not contain every metric/replicate")
    else:
        raise ValueError(f"unknown null checkpoint stage: {stage}")


def _save_null_checkpoint(checkpoint_root: Path, *, stage: str, slot_key: str,
                          campaign_hash: str, result: dict[str, Any]) -> Path:
    path = _null_checkpoint_path(checkpoint_root, stage, slot_key)
    payload = {"checkpoint_version": 1, "stage": stage, "slot_key": slot_key,
               "campaign_manifest_sha256": campaign_hash,
               "result_sha256": _json_hash(result), "result": result}
    _atomic_json_write_once(path, payload)
    return path


def _read_null_checkpoint(path: Path, *, stage: str, slot_key: str,
                          campaign_hash: str, payload: dict[str, Any],
                          permutations: int) -> dict[str, Any]:
    checkpoint = json_read(path)
    if (checkpoint.get("checkpoint_version") != 1 or checkpoint.get("stage") != stage or
            checkpoint.get("slot_key") != slot_key or
            checkpoint.get("campaign_manifest_sha256") != campaign_hash):
        raise ValueError(f"checkpoint provenance mismatch: {path}")
    result = checkpoint.get("result")
    if not isinstance(result, dict) or checkpoint.get("result_sha256") != _json_hash(result):
        raise ValueError(f"checkpoint content hash mismatch: {path}")
    _validate_null_checkpoint_result(stage, payload, result, permutations)
    return result


def _load_null_checkpointed_slots(checkpoint_root: Path, *, stage: str,
                                  payloads: list[dict[str, Any]], campaign_hash: str,
                                  permutations: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cached, missing = [], []
    for payload in payloads:
        slot_key = _null_slot_key(stage, payload)
        path = _null_checkpoint_path(checkpoint_root, stage, slot_key)
        if path.exists():
            cached.append(_read_null_checkpoint(path, stage=stage, slot_key=slot_key,
                campaign_hash=campaign_hash, payload=payload, permutations=permutations))
        else:
            missing.append(payload)
    return cached, missing


def _run_checkpointed_null_slots(payloads: list[dict[str, Any]], *, stage: str,
                                 worker: Any, checkpoint_root: Path, campaign_hash: str,
                                 permutations: int, workers: int, progress_label: str
                                 ) -> tuple[list[dict[str, Any]], int, list[Path]]:
    cached, pending = _load_null_checkpointed_slots(checkpoint_root, stage=stage,
        payloads=payloads, campaign_hash=campaign_hash, permutations=permutations)
    results = list(cached)
    files = [_null_checkpoint_path(checkpoint_root, stage, _null_slot_key(stage, payload))
             for payload in payloads if _null_checkpoint_path(checkpoint_root, stage,
                 _null_slot_key(stage, payload)).exists()]
    total = len(payloads)
    print(f"{progress_label}: reused {len(cached)} checkpointed; pending {len(pending)}/{total}", flush=True)

    def commit(payload: dict[str, Any], result: dict[str, Any]) -> None:
        slot_key = _null_slot_key(stage, payload)
        _validate_null_checkpoint_result(stage, payload, result, permutations)
        path = _save_null_checkpoint(checkpoint_root, stage=stage, slot_key=slot_key,
                                     campaign_hash=campaign_hash, result=result)
        results.append(result)
        files.append(path)
        completed = len(results)
        if completed % 15 == 0 or completed == total:
            print(f"{progress_label}: completed {completed}/{total} slots", flush=True)

    if workers == 1:
        for payload in pending:
            commit(payload, worker(payload))
    elif pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            future_payloads = {pool.submit(worker, payload): payload for payload in pending}
            for future in concurrent.futures.as_completed(future_payloads):
                commit(future_payloads[future], future.result())
    return results, len(cached), sorted(set(files))


def _null_parent_hashes(ctx: dict[str, Any]) -> dict[str, str]:
    return {"core_provenance": sha256(CORE / "PROVENANCE.json"),
            "core_metrics": sha256(CORE / "CORE_METRICS_LONG.csv"),
            "embedding_index": sha256(CORE / "EMBEDDING_INDEX.csv"),
            "evaluation_support": sha256(CORE / "EVALUATION_SUPPORT_MANIFEST.json"),
            "analysis_driver_source": sha256(Path(__file__).resolve()),
            "core_metric_source": sha256(PROJECT / "tools" / "final_thesis_core_metrics.py"),
            "representation_metric_source": sha256(PROJECT / "src" / "neurobridge" / "eval" / "representation.py"),
            "trial_pairing_null_source": sha256(PROJECT / "src" / "neurobridge" / "experiments" / "lag_shuffle.py")}


def _prepare_null_checkpoint_campaign(outdir: Path, *, parent_hashes: dict[str, str],
                                      settings: dict[str, Any], label_slot_keys: list[str],
                                      pair_slot_keys: list[str]) -> tuple[Path, str, Path]:
    if len(label_slot_keys) != len(set(label_slot_keys)) or len(pair_slot_keys) != len(set(pair_slot_keys)):
        raise ValueError("null checkpoint campaign contains duplicate slot identities")
    if outdir.exists():
        unexpected = [child.name for child in outdir.iterdir() if child.name != "_checkpoints"]
        if unexpected:
            raise FileExistsError(f"null-controls branch already has published artifacts; preserve it: {outdir}")
    checkpoint_root = outdir / "_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    body = {"checkpoint_campaign_version": 1, "parent_hashes": parent_hashes,
            "settings": settings, "expected_label_slots": label_slot_keys,
            "expected_trial_pairing_slots": pair_slot_keys}
    campaign_hash = _json_hash(body)
    manifest = {**body, "campaign_manifest_sha256": campaign_hash}
    manifest_path = checkpoint_root / "CAMPAIGN_MANIFEST.json"
    _atomic_json_write_once(manifest_path, manifest)
    if json_read(manifest_path) != manifest:
        raise ValueError("null checkpoint campaign manifest differs from requested campaign")
    return checkpoint_root, campaign_hash, manifest_path


def _normalize_index_row(row: dict[str, str]) -> dict[str, Any]:
    item: dict[str, Any] = dict(row)
    item["dataset"] = "Synthetic" if row["dataset"].lower() == "synthetic" else "Real"
    for key in ("seed", "candidate_index"):
        try:
            item[key] = int(float(row[key]))
        except (KeyError, ValueError):
            item[key] = None
    item["near_collapse"] = str(row.get("near_collapse", "False")).lower() == "true"
    return item


def load_core_context() -> dict[str, Any]:
    """Verify and load the frozen parent branch without modifying it."""
    provenance = json_read(CORE / "PROVENANCE.json")
    audit_path = CORE / "AUDIT_MANIFEST.json"
    support_path = CORE / "EVALUATION_SUPPORT_MANIFEST.json"
    require = core.require
    require(sha256(audit_path) == provenance["audit_manifest_sha256"], "core audit hash differs from provenance")
    require(sha256(support_path) == provenance["evaluation_support_manifest_sha256"],
            "core support hash differs from provenance")
    for rel in ("CORE_METRICS_LONG.csv", "EMBEDDING_INDEX.csv", "DOWNSTREAM_PROBE_SELECTION.csv"):
        expected = provenance["output_artifacts"].get(str((CORE / rel).relative_to(PROJECT)))
        require(expected is not None and sha256(CORE / rel) == expected, f"core parent artifact hash mismatch: {rel}")
    # The core provenance predates a later issue-log append. Keep that history
    # visible, but do not alter the immutable core branch or any metric table.
    issue_log_rel = str((CORE / "AUDIT_ISSUES_AND_FIXES.md").relative_to(PROJECT))
    stale_issue_log_hash = provenance["output_artifacts"].get(issue_log_rel) != sha256(CORE / "AUDIT_ISSUES_AND_FIXES.md")
    index = [_normalize_index_row(row) for row in csv_read(CORE / "EMBEDDING_INDEX.csv")]
    require(len(index) == 122, f"expected 122 indexed embeddings, found {len(index)}")
    seen = set()
    for item in index:
        require(item["trial_id"] not in seen, f"duplicate embedding slot {item['trial_id']}")
        seen.add(item["trial_id"])
        manifest_path = Path(item["manifest_path"])
        require(manifest_path.is_file() and sha256(manifest_path) == item["manifest_sha256"],
                f"embedding manifest mismatch: {item['trial_id']}")
        if item["dataset"] == "Synthetic":
            embed_path = Path(item["embedding_path"])
            require(embed_path.is_file() and sha256(embed_path) == item["embedding_sha256"],
                    f"Synthetic embedding mismatch: {item['trial_id']}")
        else:
            folder = Path(item["embedding_path"])
            manifest = json_read(manifest_path)
            for filename, digest in manifest["artifact_sha256"].items():
                require(sha256(folder / filename) == digest,
                        f"Real embedding payload mismatch: {item['trial_id']}:{filename}")
    metrics = csv_read(CORE / "CORE_METRICS_LONG.csv")
    support = json_read(support_path)
    selected = csv_read(CORE / "DOWNSTREAM_PROBE_SELECTION.csv")
    return {"provenance": provenance, "stale_issue_log_hash": stale_issue_log_hash,
            "index": index, "metrics": metrics, "support": support,
            "probe_selection": selected}


def load_item_arrays(item: dict[str, Any]) -> dict[str, np.ndarray]:
    if item["dataset"] == "Synthetic":
        return core._synthetic_embedding(item)
    return core._real_embedding(item)


def coordinate_lookup(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[tuple[int, int], int]:
    return {(int(data["trial_id"][i]), int(data["time_id"][i])): int(i) for i in indices}


def paired_vectorized_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Match core metrics while vectorizing its fixed 100k RSA pair sample."""
    # The core metric input contract casts embeddings to float64 before
    # computing pair distances; mirror that even when an artifact is float32.
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("paired representations must have the same N x D shape")
    n = len(a)
    out = {"procrustes_r2": procrustes_r2(a, b), "linear_cka": linear_cka(a, b)}
    left, right = _rsa_pair_indices(n)
    if left is None:
        from scipy.spatial.distance import pdist, squareform
        da, db = pdist(a, metric="euclidean"), pdist(b, metric="euclidean")
        xa, xb = da, db
    else:
        # Match the frozen core's sampled Euclidean distance arithmetic exactly.
        # `np.linalg.norm` is mathematically equivalent but can perturb nearly
        # tied distances by a few ulps and thereby change Spearman ranks.
        xa = _sampled_pair_distances(a, left, right, "euclidean")
        xb = _sampled_pair_distances(b, left, right, "euclidean")
    out["rsa_spearman"] = float(spearmanr(xa, xb).statistic)
    out["rsa_pearson"] = float(pearsonr(xa, xb).statistic)
    return out


@lru_cache(maxsize=8)
def _rsa_pair_indices(n: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Cache the exact deterministic random pair sample used by core RSA."""
    n_pairs = n * (n - 1) // 2
    if n_pairs <= PAIR_COUNT:
        return None, None
    # The frozen A/B RSA rows use distance_geometry_correlation's default
    # random_state=0 (not the separate seed used by representation diagnostics).
    # Match its endpoint draws exactly, including dropping self-pairs.
    rng = np.random.default_rng(0)
    left = rng.integers(0, n, size=PAIR_COUNT)
    right = rng.integers(0, n, size=PAIR_COUNT)
    keep = left != right
    return left[keep], right[keep]


def _core_metric_value(rows: list[dict[str, str]], *, dataset: str, arch: str,
                       objective: str, population: str, seed: int | None,
                       representation: str, category: str, metric: str,
                       reference: str | None = None) -> float:
    matches = [r for r in rows if r["dataset"].lower() == dataset.lower()
               and r["architecture"] == arch and r["objective"] == objective
               and r["population"] == population
               and (seed is None or int(float(r["seed"])) == seed)
               and r["representation"] == representation and r["category"] == category
               and r["metric"] == metric
               and (reference is None or r["reference"] == reference)]
    if len(matches) != 1:
        raise ValueError(f"expected one core metric row, found {len(matches)} for "
                         f"{dataset}/{arch}/{objective}/{population}/{seed}/{representation}/{metric}/{reference}")
    return float(matches[0]["value"])


def _artifact_hashes(paths: list[Path], root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256(path) for path in paths}


def _finalize_provenance(outdir: Path, *, branch: str, parent_hashes: dict[str, str],
                         settings: dict[str, Any], output_files: list[Path], extra: dict[str, Any] | None = None) -> Path:
    path = outdir / ("NULL_PROVENANCE.json" if branch == "null_controls" else
                     "UNCERTAINTY_PROVENANCE.json" if branch == "uncertainty" else "REAL_LAG_PROVENANCE.json")
    payload = {"provenance_version": "neurobridge_final_downstream_v1", "branch": branch,
               "parent_hashes": parent_hashes, "settings": settings,
               "no_encoder_training_or_hpo": True,
               "artifacts": _artifact_hashes(output_files, PROJECT)}
    if extra:
        payload.update(extra)
    write_once(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _plot_save(fig: plt.Figure, path_no_suffix: Path) -> list[Path]:
    png, pdf = path_no_suffix.with_suffix(".png"), path_no_suffix.with_suffix(".pdf")
    for path in (png, pdf):
        if path.exists():
            raise FileExistsError(f"write-once figure exists: {path}")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def run_controlled_lag(ctx: dict[str, Any]) -> dict[str, Any]:
    from final_real_embeddings import (  # local project helper; read-only model/data access
        WINDOW, TRIAL_LENGTH, centered_batches, channel_indices, read_json as real_json_read,
        stage1_arrays, valid_centers,
    )
    from neurobridge.experiments.real_monkey_lag_eval import _curve_summary, _lag_curves_on_joint_support

    outdir = EVAL / "controlled_lag"
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"controlled lag branch is not empty; preserve it and use a new version: {outdir}")
    spec_path = FINAL / "freeze" / "REAL_FINAL_EXPERIMENT_SPEC.json"
    spec = real_json_read(spec_path)
    part_path = Path(spec["somatotopic_partition_path"])
    if not part_path.is_absolute():
        part_path = PROJECT / part_path
    partition = real_json_read(part_path)
    arrays, split, source_hashes = stage1_arrays(PROJECT, spec)
    spikes = arrays["spikes"].reshape(-1, TRIAL_LENGTH, 65)
    valid_bins = arrays["valid_bins"].reshape(-1, TRIAL_LENGTH).astype(bool)
    shift = 10
    shifted = np.zeros_like(spikes)
    shifted[:, shift:, :] = spikes[:, :-shift, :]
    shifted_valid_bins = valid_bins.copy()
    shifted_valid_bins[:, :shift] = False
    if not np.array_equal(shifted[:, shift:, :], spikes[:, :-shift, :]) or np.any(shifted[:, :shift, :] != 0):
        raise AssertionError("R10 input shift is not a no-wrap +10-bin copy")
    test_trials = np.asarray(split["test"], dtype=np.int64)
    b_indices = channel_indices(partition, "B_DISTAL")
    radius = WINDOW // 2
    input_windows_checked = 0
    for trial in test_trials:
        # Index time first so NumPy's advanced channel index does not move the
        # channel axis ahead of the temporal axis.
        original_trial = spikes[int(trial)][:, b_indices]
        shifted_trial = shifted[int(trial)][:, b_indices]
        source_centers = np.flatnonzero(valid_centers(valid_bins[int(trial)]))
        shifted_center_mask = valid_centers(shifted_valid_bins[int(trial)])
        target_centers = np.flatnonzero(shifted_center_mask)
        target_by_source = target_centers - shift
        keep = np.isin(source_centers, target_by_source)
        source_centers = source_centers[keep]
        target_centers = source_centers + shift
        if not np.array_equal(target_centers, np.flatnonzero(valid_centers(shifted_valid_bins[int(trial)]))):
            raise AssertionError("R0/R10 valid center mapping has a gap or wrap")
        padded_original = np.pad(original_trial, ((radius, radius), (0, 0)), mode="constant")
        padded_shifted = np.pad(shifted_trial, ((radius, radius), (0, 0)), mode="constant")
        for start in range(0, len(source_centers), 64):
            source = source_centers[start:start + 64]
            target = source + shift
            win0 = np.stack([padded_original[t:t + WINDOW] for t in source])
            win10 = np.stack([padded_shifted[t:t + WINDOW] for t in target])
            if not np.array_equal(win0, win10):
                raise AssertionError(f"R10 B window does not equal frozen R0 B window shifted by 10: trial={trial}")
            input_windows_checked += len(source)

    # The encoders map each centered window independently. Therefore exact
    # input-window identity implies exact output identity under the same frozen
    # encoder. Build R10 by a within-trial, no-wrap shift of the R0 embeddings.
    real_items = [item for item in ctx["index"] if item["dataset"] == "Real"]
    real_by_key = {(i["architecture"], i["objective"], int(i["seed"]), i["population"]): i for i in real_items}
    curves, summaries, plotted = [], [], defaultdict(list)
    embed_parent_hashes = {}
    completed = 0
    total = 2 * 4 * len(SEEDS)
    for arch in sorted({i["architecture"] for i in real_items}):
        for objective in sorted({i["objective"] for i in real_items}):
            for seed in SEEDS:
                a_item = real_by_key[(arch, objective, seed, "A_PROXIMAL")]
                b_item = real_by_key[(arch, objective, seed, "B_DISTAL")]
                a = load_item_arrays(a_item)
                b0 = load_item_arrays(b_item)
                b10 = {key: value.copy() for key, value in b0.items()}
                b10["embedding_unit"] = np.zeros_like(b0["embedding_unit"])
                b10["valid_mask"] = np.zeros_like(b0["valid_mask"], dtype=bool)
                b_lookup = coordinate_lookup(b0, np.arange(len(b0["trial_id"])))
                b10_lookup = coordinate_lookup(b10, np.arange(len(b10["trial_id"])))
                for trial in test_trials:
                    shifted_center_mask = valid_centers(shifted_valid_bins[int(trial)])
                    for t in range(shift, TRIAL_LENGTH):
                        dest = b10_lookup.get((int(trial), t))
                        src = b_lookup.get((int(trial), t - shift))
                        if dest is not None and src is not None:
                            b10["embedding_unit"][dest] = b0["embedding_unit"][src]
                            b10["valid_mask"][dest] = bool(b0["valid_mask"][src]) and bool(shifted_center_mask[t])
                valid_a = a["valid_mask"].astype(bool)
                valid_b0 = b0["valid_mask"].astype(bool)
                valid_b10 = b10["valid_mask"].astype(bool)
                # Input/output equivalence check across every held-out center
                # used by the intervention; no new fit or test-selected choice.
                checked_embeddings = 0
                for trial in test_trials:
                    for t in range(shift, TRIAL_LENGTH):
                        dest = b10_lookup.get((int(trial), t))
                        src = b_lookup.get((int(trial), t - shift))
                        if dest is not None and src is not None and valid_b10[dest]:
                            if not np.array_equal(b10["embedding_unit"][dest], b0["embedding_unit"][src]):
                                raise AssertionError("R10 B embedding mapping differs from frozen R0 encoder output")
                            checked_embeddings += 1
                if checked_embeddings != input_windows_checked:
                    raise AssertionError(f"R10 embedding/input support differs: {checked_embeddings} vs {input_windows_checked}")
                curve_r0, curve_r10, n_common = _lag_curves_on_joint_support(
                    a, b0, b10, valid_a, valid_b0, valid_b10, branch="held_out")
                summaries_curves = []
                for setting, curve in (("R0", curve_r0), ("R10_B_shifted_plus10", curve_r10)):
                    curve_summary = _curve_summary(curve)
                    vals = np.asarray([curve[int(lag)] for lag in LAGS], dtype=float)
                    maximum = float(np.nanmax(vals))
                    ties = int(np.sum(np.isclose(vals, maximum, rtol=0, atol=1e-12)))
                    best_lag = int(curve_summary["best_lag_bins"])
                    interior = best_lag not in (min(LAGS), max(LAGS))
                    identifiable = bool(interior and ties == 1)
                    summaries_curves.append((best_lag, curve_summary, identifiable, ties))
                    for lag in LAGS:
                        curves.append({"architecture": arch, "objective": objective, "training_seed": seed,
                                       "population_pair": "A_PROXIMAL_vs_B_DISTAL", "setting": setting,
                                       "lag_bins": int(lag), "s_lag_r2": float(curve[int(lag)]),
                                       "n_comparisons": int(n_common), "representation": "unit"})
                    plotted[(arch, objective)].append((setting, seed, curve))
                (l0, s0, id0, ties0), (l10, s10, id10, ties10) = summaries_curves
                identifiable_delta = bool(id0 and id10)
                delta = l10 - l0 if identifiable_delta else math.nan
                summaries.append({"architecture": arch, "objective": objective, "training_seed": seed,
                    "population_pair": "A_PROXIMAL_vs_B_DISTAL", "representation": "unit",
                    "n_common_reference_rows_per_lag": int(n_common),
                    "l_hat_r0_bins": l0, "s_max_r0": s0["best_procrustes_r2"],
                    "peak_margin_r0": s0["peak_margin_r2"], "boundary_r0": s0["boundary_peak"],
                    "peak_ties_r0": ties0, "identifiable_r0": id0,
                    "l_hat_r10_bins": l10, "s_max_r10": s10["best_procrustes_r2"],
                    "peak_margin_r10": s10["peak_margin_r2"], "boundary_r10": s10["boundary_peak"],
                    "peak_ties_r10": ties10, "identifiable_r10": id10,
                    "delta_l_hat_bins": delta, "expected_controlled_delta_bins": 10,
                    "delta_vs_plus10_absolute_error_bins": abs(delta - 10) if identifiable_delta else math.nan,
                    "delta_identifiable": identifiable_delta,
                    "interpretation": "relative temporal correspondence; R10 response to digital intervention, not causal delay"})
                embed_parent_hashes[f"{arch}/{objective}/{seed}/A_manifest"] = a_item["manifest_sha256"]
                embed_parent_hashes[f"{arch}/{objective}/{seed}/B_manifest"] = b_item["manifest_sha256"]
                completed += 1
                print(f"controlled lag: completed {completed}/{total} architecture-objective-seed pairs",
                      flush=True)

    outdir.mkdir(parents=True, exist_ok=True)
    curves_path = outdir / "REAL_LAG_CURVES_R0_R10.csv"
    summary_path = outdir / "REAL_LAG_SUMMARY.csv"
    fields_curves = ["architecture", "objective", "training_seed", "population_pair", "setting",
                     "lag_bins", "s_lag_r2", "n_comparisons", "representation"]
    fields_summary = list(summaries[0])
    csv_write_once(curves_path, curves, fields_curves)
    csv_write_once(summary_path, summaries, fields_summary)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    for ax, (arch, objective) in zip(axes.flat, sorted(plotted)):
        for setting, seed, curve in plotted[(arch, objective)]:
            scores = [curve[int(lag)] for lag in LAGS]
            color = "#1769aa" if setting == "R0" else "#d55e00"
            ls = "-" if setting == "R0" else "--"
            ax.plot(LAGS, scores, color=color, linestyle=ls, alpha=.28, linewidth=1)
            summ = next(row for row in summaries if row["architecture"] == arch and row["objective"] == objective
                        and row["training_seed"] == seed)
            peak = summ["l_hat_r0_bins"] if setting == "R0" else summ["l_hat_r10_bins"]
            ax.axvline(peak, color=color, alpha=.24, linewidth=.8)
        ax.set_title(f"{arch} / {objective}")
        ax.grid(alpha=.2)
    fig.supxlabel("Candidate lag (ms; B(t + lag) vs A(t))")
    fig.supylabel("Procrustes R²")
    fig.suptitle("Real held-out temporal correspondence: R0 and B-only +10-bin intervention\n"
                 "thin lines are fixed training seeds; peak lines use same colors")
    fig.legend(handles=[plt.Line2D([0], [0], color="#1769aa", label="R0"),
                        plt.Line2D([0], [0], color="#d55e00", linestyle="--", label="R10 B shifted")],
               loc="upper center", ncol=2, bbox_to_anchor=(.5, .94))
    figure_files = _plot_save(fig, figdir / "real_lag_curves_r0_r10")
    identifiable = [row for row in summaries if row["delta_identifiable"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    if identifiable:
        x = np.arange(len(identifiable))
        y = [row["delta_l_hat_bins"] for row in identifiable]
        labels = [f"{r['architecture']}\n{r['objective']}\nseed {r['training_seed']}" for r in identifiable]
        ax.scatter(x, y, color="#1769aa", s=32)
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
        ax.axhline(10, color="#d55e00", linestyle="--", label="known digital shift: +10 ms")
        ax.legend()
    else:
        ax.text(.5, .5, "No pair has both unique interior peaks.\nDelta is unresolved/censored.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
    ax.set_ylabel("Δ l-hat (R10 − R0), ms")
    ax.set_title("Controlled lag response; only jointly identifiable pairs shown")
    ax.grid(axis="y", alpha=.2)
    figure_files += _plot_save(fig, figdir / "real_lag_delta_identifiable")
    parent_hashes = {"core_provenance": sha256(CORE / "PROVENANCE.json"),
                     "core_metrics": sha256(CORE / "CORE_METRICS_LONG.csv"),
                     "evaluation_support": sha256(CORE / "EVALUATION_SUPPORT_MANIFEST.json"),
                     "final_spec": sha256(spec_path), "partition": sha256(part_path),
                     "stage1_data": source_hashes["data_npz_sha256"],
                     "stage1_split": source_hashes["split_json_sha256"]}
    files = [curves_path, summary_path, *figure_files]
    provenance_path = _finalize_provenance(outdir, branch="controlled_lag", parent_hashes=parent_hashes,
        settings={"candidate_lags_bins": list(LAGS), "positive_lag": "compare A(trial,t) with B(trial,t+lag)",
                  "shift": "B_R10(t+10)=B_R0(t), no wrap, first 10 bins invalid",
                  "held_out_only": True, "representation": "unit-normalized frozen embeddings",
                  "window_size": WINDOW, "sampling_ms": 1, "training_seeds": list(SEEDS),
                  "common_support_independent_of_lag": True}, output_files=files,
        extra={"input_windows_exact_matches": int(input_windows_checked),
               "embedding_rows_exactly_remapped": int(sum(r["n_common_reference_rows_per_lag"] for r in summaries)),
               "embedding_parent_manifests": embed_parent_hashes,
               "all_delta_claims_require_both_peaks_unique_and_interior": True,
               "causal_or_synaptic_delay_claim": False})
    return {"summary_rows": len(summaries), "curve_rows": len(curves),
            "input_windows_checked": input_windows_checked, "output": str(outdir),
            "provenance": str(provenance_path)}


def _trial_label_permutation(data: dict[str, np.ndarray], labels: np.ndarray, seed: int) -> np.ndarray:
    """Permute trial-level labels within each frozen split, retaining each trial's rows."""
    trial = np.asarray(data["trial_id"], dtype=np.int64)
    split = np.asarray(data["split"]).astype(str)
    labels = np.asarray(labels)
    trial_labels: dict[int, Any] = {}
    for tid in np.unique(trial):
        values = np.unique(labels[trial == tid])
        if len(values) != 1:
            raise ValueError(f"label is not trial-level for trial {tid}")
        trial_labels[int(tid)] = values[0]
    out = np.empty_like(labels)
    rng = np.random.default_rng(seed)
    for split_name in ("train", "validation", "test"):
        ids = np.unique(trial[split == split_name])
        permuted_source_trials = rng.permutation(ids)
        for target_trial, source_trial in zip(ids, permuted_source_trials):
            out[trial == target_trial] = trial_labels[int(source_trial)]
    return out


def _label_null_slot(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker: validation-selected multinomial probe under trial label shuffles."""
    item = payload["item"]
    representation = payload["representation"]
    seeds = payload["permutation_seeds"]
    data = load_item_arrays(item)
    labels = data["labels"] if item["dataset"] == "Synthetic" else data["target"].astype(int)
    masks = core._split_masks(data, item["dataset"], item["population"])
    x = data[f"embedding_{representation}"]
    values = []
    metric = "condition_balanced_accuracy" if item["dataset"] == "Synthetic" else "direction_balanced_accuracy"
    with threadpool_limits(limits=1):
        for seed in seeds:
            permuted_labels = _trial_label_permutation(data, labels, int(seed))
            rows, _ = core._probe_metrics(x, permuted_labels, masks,
                seed=(int(item["seed"]) if item["seed"] is not None else 0),
                behavior={} if item["dataset"] == "Real" else None)
            match = next(row for row in rows if row["metric"] == metric)
            values.append(float(match["value"]))
    observed = float(payload["observed"])
    values_arr = np.asarray(values, dtype=float)
    return {"item": item, "representation": representation, "metric": metric,
            "observed": observed, "null_values": values_arr.tolist(),
            "null_mean": float(values_arr.mean()), "null_sd": float(values_arr.std(ddof=1)) if len(values_arr) > 1 else 0.0,
            "empirical_p": float((1 + np.sum(values_arr >= observed)) / (len(values_arr) + 1)),
            "percentile": float(100 * (np.sum(values_arr < observed) + .5 * np.sum(values_arr == observed)) / len(values_arr)),
            "effect": float(observed - values_arr.mean()), "replicates": int(len(values_arr))}


def _trial_pair_samples(ctx: dict[str, Any], item_a: dict[str, Any], item_b: dict[str, Any],
                        data_a: dict[str, np.ndarray], data_b: dict[str, np.ndarray],
                        representation: str, trial_map: dict[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    dataset = item_a["dataset"]
    if dataset == "Synthetic":
        coordinates = ctx["support"]["synthetic"]["lag_common_support"]["coordinates_trial_time"]
        lag = 10
    else:
        coordinates = ctx["support"]["real"]["coordinates_trial_time"]
        lag = 0
    ia = coordinate_lookup(data_a, np.arange(len(data_a["trial_id"])))
    ib = coordinate_lookup(data_b, np.arange(len(data_b["trial_id"])))
    idx_a, idx_b = [], []
    for trial, time in coordinates:
        trial, time = int(trial), int(time)
        paired_trial = trial_map.get(trial, trial) if trial_map is not None else trial
        key_a, key_b = (trial, time), (int(paired_trial), time + lag)
        if key_a not in ia or key_b not in ib:
            raise ValueError(f"paired support coordinate missing: A={key_a} B={key_b}")
        if not bool(data_a["valid_mask"][ia[key_a]]) or not bool(data_b["valid_mask"][ib[key_b]]):
            raise ValueError(f"paired support uses invalid embedding coordinate: {key_a} -> {key_b}")
        if str(data_a["split"][ia[key_a]]) != "test" or str(data_b["split"][ib[key_b]]) != "test":
            raise ValueError("cross-population null attempted to use non-test rows")
        idx_a.append(ia[key_a])
        idx_b.append(ib[key_b])
    return data_a[f"embedding_{representation}"][idx_a], data_b[f"embedding_{representation}"][idx_b]


def _pair_null_group(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker for all trial-pairing replicates of one frozen A/B model pair."""
    item_a, item_b = payload["item_a"], payload["item_b"]
    dataset = item_a["dataset"]
    data_a, data_b = load_item_arrays(item_a), load_item_arrays(item_b)
    support = json_read(CORE / "EVALUATION_SUPPORT_MANIFEST.json")
    coords = support["synthetic"]["lag_common_support"]["coordinates_trial_time"] if dataset == "Synthetic" else support["real"]["coordinates_trial_time"]
    coords = np.asarray(coords, dtype=np.int64)
    trials, times = coords[:, 0], coords[:, 1]
    lag = 10 if dataset == "Synthetic" else 0
    n_trials = max(int(data_a["trial_id"].max()), int(data_b["trial_id"].max())) + 1
    n_times = max(int(data_a["time_id"].max()), int(data_b["time_id"].max())) + 1
    grid_a = np.full((n_trials, n_times), -1, dtype=np.int64)
    grid_b = np.full_like(grid_a, -1)
    grid_a[data_a["trial_id"].astype(int), data_a["time_id"].astype(int)] = np.arange(len(data_a["trial_id"]))
    grid_b[data_b["trial_id"].astype(int), data_b["time_id"].astype(int)] = np.arange(len(data_b["trial_id"]))
    ia = grid_a[trials, times]
    ib_observed = grid_b[trials, times + lag]
    if np.any(ia < 0) or np.any(ib_observed < 0):
        raise ValueError("frozen A/B support contains coordinates absent from the embedding index")
    if (np.any(data_a["split"][ia].astype(str) != "test") or
            np.any(data_b["split"][ib_observed].astype(str) != "test") or
            not np.all(data_a["valid_mask"][ia].astype(bool)) or
            not np.all(data_b["valid_mask"][ib_observed].astype(bool))):
        raise ValueError("pairing null support is not test-only")
    unique_trials = np.asarray(sorted(np.unique(trials).tolist()), dtype=np.int64)
    seed_start = int(payload.get("seed_start", 810000 if dataset == "Synthetic" else 820000))
    results = {"raw": defaultdict(list), "unit": defaultdict(list)}
    with threadpool_limits(limits=1):
        for rep in range(payload["permutations"]):
            shuffled = shuffled_trial_ids(unique_trials, np.random.default_rng(seed_start + rep))
            mapped_trial = np.arange(n_trials, dtype=np.int64)
            mapped_trial[unique_trials] = shuffled
            ib = grid_b[mapped_trial[trials], times + lag]
            if np.any(ib < 0) or np.any(data_b["split"][ib].astype(str) != "test"):
                raise ValueError("trial-deranged pairing uses an absent/non-test B row")
            for representation in ("raw", "unit"):
                x = data_a[f"embedding_{representation}"][ia]
                y = data_b[f"embedding_{representation}"][ib]
                for metric, value in paired_vectorized_metrics(x, y).items():
                    results[representation][metric].append(float(value))
    return {"item_a": item_a, "item_b": item_b,
            "null_values": {rep: {metric: values for metric, values in metric_rows.items()}
                            for rep, metric_rows in results.items()},
            "seed_start": seed_start, "permutations": payload["permutations"],
            "n_matched_rows": len(coords)}


def run_null_controls(ctx: dict[str, Any], *, permutations: int, workers: int) -> dict[str, Any]:
    if permutations < 2:
        raise ValueError("at least two permutations are required")
    outdir = EVAL / "null_controls"
    items = ctx["index"]
    label_seeds = [730000 + i for i in range(permutations)]
    pair_seeds = {
        "Synthetic": [810000 + i for i in range(permutations)],
        "Real": [820000 + i for i in range(permutations)],
    }
    label_payloads = []
    for item in items:
        metric = "condition_balanced_accuracy" if item["dataset"] == "Synthetic" else "direction_balanced_accuracy"
        for rep in ("raw", "unit"):
            label_payloads.append({"item": item, "representation": rep,
                "permutation_seeds": label_seeds,
                "observed": _core_metric_value(ctx["metrics"], dataset=item["dataset"], arch=item["architecture"],
                    objective=item["objective"], population=item["population"], seed=item["seed"],
                    representation=rep, category="accessibility", metric=metric)})
    # One seeded trial derangement per dataset/replicate is regenerated inside
    # every worker, so all model/objective/seed slots use the same mapping.
    pair_index = {(item["dataset"], item["architecture"], item["objective"], item["seed"], item["population"]): item
                  for item in items if item["dataset"] != "Synthetic" or item["architecture"] != "pca"}
    pair_payloads = []
    for dataset, pop_a, pop_b, metric_category, reference in (
        ("Synthetic", "A", "B", "cross_population_correspondence", "known_true_shift_plus10"),
        ("Real", "A_PROXIMAL", "B_DISTAL", "cross_population_consistency", "matched_A_B")):
        for arch in sorted({i["architecture"] for i in items if i["dataset"] == dataset and i["architecture"] != "pca"}):
            for objective in sorted({i["objective"] for i in items if i["dataset"] == dataset and i["architecture"] == arch}):
                for seed in SEEDS:
                    a_item = pair_index[(dataset, arch, objective, seed, pop_a)]
                    b_item = pair_index[(dataset, arch, objective, seed, pop_b)]
                    pair_payloads.append({"item_a": a_item, "item_b": b_item, "permutations": permutations,
                                          "seed_start": pair_seeds[dataset][0]})
    if len(label_payloads) != 244 or len(pair_payloads) != 48:
        raise ValueError(f"frozen null campaign slot count mismatch: label={len(label_payloads)} pair={len(pair_payloads)}")
    parent_hashes = _null_parent_hashes(ctx)
    checkpoint_settings = {"permutations": permutations, "training_seeds": list(SEEDS),
        "label_permutation_seeds": [label_seeds[0], label_seeds[-1]],
        "trial_pairing_seeds": {dataset: [seeds[0], seeds[-1]] for dataset, seeds in pair_seeds.items()},
        "support_hash": parent_hashes["evaluation_support"],
        "embedding_index_hash": parent_hashes["embedding_index"],
        "parent_core_provenance_hash": parent_hashes["core_provenance"],
        "source_hashes": {key: value for key, value in parent_hashes.items()
                           if key.endswith("_source")},
        "label_permutation_unit": "complete-trial label within each frozen split",
        "pairing_permutation_unit": "complete B trial identity derangement; within-trial time preserved"}
    label_slot_keys = [_null_slot_key("label_shuffle", payload) for payload in label_payloads]
    pair_slot_keys = [_null_slot_key("trial_pairing", payload) for payload in pair_payloads]
    checkpoint_root, campaign_hash, checkpoint_manifest_path = _prepare_null_checkpoint_campaign(
        outdir, parent_hashes=parent_hashes, settings=checkpoint_settings,
        label_slot_keys=label_slot_keys, pair_slot_keys=pair_slot_keys)
    label_results, reused_label_slots, label_checkpoint_files = _run_checkpointed_null_slots(
        label_payloads, stage="label_shuffle", worker=_label_null_slot,
        checkpoint_root=checkpoint_root, campaign_hash=campaign_hash,
        permutations=permutations, workers=workers, progress_label="label-shuffle null")
    pair_results, reused_pair_slots, pair_checkpoint_files = _run_checkpointed_null_slots(
        pair_payloads, stage="trial_pairing", worker=_pair_null_group,
        checkpoint_root=checkpoint_root, campaign_hash=campaign_hash,
        permutations=permutations, workers=workers, progress_label="cross-population null")
    label_rows, label_replicates = [], []
    for result in sorted(label_results, key=lambda r: (r["item"]["dataset"], r["item"]["architecture"],
                                                        r["item"]["objective"], str(r["item"]["population"]),
                                                        str(r["item"]["seed"]), r["representation"])):
        item = result["item"]
        base = {"dataset": item["dataset"], "architecture": item["architecture"],
                "objective": item["objective"], "population": item["population"],
                "training_seed": item["seed"], "trial_id": item["trial_id"],
                "representation": result["representation"], "metric": result["metric"],
                "observed": result["observed"], "null_mean": result["null_mean"],
                "null_sd": result["null_sd"], "empirical_p": result["empirical_p"],
                "percentile_within_null": result["percentile"], "observed_minus_null_mean": result["effect"],
                "replicates": result["replicates"], "permutation_seed_start": label_seeds[0],
                "permutation_seed_end": label_seeds[-1], "unit_of_permutation": "complete trial label within split"}
        label_rows.append(base)
        label_replicates.extend({**{k: base[k] for k in ("dataset", "architecture", "objective", "population",
                                "training_seed", "trial_id", "representation", "metric")},
                                "replicate": i, "permutation_seed": label_seeds[i], "null_balanced_accuracy": value}
                                for i, value in enumerate(result["null_values"]))
    pair_results.sort(key=lambda r: (r["item_a"]["dataset"], r["item_a"]["architecture"],
                                     r["item_a"]["objective"], int(r["item_a"]["seed"])))
    pair_rows, pair_replicates = [], []
    for result in pair_results:
        a_item, b_item = result["item_a"], result["item_b"]
        dataset, arch, objective, seed = (a_item["dataset"], a_item["architecture"],
                                          a_item["objective"], int(a_item["seed"]))
        pop_a, pop_b = a_item["population"], b_item["population"]
        metric_category = "cross_population_correspondence" if dataset == "Synthetic" else "cross_population_consistency"
        reference = "known_true_shift_plus10" if dataset == "Synthetic" else "matched_A_B"
        a, b = load_item_arrays(a_item), load_item_arrays(b_item)
        for representation in ("raw", "unit"):
            obs_a, obs_b = _trial_pair_samples(ctx, a_item, b_item, a, b, representation)
            observed = paired_vectorized_metrics(obs_a, obs_b)
            for metric, value in observed.items():
                reference_metric = _core_metric_value(ctx["metrics"], dataset=dataset, arch=arch,
                    objective=objective, population="A_vs_B" if dataset == "Synthetic" else "A_PROXIMAL_vs_B_DISTAL",
                    seed=seed, representation=representation, category=metric_category,
                    metric=metric, reference=reference)
                if not np.isclose(value, reference_metric, rtol=1e-12, atol=1e-12):
                    raise AssertionError(f"observed pairing metric disagrees with frozen core: {dataset}/{arch}/{objective}/{seed}/{representation}/{metric}: {value} vs {reference_metric}")
                arr = np.asarray(result["null_values"][representation][metric], dtype=float)
                p = float((1 + np.sum(arr >= value)) / (len(arr) + 1))
                pct = float(100 * (np.sum(arr < value) + .5 * np.sum(arr == value)) / len(arr))
                seed_start = 810000 if dataset == "Synthetic" else 820000
                base = {"dataset": dataset, "architecture": arch, "objective": objective,
                    "training_seed": seed, "population_pair": f"{pop_a}_vs_{pop_b}",
                    "representation": representation, "metric": metric, "observed": float(value),
                    "null_mean": float(arr.mean()), "null_sd": float(arr.std(ddof=1)) if len(arr)>1 else 0.0,
                    "empirical_p": p, "percentile_within_null": pct,
                    "observed_minus_null_mean": float(value-arr.mean()), "replicates": permutations,
                    "permutation_seed_start": seed_start,
                    "permutation_seed_end": seed_start + permutations - 1,
                    "unit_of_permutation": "deranged complete B trial identity; within-trial time preserved",
                    "n_matched_rows": result["n_matched_rows"], "reference": reference}
                pair_rows.append(base)
                pair_replicates.extend({**{k: base[k] for k in ("dataset", "architecture", "objective",
                    "training_seed", "population_pair", "representation", "metric")},
                    "replicate": i, "permutation_seed": seed_start + i, "null_value": v}
                    for i, v in enumerate(arr.tolist()))

    temporal_rows = [{"dataset": "Synthetic", "analysis": "temporal_order_null", "status": "NOT_COMPUTED",
        "reason": "No frozen temporal-order null definition exists in the repository. Existing lag_shuffle is a complete-trial identity derangement and is used only for the separate cross-population trial-pairing null.",
        "lag_grid_bins": json.dumps(list(LAGS)), "r10_performed": False}]
    summary_rows = []
    for row in label_rows:
        summary_rows.append({"analysis": "accessibility_label_shuffle", **row,
                             "claim_status": "empirical comparison; uncorrected across model family"})
    for row in pair_rows:
        summary_rows.append({"analysis": "cross_population_trial_pairing", **row,
                             "claim_status": "empirical comparison; uncorrected across model family"})
    summary_rows.extend({"analysis": r["analysis"], "dataset": r["dataset"], "metric": r["analysis"],
                         "observed": "", "null_mean": "", "null_sd": "", "empirical_p": "",
                         "observed_minus_null_mean": "", "claim_status": r["status"], "reason": r["reason"]}
                        for r in temporal_rows)

    outdir.mkdir(parents=True, exist_ok=True)
    fields_label = list(label_rows[0]) if label_rows else []
    fields_label_rep = list(label_replicates[0]) if label_replicates else []
    fields_pair = list(pair_rows[0]) if pair_rows else []
    fields_pair_rep = list(pair_replicates[0]) if pair_replicates else []
    paths = [outdir / "NULL_LABEL_SHUFFLE.csv", outdir / "NULL_LABEL_SHUFFLE_REPLICATES.csv",
             outdir / "NULL_AB_TRIAL_PAIRING.csv", outdir / "NULL_AB_TRIAL_PAIRING_REPLICATES.csv",
             outdir / "NULL_TEMPORAL.csv", outdir / "NULL_SUMMARY.csv"]
    csv_write_once(paths[0], label_rows, fields_label)
    csv_write_once(paths[1], label_replicates, fields_label_rep)
    csv_write_once(paths[2], pair_rows, fields_pair)
    csv_write_once(paths[3], pair_replicates, fields_pair_rep)
    csv_write_once(paths[4], temporal_rows, list(temporal_rows[0]))
    csv_write_once(paths[5], summary_rows, list(summary_rows[0]))
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    # Accessibility: observed BA against each configuration's trial-label null.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, dataset in zip(axes, ("Synthetic", "Real")):
        rows = [r for r in label_rows if r["dataset"] == dataset]
        for rep, marker in (("raw", "o"), ("unit", "s")):
            group = [r for r in rows if r["representation"] == rep]
            ax.scatter([r["observed"] for r in group], [r["null_mean"] for r in group],
                       marker=marker, alpha=.58, label=rep, s=24)
            ax.errorbar([r["observed"] for r in group], [r["null_mean"] for r in group],
                        yerr=[r["null_sd"] for r in group], fmt="none", ecolor="#555555", alpha=.12, capsize=0)
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], color="#d55e00", linestyle="--", linewidth=1)
        ax.set_title(dataset)
        ax.set_xlabel("Observed held-out Balanced Accuracy")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Trial-label null mean ± SD")
    axes[1].legend(title="Embedding")
    fig.suptitle(f"Held-out accessibility vs complete-trial label null ({permutations} permutations)")
    figure_files = _plot_save(fig, figdir / "observed_vs_label_shuffle_balanced_accuracy")
    # Pairing null: each metric is shown in its own panel; models are pooled only
    # for the descriptive distribution, while CSV retains every configuration.
    metric_names = ("procrustes_r2", "rsa_spearman", "rsa_pearson", "linear_cka")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for row_ix, dataset in enumerate(("Synthetic", "Real")):
        for col_ix, metric in enumerate(metric_names):
            ax = axes[row_ix, col_ix]
            group = [r for r in pair_rows if r["dataset"] == dataset and r["metric"] == metric]
            null_by_rep = [r["null_value"] for r in pair_replicates if r["dataset"] == dataset and r["metric"] == metric]
            ax.violinplot(null_by_rep, positions=[0], widths=.65, showmeans=True, showextrema=False)
            for rep, marker in (("raw", "o"), ("unit", "s")):
                vals = [r["observed"] for r in group if r["representation"] == rep]
                jitter = np.linspace(-.16, .16, len(vals)) if vals else []
                ax.scatter(np.asarray(jitter) + .55, vals, marker=marker, s=18, alpha=.65, label=rep)
            ax.set_title(f"{dataset}: {metric}")
            ax.set_xticks([0, .55], ["pairing null", "observed"])
            ax.grid(axis="y", alpha=.2)
            if row_ix == 0 and col_ix == 0:
                ax.legend()
    fig.supylabel("Metric value")
    fig.suptitle(f"A/B correspondence vs complete-trial B-identity derangement ({permutations} permutations)")
    figure_files += _plot_save(fig, figdir / "observed_vs_trial_pairing_null")
    # No temporal null is drawn: showing an invented curve would be misleading.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(.5, .62, "Temporal-order null: NOT COMPUTED", ha="center", va="center",
            fontsize=17, weight="bold", transform=ax.transAxes)
    ax.text(.5, .38, "No frozen temporal-order null definition was found.\n"
            "The existing trial-identity derangement is reported separately as an A/B pairing null.",
            ha="center", va="center", fontsize=12, transform=ax.transAxes)
    figure_files += _plot_save(fig, figdir / "temporal_null_definition_status")
    checkpoint_files = [checkpoint_manifest_path, *label_checkpoint_files, *pair_checkpoint_files]
    _finalize_provenance(outdir, branch="null_controls", parent_hashes=parent_hashes,
        settings={"label_shuffle_replicates": permutations, "label_shuffle_seed_start": label_seeds[0],
                  "label_shuffle_seed_end": label_seeds[-1], "trial_pairing_replicates": permutations,
                  "synthetic_pairing_seed_start": pair_seeds["Synthetic"][0],
                  "real_pairing_seed_start": pair_seeds["Real"][0], "seed_mapping_shared_across_model_slots": True,
                  "train_validation_test_preserved": True, "complete_trial_units_only": True,
                  "statistic_direction": "higher values compared against null; p=(1+#null>=observed)/(B+1)",
                  "null_percentile": "midrank percentile, ties receive half weight",
                  "temporal_order_null": "NOT_COMPUTED; no frozen definition found",
                  "optional_random_representation_baseline": "NOT_APPLICABLE; no frozen definition found",
                   "worker_processes": workers}, output_files=[*paths, *figure_files, *checkpoint_files],
        extra={"label_null_trial_class_counts_preserved_within_split": True,
               "pair_null_preserves_trial_internal_time_order": True,
               "checkpoint_campaign_manifest_sha256": campaign_hash,
               "label_checkpoint_slots_reused": reused_label_slots,
               "label_checkpoint_slots_computed": len(label_results) - reused_label_slots,
               "trial_pairing_checkpoint_slots_reused": reused_pair_slots,
               "trial_pairing_checkpoint_slots_computed": len(pair_results) - reused_pair_slots,
               "checkpoint_slot_results_atomic_write_once": True,
               "temporal_null_claim_status": "NOT_COMPUTED",
               "temporal_null_blocker": "No frozen temporal-order null definition exists. Existing lag_shuffle implements the distinct complete-trial A/B pairing derangement and is not repurposed as a temporal-order null.",
               "core_issue_log_hash_was_already_stale": ctx["stale_issue_log_hash"],
               "no_core_metric_modified": True})
    return {"label_summary_rows": len(label_rows), "label_null_replicate_rows": len(label_replicates),
            "pair_metric_summary_rows": len(pair_rows), "pair_null_replicate_rows": len(pair_replicates),
            "temporal_null": "NOT_COMPUTED", "permutations": permutations, "output": str(outdir)}


def pilot_null_controls(ctx: dict[str, Any], *, pilot_replicates: int, target_replicates: int,
                        workers: int) -> dict[str, Any]:
    """Time a small read-only pilot and estimate the full campaign cost."""
    if pilot_replicates < 2:
        raise ValueError("pilot requires at least two replicates")
    timings = {}
    for dataset in ("Synthetic", "Real"):
        item = next(i for i in ctx["index"] if i["dataset"] == dataset and i["architecture"] != "pca")
        metric = "condition_balanced_accuracy" if dataset == "Synthetic" else "direction_balanced_accuracy"
        observed = _core_metric_value(ctx["metrics"], dataset=dataset, arch=item["architecture"],
            objective=item["objective"], population=item["population"], seed=item["seed"],
            representation="raw", category="accessibility", metric=metric)
        start = time.perf_counter()
        _label_null_slot({"item": item, "representation": "raw",
                          "permutation_seeds": list(range(910000, 910000 + pilot_replicates)),
                          "observed": observed})
        timings[f"label_{dataset.lower()}_per_slot_seconds"] = (time.perf_counter() - start) / pilot_replicates
    # One paired sample per dataset; the timing includes all four supported
    # correspondence metrics on the exact frozen held-out support.
    pair_per_rep = {}
    for dataset, pop_a, pop_b in (("Synthetic", "A", "B"), ("Real", "A_PROXIMAL", "B_DISTAL")):
        item_a = next(i for i in ctx["index"] if i["dataset"] == dataset and i["architecture"] != "pca"
                      and i["population"] == pop_a)
        item_b = next(i for i in ctx["index"] if i["dataset"] == dataset and i["architecture"] == item_a["architecture"]
                      and i["objective"] == item_a["objective"] and i["seed"] == item_a["seed"] and i["population"] == pop_b)
        start = time.perf_counter()
        _pair_null_group({"item_a": item_a, "item_b": item_b, "permutations": pilot_replicates})
        pair_per_rep[dataset] = (time.perf_counter() - start) / pilot_replicates
    slots = 2 * len(ctx["index"])
    label_slot_seconds = (timings["label_synthetic_per_slot_seconds"] +
                          timings["label_real_per_slot_seconds"]) / 2
    serial_seconds = label_slot_seconds * slots * target_replicates
    parallel_seconds = serial_seconds / max(1, workers) * 1.5
    pair_cfgs_per_dataset = 2 * 4 * len(SEEDS)  # architectures*objectives*seeds; timing already covers raw and unit
    pair_seconds = sum(pair_per_rep.values()) * pair_cfgs_per_dataset * target_replicates
    result = {"pilot_replicates_per_sample_slot": pilot_replicates,
              "target_replicates": target_replicates, "observed_embedding_representation_slots": slots,
              "worker_count_assumption": workers, **timings,
              "label_shuffle_estimated_serial_hours": serial_seconds / 3600,
              "label_shuffle_estimated_parallel_hours_idealized": parallel_seconds / 3600,
              "pair_null_per_rep_per_two_representations_seconds": pair_per_rep,
              "pair_null_estimated_hours": pair_seconds / 3600,
              "pair_null_estimated_parallel_hours_idealized": pair_seconds / max(1, workers) * 1.5 / 3600,
              "combined_estimated_parallel_hours_idealized":
                  (parallel_seconds + pair_seconds / max(1, workers) * 1.5) / 3600,
              "interpretation": "estimates only; measured on this machine and exact stored support"}
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=("lag", "nulls", "null-pilot"), required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--pilot-replicates", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    ctx = load_core_context()
    if args.block == "lag":
        print(json.dumps(run_controlled_lag(ctx), indent=2))
    elif args.block == "nulls":
        print(json.dumps(run_null_controls(ctx, permutations=args.permutations, workers=args.workers), indent=2))
    elif args.block == "null-pilot":
        pilot_null_controls(ctx, pilot_replicates=args.pilot_replicates,
                            target_replicates=args.permutations, workers=args.workers)


if __name__ == "__main__":
    main()
