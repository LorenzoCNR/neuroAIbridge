"""Complete missing frozen downstream analyses in an isolated v2 branch.

No encoder is trained and no artifact in final_evaluation/v1 is modified.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "src"))
from neurobridge.eval.representation import _sampled_pair_distances
import final_gap_closure as gap1
import final_thesis_core_metrics as core
import final_thesis_downstream_eval as de
import final_thesis_uncertainty as uncertainty
from neurobridge.experiments.presentation_rebuild import _synthetic_config
from neurobridge.experiments.staged_shared_latent import stage_generate

BASE = PROJECT / "outputs/final_thesis_v1/final_evaluation"
ROOT = BASE / "gap_closure_v2"
SOURCE_ROOT = ROOT / "reconstructed_source"
SOURCE_NPZ = (SOURCE_ROOT / "outputs/runs/clean_rebuild_2026-09-23_seed42_synthetic"
              / "stage01_data/shared_data.npz")
SOURCE_CONFIG = SOURCE_NPZ.with_name("config.json")
CORE = BASE / "core_metrics"
NULLS = BASE / "null_controls"
REPS = ("raw", "unit")
Z_SEED_START = 830000
BACC_SEED_START = {"Synthetic": 860000, "Real": 870000}
LABEL_SEED_START = 731000
LABEL_EXTENSION_REPS = 2000
LABEL_TOTAL_REPS = 3000
EXPECTED_DATA_SHA = "079e283ab896c147cff12e5f089f9080e89418925eb6629da4a76af1c50baab8"
EXPECTED_CONFIG_SHA = "fdd80c6e0bf918883b58968401bb2d73faea21336d886694abdce41c5bc0ea25"
EXPECTED_CANONICAL_CONFIG_SHA = "53fe5f508f0c23b3db0a487bf2da5332423227b0c0d56109ad61d39d4f8b1dd7"
EXPECTED_GENERATOR_SHA = "7f98e8d63c073b82f79fcdcdb7f8dc07763596741b8464bd500907e4e0c524f8"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"immutable artifact differs: {path}")
        return
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with tmp.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("refusing to write an empty table")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def write_json_once(path: Path, payload: Any) -> None:
    write_once(path, (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode())


def build_exact_source() -> dict[str, Any]:
    """Rebuild the exact missing stage-1 parent in an isolated output branch."""
    source_code = PROJECT / "src/neurobridge/experiments/staged_shared_latent.py"
    cfg = _synthetic_config(42)
    resolved_config_hash = canonical_sha(core.synth._public_config(cfg))
    if resolved_config_hash != EXPECTED_CANONICAL_CONFIG_SHA:
        raise AssertionError(f"resolved config differs from frozen canonical hash: {resolved_config_hash}")
    if sha(source_code) != EXPECTED_GENERATOR_SHA:
        raise AssertionError("generator source does not match frozen core provenance")
    if not SOURCE_NPZ.exists():
        if SOURCE_ROOT.exists() and any(SOURCE_ROOT.iterdir()):
            raise FileExistsError(f"unexpected partial reconstruction tree: {SOURCE_ROOT}")
        stage_generate(SOURCE_ROOT, cfg)
    source_sha, config_sha = sha(SOURCE_NPZ), sha(SOURCE_CONFIG)
    cfg_on_disk = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    cfg_canonical_sha = canonical_sha(cfg_on_disk)
    if source_sha != EXPECTED_DATA_SHA:
        raise AssertionError(f"regenerated parent SHA differs: {source_sha} != {EXPECTED_DATA_SHA}")
    if config_sha != EXPECTED_CONFIG_SHA:
        raise AssertionError(f"regenerated config SHA differs: {config_sha} != {EXPECTED_CONFIG_SHA}")
    if cfg_canonical_sha != EXPECTED_CANONICAL_CONFIG_SHA:
        raise AssertionError(f"regenerated config values differ: {cfg_canonical_sha}")
    record = {"status": "EXACT_PARENT_RECONSTRUCTED", "expected_shared_data_sha256": EXPECTED_DATA_SHA,
        "reconstructed_shared_data_sha256": source_sha, "expected_config_sha256": EXPECTED_CONFIG_SHA,
        "reconstructed_config_sha256": config_sha, "canonical_config_sha256": cfg_canonical_sha,
        "generator_source_sha256": sha(source_code), "expected_generator_source_sha256": EXPECTED_GENERATOR_SHA,
        "source_path": str(SOURCE_NPZ.relative_to(PROJECT)), "config_path": str(SOURCE_CONFIG.relative_to(PROJECT)),
        "reconstruction": "frozen deterministic stage_generate; seed 42; data config recovered from versioned run builder",
        "created_utc": datetime.now(timezone.utc).isoformat(), "training_performed": False}
    provenance_path = ROOT / "RECONSTRUCTED_SYNTHETIC_PARENT_PROVENANCE.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        for key in ("status", "reconstructed_shared_data_sha256", "reconstructed_config_sha256",
                    "canonical_config_sha256", "generator_source_sha256", "source_path", "config_path"):
            if previous.get(key) != record.get(key):
                raise ValueError(f"existing reconstruction provenance disagrees at {key}")
        record = previous
    else:
        write_json_once(provenance_path, record)
    return record


def sampled_pair_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n * (n - 1) // 2 <= core.PAIR_COUNT:
        return np.triu_indices(n, k=1)
    rng = np.random.default_rng(0)
    left = rng.integers(0, n, size=core.PAIR_COUNT)
    right = rng.integers(0, n, size=core.PAIR_COUNT)
    keep = left != right
    return left[keep], right[keep]


def paired_rsa(x: np.ndarray, y: np.ndarray, left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    dx = _sampled_pair_distances(np.asarray(x, float), left, right, "euclidean")
    dy = _sampled_pair_distances(np.asarray(y, float), left, right, "euclidean")
    if len(dx) < 2 or np.allclose(dx, dx[0]) or np.allclose(dy, dy[0]):
        return float("nan"), float("nan")
    return float(spearmanr(dx, dy).statistic), float(pearsonr(dx, dy).statistic)


def bootstrap_z_slot(ctx: dict[str, Any], item: dict[str, Any], rep: str,
                     z_data: dict[str, np.ndarray], ids: np.ndarray,
                     draws: list[list[int]], seeds: list[int]) -> tuple[list[dict], list[dict]]:
    pop = item["population"]
    meta = core._synthetic_embedding(item)
    mask = core._test_mask(item, meta, pop)
    trial = meta["trial_id"][mask].astype(int)
    time_id = meta["time_id"][mask].astype(int)
    embedding = meta[f"embedding_{rep}"][mask].astype(float)
    target = np.asarray(z_data[f"Z_{pop}"])[trial, time_id].astype(float)
    if not np.array_equal(meta["labels"][mask].astype(int), np.asarray(z_data["labels"])[trial].astype(int)):
        raise AssertionError(f"source labels disagree with frozen embedding metadata: {item['trial_id']}")
    direct = core._metric_geometry(embedding, target)
    observed = {m: float(direct[m]) for m in ("procrustes_r2", "rsa_spearman", "rsa_pearson")}
    for metric, value in observed.items():
        frozen = de._core_metric_value(ctx["metrics"], dataset="Synthetic", arch=item["architecture"],
            objective=item["objective"], population=pop, seed=item["seed"], representation=rep,
            category="geometry", metric=metric, reference="Z")
        if not np.isclose(value, frozen, rtol=0, atol=1e-10):
            raise AssertionError(f"reconstructed parent point estimate differs: {item['trial_id']} {rep} {metric}: {value} != {frozen}")

    row_indices = {int(tid): np.flatnonzero(trial == tid) for tid in ids}
    trial_suff = gap1.trial_stats(embedding, target, trial, ids)
    count_matrix = np.asarray([np.bincount(draw, minlength=len(ids)) for draw in draws], dtype=float)
    proc_values = gap1.procrustes_from_stats(trial_suff, count_matrix)
    left, right = sampled_pair_indices(len(trial))
    spearman_values = np.empty(len(draws), dtype=float)
    pearson_values = np.empty(len(draws), dtype=float)
    for r, draw in enumerate(draws):
        row = np.concatenate([row_indices[int(ids[j])] for j in draw])
        spearman_values[r], pearson_values[r] = paired_rsa(embedding[row], target[row], left, right)
    arrays = {"procrustes_r2": proc_values, "rsa_spearman": spearman_values, "rsa_pearson": pearson_values}
    common = {"dataset": "Synthetic", "architecture": item["architecture"], "objective": item["objective"],
        "population": pop, "training_seed": item["seed"], "trial_id": item["trial_id"],
        "representation": rep, "reference": "Z", "n_trials": len(ids),
        "resampling_unit": "complete held-out trial"}
    summaries, replicates = [], []
    for metric, values in arrays.items():
        finite = np.asarray(values, float)
        finite = finite[np.isfinite(finite)]
        summaries.append({**common, "metric": metric, "observed": observed[metric],
            "bootstrap_mean": float(np.mean(finite)), "bootstrap_se": float(np.std(finite, ddof=1)),
            "ci_2_5": float(np.quantile(finite, .025)), "ci_97_5": float(np.quantile(finite, .975)),
            "finite_replicates": len(finite), "requested_replicates": len(draws),
            "embedding_parent_sha256": item["embedding_sha256"],
            "status": "OK" if len(finite) >= 950 else "LOW_FINITE_REPLICATES"})
        replicates.extend({"dataset": "Synthetic", "architecture": item["architecture"],
            "objective": item["objective"], "population": pop, "training_seed": item["seed"],
            "trial_id": item["trial_id"], "representation": rep, "metric": metric,
            "replicate": i, "bootstrap_seed": seeds[i], "value": float(v)}
            for i, v in enumerate(values))
    return summaries, replicates


def run_z_bootstrap(ctx: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "synthetic_z_bootstrap"
    if (out / "PROVENANCE.json").exists():
        raise FileExistsError("synthetic-Z bootstrap branch already sealed")
    source = build_exact_source()
    with np.load(SOURCE_NPZ, allow_pickle=False) as f:
        z_data = {k: f[k] for k in ("Z_A", "Z_B", "labels")}
    items = [i for i in ctx["index"] if i["dataset"] == "Synthetic"]
    if len(items) != 50:
        raise AssertionError(f"expected 50 Synthetic PCA/neural population slots, got {len(items)}")
    first = core._synthetic_embedding(items[0])
    ids = np.unique(first["trial_id"][core._test_mask(items[0], first, items[0]["population"])]).astype(int)
    if len(ids) != 40:
        raise AssertionError(f"expected 40 frozen held-out trials, got {len(ids)}")
    seeds = [Z_SEED_START + i for i in range(1000)]
    draws = [np.random.default_rng(s).integers(len(ids), size=len(ids)).tolist() for s in seeds]
    checkpoint_dir = out / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    campaign = {"replicates": 1000, "seed_start": seeds[0], "seed_end": seeds[-1],
        "held_out_trials": ids.tolist(), "reconstructed_parent_sha256": source["reconstructed_shared_data_sha256"],
        "core_provenance_sha256": sha(CORE / "PROVENANCE.json"),
        "metric_source_sha256": sha(PROJECT / "src/neurobridge/eval/representation.py")}
    campaign_hash = canonical_sha(campaign)
    write_json_once(checkpoint_dir / "CAMPAIGN.json", {**campaign, "campaign_sha256": campaign_hash})
    summaries, replicate_rows = [], []
    for idx, item in enumerate(items, 1):
        for rep in REPS:
            slot = hashlib.sha256(f"{item['trial_id']}|{rep}".encode()).hexdigest()
            checkpoint = checkpoint_dir / f"{slot}.json"
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                if saved.get("campaign_sha256") != campaign_hash:
                    raise ValueError(f"bootstrap checkpoint provenance mismatch: {checkpoint}")
                slot_summary, slot_replicates = saved["summary"], saved["replicates"]
            else:
                slot_summary, slot_replicates = bootstrap_z_slot(ctx, item, rep, z_data, ids, draws, seeds)
                write_json_once(checkpoint, {"campaign_sha256": campaign_hash,
                    "summary": slot_summary, "replicates": slot_replicates})
            summaries.extend(slot_summary)
            replicate_rows.extend(slot_replicates)
        if idx % 5 == 0 or idx == len(items):
            print(f"Synthetic Z trial bootstrap: completed {idx}/{len(items)} model slots", flush=True)
    summary_path = out / "SYNTHETIC_Z_TRIAL_BOOTSTRAP.csv"
    replicates_path = out / "SYNTHETIC_Z_TRIAL_BOOTSTRAP_REPLICATES.csv"
    seed_path = out / "BOOTSTRAP_SEEDS.csv"
    write_once(summary_path, csv_bytes(summaries))
    write_once(replicates_path, csv_bytes(replicate_rows))
    write_once(seed_path, csv_bytes([{"replicate": i, "seed": seed,
        "resampling_unit": "complete held-out Synthetic trial"} for i, seed in enumerate(seeds)]))
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "COMPLETE",
        "parent_hashes": {"reconstructed_shared_data_sha256": source["reconstructed_shared_data_sha256"],
            "expected_original_shared_data_sha256": EXPECTED_DATA_SHA,
            "reconstructed_config_sha256": source["reconstructed_config_sha256"],
            "canonical_config_sha256": source["canonical_config_sha256"],
            "generator_source_sha256": source["generator_source_sha256"],
            "core_provenance_sha256": sha(CORE / "PROVENANCE.json"),
            "core_metrics_sha256": sha(CORE / "CORE_METRICS_LONG.csv"),
            "embedding_index_sha256": sha(CORE / "EMBEDDING_INDEX.csv"),
            "support_sha256": sha(CORE / "EVALUATION_SUPPORT_MANIFEST.json")},
        "settings": {"replicates": 1000, "training_seed_variability_separate": True,
            "training_seeds": [1101, 1201, 1301], "held_out_trial_count": 40,
            "metrics": ["procrustes_r2", "rsa_spearman", "rsa_pearson"],
            "representations": list(REPS), "populations": ["A", "B"], "includes_pca_reference": True,
            "observed_point_estimates_match_frozen_core_atol_1e_10": True, "encoder_training": False},
        "artifacts": {str(p.relative_to(PROJECT)): sha(p) for p in (summary_path, replicates_path, seed_path)}}
    write_json_once(out / "PROVENANCE.json", provenance)
    return {"summary_rows": len(summaries), "replicate_rows": len(replicate_rows), "status": "COMPLETE"}


def run_stratified_bacc(ctx: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "stratified_bacc_bootstrap"
    if (out / "PROVENANCE.json").exists():
        raise FileExistsError("stratified-BACC branch already sealed")
    seeds = {d: [BACC_SEED_START[d] + i for i in range(1000)] for d in ("Synthetic", "Real")}
    cached_draws: dict[tuple, list[dict[int, np.ndarray]]] = {}
    rows = []
    for slot, item in enumerate(ctx["index"], 1):
        dataset = item["dataset"]
        data = de.load_item_arrays(item)
        mask = core._test_mask(item, data, item["population"])
        trial = data["trial_id"][mask].astype(int)
        labels = data["labels"][mask].astype(int) if dataset == "Synthetic" else data["target"][mask].astype(int)
        trial_ids = np.unique(trial)
        trial_class = {}
        for tid in trial_ids:
            classes_in_trial = np.unique(labels[trial == tid])
            if len(classes_in_trial) != 1:
                raise ValueError(f"label is not trial-level in {item['trial_id']} trial {tid}")
            trial_class[int(tid)] = int(classes_in_trial[0])
        classes = sorted(set(trial_class.values()))
        if len(classes) != 8:
            raise ValueError(f"expected 8 held-out classes for {item['trial_id']}, got {classes}")
        per_class = {c: np.asarray([t for t in trial_ids if trial_class[int(t)] == c], dtype=int) for c in classes}
        key = (dataset, tuple((c, tuple(v.tolist())) for c, v in per_class.items()))
        if key not in cached_draws:
            draws = []
            for seed in seeds[dataset]:
                rng = np.random.default_rng(seed)
                draws.append({c: rng.integers(len(per_class[c]), size=len(per_class[c])) for c in classes})
            cached_draws[key] = draws
        for rep in REPS:
            pred = uncertainty._fit_probe_predictions(ctx, item, data, rep)["cls_pred"]
            direct = float(core.balanced_accuracy_score(labels, pred))
            metric = "condition_balanced_accuracy" if dataset == "Synthetic" else "direction_balanced_accuracy"
            observed = de._core_metric_value(ctx["metrics"], dataset=dataset, arch=item["architecture"],
                objective=item["objective"], population=item["population"], seed=item["seed"],
                representation=rep, category="accessibility", metric=metric)
            if not np.isclose(direct, observed, rtol=0, atol=1e-10):
                raise AssertionError(f"frozen probe reproduction mismatch: {item['trial_id']} {rep}")
            per_trial = {}
            for tid in trial_ids:
                ix = trial == tid
                per_trial[int(tid)] = (int(np.sum(pred[ix] == labels[ix])), int(np.sum(ix)))
            scores = np.empty(1000, dtype=float)
            for b, draw_by_class in enumerate(cached_draws[key]):
                recalls = []
                for c in classes:
                    ids = per_class[c][draw_by_class[c]]
                    correct = sum(per_trial[int(t)][0] for t in ids)
                    total = sum(per_trial[int(t)][1] for t in ids)
                    recalls.append(correct / total if total else np.nan)
                scores[b] = float(np.mean(recalls)) if np.all(np.isfinite(recalls)) else np.nan
            finite = scores[np.isfinite(scores)]
            rows.append({"dataset": dataset, "architecture": item["architecture"],
                "objective": item["objective"], "population": item["population"],
                "training_seed": item["seed"], "trial_id": item["trial_id"], "representation": rep,
                "metric": metric, "observed": observed, "bootstrap_mean": float(finite.mean()),
                "bootstrap_se": float(finite.std(ddof=1)), "ci_2_5": float(np.quantile(finite, .025)),
                "ci_97_5": float(np.quantile(finite, .975)), "finite_replicates": len(finite),
                "requested_replicates": 1000, "n_held_out_trials": len(trial_ids),
                "class_trial_counts": json.dumps({str(c): len(per_class[c]) for c in classes}, sort_keys=True),
                "resampling_unit": "complete held-out trial resampled within observed label class",
                "estimand": "conditional on observed class trial counts; does not quantify class-prevalence uncertainty",
                "probe_refit_per_replicate": False, "status": "OK" if len(finite) == 1000 else "LOW_FINITE_REPLICATES"})
        if slot % 10 == 0 or slot == len(ctx["index"]):
            print(f"Stratified BACC bootstrap: completed {slot}/{len(ctx['index'])} model slots", flush=True)
    table = out / "STRATIFIED_BACC_TRIAL_BOOTSTRAP.csv"
    seed_path = out / "BOOTSTRAP_SEEDS.csv"
    write_once(table, csv_bytes(rows))
    write_once(seed_path, csv_bytes([{"dataset": d, "replicate": i, "seed": seed,
        "resampling_unit": "complete held-out trial within class"} for d in seeds for i, seed in enumerate(seeds[d])]))
    decision = out / "BACC_BOOTSTRAP_METHOD_DECISION.md"
    text = """# Balanced-accuracy bootstrap decision

Use the class-stratified complete-trial bootstrap as the preferred sampling-uncertainty interval for Balanced Accuracy. Each observed test class contributes exactly its original number of held-out trials to each replicate; whole trials, not windows, are resampled. This matches balanced accuracy's equal-class-recall estimand and avoids undefined replicates with an absent class.

The interval is conditional on observed held-out class trial counts; it does not quantify uncertainty in class prevalence. The unstratified intervals in `gap_closure_v1` remain preserved as sensitivity results. The frozen probe is reproduced using its existing train/validation-selected settings and test predictions; probes and encoders are not refit per bootstrap replicate. Training-seed variability remains separate.
"""
    write_once(decision, text.encode())
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "COMPLETE",
        "parent_hashes": {"core_provenance_sha256": sha(CORE / "PROVENANCE.json"),
            "core_metrics_sha256": sha(CORE / "CORE_METRICS_LONG.csv"),
            "embedding_index_sha256": sha(CORE / "EMBEDDING_INDEX.csv"),
            "evaluation_support_sha256": sha(CORE / "EVALUATION_SUPPORT_MANIFEST.json"),
            "probe_selection_sha256": sha(CORE / "DOWNSTREAM_PROBE_SELECTION.csv")},
        "settings": {"replicates": 1000, "seed_start_by_dataset": BACC_SEED_START,
            "class_stratified": True, "complete_trials_only": True, "probe_refit_per_replicate": False,
            "encoder_training": False, "training_seed_variability_separate": True},
        "artifacts": {str(p.relative_to(PROJECT)): sha(p) for p in (table, seed_path, decision)}}
    write_json_once(out / "PROVENANCE.json", provenance)
    return {"rows": len(rows), "status": "COMPLETE"}


def _label_payloads(ctx: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    payloads = []
    for item in ctx["index"]:
        metric = "condition_balanced_accuracy" if item["dataset"] == "Synthetic" else "direction_balanced_accuracy"
        for rep in REPS:
            payloads.append({"item": item, "representation": rep, "permutation_seeds": seeds,
                "observed": de._core_metric_value(ctx["metrics"], dataset=item["dataset"],
                    arch=item["architecture"], objective=item["objective"], population=item["population"],
                    seed=item["seed"], representation=rep, category="accessibility", metric=metric)})
    return payloads


def run_label_extension(ctx: dict[str, Any], workers: int) -> dict[str, Any]:
    """Reuse the frozen 1000 label permutations and add 2000, without touching them."""
    out = ROOT / "label_shuffle_holm_3000"
    if (out / "PROVENANCE.json").exists():
        raise FileExistsError("label-shuffle extension branch already sealed")
    original_path = NULLS / "NULL_LABEL_SHUFFLE_REPLICATES.csv"
    original_summary_path = NULLS / "NULL_LABEL_SHUFFLE.csv"
    with original_path.open(newline="", encoding="utf-8") as f:
        original = list(csv.DictReader(f))
    with original_summary_path.open(newline="", encoding="utf-8") as f:
        original_summary = list(csv.DictReader(f))
    if len(original) != 244000 or len(original_summary) != 244:
        raise ValueError(f"unexpected original label-null sizes: replicates={len(original)}, summaries={len(original_summary)}")
    if {int(r["permutation_seed"]) for r in original[:1000]} != set(range(730000, 731000)):
        raise ValueError("original label-null seed range is not 730000-730999")
    ext_seeds = list(range(LABEL_SEED_START, LABEL_SEED_START + LABEL_EXTENSION_REPS))
    payloads = _label_payloads(ctx, ext_seeds)
    if len(payloads) != 244:
        raise ValueError(f"frozen label-null slot count differs: {len(payloads)} != 244")
    parent_hashes = {"core_provenance_sha256": sha(CORE / "PROVENANCE.json"),
        "core_metrics_sha256": sha(CORE / "CORE_METRICS_LONG.csv"),
        "embedding_index_sha256": sha(CORE / "EMBEDDING_INDEX.csv"),
        "support_sha256": sha(CORE / "EVALUATION_SUPPORT_MANIFEST.json"),
        "original_label_summary_sha256": sha(original_summary_path),
        "original_label_replicates_sha256": sha(original_path),
        "driver_sha256": sha(Path(__file__).resolve()), "downstream_eval_source_sha256": sha(Path(de.__file__).resolve())}
    settings = {"original_replicates_reused": 1000, "additional_replicates": 2000,
        "total_replicates": 3000, "original_seed_range": [730000, 730999],
        "extension_seed_range": [ext_seeds[0], ext_seeds[-1]],
        "families": {"Synthetic_condition_balanced_accuracy": 100, "Real_direction_balanced_accuracy": 144},
        "permutation_unit": "complete trial label within each frozen train/validation/test split",
        "same permutation seeds across all candidate/model slots": True}
    checkpoint_root = out / "_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    campaign = {"parent_hashes": parent_hashes, "settings": settings,
        "slot_ids": [de._null_slot_key("label_shuffle", payload) for payload in payloads]}
    campaign_hash = canonical_sha(campaign)
    write_json_once(checkpoint_root / "CAMPAIGN.json", {**campaign, "campaign_sha256": campaign_hash})
    results: dict[str, list[float]] = {}
    pending = []
    for payload in payloads:
        slot = hashlib.sha256(de._null_slot_key("label_shuffle", payload).encode()).hexdigest()
        checkpoint = checkpoint_root / f"{slot}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved.get("campaign_sha256") != campaign_hash or len(saved.get("values", [])) != LABEL_EXTENSION_REPS:
                raise ValueError(f"label-null checkpoint is invalid: {checkpoint}")
            results[slot] = saved["values"]
        else:
            pending.append((slot, checkpoint, payload))
    print(f"Label-shuffle extension: reused {len(results)}; pending {len(pending)}/{len(payloads)} slots; {workers} workers", flush=True)
    def save_result(slot: str, checkpoint: Path, payload: dict[str, Any], result: dict[str, Any]) -> None:
        if len(result.get("null_values", [])) != LABEL_EXTENSION_REPS:
            raise ValueError("worker returned incomplete label-shuffle extension")
        values = result["null_values"]
        write_json_once(checkpoint, {"campaign_sha256": campaign_hash, "values": values})
        results[slot] = values
    completed = len(results)
    if workers <= 1:
        for slot, checkpoint, payload in pending:
            save_result(slot, checkpoint, payload, de._label_null_slot(payload))
            completed += 1
            if completed % 15 == 0 or completed == len(payloads):
                print(f"Label-shuffle extension: completed {completed}/{len(payloads)} slots", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(de._label_null_slot, payload): (slot, checkpoint, payload)
                          for slot, checkpoint, payload in pending}
            for future in concurrent.futures.as_completed(future_map):
                slot, checkpoint, payload = future_map[future]
                save_result(slot, checkpoint, payload, future.result())
                completed += 1
                if completed % 15 == 0 or completed == len(payloads):
                    print(f"Label-shuffle extension: completed {completed}/{len(payloads)} slots", flush=True)

    extension_rows = []
    extension_by_key: dict[tuple, np.ndarray] = {}
    for payload in payloads:
        item, rep = payload["item"], payload["representation"]
        slot = hashlib.sha256(de._null_slot_key("label_shuffle", payload).encode()).hexdigest()
        metric = "condition_balanced_accuracy" if item["dataset"] == "Synthetic" else "direction_balanced_accuracy"
        key = (item["dataset"], item["architecture"], item["objective"], item["population"],
               str(item["seed"]), item["trial_id"], rep, metric)
        vals = np.asarray(results[slot], dtype=float)
        extension_by_key[key] = vals
        extension_rows.extend({"dataset": key[0], "architecture": key[1], "objective": key[2],
            "population": key[3], "training_seed": key[4], "trial_id": key[5], "representation": key[6],
            "metric": key[7], "replicate": 1000 + i, "permutation_seed": ext_seeds[i],
            "null_balanced_accuracy": float(value)} for i, value in enumerate(vals))
    original_by_key: dict[tuple, list[float]] = {}
    for r in original:
        key = (r["dataset"], r["architecture"], r["objective"], r["population"],
               str(r["training_seed"]), r["trial_id"], r["representation"], r["metric"])
        original_by_key.setdefault(key, []).append(float(r["null_balanced_accuracy"]))
    combined_summary = []
    for r in original_summary:
        key = (r["dataset"], r["architecture"], r["objective"], r["population"],
               str(r["training_seed"]), r["trial_id"], r["representation"], r["metric"])
        values = np.r_[np.asarray(original_by_key[key], dtype=float), extension_by_key[key]]
        observed = float(r["observed"])
        combined_summary.append({**r, "null_mean": float(values.mean()), "null_sd": float(values.std(ddof=1)),
            "empirical_p": float((1 + np.sum(values >= observed)) / (LABEL_TOTAL_REPS + 1)),
            "percentile_within_null": float(100 * (np.sum(values < observed) + .5 * np.sum(values == observed)) / len(values)),
            "observed_minus_null_mean": observed - float(values.mean()), "replicates": LABEL_TOTAL_REPS,
            "permutation_seed_start": 730000, "permutation_seed_end": ext_seeds[-1],
            "unit_of_permutation": "complete trial label within split"})

    ext_path = out / "NULL_LABEL_SHUFFLE_EXTENSION_REPLICATES_2000.csv"
    summary_path = out / "NULL_LABEL_SHUFFLE_SUMMARY_3000.csv"
    write_once(ext_path, csv_bytes(extension_rows))
    write_once(summary_path, csv_bytes(combined_summary))
    holm_rows, holm_summary = [], []
    for dataset in ("Synthetic", "Real"):
        family = [r for r in combined_summary if r["dataset"] == dataset]
        family.sort(key=lambda x: float(x["empirical_p"]))
        previous = 0.0
        for rank, r in enumerate(family):
            p_adj = max(previous, min(1.0, (len(family) - rank) * float(r["empirical_p"])))
            previous = p_adj
            holm_rows.append({"dataset": dataset, "architecture": r["architecture"], "objective": r["objective"],
                "population": r["population"], "training_seed": r["training_seed"], "trial_id": r["trial_id"],
                "representation": r["representation"], "metric": r["metric"], "observed": r["observed"],
                "empirical_p": r["empirical_p"], "holm_p_family": p_adj, "family_size": len(family)})
        holm_summary.append({"family": f"{dataset}_label_balanced_accuracy", "n_tests": len(family),
            "raw_p_lt_0_05": sum(float(r["empirical_p"]) < .05 for r in family),
            "holm_p_lt_0_05": sum(float(r["holm_p_family"]) < .05 for r in holm_rows if r["dataset"] == dataset),
            "total_permutations": LABEL_TOTAL_REPS, "minimum_attainable_p": 1 / (LABEL_TOTAL_REPS + 1),
            "minimum_attainable_holm_p": min(1.0, len(family) / (LABEL_TOTAL_REPS + 1))})
    holm_path = out / "HOLM_LABEL_FAMILY_RESULTS_3000.csv"
    holm_summary_path = out / "HOLM_LABEL_FAMILY_SUMMARY_3000.csv"
    seeds_path = out / "PERMUTATION_SEEDS.csv"
    write_once(holm_path, csv_bytes(holm_rows)); write_once(holm_summary_path, csv_bytes(holm_summary))
    write_once(seeds_path, csv_bytes([{"stage": "original_reused", "replicate": i, "seed": 730000 + i} for i in range(1000)] +
        [{"stage": "extension", "replicate": 1000 + i, "seed": ext_seeds[i]} for i in range(LABEL_EXTENSION_REPS)]))
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "COMPLETE",
        "parent_hashes": parent_hashes, "settings": settings, "no_model_selection": True,
        "no_encoder_training": True,
        "artifacts": {str(p.relative_to(PROJECT)): sha(p) for p in (ext_path, summary_path, holm_path, holm_summary_path, seeds_path)}}
    write_json_once(out / "PROVENANCE.json", provenance)
    return {"extension_replicate_rows": len(extension_rows), "summary_rows": len(combined_summary),
            "holm_summary": holm_summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("verify-source", "z-bootstrap", "stratified-bacc", "label-extension"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "verify-source":
        print(json.dumps(build_exact_source(), indent=2))
        return
    ctx = de.load_core_context()
    if args.stage == "z-bootstrap":
        result = run_z_bootstrap(ctx)
    elif args.stage == "stratified-bacc":
        result = run_stratified_bacc(ctx)
    else:
        result = run_label_extension(ctx, args.workers)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
