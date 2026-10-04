"""Seed-variability and held-out complete-trial bootstrap for frozen outputs."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import final_thesis_core_metrics as core  # noqa: E402
import final_thesis_downstream_eval as de  # noqa: E402
from neurobridge.eval.representation import linear_cka, procrustes_r2  # noqa: E402


def trial_bootstrap_draws(data: dict[str, np.ndarray], labels: np.ndarray,
                          row_mask: np.ndarray, *, n_replicates: int,
                          seed_start: int) -> tuple[list[np.ndarray], list[int]]:
    """Resample complete held-out trial IDs with replacement, without stratification."""
    trial = np.asarray(data["trial_id"], dtype=np.int64)
    labels = np.asarray(labels)
    heldout_trials = np.unique(trial[np.asarray(row_mask, dtype=bool)])
    for tid in heldout_trials:
        values = np.unique(labels[trial == tid])
        if len(values) != 1:
            raise ValueError(f"task label is not constant within trial {tid}")
    seeds = [seed_start + i for i in range(n_replicates)]
    draws = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        draws.append(rng.choice(heldout_trials, size=len(heldout_trials), replace=True).astype(np.int64))
    return draws, seeds


def bootstrap_row_indices(data: dict[str, np.ndarray], row_mask: np.ndarray,
                          sampled_trials: np.ndarray) -> np.ndarray:
    trial = np.asarray(data["trial_id"], dtype=np.int64)
    mask = np.asarray(row_mask, dtype=bool)
    by_trial = {int(t): np.flatnonzero(mask & (trial == t)) for t in np.unique(trial[mask])}
    if any(not len(by_trial.get(int(t), [])) for t in sampled_trials):
        raise ValueError("a resampled held-out trial has no supported rows")
    return np.concatenate([by_trial[int(t)] for t in sampled_trials])


def sampled_rsa_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError("RSA inputs have different sample counts")
    left, right = de._rsa_pair_indices(len(x))
    if left is None:
        from scipy.spatial.distance import pdist
        dx, dy = pdist(x), pdist(y)
    else:
        dx = np.linalg.norm(x[left] - x[right], axis=1)
        dy = np.linalg.norm(y[left] - y[right], axis=1)
    if len(dx) < 2 or np.allclose(dx, dx[0]) or np.allclose(dy, dy[0]):
        return float("nan")
    return float(spearmanr(dx, dy).statistic)


def _selection_for(ctx: dict[str, Any], item: dict[str, Any], representation: str) -> dict[str, str]:
    matches = [r for r in ctx["probe_selection"]
               if r["dataset"].lower() == item["dataset"].lower()
               and r["architecture"] == item["architecture"]
               and r["objective"] == item["objective"]
               and r["population"] == item["population"]
               and r["representation"] == representation
               and (item["seed"] is None or int(float(r["seed"])) == int(item["seed"]))]
    if len(matches) != 1:
        raise ValueError(f"probe-selection row not unique for {item['trial_id']}/{representation}: {len(matches)}")
    return matches[0]


def _parse_mapping_field(value: str) -> dict[str, Any]:
    """Parse serialized mapping fields from the frozen CSV safely.

    Historical probe-selection rows contain Python dict repr (single quotes),
    while a few rows use valid JSON.  Do not use ``eval`` on this artifact.
    """
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError("expected a serialized mapping")
    return parsed


def _fit_probe_predictions(ctx: dict[str, Any], item: dict[str, Any], data: dict[str, np.ndarray],
                           representation: str) -> dict[str, Any]:
    dataset, pop = item["dataset"], item["population"]
    masks = core._split_masks(data, dataset, pop)
    test_mask = core._test_mask(item, data, pop)
    x = data[f"embedding_{representation}"]
    selection = _selection_for(ctx, item, representation)
    y_cls = data["labels"] if dataset == "Synthetic" else data["target"].astype(int)
    seed = int(item["seed"]) if item["seed"] is not None else 0
    classifier = make_pipeline(StandardScaler(), LogisticRegression(
        C=float(selection["selected_C"]), solver="lbfgs", max_iter=1000, random_state=seed))
    classifier.fit(x[masks["train"]], y_cls[masks["train"]])
    cls_pred = classifier.predict(x[test_mask])
    if dataset == "Synthetic":
        expected_metric = "condition_balanced_accuracy"
    else:
        expected_metric = "direction_balanced_accuracy"
    cls_observed = balanced_accuracy_score(y_cls[test_mask], cls_pred)
    core_observed = de._core_metric_value(ctx["metrics"], dataset=dataset, arch=item["architecture"],
        objective=item["objective"], population=pop, seed=item["seed"], representation=representation,
        category="accessibility", metric=expected_metric)
    if not np.isclose(cls_observed, core_observed, rtol=1e-10, atol=1e-12):
        raise AssertionError(f"frozen probe reproduction differs for {item['trial_id']} {representation}: {cls_observed} vs {core_observed}")
    result = {"test_mask": test_mask, "y_cls": y_cls, "cls_pred": cls_pred,
              "masks": masks, "selection": selection, "x": x}
    if dataset == "Synthetic":
        progress = data["progress"]
        alpha = float(selection["selected_progress_alpha"])
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x[masks["train"]], progress[masks["train"]])
        result["progress_pred"] = model.predict(x[test_mask])
        result["progress"] = progress
        pred_r2 = r2_score(progress[test_mask], result["progress_pred"])
        expected = de._core_metric_value(ctx["metrics"], dataset=dataset, arch=item["architecture"],
            objective=item["objective"], population=pop, seed=item["seed"], representation=representation,
            category="accessibility", metric="progress_r2")
        if not np.isclose(pred_r2, expected, rtol=1e-10, atol=1e-12):
            raise AssertionError("recreated frozen progress probe differs from core")
    else:
        behavior_alphas = _parse_mapping_field(selection["behavior_alphas"])
        for name in ("position", "velocity"):
            target = data[name]
            model = make_pipeline(StandardScaler(), Ridge(alpha=float(behavior_alphas[name])))
            model.fit(x[masks["train"]], target[masks["train"]])
            pred = model.predict(x[test_mask])
            result[f"{name}_pred"] = pred
            result[name] = target
            pred_r2 = r2_score(target[test_mask], pred, multioutput="variance_weighted")
            expected = de._core_metric_value(ctx["metrics"], dataset=dataset, arch=item["architecture"],
                objective=item["objective"], population=pop, seed=item["seed"], representation=representation,
                category="accessibility", metric=f"{name}_r2")
            if not np.isclose(pred_r2, expected, rtol=1e-9, atol=1e-11):
                raise AssertionError(f"recreated frozen {name} probe differs from core")
    return result


def _diagnose_progress_reproduction(ctx: dict[str, Any], *, thread_limit: int | None) -> list[dict[str, Any]]:
    """Read-only diagnostic for exact-core progress-probe reproducibility."""
    mismatches = []
    def check_slots():
        for item in ctx["index"]:
            if item["dataset"] != "Synthetic":
                continue
            data = de.load_item_arrays(item)
            masks = core._split_masks(data, item["dataset"], item["population"])
            test_mask = core._test_mask(item, data, item["population"])
            for rep in ("raw", "unit"):
                selection = _selection_for(ctx, item, rep)
                target = data["progress"]
                model = make_pipeline(StandardScaler(), Ridge(alpha=float(selection["selected_progress_alpha"])))
                model.fit(data[f"embedding_{rep}"][masks["train"]], target[masks["train"]])
                pred = model.predict(data[f"embedding_{rep}"][test_mask])
                observed = float(r2_score(target[test_mask], pred))
                expected = de._core_metric_value(ctx["metrics"], dataset=item["dataset"],
                    arch=item["architecture"], objective=item["objective"], population=item["population"],
                    seed=item["seed"], representation=rep, category="accessibility", metric="progress_r2")
                if not np.isclose(observed, expected, rtol=1e-10, atol=1e-12):
                    mismatches.append({"trial_id": item["trial_id"], "architecture": item["architecture"],
                        "objective": item["objective"], "population": item["population"],
                        "seed": item["seed"], "representation": rep,
                        "selected_progress_alpha": selection["selected_progress_alpha"],
                        "n_train": int(masks["train"].sum()), "n_validation": int(masks["validation"].sum()),
                        "n_test": int(test_mask.sum()), "computed_r2": observed,
                        "core_r2": expected, "absolute_difference": abs(observed - expected),
                        "relative_tolerance": 1e-10, "absolute_tolerance": 1e-12,
                        "thread_limit": thread_limit})
            del data
    if thread_limit is None:
        check_slots()
    else:
        with threadpool_limits(limits=thread_limit):
            check_slots()
    return mismatches


def _summary(base: dict[str, Any], observed: float, samples: list[float]) -> dict[str, Any]:
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) >= 2:
        lo, hi = np.quantile(arr, [.025, .975])
        mean, se = float(np.mean(arr)), float(np.std(arr, ddof=1))
        status = "OK"
    elif len(arr) == 1:
        lo = hi = mean = float(arr[0])
        se = float("nan")
        status = "INSUFFICIENT_FINITE_REPLICATES"
    else:
        lo = hi = mean = se = float("nan")
        status = "NO_FINITE_REPLICATES"
    return {**base, "observed": float(observed), "bootstrap_mean": mean,
            "bootstrap_se": se, "ci_2_5": float(lo), "ci_97_5": float(hi),
            "finite_replicates": int(len(arr)), "requested_replicates": int(len(samples)),
            "status": status}


def _csv_payload(rows: list[dict[str, Any]]) -> str:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _write_partial_seed_branch(output: Path, rows: list[dict[str, Any]],
                               parent_hashes: dict[str, str], *, replicates: int = 1000) -> dict[str, Any]:
    """Publish seed variability before bootstrap, explicitly marking the branch partial."""
    output.mkdir(parents=True, exist_ok=True)
    seed_path = output / "TRAINING_SEED_VARIABILITY.csv"
    status_path = output / "UNCERTAINTY_STATUS.json"
    payload = _csv_payload(rows)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if seed_path.exists():
        if hashlib.sha256(seed_path.read_bytes()).hexdigest() != digest:
            raise ValueError("existing seed-variability CSV differs; refusing overwrite")
    else:
        tmp = seed_path.with_name(f".{seed_path.name}.tmp")
        tmp.write_text(payload, encoding="utf-8", newline="")
        tmp.replace(seed_path)
    status = {"branch": "uncertainty", "status": "PARTIAL_BOOTSTRAP_PENDING",
              "training_seed_variability_status": "COMPLETE",
              "bootstrap_status": "PENDING", "requested_bootstrap_replicates": replicates,
              "training_seed_variability_rows": len(rows),
              "training_seed_variability_sha256": digest,
              "parent_hashes": parent_hashes,
              "seed_interpretation": {
                  "1101": "HPO-selected checkpoint; not an independent final refit",
                  "1201": "independent final training seed",
                  "1301": "independent final training seed"},
              "near_collapse_seed_values_retained": True,
              "no_bootstrap_summary_or_seed_ci_figures_published": True}
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing != status:
            raise ValueError("existing uncertainty status differs; refusing overwrite")
    else:
        tmp = status_path.with_name(f".{status_path.name}.tmp")
        tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(status_path)
    return status


def _stream_slot_probes(ctx: dict[str, Any], item: dict[str, Any], *,
                        load_arrays=de.load_item_arrays,
                        fit_predictions=_fit_probe_predictions):
    """Load one slot once, fit both probe views, and return only predictions."""
    data = load_arrays(item)
    return {rep: fit_predictions(ctx, item, data, rep) for rep in ("raw", "unit")}


class _CellCheckpoint:
    """Atomic write-once summaries per configuration/representation cell."""

    def __init__(self, root: Path, settings: dict[str, Any]):
        self.root = root
        self.blocks_dir = root / "blocks"
        self.state_path = root / "WORK_STATE.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.blocks_dir.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            stored = json.loads(self.state_path.read_text(encoding="utf-8"))
            if stored != settings:
                raise ValueError("uncertainty checkpoint settings/parent hashes differ; refusing unsafe resume")
        else:
            self._write_atomic(self.state_path, json.dumps(settings, indent=2, sort_keys=True) + "\n")
        self.blocks: dict[str, list[dict[str, Any]]] = {}
        for path in sorted(self.blocks_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            key_text = json.dumps(payload["key"], sort_keys=True, separators=(",", ":"))
            expected = hashlib.sha256(key_text.encode("utf-8")).hexdigest() + ".json"
            if path.name != expected:
                raise ValueError(f"uncertainty checkpoint cell key/hash mismatch: {path}")
            self.blocks[key_text] = payload["rows"]

    @staticmethod
    def _write_atomic(path: Path, payload: str) -> None:
        temp = path.with_name(f".{path.name}.tmp")
        temp.write_text(payload, encoding="utf-8", newline="")
        temp.replace(path)

    @staticmethod
    def _key_text(key: dict[str, Any]) -> str:
        return json.dumps(key, sort_keys=True, separators=(",", ":"))

    def has(self, key: dict[str, Any]) -> bool:
        return self._key_text(key) in self.blocks

    def save(self, key: dict[str, Any], rows: list[dict[str, Any]]) -> None:
        key_text = self._key_text(key)
        if key_text in self.blocks:
            raise FileExistsError(f"uncertainty checkpoint cell already exists: {key}")
        filename = hashlib.sha256(key_text.encode("utf-8")).hexdigest() + ".json"
        path = self.blocks_dir / filename
        payload = {"key": key, "rows": rows}
        self._write_atomic(path, json.dumps(payload, sort_keys=True, allow_nan=True) + "\n")
        self.blocks[key_text] = rows

    def all_rows(self) -> list[dict[str, Any]]:
        return [row for rows in self.blocks.values() for row in rows]


def _emit_metric(rows: list[dict[str, Any]], ctx: dict[str, Any], *, item: dict[str, Any],
                 representation: str, metric: str, category: str, observed_metric: str,
                 reference: str | None, values: list[float], unit: str,
                 n_trials: int, n_samples: int) -> None:
    observed = de._core_metric_value(ctx["metrics"], dataset=item["dataset"], arch=item["architecture"],
        objective=item["objective"], population=item["population"], seed=item["seed"],
        representation=representation, category=category, metric=observed_metric, reference=reference)
    base = {"dataset": item["dataset"], "architecture": item["architecture"],
            "objective": item["objective"], "population": item["population"],
            "training_seed": item["seed"] if item["seed"] is not None else "deterministic",
            "trial_id": item["trial_id"], "representation": representation,
            "metric": metric, "reference": reference or "none", "uncertainty_type": "held_out_trial_bootstrap",
            "unit_of_resampling": "complete held-out trial; unstratified cluster bootstrap",
            "n_heldout_trials": n_trials, "n_supported_samples": n_samples}
    rows.append(_summary(base, observed, values))


def _build_seed_variability(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    chosen = []
    for row in ctx["metrics"]:
        dataset = row["dataset"].lower()
        category, metric, reference = row["category"], row["metric"], row["reference"]
        keep = False
        if dataset == "synthetic":
            keep = ((category == "geometry" and reference == "Z" and metric in ("procrustes_r2", "rsa_spearman")) or
                    (category == "accessibility" and metric in ("condition_balanced_accuracy", "progress_r2")) or
                    (category == "temporal_fidelity" and metric == "s_lag_r2" and
                     row["lag_bins"] not in ("", "nan") and float(row["lag_bins"]) == 10.0))
        elif dataset == "real":
            keep = ((category == "cross_population_consistency" and reference == "matched_A_B" and
                     metric in ("procrustes_r2", "rsa_spearman", "linear_cka")) or
                    (category == "accessibility" and metric in ("direction_balanced_accuracy", "position_r2", "velocity_r2")))
        if keep and row["seed"] not in ("", "nan", "deterministic"):
            chosen.append(row)
    keys = ("dataset", "architecture", "objective", "population", "representation", "reference", "category", "metric", "lag_bins")
    groups: dict[tuple[str, ...], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in chosen:
        key = tuple(row.get(k, "") or "" for k in keys)
        seed = int(float(row["seed"]))
        if seed in de.SEEDS:
            if seed in groups[key]:
                raise ValueError(f"duplicate metric for seed {seed}: {key}")
            try:
                value = float(row["value"])
            except (TypeError, ValueError):
                value = float("nan")
            groups[key][seed] = {
                "value": value,
                "fit_status": row.get("fit_status", "unknown"),
                "near_collapse": row.get("near_collapse", "unknown"),
            }
    for key, by_seed in sorted(groups.items()):
        if set(by_seed) != set(de.SEEDS):
            raise ValueError(f"expected three seed records for {key}; got {sorted(by_seed)}")
        values = {seed: by_seed[seed]["value"] for seed in de.SEEDS}
        arr = np.asarray([values[s] for s in de.SEEDS], dtype=float)
        finite = arr[np.isfinite(arr)]
        independent = np.asarray([values[s] for s in (1201, 1301)], dtype=float)
        independent = independent[np.isfinite(independent)]
        row = dict(zip(keys, key))
        for seed in de.SEEDS:
            value = values[seed]
            row[f"seed_{seed}"] = value
            row[f"seed_{seed}_fit_status"] = by_seed[seed]["fit_status"]
            row[f"seed_{seed}_near_collapse"] = by_seed[seed]["near_collapse"]
            row[f"seed_{seed}_metric_defined"] = bool(np.isfinite(value))
            row[f"seed_{seed}_metric_status"] = ("FINITE" if np.isfinite(value)
                                                   else "NA_SOURCE_METRIC_NONFINITE")
            row[f"seed_{seed}_role"] = ("HPO-selected checkpoint seed; not an independent final refit"
                                          if seed == 1101 else "independent final training seed")
        if len(finite):
            mean, median = float(np.mean(finite)), float(np.median(finite))
            minimum, maximum = float(np.min(finite)), float(np.max(finite))
        else:
            mean = median = minimum = maximum = float("nan")
        sd = float(np.std(finite, ddof=1)) if len(finite) >= 2 else float("nan")
        row.update({"n_training_seed_records": 3, "n_training_seeds": 3,
                    "n_finite_metric_values": int(len(finite)),
                    "mean": mean, "median": median, "sd": sd, "sd_sample": sd, "minimum": minimum,
                    "maximum": maximum, "range": float(maximum - minimum) if len(finite) else float("nan"),
                    "near_collapse_seed_count": sum(str(by_seed[s]["near_collapse"]).lower() == "true"
                                                     for s in de.SEEDS),
                    "n_independent_final_seeds": int(len(independent)),
                    "independent_final_seed_mean_1201_1301": float(np.mean(independent)) if len(independent) else float("nan"),
                    "independent_final_seed_range_1201_1301": float(np.ptp(independent)) if len(independent) >= 2 else float("nan"),
                    "summary_status": "all_three_defined_mixed_hpo_and_final_seeds" if len(finite) == 3 else
                                     "partial_defined_metric_values" if len(finite) else "no_defined_metric_values",
                    "uncertainty_type": "descriptive_seed_variability_not_population_CI",
                    "seed_1101_is_independent_replication": False})
        rows.append(row)
    return rows


def run_uncertainty(ctx: dict[str, Any], *, replicates: int, pilot: bool = False) -> dict[str, Any]:
    if replicates < 2:
        raise ValueError("at least two bootstrap replicates are required")
    output = de.EVAL / "uncertainty"
    parent_hashes = {"core_provenance": de.sha256(de.CORE / "PROVENANCE.json"),
                     "core_metrics": de.sha256(de.CORE / "CORE_METRICS_LONG.csv"),
                     "embedding_index": de.sha256(de.CORE / "EMBEDDING_INDEX.csv"),
                     "probe_selection": de.sha256(de.CORE / "DOWNSTREAM_PROBE_SELECTION.csv"),
                     "evaluation_support": de.sha256(de.CORE / "EVALUATION_SUPPORT_MANIFEST.json"),
                     "analysis_source": de.sha256(Path(__file__).resolve())}
    seed_variability = _build_seed_variability(ctx)
    if not pilot:
        if (output / "UNCERTAINTY_PROVENANCE.json").exists():
            raise FileExistsError(f"uncertainty branch is already finalized; preserve it: {output}")
        # Existing branch is permitted only as the explicitly marked partial seed artifact
        # or as an interrupted publication without final provenance.
        allowed = {"TRAINING_SEED_VARIABILITY.csv", "UNCERTAINTY_STATUS.json",
                   "TRIAL_BOOTSTRAP_PRIMARY_METRICS.csv", "UNCERTAINTY_SUMMARY.csv",
                   "TRIAL_BOOTSTRAP_SEEDS.csv", "figures"}
        if output.exists():
            unexpected = {p.name for p in output.iterdir()} - allowed
            if unexpected:
                raise FileExistsError(f"uncertainty branch contains unrecognized artifacts: {sorted(unexpected)}")
            if (output / "figures").exists():
                figure_names = {p.name for p in (output / "figures").iterdir()}
                expected_figures = {"synthetic_seed_dots_trial_bootstrap_ci.png",
                                    "synthetic_seed_dots_trial_bootstrap_ci.pdf",
                                    "real_seed_dots_trial_bootstrap_ci.png",
                                    "real_seed_dots_trial_bootstrap_ci.pdf"}
                if figure_names - expected_figures:
                    raise FileExistsError(f"unexpected uncertainty figures: {sorted(figure_names - expected_figures)}")
        _write_partial_seed_branch(output, seed_variability, parent_hashes, replicates=replicates)
    items = ctx["index"]
    syn_data, _ = core._synthetic_latents(de.PROJECT)
    all_draws: dict[str, list[np.ndarray]] = {}
    all_seeds: dict[str, list[int]] = {}
    for dataset in ("Synthetic", "Real"):
        item = next(i for i in items if i["dataset"] == dataset and i["architecture"] != "pca")
        data = de.load_item_arrays(item)
        labels = data["labels"] if dataset == "Synthetic" else data["target"]
        mask = core._test_mask(item, data, item["population"])
        seed_start = 830000 if dataset == "Synthetic" else 840000
        all_draws[dataset], all_seeds[dataset] = trial_bootstrap_draws(
            data, labels, mask, n_replicates=replicates, seed_start=seed_start)

    if pilot:
        # Reproduce the frozen probe metrics while streaming one slot at a time.
        checks = 0
        for slot_ix, item in enumerate(items, start=1):
            _stream_slot_probes(ctx, item)
            checks += 2
            if slot_ix % 15 == 0 or slot_ix == len(items):
                print(f"probe qualification: {slot_ix}/{len(items)} slots", flush=True)
        out = {"mode": "PILOT_NO_OUTPUTS", "replicates": replicates,
               "seed_variability_rows": len(seed_variability), "embedding_representation_slots": checks,
               "probe_reproduction_checks": "passed", "data_loading": "streamed one slot at a time",
               "note": "pilot fits no encoder and writes no branch files"}
        return out

    checkpoint = _CellCheckpoint(de.EVAL / ".uncertainty_work", {
        "branch": "uncertainty", "parent_hashes": parent_hashes,
        "replicates": replicates, "training_seeds": list(de.SEEDS),
        "synthetic_bootstrap_seed_start": 830000, "real_bootstrap_seed_start": 840000,
        "resampling_unit": "complete held-out trial; unstratified",
        "probe_policy": "fixed train-fitted, validation-selected downstream probes",
    })
    print(f"cell checkpoint: reusing {len(checkpoint.blocks)}/340 completed cells", flush=True)
    bootstrap_rows: list[dict[str, Any]] = checkpoint.all_rows()
    # Single-population Synthetic and Real accessibility/geometry metrics.
    for slot_ix, item in enumerate(items, start=1):
        dataset, population = item["dataset"], item["population"]
        if dataset not in ("Synthetic", "Real"):
            continue
        if all(checkpoint.has({"stage": "single_population", "trial_id": item["trial_id"],
                               "representation": rep}) for rep in ("raw", "unit")):
            continue
        data = de.load_item_arrays(item)
        for rep in ("raw", "unit"):
            cell_key = {"stage": "single_population", "trial_id": item["trial_id"],
                        "representation": rep}
            if checkpoint.has(cell_key):
                continue
            block_start = len(bootstrap_rows)
            pred = _fit_probe_predictions(ctx, item, data, rep)
            support_mask = pred["test_mask"]
            labels = data["labels"] if dataset == "Synthetic" else data["target"].astype(int)
            draws = all_draws[dataset]
            trial_ids = np.unique(data["trial_id"][support_mask])
            expected_n_trials = len(trial_ids)
            n_samples = int(support_mask.sum())
            if dataset == "Synthetic":
                z = syn_data[f"Z_{population}"]
                trial_coord = data["trial_id"].astype(int)
                time_coord = data["time_id"].astype(int)
                reference = z[trial_coord[support_mask], time_coord[support_mask]]
                embedding = data[f"embedding_{rep}"][support_mask]
                indices_in_support = np.flatnonzero(support_mask)
                # Map original row index -> row position in the frozen held-out support.
                pos = np.full(len(support_mask), -1, dtype=np.int64)
                pos[indices_in_support] = np.arange(len(indices_in_support))
                geo_values = {"procrustes_r2": [], "rsa_spearman": []}
                ytest = labels[support_mask]
                pred_cls = pred["cls_pred"]
                ba, prog = [], []
                yprogress = data["progress"][support_mask]
                pprogress = pred["progress_pred"]
                for draw in draws:
                    sampled_rows = bootstrap_row_indices(data, support_mask, draw)
                    p = pos[sampled_rows]
                    xboot, zboot = embedding[p], reference[p]
                    geo_values["procrustes_r2"].append(procrustes_r2(xboot, zboot))
                    geo_values["rsa_spearman"].append(sampled_rsa_spearman(xboot, zboot))
                    ba.append(balanced_accuracy_score(ytest[p], pred_cls[p]))
                    prog.append(r2_score(yprogress[p], pprogress[p]))
                for metric in ("procrustes_r2", "rsa_spearman"):
                    _emit_metric(bootstrap_rows, ctx, item=item, representation=rep,
                        metric=metric, category="geometry", observed_metric=metric, reference="Z",
                        values=geo_values[metric], unit="score", n_trials=expected_n_trials, n_samples=n_samples)
                _emit_metric(bootstrap_rows, ctx, item=item, representation=rep,
                    metric="condition_balanced_accuracy", category="accessibility",
                    observed_metric="condition_balanced_accuracy", reference=None, values=ba,
                    unit="balanced_accuracy", n_trials=expected_n_trials, n_samples=n_samples)
                _emit_metric(bootstrap_rows, ctx, item=item, representation=rep,
                    metric="progress_r2", category="accessibility", observed_metric="progress_r2",
                    reference=None, values=prog, unit="R2", n_trials=expected_n_trials, n_samples=n_samples)
            else:
                ytest = labels[support_mask]
                pos = np.full(len(support_mask), -1, dtype=np.int64)
                indices_in_support = np.flatnonzero(support_mask)
                pos[indices_in_support] = np.arange(len(indices_in_support))
                ba = []
                behavior_scores = {"position": [], "velocity": []}
                target_values = {name: data[name][support_mask] for name in behavior_scores}
                prediction_values = {name: pred[f"{name}_pred"] for name in behavior_scores}
                for draw in draws:
                    sampled_rows = bootstrap_row_indices(data, support_mask, draw)
                    p = pos[sampled_rows]
                    ba.append(balanced_accuracy_score(ytest[p], pred["cls_pred"][p]))
                    for behavior in behavior_scores:
                        behavior_scores[behavior].append(r2_score(target_values[behavior][p],
                            prediction_values[behavior][p], multioutput="variance_weighted"))
                _emit_metric(bootstrap_rows, ctx, item=item, representation=rep,
                    metric="direction_balanced_accuracy", category="accessibility",
                    observed_metric="direction_balanced_accuracy", reference=None, values=ba,
                    unit="balanced_accuracy", n_trials=expected_n_trials, n_samples=n_samples)
                for behavior in ("position", "velocity"):
                    _emit_metric(bootstrap_rows, ctx, item=item, representation=rep,
                        metric=f"{behavior}_r2", category="accessibility", observed_metric=f"{behavior}_r2",
                        reference=None, values=behavior_scores[behavior], unit="R2",
                        n_trials=expected_n_trials, n_samples=n_samples)
            expected_rows = 4 if dataset == "Synthetic" else 3
            cell_rows = bootstrap_rows[block_start:]
            if len(cell_rows) != expected_rows:
                raise AssertionError(f"incomplete bootstrap cell {cell_key}: {len(cell_rows)} rows")
            checkpoint.save(cell_key, cell_rows)
        del data
        if slot_ix % 15 == 0 or slot_ix == len(items):
            print(f"trial bootstrap: single-population slots {slot_ix}/{len(items)} prepared",
                  flush=True)

    # Paired Real A/B consistency. The same sampled trial identities are used for both populations.
    real_by_key = {(i["architecture"], i["objective"], int(i["seed"]), i["population"]): i
                   for i in items if i["dataset"] == "Real"}
    for arch in sorted({i["architecture"] for i in items if i["dataset"] == "Real"}):
        for objective in sorted({i["objective"] for i in items if i["dataset"] == "Real" and i["architecture"] == arch}):
            for seed in de.SEEDS:
                cell_keys = [{"stage": "real_ab", "architecture": arch,
                              "objective": objective, "seed": seed,
                              "representation": rep} for rep in ("raw", "unit")]
                if all(checkpoint.has(key) for key in cell_keys):
                    continue
                ia = real_by_key[(arch, objective, seed, "A_PROXIMAL")]
                ib = real_by_key[(arch, objective, seed, "B_DISTAL")]
                da, db = de.load_item_arrays(ia), de.load_item_arrays(ib)
                idx_a = de.coordinate_lookup(da, np.flatnonzero(core._test_mask(ia, da, "A_PROXIMAL")))
                idx_b = de.coordinate_lookup(db, np.flatnonzero(core._test_mask(ib, db, "B_DISTAL")))
                coords = ctx["support"]["real"]["coordinates_trial_time"]
                coord_positions: dict[int, list[int]] = defaultdict(list)
                coord_list = [(int(trial), int(time)) for trial, time in coords]
                for position, (trial, _time) in enumerate(coord_list):
                    coord_positions[trial].append(position)
                coord_positions = {trial: np.asarray(pos, dtype=np.int64)
                                   for trial, pos in coord_positions.items()}
                common_a = np.asarray([idx_a[c] for c in coord_list], dtype=np.int64)
                common_b = np.asarray([idx_b[c] for c in coord_list], dtype=np.int64)
                for rep in ("raw", "unit"):
                    cell_key = cell_keys[0 if rep == "raw" else 1]
                    if checkpoint.has(cell_key):
                        continue
                    block_start = len(bootstrap_rows)
                    scores = {"procrustes_r2": [], "rsa_spearman": [], "linear_cka": []}
                    for draw in all_draws["Real"]:
                        sampled_positions = np.concatenate([coord_positions[int(tid)] for tid in draw])
                        ax = da[f"embedding_{rep}"][common_a[sampled_positions]]
                        bx = db[f"embedding_{rep}"][common_b[sampled_positions]]
                        scores["procrustes_r2"].append(procrustes_r2(ax, bx))
                        scores["rsa_spearman"].append(sampled_rsa_spearman(ax, bx))
                        scores["linear_cka"].append(linear_cka(ax, bx))
                    pair_item = {**ia, "population": "A_PROXIMAL_vs_B_DISTAL", "dataset": "Real"}
                    for metric in scores:
                        _emit_metric(bootstrap_rows, ctx, item=pair_item, representation=rep,
                            metric=metric, category="cross_population_consistency", observed_metric=metric,
                            reference="matched_A_B", values=scores[metric], unit="score",
                            n_trials=len(coord_positions), n_samples=len(coords))
                    cell_rows = bootstrap_rows[block_start:]
                    if len(cell_rows) != 3:
                        raise AssertionError(f"incomplete bootstrap cell {cell_key}: {len(cell_rows)} rows")
                    checkpoint.save(cell_key, cell_rows)
                print(f"trial bootstrap: Real A/B {arch}/{objective}/seed{seed} done", flush=True)

    # Synthetic true-lag temporal score bootstrap at the frozen +10-bin lag.
    syn_by_key = {(i["architecture"], i["objective"], int(i["seed"]), i["population"]): i
                  for i in items if i["dataset"] == "Synthetic" and i["architecture"] != "pca"}
    syn_lag_coords = np.asarray(ctx["support"]["synthetic"]["lag_common_support"]["coordinates_trial_time"], dtype=np.int64)
    lag_grouped = defaultdict(list)
    for trial, time in syn_lag_coords:
        lag_grouped[int(trial)].append((int(trial), int(time)))
    for arch in sorted({i["architecture"] for i in items if i["dataset"] == "Synthetic" and i["architecture"] != "pca"}):
        for objective in sorted({i["objective"] for i in items if i["dataset"] == "Synthetic" and i["architecture"] == arch}):
            for seed in de.SEEDS:
                cell_keys = [{"stage": "synthetic_lag", "architecture": arch,
                              "objective": objective, "seed": seed,
                              "representation": rep} for rep in ("raw", "unit")]
                if all(checkpoint.has(key) for key in cell_keys):
                    continue
                ia, ib = syn_by_key[(arch, objective, seed, "A")], syn_by_key[(arch, objective, seed, "B")]
                da, db = de.load_item_arrays(ia), de.load_item_arrays(ib)
                la = de.coordinate_lookup(da, np.arange(len(da["trial_id"])))
                lb = de.coordinate_lookup(db, np.arange(len(db["trial_id"])))
                lag_coords = [(int(trial), int(time)) for trial, time in syn_lag_coords]
                lag_positions: dict[int, list[int]] = defaultdict(list)
                for position, (trial, _time) in enumerate(lag_coords):
                    lag_positions[trial].append(position)
                lag_positions = {trial: np.asarray(pos, dtype=np.int64)
                                 for trial, pos in lag_positions.items()}
                common_a = np.asarray([la[c] for c in lag_coords], dtype=np.int64)
                common_b = np.asarray([lb[(c[0], c[1] + 10)] for c in lag_coords], dtype=np.int64)
                for rep in ("raw", "unit"):
                    cell_key = cell_keys[0 if rep == "raw" else 1]
                    if checkpoint.has(cell_key):
                        continue
                    block_start = len(bootstrap_rows)
                    values = []
                    for draw in all_draws["Synthetic"]:
                        sampled_positions = np.concatenate([lag_positions[int(tid)] for tid in draw])
                        idxa, idxb = common_a[sampled_positions], common_b[sampled_positions]
                        values.append(procrustes_r2(db[f"embedding_{rep}"][idxb], da[f"embedding_{rep}"][idxa]))
                    lag_item = {**ia, "population": "A_vs_B", "dataset": "Synthetic"}
                    _emit_metric(bootstrap_rows, ctx, item=lag_item, representation=rep,
                        metric="s_lag_r2_at_true_plus10", category="temporal_fidelity",
                        observed_metric="s_true_plus10", reference="B(t+lag)_vs_A(t)",
                        values=values, unit="Procrustes R2", n_trials=len(lag_grouped),
                        n_samples=len(syn_lag_coords))
                    cell_rows = bootstrap_rows[block_start:]
                    if len(cell_rows) != 1:
                        raise AssertionError(f"incomplete bootstrap cell {cell_key}: {len(cell_rows)} rows")
                    checkpoint.save(cell_key, cell_rows)
                print(f"trial bootstrap: Synthetic lag {arch}/{objective}/seed{seed} done", flush=True)

    stage = Path(tempfile.mkdtemp(prefix=".uncertainty-publish-", dir=output.parent))
    variability_path = stage / "TRAINING_SEED_VARIABILITY.csv"
    bootstrap_path = stage / "TRIAL_BOOTSTRAP_PRIMARY_METRICS.csv"
    summary_path = stage / "UNCERTAINTY_SUMMARY.csv"
    seed_path = stage / "TRIAL_BOOTSTRAP_SEEDS.csv"
    def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if path.exists():
            raise FileExistsError(f"uncertainty artifact exists: {path}")
        temp = path.with_name(f".{path.name}.tmp")
        temp.write_text(_csv_payload(rows), encoding="utf-8", newline="")
        temp.replace(path)
    save_csv(variability_path, seed_variability)
    save_csv(bootstrap_path, bootstrap_rows)
    seed_rows = [{"dataset": d, "replicate": i, "bootstrap_seed": seed,
                  "resampling_unit": "complete held-out trial",
                  "stratification": "none; all held-out trials in one pool; class proportions may vary"}
                 for d in ("Synthetic", "Real") for i, seed in enumerate(all_seeds[d])]
    save_csv(seed_path, seed_rows)
    summary_rows = []
    for (dataset, metric, representation), group in _group_rows(bootstrap_rows, ("dataset", "metric", "representation")):
        ses = np.asarray([float(r["bootstrap_se"]) for r in group], dtype=float)
        widths = np.asarray([float(r["ci_97_5"]) - float(r["ci_2_5"]) for r in group], dtype=float)
        summary_rows.append({"uncertainty_type": "held_out_trial_bootstrap", "dataset": dataset,
            "metric": metric, "representation": representation, "configurations": len(group),
            "median_bootstrap_se": float(np.nanmedian(ses)) if np.isfinite(ses).any() else float("nan"),
            "median_ci_width": float(np.nanmedian(widths)) if np.isfinite(widths).any() else float("nan"),
            "bootstrap_replicates": replicates,
            "interpretation": "sampling uncertainty across held-out trials; conditional on frozen embeddings/probes"})
    for (dataset, metric, representation), group in _group_rows(seed_variability, ("dataset", "metric", "representation")):
        seed_sds = np.asarray([float(r["sd"]) for r in group], dtype=float)
        summary_rows.append({"uncertainty_type": "training_seed_variability", "dataset": dataset,
            "metric": metric, "representation": representation, "configurations": len(group),
            "median_training_seed_sd": float(np.nanmedian(seed_sds)) if np.isfinite(seed_sds).any() else float("nan"),
            "n_training_seed_records": 3, "n_independent_final_seeds": 2,
            "interpretation": "descriptive mixed HPO-selected seed 1101 plus independent final seeds 1201/1301; not a population-level confidence interval"})
    save_csv(summary_path, summary_rows)
    figure_dir = stage / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = _uncertainty_figures(bootstrap_rows, figure_dir)
    outputs = [variability_path, bootstrap_path, summary_path, seed_path, *figure_paths]
    # Publish individual files only after the complete staged branch exists.
    # The pending status remains visible until the final provenance is sealed.
    published_hashes: dict[str, str] = {}
    for staged_file in outputs:
        relative = staged_file.relative_to(stage)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.suffix.lower() == ".csv":
                if de.sha256(destination) != de.sha256(staged_file):
                    raise ValueError(f"existing partial CSV differs from deterministic staged result: {destination}")
            elif destination.stat().st_size == 0:
                raise ValueError(f"existing partial figure is empty: {destination}")
        else:
            staged_file.replace(destination)
        published_hashes[str(destination.relative_to(de.PROJECT))] = de.sha256(destination)
    provenance_path = output / "UNCERTAINTY_PROVENANCE.json"
    if provenance_path.exists():
        raise FileExistsError(f"uncertainty provenance already exists: {provenance_path}")
    provenance = {"provenance_version": "neurobridge_final_downstream_v1",
        "branch": "uncertainty", "parent_hashes": parent_hashes,
        "settings": {"training_seeds": list(de.SEEDS), "bootstrap_replicates": replicates,
                  "synthetic_bootstrap_seed_range": [all_seeds["Synthetic"][0], all_seeds["Synthetic"][-1]],
                  "real_bootstrap_seed_range": [all_seeds["Real"][0], all_seeds["Real"][-1]],
                  "bootstrap_seed_lists": {k: v for k, v in all_seeds.items()},
                  "bootstrap_sampling": "complete held-out trials resampled with replacement, unstratified",
                  "probe_bootstrap_policy": "frozen validation-selected C/alpha and train-fitted probe held fixed; only test trials resampled",
                  "probe_retraining_within_bootstrap": False,
                  "held_out_rows_only": True, "no_encoder_retraining": True,
                  "seed_1101_role": "HPO-selected checkpoint seed; not an independent final refit",
                  "independent_final_training_seeds": [1201, 1301]},
        "no_encoder_training_or_hpo": True,
        "artifacts": published_hashes,
        "extra": {"synthetic_temporal_bootstrap_score": "Procrustes S_lag at known +10 bins only; lag argmax not assigned a CI",
               "real_progress_r2_excluded_from_behavior_primary": True,
               "near_collapse_seed_records_retained": True,
               "training_seed_sd_not_population_CI": True,
               "cell_level_atomic_resume_checkpoint": "used during execution; removed after atomic publication",
               "no_core_metric_modified": True}}
    provenance_payload = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    provenance_tmp = provenance_path.with_name(f".{provenance_path.name}.tmp")
    provenance_tmp.write_text(provenance_payload, encoding="utf-8", newline="")
    provenance_tmp.replace(provenance_path)
    status_path = output / "UNCERTAINTY_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["status"] = "COMPLETE"
    status["bootstrap_status"] = "COMPLETE"
    status["completed_bootstrap_replicates"] = replicates
    status["final_provenance_sha256"] = de.sha256(provenance_path)
    status["published_artifacts"] = sorted(published_hashes)
    status_tmp = status_path.with_name(f".{status_path.name}.tmp")
    status_tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="")
    status_tmp.replace(status_path)
    shutil.rmtree(stage, ignore_errors=True)
    # The final immutable tables, all bootstrap seeds, and provenance now exist.
    # Intermediate cell files are no longer needed after a successful publish.
    try:
        shutil.rmtree(checkpoint.root)
    except OSError:
        pass
    return {"training_seed_variability_rows": len(seed_variability),
            "bootstrap_summary_rows": len(bootstrap_rows), "bootstrap_replicates": replicates,
            "bootstrap_seed_rows": len(seed_rows), "output": str(output),
            "provenance": str(provenance_path)}


def _group_rows(rows: list[dict[str, Any]], fields: tuple[str, ...]):
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(k, "")) for k in fields)].append(row)
    return groups.items()


def _uncertainty_figures(rows: list[dict[str, Any]], figure_dir: Path) -> list[Path]:
    outputs = []
    panels = {
        "Synthetic": [("procrustes_r2", "Z recovery: Procrustes R2"),
                      ("rsa_spearman", "Z recovery: RSA Spearman"),
                      ("condition_balanced_accuracy", "Condition decoding: BA"),
                      ("progress_r2", "Temporal progress: R2"),
                      ("s_lag_r2_at_true_plus10", "A/B S_lag at known +10 ms")],
        "Real": [("procrustes_r2", "A/B Procrustes R2"),
                 ("rsa_spearman", "A/B RSA Spearman"),
                 ("linear_cka", "A/B linear CKA"),
                 ("direction_balanced_accuracy", "Direction decoding: BA"),
                 ("position_r2", "Position R2"), ("velocity_r2", "Velocity R2")],
    }
    seed_colors = {1101: "#0072B2", 1201: "#D55E00", 1301: "#009E73"}
    for dataset, entries in panels.items():
        ncols = 3
        nrows = math.ceil(len(entries) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5.4 * nrows), squeeze=False)
        for ax, (metric, title) in zip(axes.flat, entries):
            group = [r for r in rows if r["dataset"] == dataset and r["metric"] == metric]
            configs = sorted({f"{r['architecture']}|{r['objective']}|{r['population']}" for r in group})
            positions = {name: i for i, name in enumerate(configs)}
            for r in group:
                config = f"{r['architecture']}|{r['objective']}|{r['population']}"
                seed = str(r["training_seed"])
                seed_pos = {"1101": -.18, "1201": 0, "1301": .18}.get(seed, 0)
                rep_pos = -.045 if r["representation"] == "raw" else .045
                x = positions[config] + seed_pos + rep_pos
                y = float(r["observed"])
                lo, hi = float(r["ci_2_5"]), float(r["ci_97_5"])
                color = seed_colors.get(int(seed) if seed.isdigit() else -1, "#555555")
                marker = "o" if r["representation"] == "raw" else "s"
                if np.isfinite(y):
                    if np.isfinite(lo) and np.isfinite(hi):
                        ax.vlines(x, lo, hi, color=color, alpha=.72, linewidth=.8)
                        ax.hlines([lo, hi], x - .018, x + .018, color=color, alpha=.72, linewidth=.8)
                    ax.plot(x, y, marker=marker, color=color, markersize=3.5,
                            alpha=.82, linestyle="none")
            ax.set_title(title)
            ax.set_xticks(range(len(configs)), configs, rotation=82, ha="right", fontsize=7)
            ax.grid(axis="y", alpha=.2)
        for ax in axes.flat[len(entries):]:
            ax.axis("off")
        handles = [plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=seed_colors[1101],
                              label="1101 HPO-selected"),
                   plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=seed_colors[1201],
                              label="1201 final seed"),
                   plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=seed_colors[1301],
                              label="1301 final seed")]
        handles += [plt.Line2D([0],[0], marker="o", color="#444444", label="raw"),
                    plt.Line2D([0],[0], marker="s", color="#444444", label="unit")]
        fig.legend(handles=handles, title="Training seed / embedding", loc="upper center", ncol=5)
        fig.suptitle(f"{dataset}: training-seed observations with held-out trial-bootstrap 95% intervals", y=1.01)
        for suffix in ("png", "pdf"):
            path = figure_dir / f"{dataset.lower()}_seed_dots_trial_bootstrap_ci.{suffix}"
            if path.exists():
                raise FileExistsError(f"figure exists: {path}")
            fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
            outputs.append(path)
        plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    ctx = de.load_core_context()
    # Preserve the exact numerical execution mode used by the core probe fit.
    # A global BLAS thread cap changes float32 Ridge R2 at ~1e-6 despite
    # identical data/splits/alpha, violating the core reproduction gate.
    print(json.dumps(run_uncertainty(ctx, replicates=args.replicates, pilot=args.pilot), indent=2))


if __name__ == "__main__":
    main()
