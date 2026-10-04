"""Lag-only re-evaluation on a two-sigma search range.

Reuses frozen held-out embeddings from the previous controlled-lag branch.
Never constructs windows, loads raw neural data, initializes an encoder, or
performs optimizer steps.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from neurobridge.experiments.final_real_lag_fit import ARCHITECTURES, OBJECTIVES, SEEDS  # noqa: E402
from tools import final_real_controlled_lag_v2 as parent  # noqa: E402

EVAL_ROOT = Path("outputs/final_thesis_v1/final_evaluation")
PARENT_ROOT = EVAL_ROOT / "controlled_lag_v2"
OUTPUT_ROOT = EVAL_ROOT / "controlled_lag_v3_2sd_corrected"
RANGE_MS = 80
SENSITIVITY_RANGES_MS = (50, 60, 80)
SMOOTHING_SD_MS = 40
SHIFTS_MS = (100, 160, 200)
EXPECTED_SPLIT_SHA = "e1c115122c0424c124e0268bebd2a1d976a081564288538e785269ed9c5a666a"
EXPECTED_PARTITION_SHA = "027ca4b9a6ba53471b72cf93a11e9514968af4399df0dc347de4c44973b98025"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to publish an empty table: {path.name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_core_test_embeddings(test_ids: list[int]) -> tuple[dict, dict, dict[str, str], dict]:
    index_path = ROOT / "outputs/final_thesis_v1/final_evaluation/core_metrics/EMBEDDING_INDEX.csv"
    if not index_path.is_file():
        raise RuntimeError("frozen core embedding index is missing")
    core = parent._load_core_embedding_index(ROOT, {"test": test_ids})
    expected = {
        (pop, arch, obj, seed)
        for pop in ("A_PROXIMAL", "B_DISTAL")
        for arch in ARCHITECTURES for obj in OBJECTIVES for seed in SEEDS
    }
    if set(core) != expected:
        raise RuntimeError(f"expected {len(expected)} frozen A/B slots, found {len(core)}")
    a_rows, b_rows, eligibility = {}, {}, {}
    parent_hashes: dict[str, str] = {"core_embedding_index": sha256(index_path)}
    for key, values in core.items():
        test = parent._core_test_rows(values)
        raw = np.asarray(test["embedding_raw"])
        unit = np.asarray(test["embedding_unit"])
        n = len(test["trial_id"])
        if raw.shape != (n, 3) or unit.shape != (n, 3):
            raise RuntimeError(f"frozen test embedding shape is not N x 3: {key}")
        if not np.isfinite(raw).all() or not np.isfinite(unit).all():
            raise RuntimeError(f"non-finite frozen test embedding: {key}")
        reduced = {
            "embedding_unit": unit,
            "trial_id": np.asarray(test["trial_id"], dtype=np.int64),
            "time_id": np.asarray(test["time_id"], dtype=np.int64),
            "valid_mask": np.ones(n, dtype=bool),
            "split": np.repeat("test", n),
        }
        if set(np.unique(reduced["trial_id"])) != set(test_ids):
            raise RuntimeError(f"test trial set changed in core slot {key}")
        target = a_rows if key[0] == "A_PROXIMAL" else b_rows
        target[key[1:]] = reduced
        eligibility[key] = {
            "fit_status": str(values["fit_status"]),
            "near_collapse": str(values["near_collapse"]).lower() == "true",
        }
        parent_hashes[f"core:{key[0]}:{key[1]}:{key[2]}:{key[3]}:manifest"] = values["parent_manifest_sha256"]
        parent_hashes[f"core:{key[0]}:{key[1]}:{key[2]}:{key[3]}:unit"] = values["parent_unit_sha256"]
        parent_hashes[f"core:{key[0]}:{key[1]}:{key[2]}:{key[3]}:metadata"] = values["parent_metadata_sha256"]
    return a_rows, b_rows, parent_hashes, eligibility


def load_shifted_embedding(path: Path, identity: dict[str, Any], test_ids: list[int]) -> tuple[dict, str, str]:
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"cached shifted embedding is incomplete: {path}")
    manifest = read_json(manifest_path)
    if sha256(path) != manifest.get("artifact_sha256"):
        raise RuntimeError(f"shifted embedding hash mismatch: {path}")
    for key, value in identity.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"shifted embedding identity mismatch ({key}): {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {"embedding_raw", "embedding_unit", "trial_id", "time_id",
                "global_time_id", "valid_mask", "split"}
    if set(arrays) != required:
        raise RuntimeError(f"shifted embedding fields differ from contract: {path}")
    n = len(arrays["trial_id"])
    if (arrays["embedding_raw"].shape != (n, 3) or
            arrays["embedding_unit"].shape != (n, 3) or
            any(len(value) != n for value in arrays.values()) or
            set(np.unique(arrays["split"])) != {"test"} or
            set(np.unique(arrays["trial_id"])) != set(test_ids) or
            not np.isfinite(arrays["embedding_raw"]).all() or
            not np.isfinite(arrays["embedding_unit"]).all()):
        raise RuntimeError(f"shifted test embedding cardinality/content invalid: {path}")
    if not np.array_equal(
        arrays["global_time_id"],
        arrays["trial_id"].astype(np.int64) * 600 + arrays["time_id"].astype(np.int64),
    ):
        raise RuntimeError(f"trial/time coordinate invariant failed: {path}")
    if int(np.asarray(arrays["valid_mask"], dtype=bool).sum()) != int(manifest["valid_count"]):
        raise RuntimeError(f"valid row count differs from manifest: {path}")
    return arrays, sha256(path), sha256(manifest_path)


def candidate_lags(center: int, radius: int = RANGE_MS) -> tuple[int, ...]:
    return tuple(range(center - radius, center + radius + 1))


def common_reference(a: dict, b0: dict, shifted: dict) -> list[tuple[int, int]]:
    a_map = parent._coordinate_index(a)
    b0_map = parent._coordinate_index(b0)
    shifted_maps = {key: parent._coordinate_index(values) for key, values in shifted.items()}
    r0_lags = candidate_lags(0)
    shifted_lags = {key: candidate_lags(key[1]) for key in shifted_maps}
    refs = []
    for trial, time_id in a_map:
        if not all((trial, time_id + lag) in b0_map for lag in r0_lags):
            continue
        if not all(
            (trial, time_id + lag) in shifted_maps[key]
            for key, lags in shifted_lags.items()
            for lag in lags
        ):
            continue
        refs.append((trial, time_id))
    refs.sort()
    if len(refs) < 3:
        raise RuntimeError(f"two-sigma common held-out support is too small: {len(refs)}")
    return refs


def summarize_curve(curve: dict[int, float]) -> dict[str, Any]:
    finite = sorted((int(lag), float(score)) for lag, score in curve.items() if np.isfinite(score))
    if len(finite) < 3:
        raise RuntimeError("lag curve has fewer than three finite scores")
    lag_hat, s_max = max(finite, key=lambda item: item[1])
    next_score = max((score for lag, score in finite if lag != lag_hat), default=float("nan"))
    lo, hi = finite[0][0], finite[-1][0]
    return {
        "lag_hat_ms": lag_hat, "S_max": s_max,
        "peak_margin": float(s_max - next_score),
        "boundary_censored": bool(lag_hat in {lo, hi}),
        "lag_min_ms": lo, "lag_max_ms": hi,
    }


def as_stats(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if not len(finite):
        return {"n": 0, "mean": "", "median": "", "sd": "", "min": "", "max": ""}
    return {
        "n": int(len(finite)), "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "min": float(np.min(finite)), "max": float(np.max(finite)),
    }


def pair_eligibility(*records: dict[str, Any]) -> tuple[str, bool]:
    near_collapse = any(bool(row["near_collapse"]) for row in records)
    eligible = all(row["fit_status"] == "ELIGIBLE" and not row["near_collapse"] for row in records)
    if eligible:
        return "ELIGIBLE", False
    if near_collapse:
        return "INELIGIBLE_NEAR_COLLAPSE", True
    return "INELIGIBLE_PARENT_FIT", False


def plot_curves(curves: list[dict[str, Any]], support_n: int, output: Path) -> list[Path]:
    colors = {100: "#0072B2", 160: "#E69F00", 200: "#009E73"}
    fig, axes = plt.subplots(2, 4, figsize=(19, 8), sharey=True)
    for ax, (architecture, objective) in zip(
        axes.flat, [(a, o) for a in ARCHITECTURES for o in OBJECTIVES]
    ):
        rows = [
            r for r in curves
            if r["architecture"] == architecture and r["objective"] == objective
            and r["fit_status"] == "ELIGIBLE"
        ]
        for branch, label, style in (
            ("R0_baseline", "R0", ":"),
            ("frozen_R0_encoder", "B frozen", "-"),
            ("retrained_B_encoder", "B refit", "--"),
        ):
            shifts = [0] if branch == "R0_baseline" else list(SHIFTS_MS)
            for shift in shifts:
                subset = [r for r in rows if r["branch"] == branch and int(r["shift_ms"]) == shift]
                grouped: dict[int, list[float]] = defaultdict(list)
                for row in subset:
                    grouped[int(row["lag_ms"])].append(float(row["procrustes_r2"]))
                if not grouped:
                    continue
                xs = sorted(grouped)
                means = np.asarray([np.mean(grouped[x]) for x in xs])
                sds = np.asarray([np.std(grouped[x], ddof=1) if len(grouped[x]) > 1 else 0.0 for x in xs])
                color = "#555555" if shift == 0 else colors[shift]
                legend = "R0" if shift == 0 else f"{label} +{shift}"
                ax.plot(xs, means, color=color, linestyle=style, linewidth=1.25, label=legend)
                ax.fill_between(xs, means - sds, means + sds, color=color, alpha=0.09)
        ax.set_title(f"{architecture} / {objective}", fontsize=9)
        ax.set_xlabel("Candidate lag (ms; B(t + lag) vs A(t))")
        ax.grid(alpha=0.22)
    axes[0, 0].set_ylabel("Procrustes R²")
    axes[1, 0].set_ylabel("Procrustes R²")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=7, fontsize=8)
    fig.suptitle(
        f"Real held-out lag curves; ±{RANGE_MS} ms = 2× documented smoothing SD\n"
        f"fit-eligible seed mean ± SD; identical common support N={support_n}",
        y=0.985,
    )
    fig.subplots_adjust(top=0.84, bottom=0.10, left=0.055, right=0.99, hspace=0.31, wspace=0.15)
    paths = []
    for ext, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        path = output / f"REAL_LAG_CURVES_2SD.{ext}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_recovery(summary: list[dict[str, Any]], output: Path) -> list[Path]:
    selected = [r for r in summary if r["branch"] != "R0_baseline" and r["delta_identifiable"]]
    eligible = [r for r in selected if r["delta_fit_eligible"]]
    flagged = [r for r in selected if not r["delta_fit_eligible"]]
    intervention = [r for r in summary if r["branch"] != "R0_baseline"]
    boundary = sum(bool(r["boundary_censored"]) for r in intervention)
    r0 = [r for r in summary if r["branch"] == "R0_baseline"]
    r0_boundary = sum(bool(r["boundary_censored"]) for r in r0)
    unresolved = len(intervention) - len(selected)
    fig, ax = plt.subplots(figsize=(8, 6))
    for branch, marker, color in (
        ("frozen_R0_encoder", "o", "#0072B2"),
        ("retrained_B_encoder", "s", "#D55E00"),
    ):
        subset = [r for r in eligible if r["branch"] == branch]
        ax.scatter(
            [r["expected_shift_ms"] for r in subset],
            [r["delta_lag_hat_ms"] for r in subset],
            marker=marker, color=color, alpha=0.78, label=f"{branch} (fit-eligible)", zorder=3,
        )
    if flagged:
        ax.scatter(
            [r["expected_shift_ms"] for r in flagged],
            [r["delta_lag_hat_ms"] for r in flagged],
            marker="x", color="#666666", alpha=0.9, label="near-collapse/ineligible",
            zorder=4,
        )
    ax.plot([100, 200], [100, 200], color="black", linestyle="--", label="identity")
    ax.set(
        xlabel="Imposed B delay (ms)",
        ylabel="Estimated lag change Δ (ms)",
        title="Controlled temporal-shift response; interior peaks only",
    )
    ax.grid(alpha=0.22)
    ax.legend()
    ax.text(
        0.98, 0.03,
        f"Interior Δ: {len(selected)}/144\nFit-eligible: {len(eligible)}/144\n"
        f"Flagged among interior: {len(flagged)}/144\nUnresolved/censored: {unresolved}/144\n"
        f"Intervention boundary peaks: {boundary}/144\nR0 boundary peaks: {r0_boundary}/24",
        transform=ax.transAxes, ha="right", va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    paths = []
    for ext, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        path = output / f"REAL_LAG_SHIFT_RECOVERY_2SD.{ext}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_sensitivity(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["range_radius_ms"]), row["branch"])].append(row)
    radii = list(SENSITIVITY_RANGES_MS)
    x = np.arange(len(radii), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for offset, (branch, label, color) in enumerate((
        ("frozen_R0_encoder", "B frozen", "#0072B2"),
        ("retrained_B_encoder", "B refit", "#D55E00"),
    )):
        counts = [
            sum(bool(r["delta_fit_eligible"]) for r in grouped[(radius, branch)])
            for radius in radii
        ]
        bars = ax.bar(x + (offset - 0.5) * width, counts, width, color=color, label=label)
        ax.bar_label(bars, padding=2)
    ax.set_xticks(x, [f"±{radius} ms" for radius in radii])
    ax.set_ylim(0, 80)
    ax.set_ylabel("Fit-eligible identifiable Δ estimates (of 72 per branch)")
    ax.set_title("Lag-grid sensitivity; support fixed to the ±80 ms analysis")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    paths = []
    for ext, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        path = output / "LAG_GRID_SENSITIVITY_50_60_80MS"
        path = path.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    if RANGE_MS != 2 * SMOOTHING_SD_MS:
        raise RuntimeError("primary lag range must equal two documented smoothing SDs")
    staging_root = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + ".staging")
    if (ROOT / OUTPUT_ROOT).exists() or (ROOT / staging_root).exists():
        raise FileExistsError("new immutable output/staging path already exists; preserve it")
    parent_spec_path = ROOT / PARENT_ROOT / "CONTROLLED_LAG_V2_SPEC.json"
    test_manifest_path = ROOT / PARENT_ROOT / "inputs/test_only/shift_000ms_B_DISTAL/manifest.json"
    status_path = ROOT / PARENT_ROOT / "REFIT_B_FIT_STATUS.csv"
    prior_curves_path = ROOT / PARENT_ROOT / "REAL_LAG_CURVES_R0_FROZEN_REFIT.csv"
    prior_summary_path = ROOT / PARENT_ROOT / "REAL_LAG_SUMMARY_R0_FROZEN_REFIT.csv"
    for path in (parent_spec_path, test_manifest_path, status_path, prior_curves_path, prior_summary_path):
        if not path.is_file():
            raise RuntimeError(f"required immutable parent missing: {path}")
    spec = read_json(parent_spec_path)
    test_manifest = read_json(test_manifest_path)
    if (
        spec["frozen_split_sha256"] != EXPECTED_SPLIT_SHA
        or spec["somatotopic_partition_sha256"] != EXPECTED_PARTITION_SHA
        or tuple(spec["imposed_B_shifts_ms"]) != SHIFTS_MS
        or test_manifest["original_split_sha256"] != EXPECTED_SPLIT_SHA
        or test_manifest["partition_sha256"] != EXPECTED_PARTITION_SHA
        or test_manifest["test_window_count"] != len(test_manifest["test_trial_ids"]) * 600
    ):
        raise RuntimeError("frozen split/partition/shift parents do not match")
    test_ids = list(map(int, test_manifest["test_trial_ids"]))
    a_rows, b0_rows, parent_hashes, core_eligibility = load_core_test_embeddings(test_ids)
    parent_hashes.update({
        "v2_spec": sha256(parent_spec_path),
        "v2_test_manifest": sha256(test_manifest_path),
        "v2_fit_status": sha256(status_path),
        "v2_per_seed_curves": sha256(prior_curves_path),
        "v2_per_seed_summary": sha256(prior_summary_path),
    })
    core_slots: dict[tuple[str, str, int], dict[str, dict]] = {}
    shifted: dict[tuple[str, int, str, str, int], dict] = {}
    for architecture in ARCHITECTURES:
        for objective in OBJECTIVES:
            for seed in SEEDS:
                key = (architecture, objective, seed)
                core_slots[key] = {
                    "A": a_rows[key], "B0": b0_rows[key],
                    "A_fit": core_eligibility[("A_PROXIMAL", *key)],
                    "B0_fit": core_eligibility[("B_DISTAL", *key)],
                }
                for branch in ("frozen_R0_encoder", "retrained_B_encoder"):
                    for delay in SHIFTS_MS:
                        path = (
                            ROOT / PARENT_ROOT / "embeddings" / branch
                            / f"shift_{delay}ms"
                            / f"{architecture}_{objective}_s{seed}.npz"
                        )
                        identity = {
                            "branch": branch, "shift_ms": delay,
                            "architecture": architecture, "objective": objective,
                            "seed": seed, "population": "B_DISTAL",
                        }
                        arrays, artifact_hash, manifest_hash = load_shifted_embedding(path, identity, test_ids)
                        shifted[(branch, delay, *key)] = arrays
                        parent_hashes[f"{branch}:d{delay}:{architecture}:{objective}:s{seed}:npz"] = artifact_hash
                        parent_hashes[f"{branch}:d{delay}:{architecture}:{objective}:s{seed}:manifest"] = manifest_hash
    del a_rows, b0_rows

    # Masks may differ between R0 and shifted conditions because no-wrap shifts
    # intentionally invalidate different edge rows. Within each exact condition,
    # however, masks must be identical across architecture/objective/seed.
    condition_support: dict[tuple[str, int], set[tuple[int, int]]] = {}
    for key, slots in core_slots.items():
        for name, values in (("A_R0", slots["A"]), ("B_R0", slots["B0"])):
            observed = set(parent._coordinate_index(values))
            if name in condition_support and observed != condition_support[name]:
                raise RuntimeError(f"held-out support differs across model slots for {name}")
            condition_support.setdefault(name, observed)
        for branch in ("frozen_R0_encoder", "retrained_B_encoder"):
            for delay in SHIFTS_MS:
                name = (branch, delay)
                observed = set(parent._coordinate_index(shifted[(branch, delay, *key)]))
                if name in condition_support and observed != condition_support[name]:
                    raise RuntimeError(f"shifted held-out support differs across model slots for {name}")
                condition_support.setdefault(name, observed)
    if not condition_support:
        raise RuntimeError("no frozen A/B support was loaded")

    representative = next(iter(core_slots))
    refs = common_reference(
        core_slots[representative]["A"],
        core_slots[representative]["B0"],
        {
            (branch, delay): shifted[(branch, delay, *representative)]
            for branch in ("frozen_R0_encoder", "retrained_B_encoder")
            for delay in SHIFTS_MS
        },
    )
    counts_by_trial = {
        trial: sum(ref_trial == trial for ref_trial, _ in refs)
        for trial in test_ids
    }
    if len(refs) != 1560 or set(counts_by_trial.values()) != {40}:
        raise RuntimeError(f"unexpected two-sigma common support: {len(refs)} rows")

    with status_path.open(encoding="utf-8-sig", newline="") as stream:
        fit_status = {
            (int(row["shift_ms"]), row["architecture"], row["objective"], int(row["seed"])): row
            for row in csv.DictReader(stream)
        }
    expected_fits = {
        (delay, arch, obj, seed)
        for delay in SHIFTS_MS for arch in ARCHITECTURES
        for obj in OBJECTIVES for seed in SEEDS
    }
    if set(fit_status) != expected_fits:
        raise RuntimeError("fit qualification parent does not enumerate exactly 72 B refits")

    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for (architecture, objective, seed), slots in core_slots.items():
        a = slots["A"]
        r0_curve = parent._score_curve(a, slots["B0"], refs, candidate_lags(0))
        r0_summary = summarize_curve(r0_curve)
        r0_fit_status, r0_near_collapse = pair_eligibility(slots["A_fit"], slots["B0_fit"])
        r0_summary.update({
            "population_A": "A_PROXIMAL", "population_B": "B_DISTAL",
            "architecture": architecture, "objective": objective, "seed": seed,
            "branch": "R0_baseline", "shift_ms": 0, "expected_shift_ms": 0,
            "common_reference_rows": len(refs), "trial_count": len(test_ids),
            "fit_status": r0_fit_status, "near_collapse": r0_near_collapse,
            "A_R0_fit_status": slots["A_fit"]["fit_status"],
            "A_R0_near_collapse": slots["A_fit"]["near_collapse"],
            "B_R0_fit_status": slots["B0_fit"]["fit_status"],
            "B_R0_near_collapse": slots["B0_fit"]["near_collapse"],
            "delta_lag_hat_ms": "", "delta_identifiable": "",
            "lag_grid_radius_ms": RANGE_MS,
        })
        summary_rows.append(r0_summary)
        curve_rows.extend({
            "architecture": architecture, "objective": objective, "seed": seed,
            "branch": "R0_baseline", "shift_ms": 0, "lag_ms": lag,
            "procrustes_r2": score, "common_reference_rows": len(refs),
            "fit_status": r0_fit_status, "near_collapse": r0_near_collapse,
        } for lag, score in sorted(r0_curve.items()))

        for branch in ("frozen_R0_encoder", "retrained_B_encoder"):
            for delay in SHIFTS_MS:
                values = shifted[(branch, delay, architecture, objective, seed)]
                curve = parent._score_curve(a, values, refs, candidate_lags(delay))
                fitted = fit_status[(delay, architecture, objective, seed)]
                summary = summarize_curve(curve)
                delta_identifiable = not r0_summary["boundary_censored"] and not summary["boundary_censored"]
                shifted_fit = {
                    "fit_status": fitted["status"],
                    "near_collapse": fitted["near_collapse"].lower() == "true",
                }
                paired_fits = (
                    (slots["A_fit"], slots["B0_fit"])
                    if branch == "frozen_R0_encoder"
                    else (slots["A_fit"], slots["B0_fit"], shifted_fit)
                )
                pair_status, pair_near_collapse = pair_eligibility(*paired_fits)
                summary.update({
                    "population_A": "A_PROXIMAL", "population_B": "B_DISTAL",
                    "architecture": architecture, "objective": objective, "seed": seed,
                    "branch": branch, "shift_ms": delay, "expected_shift_ms": delay,
                    "common_reference_rows": len(refs), "trial_count": len(test_ids),
                    "fit_status": pair_status,
                    "near_collapse": pair_near_collapse,
                    "A_R0_fit_status": slots["A_fit"]["fit_status"],
                    "A_R0_near_collapse": slots["A_fit"]["near_collapse"],
                    "B_R0_fit_status": slots["B0_fit"]["fit_status"],
                    "B_R0_near_collapse": slots["B0_fit"]["near_collapse"],
                    "B_shift_fit_status": shifted_fit["fit_status"] if branch == "retrained_B_encoder" else "FROZEN_R0_REUSED",
                    "B_shift_near_collapse": shifted_fit["near_collapse"] if branch == "retrained_B_encoder" else slots["B0_fit"]["near_collapse"],
                    "r0_lag_hat_ms": r0_summary["lag_hat_ms"],
                    "r0_boundary_censored": r0_summary["boundary_censored"],
                    "delta_lag_hat_ms": summary["lag_hat_ms"] - r0_summary["lag_hat_ms"] if delta_identifiable else "",
                    "delta_identifiable": delta_identifiable,
                    "delta_fit_eligible": delta_identifiable and pair_status == "ELIGIBLE",
                    "lag_grid_radius_ms": RANGE_MS,
                })
                summary_rows.append(summary)
                curve_rows.extend({
                    "architecture": architecture, "objective": objective, "seed": seed,
                    "branch": branch, "shift_ms": delay, "lag_ms": lag,
                    "procrustes_r2": score, "common_reference_rows": len(refs),
                    "fit_status": pair_status, "near_collapse": pair_near_collapse,
                } for lag, score in sorted(curve.items()))

                for radius in SENSITIVITY_RANGES_MS:
                    r0_sub = {lag: value for lag, value in r0_curve.items() if -radius <= lag <= radius}
                    sub = {lag: value for lag, value in curve.items()
                           if delay - radius <= lag <= delay + radius}
                    rs, ss = summarize_curve(r0_sub), summarize_curve(sub)
                    identifiable = not rs["boundary_censored"] and not ss["boundary_censored"]
                    sensitivity_rows.append({
                        "range_radius_ms": radius, "architecture": architecture,
                        "objective": objective, "seed": seed, "branch": branch,
                        "shift_ms": delay, "r0_lag_hat_ms": rs["lag_hat_ms"],
                        "r0_boundary_censored": rs["boundary_censored"],
                        "intervention_lag_hat_ms": ss["lag_hat_ms"],
                        "intervention_boundary_censored": ss["boundary_censored"],
                        "delta_lag_hat_ms": ss["lag_hat_ms"] - rs["lag_hat_ms"] if identifiable else "",
                        "delta_identifiable": identifiable,
                        "delta_fit_eligible": identifiable and pair_status == "ELIGIBLE",
                        "fit_status": pair_status,
                        "near_collapse": pair_near_collapse,
                        "common_reference_rows": len(refs),
                    })

    if len(curve_rows) != 27048 or len(summary_rows) != 168 or len(sensitivity_rows) != 432:
        raise RuntimeError(
            f"unexpected output row counts: curves={len(curve_rows)}, summaries={len(summary_rows)}, "
            f"sensitivity={len(sensitivity_rows)}"
        )

    aggregate_rows = []
    aggregate_keys = {
        (r["architecture"], r["objective"], r["branch"], int(r["shift_ms"]))
        for r in summary_rows
    }
    for architecture, objective, branch, shift in sorted(aggregate_keys):
        group = [r for r in summary_rows if
                 r["architecture"] == architecture and r["objective"] == objective and
                 r["branch"] == branch and int(r["shift_ms"]) == shift]
        resolved = [r for r in group if not r["boundary_censored"]]
        delta_rows = [r for r in group if r["branch"] != "R0_baseline" and r["delta_identifiable"]]
        eligible_resolved = [r for r in resolved if r["fit_status"] == "ELIGIBLE"]
        eligible_delta_rows = [r for r in delta_rows if r["delta_fit_eligible"]]
        scores = as_stats([r["S_max"] for r in group])
        lags = as_stats([r["lag_hat_ms"] for r in resolved])
        eligible_lags = as_stats([r["lag_hat_ms"] for r in eligible_resolved])
        deltas = as_stats([r["delta_lag_hat_ms"] for r in delta_rows])
        eligible_deltas = as_stats([r["delta_lag_hat_ms"] for r in eligible_delta_rows])
        aggregate_rows.append({
            "architecture": architecture, "objective": objective,
            "branch": branch, "shift_ms": shift, "seed_count": len(group),
            "boundary_peak_count": sum(bool(r["boundary_censored"]) for r in group),
            "interior_peak_count": len(resolved), "delta_identifiable_count": len(delta_rows),
            "fit_eligible_interior_peak_count": len(eligible_resolved),
            "fit_eligible_delta_identifiable_count": len(eligible_delta_rows),
            "ineligible_identifiable_count": len(delta_rows) - len(eligible_delta_rows),
            "near_collapse_count": sum(bool(r["near_collapse"]) for r in group),
            "S_max_mean": scores["mean"], "S_max_sd": scores["sd"],
            "interior_lag_n": lags["n"], "interior_lag_mean_ms": lags["mean"],
            "interior_lag_sd_ms": lags["sd"], "interior_lag_min_ms": lags["min"],
            "interior_lag_max_ms": lags["max"],
            "eligible_interior_lag_n": eligible_lags["n"],
            "eligible_interior_lag_mean_ms": eligible_lags["mean"],
            "eligible_interior_lag_sd_ms": eligible_lags["sd"],
            "identifiable_delta_n": deltas["n"], "delta_mean_ms": deltas["mean"],
            "delta_sd_ms": deltas["sd"], "delta_min_ms": deltas["min"],
            "delta_max_ms": deltas["max"],
            "eligible_delta_n": eligible_deltas["n"],
            "eligible_delta_mean_ms": eligible_deltas["mean"],
            "eligible_delta_sd_ms": eligible_deltas["sd"],
            "eligible_delta_min_ms": eligible_deltas["min"],
            "eligible_delta_max_ms": eligible_deltas["max"],
        })

    curve_aggregate_rows = []
    curve_keys = {
        (r["architecture"], r["objective"], r["branch"], int(r["shift_ms"]), int(r["lag_ms"]))
        for r in curve_rows
    }
    for architecture, objective, branch, shift, lag in sorted(curve_keys):
        group = [r for r in curve_rows if
                 r["architecture"] == architecture and r["objective"] == objective and
                 r["branch"] == branch and int(r["shift_ms"]) == shift and int(r["lag_ms"]) == lag]
        stats = as_stats([r["procrustes_r2"] for r in group])
        eligible_group = [r for r in group if r["fit_status"] == "ELIGIBLE"]
        eligible_stats = as_stats([r["procrustes_r2"] for r in eligible_group])
        curve_aggregate_rows.append({
            "architecture": architecture, "objective": objective, "branch": branch,
            "shift_ms": shift, "lag_ms": lag, "seed_count": stats["n"],
            "procrustes_r2_mean": stats["mean"], "procrustes_r2_sd": stats["sd"],
            "near_collapse_seed_count": sum(bool(r["near_collapse"]) for r in group),
            "fit_eligible_seed_count": eligible_stats["n"],
            "fit_eligible_procrustes_r2_mean": eligible_stats["mean"],
            "fit_eligible_procrustes_r2_sd": eligible_stats["sd"],
            "common_reference_rows": len(refs),
        })

    sensitivity_summary = []
    for radius in SENSITIVITY_RANGES_MS:
        for branch in ("frozen_R0_encoder", "retrained_B_encoder"):
            group = [r for r in sensitivity_rows
                     if int(r["range_radius_ms"]) == radius and r["branch"] == branch]
            sensitivity_summary.append({
                "range_radius_ms": radius, "branch": branch, "comparison_count": len(group),
                "delta_identifiable_count": sum(bool(r["delta_identifiable"]) for r in group),
                "delta_unresolved_count": sum(not bool(r["delta_identifiable"]) for r in group),
                "delta_fit_eligible_identifiable_count": sum(bool(r["delta_fit_eligible"]) for r in group),
                "delta_ineligible_identifiable_count": sum(
                    bool(r["delta_identifiable"]) and not bool(r["delta_fit_eligible"]) for r in group
                ),
                "intervention_boundary_peak_count": sum(bool(r["intervention_boundary_censored"]) for r in group),
                "r0_boundary_peak_count": sum(bool(r["r0_boundary_censored"]) for r in group),
            })

    fit_rows = []
    for delay in SHIFTS_MS:
        for architecture in ARCHITECTURES:
            for objective in OBJECTIVES:
                group = [row for key, row in fit_status.items()
                         if key[:3] == (delay, architecture, objective)]
                fit_rows.append({
                    "shift_ms": delay, "architecture": architecture, "objective": objective,
                    "fit_count": len(group),
                    "eligible_count": sum(row["status"] == "ELIGIBLE" for row in group),
                    "near_collapse_count": sum(row["near_collapse"].lower() == "true" for row in group),
                })

    r0_boundary_count = sum(
        bool(r["boundary_censored"]) for r in summary_rows if r["branch"] == "R0_baseline"
    )
    intervention_rows = [r for r in summary_rows if r["branch"] != "R0_baseline"]
    identified = [r for r in intervention_rows if r["delta_identifiable"]]
    eligible_identified = [r for r in identified if r["delta_fit_eligible"]]
    ineligible_identified = [r for r in identified if not r["delta_fit_eligible"]]
    intervention_boundary_count = sum(bool(r["boundary_censored"]) for r in intervention_rows)
    sensitivity_lines = []
    for radius in SENSITIVITY_RANGES_MS:
        by_branch = {
            row["branch"]: row for row in sensitivity_summary
            if int(row["range_radius_ms"]) == radius
        }
        sensitivity_lines.append(
            f"| ±{radius} | "
            f"{by_branch['frozen_R0_encoder']['delta_identifiable_count']}/72; "
            f"{by_branch['frozen_R0_encoder']['delta_fit_eligible_identifiable_count']}/72 | "
            f"{by_branch['retrained_B_encoder']['delta_identifiable_count']}/72; "
            f"{by_branch['retrained_B_encoder']['delta_fit_eligible_identifiable_count']}/72 |"
        )

    stage = ROOT / OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + ".staging")
    stage.mkdir(parents=True)
    (stage / "figures").mkdir()
    tables = {
        "REAL_LAG_CURVES_2SD.csv": curve_rows,
        "REAL_LAG_SUMMARY_2SD.csv": summary_rows,
        "REAL_LAG_SEED_AGGREGATES_2SD.csv": aggregate_rows,
        "REAL_LAG_CURVE_SEED_AGGREGATES_2SD.csv": curve_aggregate_rows,
        "LAG_GRID_SENSITIVITY_50_60_80MS.csv": sensitivity_rows,
        "LAG_GRID_SENSITIVITY_SUMMARY.csv": sensitivity_summary,
        "REFIT_FIT_QUALIFICATION_COUNTS.csv": fit_rows,
    }
    for name, rows in tables.items():
        write_csv(stage / name, rows)

    figure_paths = (
        plot_curves(curve_rows, len(refs), stage / "figures")
        + plot_recovery(summary_rows, stage / "figures")
        + plot_sensitivity(sensitivity_rows, stage / "figures")
    )
    report = [
        "# Real controlled lag: ±2σ re-evaluation",
        "",
        "## What was changed",
        "",
        "The earlier ±20 ms scan is retained unchanged as historical output. This separate branch "
        "recomputes lag curves only, reusing already frozen held-out A/B embeddings. No encoder was "
        "trained or rerun; no embeddings or checkpoints were regenerated.",
        "",
        "The repository dataset-loader documentation states 1-ms bins and Gaussian spike smoothing with "
        "σ=40 ms. The primary scan therefore spans ±80 ms (2σ), sampled every 1 ms: R0 uses -80..+80 ms, "
        "and imposed delays use d+[-80,+80] ms. This radius is a search span, not a confidence interval.",
        "",
        "## Support and protocol",
        "",
        f"- Imposed B shifts: {', '.join(map(str, SHIFTS_MS))} ms; both the frozen-R0 and retrained-B branches are retained.",
        f"- Same-trial, no-wrap matching; common held-out support across every lag, shift and branch: {len(refs)} rows "
        f"({len(test_ids)} test trials × 40 time points).",
        "- The ±50/±60/±80 sensitivity table uses the same conservative support defined by the full ±80 scan.",
        "- Metric: existing unit-embedding Procrustes R² lag profile. These are relative temporal correspondence and digital-intervention response, not biological or causal delays.",
        "",
        "## Results",
        "",
        f"- R0 boundary peaks: {r0_boundary_count}/24.",
        f"- Intervention boundary peaks: {intervention_boundary_count}/144.",
        f"- Δ estimates identifiable only when both R0 and intervention maxima are interior: {len(identified)}/144; "
        f"unresolved/censored: {len(intervention_rows) - len(identified)}/144.",
        f"- Of the {len(identified)} geometrically interior Δ estimates, {len(ineligible_identified)} involve a near-collapse/ineligible parent fit; "
        f"{len(eligible_identified)} are fit-eligible. Geometric identifiability is not the same as fit eligibility.",
        "- Fit eligibility remains unchanged for the B-refit campaign: 72 fits, 67 eligible and 5 near-collapse. "
        "Pair-level eligibility also requires the frozen A and R0-B anchor fits to be eligible.",
        "- Sensitivity counts use the fixed ±80-ms common support; each cell is geometric interior / fit-eligible interior, out of 72:",
        "",
        "| Search radius | B frozen | B refit |",
        "|---:|---:|---:|",
        *sensitivity_lines,
        "",
        "- All per-seed lag scores are retained, including flagged fits; eligibility-aware plots/aggregates exclude flagged fits from eligible summaries.",
        "- Seed mean/SD/range are descriptive for n=3; no inferential p-values or confidence intervals are claimed.",
        "",
        "## Outputs",
        "",
        *[f"- {name}" for name in tables],
        *[f"- figures/{path.name}" for path in figure_paths],
        "- PROVENANCE.json",
        "",
        "## Integrity",
        "",
        "Only frozen embedding artifacts, trial/time/validity metadata, manifests and fit-status records were read. "
        "No raw neural container, behavior labels, optimizer, training routine, HPO result, or core metric table was used "
        "to select lag settings.",
    ]
    (stage / "REAL_CONTROLLED_LAG_2SD_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    output_hashes = {
        str(path.relative_to(stage)).replace("\\", "/"): sha256(path)
        for path in sorted(stage.rglob("*")) if path.is_file()
    }
    loader_path = ROOT / "legacy_ai_for_all/CEBRA/cebra/datasets/monkey_reaching.py"
    provenance = {
        "study_id": "real_controlled_lag_v3_2sd",
        "decision_class": "USER_AUTHORIZED_NEW_LAG_GRID",
        "smoothing_sd_ms": SMOOTHING_SD_MS,
        "primary_radius_ms": RANGE_MS,
        "sensitivity_radii_ms": list(SENSITIVITY_RANGES_MS),
        "resolution_ms": 1,
        "R0_lags_ms": list(candidate_lags(0)),
        "intervention_lags_ms": {str(d): list(candidate_lags(d)) for d in SHIFTS_MS},
        "support_rows": len(refs), "support_trial_count": len(test_ids),
        "support_rows_per_trial": 40,
        "support_rule": "one held-out support shared by every candidate lag, shift, branch, architecture, objective and seed; same-trial only; no wrap",
        "parent_v2_spec_sha256": sha256(parent_spec_path),
        "parent_split_sha256": EXPECTED_SPLIT_SHA,
        "parent_partition_sha256": EXPECTED_PARTITION_SHA,
        "parent_fit_status_sha256": sha256(status_path),
        "parent_artifact_hashes": parent_hashes,
        "smoothing_documentation": {
            "source_file": "monkey_reaching.py (repository dataset loader)",
            "lines": [58, 59], "source_sha256": sha256(loader_path),
            "meaning": "Gaussian kernel standard deviation is 40 ms; two-SD scan radius is 80 ms",
        },
        "fit_count": 72, "training_performed": False,
        "embedding_generation_performed": False,
        "lag_metric_recomputed_on_frozen_embeddings": True,
        "raw_neural_container_loaded": False, "behavior_labels_loaded": False,
        "test_used_for_model_or_hyperparameter_selection": False,
        "old_v2_artifacts_modified_or_deleted": False,
        "row_counts": {
            "curves": len(curve_rows), "summaries": len(summary_rows),
            "seed_aggregates": len(aggregate_rows), "curve_seed_aggregates": len(curve_aggregate_rows),
            "grid_sensitivity": len(sensitivity_rows),
            "grid_sensitivity_summary": len(sensitivity_summary),
        },
        "results": {
            "r0_boundary_peaks": r0_boundary_count, "r0_total": 24,
            "intervention_boundary_peaks": intervention_boundary_count,
            "intervention_total": len(intervention_rows),
            "delta_identifiable": len(identified),
            "delta_fit_eligible": len(eligible_identified),
            "delta_identifiable_but_ineligible": len(ineligible_identified),
            "delta_unresolved": len(intervention_rows) - len(identified),
        },
        "source_script_sha256": sha256(Path(__file__)),
        "output_hashes_before_provenance": output_hashes,
    }
    (stage / "PROVENANCE.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, ROOT / OUTPUT_ROOT)
    print(f"Completed lag-only 2SD re-evaluation: {ROOT / OUTPUT_ROOT}")
    print(
        f"Support={len(refs)} rows; R0 boundary={r0_boundary_count}/24; "
        f"intervention boundary={intervention_boundary_count}/144; "
        f"geometric delta={len(identified)}/144; fit-eligible delta={len(eligible_identified)}/144; "
        f"flagged interior={len(ineligible_identified)}/144."
    )
    print("Training=0; embedding generation=0; old outputs modified/deleted=0.")


if __name__ == "__main__":
    main()
