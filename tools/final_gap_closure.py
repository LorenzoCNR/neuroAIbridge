"""Read-only downstream analyses of frozen NeuroBridge embeddings.

Outputs are confined to the versioned gap_closure_v1 branch. No encoder is fit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
import final_thesis_downstream_eval as de
import final_thesis_core_metrics as core
import final_thesis_uncertainty as un
from neurobridge.eval.representation import procrustes_r2

ROOT = PROJECT / "outputs/final_thesis_v1/final_evaluation/gap_closure_v1"
REPS = ("raw", "unit")
SEED_START = {"Synthetic": 830000, "Real": 840000}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"immutable artifact differs: {path}")
        return
    path.write_text(payload, encoding="utf-8", newline="")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    import io
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    write_once(path, buf.getvalue())


def provenance(stage: Path, ctx: dict, settings: dict, artifacts: list[Path], parents: dict) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": digest(Path(__file__)),
        "claim_freeze_sha256": digest(ROOT / "FINAL_CLAIM_FREEZE.md"),
        "core_provenance_sha256": digest(de.CORE / "PROVENANCE.json"),
        "support_sha256": digest(de.CORE / "EVALUATION_SUPPORT_MANIFEST.json"),
        "embedding_index_sha256": digest(de.CORE / "EMBEDDING_INDEX.csv"),
        "training_seeds": list(de.SEEDS),
        "split": "frozen held_out test; probe parameters from frozen validation selection",
        "settings": settings,
        "parents": parents,
        "artifacts": {str(p.relative_to(PROJECT)): digest(p) for p in artifacts},
        "no_encoder_fit": True,
    }
    write_once(stage / "PROVENANCE.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")


def trial_stats(x: np.ndarray, y: np.ndarray, trial_ids: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Sufficient statistics for the exact centered/scaled 3-D Procrustes R2."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = np.zeros((len(ids), 1 + 3 + 3 + 1 + 1 + 9), float)
    for i, tid in enumerate(ids):
        a, b = x[trial_ids == tid], y[trial_ids == tid]
        out[i] = np.r_[len(a), a.sum(0), b.sum(0), (a*a).sum(), (b*b).sum(), (a.T @ b).ravel()]
    return out


def procrustes_from_stats(stats: np.ndarray, count: np.ndarray) -> np.ndarray:
    s = count @ stats
    n = s[:, 0]
    sx, sy = s[:, 1:4], s[:, 4:7]
    vx = s[:, 7] - np.einsum("ij,ij->i", sx, sx) / n
    vy = s[:, 8] - np.einsum("ij,ij->i", sy, sy) / n
    cross = s[:, 9:].reshape(-1, 3, 3) - sx[:, :, None] * sy[:, None, :] / n[:, None, None]
    numerator = np.linalg.svd(cross, compute_uv=False).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = 2 * numerator / np.sqrt(vx * vy) - 1
    score[(vx <= 1e-16) | (vy <= 1e-16)] = np.nan
    return score


def counts_for_draws(ids: np.ndarray, seed_start: int, nrep: int) -> tuple[np.ndarray, list[int]]:
    seeds = [seed_start + r for r in range(nrep)]
    counts = np.zeros((nrep, len(ids)), np.int16)
    for r, seed in enumerate(seeds):
        draw = np.random.default_rng(seed).integers(len(ids), size=len(ids))
        counts[r] = np.bincount(draw, minlength=len(ids))
    return counts, seeds


def add_summary(rows: list[dict], base: dict, observed: float, samples: np.ndarray,
                nrep: int, source_hashes: str) -> None:
    finite = np.asarray(samples, float)
    finite = finite[np.isfinite(finite)]
    rows.append({**base, "observed": observed,
        "bootstrap_mean": float(finite.mean()) if len(finite) else np.nan,
        "bootstrap_se": float(finite.std(ddof=1)) if len(finite)>1 else np.nan,
        "ci_2_5": float(np.quantile(finite, .025)) if len(finite) else np.nan,
        "ci_97_5": float(np.quantile(finite, .975)) if len(finite) else np.nan,
        "finite_replicates": len(finite), "requested_replicates": nrep,
        "embedding_parent_sha256": source_hashes,
        "status": "OK" if len(finite)>=950 else "LOW_FINITE_REPLICATES"})


def run_bootstrap(ctx: dict, nrep: int) -> dict:
    stage = ROOT / "trial_bootstrap"
    if (stage / "PROVENANCE.json").exists():
        raise FileExistsError("bootstrap branch already sealed")
    ids_by_dataset, counts, seeds = {}, {}, {}
    for dataset in ("Synthetic", "Real"):
        item = next(i for i in ctx["index"] if i["dataset"] == dataset and i["architecture"] != "pca")
        data = de.load_item_arrays(item)
        mask = core._test_mask(item, data, item["population"])
        ids = np.unique(data["trial_id"][mask]).astype(int)
        ids_by_dataset[dataset] = ids
        counts[dataset], seeds[dataset] = counts_for_draws(ids, SEED_START[dataset], nrep)
    rows, failures = [], []
    for item in ctx["index"]:
        data = de.load_item_arrays(item)
        dataset = item["dataset"]
        ids = ids_by_dataset[dataset]
        mask = core._test_mask(item, data, item["population"])
        trial = data["trial_id"][mask].astype(int)
        if not np.array_equal(np.unique(trial), ids):
            raise ValueError("held-out trial set differs across frozen slots")
        label = data["labels"][mask] if dataset == "Synthetic" else data["target"][mask].astype(int)
        for rep in REPS:
            base = {k: item.get(k) for k in ("dataset", "architecture", "objective", "population", "seed", "trial_id", "fit_status", "near_collapse")}
            base.update(representation=rep, resampling_unit="complete held-out trial", n_trials=len(ids))
            try:
                pred = un._fit_probe_predictions(ctx, item, data, rep)["cls_pred"]
            except Exception as exc:
                failures.append({**base, "stage": "probe_reproduction", "error": repr(exc)})
                continue
            # Fixed 8-class definition; missing true classes in a draw are non-finite.
            correct = np.zeros((len(ids), 8), float)
            total = np.zeros_like(correct)
            for j, tid in enumerate(ids):
                for c in range(8):
                    m = (trial == tid) & (label == c)
                    total[j, c] = m.sum()
                    correct[j, c] = np.sum(pred[m] == c)
            den = counts[dataset] @ total
            num = counts[dataset] @ correct
            with np.errstate(divide="ignore", invalid="ignore"):
                scores = np.where(np.all(den>0, axis=1), np.mean(num / den, axis=1), np.nan)
            observed = de._core_metric_value(ctx["metrics"], dataset=dataset, arch=item["architecture"],
                objective=item["objective"], population=item["population"], seed=item["seed"],
                representation=rep, category="accessibility",
                metric="condition_balanced_accuracy" if dataset=="Synthetic" else "direction_balanced_accuracy")
            add_summary(rows, {**base,"metric":"condition_balanced_accuracy" if dataset=="Synthetic" else "direction_balanced_accuracy",
                               "category":"accessibility"}, observed, scores, nrep, item["embedding_sha256"])
    # Real A/B whole-trial paired geometry, with exact frozen common support.
    by_key = {(i["architecture"],i["objective"],i["seed"],i["population"]):i for i in ctx["index"] if i["dataset"]=="Real"}
    coords = np.asarray(ctx["support"]["real"]["coordinates_trial_time"], int)
    if not np.array_equal(np.unique(coords[:,0]), ids_by_dataset["Real"]):
        raise ValueError("Real A/B support trial IDs differ from bootstrap IDs")
    for arch, objective, seed, _ in sorted(by_key):
        if _ != "A_PROXIMAL":
            continue
        ia, ib = by_key[(arch,objective,seed,"A_PROXIMAL")], by_key[(arch,objective,seed,"B_DISTAL")]
        da, db = de.load_item_arrays(ia), de.load_item_arrays(ib)
        la = de.coordinate_lookup(da,np.arange(len(da["trial_id"])))
        lb = de.coordinate_lookup(db,np.arange(len(db["trial_id"])))
        xa = np.asarray([la[tuple(c)] for c in coords]); xb = np.asarray([lb[tuple(c)] for c in coords])
        for rep in REPS:
            x, y = da[f"embedding_{rep}"][xa], db[f"embedding_{rep}"][xb]
            stats = trial_stats(x,y,coords[:,0],ids_by_dataset["Real"])
            if not np.isclose(procrustes_from_stats(stats,np.ones((1,len(stats))))[0],procrustes_r2(x,y),atol=1e-10):
                raise AssertionError("Procrustes sufficient statistics disagree with frozen metric")
            observed = de._core_metric_value(ctx["metrics"],dataset="Real",arch=arch,objective=objective,
                population="A_PROXIMAL_vs_B_DISTAL",seed=seed,representation=rep,
                category="cross_population_consistency",metric="procrustes_r2",reference="matched_A_B")
            direct = procrustes_r2(x,y)
            if not np.isclose(direct,observed,atol=1e-10):
                raise AssertionError("A/B observed Procrustes differs from frozen core")
            scores = procrustes_from_stats(stats,counts["Real"])
            add_summary(rows,{"dataset":"Real","architecture":arch,"objective":objective,"population":"A_PROXIMAL_vs_B_DISTAL",
                "seed":seed,"trial_id":ia["trial_id"]+"|"+ib["trial_id"],"representation":rep,
                "metric":"procrustes_r2","category":"cross_population_consistency","n_trials":len(ids_by_dataset["Real"]),
                "resampling_unit":"paired complete held-out trial"},observed,scores,nrep,
                ia["embedding_sha256"]+"|"+ib["embedding_sha256"])
    # Synthetic +10 score; no new latent source is needed for the A/B temporal claim.
    by_key = {(i["architecture"],i["objective"],i["seed"],i["population"]):i for i in ctx["index"] if i["dataset"]=="Synthetic" and i["architecture"]!="pca"}
    coords = np.asarray(ctx["support"]["synthetic"]["lag_common_support"]["coordinates_trial_time"],int)
    for arch, objective, seed, _ in sorted(by_key):
        if _ != "A":
            continue
        ia, ib = by_key[(arch,objective,seed,"A")],by_key[(arch,objective,seed,"B")]
        da, db = de.load_item_arrays(ia),de.load_item_arrays(ib)
        la = de.coordinate_lookup(da,np.arange(len(da["trial_id"])))
        lb = de.coordinate_lookup(db,np.arange(len(db["trial_id"])))
        xa = np.asarray([la[tuple(c)] for c in coords]); xb = np.asarray([lb[(int(c[0]),int(c[1])+10)] for c in coords])
        for rep in REPS:
            x,y=da[f"embedding_{rep}"][xa],db[f"embedding_{rep}"][xb]
            stats=trial_stats(x,y,coords[:,0],ids_by_dataset["Synthetic"])
            if not np.isclose(procrustes_from_stats(stats,np.ones((1,len(stats))))[0],procrustes_r2(x,y),atol=1e-10):
                raise AssertionError("Synthetic lag sufficient statistics disagree")
            observed=de._core_metric_value(ctx["metrics"],dataset="Synthetic",arch=arch,objective=objective,
                population="A_vs_B",seed=seed,representation=rep,category="temporal_fidelity",
                metric="s_true_plus10",reference="B(t+lag)_vs_A(t)")
            if not np.isclose(procrustes_r2(x,y),observed,atol=1e-10):
                raise AssertionError("Synthetic +10 score differs from frozen core")
            scores=procrustes_from_stats(stats,counts["Synthetic"])
            add_summary(rows,{"dataset":"Synthetic","architecture":arch,"objective":objective,"population":"A_vs_B",
                "seed":seed,"trial_id":ia["trial_id"]+"|"+ib["trial_id"],"representation":rep,
                "metric":"s_true_plus10","category":"temporal_fidelity","n_trials":len(ids_by_dataset["Synthetic"]),
                "resampling_unit":"paired complete held-out trial"},observed,scores,nrep,
                ia["embedding_sha256"]+"|"+ib["embedding_sha256"])
    # Z geometry cannot be recomputed when the exact frozen source is absent.
    missing = PROJECT / core.SYN_RUN / "stage01_data/shared_data.npz"
    if not missing.is_file():
        failures.append({"stage":"synthetic_Z_geometry_bootstrap","status":"MISSING_EXACT_PARENT",
                         "expected_path":str(missing),"required_sha256":ctx["provenance"]["input_artifacts"]["synthetic_data"]["sha256"] if "input_artifacts" in ctx["provenance"] else "see core provenance"})
    out=stage/"TRIAL_BOOTSTRAP_PRIMARY_METRICS.csv"
    write_csv(out,rows)
    fail=stage/"BOOTSTRAP_FAILURES.json"
    write_once(fail,json.dumps(failures,indent=2,sort_keys=True)+"\n")
    seed_path=stage/"TRIAL_BOOTSTRAP_SEEDS.csv"
    write_csv(seed_path,[{"dataset":d,"replicate":r,"seed":seed,"resampling_unit":"complete held-out trial"}
        for d in ("Synthetic","Real") for r,seed in enumerate(seeds[d])])
    status=stage/"UNCERTAINTY_STATUS.json"
    write_once(status,json.dumps({"status":"PARTIAL_MISSING_SYNTHETIC_Z_PARENT" if failures else "COMPLETE",
        "replicates":nrep,"completed_rows":len(rows),"failures":len(failures),
        "training_seed_variability":"separate existing uncertainty/TRAINING_SEED_VARIABILITY.csv",
        "existing_uncertainty_status_unchanged":True},indent=2)+"\n")
    provenance(stage,ctx,{"replicates":nrep,"seeds":seeds,"resampling":"unstratified complete trial",
        "missing_class_policy":"nonfinite replicate, not silently redefined balanced accuracy"},
        [out,fail,seed_path,status],{"existing_seed_variability_sha256":digest(de.EVAL/"uncertainty/TRAINING_SEED_VARIABILITY.csv")})
    return {"rows":len(rows),"failures":len(failures),"status":"PARTIAL" if failures else "COMPLETE"}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("bootstrap",))
    parser.add_argument("--replicates",type=int,default=1000)
    args=parser.parse_args()
    ctx=de.load_core_context()
    if args.stage=="bootstrap":
        print(json.dumps(run_bootstrap(ctx,args.replicates),indent=2))


if __name__=="__main__":
    main()
