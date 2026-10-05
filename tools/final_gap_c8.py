"""Frozen-embedding, train-defined C8 condition-centroid equivariance probe."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
sys.path.insert(0,str(HERE))
import final_thesis_downstream_eval as de
import final_thesis_core_metrics as core

ROOT=PROJECT/"outputs/final_thesis_v1/final_evaluation/gap_closure_v1"
OUT=ROOT/"symmetry_c8"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=payload:raise FileExistsError(f"immutable artifact differs: {path}")
    else:path.write_bytes(payload)


def csv_bytes(rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    h=io.StringIO(newline="");w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
    return h.getvalue().encode()


def centroids(x,labels,mask):
    return np.stack([np.asarray(x[mask & (labels==c)],float).mean(0) for c in range(8)])


def evaluate(train,test):
    """Fit generator only on train centroids; evaluate all nonidentity C8 shifts on test."""
    mean=train.mean(0)
    scale=np.linalg.norm(train-mean)
    if scale<=1e-10: return float("nan"),float("nan"),float("nan")
    a=(train-mean)/scale
    b=np.roll(a,-1,axis=0)
    u,_,v=np.linalg.svd(a.T@b,full_matrices=False)
    generator=u@v
    # Reflection is not a C8 generator: enforce the proper-rotation component.
    if np.linalg.det(generator)<0:
        u[:,-1]*=-1;generator=u@v
    axis_angle=Rotation.from_matrix(generator.T).as_rotvec()
    axis_norm=np.linalg.norm(axis_angle)
    if axis_norm<=1e-10:return float("nan"),float("nan"),float("nan")
    generator=Rotation.from_rotvec(axis_angle/axis_norm*(np.pi/4)).as_matrix().T
    held=(test-mean)/scale
    numerator=0.;denominator=0.;power=np.eye(3)
    for g in range(8):
        target=np.roll(held,-g,axis=0)
        numerator+=np.square(held@power-target).sum()
        denominator+=np.square(target).sum()
        power=power@generator
    score=1-numerator/denominator if denominator>1e-12 else float("nan")
    train_one=1-np.square(a@generator-b).sum()/np.square(b).sum()
    closure=float(np.linalg.norm(np.linalg.matrix_power(generator,8)-np.eye(3)))
    return float(score),float(train_one),closure


def main():
    if (OUT/"PROVENANCE.json").exists():raise FileExistsError("C8 branch already sealed")
    ctx=de.load_core_context();rows=[];failures=[]
    for item in ctx["index"]:
        if item["dataset"]!="Synthetic":continue
        data=de.load_item_arrays(item)
        xlabels=data["labels"].astype(int)
        if set(np.unique(xlabels))!=set(range(8)):raise ValueError("expected exactly eight circular conditions")
        masks=core._split_masks(data,"Synthetic",item["population"])
        test=core._test_mask(item,data,item["population"])
        train=masks["train"]
        if not np.array_equal(test, masks["test"]):
            raise ValueError("C8 test support differs from frozen held-out mask")
        for rep in ("raw","unit"):
            x=data[f"embedding_{rep}"]
            tr=centroids(x,xlabels,train);te=centroids(x,xlabels,test)
            score,train_one,closure=evaluate(tr,te)
            row={k:item.get(k) for k in ("architecture","objective","population","seed","trial_id","fit_status","near_collapse")}
            row.update({"dataset":"Synthetic","representation":rep,"metric":"c8_centroid_equivariance_r2",
                        "held_out_score":score,"train_generator_one_step_r2":train_one,
                        "generator_order8_closure_frobenius":closure,
                        "train_trials":len(np.unique(data["trial_id"][train])),
                        "held_out_trials":len(np.unique(data["trial_id"][test])),
                        "embedding_sha256":item["embedding_sha256"],
                        "evaluation_split":"held_out","generator_fit_split":"train"})
            if not np.isfinite(score):failures.append({**row,"reason":"nonfinite equivariance score"})
            rows.append(row)
    aggregates=[]
    grouped=defaultdict(list)
    for row in rows:
        grouped[(row["architecture"],row["objective"],row["population"],row["representation"])].append(row)
    for (arch,obj,pop,rep),group in sorted(grouped.items()):
        vals=np.asarray([r["held_out_score"] for r in group],float)
        aggregates.append({"architecture":arch,"objective":obj,"population":pop,"representation":rep,
            "metric":"c8_centroid_equivariance_r2","n_seed_records":len(group),
            "seeds":"|".join(str(r["seed"]) for r in group),
            "mean":float(np.nanmean(vals)),"sd":float(np.nanstd(vals,ddof=1)) if len(vals)>1 else np.nan,
            "min":float(np.nanmin(vals)),"max":float(np.nanmax(vals)),
            "interpretation":"descriptive seed variability; seed 1101 HPO-selected"})
    metric=OUT/"C8_EQUIVARIANCE_HELD_OUT.csv";agg=OUT/"C8_SEED_VARIABILITY.csv";fail=OUT/"C8_FAILURES.json"
    once(metric,csv_bytes(rows));once(agg,csv_bytes(aggregates));once(fail,(json.dumps(failures,indent=2)+"\n").encode())
    # Fixed order, all individual seed values visible; PCA is a deterministic reference.
    fig,axes=plt.subplots(1,2,figsize=(15,6),sharey=True)
    groups=sorted(set((r["architecture"],r["objective"]) for r in rows))
    labels=[a if a=="pca" else f"{a} / {o}" for a,o in groups]
    for ax,rep in zip(axes,("raw","unit")):
        for j,(arch,obj) in enumerate(groups):
            for pop,off,color in (("A",-.13,"#2677ae"),("B",.13,"#d46a36")):
                values=[r["held_out_score"] for r in rows if r["architecture"]==arch and r["objective"]==obj
                        and r["population"]==pop and r["representation"]==rep]
                ax.scatter(np.full(len(values),j+off),values,s=21,color=color,alpha=.82,label=pop if j==0 else None)
        ax.axhline(0,color="gray",lw=.8);ax.set_title(rep);ax.set_xticks(range(len(groups)),labels,rotation=65,ha="right")
        ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Held-out C8 centroid equivariance R²")
    axes[0].legend(title="Population");fig.suptitle("Synthetic circular-condition symmetry (train-defined generator)")
    fig.tight_layout();(OUT/"figures").mkdir(parents=True,exist_ok=True)
    figs=[]
    for ext in ("png","pdf"):
        p=OUT/"figures"/f"c8_held_out_seed_dots.{ext}"
        if p.exists():raise FileExistsError(p)
        fig.savefig(p,dpi=300,bbox_inches="tight");figs.append(p)
    plt.close(fig)
    prov={"created_utc":datetime.now(timezone.utc).isoformat(),"script_sha256":sha(__file__),
        "claim_freeze_sha256":sha(ROOT/"FINAL_CLAIM_FREEZE.md"),
        "core_provenance_sha256":sha(de.CORE/"PROVENANCE.json"),
        "embedding_index_sha256":sha(de.CORE/"EMBEDDING_INDEX.csv"),
        "definition":{"group":"C8","action_on_labels":"(c+g) mod 8","generator":"proper orthogonal 3x3 Procrustes train-centroid one-step map",
            "evaluation":"constrain train-derived proper-rotation axis to 45-degree generator; compose powers g=0..7; compare held-out condition centroids; no held-out fitting",
            "score":"1 - summed squared equivariance residual / summed squared target norm",
            "closure_diagnostic":"Frobenius norm of generator^8 - identity",
            "scope":"condition-centroid symmetry, not exact sample-level equivariance",
            "train_seed_1101":"HPO-selected checkpoint, not independent refit"},
        "split":"frozen train generator; frozen held_out evaluation","no_encoder_fit":True,
        "artifacts":{str(p.relative_to(PROJECT)):sha(p) for p in [metric,agg,fail,*figs]}}
    once(OUT/"PROVENANCE.json",(json.dumps(prov,indent=2,sort_keys=True)+"\n").encode())
    print(json.dumps({"rows":len(rows),"aggregates":len(aggregates),"failures":len(failures),"output":str(OUT)}))


if __name__=="__main__":main()
