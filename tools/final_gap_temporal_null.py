"""Prespecified within-trial temporal-block null on frozen Synthetic A/B embeddings."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
sys.path.insert(0,str(HERE))
import final_thesis_downstream_eval as de
from neurobridge.eval.representation import procrustes_r2

ROOT=PROJECT/"outputs/final_thesis_v1/final_evaluation/gap_closure_v1"
OUT=ROOT/"temporal_null"
B=1000
BLOCK=10
SEED_START=850000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=payload:
            raise FileExistsError(f"immutable artifact differs: {path}")
    else:
        path.write_bytes(payload)


def csv_bytes(rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    handle=io.StringIO(newline="")
    writer=csv.DictWriter(handle,fieldnames=fields)
    writer.writeheader();writer.writerows(rows)
    return handle.getvalue().encode()


def permutations(n_trials, n_blocks):
    result=np.empty((B,n_trials,n_blocks),dtype=np.int16)
    for r in range(B):
        rng=np.random.default_rng(SEED_START+r)
        for t in range(n_trials):
            while True:
                p=rng.permutation(n_blocks)
                if np.all(p!=np.arange(n_blocks)):
                    break
            result[r,t]=p
    return result


def null_scores(x,y,perms):
    _,n_trials,n_blocks=perms.shape
    x=np.asarray(x,float).reshape(n_trials,n_blocks,BLOCK,3)
    y=np.asarray(y,float).reshape(n_trials,n_blocks,BLOCK,3)
    x=(x-x.mean(axis=(0,1,2),keepdims=True));x=x/np.linalg.norm(x)
    y=(y-y.mean(axis=(0,1,2),keepdims=True));y=y/np.linalg.norm(y)
    cross=np.einsum("tbkd,tjke->tbjde",x,y,optimize=True)
    scores=np.empty(len(perms),float)
    for r,p in enumerate(perms):
        mat=cross[np.arange(n_trials)[:,None],np.arange(n_blocks)[None,:],p].sum(axis=(0,1))
        scores[r]=2*np.linalg.svd(mat,compute_uv=False).sum()-1
    return scores


def main():
    if (OUT/"PROVENANCE.json").exists():
        raise FileExistsError("temporal null branch already sealed")
    ctx=de.load_core_context()
    coords=np.asarray(ctx["support"]["synthetic"]["lag_common_support"]["coordinates_trial_time"],int)
    trial_ids=np.unique(coords[:,0]);n_trials=len(trial_ids)
    row_counts=np.asarray([np.sum(coords[:,0]==t) for t in trial_ids])
    if len(set(row_counts))!=1 or row_counts[0]%BLOCK:
        raise ValueError("frozen support is incompatible with fixed 10-bin blocks")
    n_blocks=int(row_counts[0]//BLOCK)
    perms=permutations(n_trials,n_blocks)
    seed_rows=[{"replicate":r,"seed":SEED_START+r,
        "permutation_sha256":hashlib.sha256(perms[r].tobytes()).hexdigest()} for r in range(B)]
    index={(i["architecture"],i["objective"],i["seed"],i["population"]):i for i in ctx["index"]
           if i["dataset"]=="Synthetic" and i["architecture"]!="pca"}
    summaries=[];replicates=[];figure_data=None
    for arch,objective,seed,pop in sorted(index):
        if pop!="A":continue
        ia,ib=index[(arch,objective,seed,"A")],index[(arch,objective,seed,"B")]
        da,db=de.load_item_arrays(ia),de.load_item_arrays(ib)
        la=de.coordinate_lookup(da,np.arange(len(da["trial_id"])))
        lb=de.coordinate_lookup(db,np.arange(len(db["trial_id"])))
        idx_a=np.asarray([la[tuple(c)] for c in coords]);idx_b=np.asarray([lb[(int(c[0]),int(c[1])+10)] for c in coords])
        if not (np.all(da["split"][idx_a].astype(str)=="test") and np.all(db["split"][idx_b].astype(str)=="test")):
            raise ValueError("temporal null escaped held-out support")
        if not (np.all(da["valid_mask"][idx_a]) and np.all(db["valid_mask"][idx_b])):
            raise ValueError("temporal null uses invalid embedding")
        for rep in ("raw","unit"):
            x=da[f"embedding_{rep}"][idx_a];y=db[f"embedding_{rep}"][idx_b]
            observed=procrustes_r2(x,y)
            core_value=de._core_metric_value(ctx["metrics"],dataset="Synthetic",arch=arch,objective=objective,
                population="A_vs_B",seed=seed,representation=rep,category="temporal_fidelity",
                metric="s_true_plus10",reference="B(t+lag)_vs_A(t)")
            if not np.isclose(observed,core_value,atol=1e-10):
                raise AssertionError("observed temporal statistic differs from frozen core")
            scores=null_scores(x,y,perms)
            base={"dataset":"Synthetic","architecture":arch,"objective":objective,"seed":seed,
                  "population":"A_vs_B","representation":rep,"metric":"s_true_plus10",
                  "A_embedding_sha256":ia["embedding_sha256"],"B_embedding_sha256":ib["embedding_sha256"]}
            summaries.append({**base,"observed":observed,"null_mean":float(scores.mean()),
                "null_sd":float(scores.std(ddof=1)),"effect":float(observed-scores.mean()),
                "empirical_p":float((1+np.sum(scores>=observed))/(B+1)),"replicates":B,
                "permutation_unit":"10-bin B blocks within each matched trial; deranged; A fixed"})
            replicates.extend({**base,"replicate":r,"seed":SEED_START+r,"null_score":float(value)}
                              for r,value in enumerate(scores))
            if figure_data is None and rep=="unit":figure_data=(base,observed,scores)
        print(f"temporal null {arch}/{objective}/seed{seed}",flush=True)
    # Holm family correction over all 48 raw/unit architecture-objective-seed tests.
    ordered=sorted(range(len(summaries)),key=lambda i:summaries[i]["empirical_p"])
    prev=0.0
    for rank,i in enumerate(ordered):
        prev=max(prev,min(1.0,(len(ordered)-rank)*summaries[i]["empirical_p"]))
        summaries[i]["holm_p_all_raw_unit_models"]=prev
    sum_path=OUT/"NULL_TEMPORAL_SUMMARY.csv";rep_path=OUT/"NULL_TEMPORAL_REPLICATES.csv";seed_path=OUT/"NULL_TEMPORAL_SEEDS.csv"
    once(sum_path,csv_bytes(summaries));once(rep_path,csv_bytes(replicates));once(seed_path,csv_bytes(seed_rows))
    figpaths=[]
    if figure_data:
        base,obs,scores=figure_data
        fig,ax=plt.subplots(figsize=(7,4));ax.hist(scores,bins=35,color="#6c8caf",alpha=.8)
        ax.axvline(obs,color="#c34432",linewidth=2,label="observed +10")
        ax.set(xlabel="Procrustes R² at frozen +10-bin shift",ylabel="Temporal-order null replicates",
               title=f"Synthetic temporal null: {base['architecture']} / {base['objective']} / seed {base['seed']}")
        ax.legend();fig.tight_layout();(OUT/"figures").mkdir(parents=True,exist_ok=True)
        for ext in ("png","pdf"):
            p=OUT/"figures"/f"representative_temporal_null.{ext}"
            if p.exists():raise FileExistsError(p)
            fig.savefig(p,dpi=300);figpaths.append(p)
        plt.close(fig)
    prov={"created_utc":datetime.now(timezone.utc).isoformat(),"script_sha256":sha(__file__),
        "claim_freeze_sha256":sha(ROOT/"FINAL_CLAIM_FREEZE.md"),"core_provenance_sha256":sha(de.CORE/"PROVENANCE.json"),
        "support_sha256":sha(de.CORE/"EVALUATION_SUPPORT_MANIFEST.json"),
        "settings":{"replicates":B,"seed_start":SEED_START,"block_size":BLOCK,"blocks_per_trial":n_blocks,
                    "trial_count":n_trials,"shift":10,"representation":["raw","unit"],
                    "unit":"deranged 10-bin B blocks independently within each same-identity trial",
                    "split":"held_out","primary_statistic":"s_true_plus10","no_encoder_fit":True},
        "artifacts":{str(p.relative_to(PROJECT)):sha(p) for p in [sum_path,rep_path,seed_path,*figpaths]}}
    once(OUT/"PROVENANCE.json",(json.dumps(prov,indent=2,sort_keys=True)+"\n").encode())
    print(json.dumps({"comparisons":len(summaries),"replicates_each":B,"output":str(OUT)}))


if __name__=="__main__":main()
