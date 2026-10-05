"""Additional frozen primary uncertainty rows, separate from sealed bootstrap branch."""
import csv
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
sys.path.insert(0,str(HERE))
import final_gap_closure as base
import final_thesis_downstream_eval as de
import final_thesis_core_metrics as core
import final_thesis_uncertainty as un
from neurobridge.eval.representation import linear_cka

OUT=base.ROOT/"trial_bootstrap_completion_v1"
NREP=1000


def r2_stats(y,pred,trial,ids):
    y=np.asarray(y,float).reshape(len(y),-1)
    pred=np.asarray(pred,float).reshape(len(pred),-1)
    stats=[]
    for tid in ids:
        a=y[trial==tid];b=pred[trial==tid]
        stats.append(np.r_[len(a),a.sum(0),np.square(a).sum(0),np.square(a-b).sum(0)])
    return np.asarray(stats)


def r2_scores(stats,counts):
    s=counts@stats
    d=(s.shape[1]-1)//3
    n=s[:,0];sy=s[:,1:1+d];sy2=s[:,1+d:1+2*d];err=s[:,1+2*d:]
    denom=(sy2-np.square(sy)/n[:,None]).sum(1)
    with np.errstate(divide="ignore",invalid="ignore"):
        result=1-err.sum(1)/denom
    result[denom<=1e-12]=np.nan
    return result


def cka_stats(x,y,trial,ids):
    rows=[]
    for tid in ids:
        a=np.asarray(x[trial==tid],float);b=np.asarray(y[trial==tid],float)
        rows.append(np.r_[len(a),a.sum(0),b.sum(0),(a.T@a).ravel(),(b.T@b).ravel(),(a.T@b).ravel()])
    return np.asarray(rows)


def cka_scores(stats,counts):
    s=counts@stats;n=s[:,0];sx=s[:,1:4];sy=s[:,4:7]
    xx=s[:,7:16].reshape(-1,3,3)-sx[:,:,None]*sx[:,None,:]/n[:,None,None]
    yy=s[:,16:25].reshape(-1,3,3)-sy[:,:,None]*sy[:,None,:]/n[:,None,None]
    xy=s[:,25:34].reshape(-1,3,3)-sx[:,:,None]*sy[:,None,:]/n[:,None,None]
    numer=np.square(xy).sum(axis=(1,2))
    denom=np.sqrt(np.square(xx).sum(axis=(1,2))*np.square(yy).sum(axis=(1,2)))
    with np.errstate(divide="ignore",invalid="ignore"):
        out=numer/denom
    out[denom<=1e-16]=np.nan
    return out


def main():
    if (OUT/"PROVENANCE.json").exists():raise FileExistsError("completion branch sealed")
    ctx=de.load_core_context();ids={};counts={};seed_lists={}
    for dataset in ("Synthetic","Real"):
        item=next(i for i in ctx["index"] if i["dataset"]==dataset and i["architecture"]!="pca")
        data=de.load_item_arrays(item);mask=core._test_mask(item,data,item["population"])
        ids[dataset]=np.unique(data["trial_id"][mask]).astype(int)
        counts[dataset],seed_lists[dataset]=base.counts_for_draws(ids[dataset],base.SEED_START[dataset],NREP)
    rows=[];failures=[]
    for item in ctx["index"]:
        data=de.load_item_arrays(item);dataset=item["dataset"]
        mask=core._test_mask(item,data,item["population"])
        trial=data["trial_id"][mask].astype(int)
        if not np.array_equal(np.unique(trial),ids[dataset]):raise ValueError("held-out IDs differ")
        for rep in base.REPS:
            try:pred=un._fit_probe_predictions(ctx,item,data,rep)
            except Exception as exc:
                failures.append({"trial_id":item["trial_id"],"representation":rep,"error":repr(exc)})
                continue
            targets=( {"progress":("progress_r2",data["progress"][mask],pred["progress_pred"])}
                      if dataset=="Synthetic" else
                      {name:(f"{name}_r2",data[name][mask],pred[f"{name}_pred"])
                       for name in ("position","velocity")})
            for name,(metric,y,p) in targets.items():
                stats=r2_stats(y,p,trial,ids[dataset])
                observed=de._core_metric_value(ctx["metrics"],dataset=dataset,arch=item["architecture"],
                    objective=item["objective"],population=item["population"],seed=item["seed"],
                    representation=rep,category="accessibility",metric=metric)
                direct=r2_scores(stats,np.ones((1,len(stats))))[0]
                if not np.isclose(direct,observed,atol=1e-5,rtol=0):
                    raise AssertionError(f"R2 stats/core mismatch {item['trial_id']} {rep} {metric}: {direct} vs {observed}")
                scores=r2_scores(stats,counts[dataset])
                base.add_summary(rows,{"dataset":dataset,"architecture":item["architecture"],
                    "objective":item["objective"],"population":item["population"],"seed":item["seed"],
                    "trial_id":item["trial_id"],"representation":rep,"metric":metric,
                    "category":"accessibility","n_trials":len(ids[dataset]),
                    "resampling_unit":"complete held-out trial"},observed,scores,NREP,item["embedding_sha256"])
    real={(i["architecture"],i["objective"],i["seed"],i["population"]):i for i in ctx["index"] if i["dataset"]=="Real"}
    coords=np.asarray(ctx["support"]["real"]["coordinates_trial_time"],int)
    if not np.array_equal(np.unique(coords[:,0]),ids["Real"]):raise ValueError("real common support IDs differ")
    for arch,obj,seed,pop in sorted(real):
        if pop!="A_PROXIMAL":continue
        ia,ib=real[(arch,obj,seed,"A_PROXIMAL")],real[(arch,obj,seed,"B_DISTAL")]
        da,db=de.load_item_arrays(ia),de.load_item_arrays(ib)
        la=de.coordinate_lookup(da,np.arange(len(da["trial_id"])))
        lb=de.coordinate_lookup(db,np.arange(len(db["trial_id"])))
        idxa=np.asarray([la[tuple(c)] for c in coords]);idxb=np.asarray([lb[tuple(c)] for c in coords])
        for rep in base.REPS:
            x=da[f"embedding_{rep}"][idxa];y=db[f"embedding_{rep}"][idxb]
            stats=cka_stats(x,y,coords[:,0],ids["Real"])
            direct=cka_scores(stats,np.ones((1,len(stats))))[0]
            expected=de._core_metric_value(ctx["metrics"],dataset="Real",arch=arch,objective=obj,
                population="A_PROXIMAL_vs_B_DISTAL",seed=seed,representation=rep,
                category="cross_population_consistency",metric="linear_cka",reference="matched_A_B")
            if not np.isclose(direct,expected,atol=1e-10) or not np.isclose(direct,linear_cka(x,y),atol=1e-10):
                raise AssertionError("CKA stats/core mismatch")
            scores=cka_scores(stats,counts["Real"])
            base.add_summary(rows,{"dataset":"Real","architecture":arch,"objective":obj,
                "population":"A_PROXIMAL_vs_B_DISTAL","seed":seed,"trial_id":ia["trial_id"]+"|"+ib["trial_id"],
                "representation":rep,"metric":"linear_cka","category":"cross_population_consistency",
                "n_trials":len(ids["Real"]),"resampling_unit":"paired complete held-out trial"},
                expected,scores,NREP,ia["embedding_sha256"]+"|"+ib["embedding_sha256"])
    output=OUT/"TRIAL_BOOTSTRAP_ADDITIONAL_METRICS.csv";fail=OUT/"FAILURES.json"
    base.write_csv(output,rows)
    base.write_once(fail,json.dumps(failures,indent=2)+"\n")
    prov={"created_utc":datetime.now(timezone.utc).isoformat(),"script_sha256":base.digest(Path(__file__)),
        "claim_freeze_sha256":base.digest(base.ROOT/"FINAL_CLAIM_FREEZE.md"),
        "core_provenance_sha256":base.digest(de.CORE/"PROVENANCE.json"),
        "support_sha256":base.digest(de.CORE/"EVALUATION_SUPPORT_MANIFEST.json"),
        "embedding_index_sha256":base.digest(de.CORE/"EMBEDDING_INDEX.csv"),
        "settings":{"replicates":NREP,"seed_lists":seed_lists,"training_seeds":list(de.SEEDS),
            "split":"held_out; frozen train-fitted validation-selected probes","no_encoder_fit":True,
            "remaining_missing":"Synthetic Z Procrustes/RSA and Real A/B RSA Spearman"},
        "artifacts":{str(p.relative_to(PROJECT)):base.digest(p) for p in (output,fail)}}
    base.write_once(OUT/"PROVENANCE.json",json.dumps(prov,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"additional_rows":len(rows),"failures":len(failures)}))


if __name__=="__main__":main()
