"""Resumable exact-definition Real A/B RSA trial bootstrap (frozen embeddings only)."""
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
import final_thesis_uncertainty as un

OUT=base.ROOT/"trial_bootstrap_real_rsa_v1"
NREP=1000


def main():
    if (OUT/"PROVENANCE.json").exists():raise FileExistsError("RSA bootstrap branch sealed")
    ctx=de.load_core_context()
    index={(i["architecture"],i["objective"],i["seed"],i["population"]):i for i in ctx["index"] if i["dataset"]=="Real"}
    coords=np.asarray(ctx["support"]["real"]["coordinates_trial_time"],int)
    ids=np.unique(coords[:,0]);by_trial=[np.flatnonzero(coords[:,0]==tid) for tid in ids]
    if len(set(map(len,by_trial)))!=1:raise ValueError("unequal support lengths require explicit resampling audit")
    draws=[]
    for r in range(NREP):
        draws.append(np.random.default_rng(840000+r).integers(len(ids),size=len(ids)))
    checkpoint=OUT/"cells";checkpoint.mkdir(parents=True,exist_ok=True)
    parent={"core_provenance":base.digest(de.CORE/"PROVENANCE.json"),
        "support":base.digest(de.CORE/"EVALUATION_SUPPORT_MANIFEST.json"),
        "embedding_index":base.digest(de.CORE/"EMBEDDING_INDEX.csv"),
        "script":base.digest(Path(__file__)),"claim_freeze":base.digest(base.ROOT/"FINAL_CLAIM_FREEZE.md")}
    state=OUT/"WORK_STATE.json"
    state_payload=json.dumps({"parent_hashes":parent,"replicates":NREP,"seed_start":840000,
        "split":"held_out","metric":"frozen sampled-pair RSA Spearman","representation":["raw","unit"]},sort_keys=True,indent=2)+"\n"
    base.write_once(state,state_payload)
    rows=[];done=0
    for arch,obj,seed,pop in sorted(index):
        if pop!="A_PROXIMAL":continue
        ia,ib=index[(arch,obj,seed,"A_PROXIMAL")],index[(arch,obj,seed,"B_DISTAL")]
        da,db=de.load_item_arrays(ia),de.load_item_arrays(ib)
        la=de.coordinate_lookup(da,np.arange(len(da["trial_id"])))
        lb=de.coordinate_lookup(db,np.arange(len(db["trial_id"])))
        idxa=np.asarray([la[tuple(c)] for c in coords]);idxb=np.asarray([lb[tuple(c)] for c in coords])
        for rep in ("raw","unit"):
            key=f"{arch}__{obj}__s{seed}__{rep}"
            path=checkpoint/f"{key}.json"
            if path.exists():
                row=json.loads(path.read_text())
                if row["A_embedding_sha256"]!=ia["embedding_sha256"] or row["B_embedding_sha256"]!=ib["embedding_sha256"]:
                    raise ValueError("RSA checkpoint embedding parent differs")
                rows.append(row);done+=1;continue
            x=da[f"embedding_{rep}"][idxa];y=db[f"embedding_{rep}"][idxb]
            expected=de._core_metric_value(ctx["metrics"],dataset="Real",arch=arch,objective=obj,
                population="A_PROXIMAL_vs_B_DISTAL",seed=seed,representation=rep,
                category="cross_population_consistency",metric="rsa_spearman",reference="matched_A_B")
            direct=un.sampled_rsa_spearman(x,y)
            if not np.isclose(direct,expected,atol=1e-12):raise AssertionError("observed RSA differs from frozen core")
            scores=[]
            for draw in draws:
                positions=np.concatenate([by_trial[j] for j in draw])
                scores.append(un.sampled_rsa_spearman(x[positions],y[positions]))
            finite=np.asarray(scores,float);finite=finite[np.isfinite(finite)]
            row={"dataset":"Real","architecture":arch,"objective":obj,"population":"A_PROXIMAL_vs_B_DISTAL",
                "seed":seed,"representation":rep,"metric":"rsa_spearman","observed":expected,
                "bootstrap_mean":float(finite.mean()),"bootstrap_se":float(finite.std(ddof=1)),
                "ci_2_5":float(np.quantile(finite,.025)),"ci_97_5":float(np.quantile(finite,.975)),
                "finite_replicates":len(finite),"requested_replicates":NREP,
                "resampling_unit":"paired complete held-out trial","n_trials":len(ids),
                "A_embedding_sha256":ia["embedding_sha256"],"B_embedding_sha256":ib["embedding_sha256"]}
            base.write_once(path,json.dumps(row,sort_keys=True)+"\n")
            rows.append(row);done+=1
            if done%8==0:print(f"Real RSA bootstrap {done}/48 cells",flush=True)
    rows.sort(key=lambda r:(r["architecture"],r["objective"],r["seed"],r["representation"]))
    output=OUT/"REAL_AB_RSA_TRIAL_BOOTSTRAP.csv";base.write_csv(output,rows)
    seed_path=OUT/"BOOTSTRAP_SEEDS.csv"
    base.write_csv(seed_path,[{"replicate":r,"seed":840000+r} for r in range(NREP)])
    prov={"created_utc":datetime.now(timezone.utc).isoformat(),"parent_hashes":parent,
        "settings":{"replicates":NREP,"seed_start":840000,"training_seeds":list(de.SEEDS),
            "split":"frozen held_out","representation":["raw","unit"],
            "metric":"core sampled-pair RSA Spearman, 100k pairs, random_state 0",
            "resampling":"same complete trial IDs with replacement for A and B","no_encoder_fit":True},
        "artifacts":{str(p.relative_to(PROJECT)):base.digest(p) for p in [output,seed_path,*checkpoint.glob("*.json")]}}
    base.write_once(OUT/"PROVENANCE.json",json.dumps(prov,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"rows":len(rows),"replicates_each":NREP,"output":str(OUT)}))


if __name__=="__main__":main()
