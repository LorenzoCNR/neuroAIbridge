"""Apply the pre-frozen Holm families to existing immutable permutation results."""
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT=Path(__file__).resolve().parent.parent
ROOT=PROJECT/"outputs/final_thesis_v1/final_evaluation/gap_closure_v1"
BASE=PROJECT/"outputs/final_thesis_v1/final_evaluation"
OUT=ROOT/"claim_inference"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with path.open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))


def write_once(path,content):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=content:raise FileExistsError(path)
    else:path.write_bytes(content)


def csv_content(rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    handle=io.StringIO(newline="");writer=csv.DictWriter(handle,fieldnames=fields)
    writer.writeheader();writer.writerows(rows);return handle.getvalue().encode()


def main():
    source_paths={
        "label":BASE/"null_controls/NULL_LABEL_SHUFFLE.csv",
        "pair":BASE/"null_controls/NULL_AB_TRIAL_PAIRING.csv",
        "temporal":ROOT/"temporal_null/NULL_TEMPORAL_SUMMARY.csv"}
    if (OUT/"PROVENANCE.json").exists():raise FileExistsError("multiplicity branch already sealed")
    families={}
    for source,path in source_paths.items():
        for row in read(path):
            dataset=row["dataset"]
            metric=row["metric"]
            if source=="label" and metric not in ("condition_balanced_accuracy","direction_balanced_accuracy"):
                continue
            if source=="pair" and metric!="procrustes_r2":continue
            if source=="temporal" and metric!="s_true_plus10":continue
            family=f"{dataset}_{source}_{metric}"
            families.setdefault(family,[]).append({"family":family,"source":source,"dataset":dataset,
                "architecture":row["architecture"],"objective":row["objective"],
                "population":row.get("population",row.get("population_pair","A_vs_B")),
                "seed":row.get("seed",row.get("training_seed","")),
                "representation":row["representation"],"metric":metric,
                "observed":row["observed"],"empirical_p":float(row["empirical_p"]),
                "source_sha256":sha(path)})
    out=[];summary=[]
    for family,rows in sorted(families.items()):
        n=len(rows);order=sorted(range(n),key=lambda i:rows[i]["empirical_p"])
        adjusted=0.
        for rank,i in enumerate(order):
            adjusted=max(adjusted,min(1.,(n-rank)*rows[i]["empirical_p"]))
            rows[i]["holm_p_family"]=adjusted
            rows[i]["family_size"]=n
        out.extend(rows)
        summary.append({"family":family,"n_tests":n,"raw_p_lt_0_05":sum(r["empirical_p"]<.05 for r in rows),
            "holm_p_lt_0_05":sum(r["holm_p_family"]<.05 for r in rows),
            "minimum_attainable_p_with_1000_permutations":1/1001,
            "minimum_attainable_holm_p":min(1,n/1001)})
    table=OUT/"HOLM_PRIMARY_NULL_FAMILIES.csv";summ=OUT/"HOLM_FAMILY_SUMMARY.csv"
    write_once(table,csv_content(out));write_once(summ,csv_content(summary))
    prov={"created_utc":datetime.now(timezone.utc).isoformat(),"script_sha256":sha(__file__),
        "claim_freeze_sha256":sha(ROOT/"FINAL_CLAIM_FREEZE.md"),
        "sources":{str(p.relative_to(PROJECT)):sha(p) for p in source_paths.values()},
        "settings":{"method":"Holm step-down within five prespecified dataset/null/primary-metric families",
            "alpha":.05,"no_model_selection":True,"no_encoder_training":True,
            "raw_and_unit":"included together in each family"},
        "artifacts":{str(p.relative_to(PROJECT)):sha(p) for p in (table,summ)}}
    write_once(OUT/"PROVENANCE.json",(json.dumps(prov,indent=2,sort_keys=True)+"\n").encode())
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
