# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 20:27:04 2025

@author: zloll
"""

import os
os.chdir(r'D:\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\AI_for_all\Neuro_Bridge')
import torch
import warnings   # <── aggiungi questa riga

import random
import joblib
from pathlib import Path
from src.neurobridge.utils.paths import setup_paths, project_paths
from src.neurobridge.utils.project_store import ProjectStore
from src.neurobridge.data.io import load_data
from src.neurobridge.data.preprocess import create_trials_id, f_resample  
from src.neurobridge.models.cebra_adapter import *
from src.neurobridge.utils.config import *
from src.neurobridge.utils.debug import explore_obj
from src.neurobridge.viz.manifold_plots import plot_direction_averaged_embedding
from pathlib import Path
from joblib import load

os.getcwd()

# ---------------- CONFIG Subjects ---------
project_name = "hasson_common_space"
pipe_path="src/neurobridge/models"     
data_top_dir="data"
change_dir=False
trial_length_full=599
steady_length = 199

def main():
    from cebra import CEBRA
    sub_data_rel, out_dir_rel = project_paths(project_name)
    project_root, pipeline_path, default_output_dir, default_input_dir = setup_paths(
        data_dir=data_top_dir,
        sub_data=sub_data_rel,
        out_dir=out_dir_rel,
        pipe_path=pipe_path,
        change_dir=False, 
    )

    print(f"Project root      : {project_root}")
    print(f"Pipeline path     : {pipeline_path}")
    print(f"Default output dir: {default_output_dir}")
    print(f"Default input dir : {default_input_dir}")
    
    models_dir = Path(default_output_dir) / "models"
    embeds_dir = Path(default_output_dir) / "embeds"
    string_match = "_MS_"
    
    result_files = [f for f in embeds_dir.glob("*.pkl") if string_match in f.name]
    
    if not result_files:
        print(f"⚠ Nessun file '{string_match}' trovato in:")
        print("   •", models_dir.resolve())
        print("   •", embeds_dir.resolve())
        # opzionale: elenco rapido contenuti
        if models_dir.exists():
            print("\n[DIAG] Contenuto models_dir (prime 10 voci):")
            for f in list(models_dir.glob("*"))[:10]:
                print("   ", f.name)
        if embeds_dir.exists():
            print("\n[DIAG] Contenuto embeds_dir (prime 10 voci):")
            for f in list(embeds_dir.glob("*"))[:10]:
                print("   ", f.name)
        # qui RITORNA davvero (solo se non hai trovato nulla)
        return {}, embeds_dir

    print(f"\n📦 Trovati {len(result_files)} file multi-session:")
    for f in result_files:
        print("   •", f.name)

    all_results = {}
    for file in result_files:
        try:
            key = file.stem  # mantieni il nome-file (senza .pkl) come chiave
            all_results[key] = load(file)
            print(f"✔ Caricato: {key}")
        except Exception as e:
            print(f"⚠ Errore su {file.name}: {e}")
            
        out_mat = embeds_dir / "results_08_10_25.mat"
        savemat(str(out_mat), all_results)
        print(f"💾 File salvato in: {out_mat.resolve()}")
                   
                   

    return all_results, models_dir


if __name__ == "__main__": 
    results_, mod_dir = main()
    print(f"\nTotale modelli caricati: {len(results_)}")
 
def flatten_nested_dict(d: dict, sep="_") -> dict:
    """Appiattisce un dizionario annidato in chiavi concatenate."""
    flat = {}
    for outer_key, inner_dict in d.items():
        if isinstance(inner_dict, dict):
            for inner_key, value in inner_dict.items():
                print(inner_key)
                #new_key = f"{inner_key}{sep}{outer_key}"
                #f<lat[new_key] = value
        else:
            flat[outer_key] = inner_dict
    return flat




#results_full={}
# results_full['k_cond_2_neural_active']=results_["k_cond2_MS_sum_s_10_o_8_nhu_64_temp1_results"]['X_used']
# results_full['k_cond_2_label_active']=results_["k_cond2_MS_sum_s_10_o_8_nhu_64_temp1_results"]['y_used']
# results_full['k_cond_2_embed_active']=results_["k_cond2_MS_sum_s_10_o_8_nhu_64_temp1_results"]['Z_embedding'] 
                                                                                         
savemat("results_08102025.mat",results_full)


    
    

