# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 15:59:13 2025

@author: zloll
"""
"""
inspect class constructors
import inspect
from cebra import CEBRA
print(inspect.signature(CEBRA.__init__))

"""


import os
os.getcwd()
i_dir=r"C:\\Users\\loren\\Desktop\\USB_Content\\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\AI_for_all\Neuro_Bridge"
#i_dir=r"C:\Users\zloll\Desktop\Condivisione\Back_USB_05_11\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\AI_for_all\Neuro_Bridge"
os.chdir(i_dir)
import sys
#sys.path.append(r"C:\Users\loren\Desktop\USB_Content\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\AI_for_all\Neuro_Bridge")
sys.path.insert(0, r"C:\Users\loren\Desktop\USB_Content\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\AI_for_all\Neuro_Bridge\src")
sys.path.insert(0,r"C:\Users\zloll\Desktop\Condivisione\Back_USB_05_11\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\AI_for_all\Neuro_Bridge\src")
import sys
sys.path.insert(0, r"C:\Users\loren\Desktop\USB_Content\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\AI_for_all\Neuro_Bridge\src")
import neurobridge

from neurobridge.utils.paths import setup_paths, project_paths


import torch
import random
import joblib
import numpy as np
from pathlib import Path
from neurobridge.utils.paths import setup_paths, project_paths
from neurobridge.utils.project_store import ProjectStore
from neurobridge.data.io import load_data
from neurobridge.data.preprocess import create_trials_id, f_resample  
from neurobridge.models.cebra_adapter import *
from neurobridge.utils.config import *
from neurobridge.utils.debug import explore_obj
from neurobridge.viz.manifold_plots import plot_direction_averaged_embedding

# Random Seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
# Config GPU
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# ---------------- CONFIG ----------------
project_name = "hasson_common_space"
data_file_k  = "dati_mirco_18_03_k"
data_file   = "dati_mirco_18_03_k"
data_format  = "mat"
X_key         = "k_cond2_active_neural"     
y_key         = "k_cond2_active_trial"    
trial_key     = "k_cond2_active_trial_id"
pipe_path="src/neurobridge/models"     
data_top_dir="data"
change_dir=False
trial_length_full=800
steady_length = 0

def main():
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
    data = load_data(default_input_dir, data_file, data_format)
    if data is None:
        raise RuntimeError("Data not found or failed to load.")
    X_full = np.asarray(data[X_key])
    y_full = np.asarray(data[y_key]).flatten().astype(int)
    trial_id_full = np.asarray(data[trial_key]).flatten().astype(int)
        ############ remove records before signal ################
    # could make a function in preprocess
    n_trials = len(y_full) // trial_length_full
       
    # Maschere
    keep_mask_move   = np.ones(len(y_full), dtype=bool)
    keep_mask_steady = np.zeros(len(y_full), dtype=bool)
    c_t_list_full, _, _ = create_trials_id(trial_id_full,y_full)
    for i in range(n_trials):
        start, end = c_t_list_full[i]                   
        s_steady   = start
        e_steady   = start + steady_length          
        keep_mask_move[s_steady:e_steady] = False   
        keep_mask_steady[s_steady:e_steady] = True 
    
    # Applica maschere
    X_move    = X_full[keep_mask_move]
    #X_steady  = X_full[keep_mask_steady]
    y_move    = y_full[keep_mask_move]
    #y_steady  = y_full[keep_mask_steady]
    trial_id_move   = trial_id_full[keep_mask_move]
    #trial_id_steady = trial_id_full[keep_mask_steady]
 
    
    # ---------- eventually resample data ------------- #
    #try:
    methods = {0: "sum", 1: "center"}  
    overlap = 5
    mode = "overlapping"                # o "disjoint"
    normalization = True
    step=10
    c_t_list, trial_len, _ = create_trials_id(trial_id_move,y_move)   

       # c_t: array con indici di start/end per ciascun trial (come da tua logica)
   # except Exception:
    #    # change_idx = np.where(np.diff(trial_id) != 0)[0] + 1
    #    # c_t = np.concatenate([[0], change_idx, [len(trial_id)]])
         # raise
    resampled, new_lengths, new_indices = f_resample(
        datasets=[X_move, y_move],      
        trials=c_t_list,
        step=step,
        overlap=overlap,
        methods=methods,
        mode=mode,
        normalization=normalization
     )
    ### reampled data
    X_res = resampled[0]                             
    y_res = resampled[1].reshape(-1, 1).astype(int) 
    X_=X_res
    y_=y_res
    
    # -----------------------------------------------------#
    
    # --------------------- Prepare Data ---------------------------#
    #print(c_t_list)
    # maybe redundnat
    data_={"X":X_,"y":y_}
    #train_data_=['X', 'y']
    # neural data (full sample)
    #transform_data_=['X']
    model_type="cebra_behavior"
    param_file = Path(project_root)/"configs"/ "cebra_model_params.yaml"
    fixed_params, grid_params = load_model_params(param_file, model_type)
    #eventaully override
    fixed_params['max_iterations'] = 24000
    fixed_params['temperature'] = 1.26
    fixed_params['num_hidden_units'] = 64
  
    model_params=fixed_params
    
    cebra1 = CebraAdapter(model_type=model_type, params=fixed_params)
    
    cebra1.fit(data_['X'],data_['y'])
    X_hat=cebra1.transform(data_['X'])
    y_dir_=data_['y'].flatten()
    
    
    # ------------------------ Dynamic NAme -------------------- #
    met=methods[0]
    # analyze just data post signal
    steady_len_=0
    trial_len_=new_lengths[0][0]
    const_len= True
    original_label_order = np.unique(y_dir_).tolist()
    ### title with orginal data
   # dynamic title
    parts = X_key.split("_")
    prefix = "_".join(parts[:2]) 
    title_= f"{prefix}_{met}_s_{step}_o_{overlap}_nhu_{model_params['num_hidden_units']}_temp{model_params['temperature']}"
    #timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    #title_html = f"{title_base}_{timestamp}.html"

    
        
    # --------------------Sphere PLOT ----------------------------- #
    
    c_s="maroon"
    output_folder = default_output_dir
    results_list=[]
    #title='CEBRA-behavior trained with target label'
    ww=0
    const_len=True
    ### parameters' values to name the plot   
    n_h_u=model_params['num_hidden_units']
    temp=model_params['temperature']
    iters=model_params['max_iterations']
    step=step
    overlap=overlap
    plots_dir=  Path(default_output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    
    plot_direction_averaged_embedding(
                    #¶x_x,
                    X_hat,
                    #y_y,
                    y_dir_,
                    original_label_order,
                    c_s,
                    plots_dir,
                    title_,
                    trial_len_,
                    steady_len_,
                    const_len,
                    0) 

 # -------------------------- SAVE Results  -------------------------- #
    models_ = {"cebra1": cebra1}
    results_full = {
            "X_used": X_res,
            "y_used": y_dir_,
            "Z_embedding": X_hat,
            "trial_indices": new_indices,
            "trial_lengths": trial_len_,
        }
    
    models_dir = Path(default_output_dir) / "models"
    embeds_dir = Path(default_output_dir) / "embeds"
    models_dir.mkdir(parents=True, exist_ok=True)
    embeds_dir.mkdir(parents=True, exist_ok=True)
    
    
    joblib.dump(models_, models_dir / f"{title_}_models.pkl")
    joblib.dump(results_full, embeds_dir / f"{title_}_results.pkl")
    
    print(f"Saved: {models_dir / f'{title_}_models.pkl'}")
    print(f"Saved: {embeds_dir / f'{title_}_results.pkl'}")
    
    return X_hat, y_dir_,  trial_len_


if __name__ == "__main__":
    X_hat_out, y_out,trial_len_= main()


# # Path alla cartella dove salvi gli embed
# from pathlib import Path
# from joblib import load
# embeds_dir = Path(r"J:\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\AI_for_all\Neuro_Bridge\outputs\projects\hasson_common_space\embeds")

# # Cerca tutti i file che finiscono con _results.pkl
# result_files = list(embeds_dir.glob("*_results.pkl"))

# # Carica tutti i file in un dizionario
# all_results = {}

# for file in result_files:
#     try:
#         # Usa lo stem del filename come chiave (senza estensione)
#         key = file.stem.replace("_results", "")
#         all_results[key] = load(file)
#         print(f"✔ Caricato: {key}")
#     except Exception as e:
#         print(f"⚠ Errore su {file.name}: {e}")
