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
os.chdir(r'J:\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\AI_for_all\Neuro_Bridge')
import torch
import warnings   # <── aggiungi questa riga

import random
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

from src.neurobridge.utils.paths import setup_paths, project_paths
from src.neurobridge.utils.project_store import ProjectStore
from src.neurobridge.data.io import load_data
from src.neurobridge.data.preprocess import create_trials_id, f_resample  
from src.neurobridge.models.cebra_adapter import *
from src.neurobridge.utils.config import *
from src.neurobridge.utils.debug import explore_obj
from src.neurobridge.viz.manifold_plots import plot_direction_averaged_embedding
from src.neurobridge.viz.manifold_plots import plot_datasets_in_groups

# Random Seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
# Config GPU
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# ---------------- CONFIG Subjects ---------
project_name = "cebra_rats_B&G"
# SUBJECTS = {
#     "achilles": {
#         "data_file": "achilles",
       
#     },
  
SUBJECTS = {
    "achilles": {
        "data_file": "achilles",
       
    },
    "cicero": {
        "data_file": "cicero"} , #
    "buddy": {
        "data_file": "buddy"},
   
    "gatsby": {
        "data_file": "gatsby"},
}

pipe_path="src/neurobridge/models"     
data_top_dir="data"
change_dir=False


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
    
    achilles = load_data(default_input_dir, SUBJECTS['achilles']["data_file"],"jl")
    buddy = load_data(default_input_dir, SUBJECTS['buddy']["data_file"],"jl")
    cicero = load_data(default_input_dir, SUBJECTS['cicero']["data_file"],"jl")
    gatsby = load_data(default_input_dir, SUBJECTS['gatsby']["data_file"],"jl")
   #                      if data is None:
   #     raise RuntimeError("Data not found or failed to load.")
   # X_full = np.asarray(data[X_key])
   # y_full = np.asarray(data[y_key]).flatten().astype(int)
   # trial_id_full = np.asarray(data[trial_key]).flatten().astype(int)
    
    

     # --------------------- Single Sessiion ---------------------------#
     #print(c_t_list)
     # maybe redundnat
    rat_dict = {
    "achilles": achilles,
    "buddy": buddy,
    "cicero": cicero,
    "gatsby": gatsby
   }
    rat_results={} 
    for name, rat in rat_dict.items(): 
         print(f"\nProcessing rat: {name}")
     
         data_={"X":rat['spikes'],"y":rat['position']}
         model_type="cebra_behavior"
         param_file = Path(project_root)/"configs"/ "cebra_model_params.yaml"
         fixed_params, grid_params = load_model_params(param_file, model_type)
         #eventaully override
         fixed_params['max_iterations'] = 16000
         fixed_params['num_hidden_units'] = 32
         fixed_params['temperature'] = 1.1
         cebra_ss = CebraAdapter(model_type=model_type, params=fixed_params)
         cebra_ss.fit(data_['X'],data_['y'])
         X_hat=cebra_ss.transform(data_['X'])
         y_dir_=data_['y']
         
         rat_results[name] = {
            "embedding": X_hat,
            "behavior": y_dir_,
            "params": fixed_params,
            "raw_data": rat,   # opzionale: conserva dati grezzi
        }

         
    return rat_results

    
if __name__ == "__main__":
    r_r_32_0_1_1_16000=main()
    
dataset_dict = {name: r["embedding"] for name, r in   r_r_32_1.items()}
label_dict   = {name: r["behavior"] for name, r in   r_r_32_1.items()}


# Generate grouped 3D plots (2 per figure)
plot_datasets_in_groups(
    dataset_dict=dataset_dict,
    label=label_dict,
    group_size=4,
    title= "CEBRA Platform"
)

####•FOR VIDEO rename embediing in dictionary--> X_hat and label/behaviour--> y
###
import sys
sys.path
neurob_path = Path(r"J:\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\AI_for_all\Neuro_Bridge")
if str(neurob_path) not in sys.path:
    sys.path.append(str(neurob_path)) 
    
from Plot_Video_decoding_Rats import *
####•FOR VIDEO rename embediing in dictionary--> X_hat and label/behaviour--> y
### qui lo faccio manualmente
achilles_CEBRA=r_r_32_1['achilles']

achilles_res_CEBRA={}
achilles_res_CEBRA['X_hat']=achilles_CEBRA['embedding']
achilles_res_CEBRA['y']=achilles_CEBRA['behavior']
achilles_res_CEBRA['X']=achilles_CEBRA['raw_data']['spikes']


#writer = FFMpegWriter(fps=15, metadata=dict(artist='NeuroBridge'), bitrate=10000)
#ani.save(str(save_path), writer=writer, dpi=150)
make_video_achilles(
    achilles_res_CEBRA,
    title_track='Decoded (Cebra+KNN)',
    title_scatter='Manifold Cebra',
    animate=True,
    save_path=r"J:\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\AI_for_all\Neuro_Bridge\achilles_video.mp4"
)   

    

 #    # ------------------------ Dynamic NAme Single Session-------------------- #
 #    met=methods[0]
 #    # analyze just data post signal
 #    steady_len_=0
 #    trial_len_=new_lengths[0][0]
 #    const_len= True
 #    original_label_order = np.unique(y_dir_).tolist()
 #    ### title with orginal data
 #   # dynamic title
 #    X_key = SUBJECTS["k"]["X_key"]
 #    parts = X_key.split("_")
 #    prefix = "_".join(parts[:2]) 
 #    title_= f"{prefix}_{met}_s_{step}_o_{overlap}_nhu_{model_params['num_hidden_units']}_temp{model_params['temperature']}"
 #    #timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
 #    #title_html = f"{title_base}_{timestamp}.html"
    
    
    
        
 #    # --------------------Sphere PLOT ----------------------------- #
    
 #    c_s="maroon"
 #    output_folder = default_output_dir
 #    results_list=[]
 #    #title='CEBRA-behavior trained with target label'
 #    ww=0
 #    const_len=True
 #    ### parameters' values to name the plot   
 #    iters=model_params['max_iterations']
 #    plots_dir=  Path(default_output_dir) / "plots"
 #    plots_dir.mkdir(parents=True, exist_ok=True)

 #    plot_direction_averaged_embedding(
 #                    #¶x_x,
 #                    X_hat,
 #                    #y_y,
 #                    y_dir_,
 #                    original_label_order,
 #                    c_s,
 #                    plots_dir,
 #                    title_,
 #                    trial_len_,
 #                    steady_len_,
 #                    const_len,
 #                    0) 

 # # -------------------------- SAVE Results  -------------------------- #
 #    models_ = {"cebra_ss": cebra_ss}
 #    results_ss = {
 #            "X_used": data_['X'],
 #            "y_used":data_['y'],
 #            "Z_embedding": X_hat,
 #            "trial_indices": new_indices,
 #            "trial_lengths": trial_len_,
 #        }
    
 #    models_dir = Path(default_output_dir) / "models"
 #    embeds_dir = Path(default_output_dir) / "embeds"
 #    models_dir.mkdir(parents=True, exist_ok=True)
 #    embeds_dir.mkdir(parents=True, exist_ok=True)
    
    
 #    joblib.dump(models_, models_dir / f"{title_}_models.pkl")
 #    joblib.dump(results_ss, embeds_dir / f"{title_}_results.pkl")
    
 #    print(f"Saved: {models_dir / f'{title_}_models.pkl'}")
 #    print(f"Saved: {embeds_dir / f'{title_}_results.pkl'}")
'''
#===================== CEBRA MULTI-SESSION (K + S) =====================


    
    #new_lengths=len_k
    #new_indices= idx_k
    #train_data_=['X', 'y']
    # neural data (full sample)
    #transform_data_=['X']
    model_type="cebra_behavior"
    param_file = Path(project_root)/"configs"/ "cebra_model_params.yaml"
    fixed_params, grid_params = load_model_params(param_file, model_type)
    #eventaully override
    fixed_params['max_iterations'] =32000
    fixed_params['temperature'] = 1
    fixed_params['num_hidden_units'] = 64
  
    model_params=fixed_params
    cebra_ms = CebraAdapter(model_type=model_type, params=fixed_params)
    # IMPORTANT: pass [Xk, Xs] and [yk, ys] (your earlier order was wrong)
    datas_X=[Xk,Xs]
    datas_y=[yk,ys]
    names_=["k","s"]
    cebra_ms.fit(datas_X, datas_y)
    if getattr(cebra_ms, "_is_multi", False):
        warnings.warn(
            f"⚠️  CEBRA multi-session training active ({cebra_ms._n_sessions} sessions detected).",
            category=UserWarning,
        )
        

       
    multi_embeddings={}
    for i, (name, X) in enumerate(zip(names_, datas_X)):
        multi_embeddings[name] = cebra_ms.transform(X, session_id=i)
    X_hat_k = cebra_ms.transform(Xk, session_id=0)
    y_dir_k = yk.flatten()
    
    # Transform session S reusing the SAME variable names as you did
    X_hat_s = cebra_ms.transform(Xs, session_id=1)
    y_dir_s = ys.flatten()
    
    # ------------------------ Dynamic NAme Multi Session-------------------- #
    models_dir = Path(default_output_dir) / "models"
    embeds_dir = Path(default_output_dir) / "embeds"
    models_dir.mkdir(parents=True, exist_ok=True)
    embeds_dir.mkdir(parents=True, exist_ok=True)
    met=methods[0]
    # analyze just data post signal
    steady_len_=0
    trial_len_k=len_k[0][0]
    const_len= True
    original_label_order_k = np.unique(y_dir_k).tolist()
    ### title with orginal data
   # dynamic title
    X_key_k = SUBJECTS["k"]["X_key"]
    parts_k= X_key_k.split("_")
    prefix_k = "_".join(parts_k[:2]) 
    title_ms_k = f"{prefix_k}_MS_{met}_s_{step}_o_{overlap}_nhu_{model_params['num_hidden_units']}_temp{model_params['temperature']}"    #timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    #title_html = f"{title_base}_{timestamp}.html"
    
    # --------------------Sphere PLOT K----------------------------- #
    
    c_s="maroon"
    output_folder = default_output_dir
    results_list=[]
    #title='CEBRA-behavior trained with target label'
    ww=0
    const_len=True
    ### parameters' values to name the plot   
    iters=model_params['max_iterations']
    plots_dir=  Path(default_output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_direction_averaged_embedding(
                    #¶x_x,
                    X_hat_k,
                    #y_y,
                    y_dir_k,
                    original_label_order_k,
                    c_s,
                    plots_dir,
                    title_ms_k,
                    trial_len_k,
                    steady_len_,
                    const_len,
                    0) 
    
    
    # Save MS K
    models_ms_k = {"cebra_ms": cebra_ms}
    results_ms_k = {
        "X_used": Xk,
        "y_used": yk,
        "Z_embedding": X_hat_k,
        "trial_indices": idx_k,
        "trial_lengths": trial_len_k,
    }
    joblib.dump(models_ms_k, models_dir / f"{title_ms_k}_models.pkl")
    joblib.dump(results_ms_k, embeds_dir / f"{title_ms_k}_results.pkl")
    print(f"Saved: {models_dir / f'{title_ms_k}_models.pkl'}")
    print(f"Saved: {embeds_dir / f'{title_ms_k}_results.pkl'}")


    # Build a title for S using its own X_key but same style
    X_key_s = SUBJECTS["s"]["X_key"]
    parts_s = X_key_s.split("_")
    trial_len_s=len_s[0][0]
    prefix_s = "_".join(parts_s[:2])
    title_ms_s = f"{prefix_s}_MS_{met}_s_{step}_o_{overlap}_nhu_{model_params['num_hidden_units']}_temp{model_params['temperature']}"

    # Optional plot for S (same function & variable names)
    original_label_order_s = np.unique(y_dir_s).tolist()
    plot_direction_averaged_embedding(
        X_hat_s,
        y_dir_s,
        original_label_order_s,
        c_s,
        plots_dir,
        title_ms_s,
        trial_len_s,
        steady_len_,
        const_len,
        0,
    )

    # Save MS S
    models_ms_s = {"cebra_ms": cebra_ms}
    results_ms_s = {
        "X_used": Xs,
        "y_used": ys,
        "Z_embedding": X_hat_s,
        "trial_indices": idx_s,
        "trial_lengths": trial_len_s,
    }
    joblib.dump(models_ms_s, models_dir / f"{title_ms_s}_models.pkl")
    joblib.dump(results_ms_s, embeds_dir / f"{title_ms_s}_results.pkl")
    print(f"Saved: {models_dir / f'{title_ms_s}_models.pkl'}")
    print(f"Saved: {embeds_dir / f'{title_ms_s}_results.pkl'}")

    return multi_embeddings


'''