import os
# windows
i_dir=r'J:\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\AI_for_all\Neuro_Bridge'
# ubuntu
#i_dir

os.chdir(i_dir)
from src.neurobridge.utils.paths import setup_paths, project_paths
from src.neurobridge.utils.project_store import ProjectStore
from src.neurobridge.data.io import load_data

# ---------------- CONFIG ----------------
project_name = "hasson_common_space"
data_file    = "dati_mirco_18_03_s"
data_format  = "mat"

# 1. Costruisci i percorsi standardizzati (relativi)
sub_data_rel, out_dir_rel = project_paths(project_name)

# 2. Setup dei path assoluti
project_root, pipeline_path, default_output_dir, default_input_dir = setup_paths(
    data_dir="data",           # cartella top-level dei dati
    sub_data=sub_data_rel,     # es. "projects/hasson_common_space"
    out_dir=out_dir_rel,       # es. "outputs/projects/hasson_common_space"
    pipe_path="src/neurobridge/encoders",      # dove mettere la pipeline
    change_dir=False
)


############ -------------------PROVE VARIE--------------------- ##############
import sys, pathlib
sys.path.insert(0, r"J:\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\AI_for_all\Neuro_Bridge")


#---------------------------------------
# funzioni nel modulo distance functions cartella src/neurobrige/math
#---------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. clamp min

x = torch.tensor([0.0, 1.5, -3.2])
print("Originale:", x)

clamped=x.clamp_min(0.5)
print("clamped", clamped)

w = torch.tensor([2.0], requires_grad=True)

# 2. Decoratore @torch.no_grad
@torch.no_grad()
def f(x):
    return x*w 

out_=f(torch.tensor[3.0])

#---------------------------------------
# funzioni nel modulo block cartella src/neurobrige/mmodels
#---------------------------------------
# funzioni accessorie ai modelli 

x=torch.randn(2,8,64)

#---------------------------------------
# funzioni nel modulo distance functions cartella src/neurobrige/math
#---------------------------------------

import torch
from src.neurobridge.utils.paths import setup_paths
from src.neurobridge.models.blocks import _Skip, Squeeze, _Norm, _MeanAndConv

def test_skip():
    x = torch.randn(2, 8, 64)
    block = _Skip(nn.Conv1d(8, 8, kernel_size=3, padding=1), crop=(0,0))
    y = block(x)
    assert y.shape == x.shape