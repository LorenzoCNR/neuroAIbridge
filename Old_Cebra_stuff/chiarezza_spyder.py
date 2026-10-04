# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:38:29 2026

@author: zloll
"""

import os
import sys


# %%
from pathlib import Path
import os
import joblib as jl
import sys
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader
from scipy.io import savemat
from sklearn.decomposition import PCA


from experiments.Spike_simulator import PROJECT_ROOT
# %%
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd().resolve()

if not (PROJECT_ROOT / "src" / "neurobridge").exists():
    PROJECT_ROOT = Path.cwd().resolve()

if not (PROJECT_ROOT / "src" / "neurobridge").exists():
    raise FileNotFoundError(f"Cannot find Neuro_Bridge project root from {PROJECT_ROOT}")

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))


os.getcwd()
idir=r'C:\Users\zloll\Desktop\Condivisione\Back_USB_05_11\AI_PhD_Neuro_CNR\Empirics\Neuro_Bridge'
idir=r'C:\Users\loren\Desktop\USB_Content\AI_PhD_Neuro_CNR\Empirics\Neuro_Bridge'
os.chdir(idir)

from neurobridge.data.dataset import TemporalWindowDataset
from neurobridge.data.sim.builders import *
from neurobridge.data.sim import LatentTrajectoryGenerator, SpikeEmissionGenerator, build_structured_B
from neurobridge.data.sim.builders import apply_temporal_lag
from neurobridge.eval.representation import (
    evaluate_latent_recovery,
    lagged_alignment_by_trial_time,
    lagged_alignment_scores,
    procrustes_align,
)
from neurobridge.losses import soft_contrastive_loss, supervised_infonce_loss, time_offset_infonce_loss
from neurobridge.models import (
    TemporalCNNEncoder,
    TemporalLSTMEncoder,
    TemporalMLPEncoder,
    TemporalTransformerEncoder,
)
from cebra import CEBRA
from neurobridge.sampling.batch_similarity import (
    batch_structured_similarity,
    batch_structured_similarity_from_specs,
)
from neurobridge.sampling.f_windows import build_windows
from neurobridge.train.loop import encode_windows, train_epoch
from neurobridge.viz.manifold_plots import (
    plot_condition_centroids_2d,
    plot_condition_trajectories_2d,
    plot_condition_trajectories_sphere,
    plot_embedding_2d,
    plot_embedding_sphere,
    _condition_color
)



import numpy as np

n_trials=160
L_trials=100
k=3
phi=0.4
conditions=np.arange(1,9)

generatore=LatentTrajectoryGenerator(n_trials, L_trials, k, phi, conditions)

generatore.condition_mode
generatore.noise_scale
t, s, p = generatore._build_movement_profile()
t.max()
s.max()
Z, cond=generatore.generate_latent()
Z.shape

Z[:,:,0].mean(axis=0).shape
Z[:,:,1].mean(axis=0).shape
plt.scatter(Z[:,:,0].mean(axis=0), 
Z[:,:,1].mean(axis=0))

fig = go.Figure()
for idx, label in enumerate(conditions):
    print(idx, label)
    mask = cond ==label
    Z_mean = Z[mask].mean(axis=0)

    hex_color= _condition_color(idx, len(conditions))
    fig.add_trace(go.Scatter(
        x=Z_mean[:,0],
        y=Z_mean[:, 1],
        mode="markers",
        marker=dict(color=hex_color, size=4, opacity=0.75),
        name=f"cond {label}",
    ))
fig.show(renderer="browser")



generatore_lin=LatentTrajectoryGenerator(n_trials, L_trials, k, phi,
                                         condition_mode="linear")

generatore_lin.condition_mode
generatore_lin.noise_scale
t, s, p = generatore_lin._build_movement_profile()
t.max()
s.max()
Z, cond=generatore.generate_latent()
Z.shape

Z[:,:,0].mean(axis=0).shape
Z[:,:,1].mean(axis=0).shape
plt.scatter(Z[:,:,0].mean(axis=0), 
Z[:,:,1].mean(axis=0))

fig = go.Figure()
for idx, label in enumerate(conditions):
    print(idx, label)
    mask = cond ==label
    Z_mean = Z[mask].mean(axis=0)

    hex_color= _condition_color(idx, len(conditions))
    fig.add_trace(go.Scatter(
        x=Z_mean[:,0],
        y=Z_mean[:, 1],
        mode="markers",
        marker=dict(color=hex_color, size=4, opacity=0.75),
        name=f"cond {label}",
    ))
fig.show(renderer="browser")