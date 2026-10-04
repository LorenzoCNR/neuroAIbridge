#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 19:40:38 2024

@author: lorenzo
"""

import os
import sys
from pathlib import Path

# 
# from some_functions import plot_embs
# from data import LabelsDistance, TrialEEG, DatasetEEG, DatasetEEGTorch
# from data.preprocessing import normalize_signals
# from models import EncoderContrastiveWeights
# from helpers.model_utils import plot_training_metrics, count_model_parameters, train_model
# from helpers.visualization import plot_latent_trajectories_3d, plot_latents_3d
# from layers.custom_layers import _Skip, Squeeze, _Norm, _MeanAndConv

# 
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import random
from torch import nn
import torch
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
import joblib as jl
import pandas as pd
import seaborn as sns# 
from cebra import CEBRA
import cebra

# DATI monkey sono quelli che vengono dati dalla demo ###
## riprendo e arricchisco dai dati in figura ###

#
####
data_dir=r'/media/lorenzo/21DB-AB79/AI_PhD_Neuro_CNR/Empirics/GIT_stuff/cebra-figures/data/EDFigure1.h5'
data_fig = pd.read_hdf(data_dir)
data_fig_dic=data_fig.to_dict()
ephys_EDF1 = data_fig["monkey"]["neural"]



#data.reset_index(inplace=True)
data_dic=data.to_dict()
#df_behavior=data['behavior'][3]

### congurenza dati miei  dati figura

################################ MOVIMENTI ATTIVI #######################
#### coincidono pos con active
active = data_fig["behavior"]["active"]
active_target = data_fig["behavior"]["active_target"]

#pos_act=dati_monkey['pos']
#print(np.array_equal(dati_monkey['pos'],active))
#rint(np.allclose(dati_monkey['pos'],active))

####PASSIVI
### Questi dati non li ritrovo nel db che ci danno (sono solo nei dati figura)
passive = data["behavior"]["passive"]
passive_target = data["behavior"]["passive_target"]
### li aggiungo al data dict
#dati_monkey['passive_target']=passive_target 
#dati_monkey['pos_passive']=passive


### figura a riprova (alterna pos act con active)
fig = plt.figure(figsize=(8, 4))
ax1 = plt.subplot(121)
ax1.scatter(active[:, 0], active[:, 1], color=plt.cm.hsv(1 / 8 * active_target), s=1)
ax1.axis("off")

ax2 = plt.subplot(122)
ax2.scatter(passive[:, 0], passive[:, 1], color=plt.cm.hsv(1 / 8 * passive_target), s=1)
ax2.axis("off")

###
#trial_len=600
#dati_monkey['active_target']=np.repeat(dati_monkey['movement_dir'], trial_len)
## check match
#print(np.array_equal(dati_monkey['mov_dir_exp'],active_target))


#### aggiungo i dati di traiettoria da prevedere (in data figures)
#dati_monkey['to predict']=data_dic['trajectory']['true']


# #### rinomino
# dati_monkey['pos_active']=dati_monkey.pop('pos')
# dati_monkey['spikes_active']=dati_monkey.pop('spikes')
# dati_monkey['vel_active']=dati_monkey.pop('vel')
# ### passive è una colonna inutile fatta solo di false
# np.unique(dati_monkey['movement_dir'])
# overview = data["overview"]["cebra-behavior"]

# features_pos = data["behavior_time"]["behavior"]["embedding"]
# labels_pos = data["behavior_time"]["behavior"]["label"]

##### FIGURA 3b 
'''
omparison of embeddings of active trials generated with CEBRA-Behavior, 
CEBRA-Time, conv-pi-VAE variants, tSNE, and UMAP. The embeddings of trials 
(n=364) of each direction are post-hoc averaged.
'''

overview = data["overview"]
fig = plt.figure(figsize=(30, 5))
plt.subplots_adjust(wspace=0, hspace=0)
emissions_list = [
    overview["cebra-behavior"],
    overview["pivae_w"],
    overview["cebra-time"],
    overview["pivae_wo"],
    overview["autolfads"],
    overview["tsne"],
    overview["umap"],
]
num_plots = len(emissions_list)
labels = overview["label"]

def plot_subplot(ax, j, lw = 2):

    if j == 0:
        idx1, idx2 = (2, 0)

    elif j == 1 or j == 3:
        idx1, idx2 = (2, 3)

    elif j == 5 or j == 6:
        idx1, idx2 = (0, 1)

    else:
        idx1, idx2 = (0, 1)


    if j == 0:
        trials = emissions_list[j].reshape(-1, 600, 4)
        trials_labels = labels.reshape(-1, 600)[:, 1]
        mean_trials = []
        for i in range(8):
            mean_trial = trials[trials_labels == i].mean(axis=0)
            mean_trials.append(mean_trial)
        for trial, label in zip(mean_trials, np.arange(8)):
            ax.plot(
                trial[:, idx1], trial[:, idx2], color=plt.cm.hsv(1 / 8 * label),
                linewidth = lw
            )
    elif j == 1:
        trials = emissions_list[j].reshape(-1, 600, 4)
        trials_labels = labels.reshape(-1, 600)[:, 1]
        mean_trials = []
        for i in range(8):
            mean_trial = trials[trials_labels == i].mean(axis=0)
            mean_trials.append(mean_trial)
        for trial, label in zip(mean_trials, np.arange(8)):
            ax.plot(
                trial[:, idx1], trial[:, idx2], color=plt.cm.hsv(1 / 8 * label),
              linewidth = lw
            )
    elif j == 4:
        mean_trials = emissions_list[j]
        for trial, label in zip(mean_trials, np.arange(8)):
            ax.plot(
                trial[:, idx1], trial[:, idx2], color=plt.cm.hsv(1 / 8 * label),
              linewidth = lw
            )
    else:
        trials = emissions_list[j].reshape(-1, 600, emissions_list[j].shape[-1])
        trials_labels = labels.reshape(-1, 600)[:, 1]
        mean_trials = []
        for i in range(8):
            mean_trial = trials[trials_labels == i].mean(axis=0)
            mean_trials.append(mean_trial)
        for trial, label in zip(mean_trials, np.arange(8)):
            ax.plot(
                trial[:, idx1], trial[:, idx2], color=plt.cm.hsv(1 / 8 * label),
              linewidth = lw
            )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        [
            "CEBRA-Behavior",
            "pi-VAE w/ label",
            "CEBRA-Time",
            "pi-VAE w/o",
            "autoLFADS",
            "tSNE",
            "UMAP",
        ][j],
        fontsize=20,
    )
    return ax


for j in range(num_plots):
    if j == 0:
        idx1, idx2 = (2, 0)
        ax = fig.add_subplot(1, num_plots, j + 1)
    elif j == 1 or j == 3:
        idx1, idx2 = (2, 3)
        ax = fig.add_subplot(1, num_plots, j + 1)
    elif j == 5 or j == 6:
        idx1, idx2 = (0, 1)
        ax = fig.add_subplot(1, num_plots, j + 1)
    else:
        idx1, idx2 = (0, 1)
        ax = fig.add_subplot(1, num_plots, j + 1)
    plot_subplot(ax, j, lw = 3)
plt.show()

# # Uncomment to generate high-quality plots of the paper

# for j in range(num_plots):
#   fig = plt.figure(figsize=(5, 5), dpi = 300)
#   ax = plot_subplot(plt.gca(), j, lw= 3)
#   ax.set_aspect("equal")
#   plt.savefig(f"monkey_{j}.svg", bbox_inches = "tight", transparent = True)
#   plt.show()

methods = [
  "cebra-behavior",
  "pivae_w",
  "cebra-time",
  "pivae_wo",
  "autolfads",
  "tsne",
  "umap",
]

emissions_list = [
    overview["cebra-behavior"],
    overview["pivae_w"],
    overview["cebra-time"],
    overview["pivae_wo"],
    overview["tsne"],
    overview["umap"],
]

for m, e in zip(methods, emissions_list):
  print(m, e.shape)

##### FIGURA 3c
### simil questa me la ritrovo in demo

'''
CEBRA-Behavior trained with x,y position of the hand. Left panel is color-coded 
to x position and right panel is color-coded to y position.
'''
def set_pane_axis(ax):
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])


features_pos = data["behavior_time"]["behavior"]["embedding"]
labels_pos = data["behavior_time"]["behavior"]["label"]
dx1, idx2, idx3 = (0, 1, 2)
fig = plt.figure(figsize=(12, 6))
fig.suptitle("CEBRA-Behavior, labels - continuous position", fontsize=20)
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.set_title(f"x pos")
x = ax1.scatter(
    features_pos[:, idx1],
    features_pos[:, idx2],
    features_pos[:, idx3],
    cmap="seismic",
    c=labels_pos[:, 0],
    s=0.05,
    vmin=-15,
    vmax=15,
)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.set_title(f"y pos")
y = ax2.scatter(
    features_pos[:, idx1],
    features_pos[:, idx2],
    features_pos[:, idx3],
    cmap="seismic",
    c=labels_pos[:, 1],
    s=0.05,
    vmin=-15,
    vmax=15,
)
yc = plt.colorbar(y, fraction=0.03, pad=0.05, ticks=np.linspace(-15, 15, 7))
yc.ax.tick_params(labelsize=15)
yc.ax.set_title("(cm)", fontsize=10)
set_pane_axis(ax1)
set_pane_axis(ax2)

##### FIGURA 3D
### simil questa me la ritrovo in demo

'''

CEBRA-Time without any external behavior variables. As in \textbf{c}, left and
right are color-coded to x and y position, respectively.

'''

features_time = data["behavior_time"]["time"]["embedding"]
labels_time = data["behavior_time"]["time"]["label"]
idx1, idx2, idx3 = (0, 1, 2)
fig = plt.figure(figsize=(12, 6))
fig.suptitle("CEBRA-Time, labels - continuous position", fontsize=20)
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.set_title(f"x pos")
x = ax1.scatter(
    features_time[:, idx1],
    features_time[:, idx2],
    features_time[:, idx3],
    cmap="seismic",
    c=labels_time[:, 0],
    s=0.05,
    vmin=-15,
    vmax=15,
)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.set_title(f"y pos")
y = ax2.scatter(
    features_time[:, idx1],
    features_time[:, idx2],
    features_time[:, idx3],
    cmap="seismic",
    c=labels_time[:, 1],
    s=0.05,
    vmin=-15,
    vmax=15,
)
yc = plt.colorbar(y, fraction=0.03, pad=0.05, ticks=np.linspace(-15, 15, 7))
yc.ax.tick_params(labelsize=15)
yc.ax.set_title("(cm)", fontsize=10)
set_pane_axis(ax1)
set_pane_axis(ax2)

### rename data


###Figure 3e
'''
Left, CEBRA-Behavior embedding trained with a 4D latent space, with target
 direction and active OR passive trials (trained separately) as behavior labels. 
 Plotted separately, active vs. passive training condition.

'''
active_emission = data["cebra_ap_sep"]["active"]["embedding"]
active_label = data["cebra_ap_sep"]["active"]["label"]
passive_emission = data["cebra_ap_sep"]["passive"]["embedding"]
passive_label = data["cebra_ap_sep"]["passive"]["label"]
feature_num = active_emission.shape[-1]

fig = plt.figure(
    figsize=(20, 10),
)
fig.suptitle(
    "Separately trained active/passive trials with direction (1-8) labels", fontsize=20
)
idx1, idx2, idx3 = 1, 2, 0
active_trials_index = active_label

active_trials = active_emission.reshape(-1, 600, feature_num)
active_trials_labels = active_label.reshape(-1, 600)[:, 0].squeeze()
mean_active_trials = []

for i in range(8):
    mean_active_trial = active_trials[active_trials_labels == i].mean(
        axis=0
    ) / np.linalg.norm(active_trials[active_trials_labels == i].mean(axis=0))
    mean_active_trials.append(mean_active_trial)

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
for trial, label in zip(mean_active_trials, np.arange(8)):
    ax1.plot(
        trial[:, idx1], trial[:, idx2], trial[:, idx3], color=plt.cm.hsv(1 / 8 * label)
    )
    ax1.plot(
        trial[0, idx1],
        trial[0, idx2],
        trial[0, idx3],
        color=plt.cm.hsv(1 / 8 * label),
        marker="o",
        markersize=10,
    )
    ax1.plot(
        trial[-1, idx1],
        trial[-1, idx2],
        trial[-1, idx3],
        color=plt.cm.hsv(1 / 8 * label),
        marker="<",
        markersize=10,
    )
set_pane_axis(ax1)
ax1.set_title("Active", fontsize=20)


passive_trials_index = passive_label
passive_trials = passive_emission.reshape(-1, 600, feature_num)
passive_trials_labels = passive_label.reshape(-1, 600)[:, 0].squeeze()
mean_passive_trials = []

for i in range(8):
    mean_passive_trial = passive_trials[passive_trials_labels == i].mean(
        axis=0
    ) / np.linalg.norm(passive_trials[passive_trials_labels == i].mean(axis=0))
    mean_passive_trials.append(mean_passive_trial)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
for trial, label in zip(mean_passive_trials, np.arange(8)):
    ax2.plot(
        trial[:, idx1], trial[:, idx2], trial[:, idx3], color=plt.cm.hsv(1 / 8 * label)
    )
    ax2.plot(
        trial[0, idx1],
        trial[0, idx2],
        trial[0, idx3],
        color=plt.cm.hsv(1 / 8 * label),
        marker="o",
        markersize=10,
    )
    ax2.plot(
        trial[-1, idx1],
        trial[-1, idx2],
        trial[-1, idx3],
        color=plt.cm.hsv(1 / 8 * label),
        marker="<",
        markersize=10,
    )
set_pane_axis(ax2)
ax2.set_title("Passive", fontsize=20)


####Figure 3f
'''
   Left, CEBRA-Behavior embedding trained with a 4D latent space, with target 
   direction and active and passive trials as behavior labels, but plotted
   separately, active vs. passive trials.
   
 '''
 target_emission = data["cebra_ap_all"]["embedding"]
target_label = data["cebra_ap_all"]["label"]

fig = plt.figure(
    figsize=(20, 10),
)
fig.suptitle(
    "Concurrently trained active/passive trials with direction (1-8) + active/passive trial info labels",
    fontsize=20,
)
idx1, idx2, idx3 = 2, 0, 1
active_trials_index = target_label < 8
active_trials = target_emission[active_trials_index].reshape(-1, 600, feature_num)
active_trials_labels = (
    target_label[active_trials_index].reshape(-1, 600)[:, 0].squeeze()
)
mean_active_trials = []

for i in range(8):
    mean_active_trial = (
        active_trials[active_trials_labels == i].mean(axis=0)
        / np.linalg.norm(active_trials[active_trials_labels == i].mean(axis=0), axis=1)[
            :, None
        ]
    )
    mean_active_trials.append(mean_active_trial)

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
for trial, label in zip(mean_active_trials, np.arange(8)):
    ax1.plot(
        trial[:, idx1], trial[:, idx2], trial[:, idx3], color=plt.cm.hsv(1 / 8 * label)
    )
    ax1.plot(
        trial[0, idx1],
        trial[0, idx2],
        trial[0, idx3],
        color=plt.cm.hsv(1 / 8 * label),
        marker="o",
        markersize=10,
    )
    ax1.plot(
        trial[-1, idx1],
        trial[-1, idx2],
        trial[-1, idx3],
        color=plt.cm.hsv(1 / 8 * label),
        marker="<",
        markersize=10,
    )
set_pane_axis(ax1)
ax1.set_title("Active", fontsize=20)

# ax3 = fig.add_subplot(2,2,3)
passive_trials_index = target_label >= 8
passive_trials = target_emission[passive_trials_index].reshape(-1, 600, feature_num)
passive_trials_labels = (
    target_label[passive_trials_index].reshape(-1, 600)[:, 0].squeeze()
)
mean_passive_trials = []

for i in range(8, 16):
    mean_passive_trial = (
        passive_trials[passive_trials_labels == i].mean(axis=0)
        / np.linalg.norm(
            passive_trials[passive_trials_labels == i].mean(axis=0), axis=1
        )[:, None]
    )
    mean_passive_trials.append(mean_passive_trial)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")

for trial, label in zip(mean_passive_trials, np.arange(8, 16)):
    ax2.plot(
        trial[:, idx1],
        trial[:, idx2],
        trial[:, idx3],
        color=plt.cm.hsv(1 / 8 * (label - 8)),
    )
    ax2.plot(
        trial[0, idx1],
        trial[0, idx2],
        trial[0, idx3],
        color=plt.cm.hsv(1 / 8 * (label - 8)),
        marker="o",
        markersize=10,
    )
    ax2.plot(
        trial[-1, idx1],
        trial[-1, idx2],
        trial[-1, idx3],
        color=plt.cm.hsv(1 / 8 * (label - 8)),
        marker="<",
        markersize=10,
    )
set_pane_axis(ax2)
ax2.set_title("Passive", fontsize=20)




