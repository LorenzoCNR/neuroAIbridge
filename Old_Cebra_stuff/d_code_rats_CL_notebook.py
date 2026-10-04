#!/usr/bin/evnv python
# coding: utf-8

# In[215]:
# windows
# input_dir=r'F:\........'
# main_root=r'F:\....'

#### Ubuntu
#main_root = r"/media/zlollo/........."
#input_dir = r"/media/zlollo/........."

import os
import sys
import math
from pathlib import Path
import argparse
import logging
import mimetypes
from statsmodels.tsa.seasonal import seasonal_decompose
### load path (use the function in module some functions)
import json
import copy
import time
#import shap
import numpy as np
import yaml
import pickle
import warnings
#import matplotlib.animation as animation
#print(animation.writers.list())
#import umap 
import openTSNE
import random
from src.neurobridge.utils.config import *
from src.neurobridge.utils.debug import explore_obj
from src.neurobridge.viz.manifold_plots import plot_direction_averaged_embedding
from src.neurobridge.viz.manifold_plots import plot_datasets_in_groups

import typing
import joblib as jl
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import ParameterGrid, train_test_split,  ParameterSampler, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
from scipy import stats
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.markers import MarkerStyle
from joblib import Parallel, delayed
import torch
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import optimize as opt
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

i_dir=r'J:\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\AI_for_all\Neuro_Bridge\Contrastive_Stuff_INTEGRA'
os.chdir(i_dir)
os.getcwd()
from some_functions import *
from model_utils import *
# Random Seeds
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Config GPU
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
# backend or inline plots
# %matplotlib inline
# matplotlib.use('Agg')
# matplotlib.use('TkAgg')
# matplotlib.use('QtAgg')  
# from data.eeg_dataset import *
# plt.ion()
# plt.show()
# plt.pause(10)  

# 2) NAME of folder containing data (input) directory. 
data_dir="data"
# (specific) project data folder
sub_data="rat_hippocampus"

# 3) PIPELINE folder name
pipe_path= "EEG-ANN-Pipeline"

# 4) OUTPUT folder: folder to store processed output
#    (if not existing is created)
out_dir="contrastive_output"


project_root, eeg_pipeline_path, default_output_dir, default_input_dir = setup_paths(data_dir,sub_data,out_dir, pipe_path,change_dir=False)



from data import LabelsDistance, TrialEEG, DatasetEEG, DatasetEEGTorch
from data.preprocessing import normalize_signals
from models import EncoderContrastiveWeights
from helpers.model_utils import plot_training_metrics, count_model_parameters, train_model
from helpers.visualization import plot_latent_trajectories_3d, plot_latents_3d
from helpers.visualization import plot_latent_trajectories_3d, plot_latents_3d
from helpers.distance_functions import *
from layers.custom_layers import _Skip, Squeeze, _Norm, _MeanAndConv

try:
    from torch.utils.data import DataLoader
    print("dataloader è stato importato correttamente!")
except ImportError:
    print("Errore: numpy non è stato importato!")
#from torch.utils.data import DataLoader

input_dir = default_input_dir




# # identify trials based upon behavioral data
# # identify trials based upon behavioral data
def create_trial_ids(behav_data):

    trial_ids = np.ones(len(behav_data), dtype=int)
    c_t = [0]
    current_trial = 1

    for i in range(1, len(behav_data)):
        if behav_data[i] != behav_data[i - 1]:
            if behav_data[i - 1] == 0 and behav_data[i] == 1:
                current_trial += 1
                c_t.append(i)
                # include last index
    c_t.append(len(behav_data))
    return trial_ids, np.array(c_t)


def build_model(filters, dropout, latents, num_timepoints, chns, num_units=None, groups=1,normalize=True):
    """
    Build a cnn1d model with:
    - chns: Input channels.
    - filters: convolutional layer(s) filters.
    - latents: outpuit dimension (latent space).
    - num_timepoints: window dimension (test the optimal one).
    - dropout:  dropout.
    - num_units: optional intermediate filters.
    """
    if num_units is None:
        num_units = filters  # Se num_units non è specificato, usa filters.

    layers = [
        Squeeze(),
        nn.Conv1d(chns, filters, kernel_size=2),
        nn.GELU(),
        _Skip(nn.Conv1d(filters, filters, kernel_size=3), nn.GELU()),
        _Skip(nn.Conv1d(filters, filters, kernel_size=3), nn.GELU()),
        _Skip(nn.Conv1d(filters, filters, kernel_size=3), nn.GELU()),
        nn.Conv1d(filters, latents, kernel_size=3),
    ]

    if normalize:
        layers.append(_Norm())  #

    layers.extend([
        nn.Flatten(),  # 
        #nn.Dropout(dropout),  # 
    ])

    return nn.Sequential(*layers)
#### Build the model encoder.
'''
def build_model(filters, dropout, latents, num_timepoints, chns):
    
    return nn.Sequential(
        nn.Conv2d(1, filters, kernel_size=(1, num_timepoints)),
        nn.BatchNorm2d(filters),
        nn.Conv2d(filters, filters, kernel_size=(chns, 1), groups=filters),
        nn.BatchNorm2d(filters),
        nn.Dropout(dropout),
        nn.Flatten(),
        nn.Linear(filters, filters),
        nn.SELU(),
        nn.Dropout(dropout),
        nn.Linear(filters, latents)
    )

'''

# fig.savefig(Path(output_dir) / file_name)

def generate_embeddings(model, dataset_pytorch, batch_size, device):
    model.eval()
    x_used, x_hat, l_pos_, l_dir_ = [], [], [],[]
    with torch.no_grad():
        for i in range(0, dataset_pytorch.num_trials, batch_size):
            x = dataset_pytorch.eeg_signals[i:i+batch_size].to(device)
            l_pos = dataset_pytorch.labels['position'][i:i+batch_size]
            l_dir = dataset_pytorch.labels['direction'][i:i+batch_size]

            f_x = model(x)
            x_used.append(x.cpu().numpy())
            x_hat.append(f_x.cpu().numpy())
            l_pos_.append(l_pos.cpu().numpy())
            l_dir_.append(l_dir.cpu().numpy())

    x_used = np.concatenate(x_used)
    x_hat=np.concatenate(x_hat)
    l_pos_ = np.concatenate(l_pos_)
    l_dir_ = np.concatenate(l_dir_)
    #labels_direction_2 = 1 - labels_direction_1
    labels_ = np.stack((l_pos_, l_dir_), axis=1)
    return x_used,x_hat, labels_

def reverse_windowing(X_windowed, window_size=10, shift=1):
    num_windows, _, num_channels, ww = X_windowed.shape
    T = (num_windows - 1) * shift + window_size

    X_reconstructed = np.zeros((T, num_channels))
    count_matrix = np.zeros((T, num_channels))

    for i in range(num_windows):
        window = X_windowed[i, 0].T  # shape (10, 120)
        start = i * shift
        end = start + window_size
        X_reconstructed[start:end] += window
        count_matrix[start:end] += 1

    count_matrix[count_matrix == 0] = 1  # sicurezza
    return X_reconstructed / count_matrix

#### DATA AND MODEL PARAMETERS
d_name='gatsby'
d_format='jl'
data = load_data(input_dir, d_name,d_format)
chns=data['spikes'].shape[1]

X=data['spikes']
y=data['position']

trial_ids, c_t = create_trial_ids(data['position'][:, 1])
    # Build and train the model
print(chns)
### sampling rate
fs=40
valid_split=0.07
### sample sizes
ww=10
shift=1
dropout = 0.7
## network hidden layers channles
## output channels (latents)
latents = 3
# learning rate
l_rate = 0.0001
#
sigma_pos = 0.3
sigma_time = 0.025
#
batch_size=1024
#
normalize=True
filters = 32
num_units=filters
epochs=800
tau=0.54 #0.55


trials = [
    TrialEEG(
       X[ c_t[i]:c_t[i+1]].T,  # Segnali EEG
          # Labels
         # ,
          [
            ('position', y[c_t[i]:c_t[i+1],0].T),
           ( 'direction',y[c_t[i]:c_t[i+1],1].T        
       ) ],
          # Timepoints
        np.linspace(c_t[i] / fs, c_t[i+1] / fs, c_t[i+1] - c_t[i])  
    )
    for i in range(len(c_t) - 1)
]


dataset = DatasetEEG(trials)
print(dataset)
print(dataset.trials[1])
explore_obj(dataset.trials[1])
dataset.trials[0].eeg_signals.shape

# dataset_windows=dataset.create_windows(ww, shift)
# print(dataset_windows)
# # check
# #explore_obj(dataset_windows.trials[2500])
# dataset_pytorch = DatasetEEGTorch(dataset_windows)
# explore_obj(dataset_pytorch)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataset_pytorch.to_device(device)
# print(device)

# ####---------------- TRAIN AND VALID------------------#######
# create trainin and valid data portion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_training, dataset_validation = dataset.split_dataset(validation_size=valid_split)
explore_obj(dataset_training)
explore_obj(dataset_validation)
# ###  window datasets
dataset_windows_training = dataset_training.create_windows(ww,shift)
dataset_windows_validation = dataset_validation.create_windows(ww, shift)
explore_obj(dataset_windows_training)
dataset_training_pytorch = DatasetEEGTorch(dataset_windows_training)
dataset_training_pytorch.to_device(device)
explore_obj(dataset_training_pytorch)

# ### Convert to PyTorch datasets
dataset_validation_pytorch = DatasetEEGTorch(dataset_windows_validation)
dataset_validation_pytorch.to_device(device)
explore_obj(dataset_validation_pytorch)

# DataLoader
dataloader = DataLoader(dataset_training_pytorch, batch_size=batch_size, shuffle=False)
dataloader_validation = DataLoader(dataset_validation_pytorch, batch_size=batch_size, shuffle=False)

#ataset_pytorch = DatasetEEGTorch(dataset_windows_training)
#dataset_pytorch[0][1]['position']

explore_obj(dataloader)
for batch in dataloader:
      #print(type(batch))  # Stampa il tipo (list, dict, tensor, ecc.)
      print(batch)        # Stampa il contenuto
      break 
      

    # Define label distances
labels_distance = LabelsDistance({
        'position':lambda l1, l2: position_distance(l1, l2, sigma_pos),
        'direction': direction_distance,
})

# Costruzione modello
model = EncoderContrastiveWeights(
    layers=build_model(filters, dropout, latents, ww, chns, normalize=True),
    labels_distance=labels_distance,
    labels_weights=[0.6, 0.4],
    temperature=tau,
    train_temperature=False
)
model.to(device)
print("Training dataset size:", len(dataset_training.trials))
print("Validation dataset size:", len(dataset_validation.trials))
print("Model architecture:", model)
print(f"Model has {count_model_parameters(model)} parameters.")
print(f"rat name is {d_name}")
print(f"{device}")
optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)
batch = next(iter(dataloader))
loss_dict = model.process_batch(batch, optimizer)
metrics = train_model(model, optimizer, dataloader, epochs=epochs, dataloader_validation=dataloader_validation)
    # Save metrics and model
    #torch.save(model.state_dict(), f"{output_dir}/{name}_model.pth")
    #np.save(f"{output_dir}/{name}_metrics.npy", metrics)
    #plt.figure()

 # Evaluate the model and plot latent spaces
model.eval()
plot_training_metrics(metrics)


    # Save metrics and model
    #torch.save(model.state_dict(), f"{output_dir}/{name}_model.pth")
    #np.save(f"{output_dir}/{name}_metrics.npy", metrics)
    #plt.figure()
    #plt.show()
     # Evaluate the model and plot latent spaces
    #model Evali
    # Generate embeddings for training data
    
x_used_train,x_hat_train, labels_train_ = generate_embeddings(model, dataset_training_pytorch, batch_size, device)
l_opposite_train=1-labels_train_[:,1]
labels_train = np.concatenate([labels_train_, l_opposite_train[:, None]], axis=1)
    # Generate embeddings for validation data
x_used_val,x_hat_val, labels_val_= generate_embeddings(model, dataset_validation_pytorch, batch_size, device)
l_opposite_val=1-labels_val_[:,1]
labels_val= np.concatenate([labels_val_, l_opposite_val[:, None]], axis=1)


#♦rat_result_CL_platform = {}
X_reconstructed = reverse_windowing(x_used_train, window_size=10, shift=1)
print(X_reconstructed.shape)  
####•FOR VIDEO plot name/rename embediing in dictionary--> X_hat and label/behaviour--> y

rat_result_CL_platform[d_name] = {
        "X": X_reconstructed,
        "X_hat": x_hat_train,
        "y": labels_train,
    }

dataset_dict = {name: r["X_hat"] for name, r in   rat_result_CL_platform.items()}
label_dict   = {name: r["y"] for name, r in   rat_result_CL_platform.items()}


# Generate grouped 3D plots (2 per figure)
plot_datasets_in_groups(
    dataset_dict=dataset_dict,
    label=label_dict,
    group_size=4,
    title= "CONAN Platform"
)
##################### VIDEO PLOT
import sys
sys.path
neurob_path = Path(r"J:\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\AI_for_all\Neuro_Bridge")
if str(neurob_path) not in sys.path:
    sys.path.append(str(neurob_path)) 
    
from Plot_Video_decoding_Rats import *

achilles_res=rat_result_CL_platform['achilles']
buddy_res=rat_result_CL_platform['buddy']
#writer = FFMpegWriter(fps=15, metadata=dict(artist='NeuroBridge'), bitrate=10000)
#ani.save(str(save_path), writer=writer, dpi=150)
make_video_achilles(
    achilles_res,
    title_track='Decoded (Conan+KNN)',
    title_scatter='Manifold Conan',
    animate=True,
    save_path=r"J:\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\AI_for_all\Neuro_Bridge\achilles_video.mp4"
)   
# metrics='cosine'
# posdir_decode_CL = decoding_knn(z_train, z_val, labels_train, labels_val,metrics, neighbors)

# return z_train, z_val ,labels_train, labels_val,posdir_decode_CL
    
   
    
   #  if __name__ == "__main__":
    #    # #
     #        input_dir = r"F:\CNR_neuroscience\Consistency_across\Codice Davide"
      #       output_dir = r"F:\CNR_neuroscience\Consistency_across\Codice Davide"
    #         name='rat_name'
    #         run_d_code(input_dir, output_dir, name, filters, tau, epochs, dropout, latents, ww, sigma_pos, sigma_time, train_split, valid_split, l_rate, Batch_size):
    
    
