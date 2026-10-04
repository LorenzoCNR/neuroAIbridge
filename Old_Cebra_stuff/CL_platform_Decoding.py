# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 00:26:04 2025

@author: loren
"""
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


def decoding_mod(pos_decoder, dir_decoder, emb_test, label_test):
    """
    Use pre-trained decoders to predict and evaluate on the test set.
    """
    pos_pred = pos_decoder.predict(emb_test)
    dir_pred = dir_decoder.predict(emb_test)
    
    prediction = np.stack([pos_pred, dir_pred], axis=1)
    test_score = sklearn.metrics.r2_score(label_test[:, :2], prediction)
    pos_test_err = np.median(np.abs(prediction[:, 0] - label_test[:, 0]))
    pos_test_score = sklearn.metrics.r2_score(label_test[:, 0], prediction[:, 0])
    
    return test_score, pos_test_err, pos_test_score, prediction

#### DATA AND MODEL PARAMETERS
d_name='achilles'
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
valid_split=0.2
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
epochs=200
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
num_iter=30
results=[]
k_values = [1, 3, 5, 7, 9, 16,25,36]  # puoi randomizzarli se vuoi

for i in range(num_iter):
    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)
    print(f"\n Starting {i+1}/{num_iter}")
    
    # create trainin and valid data portion
    dataset_training, dataset_validation = dataset.split_dataset(validation_size=valid_split)
    # ###  window datasets
    dataset_windows_training = dataset_training.create_windows(ww,shift)
    dataset_windows_validation = dataset_validation.create_windows(ww, shift)
    
    # ### Convert to PyTorch datasets
    dataset_validation_pytorch = DatasetEEGTorch(dataset_windows_validation)
    dataset_validation_pytorch.to_device(device)
    dataset_training_pytorch = DatasetEEGTorch(dataset_windows_training)
    dataset_training_pytorch.to_device(device)
    
    
    # DataLoader
    dataloader = DataLoader(dataset_training_pytorch, batch_size=batch_size, shuffle=False)
    dataloader_validation = DataLoader(dataset_validation_pytorch, batch_size=batch_size, shuffle=False)
           
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
    #plot_training_metrics(metrics)
    x_used_train,x_hat_train, labels_train_ = generate_embeddings(model, dataset_training_pytorch, batch_size, device)
    l_opposite_train=1-labels_train_[:,1]
    labels_train = np.concatenate([labels_train_, l_opposite_train[:, None]], axis=1)
        # Generate embeddings for validation data
    x_used_val,x_hat_val, labels_val_= generate_embeddings(model, dataset_validation_pytorch, batch_size, device)
    l_opposite_val=1-labels_val_[:,1]
    labels_val= np.concatenate([labels_val_, l_opposite_val[:, None]], axis=1)
    
    best_k = None
    best_pos_err = float('inf')  # errore minimo trovato
    best_prediction = None       # per tenere il pred migliore
    
    for k in k_values:
        # 1. Fit dei decoder su validation set
        pos_decoder = KNeighborsRegressor(n_neighbors=k).fit(x_hat_train, labels_train[:, 0])
        dir_decoder = KNeighborsClassifier(n_neighbors=k).fit(x_hat_train, labels_train[:, 1])
    
        # 2. Decoding e valutazione
        _, pos_err, _, prediction = decoding_mod(pos_decoder, dir_decoder, x_hat_val, labels_val)
    
        # 3. Se questo è il migliore finora, aggiorno
        if pos_err < best_pos_err:
            best_pos_err = pos_err
            best_k = k
            best_prediction = prediction
    
    print(f"[Iter {i}] Best k = {best_k} → Error = {best_pos_err:.4f}")
    
    result = {
                'rat': 'rat_id',
                'split_no': i,
                'seed': 42 + i,
                'k': best_k,
                'test_pos_err': best_pos_err,
    }
    results.append(result)


# Niente return, salva alla fine
import pandas as pd
df_results_CL = pd.DataFrame(results)

df_results_CL.to_csv('results_CL.csv', index=True)
