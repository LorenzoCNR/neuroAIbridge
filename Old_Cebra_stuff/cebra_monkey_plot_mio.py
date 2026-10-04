# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 00:28:34 2025

@author: loren
"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib as jl
import cebra.datasets
from cebra import CEBRA

monkey_pos = cebra.datasets.init('area2-bump-pos-active')
monkey_target = cebra.datasets.init('area2-bump-target-active')
cebra_pos_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.0001,
                        temperature=1,
                        output_dimension=3,
                        max_iterations=5000,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

cebra_pos_model.fit(monkey_pos.neural, monkey_pos.continuous_index.numpy())
cebra_pos = cebra_pos_model.transform(monkey_pos.neural)

cebra_target_model = CEBRA(model_architecture='offset10-model',
                           batch_size=512,
                           learning_rate=0.0001,
                           temperature=1,
                           output_dimension=3,
                           max_iterations=24000,
                           distance='cosine',
                           conditional='time_delta',
                           device='cuda_if_available',
                           verbose=True,
                           time_offsets=10)

cebra_target_model.fit(monkey_target.neural,
                       monkey_target.discrete_index.numpy())
cebra_target = cebra_target_model.transform(monkey_target.neural)

X_hat=cebra_target
y_dir_= monkey_target.discrete_index.numpy()+1
original_label_order =  np.sort(np.unique(y_dir_))
output_folder=os.getcwd()
from some_functions import *
c_s="maroon"
trial_length_=600
quiescent_length=0
const_len=True
title_='Cebra Manifold'
plot_direction_averaged_embedding(
            #¶x_x,
           X_hat,
            #y_y,
           y_dir_,
            
            original_label_order,
            c_s,
            output_folder,
            title_,
            trial_length_,
            quiescent_length,
            const_len,
            0) 