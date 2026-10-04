#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:15:25 2024

@author: lorenzo
"""
'''
sudo rmmod nvidia_uvm
sudo rmmod nvidia
sudo modprobe nvidia
sudo modprobe nvidia_uvm
'''

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Is CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()


# In PyTorch, la serializzazione viene eseguita con torch.save per 
# salvare modelli, stati di ottimizzatori o tensori:

# Salva un modello
torch.save(model.state_dict(), "model.pth")

# Carica un modello
model.load_state_dict(torch.load("model.pth"))
                      

                      ### Parallelizzazione ##
from torch.nn.parallel import DataParallel
model = DataParallel(model)


### Multi procesing ###
from torch.utils.data import DataLoader

DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)