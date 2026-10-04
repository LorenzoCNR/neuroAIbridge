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

import tensorflow as tf
print("GPUs disponibili:", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    


#### mixed precision accelera il training
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

mixed_precision.set_policy(policy)

### distribuzione su piùù gpu
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model()
    

#### colli di bottiglia con tf.profiler
tensorboard --logdir=logs/
####
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
dataset = dataset.shuffle(buffer_size=10000).batch(64).prefetch(tf.data.AUTOTUNE)

### serializzazione
# Salva un modello
model.save("model")

# Carica un modello
from tensorflow.keras.models import load_model
model = load_model("model")

### Multi GPU
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model()
    
    
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE