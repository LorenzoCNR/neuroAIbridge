# -*- coding: utf-8 -*-
"""Spike emission generators for synthetic neural population activity."""

import numpy as np
from .builders import (
    latent_to_drive,
    drive_to_rate,
    rate_to_spike,
    )


class SpikeEmissionGenerator:
    '''
    Task: 
         Given Latent trajectories 
         the class create a Matrix of neural spikes count per bin (final task)
         starting with a neural drive passing through a rate to spiek
         
    
    
    Inputs: 
        - number of final neurons
        - B loading matrix to cast neurons to latent space
        - a baseline value c for the neural drive
        - type of non linearity (exponential vel softplus)
        - time interval length
        
    
    Output
           - X
    
    '''
    def __init__(
            self,
            B,
            c,
            dt,
            nonlinearity,
            overdispersion=0.0,
            refractory_mean_bins=None,
            refractory_std_bins=0.0,
            burst_probability=0.0,
            burst_size_mean=0.0,
            burst_window_bins=1):
       
       self.loadings=B
       self.bias=c
       self.interval=dt
       self.nonlinearity=nonlinearity
       self.overdispersion=overdispersion
       self.refractory_mean_bins=refractory_mean_bins
       self.refractory_std_bins=refractory_std_bins
       self.burst_probability=burst_probability
       self.burst_size_mean=burst_size_mean
       self.burst_window_bins=burst_window_bins
       
    #@staticmethod  
    def _latent_to_drive(self,Z):
        return latent_to_drive(Z, self.loadings,self.bias)
        
  
#    @staticmethod
    def _drive_to_rate(self,u):
        return drive_to_rate(u, self.nonlinearity)
    
    def _rate_to_spike(self,lam):
        return rate_to_spike(
            lam,
            self.interval,
            overdispersion=self.overdispersion,
            refractory_mean_bins=self.refractory_mean_bins,
            refractory_std_bins=self.refractory_std_bins,
            burst_probability=self.burst_probability,
            burst_size_mean=self.burst_size_mean,
            burst_window_bins=self.burst_window_bins,
        )
    

    def generate_spikes(self, Z):
     ##   B, c =self.loadings, self.bias
     #   nn=B.shape[1]
     #   t_int=self.interval
    #  non_lin=self.nonlinearity
    #
        if Z.ndim != 3:
            raise ValueError("Z must be (n_trials, L, k)")    
        u=self._latent_to_drive(Z)
        lam=self._drive_to_rate(u)
        spikes=self._rate_to_spike(lam)
                
            
        return u, lam, spikes
        
     
           
             
       
   
          
   
    
# class MemorySpikeEmissionGenerator:
#     """
#     Generate spikes with a simple memory/threshold mechanism.

#     This emission model is different from the Poisson emission model:

#         Z -> u -> V -> spikes

#     where V accumulates drive over time:

#         V_t = alpha * V_{t-1} + u_t + noise

#     and a spike is emitted when V crosses a threshold. After firing, V is reset
#     and an optional refractory period can block subsequent spikes.
#     """

#     def __init__(
#             self,
#             B,
#             c,
#             alpha=0.9,
#             threshold=1.0,
#             reset_value=0.0,
#             noise_std=0.0,
#             refractory_bins=0,
#             random_state=None):

#         self.loadings = B
#         self.bias = c
#         self.alpha = alpha
#         self.threshold = threshold
#         self.reset_value = reset_value
#         self.noise_std = noise_std
#         self.refractory_bins = refractory_bins
#         self.random_state = random_state

#     def _latent_to_drive(self, Z):
#         return latent_to_drive(Z, self.loadings, self.bias)

#     def _drive_to_spike_memory(self, u):
#         return drive_to_spike_memory(
#             u,
#             alpha=self.alpha,
#             threshold=self.threshold,
#             reset_value=self.reset_value,
#             noise_std=self.noise_std,
#             refractory_bins=self.refractory_bins,
#             random_state=self.random_state,
#         )

#     def generate_spikes(self, Z):
#         if Z.ndim != 3:
#             raise ValueError("Z must be (n_trials, L, k)")

#         u = self._latent_to_drive(Z)
#         V, spikes = self._drive_to_spike_memory(u)

#         return u, V, spikes
