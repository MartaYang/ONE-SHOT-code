"""/usr/bin/env python3"""

# Copied from 
# https://github.com/huggingface/diffusers/blob/v0.31.0/
# src/diffusers/schedulers/scheduling_unipc_multistep.py
# Convert unipc for flow matching
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import (KarrasDiffusionSchedulers,
                                                   SchedulerMixin,
                                                   SchedulerOutput)
from diffusers.utils import deprecate, is_scipy_available

if is_scipy_available():
    """Importing `scipy` to check whether it's available or not."""
    import scipy.stats


class FlowMatchScheduler(SchedulerMixin, ConfigMixin):
    """The Flow Match scheduler"""
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: int = 3,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    # Modified from 
    # diffusers.schedulers.scheduling_flow_match_euler_discrete
    # .FlowMatchEulerDiscreteScheduler.set_timesteps
    def set_timesteps(
        self, 
        num_inference_steps=100, 
        denoising_strength=1.0, 
        training=False, 
        shift=None,
        device=None,
    ):
        """Set the discrete timesteps used by the scheduler"""
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps

        timesteps = self.timesteps.to(device)
        sigmas = self.sigmas.to(device)
        self.timesteps = self.timesteps.to(device)
        self.sigmas = self.sigmas.to(device)

        if training:
            x = timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing


    def step(
        self, 
        model_output, 
        timestep, 
        sample,
        return_dict=True,
        to_final=False
    ):
        """Prepare the next step"""
        timesteps = self.timesteps.to(sample.device)
        sigmas = self.sigmas.to(sample.device)

        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma = sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    
    def return_to_timestep(
        self, 
        timestep, 
        sample, 
        sample_stablized
    ):
        """Return to a specific time step"""
        timesteps = self.timesteps.to(sample.device)
        sigmas = self.sigmas.to(sample.device)

        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma = sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output
    
    
    def add_noise(
        self, 
        original_samples, 
        noise, 
        timestep
    ):
        """Add noise"""
        timesteps = self.timesteps.to(original_samples.device)
        sigmas = self.sigmas.to(original_samples.device)

        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma = sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample
    

    def training_target(
        self, 
        sample, 
        noise, 
        timestep
    ):
        """Training target"""
        target = noise - sample
        return target
    

    def training_weight(
        self,
        timestep
    ):
        """Training weight"""
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights

