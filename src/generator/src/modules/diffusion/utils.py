# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

import math
import torch
import torch.nn as nn
import numpy as np

from einops import repeat

from generator.src.utils import instantiate_from_config

###

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "cosine":
        timesteps = (torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s)

        alphas = torch.cos(timesteps / (1 + cosine_s) * np.pi / 2).pow(2)
        alphas /= alphas[0]

        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "linear":       betas = (torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2)
    elif schedule == "sqrt_linear":  betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":         betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:                            raise ValueError(f"schedule '{schedule}' unknown.")
    
    return betas.numpy()

###

def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):

    if ddim_discr_method == 'uniform':  ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, num_ddpm_timesteps // num_ddim_timesteps)))
    elif ddim_discr_method == 'quad':   ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:                               raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # Add one to get the final alpha values right (the ones from first scale to data during sampling)
    return ddim_timesteps + 1

###

def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # https://arxiv.org/abs/2010.02502

    # Select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, 'f'this results in the following sigma_t schedule for ddim sampler {sigmas}')

    return sigmas, alphas, alphas_prev

###

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    return a.gather(-1, t).reshape(b, *((1,) * (len(x_shape) - 1)))

###

def checkpoint(func, inputs, params, flag):
    # Evaluate a function without caching intermediate activations, allowing for reduced memory at the expense of extra compute in the backward pass
    return CheckpointFunction.apply(func, len(inputs), *(tuple(inputs) + tuple(params))) if flag else func(*inputs)

###

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    # Create sinusoidal timestep embeddings.
    
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:  embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    else:  embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

###

def zero_module(module):
    # Zero out the parameters of a module and return it
    for p in module.parameters():  
        p.detach().zero_()
    return module

###

def scale_module(module, scale):
    # Scale the parameters of a module and return it
    for p in module.parameters():
        p.detach().mul_(scale)
    return module

###

def mean_flat(tensor):
    # Take the mean over all non-batch dimensions
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

###

def normalization(channels):
    # Make a standard normalization layer
    return GroupNorm32(32, channels)

###

class SiLU(nn.Module):  # PyTorch 1.7 has SiLU, but we support PyTorch 1.5
    def forward(self, x):
        return x * torch.sigmoid(x)

###

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

###

def conv_nd(dims, *args, **kwargs):
    # Create a 1D, 2D, or 3D convolution module

    if dims == 1:    return nn.Conv1d(*args, **kwargs)
    elif dims == 2:  return nn.Conv2d(*args, **kwargs)
    elif dims == 3:  return nn.Conv3d(*args, **kwargs)

    raise ValueError(f"unsupported dimensions: {dims}")

###s

def linear(*args, **kwargs):
    # Create a linear module
    return nn.Linear(*args, **kwargs)

###

def avg_pool_nd(dims, *args, **kwargs):
    # Create a 1D, 2D, or 3D average pooling module

    if dims == 1:    return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:  return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:  return nn.AvgPool3d(*args, **kwargs)

    raise ValueError(f"unsupported dimensions: {dims}")

###

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            return ctx.run_function(*ctx.input_tensors)

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]

        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)

        input_grads = torch.autograd.grad(output_tensors, ctx.input_tensors + ctx.input_params, output_grads, allow_unused=True,)
        del ctx.input_tensors; del ctx.input_params; del output_tensors
        return (None, None) + input_grads

###

class HybridConditioner(nn.Module):

    def __init__(self, c_concat_config, c_crossattn_config):
        super().__init__()
        self.concat_conditioner = instantiate_from_config(c_concat_config)
        self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

    def forward(self, c_concat, c_crossattn):
        c_concat = self.concat_conditioner(c_concat)
        c_crossattn = self.crossattn_conditioner(c_crossattn)
        
        return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}


def noise_like(shape, device, repeat=False):
    noise = lambda: torch.randn(shape, device=device)
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    return repeat_noise() if repeat else noise()