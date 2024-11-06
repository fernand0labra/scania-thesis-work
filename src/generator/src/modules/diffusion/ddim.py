# Denoising Diffusion Implicit Models (extends DDPM)

import torch
import numpy as np

from tqdm import tqdm

from generator.src.modules.diffusion.utils import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

###

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.schedule = schedule
        self.ddpm_num_timesteps = model.num_timesteps


    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))

        setattr(self, name, attr)


    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        self.ddim_timesteps = make_ddim_timesteps(ddim_discretize, ddim_num_steps, self.ddpm_num_timesteps,verbose)

        self.register_buffer('betas',                               to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod',                      to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',                 to_torch(self.model.alphas_cumprod_prev))

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',                 to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',       to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod',        to_torch(np.log( 1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod',           to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',         to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # DDIM sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphas_cumprod.cpu(), self.ddim_timesteps, ddim_eta, verbose)

        self.register_buffer('ddim_sigmas',                         ddim_sigmas)
        self.register_buffer('ddim_alphas',                         ddim_alphas)
        self.register_buffer('ddim_alphas_prev',                    ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas',          np.sqrt(1. - ddim_alphas))

        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt((1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    @torch.no_grad()
    def sample(self, S, batch_size, shape, conditioning=None, callback=None, normals_sequence=None, img_callback=None, quantize_x0=False,  eta=0., mask=None, x0=None,
               temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, verbose=True, x_T=None, log_every_t=100, unconditional_guidance_scale=1.,
               unconditional_conditioning=None, **kwargs):
        
        if conditioning is not None:
            if isinstance(conditioning, dict):
                y_batch_size = conditioning[list(conditioning.keys())[0]].shape[0]
                if y_batch_size != batch_size:           print(f"Warning: Got {y_batch_size} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:  print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        
        # Sampling
        C, H, W = shape;  size = (batch_size, C, H, W)
        # print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        return self.ddim_sampling(conditioning, size, x_T, False, callback, None, quantize_x0, mask, x0, img_callback, log_every_t, temperature, noise_dropout, 
                                  score_corrector, corrector_kwargs, unconditional_guidance_scale, unconditional_conditioning,)


    @torch.no_grad()
    def ddim_sampling(self, cond, shape, x_T=None, ddim_use_original_steps=False, callback=None, timesteps=None, quantize_denoised=False, mask=None, x0=None, 
                      img_callback=None, log_every_t=100, temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        
        batch = shape[0]
        device = self.model.betas.device
        img = torch.randn(shape, device=device) if x_T is None else x_T

        if timesteps is None:              timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif not ddim_use_original_steps:  timesteps = self.ddim_timesteps[:int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        # print(f"Running DDIM Sampling with {total_steps} timesteps")

        for i, step in enumerate(time_range):  # tqdm( , desc='DDIM Sampler', total=total_steps)
            ts = torch.full((batch,), step, device=device, dtype=torch.long)
            if mask is not None:  img = self.model.q_sample(x0, ts) * mask + (1. - mask) * img

            index = total_steps - i - 1
            img, pred_x0 = self.p_sample_ddim(img, cond, ts, index, False, ddim_use_original_steps, quantize_denoised, temperature, noise_dropout, score_corrector, 
                                              corrector_kwargs, unconditional_guidance_scale, unconditional_conditioning)

            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates


    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False, temperature=1., noise_dropout=0., 
                      score_corrector=None, corrector_kwargs=None, unconditional_guidance_scale=1., unconditional_conditioning=None):
        
        b, *_, device = *x.shape, x.device
        alphas = self.model.alphas_cumprod                                  if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev                        if use_original_steps else self.ddim_alphas_prev
        sigmas = self.model.ddim_sigmas_for_original_num_steps              if use_original_steps else self.ddim_sigmas
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod    if use_original_steps else self.ddim_sqrt_one_minus_alphas

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2);  t_in = torch.cat([t] * 2);  c_in = torch.cat([unconditional_conditioning, c])
            
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        # Select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1),                  alphas[index],                  device=device)
        a_prev = torch.full((b, 1, 1, 1),               alphas_prev[index],             device=device)
        sigma_t = torch.full((b, 1, 1, 1),              sigmas[index],                  device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1),    sqrt_one_minus_alphas[index],   device=device)

        # Current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t  # Direction pointing to x_t
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0