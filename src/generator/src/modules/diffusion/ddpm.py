# Denoising Diffusion Probabilistic Models

# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
# https://github.com/CompVis/taming-transformers

import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl

from tqdm import tqdm
from functools import partial
from einops import rearrange, repeat
from contextlib import contextmanager
from torchvision.utils import make_grid

from generator.src.modules.losses.metrics import LitEma
from generator.src.utils import exists, default, count_params, instantiate_from_config
from generator.src.modules.diffusion.utils import make_beta_schedule, extract_into_tensor, noise_like

###

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode does not change anymore."""
    return self

###

def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2

###

class DDPM(pl.LightningModule):  # DDPM with Gaussian diffusion (Image space)
    def __init__(self, unet_config, timesteps=1000, beta_schedule="linear", loss_type="l2", ckpt_path=None, ignore_keys=[], load_only_unet=False, monitor="val/loss",
                 use_ema=True, first_stage_key="image", image_size=256, channels=3, log_every_t=100, clip_denoised=True, linear_start=1e-4, linear_end=2e-2,
                 cosine_s=8e-3, given_betas=None, original_elbo_weight=0., v_posterior=0., l_simple_weight=1., conditioning_key=None, parameterization="eps",
                 scheduler_config=None, use_positional_encodings=False, learn_logvar=False, logvar_init=0.,):
        super().__init__()

        self.parameterization = parameterization
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")

        self.use_ema = use_ema
        self.channels = channels
        self.loss_type = loss_type
        self.image_size = image_size
        self.cond_stage_model = None
        self.log_every_t = log_every_t
        self.learn_logvar = learn_logvar
        self.clip_denoised = clip_denoised
        self.first_stage_key = first_stage_key
        self.use_scheduler = scheduler_config is not None
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        count_params(self.model, verbose=True)  # NOTE: What is this for

        self.v_posterior = v_posterior # Posterior variance as sigma = (1-v) * beta_tilde + v * beta
        self.l_simple_weight = l_simple_weight
        self.original_elbo_weight = original_elbo_weight

        if self.use_scheduler:      self.scheduler_config = scheduler_config
        if monitor is not None:     self.monitor = monitor
        if ckpt_path is not None:   self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,  linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        to_torch = partial(torch.tensor, dtype=torch.float32)

        betas = given_betas if exists(given_betas) else \
                make_beta_schedule(beta_schedule, timesteps, linear_start, linear_end, cosine_s)
        
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        self.linear_end = linear_end
        self.linear_start = linear_start

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',   to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',    to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',       to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',     to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) + self.v_posterior * betas

        # Log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_variance',              to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped',  to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',            to_torch( betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',            to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":   lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":  lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:                                raise NotImplementedError("mu not supported")

        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()


    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            # if context is not None:  print(f"{context}: Switched to EMA weights")

        try:        yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                # if context is not None:  print(f"{context}: Restored training weights")


    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        state_dict = torch.load(path, map_location="cpu")
        if "state_dict" in list(state_dict.keys()):  state_dict = state_dict["state_dict"]
            
        for k in list(state_dict.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):  print("Deleting key {} from state_dict.".format(k));  del state_dict[k]

        missing, unexpected = self.load_state_dict(state_dict, strict=False) if not only_model else self.model.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        
        if len(missing) > 0:      print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:   print(f"Unexpected Keys: {unexpected}")


    def predict_start_from_noise(self, x_t, t, noise):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)
    

    def q_mean_variance(self, x_start, t):  # Get the distribution q(x_t | x_0)  -> (mean, variance, log_variance)
        # x_start:        [N x C x ...] Tensor of noiseless inputs.
        # t:                            Number of diffusion steps (minus 1). Here, 0 means one step.
        
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)

        return mean, variance, log_variance


    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start + \
                          extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)

        if self.parameterization == "eps":      x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":     x_recon = model_out
        if clip_denoised:                       x_recon.clamp_(-1., 1.)

        return self.q_posterior(x_start=x_recon, x_t=x, t=t)


    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        batch, *_, device = *x.shape, x.device

        noise = noise_like(x.shape, device, repeat_noise)
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        nonzero_mask = (1 - (t == 0).float()).reshape(batch, *((1,) * (len(x.shape) - 1)))  # No noise when t == 0

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        batch = shape[0]
        intermediates = [img]
        device = self.betas.device
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((batch,), i, device=device, dtype=torch.long), clip_denoised=self.clip_denoised)

            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)

        if return_intermediates:  return img, intermediates
        else:                     return img


    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), return_intermediates=return_intermediates)


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:  loss = loss.mean()

        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(target, pred) if mean else \
                   torch.nn.functional.mse_loss(target, pred, reduction='none')
            
        else:  raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss


    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        if self.parameterization == "eps":   target = noise
        elif self.parameterization == "x0":  target = x_start
        else:                                raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss_dict = {}
        log_prefix = 'train' if self.training else 'val'

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])
        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss_simple = loss.mean() * self.l_simple_weight
        loss = loss_simple + self.original_elbo_weight * loss_vlb
        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict


    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long(), *args, **kwargs)


    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:  x = x[..., None]
        return rearrange(x, 'b h w c -> b c h w').to(memory_format=torch.contiguous_format).float()


    def shared_step(self, batch):
        return self(self.get_input(batch, self.first_stage_key))


    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            self.log('lr_abs', self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)

        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}

        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)


    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:  self.model_ema(self.model)


    def _get_rows_from_list(self, samples):
        denoise_grid = rearrange(samples,      'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        return make_grid(denoise_grid, nrow=len(samples))


    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        x = self.get_input(batch, self.first_stage_key)

        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]

        # Get diffusion row
        x_start = x[:n_row]
        diffusion_row = list()

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row).to(self.device).long()

                x_noisy = self.q_sample(x_start=x_start, t=t, noise=torch.randn_like(x_start))
                diffusion_row.append(x_noisy)

        log = dict();  log["inputs"] = x;  log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:  # Get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples;  log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
                return log if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0 else \
                       {key: log[key] for key in return_keys}
        return log


    def configure_optimizers(self):
        params = list(self.model.parameters())
        if self.learn_logvar:  params = params + [self.logvar]
        return torch.optim.AdamW(params, lr=self.learning_rate)

###

class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()

        self.conditioning_key = conditioning_key
        self.diffusion_model = instantiate_from_config(diff_model_config)
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:           return self.diffusion_model(x, t)
        elif self.conditioning_key == 'adm':        return self.diffusion_model(x, t, y=c_crossattn[0])
        elif self.conditioning_key == 'concat':     return self.diffusion_model(torch.cat([x] + c_concat, dim=1), t)
        elif self.conditioning_key == 'crossattn':  return self.diffusion_model(x, t, context=torch.cat(c_crossattn, 1))
        elif self.conditioning_key == 'hybrid':     return self.diffusion_model(torch.cat([x] + c_concat, dim=1), t, context=torch.cat(c_crossattn, 1))
        else:                                       raise NotImplementedError()