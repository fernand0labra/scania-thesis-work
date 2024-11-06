import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm
from PIL import Image
from einops import rearrange, repeat
from torchvision.utils import make_grid
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from generator.src.modules.diffusion.ddim import DDIMSampler
from generator.src.modules.diffusion.ddpm import DDPM, disabled_train
from embedder.src.model import VQModelInterface, AutoencoderKL
from generator.src.modules.diffusion.utils import extract_into_tensor, noise_like
from generator.src.modules.distributions import normal_kl, DiagonalGaussianDistribution
from generator.src.utils import log_txt_as_img, default, ismap, isimage, mean_flat, instantiate_from_config

###

__conditioning_keys__ = {'concat': 'c_concat', 'crossattn': 'c_crossattn', 'adm': 'y'}

n_x = 0;  outdir = '/home/ubuntu/scania-raw-diff/src/auxiliary/imgs/'

###

class LatentDiffusion(DDPM):
    def __init__(self, first_stage_config, cond_stage_config, num_timesteps_cond=None, cond_stage_key="image", cond_stage_trainable=False,  concat_mode=True, 
                 cond_stage_forward=None, conditioning_key=None, scale_factor=1.0, scale_by_std=False, *args, **kwargs):

        self.bbox_tokenizer = None  
        self.clip_denoised = False
        self.restarted_from_ckpt = False
        self.scale_by_std = scale_by_std
        self.cond_stage_forward = cond_stage_forward
        self.num_timesteps_cond = default(num_timesteps_cond, 1)

        assert self.num_timesteps_cond <= kwargs['timesteps']

        # Backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:                     conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':  conditioning_key = None


        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_key = cond_stage_key
        self.cond_stage_trainable = cond_stage_trainable

        try:     self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:  self.num_downs = 0

        if not scale_by_std:  self.scale_factor = scale_factor
        else:                 self.register_buffer('scale_factor', torch.tensor(scale_factor))

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        
        ckpt_path = kwargs.pop("ckpt_path", None)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, kwargs.pop("ignore_keys", []))
            self.restarted_from_ckpt = True


    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()


    def instantiate_first_stage(self, config):
        self.first_stage_model = instantiate_from_config(config).eval()
        self.first_stage_model.train = disabled_train
        
        for param in self.first_stage_model.parameters():
            param.requires_grad = False


    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model

            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None

            else:
                self.cond_stage_model = instantiate_from_config(config).eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            self.cond_stage_model = instantiate_from_config(config)


    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device), force_not_quantize=force_no_decoder_quantization))

        denoise_grid = torch.stack(denoise_row)                                 # [n_log_step,  n_row,  C, H, W]
        denoise_grid = rearrange(denoise_grid,  'n b c h w -> b n c h w')       # [n_row,  n_log_step,  C, H, W]
        denoise_grid = rearrange(denoise_grid,  'b n c h w -> (b n) c h w')     # [(n_row, n_log_step), C, H, W]

        return make_grid(denoise_grid, nrow=len(denoise_row))


    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)


    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.

        input:  [N x C x ...]   Tensor of inputs.
        output: [N]             KL values (in bits), one per batch element.
        """

        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=torch.tensor([self.num_timesteps - 1] * x_start.shape[0], device=x_start.device))
        
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)


    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):  z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):                z = encoder_posterior
        elif isinstance(encoder_posterior, tuple):                       z = encoder_posterior[0] # VQModel instead of VQModelInterface                                 
        else:                                                            raise NotImplementedError(f"Encoder_posterior of type '{type(encoder_posterior)}' not implemented")
        
        return self.scale_factor * z


    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                return c.mode() if isinstance(c, DiagonalGaussianDistribution) else c
            else:
                return self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            return getattr(self.cond_stage_model, self.cond_stage_forward)(c)


    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        return torch.cat([y, x], dim=-1)


    def delta_border(self, h, w):
        # Normalized distance to image border, with min distance = 0 at border and max dist = 0.5 at image center
        mesh_grid = self.meshgrid(h, w) / torch.tensor([h - 1, w - 1]).view(1, 1, 2)    # Normalized over lower right corner
        
        dist_left_up = torch.min(mesh_grid, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - mesh_grid, dim=-1, keepdims=True)[0]
        
        return torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]  # Edge distance


    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"], self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting, self.split_input_params["clip_min_tie_weight"], self.split_input_params["clip_max_tie_weight"])

            weighting *= L_weighting.view(1, 1, Ly * Lx).to(device)

        return weighting


    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):
        _, _, h, w = x.shape

        # Number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1;  Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            
            unfold = torch.nn.Unfold(**fold_params)
            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)                                                        # Normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf), dilation=1, padding=0, stride=(stride[0] * uf, stride[1] * uf))

            unfold = torch.nn.Unfold(**fold_params)
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)                                              # Normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df), dilation=1, padding=0, stride=(stride[0] // df, stride[1] // df))

            unfold = torch.nn.Unfold(**fold_params)
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)                                            # Normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:  raise NotImplementedError

        return fold, unfold, normalization, weighting


    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False, img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None, log_every_t=None):
        
        if batch_size is not None:  b = batch_size if batch_size is not None else shape[0]; \
                                    shape = [batch_size] + list(shape)
        else:                       b = batch_size = shape[0]
        
        if x_T is None:  img = torch.randn(shape, device=self.device)
        else:            img = x_T
        
        intermediates = []
        if cond is not None:
            cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size] if not isinstance(cond, dict) else \
                   {key: cond[key][:batch_size] if not isinstance(cond[key], list) else list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
        
        timesteps = self.num_timesteps
        if not log_every_t:  log_every_t = self.log_every_t
        if start_T is not None:         timesteps = min(timesteps, start_T)
        if type(temperature) == float:  temperature = [temperature] * timesteps
        
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation', total=timesteps) if verbose else reversed(range(0, timesteps))
        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)

            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts, self.clip_denoised, False, False, quantize_denoised, True, temperature[i], noise_dropout, score_corrector, corrector_kwargs)
            
            if mask is not None:
                assert x0 is not None
                img = self.q_sample(x0, ts) * mask + (1. - mask) * img

            if callback:                                    callback(i)
            if img_callback:                                img_callback(img, i)
            if i % log_every_t == 0 or i == timesteps - 1:  intermediates.append(x0_partial)

        return img, intermediates


    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'Rather not use custom rescaling and std-rescaling simultaneously'

            x = super().get_input(batch, self.first_stage_key).to(self.device)  # Set rescale weight to 1./std of encodings
            z = self.get_first_stage_encoding(self.encode_first_stage(x)).detach()
            
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"Setting self.scale_factor to {self.scale_factor}")


    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False, cond_key=None, return_original_cond=False, batch_size=None):
        x = super().get_input(batch, k)
        if batch_size is not None:  x = x[:batch_size]
        x = x.to(self.device)

        z = self.get_first_stage_encoding(self.encode_first_stage(x)).detach()

        ### NOTE Denoising representation :: Added fernand0labra
        # if n_x < 100:
        #     if n_x % 10 == 0 :
        #         Image.fromarray(np.transpose((x * 255).cpu().numpy().astype(np.uint8), (0, 2, 3, 1))[0]).save(outdir + "x_orig_{0}.png".format(n_x//10))
        #         Image.fromarray(np.transpose((z * 255).cpu().numpy().astype(np.uint8), (0, 2, 3, 1))[0]).save(outdir + "x_latent_{0}.png".format(n_x//10))

        ###

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox']:  xc = batch[cond_key]
                elif cond_key == 'class_label':                  xc = batch
                else:                                            xc = super().get_input(batch, cond_key).to(self.device)
            else:                                                xc = x

            if not self.cond_stage_trainable or force_c_encode:
                    c = self.get_learned_conditioning(xc) if isinstance(xc, dict) or isinstance(xc, list) else \
                        self.get_learned_conditioning(xc.to(self.device))
            else:   c = xc

            if batch_size is not None:  c = c[:batch_size]
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {__conditioning_keys__[self.model.conditioning_key]: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None;  xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}

        x_target = None
        # x_target = super().get_input(batch, "target").to(self.device)
        # x_target = self.get_first_stage_encoding(self.encode_first_stage(x_target)).detach()

        out = [z, c, x_target]  
        if return_first_stage_outputs:  out.extend([x, self.decode_first_stage(z)])
        if return_original_cond:        out.append(xc)

        return out


    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:  z = torch.argmax(z.exp(), dim=1).long()
            z = rearrange(self.first_stage_model.quantize.get_codebook_entry(z, shape=None), 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                
                ks = self.split_input_params["ks"]              # Kernel Size eg. (128, 128)
                uf = self.split_input_params["vqf"]             # Vector-quantized Features
                stride = self.split_input_params["stride"]      # Stride eg. (64, 64)

                _, _, h, w = z.shape
                if ks[0] > h or ks[1] > w:              ks = (min(ks[0], h), min(ks[1], w));                print("reducing Kernel")
                if stride[0] > h or stride[1] > w:      stride = (min(stride[0], h), min(stride[1], w));    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)                                               # (bn, nc * prod(**ks), L)
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))     # (bn, nc, ks[0], ks[1], L ) :: Reshape to img shape

                # Apply model loop over last dim
                output_list = [self.first_stage_model.decode(z[:, :, :, :, i]) for i in range(z.shape[-1])] if not isinstance(self.first_stage_model, VQModelInterface) else \
                              [self.first_stage_model.decode(z[:, :, :, :, i], force_not_quantize=predict_cids or force_not_quantize)  for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1) * weighting           # (bn, nc, ks[0], ks[1], L)
                o = o.view((o.shape[0], -1, o.shape[-1]))                   # (bn, nc * ks[0] * ks[1], L) :: Reverse reshape to img shape
                
                return fold(o) / normalization  # Norm is shape (1, 1, h, w)

        return self.first_stage_model.decode(z) if not isinstance(self.first_stage_model, VQModelInterface) else \
               self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)


    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                self.split_input_params['original_image_size'] = x.shape[-2:]

                ks = self.split_input_params["ks"]              # Kernel Size eg. (128, 128)
                df = self.split_input_params["vqf"]             # Vector-quantized Features
                stride = self.split_input_params["stride"]      # Stride eg. (64, 64)

                _, _, h, w = x.shape
                if ks[0] > h or ks[1] > w:              ks = (min(ks[0], h), min(ks[1], w));                print("reducing Kernel")
                if stride[0] > h or stride[1] > w:      stride = (min(stride[0], h), min(stride[1], w));    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)

                z = unfold(x)                                               # (bn, nc * prod(**ks), L)
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))     # (bn, nc, ks[0], ks[1], L ) :: Reshape to img shape

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i]) for i in range(z.shape[-1])]
                
                o = torch.stack(output_list, axis=-1) * weighting
                o = o.view((o.shape[0], -1, o.shape[-1]))                   # (bn, nc * ks[0] * ks[1], L) :: Reverse reshape to img shape

                return fold(o) / normalization

        return self.first_stage_model.encode(x)


    def shared_step(self, batch, **kwargs):
        return self(*self.get_input(batch, self.first_stage_key))


    def forward(self, x, c, x_target, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()

        if self.model.conditioning_key is not None:
            assert c is not None
            
            if self.cond_stage_trainable:   c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  c = self.q_sample(x_start=c, t=self.cond_ids[t].to(self.device), noise=torch.randn_like(c.float()))

        return self.p_losses(x, c, t, x_target=x_target, *args, **kwargs)


    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):  pass  # Hybrid case, cond is exptected to be a dict
        else:
            if not isinstance(cond, list):  cond = [cond]
            cond = {'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn': cond}

        if hasattr(self, "split_input_params"):
            assert not return_ids;  assert len(cond) == 1  # TODO can only deal with one conditioning atm
              
            _, w = x_noisy.shape[-2:]
            ks = self.split_input_params["ks"]              # Kernel Size eg. (128, 128)
            stride = self.split_input_params["stride"]      # Stride eg. (64, 64)

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)                                             # (bn, nc * prod(**ks), L)
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))         # (bn, nc, ks[0], ks[1], L ) :: Reshape to img shape
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["image", "LR_image", "segmentation", 'bbox_img'] and self.model.conditioning_key:
                c = next(iter(cond.values()));  assert (len(c) == 1)  # TODO extend to list with more than one elem

                c = unfold(c[0])
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))     # (bn, nc, ks[0], ks[1], L )

                cond_list = [{next(iter(cond.keys())): [c[:, :, :, :, i]]} for i in range(c.shape[-1])]
            else:
                cond_list = [cond for i in range(z.shape[-1])]  # TODO make this more efficient

            # Apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0], tuple)

            o = torch.stack(output_list, axis=-1) * weighting
            o = o.view((o.shape[0], -1, o.shape[-1]))                       # (bn, nc * ks[0] * ks[1], L) :: Reverse reshape to img shape

            x_recon = fold(o) / normalization
        else:
            x_recon = self.model(x_noisy, t, **cond)

        ### NOTE Denoising representation :: Added fernand0labra
        global n_x, outdir

        # if n_x < 100:
        #     if n_x % 10 == 0 :
        #         Image.fromarray(np.transpose((x_noisy * 255).detach().cpu().numpy().astype(np.uint8), (0, 2, 3, 1))[0]).save(outdir + "x_latent_noisy_{0}.png".format(n_x//10))
                
        #         x_noise = self.decode_first_stage(x_noisy.detach())
        #         Image.fromarray(np.transpose((x_noise * 255).cpu().numpy().astype(np.uint8), (0, 2, 3, 1))[0]).save(outdir + "x_noisy_{0}.png".format(n_x//10))
        
        ###

        return x_recon[0] if isinstance(x_recon, tuple) and not return_ids else x_recon


    def p_losses(self, x_start, cond, t, noise=None, x_target=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start, t, noise)
        model_output = self.apply_model(x_noisy, t, cond)

        ### NOTE Denoising representation :: Added fernand0labra
        global n_x, outdir

        # if n_x < 100:
        #     if n_x % 10 == 0 :
        #         Image.fromarray(np.transpose((model_output.detach() * 255).cpu().numpy().astype(np.uint8), (0, 2, 3, 1))[0]).save(outdir + "x_latent_rec_{0}.png".format(n_x//10))
                
        #         x_rec = self.decode_first_stage(model_output.detach())
        #         x_rec = torch.clamp(x_rec.detach(), -1., 1.)
        #         Image.fromarray(np.transpose((x_rec * 255).cpu().numpy().astype(np.uint8), (0, 2, 3, 1))[0]).save(outdir + "x_rec_{0}.png".format(n_x//10))
        #     n_x = n_x + 1

        ###

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":     target = x_target # x_start
        elif self.parameterization == "eps":  target = noise
        else:                                 raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar.to(self.device)[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()

        loss = self.l_simple_weight * loss.mean()
        loss += (self.original_elbo_weight * loss_vlb)

        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict


    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False, return_x0=False, score_corrector=None, corrector_kwargs=None):
        model_out = self.apply_model(x, t, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":   x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":  x_recon = model_out
        else:                                raise NotImplementedError()

        if clip_denoised:      x_recon.clamp_(-1., 1.)
        if quantize_denoised:  x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        if return_codebook_ids:  return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:          return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:                    return model_mean, posterior_variance, posterior_log_variance


    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False, return_codebook_ids=False, quantize_denoised=False, return_x0=False, temperature=1., noise_dropout=0., 
                 score_corrector=None, corrector_kwargs=None):

        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x, c, t, clip_denoised, return_codebook_ids, quantize_denoised, return_x0, score_corrector, corrector_kwargs)

        if return_codebook_ids:  model_mean, _, model_log_variance, logits = outputs;  raise DeprecationWarning("Support dropped.");  
        elif return_x0:          model_mean, _, model_log_variance, x0 = outputs
        else:                    model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:  noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))  # No noise when t == 0

        if return_codebook_ids:  return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:                    return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False, x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False, mask=None, 
                      x0=None, img_callback=None, start_T=None, log_every_t=None):

        b = shape[0]
        device = self.betas.device
        img = torch.randn(shape, device=device) if x_T is None else x_T

        if not log_every_t:  log_every_t = self.log_every_t
        if timesteps is None:    timesteps = self.num_timesteps
        if start_T is not None:  timesteps = min(timesteps, start_T)
        if mask is not None:  assert x0 is not None;  assert x0.shape[2:3] == mask.shape[2:3]

        intermediates = [img]
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(range(0, timesteps))
        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                cond = self.q_sample(x_start=cond, t=self.cond_ids[ts].to(cond.device), noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised, quantize_denoised=quantize_denoised)
            if mask is not None:
                img = self.q_sample(x0, ts) * mask + (1. - mask) * img

            if callback:                                    callback(i)
            if img_callback:                                img_callback(img, i)
            if i % log_every_t == 0 or i == timesteps - 1:  intermediates.append(img)

        if return_intermediates:  return img, intermediates
        return img


    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None, verbose=True, timesteps=None, quantize_denoised=False, mask=None, x0=None, shape=None,**kwargs):
        if shape is None:  shape = (batch_size, self.channels, self.image_size, self.image_size)

        if cond is not None:
            cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size] if not isinstance(cond, dict) else \
            {key: cond[key][:batch_size] if not isinstance(cond[key], list) else list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
        
        return self.p_sample_loop(cond, shape, return_intermediates, x_T, verbose, timesteps=timesteps, quantize_denoised=quantize_denoised, mask=mask, x0=x0)


    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim, ddim_steps,**kwargs):
        return DDIMSampler(self).sample(ddim_steps,batch_size, (self.channels, self.image_size, self.image_size), cond,verbose=False,**kwargs) if ddim else \
               self.sample(cond=cond, batch_size=batch_size, return_intermediates=True,**kwargs)


    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None, quantize_denoised=True, inpaint=True, plot_denoise_rows=False, 
                   plot_progressive_rows=True, plot_diffusion_rows=True, **kwargs):
        
        use_ddim = ddim_steps is not None
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, return_first_stage_outputs=True, force_c_encode=True, return_original_cond=True, batch_size=N)
        
        N = min(x.shape[0], N);  n_row = min(x.shape[0], n_row)
        log = dict();  log["inputs"] = x;  log["reconstruction"] = xrec
        
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):  xc = self.cond_stage_model.decode(c);                                 log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:      xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"]);      log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':    xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"]);  log['conditioning'] = xc
            elif isimage(xc):                                                                                                   log["conditioning"] = xc
            
            if ismap(xc):  log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:  # Get diffusion row
            
            diffusion_row = list()
            z_start = z[:n_row]

            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row =  torch.stack(diffusion_row)                                         # [ n_log_step, n_row,  C, H, W]
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')                 # [ n_row, n_log_step,  C, H, W]
            diffusion_grid = rearrange(diffusion_grid,             'b n c h w -> (b n) c h w')  # [(n_row, n_log_step), C, H, W]
            
            log["diffusion_row"] = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])

        if sample:  # Get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, ddim_steps=ddim_steps,eta=ddim_eta)

            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL):
                with self.ema_scope("Plotting Quantized Denoised"):  # Display when quantizing x0 while sampling
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, ddim_steps=ddim_steps,eta=ddim_eta, quantize_denoised=True)

                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c, shape=(self.channels, self.image_size, self.image_size), batch_size=N)
            
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            return log if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0 else \
            {key: log[key] for key in return_keys}
        
        return log


    def configure_optimizers(self):
        params = list(self.model.parameters())
        
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        
        opt = torch.optim.AdamW(params, lr=self.learning_rate)

        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            print("Setting up LambdaLR scheduler...")

            return [opt], [{'scheduler': LambdaLR(opt, lr_lambda=instantiate_from_config(self.scheduler_config).schedule), 'interval': 'step', 'frequency': 1}]
        
        return opt


    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):  
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        
        x = nn.functional.conv2d(x, weight=self.colorize)
        return 2. * (x - x.min()) / (x.max() - x.min()) - 1.