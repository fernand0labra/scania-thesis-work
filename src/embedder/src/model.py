import clip
import torch
import kornia
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from functools import partial

from main import instantiate_from_config
from embedder.src.dataloader import ImagePaths
from embedder.src.modules.archs.codebook import VectorQuantizer
from embedder.src.modules.archs.autoencoder import Encoder, Decoder
# from generator.src.modules.archs.autoencoder import Encoder, Decoder
from generator.src.modules.distributions import DiagonalGaussianDistribution

###

class VQModel(pl.LightningModule):
    def __init__(self, ddconfig, lossconfig, n_embed, embed_dim, ckpt_path=None, ignore_keys=[], image_key="image", colorize_nlabels=None, 
                 monitor=None, remap=None, sane_index_shape=False,):
        super().__init__()

        self.automatic_optimization=False  # NOTE: Added
        self.image_key = image_key

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)

        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        if ckpt_path is not None:         self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if colorize_nlabels is not None:  self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:           self.monitor = monitor


    def init_from_ckpt(self, path, ignore_keys=list()):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(state_dict.keys())

        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):  print("Deleting key {} from state_dict.".format(k));  del state_dict[k]

        self.load_state_dict(state_dict, strict=False);  print(f"Restored from {path}")


    def encode(self, x):
        quant, emb_loss, info = self.quantize(self.quant_conv(self.encoder(x)))
        return quant, emb_loss, info


    def decode(self, quant):
        return self.decoder(self.post_quant_conv(quant))


    def decode_code(self, code_b):
        return self.decode(self.quantize.embed_code(code_b))


    def forward(self, input):
        quant, diff, _ = self.encode(input)
        return self.decode(quant), diff


    def get_input(self, batch, k):
        x = batch[k].to(self.device)
        if len(x.shape) == 3:  x = x[..., None]
        return x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
    

    def training_step(self, batch, batch_idx):  # optimizer_idx
        x = self.get_input(batch, self.image_key)        
        xrec, qloss = self(x)

        # x = self.get_input(batch, 'target')  # NOTE: Use when training different input/output

        # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
        aeopt, discopt = self.optimizers()

        # Autoencoder
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="train")
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=x.shape[0])
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=x.shape[0])

        aeopt.zero_grad(); self.manual_backward(aeloss); aeopt.step()

        # Discriminator
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="train")
        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=x.shape[0])
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=x.shape[0])

        discopt.zero_grad();  self.manual_backward(discloss);  discopt.step()


    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        # x = self.get_input(batch, 'target') # NOTE: Use when training different input/output

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        
        self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        # self.log_dict(log_dict_ae);  # self.log_dict(log_dict_disc)

        return self.log_dict


    def configure_optimizers(self):
        lr = self.learning_rate
        autoencoder_parameters = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.quantize.parameters()) + \
                                 list(self.quant_conv.parameters()) + list(self.post_quant_conv.parameters())
        
        return [torch.optim.Adam(autoencoder_parameters, lr=lr, betas=(0.5, 0.9)), \
                torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))], []


    def get_last_layer(self):
        return self.decoder.conv_out.weight


    def log_images(self, batch, **kwargs):
        
        x = self.get_input(batch, self.image_key).to(self.device)
        xrec, _ = self(x)

        if x.shape[1] > 3:  # Colorize with random projection
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)  

        log = dict();  log["inputs"] = x;  log["reconstructions"] = xrec
        return log


    def to_rgb(self, x):
        if not hasattr(self, "colorize"):  
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))

        x = F.conv2d(x, weight=self.colorize)
        return 2. * (x - x.min()) / (x.max() - x.min()) - 1.
    
###

class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim


    def encode(self, x):
        return self.quant_conv(self.encoder(x))


    def decode(self, h, force_not_quantize=False):
        return self.decoder(self.post_quant_conv(self.quantize(h)[0] if not force_not_quantize else h))
    
###

class AutoencoderKL(pl.LightningModule):
    def __init__(self, ddconfig, lossconfig, embed_dim, ckpt_path=None, ignore_keys=[], image_key="image", colorize_nlabels=None, monitor=None,):
        super().__init__()

        self.automatic_optimization=False  # NOTE: Added

        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)

        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        if colorize_nlabels is not None:  self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:           self.monitor = monitor
        if ckpt_path is not None:         self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)


    def init_from_ckpt(self, path, ignore_keys=list()):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]

        for k in list(state_dict.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):  print("Deleting key {} from state_dict.".format(k));  del state_dict[k]

        self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path}")


    def encode(self, x):  # x -> embedding -> moments -> posterior
        return DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)))


    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))


    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        return self.decode(posterior.sample() if sample_posterior else posterior.mode()), posterior


    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:  x = x[..., None]
        return x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()


    def training_step(self, batch, batch_idx):  # optimizer_idx
        inputs = self.get_input(batch, self.image_key)  
        reconstructions, posterior = self(inputs)

        # inputs = self.get_input(batch, 'target')  # NOTE: Use when training different input/output

        # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
        aeopt, discopt = self.optimizers()

        # Train encoder + decoder + logvar
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step, last_layer=self.get_last_layer(), split="train")
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        aeopt.zero_grad(); self.manual_backward(aeloss); aeopt.step()

        # Train the discriminator
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step, last_layer=self.get_last_layer(), split="train")
        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)

        discopt.zero_grad(); self.manual_backward(discloss); discopt.step()


    def validation_step(self, batch, batch_idx):
        
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        # inputs = self.get_input(batch, 'target')  # NOTE: Use when training different input/output

        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step, last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step, last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        # self.log_dict(log_dict_ae); self.log_dict(log_dict_disc)
        
        return self.log_dict


    def configure_optimizers(self):
        lr = self.learning_rate
        autoencoder_params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + \
                             list(self.quant_conv.parameters())+ list(self.post_quant_conv.parameters())
        
        return [torch.optim.Adam(autoencoder_params, lr=lr, betas=(0.5, 0.9)), 
                torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))], []


    def get_last_layer(self):
        return self.decoder.conv_out.weight


    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device)

        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:  # Colorize with random projection
                assert xrec.shape[1] > 3
                
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)

            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec

        log["inputs"] = x

        return log


    def to_rgb(self, x):
        assert self.image_key == "segmentation"

        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))

        x = F.conv2d(x, weight=self.colorize)
        return 2.*(x - x.min()) / (x.max() - x.min()) - 1.

###

class SpatialRescaler(nn.Module):
    def __init__(self, n_stages=1, method='bilinear', multiplier=0.5, in_channels=3, out_channels=None, bias=False):
        super().__init__()

        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']

        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None

        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)


    def forward(self,x):
        for _ in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        return self.channel_mapper(x) if self.remap_output else x


    def encode(self, x):
        return self(x)

###

class FrozenClipImageEmbedder(nn.Module):  # Uses the CLIP image encoder
    def __init__(self, model, jit=False, device='cuda' if torch.cuda.is_available() else 'cpu', antialias=False,):
        super().__init__()
        
        self.antialias = antialias
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)


    def preprocess(self, x):  # Normalize to [0,1] and renormalize according to clip
        x = kornia.geometry.resize(x, (224, 224), interpolation='bicubic',align_corners=True, antialias=self.antialias)
        return kornia.enhance.normalize((x + 1.) / 2., self.mean, self.std)


    def forward(self, x):
        return self.model.encode_image(self.preprocess(x))  # x is assumed to be in range [-1,1]