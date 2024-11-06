import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from omegaconf.listconfig import ListConfig

from generator.src.modules.diffusion.utils import *
from generator.src.modules.archs.transformer import SpatialTransformer

###

def convert_module_to_f16(x):  pass
def convert_module_to_f32(x):  pass

###

class TimestepBlock(nn.Module):
    # Any module where forward() takes timestep embeddings as a second argument.

    @abstractmethod  # Apply the module to `x` given `emb` timestep embeddings
    def forward(self, x, emb):  pass 

###

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    # A sequential module that passes timestep embeddings to the children that support it as an extra input.

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):         x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):  x = layer(x, context)
            else:                                        x = layer(x)
        return x

###

class Upsample(nn.Module):
    # An upsampling layer with an optional convolution.

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.dims = dims                                # Determines if the signal is 1D, 2D, or 3D
        self.channels = channels                        # Channels in the inputs and outputs
        self.use_conv = use_conv                        # Bool determining if a convolution is applied
        self.out_channels = out_channels or channels    

        if use_conv:  self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)


    def forward(self, x):
        assert x.shape[1] == self.channels

        x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest") if self.dims == 3 else \
            F.interpolate(x, scale_factor=2, mode="nearest")

        return self.conv(x) if self.use_conv else x

###

class Downsample(nn.Module):
    # A downsampling layer with an optional convolution

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()

        self.dims = dims                                # Determines if the signal is 1D, 2D, or 3D
        self.channels = channels                        # Channels in the inputs and outputs
        self.use_conv = use_conv                        # Bool determining if a convolution is applied
        self.out_channels = out_channels or channels

        stride = 2 if dims != 3 else (1, 2, 2)

        self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding) if use_conv else \
                  avg_pool_nd(dims, kernel_size=stride, stride=stride)
            

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

###

class ResBlock(TimestepBlock):
    # A residual block that can optionally change the number of channels.

    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False, use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False,):
        super().__init__()

        self.dropout = dropout                          # Rate of dropout
        self.channels = channels                        # Number of input channels
        self.use_conv = use_conv                        # Use a spatial convolution instead of 1x1 convolution to change channels in skip connection (when out_channels)
        self.updown = up or down                        # Use this block for upsampling or downsampling
        self.emb_channels = emb_channels                # Number of timestep embedding channels
        self.use_checkpoint = use_checkpoint            # Use gradient checkpointing on this module 
        self.out_channels = out_channels or channels    # Number of out channels (if specified)
        self.use_scale_shift_norm = use_scale_shift_norm

        if up:      self.h_upd = Upsample(channels, False, dims);    self.x_upd = Upsample(channels, False, dims)
        elif down:  self.h_upd = Downsample(channels, False, dims);  self.x_upd = Downsample(channels, False, dims)
        else:       self.h_upd = self.x_upd = nn.Identity()

        if self.out_channels == channels:  self.skip_connection = nn.Identity()
        elif use_conv:                     self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:                              self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

        self.in_layers = nn.Sequential(normalization(channels),
                                       nn.SiLU(),
                                       conv_nd(dims, channels, self.out_channels, 3, padding=1),)

        self.emb_layers = nn.Sequential(nn.SiLU(),
                                        linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels,),)
        
        self.out_layers = nn.Sequential(normalization(self.out_channels),
                                        nn.SiLU(),
                                        nn.Dropout(p=dropout),
                                        zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),)


    def forward(self, x, emb):
        # Apply the block to a Tensor, conditioned on a timestep embedding
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)


    def _forward(self, x, emb):
        if self.updown:  
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_conv(self.h_upd(in_rest(x)))
            x = self.x_upd(x)
        else:            
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_rest(out_norm(h) * (1 + scale) + shift)
        else:
            h = self.out_layers(h + emb_out)

        return self.skip_connection(x) + h

###

class AttentionBlock(nn.Module):
    # An attention block that allows spatial positions to attend to each other.
    # https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.

    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False):
        super().__init__()

        self.channels = channels
        if num_head_channels == -1:  self.num_heads = num_heads
        else:                        assert (channels % num_head_channels == 0), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"; \
                                     self.num_heads = channels // num_head_channels

        self.norm = normalization(channels)
        self.use_checkpoint = use_checkpoint
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))
        self.attention = QKVAttention(self.num_heads)


    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)  # NOTE: May not be correct


    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        h = self.proj_out(self.attention(self.qkv(self.norm(x))))
        return (x + h).reshape(b, c, *spatial)

###

class QKVAttention(nn.Module):
    # A module which performs QKV attention

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        input:  QKV            [N x (H x 3 x C) x T]
        output: Post-attention [N x (H x C) x T]
        """

        bs, width, length = qkv.shape                                               # assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        scale = 1 / math.sqrt(math.sqrt(ch))

        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)                    # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)

        return th.einsum("bts, bcs -> bct", weight, v).reshape(bs, -1, length)

###

class UNetModel(nn.Module):
    # The full UNet model with attention and timestep embedding
    """
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially increased efficiency.
    """

    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), 
                 conv_resample=True, dims=2, num_classes=None, use_checkpoint=False, use_fp16=False, num_heads=-1, num_head_channels=-1, num_heads_upsample=-1, 
                 use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, use_spatial_transformer=False, transformer_depth=1, 
                 context_dim=None, n_embed=None, legacy=True,):
        super().__init__()

        if context_dim is not None:
            if type(context_dim) == ListConfig:  
                context_dim = list(context_dim)

        if num_heads_upsample == -1:  num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels                          # Channels in the input Tensor
        self.model_channels = model_channels                    # Base channel count for the model
        self.out_channels = out_channels                        # Channels in the output Tensor
        self.num_res_blocks = num_res_blocks                    # Number of residual blocks per downsample
        self.attention_resolutions = attention_resolutions      # Collection of downsample rates at which attention will take place
        self.dropout = dropout                                  # Dropout probability
        self.channel_mult = channel_mult                        # Channel multiplier for each level of the UNet
        self.conv_resample = conv_resample                      # Use learned convolutions for upsampling and downsampling
        self.num_classes = num_classes                          # Class-conditional with `num_classes` classes
        self.use_checkpoint = use_checkpoint                    # Use gradient checkpointing to reduce memory usage
        self.num_heads = num_heads                              # Number of attention heads in each attention layer
        self.num_head_channels = num_head_channels              # Ignore num_heads and instead use a fixed channel width per attention head
        self.num_heads_upsample = num_heads_upsample            # (Deprecated) Works with num_heads to set a different number of heads for upsampling
        self.predict_codebook_ids = n_embed is not None
        self.dtype = th.float16 if use_fp16 else th.float32
        self._feature_size = model_channels

        ds = 1
        channels = model_channels
        time_embed_dim = model_channels * 4
        input_block_chans = [model_channels]

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))])

        self.time_embed = nn.Sequential(linear(model_channels, time_embed_dim),
                                        nn.SiLU(),
                                        linear(time_embed_dim, time_embed_dim),)

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(channels, time_embed_dim, dropout, mult * model_channels, False, use_scale_shift_norm, dims, use_checkpoint)]
                
                channels = mult * model_channels
                self._feature_size += channels

                if ds in attention_resolutions:
                    if num_head_channels == -1:  dim_head = channels // num_heads
                    else:                        dim_head = num_head_channels;  num_heads = channels // num_head_channels
                    if legacy:                   dim_head = channels // num_heads if use_spatial_transformer else num_head_channels

                    layers.append(AttentionBlock(channels, num_heads, dim_head, use_checkpoint,) if not use_spatial_transformer else \
                                  SpatialTransformer(channels, num_heads, dim_head, transformer_depth, 0, context_dim))
                    
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(channels)

            if level != len(channel_mult) - 1:
                
                out_ch = channels
                self.input_blocks.append(TimestepEmbedSequential(ResBlock(channels, time_embed_dim, dropout, out_ch, False, use_scale_shift_norm, dims, use_checkpoint, down=True,) if resblock_updown else \
                                                                 Downsample(channels, conv_resample, dims=dims, out_channels=out_ch)))
                channels = out_ch

                input_block_chans.append(channels)
                self._feature_size += channels
                ds *= 2

        if num_head_channels == -1:  dim_head = channels // num_heads
        else:                        dim_head = num_head_channels;  num_heads = channels // num_head_channels

        if legacy:                   dim_head = channels // num_heads if use_spatial_transformer else num_head_channels
        
        self.middle_block = TimestepEmbedSequential(ResBlock(channels, time_embed_dim, dropout, None, False, use_scale_shift_norm, dims, use_checkpoint,),
                                                    AttentionBlock(channels, num_heads, dim_head, use_checkpoint) if not use_spatial_transformer else \
                                                    SpatialTransformer(channels, num_heads, dim_head, transformer_depth, 0, context_dim),
                                                    ResBlock(channels, time_embed_dim, dropout, None, False, use_scale_shift_norm, dims, use_checkpoint, ),)
        
        self._feature_size += channels
        self.output_blocks = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [ResBlock(channels + input_block_chans.pop(), time_embed_dim, dropout, model_channels * mult, False, use_scale_shift_norm, dims, use_checkpoint,)]
                channels = model_channels * mult

                if ds in attention_resolutions:
                    if num_head_channels == -1:  dim_head = channels // num_heads
                    else:                        dim_head = num_head_channels;  num_heads = channels // num_head_channels
                        
                    if legacy:  dim_head = channels // num_heads if use_spatial_transformer else num_head_channels

                    layers.append(AttentionBlock(channels, num_heads_upsample, dim_head, use_checkpoint) if not use_spatial_transformer else \
                                  SpatialTransformer(channels, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim))
                    
                if level and i == num_res_blocks:
                    out_ch = channels
                    layers.append(ResBlock(channels, time_embed_dim, dropout, out_ch, False,  use_scale_shift_norm, dims, use_checkpoint, up=True,) if resblock_updown else \
                                  Upsample(channels, conv_resample, dims, out_ch))
                    
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += channels

        self.out = nn.Sequential(normalization(channels),
                                 nn.SiLU(),
                                 zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),)
        
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(normalization(channels),
                                              conv_nd(dims, model_channels, n_embed, 1),)


    def convert_to_fp16(self):  # Convert the torso of the model to float16.
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)


    def convert_to_fp32(self):  # Convert the torso of the model to float32.
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)


    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
        # x:          [N x C x ...] Tensor of inputs
        # y:          [N] Tensor of labels, if class-conditional
        # timesteps:  1D batch of timesteps.
        # context:    Conditioning plugged in via crossattn

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels, repeat_only=False))
        if self.num_classes is not None:  emb += self.label_emb(y)
        hs = [];  h = x.type(self.dtype)

        for module in self.input_blocks:   h = module(h, emb, context); hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:  h = module(th.cat([h, hs.pop()], dim=1), emb, context)

        h = h.type(x.dtype)
        return self.id_predictor(h) if self.predict_codebook_ids else self.out(h)