import math
import torch
import numpy as np
import torch.nn as nn

from einops import rearrange

from generator.src.utils import instantiate_from_config
from generator.src.modules.archs.transformer import LinearAttention

###

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Build sinusoidal embeddings (Fairseq).
    This matches the implementation in Denoising Diffusion Probabilistic Models.
    This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    return torch.nn.functional.pad(emb, (0,1,0,0)) if embedding_dim % 2 == 1 else emb  # Zero pad

###

def nonlinearity(x):
    return x * torch.sigmoid(x)

###

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

###

def make_attn(in_channels, attn_type="vanilla"):
    print(f"Making attention of type '{attn_type}' with {in_channels} in_channels")

    if attn_type == "vanilla":  return AttnBlock(in_channels)
    elif attn_type == "none":   return nn.Identity(in_channels)
    else:                       return LinAttnBlock(in_channels)

###

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:  self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x) if self.with_conv else x

###

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:  self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(torch.nn.functional.pad(x, (0,1,0,1), mode="constant", value=0)) if self.with_conv else \
                         torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

###

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.dropout = torch.nn.Dropout(dropout)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels) if temb_channels > 0 else None

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = Normalize(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.use_conv_shortcut = conv_shortcut
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:  self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:                       self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x, temb):
        h = self.conv1(nonlinearity(self.norm1(x)))
        if temb is not None:  h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]
        h = self.norm2(nonlinearity(self.dropout(self.conv2(h))))

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:  x = self.conv_shortcut(x)
            else:                       x = self.nin_shortcut(x)

        return x + h

###

class LinAttnBlock(LinearAttention):
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)

###

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        self.norm = Normalize(in_channels)

        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_);  k = self.k(h_);  v = self.v(h_)

        # Compute attention
        b, c, h, w = q.shape
        q = q.reshape(b,c,h*w).permute(0,2,1)                                       # [b, hw,  c]
        k = k.reshape(b,c,h*w)                                                      # [b,  c, hw]

        w_ = torch.bmm(q, k)            # w[b, i, j] = sum_c q[b, i, c] k[b, c, j]    [b, hw, hw]
        w_ = torch.nn.functional.softmax(w_ * (int(c)**(-0.5)), dim=2)

        # Attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)                             # (b, hw of k, hw of q)    [b, hw, hw] 
                                      # h_[b, c, j] = sum_i v[b, c, i] w_[b, i, j]    [b,  c, hw]
        h_ = self.proj_out(torch.bmm(v,w_).reshape(b,c,h,w))

        return x + h_

###

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, resolution, z_channels, 
                 double_z=True, use_linear_attn=False, attn_type="vanilla", **ignore_kwargs):
        super().__init__()
        
        self.ch = ch
        self.temb_ch = 0
        self.resolution = resolution
        self.in_channels = in_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        if use_linear_attn: attn_type = "linear"

        # Downsampling
        curr_res = resolution
        self.down = nn.ModuleList()
        self.in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList();  attn = nn.ModuleList()
            block_in =  ch * self.in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out

                if curr_res in attn_resolutions:  attn.append(make_attn(block_in, attn_type=attn_type))

            down = nn.Module();  down.block = block;  down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv);  curr_res //= 2

            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # End
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, 2*z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        
        temb = None  # Timestep embedding

        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
                
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(hs[-1], temb)), temb)

        return self.conv_out(nonlinearity(self.norm_out(h)))  # End

###

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, resolution, z_channels, 
                 give_pre_end=False, tanh_out=False, use_linear_attn=False, attn_type="vanilla", **ignorekwargs):
        super().__init__()

        self.ch = ch
        self.temb_ch = 0
        self.tanh_out = tanh_out
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        if use_linear_attn: attn_type = "linear"

        # Compute in_ch_mult, block_in and curr_res at lowest res
        curr_res = resolution // 2**(self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format( self.z_shape, np.prod(self.z_shape)))
        
        block_in = ch*ch_mult[self.num_resolutions-1]
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)  # z to block_in

        # Middle
        self.mid = nn.Module()
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList();  attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]

            for _ in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch,  dropout=dropout))
                block_in = block_out
                
                if curr_res in attn_resolutions:  attn.append(make_attn(block_in, attn_type=attn_type))

            up = nn.Module();  up.block = block;  up.attn = attn
            if i_level != 0:  
                up.upsample = Upsample(block_in, resamp_with_conv);  curr_res *= 2

            self.up.insert(0, up) # Prepend to get consistent order

        # End
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)


    def forward(self, z):       
        self.last_z_shape = z.shape

        # Middle
        temb = None  # Timestep embedding
        h = self.conv_in(z)  # z to block_in
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h, temb)), temb)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # End
        if self.give_pre_end:  return h

        h = self.conv_out(nonlinearity(self.norm_out(h)))
        return torch.tanh(h) if self.tanh_out else h