import math
import torch
import numpy as np
import torch.nn as nn

###

def nonlinearity(x):
    return x * torch.sigmoid(x)

###

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

###

def get_timestep_embedding(timesteps, embedding_dim):  # Build sinusoidal embeddings

    half_dim = embedding_dim // 2

    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]

    return torch.nn.functional.pad(emb, (0, 1, 0, 0)) if embedding_dim % 2 == 1 else \
           torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

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
        self.out_channels = in_channels if out_channels is None else out_channels

        self.use_conv_shortcut = conv_shortcut
        self.dropout = torch.nn.Dropout(dropout)

        if temb_channels > 0:   self.temb_proj = torch.nn.Linear(temb_channels, out_channels)

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = Normalize(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:  self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:                       self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x, temb):
        h = self.conv1(nonlinearity(self.norm1(x)))

        if temb is not None:  h += self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:  x = self.conv_shortcut(x)
            else:                       x = self.nin_shortcut(x)

        return x + h

###

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.norm = Normalize(in_channels)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_);  k = self.k(h_);  v = self.v(h_)

        # Compute attention
        b, c, h, w = q.shape 

        q = q.reshape(b, c, h*w).permute(0, 2, 1)           # [Batch, Height * Width, Channel]
        k = k.reshape(b, c, h*w)                            # [Batch, Channel, Height * Width]

        # w[b, i, j] = sum_c ( q[b, i, c] * k[b, c, j] )
        w_ = torch.bmm(q, k) * (int(c)**(-0.5))             # [Batch, Height * Width, Height * Width]
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # Attend to values
        w_ = w_.permute(0,2,1)                              # [Batch, Height * Width (k), Height * Width (q)]

        # h_[b, c, j] = sum_i ( v[b, c, i] w_[b, i, j] )
        h_ = torch.bmm(v.reshape(b, c, h*w), w_)    
        h_ = h_.reshape(b, c, h, w)

        return x + self.proj_out(h_)
    
###

class Encoder(nn.Module):  # Downsampling
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, 
                 in_channels, resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()

        self.num_res_blocks = num_res_blocks
        self.num_resolutions = len(ch_mult)
        self.in_channels = in_channels
        self.resolution = resolution
        self.temb_ch = 0
        self.ch = ch

        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        self.down = nn.ModuleList()
        in_ch_mult = (1,) + tuple(ch_mult)

        for i_level in range(self.num_resolutions):

            attn = nn.ModuleList()
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level];  block_out = ch * ch_mult[i_level]
            
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out

                if curr_res in attn_resolutions:  attn.append(AttnBlock(block_in))

            down = nn.Module();  down.block = block;  down.attn = attn

            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2

            self.down.append(down)

        self.mid = nn.Module()
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, 2*z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        
        temb = None  # Timestep embedding
        hs = [self.conv_in(x)]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):

                h = self.down[i_level].block[i_block](hs[-1], temb)
                hs.append(self.down[i_level].attn[i_block](h) if len(self.down[i_level].attn) > 0 else h)

            if i_level != self.num_resolutions-1:  
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(hs[-1], temb)), temb)

        return self.conv_out(nonlinearity(self.norm_out(h)))

###

class Decoder(nn.Module):  # Upsampling
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, 
                 in_channels, resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()

        self.num_res_blocks = num_res_blocks
        self.num_resolutions = len(ch_mult)
        self.give_pre_end = give_pre_end
        self.in_channels = in_channels
        self.resolution = resolution
        self.temb_ch = 0
        self.ch = ch

        curr_res = resolution // 2**(self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))
        block_in = ch * ch_mult[self.num_resolutions - 1]

        self.up = nn.ModuleList()
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        for i_level in reversed(range(self.num_resolutions)):

            attn = nn.ModuleList()
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out

                if curr_res in attn_resolutions:  attn.append(AttnBlock(block_in))

            up = nn.Module();  up.block = block;  up.attn = attn

            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2

            self.up.insert(0, up)  # NOTE ??

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)


    def forward(self, z):

        temb = None  # Timestep embedding
        self.last_z_shape = z.shape

        h = self.conv_in(z)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h, temb)), temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):

                h = self.up[i_level].block[i_block](h, temb)
                
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:  return h

        return self.conv_out(nonlinearity(self.norm_out(h)))