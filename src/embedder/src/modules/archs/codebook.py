import torch
import numpy as np
import torch.nn as nn

from einops import rearrange

###

# NOTE: due to a bug the beta term was applied to the wrong term. for
# backwards compatibility we use the buggy version by default, but you can
# specify legacy=False to fix it.

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random", sane_index_shape=False, legacy=True):
        super().__init__()

        self.beta = beta
        self.remap = remap
        self.legacy = legacy

        self.n_e = n_e;  self.e_dim = e_dim
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            self.re_embed = self.used.shape[0]

            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1

            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "f"Using {self.unknown_index} for unknown indices.")
        
        else:  self.re_embed = n_e

        self.sane_index_shape = sane_index_shape


    def remap_to_used(self, idx_array):
        idx_shape = idx_array.shape
        idx_array = idx_array.reshape(idx_shape[0],-1)
        
        used = self.used.to(idx_array)
        match = (idx_array[:, :, None] == used[None, None, ...]).long()

        new = match.argmax(-1);  unknown = match.sum(2) < 1
        
        new[unknown] = torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device) if self.unknown_index == "random" else \
                       self.unknown_index
        
        return new.reshape(idx_shape)


    def unmap_to_all(self, idx_array):
        idx_shape = idx_array.shape
        idx_array = idx_array.reshape(idx_shape[0],-1)

        used = self.used.to(idx_array)

        # Extra tokens set to zero
        if self.re_embed > self.used.shape[0]:  idx_array[idx_array>=self.used.shape[0]] = 0

        back = torch.gather(used[None, :][idx_array.shape[0] * [0], :], 1, idx_array)

        return back.reshape(idx_shape)


    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        # Reshape z -> (Batch, Height, Width, Channel) -> Flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # Distances from z to embeddings [ (z - e_j)^2 = z^2 + e_j^2 - 2 e_j * z ]
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        # Compute loss for embedding
        if not self.legacy:  loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:                loss = self.beta * torch.mean((z_q - z.detach()) ** 2) + torch.mean((z_q.detach() - z) ** 2)

        z_q = z + (z_q - z).detach()                             # Preserve gradients
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()  # Reshape back to match original input shape

        if self.remap is not None:  # Add batch axis -> Remap -> Flatten
            min_encoding_indices = self.remap_to_used(min_encoding_indices.reshape(z.shape[0],-1)).reshape(-1, 1)

        if self.sane_index_shape:   # Return indices as [Batch, Height, Width]
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        perplexity = None;  min_encodings = None
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)


    def get_codebook_entry(self, indices, shape):  

        # Shape specifying (batch, height, width, channel)
        if self.remap is not None:  # Add batch axis -> Unmap -> Flatten
            indices = self.unmap_to_all(indices.reshape(shape[0],-1)).reshape(-1)

        z_q = self.embedding(indices)  # Get quantized latent vectors

        if shape is not None:  # Reshape back to match original input shape
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()

        return z_q
