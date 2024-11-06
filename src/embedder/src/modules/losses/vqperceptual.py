import torch
import torch.nn as nn
import torch.nn.functional as F

from embedder.src.modules.losses.lpips import LPIPS
from embedder.src.modules.archs.discriminator import NLayerDiscriminator, weights_init

### 

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:  weight = value
    return weight

###

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

###

class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0, disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, 
                 disc_weight=1.0, perceptual_weight=1.0, use_actnorm=False, disc_conditional=False, disc_ndf=64, disc_loss="hinge"):
        super().__init__()

        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight

        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(disc_in_channels, disc_ndf, disc_num_layers, use_actnorm).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_conditional = disc_conditional
        self.discriminator_weight = disc_weight
        self.discriminator_factor = disc_factor
        self.discriminator_loss = hinge_d_loss


    def calculate_adaptive_weight(self, nll_loss, generator_loss, last_layer=None):
        layer = last_layer if last_layer is not None else self.last_layer[0]

        nll_grads = torch.autograd.grad(nll_loss, layer, retain_graph=True)[0]
        generator_grads = torch.autograd.grad(generator_loss, layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(generator_grads) + 1e-4)
        return torch.clamp(d_weight, 0.0, 1e4).detach() * self.discriminator_weight 


    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx, global_step, last_layer=None, cond=None, split="train"):
        reconstruction_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        
        if self.perceptual_weight <= 0:  perceptual_loss = torch.tensor([0.0])
        else:
                                         perceptual_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
                                         reconstruction_loss = reconstruction_loss + self.perceptual_weight * perceptual_loss

        negative_log_likelihood_loss = torch.mean(reconstruction_loss)

        if optimizer_idx == 0:  # Generator Update
            
            generator_loss = -torch.mean(self.discriminator(reconstructions.contiguous()) if cond is None else \
                                         self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1)))  # Logits Fake

            try:                    discriminator_weight = self.calculate_adaptive_weight(negative_log_likelihood_loss, generator_loss, last_layer)
            except RuntimeError:    discriminator_weight = torch.tensor(0.0)

            discriminator_factor = adopt_weight(self.discriminator_factor, global_step, threshold=self.discriminator_iter_start)

            loss = negative_log_likelihood_loss + \
                   discriminator_weight * discriminator_factor * generator_loss + \
                   self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): negative_log_likelihood_loss.detach().mean(),
                   "{}/rec_loss".format(split): reconstruction_loss.detach().mean(),
                   "{}/p_loss".format(split): perceptual_loss.detach().mean(),
                   "{}/d_weight".format(split): discriminator_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(discriminator_factor),
                   "{}/g_loss".format(split): generator_loss.detach().mean(),}
            
            return loss, log

        if optimizer_idx == 1:  # Discriminator Update
            
            logits_real = self.discriminator(inputs.contiguous().detach()) if cond is None else \
                          self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
            
            logits_fake = self.discriminator(reconstructions.contiguous().detach()) if cond is None else \
                          self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            discriminator_factor = adopt_weight(self.discriminator_factor, global_step, threshold=self.discriminator_iter_start)
            loss = discriminator_factor * self.discriminator_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()}
            
            return loss, log
