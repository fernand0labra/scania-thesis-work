import torch
import torch.nn as nn

from embedder.src.modules.losses.vqperceptual import *

###

class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0, disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, 
                 disc_weight=1.0, perceptual_weight=1.0, use_actnorm=False, disc_conditional=False, disc_loss="hinge"):
        super().__init__()

        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)  # Output log variance

        self.disc_loss = hinge_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.discriminator_iter_start = disc_start
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm).apply(weights_init)


    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        last_layer =  last_layer if last_layer is not None else self.last_layer[0]
        
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        return torch.clamp(torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4), 0.0, 1e4).detach() * self.discriminator_weight


    def forward(self, x, xrec, posteriors, optimizer_idx, global_step, last_layer=None, cond=None, split="train", weights=None):
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        rec_loss = torch.abs(x.contiguous() - xrec.contiguous())

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(x.contiguous(), xrec.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar

        weighted_nll_loss = weights * nll_loss if weights is not None else nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]

        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        if optimizer_idx == 0:  # Generator update
            
            logits_fake = self.discriminator(xrec.contiguous()) if cond is None else \
                          self.discriminator(torch.cat((xrec.contiguous(), cond), dim=1))
            
            g_loss = -torch.mean(logits_fake)
            d_weight = torch.tensor(0.0)

            if self.disc_factor > 0.0:
                try:                  d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:  assert not self.training

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            
            return loss, log

        if optimizer_idx == 1:  # Discriminator update

            x = x.contiguous().detach()       if cond is None else torch.cat((x.contiguous().detach(),    cond), dim=1)
            xrec = xrec.contiguous().detach() if cond is None else torch.cat((xrec.contiguous().detach(), cond), dim=1)
            
            logits_real, logits_fake = self.discriminator(x), self.discriminator(xrec)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()}
            
            return d_loss, log

