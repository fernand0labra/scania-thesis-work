import functools
import torch.nn as nn

###

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
###

class NLayerDiscriminator(nn.Module):  # Defines a PatchGAN discriminator as in Pix2Pix
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        norm_layer = nn.BatchNorm2d
        use_bias = norm_layer.func != nn.BatchNorm2d if type(norm_layer) == functools.partial else norm_layer != nn.BatchNorm2d

        kw = 4;  padw = 1; nf_mult = 1;  nf_mult_prev = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        
        for n in range(1, n_layers + 1):  # Gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1 if n == n_layers else 2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)


    def forward(self, input):
        return self.main(input)