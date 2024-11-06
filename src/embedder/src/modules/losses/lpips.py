import torch
import torch.nn as nn

from torchvision import models
from torchvision.models.vgg import VGG16_Weights
from collections import namedtuple

from embedder.src.utils import get_ckpt_path

###

class LPIPS(nn.Module):  # Learned Perceptual Similarity
    def __init__(self, use_dropout=True):
        super().__init__()
        
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vgg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout);  self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout);  self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)

        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False


    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "/home/ubuntu/scania-raw-diff/2022-CVPR-LDM/src/embedder/logs/checkpoints")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("Loaded pretrained LPIPS loss from {}".format(ckpt))


    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        model = cls()
        model.load_state_dict(torch.load(get_ckpt_path(name), map_location=torch.device("cpu")), strict=False)
        return model


    def forward(self, input, target):

        outs0 = self.net(self.scaling_layer(input));  outs1 = self.net(self.scaling_layer(target))
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        diffs = {}

        for kk in range(len(self.chns)):
            diffs[kk] = (normalize_tensor(outs0[kk]) - normalize_tensor(outs1[kk])) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))][0]
        
        val = res[0]  # NOTE Change to numpy sum
        for l in range(1, len(self.chns)):  val += res[l]

        return val

###

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()

        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])   [None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])      [None, :, None, None])


    def forward(self, inp):
        return (inp - self.shift) / self.scale

###

class NetLinLayer(nn.Module):  # 1x1 Linear Convolution
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]

        self.model = nn.Sequential(*layers)

###

class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()

        vgg_pretrained_features = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else pretrained).features

        self.slice1 = torch.nn.Sequential();  self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential();  self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        self.N_slices = 5
        for x in range(4):          self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):       self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):      self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):     self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):     self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, X):
        h = self.slice1(X);  h_relu1_2 = h
        h = self.slice2(h);  h_relu2_2 = h
        h = self.slice3(h);  h_relu3_3 = h
        h = self.slice4(h);  h_relu4_3 = h
        h = self.slice5(h);  h_relu5_3 = h

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        return vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

###

def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x/(norm_factor+eps)

###

def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)

