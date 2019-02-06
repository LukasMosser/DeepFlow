import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_activation():
    return nn.ReLU()

class GeneratorMultiChannel(nn.Module):
    def __init__(self):
        super(GeneratorMultiChannel, self).__init__()
        self.network = self.build_network()
        self.activation_facies = nn.Tanh()
        self.activation_rho = nn.Softplus()

    def build_network(self, activation=get_activation):
        blocks = []
        blocks += [nn.Conv2d(50, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), activation()]
        blocks += [nn.PixelShuffle(upscale_factor=2)]
        blocks += [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), activation()]
        blocks += [nn.PixelShuffle(upscale_factor=2)]
        blocks += [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), activation()]
        blocks += [nn.PixelShuffle(upscale_factor=2)]
        blocks += [nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), activation()]
        blocks += [nn.PixelShuffle(upscale_factor=2)]
        blocks += [nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), activation()]
        blocks += [nn.PixelShuffle(upscale_factor=2)]
        blocks += [nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), activation()]
        blocks += [nn.PixelShuffle(upscale_factor=2)]
        blocks += [nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*blocks)

    def forward(self, z):
        x = self.network(z)
        a = self.activation_facies(x[:, 0]).unsqueeze(1)
        b = self.activation_facies(x[:, 1]).unsqueeze(1)
        c = self.activation_rho(x[:, 2]).unsqueeze(1)
        return torch.cat([a, b, c], 1)

class DiscriminatorUpsampling(nn.Module):
    def __init__(self):
        super(DiscriminatorUpsampling, self).__init__()
        self.network = self.build_network()

    def build_network(self, activation=get_activation):
        blocks = []
        blocks += [nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2), activation()]
        blocks += [nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1), activation()]
        blocks += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), activation()]
        blocks += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), activation()]
        blocks += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), activation()]
        blocks += [nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), activation()]
        blocks += [nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        dec = self.network(x)
        dec = dec.view(-1, 2 * 2 * 2)
        return dec
