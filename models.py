"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
from functools import partial
from typing import List

import torch
from torch import nn

# Generator Code


# generator model
class Generator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# discriminator for the vanilla gan
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid())

    def forward(self, input):
        return self.main(input)


# rotation discriminator designed for the rotational training
class RotnetDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, nclasses=4):
        super(RotnetDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid())

        self.self_inducing = nn.Sequential(nn.LazyLinear(512), nn.ReLU(),
                                           nn.Dropout(0.4),
                                           nn.Linear(512, 256), nn.ReLU(),
                                           nn.Dropout(0.4),
                                           nn.Linear(256, nclasses))

    def forward(self, x, self_learning=False):
        features = self.main(x)
        out = self.discriminator(features)

        if not self_learning:
            return out
        z = torch.reshape(features, shape=(features.shape[0], -1))
        out_sl = self.self_inducing(z)
        return out, out_sl


# discriminator for the contrastive training
class ContrastiveDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(ContrastiveDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid())

        self.embedding = nn.Sequential(nn.LazyLinear(1024))
        self.projection = nn.Sequential(nn.Linear(1024, 128))

    def forward(self, x, self_learning=False, discriminator=True):
        # feature learner
        features = self.main(x)

        # for only self learning node is required to be activated
        if not discriminator and self_learning:
            z = torch.reshape(features, shape=(features.shape[0], -1))
            embeddings = self.embedding(z)
            projection = self.projection(embeddings)
            return projection

        # when only discrimiative node is required to be active
        out = self.discriminator(features)
        if not self_learning:
            return out

        z = torch.reshape(features, shape=(features.shape[0], -1))
        out_sl = self.projection(self.embedding(z))
        return out, out_sl
