"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
from functools import partial
from typing import List

import torch
from torch import nn

# class Discriminator(nn.Module):
#     def __init__(self,
#                  units: List[int],
#                  in_channel: int,
#                  stride: int = 1,
#                  kernel_size: int = 3,
#                  padding: int = 0,
#                  nclasses: int = 4,
#                  activation=nn.ReLU) -> None:
#         super(Discriminator, self).__init__()
#         Conv2D = partial(nn.Conv2d,
#                          kernel_size=kernel_size,
#                          stride=stride,
#                          padding=padding)
#         Pooling = partial(nn.MaxPool2d, kernel_size=2, stride=2)
#         self.activation = activation()
#         model_arch = [
#             Conv2D(in_channels=in_channel, out_channels=units[0]),
#             self.activation,
#             Pooling()
#         ]

#         for i in range(1, len(units)):
#             model_arch.extend([
#                 Conv2D(in_channels=units[i - 1], out_channels=units[i]),
#                 self.activation,
#                 Pooling()
#             ])

#         self.feature_learning = nn.Sequential(*model_arch)
#         self.classifier = nn.Sequential(nn.LazyLinear(out_features=128),
#                                         self.activation,
#                                         nn.Linear(128, 256), self.activation,
#                                          nn.Linear(256, 1))

#         self.self_learning = nn.Sequential(nn.LazyLinear(out_features=512),
#                                            self.activation, nn.Dropout(0.4),
#                                            nn.Linear(512,
#                                                      512), self.activation,
#                                            nn.Dropout(0.4),
#                                            nn.Linear(512, nclasses),
#                                            nn.Sigmoid())

#     def forward(self, x, self_learning: bool = False):
#         z = self.feature_learning(x)
#         z = torch.reshape(z, (z.shape[0], -1))
#         out = self.classifier(z)
#         if not self_learning:
#             return out
#         self_learning = self.self_learning(z)

#         return out, self_learning


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, nclass: int = 4):
        super(Discriminator, self).__init__()

        self.activation = nn.LeakyReLU()
        self.feature_learning = nn.Sequential(
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
        self.classifier = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid())

        self.self_learning = nn.Sequential(nn.LazyLinear(out_features=512),
                                           self.activation, nn.Dropout(0.4),
                                           nn.Linear(512,
                                                     512), self.activation,
                                           nn.Dropout(0.4),
                                           nn.Linear(512,
                                                     nclass), nn.Sigmoid())

    def forward(self, x, self_learning: bool = False):
        z = self.feature_learning(x)
        out = self.classifier(z)
        if not self_learning:
            return out

        z = torch.reshape(z, (z.shape[0], -1))
        self_learning = self.self_learning(z)

        return out, self_learning


# class Generator(nn.Module):
#     def __init__(self,
#                  arch: List[int],
#                  init_filters: int,
#                  stride: int = 1,
#                  kernel_size: int = 3,
#                  padding: int = 0,
#                  activation=nn.ReLU) -> None:
#         super().__init__()
#         self.activation = activation()
#         model_arch = [
#             nn.LazyConvTranspose2d(init_filters * arch[0],
#                                    kernel_size=kernel_size,
#                                    stride=stride,
#                                    padding=padding),
#             nn.BatchNorm2d(init_filters * arch[0]), self.activation
#         ]

#         for i in range(1, len(arch) - 1):
#             model_arch.extend([
#                 nn.ConvTranspose2d(init_filters * arch[i - 1],
#                                    init_filters * arch[i],
#                                    kernel_size=kernel_size,
#                                    padding=padding,
#                                    stride=stride),
#                 nn.BatchNorm2d(init_filters * arch[i]), self.activation
#             ])

#         model_arch.extend([
#             nn.ConvTranspose2d(init_filters * arch[-2],
#                                arch[-1],
#                                kernel_size=kernel_size,
#                                padding=padding,
#                                stride=stride),
#             nn.Tanh()
#         ])

#         self.model = nn.Sequential(*model_arch)

# Generator Code


class Generator(nn.Module):
    def __init__(self, ngf: int = 4, nc: int = 3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.LazyConvTranspose2d(ngf * 8, 4, 1, 0, bias=False),
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
