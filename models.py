"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
from typing import List
from torch import nn
from functools import partial
import torch

class Discriminator(nn.Module):
    
    def __init__(self, units:List[int], in_channel:int,stride:int=1, kernel_size:int=3,padding:int=0,nclasses:int=4 ,activation=nn.ReLU) -> None:
        super(Discriminator, self).__init__()
        Conv2D = partial(nn.Conv2d, kernel_size=kernel_size, stride=stride, padding=padding)
        Pooling = partial(nn.MaxPool2d, kernel_size=2, stride=2)
        self.activation = activation()
        model_arch = [Conv2D(in_channels=in_channel, out_channels=units[0]), self.activation, Pooling()]

        for i in range(1,len(units)):
            model_arch.extend([Conv2D(in_channels=units[i-1], out_channels=units[i]), self.activation, Pooling()])
        

        self.feature_learning = nn.Sequential(*model_arch)
        self.classifier = nn.Sequential(
            nn.LazyLinear(out_features=512),
            self.activation,
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            self.activation,
            nn.Dropout(0.4),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
        self.self_learning = nn.Sequential(
            nn.LazyLinear(out_features=512),
            self.activation,
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            self.activation,
            nn.Dropout(0.4),
            nn.Linear(512, nclasses),
            nn.Sigmoid()
        )
    


    def forward(self, x):
        z = self.feature_learning(x)
        z = torch.flatten(z)
        out = self.classifier(z)
        self_learning = self.self_learning(z)

        return out, self_learning