"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import torch  # for pytorch based stuff
import torch.nn as nn  # for nn stuff
from torchvision import transforms  # for transforming the vision related ops


def create_rot_transforms():
    return transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(size=(64, 64)),
    ])
