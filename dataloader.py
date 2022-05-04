"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms  # for the transforms on the images
from torchvision.transforms.functional import rotate

from utils import create_rot_transforms


class RotNetDataset(Dataset):
    """Dataset class for the rotation net and vanilla GAN"""
    def __init__(self,
                 path_to_images,
                 transform=None,
                 use_rotations=False) -> None:
        super().__init__()
        # define the image paths, rotations for the images
        self.images_path = [i for i in Path(path_to_images).glob("*.jpg")]
        self.rotations = np.random.randint(0, 4, size=len(self.images_path))

        # transformations on the image and flag if rotation is required to cary out or not
        self.transform = transform
        self.use_rotation = use_rotations

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # open image
        image = Image.open(self.images_path[index])
        angle = 0
        # if use rotation true rotate image to the angle.
        if self.use_rotation:
            angle = self.rotations[index]
            image = rotate(image, angle * 90.0)

        # apply transformation to normalize images
        if self.transform:
            image = self.transform(image)

        return image, angle


class ContrastiveDataset(Dataset):
    """Contrastive dataset class to load dataset for the contrastive learning"""
    def __init__(
        self,
        path_to_images,
        transform=None,
    ) -> None:
        super().__init__()
        # load images and define transformations on the image
        self.images_path = [i for i in Path(path_to_images).glob("*.jpg")]
        assert transform is not None, "You need to provide some transformations for the contrastive learning to work"
        self.transform = transform

        # pipeline to convert image to tensor
        self.pil_to_tensor_transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize(size=(64, 64)),
            transforms.ConvertImageDtype(torch.float32)
        ])
        # normalizer for normalization of images
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # open image and convert it to tensor
        image = Image.open(self.images_path[index])
        image = self.pil_to_tensor_transform(image)
        # transform the image
        if self.transform:
            augmented = self.transform(image)
        # normalize images to aling with augmented images
        image = self.normalize(image)

        return image, augmented
