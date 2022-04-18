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
from torchvision.transforms.functional import rotate

from utils import create_rot_transforms


class RotNetDataset(Dataset):
    def __init__(self, path_to_images, transform=None) -> None:
        super().__init__()
        self.images_path = [i for i in Path(path_to_images).glob("*.jpg")]
        self.rotations = np.random.randint(0, 4, size=len(self.images_path))
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = Image.open(self.images_path[index])
        angle = self.rotations[index]
        image = rotate(image, angle * 90.0)

        if self.transform:
            image = self.transform(image)

        return image, angle


if __name__ == "__main__":
    tranforms = create_rot_transforms()

    rotnet_dataset = RotNetDataset("images", transform=tranforms)

    image, angle = rotnet_dataset[0]
    plt.imshow(image.detach().numpy().transpose(1, 2, 0))
    plt.show()
