"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import torch
from torch.optim import Adam

from dataloader import RotNetDataset
from models import Discriminator, Generator
from trainer import BaseGANTrainer, SelfInducingGANTrainer
from utils import DEVICE, create_rot_transforms, weights_init

torch.manual_seed(999)
# create a dataset
dataset = RotNetDataset("images_small", transform=create_rot_transforms())
# discriminator = Discriminator([16, 32, 64, 64,8], in_channel=3, nclasses=4, stride=1, kernel_size=3, padding=1).to(DEVICE())
discriminator = Discriminator().to(DEVICE())
discriminator.apply(weights_init)
generator = Generator().to(DEVICE())
generator.apply(weights_init)
rotnet_trainer = BaseGANTrainer(disctiminator=discriminator,
                                generator=generator,
                                gen_opt=Adam,
                                disc_opt=Adam,
                                device=DEVICE())

rotnet_trainer.train(10,
                     dataset=dataset,
                     batch_sz=128,
                     loss_plot="plots/rotnet/loss.png",
                     plot_samples="plots/samples/rotnet/epoch.png",
                     model_name="rotnet_generator.pt")
