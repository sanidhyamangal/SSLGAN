"""
author:Sanidhya Mangal, Daniel Shu, Rishav Sen, Jatin Kodali
"""

import numpy as np  # for matrix maths
import torch
import torch.nn as nn
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataloader import RotNetDataset
from logger import logger
from models import Generator, RotnetDiscriminator
from utils import (DEVICE, create_rot_transforms, plot_sample_images,
                   weights_init)

torch.manual_seed(999)

# create a dataset
dataset = RotNetDataset("images",
                        transform=create_rot_transforms(),
                        use_rotations=True)
netD = RotnetDiscriminator().to(DEVICE())
netD.apply(weights_init)
netG = Generator().to(DEVICE())
netG.apply(weights_init)

img_list = []
G_losses = []
D_losses = []
plot_samples = "plots/sample/rotnet/epoch.png"
iters = 0
num_epochs = 50
real_label = 1
fake_label = 0
device = DEVICE()
criterion = nn.BCELoss()
self_inducing_criterion = nn.CrossEntropyLoss()
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
optimizerD = Adam(netD.parameters(), lr=0.0002)
optimizerG = Adam(netG.parameters(), lr=0.0002)
fixed_noise = torch.randn(64, 100, 1, 1, device=device)

# Commented out IPython magic to ensure Python compatibility.
print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader):

        ## Train with real data dist
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        target_label = data[1].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size, ),
                           real_label,
                           dtype=torch.float,
                           device=device)
        # Forward pass real batch through D
        output, angle_pred = netD(real_cpu, self_learning=True)
        # Calculate loss on real data dist
        errD_real = criterion(output.view(-1), label)
        # compute rotation error
        errD_rotation = self_inducing_criterion(angle_pred, target_label)
        # Calculate gradients for D in backward pass
        errD_real_rot = errD_real + errD_rotation
        errD_real_rot.backward()
        D_x = output.mean().item()

        ## Train the discriminator with the fake data dist
        # Generate batch of latent vectors
        noise = torch.randn(b_size, 100, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Compute error of D as sum over the fake and the real batches
        errD = errD_real_rot + errD_fake
        # Update D
        optimizerD.step()

        # network updater for the generator
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 10 == 0:
            logger.info(
                '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs, i, len(dataloader), errD.item(),
                   errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and
                                  (i == len(dataloader) - 1)):
            plot_name = ""
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                output = vutils.make_grid(fake, padding=2, normalize=True)
                _path_to_plot = plot_samples.split(".")
                path_to_samples = f"{_path_to_plot[0]}_{epoch}_{iters}.{_path_to_plot[-1]}"
                plot_sample_images(output, path_to_samples)

        iters += 1

torch.save(netG, "generator_rotnet.pt")
