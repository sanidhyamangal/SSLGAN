"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

from typing import Optional

import numpy as np  # for matrix multiplication
import torch  # for deep learning
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from logger import logger
from utils import generate_and_save_images, plot_gan_loss_plots


class BaseGANTrainer:
    """Trainer module for GAN"""
    def __init__(self,
                 disctiminator: nn.Module,
                 generator: nn.Module,
                 gen_opt: Optimizer,
                 disc_opt: Optimizer,
                 self_inducing_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss,
                 gan_loss: nn.BCELoss = nn.BCELoss,
                 device="cpu",
                 lr=1e-4) -> None:

        # define the generators, optimizers and loss criterions
        self.generator = generator
        self.generator_optimizer = gen_opt(generator.parameters(), lr)
        if disctiminator:
            self.discriminator_optimizer = disc_opt(disctiminator.parameters(),
                                                    lr)
            self.discriminator = disctiminator
        self.criterion = gan_loss()
        self.self_inducing_loss = self_inducing_loss()
        self.device = device

    def disc_step(self, xthat, xt, target_labels):
        """Discriminator backprop step"""
        # train disc 5 times before performing gan update
        # find true and false value for the real and fake dist
        self.discriminator.zero_grad()
        disc_true = self.discriminator(xt).view(-1)
        real_label = torch.ones_like(disc_true)
        errD_real = self.criterion(disc_true, real_label)
        errD_real.backward()

        disc_false = self.discriminator(xthat.data).view(-1)
        false_label = torch.zeros_like(disc_false)
        errD_fake = self.criterion(disc_false, false_label)

        errD = errD_fake + errD_real

        self.discriminator_optimizer.step()

        return errD.detach().cpu().numpy()

    def gen_step(self, xthat):
        """function to perform backprop on generator"""

        # compute gen loss and take a step using optimizer
        self.generator.zero_grad()
        output = self.discriminator(xthat).view(-1)
        real_labels = torch.ones_like(output)
        gen_loss = self.criterion(output, real_labels)
        gen_loss.backward()
        self.generator_optimizer.step()

        return gen_loss.detach().cpu().numpy()

    def train_batch(self, noise, target, target_labels):
        """Function to train the batch of data"""
        # generate images
        generated_image = self.generator(noise)

        # disc step
        disc_loss = self.disc_step(generated_image, target, target_labels)

        # gen step
        gen_loss = self.gen_step(generated_image)

        return gen_loss, disc_loss

    def train(self,
              epochs,
              dataset,
              batch_sz=64,
              loss_plot: Optional[str] = "",
              plot_samples: Optional[str] = "",
              model_name: Optional[str] = "generator_gan.pt"):
        """function for training the gan"""

        # logs for the loss plotting
        gen_loss = []
        disc_loss = []
        dataloader = DataLoader(dataset,
                                batch_size=batch_sz,
                                shuffle=True,
                                num_workers=2)
        latent_dim = 100

        # train models for n+1 epochs
        for epoch in range(0, epochs + 1):
            # generate the gaussian noise for the generator
            for idx, data in enumerate(dataloader):
                images, angles = data
                noise = torch.randn(
                    (batch_sz, latent_dim, 1, 1)).to(self.device)

                # train step
                _gen_loss, _disc_loss = self.train_batch(
                    noise, images.to(self.device), angles.to(self.device))

                gen_loss.append(_gen_loss), disc_loss.append(_disc_loss)

                logger.info(
                    f"Epoch: {epoch}, Iteration:{idx}, Gen Loss:{gen_loss[-1]}, Disc Loss:{disc_loss[-1]}"
                )

            if plot_samples:
                _path_to_plot = plot_samples.split(".")
                path_to_samples = f"{_path_to_plot[0]}_{epoch}.{_path_to_plot[-1]}"
                generate_and_save_images(
                    self.generator,
                    torch.randn((16, latent_dim, 1, 1)).to(self.device),
                    path_to_samples, dataset)

            torch.save(self.generator, model_name)
        # plot the loss values
        if loss_plot: plot_gan_loss_plots(disc_loss, gen_loss, loss_plot)


# class SelfInducingGANTrainer(BaseGANTrainer):
#     def disc_step(self, xthat, xt, target_labels):
#         """Discriminator backprop step"""
#         # store disc loss

#         # train disc 5 times before performing gan update
#         # for t in range(5):
#         # find true and false value for the real and fake dist
#         disc_true, labels = self.discriminator(xt, self_learning=True)
#         disc_false = self.discriminator(xthat.data)

#         # find the values for the true dist and fake dist
#         disc_loss = -torch.mean(torch.log(disc_true)) - torch.mean(
#             torch.log(1 - disc_false))

#         rotnet_loss = self.self_inducing_loss(labels, target_labels)
#         total_loss = disc_loss + 1.0 * rotnet_loss

#         # take a step on discriminator optimizer
#         self.discriminator_optimizer.zero_grad()
#         total_loss.backward()
#         self.discriminator_optimizer.step()

#         return total_loss.detach().cpu().numpy()
