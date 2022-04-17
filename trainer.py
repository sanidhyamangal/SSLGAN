"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

from typing import Optional

import numpy as np  # for matrix multiplication
import torch  # for deep learning
from torch import nn
from torch.optim import Optimizer

from utils import generate_and_save_images, plot_gan_loss_plots


class BaseGANTrainer:
    """Trainer module for GAN"""
    def __init__(self,
                 disctiminator: nn.Module,
                 generator: nn.Module,
                 gen_opt: Optimizer,
                 disc_opt: Optimizer,
                 gan_loss: nn.BCELoss = nn.BCEWithLogitsLoss,
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
        self.device = device

    def disc_step(self, xthat, xt):
        """Discriminator backprop step"""
        # store disc loss
        disc_loss_log_ = []

        # train disc 5 times before performing gan update
        for t in range(5):
            # find true and false value for the real and fake dist
            disc_true = self.discriminator(xt)
            disc_false = self.discriminator(xthat.data)

            # find the values for the true dist and fake dist
            disc_loss = -torch.mean(torch.log(disc_true)) - torch.mean(
                torch.log(1 - disc_false))

            # take a step on discriminator optimizer
            self.discriminator_optimizer.zero_grad()
            disc_loss.backward()
            self.discriminator_optimizer.step()
            disc_loss_log_.append(disc_loss.item())

        return np.mean(disc_loss_log_)

    def gen_step(self, xthat):
        """function to perform backprop on generator"""

        # compute gen loss and take a step using optimizer
        gen_loss = torch.mean(torch.log(1 - self.discriminator(xthat)))
        self.generator_optimizer.zero_grad()
        gen_loss.backward()
        self.generator_optimizer.step()

        return gen_loss

    def train_batch(self, noise, target):
        """Function to train the batch of data"""
        # generate images
        generated_image = self.generator(noise)

        # disc step
        disc_loss = self.disc_step(generated_image, target)

        # gen step
        gen_loss = self.gen_step(generated_image)

        return gen_loss.detach().cpu().numpy(), disc_loss

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
        latent_dim = self.generator.arch[0]
        self.batch_sz = batch_sz

        # train models for n+1 epochs
        for epoch in range(0, epochs + 1):
            # generate the gaussian noise for the generator
            noise = torch.randn((batch_sz, latent_dim)).to(self.device)

            # train step
            _gen_loss, _disc_loss = self.train_batch(noise, dataset)

            gen_loss.append(_gen_loss), disc_loss.append(_disc_loss)

            print(
                f"Epoch: {epoch}, Gen Loss:{gen_loss[-1]}, Disc Loss:{disc_loss[-1]}"
            )

            # for every 100 epochs save the model and generate the samples from generator
            if (epoch % 100) == 0:
                torch.save(self.generator, model_name)
                if plot_samples:
                    _path_to_plot = plot_samples.split(".")
                    path_to_samples = f"{_path_to_plot[0]}_{epoch}.{_path_to_plot[-1]}"
                    generate_and_save_images()(self.generator,
                                               torch.randn(
                                                   (16, latent_dim)).to(
                                                       self.device),
                                               path_to_samples, dataset)

        # plot the loss values
        if loss_plot: plot_gan_loss_plots(disc_loss, gen_loss, loss_plot)
