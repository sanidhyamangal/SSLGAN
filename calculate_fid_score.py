"""
author:Sanidhya Mangal, Daniel Shu, Rishav Sen, Jatin Kodali
"""

import numpy as np
import torch
from scipy.linalg import sqrtm

from utils import DEVICE

device = DEVICE()

real_images = torch.load("real_image.tensor",
                         map_location=device).detach().numpy()
rotnet_images = torch.load("rotnet.tensor",
                           map_location=device).detach().numpy()
vanilla_images = torch.load("vanilla.tensor",
                            map_location=device).detach().numpy()
contrastive_images = torch.load("contrastive.tensor",
                                map_location=device).detach().numpy()


# calculate frechet inception distance
def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
