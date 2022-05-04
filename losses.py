"""
author:Sanidhya Mangal, Daniel Shu, Rishav Sen, Jatin Kodali
"""

import torch  # for torch
import torch.nn.functional as F  # for functional layer
from torch import nn  # for nn


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer(
            "negative_mask",
            (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1),
                                                representations.unsqueeze(0),
                                                dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        numerator = torch.exp(positives / self.temperature)
        denominator = self.negative_mask * torch.exp(
            similarity_matrix / self.temperature)

        _partial_loss = -torch.log(numerator / torch.sum(denominator, dim=1))

        loss = torch.sum(_partial_loss) / (2 * self.batch_size)

        return loss
