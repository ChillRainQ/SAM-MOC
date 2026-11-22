"""Gaussian density targets matching the NWPU-MOC CC MSE branch."""
import torch
from torch import nn
from torch.nn import functional as F

from config import cfg
from misc.layer import Gaussianlayer


class GaussianDensity(nn.Module):
    def __init__(self):
        super().__init__()
        gaussian = cfg.MODEL_ARCH.gaussian
        self.gaussian = Gaussianlayer(sigma=[gaussian.sigma], kernel_size=2 * gaussian.radius + 1)

    def forward(self, dots):
        with torch.autocast(dots.device.type, enabled=False):
            return torch.cat([self.gaussian(channel) for channel in dots.float().split(1, dim=1)], dim=1)


def class_cosine_loss(density):
    channels = density.flatten(2)
    batch, classes, _ = channels.shape
    i, j = torch.triu_indices(classes, classes, offset=1, device=density.device)
    similarities = F.cosine_similarity(channels[:, i], channels[:, j], dim=-1)
    # CC averages the full symmetric matrix with ones on its diagonal.
    return (2 * similarities.sum() + batch * classes) / (batch * classes**2)
