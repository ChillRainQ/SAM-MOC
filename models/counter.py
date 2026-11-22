"""Wrap the counting network with density supervision."""
from torch import nn
from torch.nn import functional as F

from config import COS_LOSS_ALPHA, LOSS_FUNCTION, MASK_CE_WEIGHT, MIX_L1_WEIGHT
from models.loss.density import GaussianDensity, class_cosine_loss


class Counter(nn.Module):
    def __init__(self, network, loss_function=LOSS_FUNCTION, cos_alpha=COS_LOSS_ALPHA):
        super().__init__()
        self.network = network
        self.density_target = GaussianDensity()
        self.mse = nn.MSELoss()
        self.loss_function = loss_function
        self.cos_alpha = cos_alpha

    def forward(self, rgb, nir=None):
        if self.training and self.loss_function == "Mask_loss":
            return self.network(rgb, nir, return_mask=True)
        return self.network(rgb, nir)

    def loss(self, density, dots, mask_logits=None, masks=None):
        density = density.float()
        target = dots.float() if self.loss_function in ("Cos_loss", "Mix_loss") else self.density_target(dots)
        if density.shape != target.shape:
            raise ValueError(f"Prediction and target shape mismatch: {density.shape} vs {target.shape}")
        mse = self.mse(density, target)
        if self.loss_function == "Cos_loss":
            return mse + self.cos_alpha * class_cosine_loss(density)
        if self.loss_function == "Mix_loss":
            count_l1 = F.l1_loss(density.sum((1, 2, 3)), target.sum((1, 2, 3)))
            return mse + MIX_L1_WEIGHT * count_l1
        if self.loss_function == "Mask_loss":
            if mask_logits is None or masks is None or masks.shape != density.shape:
                raise ValueError("Mask_loss requires per-class mask logits and masks aligned with the density map")
            mask_ce = sum(
                F.cross_entropy(logits.float(), mask.long())
                for logits, mask in zip(mask_logits.unbind(1), masks.unbind(1), strict=True)
            )
            return mse + MASK_CE_WEIGHT * mask_ce
        return mse
