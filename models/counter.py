"""Wrap the counting network with density supervision."""
import torch
from torch import nn

from config import cfg, LOSS_FUNCTIONS
from models.loss.density import GaussianDensity, class_cosine_loss


class Counter(nn.Module):
    def __init__(self, network, loss_function=cfg.LOSS_FUNCTION, cos_alpha=cfg.alpha, use_nir=False):
        super().__init__()
        self.network = network
        self.density_target = GaussianDensity()
        self.loss_function = loss_function
        self.cos_alpha = cos_alpha
        self.use_nir = use_nir
        self.loss_fn = None
        if loss_function not in LOSS_FUNCTIONS:
            raise ValueError(f"Unknown loss function: {loss_function}")
        self.loss_mse_fn = nn.MSELoss()
        if loss_function == "Mix_loss":
            self.loss_l1_fn = nn.L1Loss()
        elif loss_function == "Mask_loss":
            self.loss_seg_fn = nn.CrossEntropyLoss()

    @property
    def loss(self):
        return self.loss_fn

    def forward(self, data, mode="train"):
        self.loss_fn = None
        return_mask = mode == "train" and self.loss_function == "Mask_loss"
        nir = data["nir"][:, :1] if self.use_nir else None
        pred_map, counts, *aux = self.network(data["rgb"], nir, return_mask=return_mask)
        outputs = {"pred_map": pred_map, "counts": counts}
        if mode != "train":
            return outputs

        gt_map = data["gt_map"][:, :pred_map.shape[1]].float()
        with torch.autocast(pred_map.device.type, enabled=False):
            pred_map = pred_map.float()
            gauss_map = self.density_target(gt_map)
            outputs["gauss_map"] = gauss_map
            if self.loss_function == "Cos_loss":
                self.loss_fn = self.build_cos_loss(pred_map, gt_map)
            elif self.loss_function == "Mix_loss":
                self.loss_fn = self.build_mix_loss(pred_map, gt_map)
            elif self.loss_function == "Mask_loss":
                pred_mask_maps = aux[0].unbind(1)
                outputs["pred_mask_maps"] = pred_mask_maps
                self.loss_fn = self.build_mask_loss(pred_map, gauss_map, data.get("mask_map"), pred_mask_maps)
            else:
                self.loss_fn = self.build_mse_loss(pred_map, gauss_map)
        return outputs

    def build_mse_loss(self, pred_map, target):
        if pred_map.shape != target.shape:
            raise ValueError(f"Prediction and target shape mismatch: {pred_map.shape} vs {target.shape}")
        return self.loss_mse_fn(pred_map, target)

    def build_cos_loss(self, pred_map, gt_map):
        return self.build_mse_loss(pred_map, gt_map) + self.cos_alpha * class_cosine_loss(pred_map)

    def build_mix_loss(self, pred_map, gt_map):
        mse = self.build_mse_loss(pred_map, gt_map)
        count_l1 = self.loss_l1_fn(pred_map.sum((1, 2, 3)), gt_map.sum((1, 2, 3)))
        return mse + cfg.MIX_L1_WEIGHT * count_l1

    def build_mask_loss(self, pred_map, gauss_map, masks, pred_mask_maps):
        if masks is None or masks.shape != pred_map.shape:
            raise ValueError("Mask_loss requires per-class masks aligned with the density map")
        mse = self.build_mse_loss(pred_map, gauss_map)
        mask_ce = sum(
            self.loss_seg_fn(logits.float(), mask.long())
            for logits, mask in zip(pred_mask_maps, masks.unbind(1), strict=True)
        )
        return mse + cfg.MASK_CE_WEIGHT * mask_ce
