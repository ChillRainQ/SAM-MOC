"""MCC counting network for the E1 MOC-6 baseline."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from config import (
    DEFAULT_CROP_SIZE, ENCODER_FEATURE_INDICES, ENCODER_FEATURE_STRIDES, FPN_WIDTH,
    HEAD_INITIAL_COUNT, HEAD_KERNEL_SIZE, HEAD_WEIGHT_STD, HEAD_WIDTH, INPUT_SIZE_MULTIPLE, MASK_CLASSES,
    NIR_CHANNELS, NIR_RESIDUAL_KERNEL_SIZE, NORM_GROUPS, RGB_MEAN,
)
from moc_adapter import CLASSES, DENSITY_SCALE, LABEL_FACTOR
from models.FPN.pyramid import FeaturePyramid, resize


class MCC(nn.Module):
    def __init__(self, encoder, channels, width=FPN_WIDTH, reference_size=DEFAULT_CROP_SIZE,
                 use_nir=False, use_masks=False):
        super().__init__()
        self.encoder = encoder
        self.fpn = FeaturePyramid(channels, width)
        self.decoder = VisualDecoder(width)
        self.head = CountingHead(width, reference_size=reference_size)
        self.nir_projections = None
        if use_nir:
            kernel = NIR_RESIDUAL_KERNEL_SIZE
            self.nir_projections = nn.ModuleList(
                nn.Conv2d(NIR_CHANNELS, width, kernel, padding=kernel // 2, bias=False)
                for _ in ENCODER_FEATURE_STRIDES
            )
            for projection in self.nir_projections:
                nn.init.zeros_(projection.weight)
        self.mask_head = nn.Conv2d(width, len(CLASSES) * MASK_CLASSES, 1) if use_masks else None

    def forward(self, rgb, nir=None, *, return_mask=False):
        if rgb.ndim != 4 or rgb.shape[1] != len(RGB_MEAN):
            raise ValueError("Expected an RGB tensor with shape [B, 3, H, W]")
        h, w = rgb.shape[-2:]
        if h % INPUT_SIZE_MULTIPLE or w % INPUT_SIZE_MULTIPLE:
            raise ValueError(f"Expected H and W divisible by {INPUT_SIZE_MULTIPLE}")
        if (self.nir_projections is None) != (nir is None):
            raise ValueError("Provide NIR exactly when the model uses RGB+NIR")
        if nir is not None and nir.shape != (rgb.shape[0], NIR_CHANNELS, h, w):
            raise ValueError("Expected an NIR tensor with shape [B, 1, H, W] aligned to RGB")
        # The upstream WKV backward supports FP32 parameters. BF16 can cover the neck and head.
        with torch.autocast(rgb.device.type, enabled=False):
            features = self.encoder(rgb.float())
        selected = tuple(features[index] for index in ENCODER_FEATURE_INDICES)
        for feature, stride in zip(selected, ENCODER_FEATURE_STRIDES, strict=True):
            if feature.shape[-2:] != (h // stride, w // stride):
                raise ValueError(f"Unexpected RWKV-SAM feature size at stride {stride}: {feature.shape}")
        pyramid = self.fpn(selected)
        if self.nir_projections is not None:
            pyramid = tuple(
                feature + projection(F.avg_pool2d(nir, stride))
                for feature, projection, stride in zip(
                    pyramid, self.nir_projections, ENCODER_FEATURE_STRIDES, strict=True
                )
            )
        feature = self.decoder(pyramid)
        density, counts = self.head(feature)
        if return_mask:
            if self.mask_head is None:
                raise ValueError("MCC must be constructed with use_masks=True to return mask logits")
            return density, counts, self.mask_head(feature).unflatten(1, (len(CLASSES), MASK_CLASSES))
        return density, counts


class VisualDecoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * width, width, 1, bias=False), nn.GroupNorm(NORM_GROUPS, width), nn.GELU()
        )

    def forward(self, pyramid):
        p2, p3, p4 = pyramid
        size = p3.shape[-2:]
        return self.fuse(torch.cat((resize(p2, size), p3, resize(p4, size)), dim=1))


class CountingHead(nn.Module):
    def __init__(self, width, initial_count=HEAD_INITIAL_COUNT, reference_size=DEFAULT_CROP_SIZE):
        super().__init__()
        self.predict = nn.Sequential(
            nn.Conv2d(width, HEAD_WIDTH, HEAD_KERNEL_SIZE, padding=HEAD_KERNEL_SIZE // 2, bias=False),
            nn.GroupNorm(NORM_GROUPS, HEAD_WIDTH),
            nn.GELU(),
            nn.Conv2d(HEAD_WIDTH, len(CLASSES), 1),
        )
        cells = (reference_size // LABEL_FACTOR) ** 2
        initial_density = initial_count * DENSITY_SCALE / cells
        nn.init.normal_(self.predict[-1].weight, std=HEAD_WEIGHT_STD)
        nn.init.constant_(self.predict[-1].bias, math.log(math.expm1(initial_density)))

    def forward(self, feature):
        density = F.softplus(self.predict(feature).float())
        return density, density.sum(dim=(-2, -1)) / DENSITY_SCALE
