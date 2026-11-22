"""MCC counting network for the E1 MOC-6 baseline."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from config import cfg
from datasets.setting.MOC_RS import cfg_data
from models.FPN.pyramid import FeaturePyramid, resize

NUM_CLASSES = len(cfg_data.CATEGORYS)


class MCC(nn.Module):
    def __init__(self, encoder, channels, width=None, reference_size=None, use_nir=False, use_masks=False):
        super().__init__()
        width = cfg.MODEL_ARCH.fpn.width if width is None else width
        self.encoder = encoder
        self.fpn = FeaturePyramid(channels, width)
        self.decoder = VisualDecoder(width)
        self.head = CountingHead(width, reference_size=reference_size)
        self.nir_projections = None
        if use_nir:
            nir = cfg.MODEL_ARCH.nir
            kernel = nir.kernel_size
            self.nir_projections = nn.ModuleList(
                nn.Conv2d(nir.channels, width, kernel, padding=kernel // 2, bias=False)
                for _ in cfg.MODEL_ARCH.backbone.strides
            )
            for projection in self.nir_projections:
                nn.init.zeros_(projection.weight)
        self.mask_classes = cfg.MODEL_ARCH.mask.classes
        self.mask_head = nn.Conv2d(width, NUM_CLASSES * self.mask_classes, 1) if use_masks else None

    def forward(self, rgb, nir=None, *, return_mask=False):
        if (self.nir_projections is None) != (nir is None):
            raise ValueError("Provide NIR exactly when the model uses RGB+NIR")
        # The upstream WKV backward supports FP32 parameters. BF16 can cover the neck and head.
        with torch.autocast(rgb.device.type, enabled=False):
            features = self.encoder(rgb.float())
        selected = tuple(features[index] for index in cfg.MODEL_ARCH.backbone.out_indices)
        pyramid = self.fpn(selected)
        if self.nir_projections is not None:
            pyramid = tuple(
                feature + projection(F.avg_pool2d(nir, stride))
                for feature, projection, stride in zip(
                    pyramid, self.nir_projections, cfg.MODEL_ARCH.backbone.strides, strict=True
                )
            )
        feature = self.decoder(pyramid)
        density, counts = self.head(feature)
        if return_mask:
            if self.mask_head is None:
                raise ValueError("MCC must be constructed with use_masks=True to return mask logits")
            return density, counts, self.mask_head(feature).unflatten(1, (NUM_CLASSES, self.mask_classes))
        return density, counts


class VisualDecoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * width, width, 1, bias=False),
            nn.GroupNorm(cfg.MODEL_ARCH.decode_head.norm_groups, width),
            nn.GELU(),
        )

    def forward(self, pyramid):
        p2, p3, p4 = pyramid
        size = p3.shape[-2:]
        return self.fuse(torch.cat((resize(p2, size), p3, resize(p4, size)), dim=1))


class CountingHead(nn.Module):
    def __init__(self, width, initial_count=None, reference_size=None):
        super().__init__()
        head = cfg.MODEL_ARCH.decode_head
        initial_count = head.initial_count if initial_count is None else initial_count
        reference_size = cfg.TRAIN_SIZE[0] if reference_size is None else reference_size
        self.predict = nn.Sequential(
            nn.Conv2d(width, head.width, head.kernel_size, padding=head.kernel_size // 2, bias=False),
            nn.GroupNorm(head.norm_groups, head.width),
            nn.GELU(),
            nn.Conv2d(head.width, NUM_CLASSES, 1),
        )
        cells = (reference_size // cfg_data.LABEL_FACTOR) ** 2
        initial_density = initial_count * cfg_data.LOG_PARA / cells
        nn.init.normal_(self.predict[-1].weight, std=head.weight_std)
        nn.init.constant_(self.predict[-1].bias, math.log(math.expm1(initial_density)))

    def forward(self, feature):
        density = F.softplus(self.predict(feature).float())
        return density, density.sum(dim=(-2, -1)) / cfg_data.LOG_PARA
