"""MCC counting network for the E1 MOC-6 baseline."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from config import cfg
from datasets.setting.MOC_RS import cfg_data
from models.FPN.FPN_Head import FPN
from models.backbone.rwkvsam import build_encoder, load_encoder

NUM_CLASSES = len(cfg_data.CATEGORYS)


class RWKVMCC(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        arch = cfg.RWKV_ARCH
        variant = arch.backbone.variant
        if pretrained:
            if not cfg.PRE_BACKBONE_WEIGHT:
                raise ValueError("Set cfg.PRE_BACKBONE_WEIGHT to matching RWKV-SAM weights")
            self.backbone = load_encoder(cfg.OVSAM_ROOT, cfg.PRE_BACKBONE_WEIGHT, variant)
        else:
            self.backbone = build_encoder(cfg.OVSAM_ROOT, variant)
        width = arch.fpn.width
        channels = list(arch.backbone.channels[variant])
        self.fpn = FPN(channels, width, len(channels))
        self.decoder = VisualDecoder(width)
        self.head = CountingHead(width)
        use_nir = cfg.MM
        use_masks = cfg.LOSS_FUNCTION == "Mask_loss"
        self.nir_projections = None
        if use_nir:
            nir = cfg.RWKV_ARCH.nir
            kernel = nir.kernel_size
            self.nir_projections = nn.ModuleList(
                nn.Conv2d(nir.channels, width, kernel, padding=kernel // 2, bias=False)
                for _ in cfg.RWKV_ARCH.backbone.strides
            )
            for projection in self.nir_projections:
                nn.init.zeros_(projection.weight)
        self.mask_classes = cfg.RWKV_ARCH.mask.classes
        self.mask_head = nn.Conv2d(width, NUM_CLASSES * self.mask_classes, 1) if use_masks else None

    def forward(self, rgb, nir=None, *, return_mask=False):
        if (self.nir_projections is None) != (nir is None):
            raise ValueError("Provide NIR exactly when the model uses RGB+NIR")
        # The upstream WKV backward supports FP32 parameters. BF16 can cover the neck and head.
        with torch.autocast(rgb.device.type, enabled=False):
            features = self.backbone(rgb.float())
        selected = tuple(features[index] for index in cfg.RWKV_ARCH.backbone.out_indices)
        pyramid = self.fpn(selected)
        if self.nir_projections is not None:
            pyramid = tuple(
                feature + projection(F.avg_pool2d(nir, stride))
                for feature, projection, stride in zip(
                    pyramid, self.nir_projections, cfg.RWKV_ARCH.backbone.strides, strict=True
                )
            )
        feature = self.decoder(pyramid)
        density = self.head(feature)
        if return_mask:
            if self.mask_head is None:
                raise ValueError("MCC must be constructed with use_masks=True to return mask logits")
            return density, self.mask_head(feature).unflatten(1, (NUM_CLASSES, self.mask_classes))
        return density


class VisualDecoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * width, width, 1, bias=False),
            nn.GroupNorm(cfg.RWKV_ARCH.decode_head.norm_groups, width),
            nn.GELU(),
        )

    def forward(self, pyramid):
        p2, p3, p4 = pyramid
        size = p3.shape[-2:]
        return self.fuse(torch.cat((F.interpolate(p2, size=size, mode="bilinear", align_corners=False), p3, F.interpolate(p4, size=size, mode="bilinear", align_corners=False)), dim=1))


class CountingHead(nn.Module):
    def __init__(self, width, initial_count=None, reference_size=None):
        super().__init__()
        head = cfg.RWKV_ARCH.decode_head
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
        return density
