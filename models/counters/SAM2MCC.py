"""Original MCC counting path with the SAM 2.1 Hiera image backbone."""
import torch
from torch import nn
from torch.nn import functional as F

from config import cfg
from datasets.setting.MOC_RS import cfg_data
from models.FPN.FPN_Head import FPN
from models.backbone.sam2 import add_nir_input, build_encoder
from .MCC import SCAModule

NUM_CLASSES = len(cfg_data.CATEGORYS)


class SAM2MCC(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        arch = cfg.SAM2_ARCH
        checkpoint = cfg.PRE_BACKBONE_WEIGHT if pretrained else None
        if pretrained and not checkpoint:
            raise ValueError("Set cfg.PRE_BACKBONE_WEIGHT to the official SAM 2.1 Large checkpoint")
        self.backbone = build_encoder(arch.backbone.hiera, checkpoint)
        self.input_multiple = arch.backbone.input_multiple
        width = arch.fpn.width
        reduced_width = arch.decode_head.width
        channels = list(arch.backbone.channels)
        self.fpn = FPN(channels, width, len(channels))
        self.reduce = nn.Conv2d(3 * width, reduced_width, 1)
        self.output_layer = SCAModule(reduced_width, NUM_CLASSES)
        use_masks = cfg.LOSS_FUNCTION == "Mask_loss"
        self.use_nir = cfg.MM
        if self.use_nir:
            add_nir_input(self.backbone, arch.nir.channels)
        self.mask_classes = arch.mask.classes
        self.mask_head = nn.Conv2d(reduced_width, NUM_CLASSES * self.mask_classes, 1) if use_masks else None

    def forward(self, rgb, nir=None, *, return_mask=False):
        if self.use_nir != (nir is not None):
            raise ValueError("Provide NIR exactly when the model uses RGB+NIR")
        # 拼接
        image = torch.cat((rgb, nir), dim=1) if self.use_nir else rgb
        height, width = image.shape[-2:]
        pad_height = (-height) % self.input_multiple
        pad_width = (-width) % self.input_multiple
        if pad_height or pad_width:
            image = F.pad(image, (0, pad_width, 0, pad_height))
        features = self.backbone(image)
        # 多尺度融合
        # The fourth level contributes through the FPN top-down path.
        x = self.fpn(features)
        x = torch.cat([F.interpolate(x[0],scale_factor=0.5), x[1],F.interpolate(x[2],scale_factor=2)], dim=1)
        feature = self.reduce(x)
        factor = cfg_data.LABEL_FACTOR
        feature = feature[:, :, :height // factor, :width // factor]
        density = self.output_layer(feature)
        if return_mask:
            if self.mask_head is None:
                raise ValueError("Mask logits require cfg.LOSS_FUNCTION = 'Mask_loss'")
            return density, self.mask_head(feature).unflatten(1, (NUM_CLASSES, self.mask_classes))
        return density
