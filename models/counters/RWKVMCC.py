"""Original MCC counting path with a RWKV-SAM image backbone."""
import torch
from torch import nn
from torch.nn import functional as F

from config import cfg
from datasets.setting.MOC_RS import cfg_data
from models.FPN.FPN_Head import FPN
from models.backbone.rwkvsam import add_nir_input, build_encoder, load_encoder
from .MCC import SCAModule

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
        reduced_width = arch.decode_head.width
        channels = list(arch.backbone.channels[variant])
        self.fpn = FPN(channels, width, len(channels))
        self.reduce = nn.Conv2d(len(channels) * width, reduced_width, 1)
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
        with torch.autocast(rgb.device.type, enabled=False):
            features = self.backbone(image.float())
        # 抽头
        selected = tuple(features[index] for index in cfg.RWKV_ARCH.backbone.out_indices)
        # 多尺度融合
        x = self.fpn(selected)
        x = torch.cat([F.interpolate(x[0],scale_factor=0.5), x[1],F.interpolate(x[2],scale_factor=2)], dim=1)
        feature = self.reduce(x)
        density = self.output_layer(feature)
        if return_mask:
            if self.mask_head is None:
                raise ValueError("Mask logits require cfg.LOSS_FUNCTION = 'Mask_loss'")
            return density, self.mask_head(feature).unflatten(1, (NUM_CLASSES, self.mask_classes))
        return density
