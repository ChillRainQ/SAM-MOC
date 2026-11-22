"""Top-down feature pyramid for the three encoder stages."""
from torch import nn
from torch.nn import functional as F

from config import FPN_KERNEL_SIZE


def resize(feature, size):
    return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)


class FeaturePyramid(nn.Module):
    def __init__(self, in_channels, width):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in in_channels)
        self.output = nn.ModuleList(
            nn.Conv2d(width, width, FPN_KERNEL_SIZE, padding=FPN_KERNEL_SIZE // 2)
            for _ in in_channels
        )

    def forward(self, features):
        levels = [layer(x) for layer, x in zip(self.lateral, features, strict=True)]
        for i in range(len(levels) - 2, -1, -1):
            levels[i] = levels[i] + resize(levels[i + 1], levels[i].shape[-2:])
        return tuple(layer(x) for layer, x in zip(self.output, levels, strict=True))
