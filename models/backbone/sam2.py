"""Build the official SAM 2 Hiera trunk and load its pretrained weights."""

from pathlib import Path

import torch
from torch import nn


def build_encoder(hiera_config, checkpoint=None):
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM 2.1 checkpoint not found: {checkpoint}")

    from sam2.modeling.backbones.hieradet import Hiera

    encoder = Hiera(**hiera_config)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        prefix = "image_encoder.trunk."
        trunk_weights = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
        encoder.load_state_dict(trunk_weights, strict=True)
    return encoder


def add_nir_input(encoder, channels):
    """Add zero-initialized NIR filters after loading the pretrained RGB filters."""
    projection = encoder.patch_embed.proj
    weight = projection.weight.detach()
    nir_weight = weight.new_zeros((weight.shape[0], channels, *weight.shape[2:]))
    projection.weight = nn.Parameter(torch.cat((weight, nir_weight), dim=1), requires_grad=projection.weight.requires_grad)
    projection.in_channels += channels
