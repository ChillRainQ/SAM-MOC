"""Load the ViTamin image backbone directly from an ovsam checkout."""
import importlib.util
import os
from pathlib import Path
import sys

import torch

from config import cfg


def build_encoder(root, variant):
    root = Path(root).resolve()
    source = root / "projects/rwkvsam/models/backbones/vitamin.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing upstream encoder source: {source}")

    # The compiler resolves CUDA source paths relative to the checkout.
    previous_directory = Path.cwd()
    write_bytecode = sys.dont_write_bytecode
    sys.path.insert(0, str(root))
    try:
        os.chdir(root)
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location("_sam_moc_vitamin", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        backbone = cfg.RWKV_ARCH.backbone
        return module.VITAMINBackbone(
            model_variant=variant,
            attn_type=backbone.attn_type,
            attn_cfg=backbone.attn_cfg.copy(),
            with_pos_embd=backbone.with_pos_embd,
            additional_final_layers=backbone.additional_final_layers,
        )
    finally:
        sys.dont_write_bytecode = write_bytecode
        os.chdir(previous_directory)
        sys.path.pop(0)


def load_encoder(root, checkpoint, variant):
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    encoder = build_encoder(root, variant)
    expected = encoder.state_dict()
    encoder_weights = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        for prefix in ("backbone_student.", "backbone.", "image_encoder."):
            if key.startswith(prefix):
                key = key.removeprefix(prefix)
                break
        if key in expected:
            if key in encoder_weights:
                raise ValueError(f"Ambiguous encoder checkpoint key: {key}")
            encoder_weights[key] = value
    encoder.load_state_dict(encoder_weights, strict=True)
    return encoder
