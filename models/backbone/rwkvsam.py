"""Load the ViTamin image backbone directly from an ovsam checkout."""
import importlib.util
import os
from pathlib import Path
import sys

import torch

from config import (
    ENCODER_ADDITIONAL_FINAL_LAYERS,
    ENCODER_ATTENTION_CONFIG,
    ENCODER_ATTENTION_TYPE,
    ENCODER_WITH_POS_EMBEDDING,
)


def load_encoder(root, checkpoint, variant):
    root = Path(root).resolve()
    source = root / "projects/rwkvsam/models/backbones/vitamin.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing upstream encoder source: {source}")

    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
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
        encoder = module.VITAMINBackbone(
            model_variant=variant,
            attn_type=ENCODER_ATTENTION_TYPE,
            attn_cfg=ENCODER_ATTENTION_CONFIG.copy(),
            with_pos_embd=ENCODER_WITH_POS_EMBEDDING,
            additional_final_layers=ENCODER_ADDITIONAL_FINAL_LAYERS,
        )
    finally:
        sys.dont_write_bytecode = write_bytecode
        os.chdir(previous_directory)
        sys.path.pop(0)

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
