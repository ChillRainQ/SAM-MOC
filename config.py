"""NWPU-MOC-style settings for training the RWKV-SAM counting model."""
from pathlib import Path

from easydict import EasyDict as edict

from project import EXP_SAVE_PATH, PROJECT_ROOT_DIR, THIRD_PARTY_DIR


cfg = edict()

# Training
cfg.SEED = 42
cfg.DATASET = "MOC_RS"
cfg.DATA_PATH = Path(PROJECT_ROOT_DIR) / "NWPU-MOC"
cfg.NET = "MCC"
cfg.MM = True
cfg.GPU_ID = [2]
cfg.BF16 = False
cfg.ALLOW_TF32 = False
cfg.PRE_BACKBONE_WEIGHT = None
cfg.OVSAM_ROOT = Path(PROJECT_ROOT_DIR) / THIRD_PARTY_DIR / "ovsam"
cfg.RESUME = False
cfg.RESUME_PATH = None
cfg.BACKBONE_FREEZE = False
cfg.TRAIN_SIZE = (512, 512)
cfg.TRAIN_BATCH_SIZE = 4
cfg.BASE_LR = 5e-5
cfg.CONV_LR = 1e-6
cfg.WEIGHT_DECAY = 1e-4
cfg.MAX_EPOCH = 200
cfg.MAX_STEPS = None
cfg.LOSS_FUNCTION = "Mse_loss"
cfg.alpha = 1.0
cfg.MIX_L1_WEIGHT = 1e-4
cfg.MASK_CE_WEIGHT = 1e-4
cfg.MASK_DIR = None

cfg.MODEL_ARCH = edict(
    backbone=dict(
        variant="large",
        channels={"tiny": (32, 64, 192), "small": (64, 128, 384), "large": (160, 320, 1024)},
        attn_type="rwkv",
        attn_cfg={"depth": 14, "mlp_ratio": 2},
        with_pos_embd=False,
        additional_final_layers=False,
        out_indices=(0, 1, -1),
        strides=(4, 8, 16),
        input_size_multiple=32,
    ),
    fpn=dict(width=128, kernel_size=3),
    decode_head=dict(width=64, kernel_size=3, norm_groups=8, initial_count=1.0, weight_std=0.001),
    nir=dict(channels=1, kernel_size=3),
    mask=dict(classes=2),
    gaussian=dict(radius=7, sigma=4),
)

# Experiments and validation (VAL_STAGE uses zero-based epochs, as in NWPU-MOC)
cfg.EXP_PATH = Path(PROJECT_ROOT_DIR) / EXP_SAVE_PATH
cfg.EXP_NAME = None
cfg.PRINT_FREQ = 20
cfg.VAL_STAGE = [0, 50, 100, 200]
cfg.VAL_FREQ = [10, 4, 4, 4]
cfg.VAL_SAMPLES = None
cfg.CP_FREQ = 10

LOSS_FUNCTIONS = ("Mse_loss", "Cos_loss", "Mix_loss", "Mask_loss")
