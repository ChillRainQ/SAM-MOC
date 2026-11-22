from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

from project import PROJECT_ROOT_DIR, THIRD_PARTY_DIR

DEFAULT_DATA_ROOT = Path(PROJECT_ROOT_DIR) / "NWPU-MOC"
DEFAULT_OVSAM_ROOT = Path(PROJECT_ROOT_DIR) / THIRD_PARTY_DIR / "ovsam"
DEFAULT_CROP_SIZE = 512
DEFAULT_BATCH_SIZE = 1
DEFAULT_VARIANT = "large"
ALLOW_TF32 = False
LOSS_FUNCTION = "Mse_loss"
LOSS_FUNCTIONS = ("Mse_loss", "Cos_loss", "Mix_loss", "Mask_loss")
COS_LOSS_ALPHA = 1.0
MIX_L1_WEIGHT = 1e-4
MASK_CE_WEIGHT = 1e-4
MASK_CLASSES = 2

ENCODER_CHANNELS = {"tiny": (32, 64, 192), "small": (64, 128, 384), "large": (160, 320, 1024)}
ENCODER_ATTENTION_TYPE = "rwkv"
ENCODER_ATTENTION_CONFIG = {"depth": 14, "mlp_ratio": 2}
ENCODER_WITH_POS_EMBEDDING = False
ENCODER_ADDITIONAL_FINAL_LAYERS = False
ENCODER_FEATURE_INDICES = (0, 1, -1)
ENCODER_FEATURE_STRIDES = (4, 8, 16)
INPUT_SIZE_MULTIPLE = 32
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)

FPN_WIDTH = 128
FPN_KERNEL_SIZE = 3
NORM_GROUPS = 8
HEAD_WIDTH = 64
HEAD_KERNEL_SIZE = 3
HEAD_INITIAL_COUNT = 1.0
HEAD_WEIGHT_STD = 0.001
TARGET_GAUSSIAN_RADIUS = 7
TARGET_GAUSSIAN_SIGMA = 4
NIR_CHANNELS = 1
NIR_RESIDUAL_KERNEL_SIZE = 3

# Used by the copied NWPU-MOC settings at import time.
cfg = SimpleNamespace(TRAIN_SIZE=(DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE), TRAIN_BATCH_SIZE=DEFAULT_BATCH_SIZE)


@dataclass(frozen=True, kw_only=True)
class RunConfig:
    mode: str = "train"
    data: Path = DEFAULT_DATA_ROOT
    ovsam: Path = DEFAULT_OVSAM_ROOT
    encoder_checkpoint: Path | None = None
    evaluate_checkpoint: Path | None = None
    output: Path | None = None
    variant: str = DEFAULT_VARIANT
    device: str = "cuda:0"
    bf16: bool = False
    nir: bool = False
    crop_size: int = DEFAULT_CROP_SIZE
    seed: int = 42
    split: str = "val"
    eval_samples: int | None = None
    epochs: int = 1
    steps: int | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    encoder_lr: float = 1e-6
    base_lr: float = 5e-5
    weight_decay: float = 1e-4
    loss_function: str = LOSS_FUNCTION
    cos_alpha: float = COS_LOSS_ALPHA
    mask_dir: Path | None = None


TRAIN_CONFIG = RunConfig(
    encoder_checkpoint=None,  # Path to matching pretrained RWKV-SAM weights
)
TEST_CONFIG = replace(
    TRAIN_CONFIG,
    mode="test",
    split="test",
    evaluate_checkpoint=None,  # Path to a trained MCC checkpoint
    output=None,
)
