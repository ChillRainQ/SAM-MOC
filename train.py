"""Run the NWPU-MOC trainer with the configured counting network."""
from pathlib import Path

from config import cfg


def main():
    import numpy as np
    import torch

    if cfg.NET == "RWKVMCC" and len(cfg.GPU_ID) != 1:
        raise ValueError("RWKV-SAM requires exactly one CUDA device in cfg.GPU_ID")
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA-enabled PyTorch")
    if any(index < 0 or index >= torch.cuda.device_count() for index in cfg.GPU_ID):
        raise ValueError(f"Invalid cfg.GPU_ID {cfg.GPU_ID}; {torch.cuda.device_count()} CUDA device(s) visible")
    torch.cuda.set_device(cfg.GPU_ID[0])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if cfg.SEED is not None:
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)

    from datasets.setting.MOC_RS import cfg_data
    from trainer import Trainer

    Trainer(cfg_data, str(Path(__file__).resolve().parent)).forward()


if __name__ == "__main__":
    main()
