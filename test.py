"""Evaluate the saved checkpoint of an existing experiment."""
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import random

import numpy as np
from numpy._core import multiarray
import torch

from config import cfg
from datasets import createTestData, createValData
from datasets.basedataset import Dataset
from models.backbone.rwkvsam import build_encoder
from models.counter import Counter
from models.counters.mcc import MCC
from project import PROJECT_ROOT_DIR
from trainer import configure_device, evaluate, make_dataset_config, plot_metrics


@dataclass(frozen=True, kw_only=True)
class TestConfig:
    model_path: Path | None = None
    gpu_id: int = 0
    split: str = "test"
    eval_samples: int | None = None


TEST_CONFIG = TestConfig(
    model_path=None,  # For example: Path("exp/<experiment>/epoch_020.pt")
    gpu_id=0,
)


def load_checkpoint_weights(path):
    # NWPU-MOC checkpoints may include NumPy metrics in their training record.
    numpy_globals = [
        np.ndarray, np.dtype, multiarray.scalar, multiarray._reconstruct,
        (multiarray.scalar, "numpy.core.multiarray.scalar"),
        (multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        *(type(np.dtype(dtype)) for dtype in (np.float32, np.float64, np.int32, np.int64)),
    ]
    with torch.serialization.safe_globals(numpy_globals):
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    state = checkpoint.get("net", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    if not isinstance(state, Mapping) or not state or not all(isinstance(key, str) for key in state):
        raise ValueError(f"Expected a model state_dict or a checkpoint containing 'net': {path}")

    def strip_parallel(key):
        return ".".join(part for part in key.split(".") if part != "module")

    normalized = OrderedDict((strip_parallel(key), value) for key, value in state.items())
    if len(normalized) != len(state):
        raise ValueError(f"Duplicate model keys after removing DataParallel prefixes: {path}")
    metadata = {strip_parallel(key): value for key, value in getattr(state, "_metadata", {}).items()}

    # A saved Counter contains the MCC weights under network.; NWPU's CC uses CCN.
    prefixes = [prefix for prefix in ("network", "CCN") if any(key.startswith(prefix + ".") for key in normalized)]
    if len(prefixes) > 1:
        raise ValueError(f"Checkpoint contains both network and CCN weights: {path}")
    if prefixes:
        prefix = prefixes[0]
        normalized = OrderedDict(
            (key[len(prefix) + 1:], value) for key, value in normalized.items() if key.startswith(prefix + ".")
        )
        metadata = {
            key[len(prefix) + 1:]: value for key, value in metadata.items()
            if key == prefix or key.startswith(prefix + ".")
        }
    normalized._metadata = metadata
    return normalized


def run_test(settings):
    if settings.model_path is None:
        raise ValueError("Set TEST_CONFIG.model_path to an existing MCC checkpoint in test.py")
    if type(settings.gpu_id) is not int or settings.gpu_id < 0:
        raise ValueError("Set TEST_CONFIG.gpu_id to a nonnegative CUDA device index")
    if settings.split not in ("val", "test"):
        raise ValueError("Set TEST_CONFIG.split to 'val' or 'test'")
    if settings.eval_samples is not None and settings.eval_samples < 1:
        raise ValueError("TEST_CONFIG.eval_samples must be positive when set")

    checkpoint_path = (Path(PROJECT_ROOT_DIR) / settings.model_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing trained MCC checkpoint: {checkpoint_path}")
    experiment = next((parent for parent in checkpoint_path.parents if (parent / "config.json").is_file()), None)
    if experiment is None:
        raise FileNotFoundError(f"No experiment config.json found above checkpoint: {checkpoint_path}")
    saved = json.loads((experiment / "config.json").read_text(encoding="utf-8"))
    if "MODEL_ARCH" in saved:
        variant = saved["MODEL_ARCH"]["backbone"]["variant"]
        train_size = tuple(saved["TRAIN_SIZE"])
        use_nir, bf16, seed = saved["MM"], saved["BF16"], saved["SEED"]
        data_path, ovsam_root = saved["DATA_PATH"], saved["OVSAM_ROOT"]
    else:
        variant = saved["variant"]
        train_size = (saved["crop_size"],) * 2
        use_nir, bf16, seed = saved["nir"], saved["bf16"], saved["seed"]
        data_path, ovsam_root = saved["data"], saved["ovsam"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = configure_device(settings.gpu_id, bf16, saved.get("ALLOW_TF32", cfg.ALLOW_TF32))

    make_loader = createTestData if settings.split == "test" else createValData
    loader = make_loader("MOC_RS", Dataset, make_dataset_config(data_path, train_size))
    state = load_checkpoint_weights(checkpoint_path)
    network = MCC(
        build_encoder(ovsam_root, variant), cfg.MODEL_ARCH.backbone.channels[variant],
        reference_size=train_size[0], use_nir=use_nir, use_masks="mask_head.weight" in state,
    )
    network.load_state_dict(state, strict=True)
    del state
    model = Counter(network, use_nir=use_nir).to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = experiment / f"{settings.split}_{timestamp}"
    output.mkdir(exist_ok=False)
    metrics = evaluate(model, loader, device, bf16, output, epoch=None, eval_samples=settings.eval_samples)
    metrics["split"] = settings.split
    metrics["checkpoint"] = checkpoint_path.relative_to(experiment).as_posix()
    metrics["gpu_id"] = settings.gpu_id
    print(json.dumps(metrics))
    (output / "metrics.json").write_text(json.dumps([metrics], indent=2), encoding="utf-8")
    plot_metrics(output, [metrics])


def main():
    run_test(TEST_CONFIG)


if __name__ == "__main__":
    main()
