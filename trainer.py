"""Train, evaluate and report the visual MOC-6 baseline."""
from datetime import datetime
from copy import copy
from functools import partial
from itertools import islice
import json
import math
from pathlib import Path
import random
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import cfg, LOSS_FUNCTIONS
from datasets import createTrainData, createValData
from datasets.basedataset import Dataset
from datasets.mask_supervision import MaskDataset, mask_collate
from datasets.setting.MOC_RS import cfg_data
from misc.utils import eval_mc, is_validation
from models.backbone.rwkvsam import build_encoder, load_encoder
from models.counter import Counter
from models.counters.mcc import MCC
from project import EXP_CODE_DIR_AND_FILE

CLASSES = tuple(name.capitalize() for name in cfg_data.CATEGORYS)
DENSITY_SCALE = cfg_data.LOG_PARA


def save_density(path, rgb, target, prediction, dots):
    fig, axes = plt.subplots(3, 6, figsize=(16, 8), layout="constrained")
    image = rgb[0].detach().cpu().permute(1, 2, 0).numpy()
    mean, std = cfg_data.MEAN_STD_RGB
    axes[0, 0].imshow(np.clip(image * std + mean, 0, 1))
    axes[0, 0].set_title("RGB")
    for i, name in enumerate(CLASSES):
        truth = target[0, i].cpu().numpy()
        pred = prediction[0, i].float().cpu().numpy()
        limit = max(float(truth.max()), float(pred.max()), 1e-6)
        axes[1, i].imshow(truth, vmin=0, vmax=limit)
        y, x = np.nonzero(dots[0, i].cpu().numpy())
        axes[1, i].scatter(x, y, s=2, c="white")
        axes[1, i].set_title(f"{name} GT {truth.sum() / DENSITY_SCALE:.0f}")
        axes[2, i].imshow(pred, vmin=0, vmax=limit)
        axes[2, i].set_title(f"Pred {pred.sum() / DENSITY_SCALE:.2f}")
    for axis in axes.flat:
        axis.axis("off")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_metrics(directory, history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for axis, key in zip(axes, ("mae", "rmse"), strict=True):
        axis.bar(CLASSES, history[-1][key])
        axis.set_title(f"{key.upper()} · macro {history[-1]['mean_' + key]:.3f}")
        axis.tick_params(axis="x", rotation=25)
    fig.savefig(directory / "class_metrics.png", dpi=140)
    plt.close(fig)
    if len(history) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for axis, key in zip(axes, ("mae", "rmse"), strict=True):
            for i, name in enumerate(CLASSES):
                axis.plot([row["epoch"] for row in history], [row[key][i] for row in history], label=name)
            axis.set(xlabel="Epoch", ylabel=key.upper())
            axis.legend()
        fig.savefig(directory / "class_trends.png", dpi=140)
        plt.close(fig)


@torch.inference_mode()
def evaluate(model, loader, device, bf16, directory, epoch, eval_samples=None):
    model.eval()
    absolute, squared, weighted = [], [], []
    predicted_counts, target_counts = [], []
    for index, batch in enumerate(islice(loader, eval_samples)):
        rgb = batch["rgb"]
        dots = batch["gt_map"][:, :len(CLASSES)]
        inputs = {"rgb": rgb.to(device)}
        if model.use_nir:
            inputs["nir"] = batch["nir"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            outputs = model(inputs, mode="val")
        density, counts = outputs["pred_map"], outputs["counts"]
        if not torch.isfinite(density).all():
            raise FloatingPointError(f"Non-finite evaluation density at sample {index}")
        pred = density.cpu()
        abs_errors, square_errors, weights = eval_mc(pred, dots, DENSITY_SCALE)
        absolute.append(abs_errors)
        squared.append(square_errors)
        weighted.append(sum(error * weight for error, weight in zip(square_errors, weights, strict=True)))
        predicted_counts.append(counts[0].cpu().numpy())
        target_counts.append(dots[0].sum((-2, -1)).numpy() / DENSITY_SCALE)
        if index == 0:
            target = model.density_target(dots.to(device))
            filename = "density.png" if epoch is None else f"density_epoch_{epoch:03}.png"
            save_density(directory / filename, rgb, target, pred, dots)
    mae = np.mean(absolute, axis=0)
    rmse = np.sqrt(np.mean(squared, axis=0))
    counts = np.asarray(predicted_counts)
    positive = np.asarray(target_counts) > 0
    positive_samples = positive.sum(axis=0)
    positive_misses = ((counts < 0.5) & positive).sum(axis=0)
    return {
        "epoch": epoch, "samples": len(absolute), "classes": CLASSES,
        "mae": mae.tolist(), "rmse": rmse.tolist(),
        "mean_mae": float(mae.mean()), "mean_rmse": float(rmse.mean()),
        "cls_weight_mse": float(np.mean(weighted)),
        "mean_pred_count": counts.mean(axis=0).tolist(),
        "exact_zero_rate": (counts == 0).mean(axis=0).tolist(),
        "below_half_rate": (counts < 0.5).mean(axis=0).tolist(),
        "gt_positive_samples": positive_samples.tolist(),
        "gt_positive_below_half_rate": [
            float(missed / total) if total else None
            for missed, total in zip(positive_misses, positive_samples, strict=True)
        ],
    }


def train_epoch(model, loader, optimizer, settings, device, log, epoch):
    model.train()
    if settings.BACKBONE_FREEZE:
        model.network.encoder.eval()
    for index, batch in enumerate(loader):
        batch = {key: value.to(device) for key, value in batch.items() if key != "nir" or model.use_nir}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=settings.BF16):
            outputs = model(batch, mode="train")
        loss = model.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {index}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        optimizer.step()
        record = {
            "epoch": epoch, "step": index + 1,
            "loss": loss.item(), "loss_function": model.loss_function,
            "grad_norm": norm.item(),
            "gt": (batch["gt_map"][:, :len(CLASSES)].sum((-2, -1)) / DENSITY_SCALE).tolist(),
            "pred": outputs["counts"].detach().tolist(),
        }
        if model.loss_function == "Mse_loss":
            record["density_mse"] = record["loss"]
        print(json.dumps(record), file=log, flush=True)
        if (index + 1) % settings.PRINT_FREQ == 0 or settings.MAX_STEPS is not None:
            print(json.dumps(record))
        if settings.MAX_STEPS is not None and index + 1 >= settings.MAX_STEPS:
            break


def configure_device(gpu_id, bf16, allow_tf32=cfg.ALLOW_TF32):
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    device = torch.device("cuda", gpu_id)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Upstream VRWKV requires a CUDA-enabled PyTorch environment")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible")
    torch.cuda.set_device(device)
    if bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable on the selected GPU")
    return device


def save_code_snapshot(output):
    output = output.resolve()
    snapshot = output / "code"
    snapshot.mkdir()

    def ignore_generated(directory, names):
        return {
            name for name in names
            if name == "__pycache__" or name.endswith(".pyc") or Path(directory, name).resolve() == output
        }

    for source in map(Path, EXP_CODE_DIR_AND_FILE):
        target = snapshot / source.name
        if source.is_dir():
            shutil.copytree(source, target, ignore=ignore_generated)
        else:
            shutil.copy2(source, target)


def validate_config(settings):
    if settings.DATASET != "MOC_RS" or settings.NET != "MCC":
        raise ValueError("Use cfg.DATASET='MOC_RS' and cfg.NET='MCC'")
    if len(settings.GPU_ID) != 1 or type(settings.GPU_ID[0]) is not int or settings.GPU_ID[0] < 0:
        raise ValueError("cfg.GPU_ID must contain one nonnegative CUDA device ID")
    if settings.RESUME and not settings.RESUME_PATH:
        raise ValueError("Set cfg.RESUME_PATH when cfg.RESUME is True")
    if not settings.RESUME and not settings.PRE_BACKBONE_WEIGHT:
        raise ValueError("Set cfg.PRE_BACKBONE_WEIGHT in config.py")
    backbone = settings.MODEL_ARCH.backbone
    if backbone.variant not in backbone.channels:
        raise ValueError(f"Unknown encoder variant: {backbone.variant}")
    height, width = settings.TRAIN_SIZE
    if height != width or height < backbone.input_size_multiple or height % backbone.input_size_multiple:
        raise ValueError(f"cfg.TRAIN_SIZE must be square and divisible by {backbone.input_size_multiple}")
    if settings.VAL_SAMPLES is not None and settings.VAL_SAMPLES < 1:
        raise ValueError("cfg.VAL_SAMPLES must be positive when set")
    if settings.LOSS_FUNCTION not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {settings.LOSS_FUNCTION}")
    if settings.LOSS_FUNCTION == "Mask_loss" and not settings.MASK_DIR:
        raise ValueError("Set cfg.MASK_DIR in config.py for Mask_loss")
    if settings.MAX_EPOCH < 1 or settings.TRAIN_BATCH_SIZE < 1 or (settings.MAX_STEPS is not None and settings.MAX_STEPS < 1):
        raise ValueError("MAX_EPOCH, TRAIN_BATCH_SIZE and MAX_STEPS must be positive when set")
    if settings.PRINT_FREQ < 1 or settings.CP_FREQ < 1:
        raise ValueError("cfg.PRINT_FREQ and cfg.CP_FREQ must be positive")
    if (
        not settings.VAL_STAGE or settings.VAL_STAGE[0] != 0
        or sorted(set(settings.VAL_STAGE)) != list(settings.VAL_STAGE)
        or len(settings.VAL_STAGE) != len(settings.VAL_FREQ) or min(settings.VAL_FREQ) < 1
    ):
        raise ValueError("Use increasing VAL_STAGE values starting at 0 and one positive VAL_FREQ per stage")
    for name in ("CONV_LR", "BASE_LR", "WEIGHT_DECAY", "alpha"):
        value = settings[name]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"cfg.{name} must be finite and nonnegative")


def make_dataset_config(data_path, train_size=None, batch_size=None):
    data_config = copy(cfg_data)
    data_config.DATA_PATH = str(data_path)
    if train_size is not None:
        data_config.TRAIN_SIZE = tuple(train_size)
    if batch_size is not None:
        data_config.TRAIN_BATCH_SIZE = batch_size
    return data_config


def run(settings):
    validate_config(settings)
    random.seed(settings.SEED)
    np.random.seed(settings.SEED)
    torch.manual_seed(settings.SEED)
    device = configure_device(settings.GPU_ID[0], settings.BF16, settings.ALLOW_TF32)

    data_config = make_dataset_config(settings.DATA_PATH, settings.TRAIN_SIZE, settings.TRAIN_BATCH_SIZE)
    val_loader = createValData(settings.DATASET, Dataset, data_config)
    val_loader.dataset[0]
    loss_function = settings.LOSS_FUNCTION
    use_masks = loss_function == "Mask_loss"
    dataset_type = partial(MaskDataset, mask_dir=settings.MASK_DIR) if use_masks else Dataset
    train_loader = createTrainData(settings.DATASET, dataset_type, data_config)
    if use_masks:
        train_loader.collate_fn = mask_collate
    train_loader.dataset[0]
    if not len(train_loader):
        raise ValueError("cfg.TRAIN_BATCH_SIZE exceeds the training split size")
    random.seed(settings.SEED)

    backbone = settings.MODEL_ARCH.backbone
    checkpoint = None
    first_epoch = 1
    if settings.RESUME:
        checkpoint = torch.load(settings.RESUME_PATH, weights_only=True, map_location="cpu")
        first_epoch = checkpoint["epoch"] + 1
        if first_epoch > settings.MAX_EPOCH:
            raise ValueError("cfg.MAX_EPOCH must exceed the completed checkpoint epoch")
        encoder = build_encoder(settings.OVSAM_ROOT, backbone.variant)
    else:
        encoder = load_encoder(settings.OVSAM_ROOT, settings.PRE_BACKBONE_WEIGHT, backbone.variant)
    if settings.BACKBONE_FREEZE:
        encoder.requires_grad_(False)
    network = MCC(
        encoder, backbone.channels[backbone.variant], reference_size=settings.TRAIN_SIZE[0],
        use_nir=settings.MM, use_masks=use_masks,
    )
    model = Counter(network, loss_function, settings.alpha, use_nir=settings.MM).to(device)
    optimizer = torch.optim.AdamW([
        {"params": network.encoder.parameters(), "lr": settings.CONV_LR},
        {"params": [parameter for name, parameter in network.named_parameters() if not name.startswith("encoder.")],
         "lr": settings.BASE_LR},
    ], weight_decay=settings.WEIGHT_DECAY)
    if checkpoint is not None:
        network.load_state_dict(checkpoint["net"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        del checkpoint

    history = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    exp_name = settings.EXP_NAME or (
        f"{timestamp}_{settings.DATASET}_MCC_{backbone.variant}_loss_{loss_function}"
        f"_alpha_{settings.alpha:g}_base_lr_{settings.BASE_LR:g}_NIR={settings.MM}"
    )
    output = Path(settings.EXP_PATH) / exp_name
    output.mkdir(parents=True, exist_ok=False)
    save_code_snapshot(output)
    snapshot = dict(settings) | {
        "DATA_PATH": Path(settings.DATA_PATH).resolve(),
        "OVSAM_ROOT": Path(settings.OVSAM_ROOT).resolve(),
        "EXP_PATH": output.parent.resolve(),
        "EXP_NAME": output.name,
    }
    (output / "config.json").write_text(json.dumps(snapshot, default=str, indent=2), encoding="utf-8")
    with (output / "train.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(first_epoch, settings.MAX_EPOCH + 1):
            train_epoch(model, train_loader, optimizer, settings, device, log, epoch)
            state = network.state_dict()
            torch.save(state, output / "latest.pt")
            if epoch % settings.CP_FREQ == 0 or epoch == settings.MAX_EPOCH:
                torch.save({"net": state, "optimizer": optimizer.state_dict(), "epoch": epoch},
                           output / f"epoch_{epoch:03d}.pth")
            if is_validation(settings.VAL_STAGE, settings.VAL_FREQ, epoch - 1):
                history.append(evaluate(model, val_loader, device, settings.BF16, output, epoch, settings.VAL_SAMPLES))
                print(json.dumps(history[-1]))
                (output / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
                plot_metrics(output, history)
