"""Train, evaluate and report the visual MOC-6 baseline."""
from dataclasses import asdict
from datetime import datetime
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
from torch.utils.data import DataLoader, Subset

from config import ALLOW_TF32, ENCODER_CHANNELS, INPUT_SIZE_MULTIPLE, LOSS_FUNCTIONS, RGB_MEAN, RGB_STD
from moc_adapter import CLASSES, DENSITY_SCALE, MOC6, count_errors
from models.backbone.rwkvsam import load_encoder
from models.counter import Counter
from models.counters.mcc import MCC
from project import EXP_CODE_DIR_AND_FILE, EXP_SAVE_PATH, PROJECT_ROOT_DIR


def save_density(path, rgb, target, prediction, dots):
    fig, axes = plt.subplots(3, 6, figsize=(16, 8), layout="constrained")
    image = rgb[0].detach().cpu().permute(1, 2, 0).numpy()
    axes[0, 0].imshow(np.clip(image * RGB_STD + RGB_MEAN, 0, 1))
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
def evaluate(model, loader, device, bf16, directory, epoch):
    model.eval()
    absolute, squared, weighted = [], [], []
    predicted_counts, target_counts = [], []
    for index, batch in enumerate(loader):
        rgb, dots = batch["rgb"], batch["dots"]
        nir = batch.get("nir")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            density, counts = model(rgb.to(device), nir.to(device) if nir is not None else None)
        if not torch.isfinite(density).all():
            raise FloatingPointError(f"Non-finite evaluation density at sample {index}")
        pred = density.cpu()
        abs_errors, square_errors, weights = count_errors(pred, dots)
        absolute.append(abs_errors)
        squared.append(square_errors)
        weighted.append(sum(error * weight for error, weight in zip(square_errors, weights, strict=True)))
        predicted_counts.append(counts[0].cpu().numpy())
        target_counts.append(dots[0].sum((-2, -1)).numpy() / DENSITY_SCALE)
        if index == 0:
            target = model.density_target(dots.to(device))
            save_density(directory / f"density_epoch_{epoch:03}.png", rgb, target, pred, dots)
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


def train_epoch(model, loader, optimizer, args, device, log, epoch):
    model.train()
    for index, batch in enumerate(loader):
        rgb, dots = batch["rgb"].to(device), batch["dots"].to(device)
        nir = batch.get("nir")
        masks = batch.get("masks")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
            density, counts, *aux = model(rgb, nir.to(device) if nir is not None else None)
        loss = model.loss(density, dots, aux[0] if aux else None, masks.to(device) if masks is not None else None)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {index}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        optimizer.step()
        record = {
            "epoch": epoch, "step": index + 1,
            "loss": loss.item(), "loss_function": model.loss_function,
            "grad_norm": norm.item(),
            "gt": (dots.sum((-2, -1)) / DENSITY_SCALE).tolist(), "pred": counts.detach().tolist(),
        }
        if model.loss_function == "Mse_loss":
            record["density_mse"] = record["loss"]
        print(json.dumps(record), file=log, flush=True)
        if index == 0 or args.steps is not None:
            print(json.dumps(record))
        if args.steps is not None and index + 1 >= args.steps:
            break


def configure_device(args):
    torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = ALLOW_TF32
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Upstream VRWKV requires a CUDA-enabled PyTorch environment")
    torch.cuda.set_device(device)
    if args.bf16 and not torch.cuda.is_bf16_supported():
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


def validate_config(args):
    if args.mode not in ("train", "test"):
        raise ValueError(f"Unknown run mode in config.py: {args.mode}")
    config_name = "TRAIN_CONFIG" if args.mode == "train" else "TEST_CONFIG"
    if args.encoder_checkpoint is None:
        raise ValueError(f"Set {config_name}.encoder_checkpoint in config.py")
    if args.variant not in ENCODER_CHANNELS:
        raise ValueError(f"Unknown encoder variant: {args.variant}")
    if args.split not in ("val", "test"):
        raise ValueError("Set split to 'val' or 'test' in config.py")
    if args.crop_size < INPUT_SIZE_MULTIPLE or args.crop_size % INPUT_SIZE_MULTIPLE:
        raise ValueError(f"crop_size must be a positive multiple of {INPUT_SIZE_MULTIPLE}")
    if args.eval_samples is not None and args.eval_samples < 1:
        raise ValueError("eval_samples must be positive when set")
    if args.mode == "test":
        if args.evaluate_checkpoint is None:
            raise ValueError("Set TEST_CONFIG.evaluate_checkpoint in config.py")
        return

    if args.evaluate_checkpoint is not None or args.split != "val":
        raise ValueError("Training configuration uses the val split and no evaluation checkpoint")
    if args.loss_function not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {args.loss_function}")
    if args.loss_function == "Mask_loss" and args.mask_dir is None:
        raise ValueError("Set TRAIN_CONFIG.mask_dir in config.py for Mask_loss")
    if args.epochs < 1 or args.batch_size < 1 or (args.steps is not None and args.steps < 1):
        raise ValueError("epochs, batch_size and steps must be positive when set")
    for name in ("encoder_lr", "base_lr", "weight_decay", "cos_alpha"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def run(args):
    validate_config(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = configure_device(args)

    validation = MOC6(args.data, args.split, use_nir=args.nir)
    validation[0]
    if args.eval_samples is not None:
        validation = Subset(validation, range(min(len(validation), args.eval_samples)))
    val_loader = DataLoader(validation, batch_size=1)
    loss_function = args.loss_function
    if args.evaluate_checkpoint is None:
        mask_dir = args.mask_dir if loss_function == "Mask_loss" else None
        training = MOC6(args.data, "train", args.crop_size, use_nir=args.nir, mask_dir=mask_dir)
        training[0]
        train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, drop_last=True)
        if not len(train_loader):
            raise ValueError("batch_size exceeds the training split size")
        random.seed(args.seed)

    encoder = load_encoder(args.ovsam, args.encoder_checkpoint, args.variant)
    checkpoint = None
    if args.evaluate_checkpoint is not None:
        checkpoint = torch.load(args.evaluate_checkpoint, weights_only=True, map_location="cpu")
    use_masks = loss_function == "Mask_loss" if checkpoint is None else "mask_head.weight" in checkpoint
    network = MCC(
        encoder, ENCODER_CHANNELS[args.variant], reference_size=args.crop_size, use_nir=args.nir, use_masks=use_masks
    )
    if checkpoint is not None:
        network.load_state_dict(checkpoint)
        del checkpoint
    model = Counter(network, loss_function, args.cos_alpha).to(device)
    if args.evaluate_checkpoint is None:
        optimizer = torch.optim.AdamW([
            {"params": network.encoder.parameters(), "lr": args.encoder_lr},
            {"params": [parameter for name, parameter in network.named_parameters() if not name.startswith("encoder.")],
             "lr": args.base_lr},
        ], weight_decay=args.weight_decay)

    history = []
    output = args.output
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        modality = "rgb_nir" if args.nir else "rgb"
        output = Path(PROJECT_ROOT_DIR) / EXP_SAVE_PATH / f"{args.mode}_{args.variant}_{modality}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    save_code_snapshot(output)
    (output / "config.json").write_text(
        json.dumps(asdict(args) | {"output": output}, default=str, indent=2), encoding="utf-8"
    )
    with (output / "train.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, (1 if args.evaluate_checkpoint else args.epochs) + 1):
            if args.evaluate_checkpoint is None:
                train_epoch(model, train_loader, optimizer, args, device, log, epoch)
                torch.save(network.state_dict(), output / "latest.pt")
            history.append(evaluate(model, val_loader, device, args.bf16, output, epoch))
            print(json.dumps(history[-1]))
            (output / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            plot_metrics(output, history)
