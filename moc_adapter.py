"""RGB/NIR data adapter and evaluation protocol for NWPU-MOC."""

from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

from config import DEFAULT_CROP_SIZE, INPUT_SIZE_MULTIPLE, RGB_MEAN, RGB_STD
from datasets.setting.MOC_RS import cfg_data
from misc.transforms import GTScaleDown, RandomCrop, RandomHorizontallyFlip

CLASSES = tuple(name.capitalize() for name in cfg_data.CATEGORYS)
LABEL_FACTOR = cfg_data.LABEL_FACTOR
DENSITY_SCALE = cfg_data.LOG_PARA
SPLIT_FILES = {"train": cfg_data.TRAIN_LST, "val": cfg_data.VAL_LST, "test": cfg_data.TEST_LST}


def open_image(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty NWPU-MOC image: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


class MOC6(Dataset):
    def __init__(self, root, split, crop_size=DEFAULT_CROP_SIZE, use_nir=False, mask_dir=None):
        self.root = Path(root)
        self.split = split
        self.use_nir = use_nir
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.names = (self.root / SPLIT_FILES[split]).read_text(encoding="utf-8-sig").split()
        self.crop_size = crop_size
        self.crop = RandomCrop(crop_size)
        self.flip = RandomHorizontallyFlip()
        self.downsample = GTScaleDown(LABEL_FACTOR)
        self.mean = torch.tensor(RGB_MEAN)[:, None, None]
        self.std = torch.tensor(RGB_STD)[:, None, None]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        rgb = open_image(self.root / "rgb" / name)
        # A placeholder lets the RGB-only path use the same paired geometry functions.
        nir = rgb
        if self.use_nir:
            nir = open_image(self.root / "ir" / name.replace("orth25", "ir"))
            if nir.size != rgb.size:
                raise ValueError(f"RGB and NIR sizes differ for {name}: {rgb.size} vs {nir.size}")
        label_name = Path(name).with_suffix(".npz")
        with np.load(self.root / "gt" / label_name, allow_pickle=False) as archive:
            dots = archive["arr_0"]
        if dots.ndim != 3 or dots.shape[-1] < len(CLASSES) or dots.shape[:2] != (rgb.height, rgb.width):
            raise ValueError(f"Invalid NWPU-MOC label shape for {name}: {dots.shape}")
        dots = dots[:, :, :len(CLASSES)]
        masks = None
        if self.mask_dir is not None:
            mask_path = self.mask_dir / label_name
            with np.load(mask_path, allow_pickle=False) as archive:
                masks = archive["mask_map"]
            if masks.shape != dots.shape or not np.all((masks == 0) | (masks == 1)):
                raise ValueError(f"Expected binary masks with shape {dots.shape} in {mask_path}")
        if self.split == "train":
            rgb, nir, dots, masks = self.augment(rgb, nir, dots, masks)
        rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float().div_(255)
        labels = torch.from_numpy(self.downsample(dots)).permute(2, 0, 1).mul_(DENSITY_SCALE)
        result = {"rgb": (rgb - self.mean) / self.std, "dots": labels}
        if masks is not None:
            out_h, out_w = labels.shape[-2:]
            rows = np.arange(out_h) * masks.shape[0] // out_h
            columns = np.arange(out_w) * masks.shape[1] // out_w
            result["masks"] = torch.from_numpy(masks[rows[:, None], columns]).permute(2, 0, 1).long()
        if self.use_nir:
            # NWPU-MOC uses the first channel of its stored IR image.
            first_band = torch.from_numpy(np.array(nir.getchannel(0))).float().div_(255).unsqueeze(0)
            mean, std = cfg_data.MEAN_STD_IR
            result["nir"] = (first_band - mean[0]) / std[0]
        return result

    def augment(self, rgb, nir, dots, masks):
        height, width = dots.shape[:2]
        crop = self.crop_size
        rate = max(random.uniform(0.8, 1.2), crop / height, crop / width)
        new_h = max(crop, int(height * rate) // INPUT_SIZE_MULTIPLE * INPUT_SIZE_MULTIPLE)
        new_w = max(crop, int(width * rate) // INPUT_SIZE_MULTIPLE * INPUT_SIZE_MULTIPLE)
        rgb = rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
        nir = nir.resize((new_w, new_h), Image.Resampling.BILINEAR)
        y, x, c = np.nonzero(dots)
        resized = np.zeros((new_h, new_w, len(CLASSES)), dtype=np.float32)
        np.add.at(resized, (y * new_h // height, x * new_w // width, c), dots[y, x, c])
        if masks is not None:
            rows = np.arange(new_h) * height // new_h
            columns = np.arange(new_w) * width // new_w
            resized = np.concatenate((resized, masks[rows[:, None], columns]), axis=-1)
        rgb, nir, dots = self.crop(rgb, nir, resized)
        rgb, nir, dots = self.flip(rgb, nir, dots)
        if masks is not None:
            dots, masks = np.split(dots, 2, axis=-1)
        if random.random() < 0.1:
            rgb = ImageOps.grayscale(rgb).convert("RGB")
        if random.random() < 0.5:
            gamma = random.uniform(0.4, 2)
            rgb = rgb.point([min(255, int(256 * (value / 255) ** gamma)) for value in range(256)] * 3)
        return rgb, nir, dots, masks


def count_errors(prediction, target):
    """Return per-class absolute errors, squared errors, and NWPU-MOC weights."""
    if prediction.ndim == target.ndim == 4 and prediction.shape[0] == target.shape[0] == 1:
        prediction, target = prediction[0], target[0]
    if prediction.shape != target.shape or target.ndim != 3 or target.shape[0] != len(CLASSES):
        raise ValueError("Expected matching [6, H, W] maps for one NWPU-MOC image")
    predicted_counts = np.asarray([layer.sum() for layer in prediction.detach().cpu().numpy()]) / DENSITY_SCALE
    true_counts = np.asarray([layer.sum() for layer in target.detach().cpu().numpy()]) / DENSITY_SCALE
    errors = true_counts - predicted_counts
    offset = 1.02
    relative_counts = (true_counts + offset) / (np.median(true_counts) + offset)
    unnormalized = np.exp(1 / np.log(relative_counts + offset))
    return np.abs(errors), errors * errors, unnormalized / unnormalized.sum()
