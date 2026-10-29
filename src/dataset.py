"""Image loading, caching and GPU-side augmentation.

Every image is converted to 3-channel RGB and resized to IMG_SIZE x IMG_SIZE once,
cached as uint8, and kept on the GPU during training. The same `load_image` is used
by the Streamlit app so training and inference preprocess identically.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402


def to_rgb_square(img: Image.Image) -> np.ndarray:
    """PIL image (any mode/size) -> uint8 array [IMG_SIZE, IMG_SIZE, 3]."""
    img = img.convert("RGB").resize((C.IMG_SIZE, C.IMG_SIZE), Image.BICUBIC)
    return np.array(img, dtype=np.uint8)


def load_image(path) -> np.ndarray:
    with Image.open(path) as im:
        im.load()
        return to_rgb_square(im)


def load_splits(split: str = "splits") -> pd.DataFrame:
    return pd.read_csv(C.PROCESSED_DIR / f"{split}.csv")


def load_fold(fold: str, split: str = "splits"):
    """Returns (uint8 tensor [N,3,H,W] on CPU, int64 labels [N], dataframe).

    `split` names a csv in data/processed ("splits" = our leak-free split)."""
    df = load_splits(split)
    df = df[df.fold == fold].reset_index(drop=True)
    stem = f"cache_{C.IMG_SIZE}_{fold}" if split == "splits" else f"cache_{C.IMG_SIZE}_{split}_{fold}"
    cache = C.PROCESSED_DIR / f"{stem}.npy"
    if cache.exists():
        arr = np.load(cache)
        if len(arr) == len(df):
            return _to_tensor(arr), torch.tensor(df["label_id"].values), df
    arr = np.stack([load_image(C.RAW_DIR / p) for p in df["path"]])
    np.save(cache, arr)
    return _to_tensor(arr), torch.tensor(df["label_id"].values), df


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


_MEAN = torch.tensor(C.MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(C.STD).view(1, 3, 1, 1)


def normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [N,3,H,W] (any device) -> normalized float32."""
    x = x_uint8.float() / 255.0
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


def augment(x_uint8: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Light, label-preserving augmentation on a GPU batch. Training data only.

    Horizontal flip, rotation +-15 deg, scale 0.9-1.1, translation +-6%,
    brightness +-15% and contrast +-15%. Returns normalized float32.
    """
    n, dev = x_uint8.shape[0], x_uint8.device

    def rnd(lo, hi):
        return torch.rand(n, device=dev, generator=gen) * (hi - lo) + lo

    ang = rnd(-15, 15) * math.pi / 180
    scale = rnd(0.9, 1.1)
    flip = torch.where(torch.rand(n, device=dev, generator=gen) < 0.5, -1.0, 1.0)
    tx, ty = rnd(-0.06, 0.06) * 2, rnd(-0.06, 0.06) * 2
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([
        torch.stack([cos * flip, -sin, tx], 1),
        torch.stack([sin * flip, cos, ty], 1),
    ], 1)
    x = x_uint8.float() / 255.0
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    contrast = rnd(0.85, 1.15).view(n, 1, 1, 1)
    bright = rnd(-0.15, 0.15).view(n, 1, 1, 1)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * contrast + mean + bright * 0.5).clamp(0, 1)
    return (x - _MEAN.to(dev)) / _STD.to(dev)
