"""Load trained checkpoints and run prediction. Shared by evaluate.py and the app."""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.dataset import normalize, to_rgb_square  # noqa: E402
from src.model import build_model  # noqa: E402


def load_model(path, device="cpu"):
    """Returns (model in eval mode, checkpoint metadata dict)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["arch"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, {k: v for k, v in ck.items() if k != "state_dict"}


@torch.no_grad()
def predict_probs(model, x_uint8: torch.Tensor, device="cpu", bs=128) -> np.ndarray:
    """uint8 [N,3,H,W] -> softmax probabilities [N, n_classes] (float32, fp32 compute)."""
    out = []
    for i in range(0, len(x_uint8), bs):
        logits = model(normalize(x_uint8[i:i + bs].to(device)))
        out.append(torch.softmax(logits.float(), 1).cpu())
    return torch.cat(out).numpy()


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """One PIL image -> uint8 [1,3,H,W], identical preprocessing to training."""
    arr = to_rgb_square(img)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
