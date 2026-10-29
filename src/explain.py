"""Grad-CAM for both architectures (hooks `model.features`).

  python src/explain.py --gallery      # figures for correct and misclassified test images
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.dataset import normalize  # noqa: E402


class GradCAM:
    """Grad-CAM (Selvaraju et al. 2017) on the last convolutional feature map."""

    def __init__(self, model):
        self.model = model.eval()
        self._act = None
        self._grad = None
        model.features.register_forward_hook(self._save)

    def _save(self, _m, _inp, out):
        self._act = out
        if out.requires_grad:                                  # skip plain no_grad predictions
            out.register_hook(lambda g: setattr(self, "_grad", g))

    def __call__(self, x_uint8: torch.Tensor, class_idx: int | None = None):
        """x_uint8: [1,3,H,W]. Returns (cam [H,W] in 0..1, probs [n_classes], class_idx)."""
        dev = next(self.model.parameters()).device
        x = normalize(x_uint8.to(dev)).requires_grad_(True)   # grad flows even if weights frozen
        with torch.enable_grad():
            logits = self.model(x)
            idx = int(logits.argmax(1)) if class_idx is None else class_idx
            self.model.zero_grad(set_to_none=True)
            logits[0, idx].backward()
        w = self._grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self._act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x_uint8.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam.numpy(), torch.softmax(logits.detach().float(), 1)[0].cpu().numpy(), idx


def overlay(img_rgb_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a jet heatmap on the image. Returns uint8 [H,W,3]."""
    from matplotlib import colormaps
    heat = colormaps["jet"](cam)[..., :3]
    out = (1 - alpha) * (img_rgb_uint8 / 255.0) + alpha * heat
    return (out.clip(0, 1) * 255).astype(np.uint8)


def gallery(n_correct: int = 1, n_wrong: int = 6) -> None:
    """Grad-CAM panels for each exported model: correct examples per class + misclassified ones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from src.dataset import load_fold
    from src.inference import load_model

    x, y, df = load_fold("test")
    for name in ("cnn", "mobilenet"):
        path = C.EXPORT_DIR / f"{name}.pt"
        preds_path = C.REPORTS_DIR / "predictions" / f"{name}.csv"
        if not path.exists() or not preds_path.exists():
            print(f"skip {name}: run evaluate.py first")
            continue
        model, _ = load_model(path)
        cam_fn = GradCAM(model)
        p = pd.read_csv(preds_path)
        rows = []
        for c in range(len(C.CLASSES)):
            ok = p[(p.label_id == c) & (p.pred == c)].sort_values("conf", ascending=False)
            rows += list(ok.index[:n_correct])
        wrong = p[p.label_id != p.pred].sort_values("conf", ascending=False)
        rows += list(wrong.index[:n_wrong])
        ncol = 4
        nrow = int(np.ceil(len(rows) / ncol))
        fig, ax = plt.subplots(nrow, ncol * 2, figsize=(ncol * 4.6, nrow * 2.4))
        ax = np.atleast_2d(ax)
        for a in ax.ravel():
            a.axis("off")
        for k, i in enumerate(rows):
            r, c0 = divmod(k, ncol)
            cam, probs, pred = cam_fn(x[i:i + 1])
            img = x[i].permute(1, 2, 0).numpy()
            ok = pred == int(y[i])
            ax[r, 2 * c0].imshow(img)
            ax[r, 2 * c0 + 1].imshow(overlay(img, cam))
            ax[r, 2 * c0].set_title(f"true: {C.CLASS_DISPLAY[C.CLASSES[int(y[i])]]}", fontsize=8)
            ax[r, 2 * c0 + 1].set_title(f"pred: {C.CLASS_DISPLAY[C.CLASSES[pred]]} {probs[pred]:.0%}",
                                        fontsize=8, color="green" if ok else "crimson")
        fig.suptitle(f"Grad-CAM, {name}: top-confidence correct per class, then most confident errors", fontsize=10)
        fig.tight_layout()
        C.FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(C.FIG_DIR / f"gradcam_{name}.png", dpi=110)
        plt.close(fig)
        print("wrote", C.FIG_DIR / f"gradcam_{name}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", action="store_true")
    a = ap.parse_args()
    if a.gallery:
        gallery()
