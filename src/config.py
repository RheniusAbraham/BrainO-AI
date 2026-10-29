"""Shared paths and constants. Every script imports from here so settings never drift."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "Dataset"                      # original dataset, never modified
PROCESSED_DIR = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "models" / "checkpoints"
EXPORT_DIR = ROOT / "models" / "exported"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

# Fixed class order: index in this list == label id used everywhere.
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_DISPLAY = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary tumor",
}

IMG_SIZE = 224
# ImageNet statistics: required by pretrained MobileNetV2, also used for the CNN
# so both models see identical inputs.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

SEED = 42
VAL_FRACTION = 0.15
# Perceptual-hash Hamming distance (of 64 bits) at or below which two images are
# treated as near-duplicates. Cross-label matches first appear at 2 (3 pairs) and
# grow fast beyond it (33 at 4, 289 at 6), so 2 is the largest safe value.
NEAR_DUP_THRESHOLD = 2


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, CKPT_DIR, EXPORT_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
