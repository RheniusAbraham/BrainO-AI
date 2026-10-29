"""Draw the pipeline diagram used as Fig. 1 in the paper.  Run: python src/make_pipeline_figure.py"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402

INK, MUTED = "#222222", "#6b7280"
BLUE, ORANGE = "#4C78A8", "#E8743B"

fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.set_xlim(-5, 102)
ax.set_ylim(0, 62)
ax.axis("off")

W = 22.0
XS = [0.0, 26.0, 52.0, 78.0]                       # four columns, 4-unit gaps
TITLE_PT, BODY_PT = 7.8, 6.6


def box(x, y, h, title, lines, face="#F4F5F7", edge=INK, tcolor=INK):
    ax.add_patch(FancyBboxPatch((x, y), W, h, boxstyle="round,pad=0.2,rounding_size=1.1",
                                fc=face, ec=edge, lw=1.1))
    ax.text(x + W / 2, y + h - 2.4, title, ha="center", va="center", fontsize=TITLE_PT,
            fontweight="bold", color=tcolor)
    ax.text(x + W / 2, y + (h - 4.6) / 2, "\n".join(lines), ha="center", va="center",
            fontsize=BODY_PT, color=INK, linespacing=1.4)


def arrow(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1.15, color=INK,
                                 shrinkA=0, shrinkB=0))


# ---- row 1: data preparation ------------------------------------------------------------------
Y1, H1 = 44.0, 15.0
box(XS[0], Y1, H1, "1  Dataset", ["7,023 MRI images", "4 classes: glioma,", "meningioma, pituitary,", "no tumor"])
box(XS[1], Y1, H1, "2  Data audit", ["exact + near duplicates", "(perceptual hash);", "41% of official test", "images have a twin"])
box(XS[2], Y1, H1, "3  Cluster split", ["duplicate clusters never", "cross splits, stratified:", "4,797 / 766 / 765", "(train / val / test)"])
box(XS[3], Y1, H1, "4  Preprocessing", ["RGB, 224 x 224,", "ImageNet normalisation,", "augmentation on", "training set only"])
for i in range(3):
    arrow((XS[i] + W + 0.5, Y1 + H1 / 2), (XS[i + 1] - 0.5, Y1 + H1 / 2))

# ---- row 2: models, selection, evaluation, application --------------------------------------
HM = 12.5
Y5a, Y5b = 21.5, 4.0
box(XS[0], Y5a, HM, "5a  Custom CNN", ["trained from scratch", "2.35 M parameters"],
    face="#E6EEF7", edge=BLUE, tcolor=BLUE)
box(XS[0], Y5b, HM, "5b  MobileNetV2", ["ImageNet weights: head", "only, then fine-tune", "the last 6 blocks"],
    face="#FCEBE2", edge=ORANGE, tcolor=ORANGE)
HR, YR = 15.0, 11.5                                  # selection / evaluation / app boxes
box(XS[1], YR, HR, "6  Model selection", ["3 seeds per model;", "best checkpoint by", "validation macro-F1 only"])
box(XS[2], YR, HR, "7  Test evaluation", ["run once on test set:", "accuracy, macro-F1,", "per-class recall,", "calibration, Grad-CAM"])
box(XS[3], YR, HR, "8  Web application", ["Streamlit: upload MRI,", "class + confidence,", "Grad-CAM heat map"])

# connector: preprocessing -> both models (down, left, then into each model box)
xr = XS[3] + W / 2
yb = Y1 - 0.2
y_turn, x_rail = 40.0, -2.6
c5a, c5b = Y5a + HM / 2, Y5b + HM / 2
ax.plot([xr, xr], [yb, y_turn], color=INK, lw=1.15, solid_capstyle="butt")
ax.plot([xr, x_rail], [y_turn, y_turn], color=INK, lw=1.15, solid_capstyle="butt")
ax.plot([x_rail, x_rail], [y_turn, c5b], color=INK, lw=1.15, solid_capstyle="butt")
arrow((x_rail, c5a), (XS[0] - 0.5, c5a))
arrow((x_rail, c5b), (XS[0] - 0.5, c5b))
ax.text((xr + x_rail) / 2, y_turn + 1.0, "same data, split, seeds and augmentation for both models",
        ha="center", va="bottom", fontsize=6.6, style="italic", color=MUTED)

# models -> selection -> evaluation -> application
arrow((XS[0] + W + 0.5, c5a), (XS[1] - 0.5, YR + HR * 0.72))
arrow((XS[0] + W + 0.5, c5b), (XS[1] - 0.5, YR + HR * 0.28))
arrow((XS[1] + W + 0.5, YR + HR / 2), (XS[2] - 0.5, YR + HR / 2))
arrow((XS[2] + W + 0.5, YR + HR / 2), (XS[3] - 0.5, YR + HR / 2))
ax.text(XS[2] + W / 2, YR - 1.4, "the test set stays untouched until this step",
        ha="center", va="top", fontsize=6.6, style="italic", color=MUTED)

fig.savefig(C.FIG_DIR / "pipeline.png", dpi=300, bbox_inches="tight", facecolor="white")
print("saved", C.FIG_DIR / "pipeline.png")
