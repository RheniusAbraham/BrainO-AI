"""BrainO AI: Streamlit interface.   Run:  streamlit run app/app.py"""
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.explain import GradCAM, overlay  # noqa: E402
from src.inference import load_model, pil_to_tensor, predict_probs  # noqa: E402

MODELS = {"cnn": "Custom CNN (trained from scratch)", "mobilenet": "MobileNetV2 (transfer learning)"}
MAX_BYTES = 15 * 1024 * 1024
LOW_CONF = 0.60

st.set_page_config(page_title="BrainO AI", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.kicker {font-size: .75rem; letter-spacing: .14em; text-transform: uppercase; color: #8a8f98;}
.result {border: 1px solid rgba(128,128,128,.3); border-radius: 12px; padding: 1.1rem 1.3rem;}
.result .label {font-size: 1.9rem; font-weight: 700; line-height: 1.2;}
.result .sub {color: #8a8f98; font-size: .9rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_model(name: str):
    model, meta = load_model(C.EXPORT_DIR / f"{name}.pt", "cpu")
    return model, GradCAM(model), meta


def show_figure(container, path: Path) -> None:
    """Show a saved figure. Loads the pixels ourselves: Streamlit 1.28 cannot serve file paths
    that contain spaces (this project's folder does)."""
    with Image.open(path) as im:
        container.image(im.copy(), use_column_width=True)


def available_models():
    return {k: v for k, v in MODELS.items() if (C.EXPORT_DIR / f"{k}.pt").exists()}


def read_upload(file) -> Image.Image:
    """Validate and decode an upload. Raises ValueError with a user-facing message."""
    raw = file.getvalue()
    if len(raw) > MAX_BYTES:
        raise ValueError("File is larger than 15 MB.")
    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise ValueError("This file could not be read as an image. Upload a PNG or JPEG.")
    img = ImageOps.exif_transpose(img)
    if min(img.size) < 64:
        raise ValueError(f"Image is too small ({img.size[0]}x{img.size[1]} px); need at least 64 px per side.")
    return img


def looks_colourful(img: Image.Image) -> bool:
    """MRI slices are essentially grayscale; strongly coloured pictures are probably not MRI."""
    hsv = np.asarray(img.convert("RGB").resize((96, 96)).convert("HSV"), dtype=np.float32)
    return float(hsv[..., 1].mean()) / 255 > 0.18


# ------------------------------------------------------------------ pages
def page_classify():
    st.markdown('<div class="kicker">Research prototype</div>', unsafe_allow_html=True)
    st.title("Brain MRI classification")
    st.caption("Upload an MRI slice. The model predicts one of four classes: glioma, meningioma, "
               "pituitary tumor, or no tumor.")
    models = available_models()
    if not models:
        st.error("No trained model found in models/exported/. Run src/train.py then src/evaluate.py first.")
        return
    with st.sidebar:
        choice = st.radio("Model", list(models.values()))
        name = next(k for k, v in models.items() if v == choice)
        show_cam = st.checkbox("Show Grad-CAM heatmap", value=True)
    file = st.file_uploader("Upload a brain MRI (PNG or JPEG)", type=["png", "jpg", "jpeg"])
    if file is None:
        st.info("Waiting for an image.")
        return
    try:
        img = read_upload(file)
    except ValueError as e:
        st.error(str(e))
        return

    model, cam_fn, meta = get_model(name)
    x = pil_to_tensor(img)
    probs = predict_probs(model, x)[0]
    top = int(probs.argmax())

    if looks_colourful(img):
        st.warning("This image is strongly coloured. MRI slices are normally grayscale, so the input may "
                   "not be an MRI. The prediction below may be meaningless.")

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.image(img.convert("RGB"), caption=f"Uploaded image ({img.size[0]}x{img.size[1]} px)",
                 use_column_width=True)
    with right:
        st.markdown(
            f'<div class="result"><div class="sub">Predicted class</div>'
            f'<div class="label">{C.CLASS_DISPLAY[C.CLASSES[top]]}</div>'
            f'<div class="sub">Confidence {probs[top]:.1%} &middot; {MODELS[name]}</div></div>',
            unsafe_allow_html=True)
        if probs[top] < LOW_CONF:
            st.warning(f"Low confidence (< {LOW_CONF:.0%}): the model is unsure about this image.")
        st.markdown("**Class probabilities**")
        for i in np.argsort(-probs):
            st.progress(float(probs[i]), text=f"{C.CLASS_DISPLAY[C.CLASSES[i]]}: {probs[i]:.1%}")

    if show_cam:
        st.subheader("Grad-CAM: where the model looked")
        names = [C.CLASS_DISPLAY[c] for c in C.CLASSES]
        target = names.index(st.selectbox("Explain class", names, index=top))
        cam, _, _ = cam_fn(x, target)
        base = x[0].permute(1, 2, 0).numpy()
        c1, c2 = st.columns(2)
        c1.image(base, caption="Model input (224x224)", use_column_width=True)
        c2.image(overlay(base, cam), caption=f"Grad-CAM for {C.CLASS_DISPLAY[C.CLASSES[target]]}",
                 use_column_width=True)
        st.caption("Warm colours mark regions that most increased the score for the chosen class. A heatmap "
                   "shows model attention, not a verified tumor outline.")
    st.divider()
    st.caption("Educational research prototype trained on a public dataset. It is not a medical device and "
               "must not be used for diagnosis or treatment decisions.")


def page_compare():
    st.title("Model comparison")
    mp = C.REPORTS_DIR / "metrics.json"
    if not mp.exists():
        st.info("Run src/evaluate.py to generate reports/metrics.json.")
        return
    m = json.loads(mp.read_text())
    n = m["test_set"]
    st.caption(f"Held-out test set: {n['n']} images "
               f"({', '.join(f'{C.CLASS_DISPLAY[c]} {k}' for c, k in n['per_class'].items())}). "
               f"{m['selection_rule'].capitalize()}.")
    rows = []
    for g in m["groups"].values():
        mean, std = g["mean"], g["std"]
        sel = g["detail"][g["selected_by_val"]]
        rows.append({
            "Model": g["display"], "Seeds": len(g["runs"]),
            "Accuracy": f"{mean['accuracy']:.3f} ± {std['accuracy']:.3f}",
            "Macro-F1": f"{mean['macro_f1']:.3f} ± {std['macro_f1']:.3f}",
            "Macro recall": f"{mean['macro_recall']:.3f} ± {std['macro_recall']:.3f}",
            "AUC (OvR)": f"{mean['auc_ovr_macro']:.3f}",
            "Train time (s)": f"{mean['train_seconds']:.0f}",
            "Trainable params": f"{sel['params_trainable']:,}",
            "CPU ms/img": f"{mean['latency_ms_cpu']:.1f}",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("± is the standard deviation over training seeds (different initialisation and data order, "
               "same split).")

    t1, t2, t3, t4 = st.tabs(["Summary charts", "Per-class recall", "Confusion matrices", "Training curves"])
    fig = C.FIG_DIR
    with t1:
        a, b = st.columns(2)
        for col, f in ((a, "compare_macro_f1.png"), (b, "compare_accuracy.png"),
                       (a, "compare_train_time.png"), (b, "calibration.png")):
            if (fig / f).exists():
                show_figure(col, fig / f)
    with t2:
        if (fig / "compare_per_class_recall.png").exists():
            show_figure(st, fig / "compare_per_class_recall.png")
        st.caption("Recall is the share of true cases of a class the model finds. Missing a tumor "
                   "(predicting 'No tumor') is the most costly error in a screening setting.")
    with t3:
        cols = st.columns(len(m["groups"]))
        for col, g in zip(cols, m["groups"]):
            if (fig / f"confusion_{g}.png").exists():
                show_figure(col, fig / f"confusion_{g}.png")
    with t4:
        for f in ("curves_cnn.png", "curves_mobilenet.png"):
            if (fig / f).exists():
                show_figure(st, fig / f)
    for name in ("cnn", "mobilenet"):
        if (fig / f"gradcam_{name}.png").exists():
            with st.expander(f"Grad-CAM gallery: {MODELS[name]}"):
                show_figure(st, fig / f"gradcam_{name}.png")


def page_about():
    st.title("About and limitations")
    s = C.REPORTS_DIR / "splits.json"
    st.markdown("""
**Data.** Public 4-class brain MRI dataset (glioma, meningioma, pituitary tumor, no tumor), pooled from
several sources.

**Models.** A custom CNN trained from scratch, and MobileNetV2 initialised from ImageNet weights, first with
only the new head trained, then fine-tuned. Both use the same splits, input size, augmentation and seeds.
""")
    if s.exists():
        d = json.loads(s.read_text())
        o = d["official_split_contamination"]
        st.markdown(f"""
**Data leakage audit.** {o['test_images_with_twin_in_train']} of the {o['test_images']} images in the dataset's
official *Testing* folder have an exact or near-identical copy in *Training*
({o['per_class_test_with_twin']['notumor']} of {o['per_class_test']['notumor']} "No tumor" images).
Testing on them would overstate accuracy, so all images were pooled and re-split into
{d['final']['train']} train / {d['final']['val']} validation / {d['final']['test']} test images, by duplicate cluster,
so no scan and none of its near-copies appears on both sides.
""")
    st.markdown("""
**Limitations**
- The dataset has no patient identifiers. Different slices from the same patient can still fall on both sides of
  the split, so test scores are probably optimistic.
- Images come from several sources, with different scanners, MRI views and crops. A model can partly learn
  source-specific cues instead of tumor appearance.
- Results are from one public dataset and have not been validated on any hospital data.
- Grad-CAM shows where the network attended, not a verified tumor segmentation.
- **This is a student research prototype, not a medical device. It must never be used to diagnose, treat or
  rule out disease.**
""")


PAGES = {"Classify MRI": page_classify, "Model comparison": page_compare, "About and limitations": page_about}
with st.sidebar:
    st.markdown("### BrainO AI")
    page = st.radio("Page", list(PAGES), label_visibility="collapsed")
PAGES[page]()
