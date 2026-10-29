# BrainO AI: Brain tumor MRI classification

Research prototype for the *AI in Healthcare* subject. It classifies a brain MRI slice as
**glioma, meningioma, pituitary tumor or no tumor** and compares two approaches:

| Model | Idea |
|---|---|
| Custom CNN | 5 conv blocks, 2.35 M parameters, trained **from scratch** |
| MobileNetV2 | ImageNet-pretrained; stage 1 trains only the new head, stage 2 fine-tunes the last 6 blocks |

A Streamlit app (`app/app.py`) does upload, prediction, class probabilities and Grad-CAM, and shows the
model comparison. **Not a medical device. Never use it for diagnosis.**

## Results (held-out test set, 765 images, 3 seeds each, mean ± std)

| Model | Accuracy | Macro-F1 | Macro recall | AUC | Train time |
|---|---|---|---|---|---|
| Custom CNN (scratch) | 0.967 ± 0.009 | 0.965 ± 0.010 | 0.965 | 0.999 | 590 s |
| MobileNetV2, head only | 0.876 ± 0.005 | 0.878 ± 0.005 | 0.877 | 0.978 | 33 s |
| MobileNetV2, fine-tuned | **0.977 ± 0.005** | **0.976 ± 0.005** | 0.975 | 0.999 | 121 s |

Per-class recall (mean over seeds): glioma 0.97 / 0.81 / 0.98, meningioma 0.93 / 0.79 / 0.96,
no tumor 0.96 / 0.91 / 0.97, pituitary 1.00 / 0.99 / 1.00 (CNN / head only / fine-tuned).

How to read this honestly
- Fine-tuned MobileNetV2 beats the scratch CNN by about 1 point, and trains about 5x faster. The
  bootstrap 95% intervals of the two selected models overlap (macro-F1 0.961 to 0.984 for both), and the
  CNN's seed-to-seed spread (0.954 to 0.974) reaches the MobileNet range, so this is a **small, not conclusive** gap.
- A frozen ImageNet backbone alone is clearly not enough (0.878): MRI differs a lot from natural photos.
  Fine-tuning is what makes transfer learning work here.
- Meningioma has the weakest recall for both models. Missing a tumor is the costly error in practice.
- Exported models were chosen by validation macro-F1 only: `cnn_s42`, `mobilenet-ft_s43`.
- Full numbers, confusion matrices and curves: `reports/metrics.json` and `reports/figures/`.

## Data audit: why we did not use the official split

`python src/audit.py` found no corrupt files and no label/CSV mismatches, but:
- 297 exact duplicate files (no conflicting labels).
- **542 of the 1,311 official Testing images (41%) have an exact or near-identical copy in Training**
  (perceptual hash distance <= 2, checked against pixel correlation; 368 of 405 "No tumor" test images). Using the
  official test set would inflate accuracy.

`python src/make_splits.py` therefore pools all unique images and splits **by duplicate cluster**,
stratified by class, seed 42: 4,797 train / 766 validation / 765 test. It asserts that no cluster or
file hash is shared between splits. Accuracy numbers here are therefore **not comparable** with
published results on the official split.

## Limitations
- No patient IDs exist, so slices of one patient can still land on both sides of the split. Test scores
  are probably optimistic.
- The dataset mixes sources, scanners, MRI views and crops. Models may partly learn source cues.
  Grad-CAM (`reports/figures/gradcam_*.png`) shows correct predictions often focus on the tumor, but several
  heatmaps are diffuse or on the skull edge, and some confident "errors" are "No tumor" scans showing a bright
  lesion, which hints at label noise in the source data.
- One public dataset, no external or clinical validation. Grad-CAM is attention, not segmentation.

## Run it

```
pip install -r requirements.txt          # needs a CUDA build of PyTorch for GPU training
python src/audit.py                      # integrity + duplicate audit
python src/make_splits.py                # leak-free splits
bash run_all.sh                          # trains both models, seeds 42/43/44 (~45 min on RTX 3080 laptop)
python src/evaluate.py                   # test metrics, figures, exports models/exported/*.pt
python src/explain.py --gallery          # Grad-CAM galleries
streamlit run app/app.py
```

The dataset stays in `Dataset/` (not tracked by Git). Training uses train and validation only;
`src/evaluate.py` is the only script that reads the test fold.

## Layout
```
src/   config, audit, make_splits, dataset, model, train, evaluate, explain, inference
app/   app.py (Streamlit)        reports/  audit.json, splits.json, metrics.json, figures/, history/
models/checkpoints/ (all runs)   models/exported/ (cnn.pt, mobilenet.pt)
```
