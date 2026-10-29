# BrainO AI: Project Guide (plain-English)

A guide to understand the project and explain it confidently, even with zero healthcare background.
All numbers below come from real runs saved in `reports/metrics.json`.

---

## 1. The project in 30 seconds

> "We built an AI that looks at a brain MRI image and says which of four categories it belongs to:
> **glioma, meningioma, pituitary tumor, or no tumor**. We trained two different models, compared them
> fairly, and built a web app where you upload a scan and see the prediction, the confidence, and a
> heatmap of where the AI looked. The best model got about **97.7% accuracy** on images it had never seen.
> It is a student research prototype, **not** a medical tool."

If you remember only one paragraph, remember that one.

---

## 2. Healthcare basics you need (the only ones)

**Brain tumor.** An abnormal lump of cells growing in or around the brain. Different types behave
differently and are treated differently, so telling them apart matters.

**MRI (Magnetic Resonance Imaging).** A scan that takes detailed pictures of the inside of the body, using
magnets, without radiation. A brain MRI is a stack of image "slices", like slicing a loaf of bread. Our
dataset has individual slices as ordinary picture files (JPG).

**The four classes** (say it like this):

| Class | What it is, in simple words |
|---|---|
| **Glioma** | A tumor that starts in the brain's supporting cells (glial cells). Can be serious. |
| **Meningioma** | A tumor in the membranes that wrap around the brain. Often slow-growing. |
| **Pituitary tumor** | A tumor on the pituitary gland, a small gland at the base of the brain that controls hormones. |
| **No tumor** | A normal brain scan. |

You are not expected to be a doctor. You only need to say: *"these are four categories a radiologist
(the doctor who reads scans) would distinguish, and we taught a computer to do the same."*

**Why use AI here at all?** Reading scans takes expert time and experts are limited. An AI could act as a
**second opinion** or help **prioritise** which scans a doctor looks at first. It would *support* the
doctor, never replace them. Always say this.

---

## 3. How the AI works (no maths)

1. **Training data.** We show the computer thousands of scans, each with the correct answer attached.
2. **Learning.** The model is a **neural network**, a program with millions of adjustable numbers. It guesses,
   checks the right answer, and nudges its numbers to be less wrong. Repeat thousands of times.
3. **Testing.** We then give it scans it has **never seen** and measure how often it is right. That is the
   honest measure of quality.

Analogy: it is like a medical student studying a big book of labelled scans, then sitting an exam on new ones.

### The two models we compared

| | **Custom CNN** | **MobileNetV2** |
|---|---|---|
| Idea | We designed it and it learned **from zero** (random start) | Someone already trained it on 1 million ordinary photos (ImageNet). We **reuse** that knowledge and adapt it to MRI |
| Technique name | Training from scratch | **Transfer learning** |
| Analogy | A student who has never seen any image before | A student who already knows how to see edges and shapes, now learning radiology |

(CNN = Convolutional Neural Network, the standard type of network for images.)

MobileNetV2 was trained in two steps:
1. **Head only:** keep the old knowledge frozen, train only the final decision layer.
2. **Fine-tune:** also let the last few layers adjust gently to MRI.

---

## 4. The most important thing we found: the dataset was leaking

This is the strongest part of your project. Understand it well.

**What is leakage?** Imagine an exam that reuses the *exact* questions from the practice sheet. A student
would score high without really understanding. In AI this makes accuracy look better than it truly is.

**What we found** (by checking image fingerprints, with pixel-level verification):
- The dataset comes with a ready-made Training folder and Testing folder.
- **41% of the Testing images (542 of 1,311) are copies, or near-copies, of Training images.**
  For "No tumor" it was 368 of 405.
- Using that split would give inflated, misleading accuracy.

**What we did:**
- Removed exact duplicate files.
- Grouped each image with its near-copies, then split the data **by group**, so copies never appear on both
  the training and test side.
- Final split: **4,797 train / 766 validation / 765 test**. The program checks automatically that nothing
  is shared between them.

Say this in your presentation: *"We did not blindly trust the dataset. We audited it, found leakage, and
fixed it, so our results are honest."* Examiners love this.

**Validation vs test** (a common question):
- *Training set*: what the model learns from.
- *Validation set*: used while building, to choose settings and pick the best version.
- *Test set*: locked away, used **once at the end** to report the final score. We never used it to make decisions.

---

## 5. Results explained

Measured on 765 unseen test images, average of 3 training runs ("seeds"):

| Model | Accuracy | Macro-F1 | Training time |
|---|---|---|---|
| Custom CNN (from scratch) | 96.7% | 0.965 | ~590 s |
| MobileNetV2, head only | 87.6% | 0.878 | ~33 s |
| **MobileNetV2, fine-tuned** | **97.7%** | **0.976** | ~121 s |

**Plain meanings**
- **Accuracy:** of all scans, the share predicted correctly. 97.7% means roughly 2 mistakes in 100.
- **Recall (also called sensitivity):** of the scans that truly *are* class X, how many did we catch? For
  tumors this is the key number: a **missed tumor** (false negative) is far worse than a false alarm.
- **Precision:** when the model says "X", how often is it right?
- **F1-score:** one number balancing precision and recall. **Macro** means we average over the four classes
  equally, so a rare class cannot hide behind a big one.
- **Confusion matrix:** a table showing what the model predicted versus the truth, so you can see *which*
  classes get mixed up.
- **Seeds:** a different random start. Running 3 shows the result is not a lucky one-off (the ± in our table).

**What the results tell us**
1. **Both models work well** (97% range) on leak-free data.
2. **Transfer learning needs fine-tuning.** The frozen version only got 87.6%, because MRI scans look very
   different from everyday photos. Adapting the model fixed that.
3. **MobileNetV2 vs our CNN:** MobileNetV2 is about 1 point better and trains about 5 times faster, but the
   difference is small and the uncertainty ranges overlap. Honest wording: *"similar accuracy; the
   pretrained model reached it much faster."* Do **not** say "MobileNet is definitely better".
4. **Weakest class is meningioma** (recall around 93% for the CNN and 96% for MobileNetV2). **Pituitary** is
   almost perfect (about 100%).
5. **Confidence is trustworthy:** the models' stated confidence matches how often they are right
   (calibration error about 0.014, which is low).

---

## 6. Grad-CAM, the "where did it look?" heatmap

Hospitals do not trust a black box that just says "tumor". **Grad-CAM** paints a heat overlay on the scan:
red/warm areas are the regions that most influenced the decision.

- In many correct predictions, the hot spot sits right on the tumor. 
- In some cases it is spread out or on the skull edge. That tells us the model sometimes uses weaker cues.
- Some "wrong" predictions were "No tumor" scans that visibly show a bright lesion, so the original dataset
  label may itself be wrong.

Be careful with wording: it shows **where the AI paid attention**, not a verified tumor outline.

---

## 7. The web app: what to demo

Run: `streamlit run app/app.py`

Pages:
1. **Classify MRI:** upload a scan → predicted class, confidence, probability bars for all four classes,
   and a Grad-CAM heatmap. You can switch between the two models in the sidebar.
2. **Model comparison:** tables and charts of the results above.
3. **About and limitations:** the data audit and the honesty section.

Safety features built in: rejects non-images, tiny files and files over 15 MB; warns if an image is
brightly coloured (probably not an MRI); warns when confidence is below 60%.

**Demo tips:** pick 2 or 3 test images beforehand (see `reports/predictions/cnn.csv` for the file names).
Show one correct prediction with its heatmap, then one low-confidence or wrong one, and say "this is why a
doctor must always check". Upload a normal photo to show it warns instead of pretending.

---

## 8. Limitations (say them before the examiner does)

- **No patient IDs in the data.** Two slices from the same patient may be in both train and test, so real
  accuracy on new patients is probably a bit lower than ours.
- **Mixed sources.** Images come from different scanners and viewing angles; the AI may partly learn
  those differences instead of tumors.
- **One public dataset only.** No testing on real hospital data, no clinical validation.
- **Possible wrong labels** in the original data.
- **Not a medical device.** It must never be used to diagnose or rule out anything.

Admitting limits makes the project look more credible, not weaker.

---

## 9. A 5-minute presentation outline

1. **Problem (30 s):** Brain tumors need fast, accurate scan reading; experts are scarce; AI can assist.
2. **Data (45 s):** 7,000 MRI images, 4 classes. *We audited it and found 41% of the test images were
   copies of training images, so we fixed the split.*
3. **Models (60 s):** Custom CNN from scratch vs MobileNetV2 transfer learning (head, then fine-tune).
4. **Results (60 s):** Show the table and confusion matrix. Best ≈ 97.7%. Fine-tuning mattered a lot.
5. **Explainability (45 s):** Show Grad-CAM. Good cases and one weak case.
6. **Demo (60 s):** Live upload.
7. **Limitations and ethics (30 s):** Not for diagnosis; no patient IDs; single dataset; needs clinical validation.

---

## 10. Questions you may be asked, with answers

**Why did you split the data yourself?**
The official test folder shared many near-identical images with the training folder (41%), which inflates
the score. We regrouped so copies never cross sides.

**Why is accuracy not enough?**
In medicine, missing a tumor is worse than a false alarm. Recall per class and the confusion matrix show
*what kind* of mistakes happen. Macro-F1 stops big classes hiding weak ones.

**What is overfitting, and did you handle it?**
Memorising training images instead of learning general patterns. We guard against it with a separate
validation set, data augmentation (small rotations, flips, brightness changes), dropout, weight decay and
early stopping, and we judge only on unseen test data.

**What is transfer learning?**
Reusing a network trained on a big, general task (ImageNet photos) as the starting point for a new task
(MRI). It usually needs less data and time.

**Why does the pretrained model without fine-tuning do poorly (87.6%)?**
Its features were learned from everyday photos, not MRI. It needs to adapt its later layers.

**Which model is better?**
MobileNetV2 fine-tuned scored slightly higher (97.7% vs 96.7%) and trained faster, but the gap is within
the uncertainty, so we call them comparable.

**How do you know the result isn't luck?**
We trained each model 3 times with different random seeds, and report the mean and spread. Chosen models
were selected using validation data only.

**Can this be used in hospitals?**
No. It was tested on one public dataset and has no clinical validation. It is a learning prototype.
Real use would need approval, testing on local hospital data and patient-level validation.

**Why 224×224 images, and what preprocessing?**
Both models expect that size. Every image is converted to 3-colour-channel RGB, resized to 224×224 and
normalised, identically in training and in the app.

**What does Grad-CAM prove?**
Only where the model looked. It helps spot suspicious behaviour but is not a tumor outline.

**What would you do next?**
Get patient IDs and split by patient; test on data from another hospital; try 3D scans; combine models;
have a radiologist review the heatmaps and the questionable labels.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| Radiologist | Doctor who reads medical scans |
| MRI | Scan producing detailed images of the body's insides |
| Neural network / CNN | Program with adjustable numbers that learns patterns from images |
| Training / validation / test | Learn / tune and pick best / final exam |
| Epoch | One full pass over the training images |
| Overfitting | Memorising instead of learning |
| Augmentation | Randomly tweaking training images (rotate, flip, brighten) to learn more robustly |
| Transfer learning | Starting from a model pre-trained on another task |
| Fine-tuning | Gently adjusting a pre-trained model to the new task |
| Recall / sensitivity | Share of real cases that were caught |
| Precision | Share of positive calls that were correct |
| F1 | Balance of precision and recall |
| Confusion matrix | Table of predicted vs true classes |
| Grad-CAM | Heatmap of where the model looked |
| Data leakage | Test data secretly overlapping with training data |
| Calibration | Whether "90% confident" really means right 90% of the time |

---

## 12. Where things are

| You want | Open |
|---|---|
| All final numbers | `reports/metrics.json` |
| Leakage audit numbers | `reports/audit.json`, `reports/splits.json` |
| Charts for slides | `reports/figures/` (confusion matrices, curves, comparison, Grad-CAM) |
| Run the demo | `streamlit run app/app.py` |
| Re-run everything | steps listed in `README.md` |
