> **Draft notes (delete this box before submitting)**
> - Fill in the author names, affiliation, email and course/conference name.
> - Every number comes from `reports/metrics.json`, `reports/audit.json` and `reports/splits.json`.
> - References marked **[verify]** exist (found by web search) but need their author initials, volume and pages checked against the publisher page before you submit.
> - Figures to insert are listed at the end, with their file names. Copy tables into Word or LaTeX as needed.
> - If your conference gives a template or page limit, send it to me and I will trim or reformat.

---

# Custom CNN versus Transfer Learning for Brain Tumor MRI Classification

**Author One, Author Two**
*Department of ____, University of ____*
*email@example.com*

---

## Abstract

Automated classification of brain MRI could help radiologists prioritize and double-check scans, but it is unclear whether a purpose-built network or a pretrained one is the better starting point for small medical datasets. We compare a custom convolutional neural network (CNN) trained from scratch with MobileNetV2 transfer learning for four-class classification (glioma, meningioma, pituitary tumor, no tumor) on a public brain MRI dataset. Before training, we audited the data and found that 41% of the images in the dataset's official test folder have an exact or near-identical copy in its training folder, so we re-split all images by duplicate cluster to prevent leakage. Both models were trained with identical data, augmentation and three random seeds, and evaluated once on a held-out test set of 765 images. Fine-tuned MobileNetV2 reached 97.7 ± 0.5% accuracy and 0.976 macro-F1, against 96.7 ± 0.9% and 0.965 for the custom CNN, and needed about one fifth of the training time (121 s versus 590 s). The accuracy difference is small and the bootstrap confidence intervals of the best runs overlap, so we regard the models as comparable in accuracy. A frozen pretrained backbone with only a new classification head reached just 87.6%, showing that fine-tuning is essential for MRI. Grad-CAM heatmaps show that correct predictions often concentrate on the tumor, but not always. We also built a small web application for inference and explanation. The system is a research prototype and not a diagnostic tool.

**Keywords:** brain tumor classification, MRI, convolutional neural network, transfer learning, MobileNetV2, Grad-CAM, data leakage.

---

## I. Introduction

Brain tumors are diagnosed mainly from magnetic resonance imaging (MRI), and the main tumor types differ in origin, behavior and treatment. Gliomas arise from glial cells, meningiomas from the membranes surrounding the brain, and pituitary tumors from the pituitary gland at the base of the brain. Reading scans requires scarce expert time, which motivates computer-aided tools that act as a second opinion or help prioritize cases.

Deep convolutional neural networks are the standard approach for this task. A practical design question arises for small student or clinic-scale projects: should one design and train a compact network from scratch, or reuse a network pretrained on a large natural-image dataset and adapt it (transfer learning) [1]? Pretrained models usually train faster and need less data, but MRI differs strongly from natural photographs, so the benefit is not guaranteed.

This paper makes three contributions:
1. A controlled comparison of a custom CNN and MobileNetV2 [2] on four-class brain MRI classification, with identical splits, preprocessing, augmentation, optimizer family and three seeds per model, including a head-only transfer-learning baseline.
2. An audit of the public dataset that quantifies duplicate leakage between its official training and testing folders (41% of test images have a twin), and a cluster-based, duplicate-aware split that removes it.
3. A fuller evaluation than accuracy alone: macro-F1, per-class recall with confidence intervals, calibration, training cost, and Grad-CAM explanations, plus an interactive demonstration application.

---

## II. Related Work

**Brain tumor classification from MRI.** Convolutional networks, from custom architectures to fine-tuned ImageNet models, are the dominant approach to classifying brain MRI. Studies on the three-class figshare data compared many pretrained CNNs and found transfer learning effective [11]. On the four-class dataset used here (7,023 images), Disci et al. fine-tuned six pretrained networks, including MobileNetV2, and reported a weighted accuracy of up to 98.73% with Xception [10]. Such results are normally measured on the dataset's own training and testing folders. We instead run one small, controlled comparison between a network trained from scratch and a pretrained one, under identical data, augmentation, seeds and a leakage-aware split.

**Dataset integrity.** Several recent works question whether high accuracy on public brain tumor benchmarks reflects generalization. Saifullah et al. remove duplicate and near-duplicate images with perceptual hashing and report that test accuracy drops slightly, while nested validation gives a more realistic estimate of generalization [12]. Chaudhary et al. build an automated de-duplication pipeline (file fingerprints and image similarity) and release a cleaned benchmark of 3,522 unique images, arguing that models can end up memorizing leaked images [13]. Vangala et al. audit several public brain tumor corpora at the image, patient and acquisition-source level. For the dominant corpus they study, they report that 28.8% of official test images have a near-twin in training and that most traceable test images share a patient with training images; they also find that removing the leaked test images leaves accuracy essentially unchanged, so stable accuracy after de-duplication does not by itself establish independence [14]. Sharma likewise partitions images by perceptual-hash cluster before splitting [15]. Our audit follows the same cluster-based idea on the four-class dataset and reports its own leakage rate, but we do not claim that it changes headline accuracy.

**Lightweight networks and explanation.** MobileNetV2 [2] is attractive for deployment because of its small size and low cost, and transfer learning [1] from ImageNet [4] is a common way to use it on small datasets. Grad-CAM [3] highlights the image regions that influenced a prediction and is widely used to inspect medical image models.

**Position of this work.** We do not propose a new architecture. Our contribution is a compact, reproducible comparison of a scratch CNN, head-only transfer learning and fine-tuned transfer learning, evaluated with macro-F1, per-class recall, calibration, training cost and Grad-CAM on a cluster-based leakage-free split.

---

## III. Dataset and Data Preparation

### A. Dataset

We used a public four-class brain MRI dataset (Kaggle "Brain Tumor MRI Dataset" [8], a merge of the figshare, SARTAJ and Br35H datasets; the uploader reports that the original SARTAJ glioma images were mislabeled and replaced them with figshare images) containing 7,023 JPEG images in a Training folder (5,712) and a Testing folder (1,311), labelled glioma, meningioma, pituitary tumor or no tumor. All files were readable and the folder labels matched the supplied metadata file. The images are heterogeneous: 3,926 are RGB, 3,093 grayscale, and 4 other modes, with 387 distinct resolutions (many 512×512), and they include axial, coronal and sagittal views with different crops.

### B. Duplicate audit

We computed two fingerprints for every image: a hash of the file bytes (exact copies) and a 64-bit perceptual hash (DCT of a 32×32 grayscale thumbnail), where two images are near-duplicates if their Hamming distance is at most 2. We chose this threshold from the data: pairs with different labels first appear at distance 2 (3 pairs) and grow quickly beyond it (33 at distance 4, 289 at distance 6), so 2 is the largest value that stays clear of unrelated scans. For matches across the official folders, pixel correlation of 64×64 thumbnails confirmed that the near-duplicates are re-encoded or resized copies of the same image (median correlation 0.999 for the "no tumor" matches).

The audit found 297 redundant exact copies (no conflicting labels), and **542 of the 1,311 official test images (41%) with an exact or near-identical twin in the training folder**, including 368 of the 405 "no tumor" test images. Evaluating on the official test folder would therefore measure performance largely on images the model has already effectively seen, and only 37 "no tumor" test images would remain if the twins were simply removed.

### C. Leakage-free split

We pooled all unique images, collapsed exact copies, and grouped images into duplicate clusters (connected components of exact and near-duplicate links). We then split by cluster, stratified by class, with a fixed seed (42): about 71% training, 14% validation and 14% test. In the validation and test sets only one image per (cluster, label) was kept, so repeated scans of one slice cannot inflate metrics (398 redundant images dropped). Automated checks assert that no cluster and no file hash appears in more than one split.

**TABLE I. Images per class after the leakage-free split**

| Class | Train | Validation | Test |
|---|---|---|---|
| Glioma | 1,160 | 220 | 217 |
| Meningioma | 1,160 | 208 | 208 |
| No tumor | 1,232 | 102 | 103 |
| Pituitary tumor | 1,245 | 236 | 237 |
| **Total** | **4,797** | **766** | **765** |

### D. Preprocessing and augmentation

All images are converted to 3-channel RGB, resized to 224×224 (bicubic) and normalized with ImageNet mean and standard deviation, identically for both models and for the application. Training images (only) are augmented on the GPU with random horizontal flips, rotations of ±15°, scaling of 0.9–1.1, translations of ±6%, contrast changes of ±15% and a small brightness shift. We checked the augmented images visually to confirm they stay anatomically plausible.

---

## IV. Methods

### A. Models

**Custom CNN (from scratch).** Five convolutional blocks with 32, 64, 128, 256 and 256 channels, each with two 3×3 convolutions, batch normalization [5] and ReLU. The first four blocks end in 2×2 max pooling; the last block keeps a 14×14 feature map. Global average pooling, dropout (0.4) and a linear layer give four outputs. Weights use He initialization [9]. The network has 2.35 M trainable parameters.

**MobileNetV2 (transfer learning).** The ImageNet-pretrained network [2], [4] with its classifier replaced by dropout (0.3) and a four-way linear layer (2.23 M parameters in total). Training has two stages: (1) *head only*: the backbone is frozen (5,124 trainable parameters) for up to 8 epochs; (2) *fine-tuning*: the last 6 of 19 feature blocks are unfrozen (1.80 M trainable parameters) with a ten times smaller learning rate. Batch normalization layers in frozen blocks keep their running statistics.

### B. Training

Both models use cross-entropy loss, the AdamW optimizer [6] with weight decay 1e-4, cosine learning-rate decay to 1% of its start value, batch size 32 and mixed-precision arithmetic. The custom CNN used learning rate 1e-3, a 40-epoch budget and early stopping patience 10. MobileNetV2 used 1e-3 (head, patience 5) and 1e-4 (fine-tuning, up to 25 epochs, patience 7). In every stage the checkpoint with the highest **validation macro-F1** is kept. Each configuration was trained with three seeds (42, 43, 44), which change initialization and data order but not the split. Hardware: NVIDIA RTX 3080 laptop GPU (8 GB), PyTorch 2.9. The test set was used only after training, in a separate evaluation script, and never for any decision. For each model family, the run exported for the application and used for the confusion matrices was chosen by validation macro-F1 only.

### C. Evaluation

We report accuracy, macro-averaged precision, recall and F1, per-class recall with Wilson 95% intervals, one-versus-rest macro ROC-AUC, expected calibration error (ECE, 15 bins) [7], a 95% bootstrap interval (2,000 resamples) for macro-F1, wall-clock training time (including validation) and CPU inference latency for one image. Results are means ± standard deviations over the three seeds. Grad-CAM [3] on the last convolutional feature map is used for qualitative explanation.

---

## V. Results

**TABLE II. Test-set results (765 images), mean ± std over 3 seeds**

| Metric | Custom CNN (scratch) | MobileNetV2 (head only) | MobileNetV2 (fine-tuned) |
|---|---|---|---|
| Accuracy | 0.967 ± 0.009 | 0.876 ± 0.005 | **0.977 ± 0.005** |
| Macro precision | 0.967 ± 0.008 | 0.885 ± 0.000 | **0.977 ± 0.004** |
| Macro recall | 0.965 ± 0.012 | 0.877 ± 0.007 | **0.975 ± 0.006** |
| Macro-F1 | 0.965 ± 0.010 | 0.878 ± 0.005 | **0.976 ± 0.005** |
| ROC-AUC (macro OvR) | 0.999 | 0.978 | 0.999 |
| ECE | 0.014 | 0.045 | 0.014 |
| Training time (s) | 590 | 33 | 121 |
| Trainable parameters | 2,354,916 | 5,124 | 1,804,740 |
| CPU latency (ms/image) | 24.3 | 15.7 | 15.7 |

Training time of the fine-tuned model includes the head-only stage.

**TABLE III. Per-class recall (mean ± std over 3 seeds)**

| Class | Custom CNN | MobileNetV2 head only | MobileNetV2 fine-tuned |
|---|---|---|---|
| Glioma | 0.971 ± 0.005 | 0.814 ± 0.032 | 0.975 ± 0.005 |
| Meningioma | 0.929 ± 0.026 | 0.793 ± 0.042 | 0.957 ± 0.013 |
| No tumor | 0.958 ± 0.031 | 0.913 ± 0.026 | 0.968 ± 0.022 |
| Pituitary tumor | 1.000 ± 0.000 | 0.989 ± 0.002 | 0.999 ± 0.002 |

**TABLE IV. Per-seed test accuracy / macro-F1**

| Seed | Custom CNN | MobileNetV2 fine-tuned |
|---|---|---|
| 42 | 0.975 / 0.974 | 0.974 / 0.973 |
| 43 | 0.957 / 0.954 | 0.974 / 0.973 |
| 44 | 0.969 / 0.968 | 0.982 / 0.981 |

**Confusion matrices of the validation-selected runs** (rows: true class; columns: predicted; order glioma, meningioma, no tumor, pituitary):

| Custom CNN (seed 42) | MobileNetV2 fine-tuned (seed 43) |
|---|---|
| 210, 3, 0, 4 | 211, 3, 0, 3 |
| 2, 199, 4, 3 | 2, 196, 4, 6 |
| 0, 2, 100, 1 | 1, 1, 101, 0 |
| 0, 0, 0, 237 | 0, 0, 0, 237 |

The bootstrap 95% interval of macro-F1 is [0.961, 0.984] for both selected runs.

**Findings.**
1. **Both end-to-end trained models are accurate** (above 96% on average) on the leakage-free test set.
2. **Fine-tuned MobileNetV2 is slightly more accurate** (+1.0 point accuracy, +1.0 point macro-F1) with a smaller spread across seeds, but the intervals of the two selected runs overlap and one CNN seed (97.5%) matches the MobileNetV2 runs, so the difference is small and not conclusive.
3. **Fine-tuned MobileNetV2 was about five times cheaper to train** (121 s versus 590 s). The CNN used its entire 40-epoch budget with its best epochs near the end (epochs 35–36), so it was still improving and might gain from longer training; the time gap partly reflects this budget.
4. **A frozen pretrained backbone is not enough:** the head-only model reached 87.6%, with meningioma and glioma recall near 0.8, so ImageNet features alone transfer poorly to MRI.
5. **Meningioma is the hardest class** for both models (recall 0.93 and 0.96). Pituitary tumors are almost always identified (recall about 1.0).
6. **Calibration is good for both models** (ECE 0.014): reported confidence tracks actual accuracy.
7. **Inference cost is small:** both models classify an image in tens of milliseconds on a CPU; MobileNetV2 is faster (15.7 versus 24.3 ms). GPU single-image timings were dominated by launch overhead and are not reported.

**Grad-CAM.** Qualitatively, correct predictions often place the heat on the tumor (for example the bright mass in meningioma and pituitary cases), but some maps are diffuse or sit on the skull border, indicating that the models sometimes rely on weaker cues. MobileNetV2's maps are coarser (7×7 feature map) than the CNN's (14×14). Several confident "errors" are scans labelled "no tumor" that visibly contain a bright lesion, which suggests some label noise in the source data. These observations are qualitative and were not quantified.

---

## VI. Discussion

For this dataset, a small custom CNN trained from scratch achieves accuracy close to a fine-tuned pretrained network, but fine-tuning MobileNetV2 reaches it faster and slightly higher, and with fewer trainable parameters in the first stage. Pretraining therefore provides efficiency more than a large accuracy gain. The poor head-only result shows that the benefit depends on adapting the pretrained features to the new domain.

The main practical lesson is about evaluation. With the dataset's official folders, a large share of the test images are ones the model has effectively already seen, and the "no tumor" class could hardly be assessed. We did not train under the official protocol, so we cannot say how much it would change the headline accuracy; an independent audit of public brain tumor data reports that removing leaked test images can leave accuracy nearly unchanged [14]. What the cluster split does guarantee is that our test score is measured on images with no copy in training. Duplicate-aware splitting is cheap and should be standard for public medical image sets. Our scores are not directly comparable to results reported on the official split.

For clinical relevance, per-class recall matters more than overall accuracy, because a missed tumor is the costly error. On average our models miss about 3% of glioma and 4 to 7% of meningioma cases, which is too high for unsupervised use.

---

## VII. Limitations

- The dataset has no patient identifiers, so slices from one patient may fall into different splits. Our splitting removes duplicates, not patient overlap; audits of similar benchmarks report that most test images share a patient with training images [14], so our test scores are probably optimistic.
- Images come from several sources, scanners, views and crops. Models may partly learn source cues rather than tumor appearance.
- Only one public dataset was used, with no external or clinical validation. The "no tumor" test class is small (103 images).
- Hyperparameters were fixed from standard defaults rather than searched. The CNN's training budget may be limiting, as noted above.
- The near-duplicate threshold is a heuristic. Grad-CAM is a qualitative attention map, not a tumor segmentation, and suspected label noise was not verified by an expert.
- The system is a research prototype and **must not be used for diagnosis or treatment decisions**.

---

## VIII. Conclusion

We compared a custom CNN with MobileNetV2 transfer learning for four-class brain tumor MRI classification on a duplicate-audited, leakage-free split. Fine-tuned MobileNetV2 (97.7% accuracy, 0.976 macro-F1) was slightly better and about five times faster to train than the scratch CNN (96.7%, 0.965), though the difference is within run-to-run uncertainty, while a frozen backbone (87.6%) was clearly worse. Duplicate leakage in the official split (41% of test images) shows that data auditing is a necessary first step. Future work includes patient-level splitting, external validation, 3D or multi-slice input, longer or tuned training, quantitative evaluation of explanations, and review of questionable labels by clinicians.

---

## References

*(Entries marked [verify] were confirmed to exist from web sources, but check author initials, volume and page details against the publisher page before submitting.)*

1. S. J. Pan and Q. Yang, "A survey on transfer learning," *IEEE Trans. Knowl. Data Eng.*, vol. 22, no. 10, pp. 1345–1359, 2010.
2. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 4510–4520.
3. R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 618–626.
4. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A large-scale hierarchical image database," in *Proc. IEEE CVPR*, 2009, pp. 248–255.
5. S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. Int. Conf. Machine Learning (ICML)*, 2015, pp. 448–456.
6. I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2019.
7. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. ICML*, 2017, pp. 1321–1330.
8. M. Nickparvar, "Brain Tumor MRI Dataset," Kaggle. [Online]. Available: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset **[verify year and access date; the source description (figshare, SARTAJ, Br35H) was taken from a mirror of the Kaggle listing]**
9. K. He, X. Zhang, S. Ren, and J. Sun, "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification," in *Proc. IEEE ICCV*, 2015, pp. 1026–1034.
10. R. Disci, F. Gurcan, and A. Soylu, "Advanced brain tumor classification in MR images using transfer learning and pre-trained deep CNN models," *Cancers*, vol. 17, no. 1, art. 121, 2025, doi: 10.3390/cancers17010121. **[verify author initials]**
11. R. Chelghoum, A. Ikhlef, A. Hameurlaine, and S. Jacquir, "Transfer learning using convolutional neural network architectures for brain tumor classification from MRI images," in *Artificial Intelligence Applications and Innovations (AIAI 2020), IFIP Advances in Information and Communication Technology*, Springer, 2020. **[verify authors, volume and pages]**
12. Saifullah, S. Baldauf, and K. Sakumura, "Reliable brain tumor classification without metadata: A step-by-step guideline with duplicate removal," in *Proc. IEEE 37th Int. Conf. Tools with Artificial Intelligence (ICTAI)*, 2025, pp. 1237–1244, doi: 10.1109/ICTAI66417.2025.00180. **[verify author initials]**
13. D. Chaudhary, D. Lu, and Y. Zhang, "MTA-Swin: A multi-token attention Swin transformer for brain tumor classification with leakage-free MRI benchmarking," 2026, doi: 10.1007/s10916-026-02418-w. **[verify author list, journal name, volume and pages; only a news summary of this paper could be read]**
14. B. P. Vangala, S. Guda, L. Peddi, and N. Vangala, "Scores that hold, benchmarks that leak: Measuring dataset contamination in public brain-tumor MRI classification," arXiv:2610.00421, Sep. 2026. **[preprint; check whether it has been published]**
15. M. Sharma, "Monte Carlo dropout uncertainty and entropy-thresholded selective prediction for architecture-agnostic brain tumor MRI triage," arXiv:2607.16317, Jul. 2026. **[preprint]**

---

## Figures to insert (all in `reports/figures/`)

| Figure | File | Suggested caption |
|---|---|---|
| Fig. 1 | `pipeline.png` | Overview of the pipeline. |
| Fig. 2 | `compare_macro_f1.png` | Test macro-F1 of the three models (bars: mean, whiskers: std, dots: seeds). |
| Fig. 3 | `compare_per_class_recall.png` | Per-class recall of the three models. |
| Fig. 4 | `confusion_cnn.png`, `confusion_mobilenet-ft.png` | Confusion matrices of the selected runs. |
| Fig. 5 | `curves_cnn.png`, `curves_mobilenet.png` | Training and validation curves. |
| Fig. 6 | `gradcam_cnn.png`, `gradcam_mobilenet.png` | Grad-CAM examples, correct and incorrect. |
| Fig. 7 | *(screenshot of the Streamlit app)* | Web application: prediction, probabilities and heatmap. |

