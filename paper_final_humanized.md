\documentclass[conference]{IEEEtran}

\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{url}

% Red markers for things you must fix. Delete these two lines when done.
\newcommand{\todo}[1]{\textcolor{red}{#1}}
\newcommand{\verify}[1]{\textcolor{red}{[verify: #1]}}

\begin{document}

\title{Custom CNN versus Transfer Learning\\for Brain Tumor MRI Classification}

\author{
\IEEEauthorblockN{\todo{First Author}}
\IEEEauthorblockA{\todo{Department of \_\_\_\_}\\
\todo{University of \_\_\_\_}\\
\todo{email@example.com}}
\and
\IEEEauthorblockN{\todo{Second Author}}
\IEEEauthorblockA{\todo{Department of \_\_\_\_}\\
\todo{University of \_\_\_\_}\\
\todo{email@example.com}}
}

\maketitle

\begin{abstract}
Automated classification of brain MRI could give radiologists a second opinion, but small projects face a practical question: is it better to train a purpose-built network from scratch or to adapt a pretrained one? We compare a custom convolutional neural network (CNN) with MobileNetV2 transfer learning on a public four-class dataset (glioma, meningioma, pituitary tumor, no tumor). Before training anything, we audited the data and found that 41\% of the images in the official test folder have an exact or near-identical copy in the training folder. We therefore pooled the images and re-split them by duplicate cluster. Both models saw identical data and augmentation, and each was trained with three random seeds and evaluated once on 765 held-out images. Fine-tuned MobileNetV2 reached $97.7 \pm 0.5\%$ accuracy and 0.976 macro-F1, against $96.7 \pm 0.9\%$ and 0.965 for the custom CNN, in about one fifth of the training time (121\,s versus 590\,s). The gap in accuracy is small and the bootstrap confidence intervals of the best runs overlap, so we regard the two models as comparable. Training only a new classification head on a frozen backbone reached just 87.6\%, which shows that fine-tuning matters for MRI. Grad-CAM heatmaps often, but not always, point at the tumor. We also built a small web application for inference and explanation. The system is a research prototype, not a diagnostic tool.
\end{abstract}

\begin{IEEEkeywords}
brain tumor classification, MRI, convolutional neural network, transfer learning, MobileNetV2, Grad-CAM, data leakage.
\end{IEEEkeywords}

%==============================================================================
\section{Introduction}

Brain tumors are diagnosed mainly from magnetic resonance imaging (MRI), and the common types differ in where they start, how they behave and how they are treated. Gliomas grow from glial cells, meningiomas from the membranes that wrap the brain, and pituitary tumors from the pituitary gland at the base of the brain. Reading these scans takes expert time that is in short supply, which is why computer-aided tools that offer a second opinion, or help sort cases by urgency, are attractive.

Convolutional neural networks are the usual choice for the task, and small projects face a practical decision. One option is to design a compact network and train it from scratch. The other is to start from a network pretrained on a large collection of natural images and adapt it, which is known as transfer learning \cite{pan2010}. Pretrained models tend to train faster and need less data, but an MRI slice looks very different from a photograph, so the benefit is not guaranteed.

This paper makes three contributions:
\begin{enumerate}
\item A controlled comparison of a custom CNN and MobileNetV2 \cite{sandler2018} on four-class brain MRI classification. Both use the same splits, preprocessing, augmentation and optimizer family, with three seeds each, and we add a head-only transfer-learning baseline.
\item An audit of the public dataset showing that 41\% of the official test images have a twin in the training folder, together with a cluster-based split that removes the overlap.
\item An evaluation that goes beyond accuracy: macro-F1, per-class recall with confidence intervals, calibration, training cost and Grad-CAM explanations, plus a small demonstration application.
\end{enumerate}
Fig.~\ref{fig:pipeline} shows the whole pipeline.

\begin{figure*}[!t]
\centering
\includegraphics[width=0.88\textwidth]{pipeline.png}
\caption{Overview of the pipeline: data audit, cluster-based split, two models trained on identical data, selection by validation score, a single test evaluation, and a web application.}
\label{fig:pipeline}
\end{figure*}

%==============================================================================
\section{Related Work}

\textit{Brain tumor classification from MRI.} Convolutional networks, whether custom-built or fine-tuned from ImageNet models, dominate MRI classification. One study on the three-class figshare data compared a set of pretrained CNNs and found transfer learning effective \cite{chelghoum2020}. On the four-class dataset we use (7,023 images), Disci \textit{et al.} fine-tuned six pretrained networks, MobileNetV2 among them, and reported a weighted accuracy of up to 98.73\% with Xception \cite{disci2025}. Results like these are normally measured on the dataset's own training and testing folders. Our setting is smaller: one controlled comparison of a network trained from scratch and a pretrained one, with identical data, augmentation and seeds, on a split we checked for leakage.

\textit{Dataset integrity.} Several recent papers ask whether high accuracy on public brain tumor benchmarks says much about generalization. Saifullah \textit{et al.} removed duplicate and near-duplicate images with perceptual hashing; test accuracy dropped slightly, and nested validation gave a more realistic picture \cite{saifullah2025}. Chaudhary \textit{et al.} built an automated de-duplication pipeline based on file fingerprints and image similarity, released a cleaned benchmark of 3,522 unique images, and argued that models can end up memorizing leaked images \cite{chaudhary2026}. Our audit applies the same cluster-based idea to the four-class dataset and reports its own leakage rate. We make no claim about how much the cleaner split changes the headline accuracy.

\textit{Lightweight networks and explanation.} MobileNetV2 \cite{sandler2018} is small and cheap to run, and starting from ImageNet \cite{deng2009} weights is a common way to use it when data are limited \cite{pan2010}. Grad-CAM \cite{selvaraju2017} marks the image regions that influenced a prediction and is widely used to inspect medical imaging models.

\textit{Scope.} We do not propose a new architecture. The value of this work lies in a compact, reproducible comparison of a scratch CNN, head-only transfer learning and fine-tuning, judged by macro-F1, per-class recall, calibration, training cost and Grad-CAM on a leakage-free split.

%==============================================================================
\section{Dataset and Data Preparation}

\subsection{Dataset}
We used the Kaggle ``Brain Tumor MRI Dataset'' \cite{nickparvar}, a merge of the figshare, SARTAJ and Br35H datasets. Its uploader reports that the original SARTAJ glioma images were mislabeled and replaced them with figshare images. The dataset holds 7,023 JPEG images, which its authors split into a Training folder (5,712 images) and a Testing folder (1,311), labelled glioma, meningioma, pituitary tumor or no tumor. Every file could be read, and the folder labels agreed with the supplied metadata file. The images are far from uniform: 3,926 are RGB, 3,093 are grayscale and 4 use other modes; there are 387 distinct resolutions (many of them $512\times512$); and the slices come from axial, coronal and sagittal views with different crops.

\subsection{Duplicate Audit}
For every image we computed two fingerprints: a hash of the file bytes, which catches exact copies, and a 64-bit perceptual hash built from the DCT of a $32\times32$ grayscale thumbnail. We call two images near-duplicates when the Hamming distance between their perceptual hashes is at most 2. The threshold came from the data itself. Pairs with different labels first appear at distance 2 (3 pairs) and then grow quickly (33 at distance 4, 289 at distance 6), so 2 is the largest value that avoids matching unrelated scans. As a check, we compared $64\times64$ thumbnails of matches across the official folders; the near-duplicates turned out to be re-encoded or resized copies of the same image (median correlation 0.999 for the ``no tumor'' matches).

The audit found 297 redundant exact copies, none with conflicting labels. More importantly, \textbf{542 of the 1,311 official test images (41\%) have an exact or near-identical twin in the training folder}, including 368 of the 405 ``no tumor'' test images. Testing on this folder would therefore measure performance largely on images the model has effectively seen already. Simply deleting the twins is not a fix either, because only 37 ``no tumor'' test images would remain.

\subsection{Leakage-Free Split}
We pooled all unique images, collapsed exact copies and grouped the rest into duplicate clusters, which are the connected components of exact and near-duplicate links. We then split by cluster, stratified by class, with a fixed seed (42), giving roughly 71\% training, 14\% validation and 14\% test images (Table~\ref{tab:split}). In the validation and test sets we kept one image per (cluster, label), so repeated scans of one slice cannot inflate the metrics; this dropped 398 redundant images. Automated checks confirm that no cluster and no file hash appears in more than one split.

\begin{table}[!t]
\caption{Images per class after the leakage-free split}
\label{tab:split}
\centering
\begin{tabular}{lrrr}
\toprule
Class & Train & Validation & Test \\
\midrule
Glioma          & 1,160 & 220 & 217 \\
Meningioma      & 1,160 & 208 & 208 \\
No tumor        & 1,232 & 102 & 103 \\
Pituitary tumor & 1,245 & 236 & 237 \\
\midrule
\textbf{Total}  & \textbf{4,797} & \textbf{766} & \textbf{765} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Preprocessing and Augmentation}
Every image is converted to 3-channel RGB, resized to $224\times224$ with bicubic interpolation and normalized with the ImageNet mean and standard deviation. Both models and the application use exactly the same steps. Only training images are augmented, on the GPU, with random horizontal flips, rotations of $\pm15^\circ$, scaling between 0.9 and 1.1, translations of $\pm6\%$, contrast changes of $\pm15\%$ and a small brightness shift. We inspected augmented samples by eye to make sure they remain anatomically plausible.

%==============================================================================
\section{Methods}

\subsection{Models}
\textit{Custom CNN (from scratch).} The network has five convolutional blocks with 32, 64, 128, 256 and 256 channels. Each block has two $3\times3$ convolutions, each followed by batch normalization \cite{ioffe2015} and ReLU. The first four blocks end with $2\times2$ max pooling, while the last keeps a $14\times14$ feature map. Global average pooling, dropout (0.4) and a linear layer produce four outputs. Weights use He initialization \cite{he2015}, and the whole network has 2.35\,M trainable parameters.

\textit{MobileNetV2 (transfer learning).} We start from the ImageNet-pretrained network \cite{sandler2018,deng2009} and replace its classifier with dropout (0.3) and a four-way linear layer, which leaves 2.23\,M parameters in total. Training happens in two stages. In the first, the backbone is frozen and only the head (5,124 parameters) is trained for up to 8 epochs. In the second, the last 6 of the 19 feature blocks are unfrozen (1.80\,M trainable parameters) and trained with a learning rate ten times smaller. Batch normalization layers in frozen blocks keep their running statistics.

\subsection{Training}
Both models are trained with cross-entropy loss, the AdamW optimizer \cite{loshchilov2019} with weight decay $10^{-4}$, cosine learning-rate decay to 1\% of the starting value, batch size 32 and mixed-precision arithmetic. The custom CNN used a learning rate of $10^{-3}$, a budget of 40 epochs and early-stopping patience of 10. MobileNetV2 used $10^{-3}$ for the head (patience 5) and $10^{-4}$ for fine-tuning (up to 25 epochs, patience 7). In every stage we keep the checkpoint with the highest \textbf{validation macro-F1}. Each configuration was trained with three seeds (42, 43, 44), which change the initialization and the data order but not the split. All runs used an NVIDIA RTX 3080 laptop GPU (8\,GB) and PyTorch 2.9. The test set was read only after training, by a separate evaluation script, and never informed any decision. The run exported for the application, and used for the confusion matrices, was chosen for each model family by validation macro-F1 alone.

\subsection{Evaluation}
We report accuracy; macro-averaged precision, recall and F1; per-class recall with Wilson 95\% intervals; one-versus-rest macro ROC-AUC; expected calibration error (ECE, 15 bins) \cite{guo2017}; a 95\% bootstrap interval for macro-F1 (2,000 resamples); wall-clock training time including validation; and CPU inference latency for a single image. Results are means $\pm$ standard deviations over the three seeds. For qualitative explanation we apply Grad-CAM \cite{selvaraju2017} to the last convolutional feature map.

\subsection{Application}
A Streamlit web application loads the two exported models. The user uploads a PNG or JPEG slice, picks a model, and sees the predicted class, the four class probabilities and, optionally, a Grad-CAM heat map. The application rejects unreadable files, files smaller than 64\,px and files over 15\,MB. It warns when the top probability is below 60\% or when the image is strongly colored, since MRI slices are essentially grayscale, and it carries a research-use-only disclaimer. A separate page shows the comparison results reported in this paper.

%==============================================================================
\section{Results}

Table~\ref{tab:main} gives the test results and Table~\ref{tab:recall} the per-class recall. Table~\ref{tab:seeds} lists the individual seeds, and Fig.~\ref{fig:cm} shows the confusion matrices of the runs selected by validation score. The 95\% bootstrap interval for macro-F1 is $[0.961, 0.984]$ for both selected runs.

\begin{table*}[!t]
\caption{Test-set results (765 images), mean $\pm$ std over 3 seeds. Training time of the fine-tuned model includes the head-only stage.}
\label{tab:main}
\centering
\begin{tabular}{lccc}
\toprule
Metric & Custom CNN (scratch) & MobileNetV2 (head only) & MobileNetV2 (fine-tuned) \\
\midrule
Accuracy               & $0.967 \pm 0.009$ & $0.876 \pm 0.005$ & $\mathbf{0.977 \pm 0.005}$ \\
Macro precision        & $0.967 \pm 0.008$ & $0.885 \pm 0.000$ & $\mathbf{0.977 \pm 0.004}$ \\
Macro recall           & $0.965 \pm 0.012$ & $0.877 \pm 0.007$ & $\mathbf{0.975 \pm 0.006}$ \\
Macro-F1               & $0.965 \pm 0.010$ & $0.878 \pm 0.005$ & $\mathbf{0.976 \pm 0.005}$ \\
ROC-AUC (macro OvR)    & 0.999 & 0.978 & 0.999 \\
ECE                    & 0.014 & 0.045 & 0.014 \\
Training time (s)      & 590 & 33 & 121 \\
Trainable parameters   & 2,354,916 & 5,124 & 1,804,740 \\
CPU latency (ms/image) & 24.3 & 15.7 & 15.7 \\
\bottomrule
\end{tabular}
\end{table*}

\begin{table}[!t]
\caption{Per-class recall, mean $\pm$ std over 3 seeds. MNv2 = MobileNetV2.}
\label{tab:recall}
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccc}
\toprule
Class & CNN & MNv2 head only & MNv2 fine-tuned \\
\midrule
Glioma          & $0.971 \pm 0.005$ & $0.814 \pm 0.032$ & $0.975 \pm 0.005$ \\
Meningioma      & $0.929 \pm 0.026$ & $0.793 \pm 0.042$ & $0.957 \pm 0.013$ \\
No tumor        & $0.958 \pm 0.031$ & $0.913 \pm 0.026$ & $0.968 \pm 0.022$ \\
Pituitary tumor & $1.000 \pm 0.000$ & $0.989 \pm 0.002$ & $0.999 \pm 0.002$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!t]
\caption{Per-seed test accuracy / macro-F1}
\label{tab:seeds}
\centering
\begin{tabular}{lcc}
\toprule
Seed & Custom CNN & MobileNetV2 fine-tuned \\
\midrule
42 & 0.975 / 0.974 & 0.974 / 0.973 \\
43 & 0.957 / 0.954 & 0.974 / 0.973 \\
44 & 0.969 / 0.968 & 0.982 / 0.981 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{confusion_cnn.png}\\[2pt]
\includegraphics[width=\columnwidth]{confusion_mobilenet-ft.png}
\caption{Confusion matrices of the validation-selected runs on the test set (rows: true class, columns: predicted; counts and row percentages). Top: custom CNN (seed 42). Bottom: fine-tuned MobileNetV2 (seed 43).}
\label{fig:cm}
\end{figure}

\textit{Accuracy.} Both models reach above 96\% accuracy on average on the leakage-free test set. Fine-tuned MobileNetV2 is ahead by about one point in accuracy and in macro-F1, and its results vary less across seeds. The lead is small, though. The intervals of the two selected runs overlap, and one CNN seed (97.5\%) matches the MobileNetV2 runs, so we do not read the difference as conclusive.

\textit{Training cost.} MobileNetV2 was also about five times cheaper to train (121\,s versus 590\,s). Part of that gap reflects our budget: the CNN used all 40 epochs and reached its best epochs near the end (epochs 35--36; Fig.~\ref{fig:curves}), so it was probably still improving and might gain from longer training.

\textit{Frozen backbone.} Training only the head is not enough. That variant reached 87.6\%, with glioma and meningioma recall near 0.8, so ImageNet features alone transfer poorly to MRI.

\textit{Per-class behavior.} Meningioma is the hardest class for both models, with recall of 0.93 for the CNN and 0.96 for fine-tuned MobileNetV2, while pituitary tumors are found almost every time (recall about 1.0).

\textit{Calibration and speed.} Both models are well calibrated (ECE 0.014), so reported confidence tracks actual accuracy. Both classify an image in under 25\,ms on a CPU, and MobileNetV2 is faster (15.7 versus 24.3\,ms). We do not report single-image GPU timings because launch overhead dominated them.

\begin{figure*}[!t]
\centering
\includegraphics[width=0.78\textwidth]{curves_cnn.png}\\[4pt]
\includegraphics[width=0.78\textwidth]{curves_mobilenet.png}
\caption{Training and validation curves of the selected runs. Top: custom CNN. Bottom: MobileNetV2 (head-only stage, then fine-tuning, marked by the dotted line).}
\label{fig:curves}
\end{figure*}

\textit{Grad-CAM.} The heatmaps in Fig.~\ref{fig:gradcam} are mixed. Correct predictions often put the heat on the tumor, as with the bright mass in the meningioma and pituitary examples, but some maps are diffuse or sit on the skull border, which suggests the models sometimes rely on weaker cues. MobileNetV2's maps are coarser ($7\times7$ feature map) than the CNN's ($14\times14$). Several confident ``errors'' are scans labelled ``no tumor'' that visibly contain a bright lesion, which hints at label noise in the source data. These are qualitative observations; we did not quantify them.

\begin{figure*}[!t]
\centering
\includegraphics[width=0.95\textwidth]{gradcam_cnn.png}\\[4pt]
\includegraphics[width=0.95\textwidth]{gradcam_mobilenet.png}
\caption{Grad-CAM examples on test images: the most confident correct prediction of each class, followed by the most confident errors (green: correct, red: wrong). Top: custom CNN. Bottom: fine-tuned MobileNetV2.}
\label{fig:gradcam}
\end{figure*}

% --- Application screenshot: take a screenshot of the app, upload it as app_screenshot.png, then remove the % signs.
% \begin{figure}[!t]
% \centering
% \includegraphics[width=\columnwidth]{app_screenshot.png}
% \caption{Web application: uploaded scan, predicted class, class probabilities and Grad-CAM heat map.}
% \label{fig:app}
% \end{figure}

%==============================================================================
\section{Discussion}

On this dataset a small CNN trained from scratch gets close to a fine-tuned pretrained network. Fine-tuning MobileNetV2 gets there faster and a little higher, so pretraining buys efficiency more than a large accuracy gain. The poor head-only result adds a caveat: the benefit appears only when the pretrained features are adapted to the new domain.

The main practical lesson concerns evaluation. On the official folders, a large share of the test images are ones a model has effectively seen already, and the ``no tumor'' class can hardly be assessed. We did not train under the official protocol, so we cannot say how much it would change the headline accuracy. What the cluster split does guarantee is that our test score comes from images with no copy in the training data. Duplicate-aware splitting is cheap and should be standard for public medical image sets. Our scores are not directly comparable to results reported on the official split.

For clinical relevance, per-class recall matters more than overall accuracy, because a missed tumor is the costly error. On average our models miss about 3\% of gliomas and 4 to 7\% of meningiomas, which is too many for unsupervised use.

%==============================================================================
\section{Limitations}

\begin{itemize}
\item The dataset has no patient identifiers, so slices from one patient may fall into different splits. Our split removes duplicates, not patient overlap, so our test scores may well be optimistic.
\item Images come from several sources, scanners, views and crops. The models may partly learn source cues rather than tumor appearance.
\item We used a single public dataset, with no external or clinical validation. The ``no tumor'' test class is small (103 images).
\item Hyperparameters were fixed from standard defaults rather than searched, and the CNN's training budget may be limiting, as noted above.
\item The near-duplicate threshold is a heuristic. Grad-CAM is a qualitative attention map, not a tumor segmentation, and no expert has verified the suspected label noise.
\item The system is a research prototype and \textbf{must not be used for diagnosis or treatment decisions}.
\end{itemize}

%==============================================================================
\section{Conclusion}

We compared a custom CNN with MobileNetV2 transfer learning for four-class brain tumor MRI classification on a duplicate-audited, leakage-free split. Fine-tuned MobileNetV2 (97.7\% accuracy, 0.976 macro-F1) came out slightly ahead of the scratch CNN (96.7\%, 0.965) and trained about five times faster, although the difference is within run-to-run variation. A frozen backbone (87.6\%) was clearly worse. The leakage we found in the official split, where 41\% of test images have a twin in training, shows that auditing the data should come first. Future work includes patient-level splitting, external validation, 3D or multi-slice input, longer or tuned training, quantitative evaluation of the explanations, and clinician review of the questionable labels.

%==============================================================================
\begin{thebibliography}{99}

\bibitem{pan2010}
S. J. Pan and Q. Yang, ``A survey on transfer learning,'' \textit{IEEE Trans. Knowl. Data Eng.}, vol. 22, no. 10, pp. 1345--1359, 2010.

\bibitem{sandler2018}
M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, ``MobileNetV2: Inverted residuals and linear bottlenecks,'' in \textit{Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)}, 2018, pp. 4510--4520.

\bibitem{selvaraju2017}
R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, ``Grad-CAM: Visual explanations from deep networks via gradient-based localization,'' in \textit{Proc. IEEE Int. Conf. Computer Vision (ICCV)}, 2017, pp. 618--626.

\bibitem{deng2009}
J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, ``ImageNet: A large-scale hierarchical image database,'' in \textit{Proc. IEEE CVPR}, 2009, pp. 248--255.

\bibitem{ioffe2015}
S. Ioffe and C. Szegedy, ``Batch normalization: Accelerating deep network training by reducing internal covariate shift,'' in \textit{Proc. Int. Conf. Machine Learning (ICML)}, 2015, pp. 448--456.

\bibitem{loshchilov2019}
I. Loshchilov and F. Hutter, ``Decoupled weight decay regularization,'' in \textit{Proc. Int. Conf. Learning Representations (ICLR)}, 2019.

\bibitem{guo2017}
C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, ``On calibration of modern neural networks,'' in \textit{Proc. ICML}, 2017, pp. 1321--1330.

\bibitem{nickparvar}
M. Nickparvar, ``Brain Tumor MRI Dataset,'' Kaggle. [Online]. Available: \url{https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset} \verify{year and access date; the source description (figshare, SARTAJ, Br35H) was taken from a mirror of the Kaggle listing}

\bibitem{he2015}
K. He, X. Zhang, S. Ren, and J. Sun, ``Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification,'' in \textit{Proc. IEEE ICCV}, 2015, pp. 1026--1034.

\bibitem{disci2025}
R. Disci, F. Gurcan, and A. Soylu, ``Advanced brain tumor classification in MR images using transfer learning and pre-trained deep CNN models,'' \textit{Cancers}, vol. 17, no. 1, art. 121, 2025, doi: 10.3390/cancers17010121. \verify{author initials}

\bibitem{chelghoum2020}
R. Chelghoum, A. Ikhlef, A. Hameurlaine, and S. Jacquir, ``Transfer learning using convolutional neural network architectures for brain tumor classification from MRI images,'' in \textit{Artificial Intelligence Applications and Innovations (AIAI 2020), IFIP Advances in Information and Communication Technology}, Springer, 2020. \verify{authors, volume and pages}

\bibitem{saifullah2025}
Saifullah, S. Baldauf, and K. Sakumura, ``Reliable brain tumor classification without metadata: A step-by-step guideline with duplicate removal,'' in \textit{Proc. IEEE 37th Int. Conf. Tools with Artificial Intelligence (ICTAI)}, 2025, pp. 1237--1244, doi: 10.1109/ICTAI66417.2025.00180. \verify{author initials}

\bibitem{chaudhary2026}
D. Chaudhary, D. Lu, and Y. Zhang, ``MTA-Swin: A multi-token attention Swin transformer for brain tumor classification with leakage-free MRI benchmarking,'' 2026, doi: 10.1007/s10916-026-02418-w. \verify{author list, journal name, volume and pages; only a news summary of this paper could be read}

\end{thebibliography}

\end{document}
