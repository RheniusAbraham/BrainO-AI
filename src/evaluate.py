"""Evaluate every trained run on the held-out test set and build the comparison report.

  python src/evaluate.py

This is the ONLY script that reads the test fold. Which run is exported for each model
family is decided by validation macro-F1 alone, never by test results.

Outputs: reports/metrics.json, reports/predictions/*.csv, reports/figures/*.png,
         models/exported/{cnn,mobilenet}.pt
"""
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix,  # noqa: E402
                             precision_recall_fscore_support, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.dataset import load_fold  # noqa: E402
from src.inference import load_model, predict_probs  # noqa: E402

GROUPS = {
    "cnn": "Custom CNN (from scratch)",
    "mobilenet-head": "MobileNetV2 (head only)",
    "mobilenet-ft": "MobileNetV2 (fine-tuned)",
}
COLORS = {"cnn": "#4C78A8", "mobilenet-head": "#9AA5B1", "mobilenet-ft": "#E8743B"}
EXPORT_FROM = {"cnn": "cnn", "mobilenet": "mobilenet-ft"}      # exported name -> group
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = [C.CLASS_DISPLAY[c] for c in C.CLASSES]


# ---------------------------------------------------------------- metrics
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(centre - half), float(centre + half)]


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    err, rel = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = float((pred[m] == y[m]).mean()), float(conf[m].mean())
            err += m.mean() * abs(acc - c)
            rel.append((c, acc, int(m.sum())))
    return float(err), rel


def bootstrap_f1(y, pred, n=2000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(precision_recall_fscore_support(y[i], pred[i], average="macro", zero_division=0)[2])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def compute(y: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(len(C.CLASSES)), zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    e, rel = ece(probs, y)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
        "macro_f1_ci95": bootstrap_f1(y, pred),
        "auc_ovr_macro": float(roc_auc_score(y, probs, multi_class="ovr", average="macro")),
        "ece": e, "reliability": rel,
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i]),
                          "recall_ci95": wilson(int(round(r[i] * s[i])), int(s[i]))}
                      for i, c in enumerate(C.CLASSES)},
        "confusion": confusion_matrix(y, pred, labels=range(len(C.CLASSES))).tolist(),
    }


def latency_ms(model, device, n=50) -> float:
    x = torch.randint(0, 256, (1, 3, C.IMG_SIZE, C.IMG_SIZE), dtype=torch.uint8)
    model = model.to(device)
    with torch.no_grad():
        for _ in range(5):
            predict_probs(model, x, device)
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            predict_probs(model, x, device)
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


# ---------------------------------------------------------------- figures
def style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110})


def fig_confusion(cm, title, path):
    cm = np.array(cm)
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.grid(False)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_curves(histories, title, path):
    """histories: list of history lists (concatenated; stage change marked)."""
    rows = [r for h in histories for r in h]
    x = np.arange(1, len(rows) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(x, [r["train_loss"] for r in rows], label="train", color="#4C78A8")
    ax[0].plot(x, [r["val_loss"] for r in rows], label="validation", color="#E8743B")
    ax[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    ax[1].plot(x, [r["train_acc"] for r in rows], label="train accuracy", color="#4C78A8")
    ax[1].plot(x, [r["val_acc"] for r in rows], label="val accuracy", color="#E8743B")
    ax[1].plot(x, [r["val_macro_f1"] for r in rows], label="val macro-F1", color="#2E8B57", ls="--")
    ax[1].set(xlabel="Epoch", ylabel="Score", title="Accuracy / macro-F1", ylim=(0.4, 1.01))
    if len(histories) > 1:
        cut = len(histories[0]) + 0.5
        for a in ax:
            a.axvline(cut, color="grey", ls=":", lw=1)
        ax[0].text(cut, ax[0].get_ylim()[1] * 0.95, " fine-tune ->", fontsize=8, va="top", color="grey")
    for a in ax:
        a.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_bars(summary, key, ylabel, title, path, ylim=None, fmt="{:.3f}"):
    names = list(summary)
    means = [summary[n]["mean"][key] for n in names]
    stds = [summary[n]["std"][key] for n in names]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(range(len(names)), means, yerr=stds, capsize=4, color=[COLORS[n] for n in names], width=0.6)
    for i, n in enumerate(names):
        pts = [r[key] for r in summary[n]["runs"].values()]
        ax.scatter([i] * len(pts), pts, color="black", s=14, zorder=3)
        ax.text(i, means[i] + (stds[i] or 0) + (ax.get_ylim()[1] * 0.01), fmt.format(means[i]), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names)), [GROUPS[n].replace(" (", "\n(") for n in names], fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
        ylabel += f" (axis starts at {ylim[0]})"
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_per_class(summary, path):
    names = list(summary)
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4))
    for k, n in enumerate(names):
        m = [summary[n]["mean"]["recall_" + c] for c in C.CLASSES]
        s = [summary[n]["std"]["recall_" + c] for c in C.CLASSES]
        ax.bar(np.arange(4) + (k - 1) * w, m, w, yerr=s, capsize=3, label=GROUPS[n], color=COLORS[n])
    ax.set_xticks(range(4), LABELS)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Recall (axis starts at 0.5)")
    ax.set_title("Per-class recall on the held-out test set (mean +/- std over seeds)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_calibration(rels, path):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="perfect calibration")
    for n, (rel, e) in rels.items():
        c, a, _ = zip(*rel)
        ax.plot(c, a, marker="o", ms=4, color=COLORS[n], label=f"{GROUPS[n]}  ECE {e:.3f}")
    ax.set(xlabel="Mean confidence", ylabel="Accuracy in bin", xlim=(0.2, 1.02), ylim=(0.2, 1.02),
           title="Reliability diagram (test set)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    C.ensure_dirs()
    style()
    (C.REPORTS_DIR / "predictions").mkdir(exist_ok=True)
    x, y_t, df = load_fold("test")
    y = y_t.numpy()
    print(f"test set: {len(y)} images, device {DEV}")

    groups, hist = {}, {}
    for g in GROUPS:
        # a run is finished once its history json exists (written after the last epoch)
        tags = sorted(p.stem for p in C.CKPT_DIR.glob(f"{g}_s*.pt")
                      if (C.REPORTS_DIR / "history" / f"{p.stem}.json").exists())
        runs = {}
        for tag in tags:
            model, meta = load_model(C.CKPT_DIR / f"{tag}.pt", DEV)
            probs = predict_probs(model, x, DEV)
            m = compute(y, probs)
            h = json.loads((C.REPORTS_DIR / "history" / f"{tag}.json").read_text())
            hist[tag] = h
            m.update({
                "seed": meta["seed"], "best_epoch": h["best_epoch"], "epochs_run": h["epochs_run"],
                "val_macro_f1": h["best_val"]["val_macro_f1"],
                "train_seconds": h["cumulative_train_seconds"],
                "params_total": h["params_total"], "params_trainable": h["params_trainable"],
                "latency_ms_gpu": latency_ms(model, DEV) if DEV == "cuda" else None,
                "latency_ms_cpu": latency_ms(model, "cpu", n=30),
            })
            pd.DataFrame({"path": df["path"], "label": df["label"], "label_id": y,
                          "pred": probs.argmax(1), "conf": probs.max(1),
                          **{f"p_{c}": probs[:, i] for i, c in enumerate(C.CLASSES)}}
                         ).to_csv(C.REPORTS_DIR / "predictions" / f"{tag}.csv", index=False)
            runs[tag] = m
            print(f"{tag}: acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  (val F1 {m['val_macro_f1']:.4f})")
        if not runs:
            continue
        # flat per-run scalars for aggregation
        flat = {t: {**{k: r[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                                        "auc_ovr_macro", "ece", "train_seconds", "latency_ms_cpu")},
                    **{f"recall_{c}": r["per_class"][c]["recall"] for c in C.CLASSES}}
                for t, r in runs.items()}
        keys = next(iter(flat.values())).keys()
        mean = {k: float(np.mean([f[k] for f in flat.values()])) for k in keys}
        std = {k: float(np.std([f[k] for f in flat.values()], ddof=1)) if len(flat) > 1 else 0.0 for k in keys}
        selected = max(runs, key=lambda t: runs[t]["val_macro_f1"])        # validation only
        groups[g] = {"display": GROUPS[g], "runs": flat, "mean": mean, "std": std,
                     "selected_by_val": selected, "detail": runs}

    # ---- export the validation-selected checkpoint of each family
    exported = {}
    for name, g in EXPORT_FROM.items():
        if g in groups:
            tag = groups[g]["selected_by_val"]
            shutil.copy(C.CKPT_DIR / f"{tag}.pt", C.EXPORT_DIR / f"{name}.pt")
            shutil.copy(C.REPORTS_DIR / "predictions" / f"{tag}.csv", C.REPORTS_DIR / "predictions" / f"{name}.csv")
            exported[name] = tag

    # ---- figures
    summary = {g: groups[g] for g in GROUPS if g in groups}
    for g, v in summary.items():
        sel = v["detail"][v["selected_by_val"]]
        fig_confusion(sel["confusion"], f"{GROUPS[g]}, seed {sel['seed']} (test, n={len(y)})",
                      C.FIG_DIR / f"confusion_{g}.png")
    for name, g, tags in (("cnn", "cnn", lambda s: [f"cnn_s{s}"]),
                          ("mobilenet", "mobilenet-ft", lambda s: [f"mobilenet-head_s{s}", f"mobilenet-ft_s{s}"])):
        if g in summary:
            seed = summary[g]["detail"][summary[g]["selected_by_val"]]["seed"]
            fig_curves([hist[t]["history"] for t in tags(seed)], f"{GROUPS[g]}, seed {seed}",
                       C.FIG_DIR / f"curves_{name}.png")
    fig_bars(summary, "macro_f1", "Macro-F1", "Test macro-F1 (bar = mean, whisker = std, dots = seeds)",
             C.FIG_DIR / "compare_macro_f1.png", ylim=(0.8, 1.0))
    fig_bars(summary, "accuracy", "Accuracy", "Test accuracy (bar = mean, whisker = std, dots = seeds)",
             C.FIG_DIR / "compare_accuracy.png", ylim=(0.8, 1.0))
    fig_bars(summary, "train_seconds", "Training time (s)", "Training time (head + fine-tune included)",
             C.FIG_DIR / "compare_train_time.png", fmt="{:.0f}")
    fig_per_class(summary, C.FIG_DIR / "compare_per_class_recall.png")
    fig_calibration({g: (summary[g]["detail"][summary[g]["selected_by_val"]]["reliability"],
                         summary[g]["detail"][summary[g]["selected_by_val"]]["ece"])
                     for g in ("cnn", "mobilenet-ft") if g in summary}, C.FIG_DIR / "calibration.png")

    splits = json.loads((C.REPORTS_DIR / "splits.json").read_text())
    out = {
        "test_set": {"n": int(len(y)), "per_class": {c: int((y == i).sum()) for i, c in enumerate(C.CLASSES)}},
        "split_info": {"final": splits["final"], "near_dup_threshold": splits["near_dup_threshold"]},
        "selection_rule": "exported checkpoint = run with the highest validation macro-F1 in each family; "
                          "test data never used for any decision",
        "exported": exported,
        "groups": groups,
    }
    for g in out["groups"].values():                       # keep reliability curves out of the json
        for r in g["detail"].values():
            r.pop("reliability", None)
    (C.REPORTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    print("wrote reports/metrics.json and figures")


if __name__ == "__main__":
    main()
