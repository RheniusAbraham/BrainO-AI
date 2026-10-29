"""Train the custom CNN or MobileNetV2. Uses train + validation folds only (never test).

  python src/train.py --model cnn --seed 42
  python src/train.py --model mobilenet --seed 42     # stage 1: head only, stage 2: fine-tune

Outputs
  models/checkpoints/<tag>.pt        best checkpoint by validation macro-F1
  reports/history/<tag>.json         per-epoch curves, timing, parameter counts
Tags: cnn_s42, mobilenet-head_s42, mobilenet-ft_s42
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.dataset import augment, load_fold, normalize  # noqa: E402
from src.model import build_model, n_params  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_loader(model, x, y, bs=128):
    """Loss / accuracy / macro-F1 on a fold held on the GPU."""
    model.eval()
    losses, preds = [], []
    for i in range(0, len(x), bs):
        xb = normalize(x[i:i + bs].to(DEV))
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            out = model(xb)
        losses.append(nn.functional.cross_entropy(out.float(), y[i:i + bs].to(DEV), reduction="sum").item())
        preds.append(out.argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    yt = y.numpy()
    return {
        "loss": sum(losses) / len(x),
        "acc": float((preds == yt).mean()),
        "macro_f1": float(f1_score(yt, preds, average="macro")),
    }


def set_train_mode(model) -> None:
    """train(), but BatchNorm layers whose weights are frozen keep their running stats."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not m.weight.requires_grad:
            m.eval()


def run_stage(model, tag, data, *, epochs, lr, bs, seed, patience, stage, base_seconds=0.0, extra_meta=None):
    (xtr, ytr), (xva, yva) = data
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    gen = torch.Generator(device=DEV).manual_seed(seed)
    perm_gen = torch.Generator().manual_seed(seed)
    xtr_d, ytr_d = xtr.to(DEV), ytr.to(DEV)

    history, best, bad = [], None, 0
    ckpt_path = C.CKPT_DIR / f"{tag}.pt"
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        set_train_mode(model)
        perm = torch.randperm(len(xtr_d), generator=perm_gen).to(DEV)
        tl, tc, seen = 0.0, 0, 0
        for i in range(0, len(perm) - bs + 1, bs):          # drop last partial batch
            idx = perm[i:i + bs]
            xb, yb = augment(xtr_d[idx], gen), ytr_d[idx]
            with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
                out = model(xb)
                loss = nn.functional.cross_entropy(out.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * len(idx)
            tc += (out.argmax(1) == yb).sum().item()
            seen += len(idx)
        sched.step()
        val = evaluate_loader(model, xva, yva)
        row = {"epoch": ep, "stage": stage, "train_loss": tl / seen, "train_acc": tc / seen,
               "val_loss": val["loss"], "val_acc": val["acc"], "val_macro_f1": val["macro_f1"],
               "lr": opt.param_groups[0]["lr"]}
        history.append(row)
        improved = best is None or val["macro_f1"] > best["val_macro_f1"] + 1e-4 or (
            abs(val["macro_f1"] - best["val_macro_f1"]) <= 1e-4 and val["loss"] < best["val_loss"])
        if improved:
            best, bad = {**row}, 0
            best["elapsed_seconds"] = time.perf_counter() - t0
            torch.save({"tag": tag, "state_dict": model.state_dict(),
                        "classes": C.CLASSES, "img_size": C.IMG_SIZE, "mean": C.MEAN, "std": C.STD,
                        "seed": seed, "epoch": ep, "val": val, **(extra_meta or {})}, ckpt_path)
        else:
            bad += 1
        print(f"[{tag}] ep {ep:02d}/{epochs} loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} | "
              f"val loss {val['loss']:.4f} acc {val['acc']:.3f} f1 {val['macro_f1']:.3f}"
              f"{' *' if improved else ''}", flush=True)
        if bad >= patience:
            print(f"[{tag}] early stop (no val improvement for {patience} epochs)", flush=True)
            break
    wall = time.perf_counter() - t0
    info = {"tag": tag, "stage": stage, "seed": seed, "epochs_run": len(history),
            "epochs_budget": epochs, "lr": lr, "batch_size": bs,
            "train_seconds": wall, "cumulative_train_seconds": base_seconds + wall,
            "seconds_to_best_epoch": best["elapsed_seconds"], "best_epoch": best["epoch"],
            "best_val": {k: best[k] for k in ("val_loss", "val_acc", "val_macro_f1")},
            "params_total": n_params(model), "params_trainable": sum(p.numel() for p in params),
            "history": history}
    (C.REPORTS_DIR / "history").mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "history" / f"{tag}.json").write_text(json.dumps(info, indent=2))
    # restore the best weights so a following stage starts from them
    model.load_state_dict(torch.load(ckpt_path, map_location=DEV, weights_only=False)["state_dict"])
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "mobilenet"], required=True)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--epochs", type=int, default=None, help="override epoch budget (fine-tune stage for mobilenet)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--split", default="splits", help="csv in data/processed to train on")
    ap.add_argument("--prefix", default="", help="checkpoint/history tag prefix, e.g. official-")
    args = ap.parse_args()

    C.ensure_dirs()
    set_seed(args.seed)
    print(f"device: {DEV} ({torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'}), seed {args.seed}")
    xtr, ytr, _ = load_fold("train", args.split)
    xva, yva, _ = load_fold("val", args.split)
    data = ((xtr, ytr), (xva, yva))
    print(f"train {tuple(xtr.shape)}  val {tuple(xva.shape)}")

    model = build_model(args.model).to(DEV)
    if args.model == "cnn":
        run_stage(model, f"{args.prefix}cnn_s{args.seed}", data, epochs=args.epochs or 40, lr=1e-3, bs=args.bs,
                  seed=args.seed, patience=10, stage="scratch", extra_meta={"arch": "cnn"})
    else:
        # Stage 1: only the new classification head learns; ImageNet features stay fixed.
        model.freeze_backbone()
        head = run_stage(model, f"{args.prefix}mobilenet-head_s{args.seed}", data, epochs=8, lr=1e-3, bs=args.bs,
                         seed=args.seed, patience=5, stage="head", extra_meta={"arch": "mobilenet"})
        # Stage 2: unfreeze the last 6 feature blocks, much smaller learning rate.
        model.unfreeze_last(6)
        run_stage(model, f"{args.prefix}mobilenet-ft_s{args.seed}", data, epochs=args.epochs or 25, lr=1e-4, bs=args.bs,
                  seed=args.seed, patience=7, stage="finetune",
                  base_seconds=head["train_seconds"], extra_meta={"arch": "mobilenet"})


if __name__ == "__main__":
    main()
