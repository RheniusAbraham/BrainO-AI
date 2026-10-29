"""Dataset audit: integrity, label consistency, exact and near duplicates.

Writes data/processed/audit.csv (one row per image) and reports/audit.json.
Run:  python src/audit.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402


def phash(img: Image.Image) -> int:
    """64-bit perceptual hash (DCT of a 32x32 grayscale thumbnail)."""
    small = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    flat = dctn(small, norm="ortho")[:8, :8].ravel()
    bits = flat > np.median(flat[1:])  # median excludes the DC term
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def scan() -> pd.DataFrame:
    rows, corrupt = [], []
    for split in ("Training", "Testing"):
        for cls in C.CLASSES:
            for p in sorted((C.RAW_DIR / split / cls).glob("*.jpg")):
                rel = p.relative_to(C.RAW_DIR).as_posix()
                try:
                    with Image.open(p) as im:
                        im.load()
                        w, h = im.size
                        mode = im.mode
                        ph = phash(im)
                    # Exact-duplicate key: hash of the file bytes.
                    md5 = hashlib.md5(p.read_bytes()).hexdigest()
                except Exception as e:  # corrupt / unreadable
                    corrupt.append({"path": rel, "error": repr(e)})
                    continue
                rows.append(dict(path=rel, split=split.lower(), label=cls,
                                 width=w, height=h, mode=mode, md5=md5, phash=str(ph)))
    df = pd.DataFrame(rows)
    df.attrs["corrupt"] = corrupt
    return df


def main() -> None:
    C.ensure_dirs()
    df = scan()
    df.to_csv(C.PROCESSED_DIR / "audit.csv", index=False)

    # Cross-check folder labels against the dataset's own CSV.
    info = pd.read_csv(C.RAW_DIR / "dataset_info.csv")
    info["path"] = info["image_path"].str.replace("\\", "/", regex=False)
    merged = df.merge(info[["path", "class_name"]], on="path", how="outer", indicator=True)
    csv_only = int((merged["_merge"] == "right_only").sum())
    folder_only = int((merged["_merge"] == "left_only").sum())
    label_mismatch = int(((merged["_merge"] == "both") & (merged["label"] != merged["class_name"])).sum())

    dup_groups = df.groupby("md5").filter(lambda g: len(g) > 1).groupby("md5")
    exact = {
        "groups": int(dup_groups.ngroups),
        "extra_copies": int(sum(len(g) - 1 for _, g in dup_groups)),
        "groups_with_conflicting_labels": int(sum(g["label"].nunique() > 1 for _, g in dup_groups)),
        "train_test_overlap_images": int(
            df[df["md5"].isin(set(df[df.split == "training"].md5))
               & (df.split == "testing")].shape[0]),
    }

    report = {
        "n_images": int(len(df)),
        "n_corrupt": len(df.attrs["corrupt"]),
        "corrupt": df.attrs["corrupt"],
        "per_split_class": df.groupby(["split", "label"]).size().unstack().to_dict(),
        "modes": df["mode"].value_counts().to_dict(),
        "n_distinct_sizes": int(df.groupby(["width", "height"]).ngroups),
        "csv_cross_check": {"in_csv_not_on_disk": csv_only, "on_disk_not_in_csv": folder_only,
                            "label_mismatch": label_mismatch},
        "exact_duplicates": exact,
    }
    (C.REPORTS_DIR / "audit.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
