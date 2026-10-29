"""Build leak-free train / val / test splits from the audit.

Why not the official Training/Testing folders? About 41% of the official Testing
images have a duplicate or near-duplicate in Training (worst for "notumor": 368 of
405). Evaluating on them would overstate accuracy, and removing them leaves only
26 "notumor" test images. So we pool every image and re-split it ourselves.

Rules
  * Exact copies (same file bytes) are collapsed to one image.
  * Images are grouped into duplicate clusters (exact copies + perceptual-hash
    neighbours within NEAR_DUP_THRESHOLD). A cluster never straddles two splits.
  * Split is stratified by class, drawn by cluster, seeded: ~71% train, ~14% val,
    ~14% test.
  * In val and test only one image is kept per (cluster, label), so repeated
    scans of the same slice cannot inflate the metrics. Train keeps near-copies.

Run:  python src/make_splits.py        (after src/audit.py)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def close_pairs(hashes: np.ndarray, thr: int):
    """All index pairs (i<j) whose 64-bit hashes differ in <= thr bits."""
    n = len(hashes)
    pairs = []
    for i in range(n):
        x = hashes[i + 1:] ^ hashes[i]
        d = _POP[x.view(np.uint8).reshape(-1, 8)].sum(1)
        for j in np.nonzero(d <= thr)[0]:
            pairs.append((i, i + 1 + int(j)))
    return pairs


def cluster_ids(df: pd.DataFrame, thr: int) -> np.ndarray:
    parent = list(range(len(df)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # exact copies
    first = {}
    for i, m in enumerate(df["md5"]):
        union(first.setdefault(m, i), i)
    # perceptual near-duplicates
    h = np.array([int(x) for x in df["phash"]], dtype=np.uint64)
    for a, b in close_pairs(h, thr):
        union(a, b)
    roots = np.array([find(i) for i in range(len(df))])
    return pd.factorize(roots)[0]


def group_split(df: pd.DataFrame, n_splits: int):
    """Hold out one stratified, cluster-respecting fold (~1/n_splits). Returns (rest, held)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=C.SEED)
    rest_idx, held_idx = next(sgkf.split(df, df["label_id"], df["cluster"]))
    return df.iloc[rest_idx], df.iloc[held_idx]


def main() -> None:
    audit = pd.read_csv(C.PROCESSED_DIR / "audit.csv").sort_values("path").reset_index(drop=True)
    audit["cluster"] = cluster_ids(audit, C.NEAR_DUP_THRESHOLD)
    audit["label_id"] = audit["label"].map({c: i for i, c in enumerate(C.CLASSES)})

    # How contaminated is the official split? (reported, not used for splitting)
    off_train = audit[audit.split == "training"]
    off_test = audit[audit.split == "testing"]
    leaking = off_test["cluster"].isin(set(off_train["cluster"]))
    official = {
        "test_images": int(len(off_test)),
        "test_images_with_twin_in_train": int(leaking.sum()),
        "per_class_test_with_twin": off_test[leaking].groupby("label").size().to_dict(),
        "per_class_test": off_test.groupby("label").size().to_dict(),
    }

    pool = audit.drop_duplicates("md5")                        # collapse exact copies
    rest, test_all = group_split(pool, 7)                      # ~14.3% test
    train, val_all = group_split(rest, 6)                      # ~14.3% of total val
    val = val_all.drop_duplicates(["cluster", "label"])
    test = test_all.drop_duplicates(["cluster", "label"])

    out = pd.concat([
        train.assign(fold="train"), val.assign(fold="val"), test.assign(fold="test"),
    ])[["path", "label", "label_id", "fold", "cluster", "md5", "width", "height", "mode"]]
    out = out.sort_values(["fold", "path"]).reset_index(drop=True)
    out.to_csv(C.PROCESSED_DIR / "splits.csv", index=False)

    # --- sanity checks: these must hold or the experiment is invalid ---------
    cl = {f: set(out[out.fold == f]["cluster"]) for f in ("train", "val", "test")}
    assert not (cl["train"] & cl["val"]), "cluster shared by train and val"
    assert not (cl["train"] & cl["test"]), "cluster shared by train and test"
    assert not (cl["val"] & cl["test"]), "cluster shared by val and test"
    md = {f: set(out[out.fold == f]["md5"]) for f in ("train", "val", "test")}
    assert not (md["train"] & md["test"]) and not (md["train"] & md["val"]) and not (md["val"] & md["test"])
    assert out["path"].is_unique
    assert set(out["label"]) == set(C.CLASSES)

    report = {
        "near_dup_threshold": C.NEAR_DUP_THRESHOLD,
        "seed": C.SEED,
        "official_split_contamination": official,
        "images_total": int(len(audit)),
        "exact_copies_collapsed": int(len(audit) - len(pool)),
        "redundant_near_copies_dropped_from_val_test": int(
            (len(val_all) - len(val)) + (len(test_all) - len(test))),
        "final": {f: int((out.fold == f).sum()) for f in ("train", "val", "test")},
        "per_class": out.groupby(["fold", "label"]).size().unstack().to_dict(),
        "checks_passed": ["no cluster shared across splits", "no md5 shared across splits",
                          "paths unique", "all four classes present"],
    }
    (C.REPORTS_DIR / "splits.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
