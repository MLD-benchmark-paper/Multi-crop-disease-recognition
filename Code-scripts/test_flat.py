#!/usr/bin/env python3
"""
TESTING: FLAT RESNET-18 BASELINE (JOINT CROP+DISEASE LABELS) — v4
==================================================================
Changes from v3 (HPC throughput; same metrics + outputs):
  - MODEL_ROOT / SAVE_ROOT now match the v5 training output (logs_flatMld).
  - TEST_ROOT readable from $TEST_PATH env var (for local-scratch staging).
  - cudnn.benchmark=True, deterministic=False (forward-only; speed only,
    no effect on argmax outputs given fixed weights).
  - NUM_WORKERS=12 + pin_memory=True, persistent_workers=True,
    prefetch_factor=4 on the DataLoader.
  - Mixed precision (autocast) on inference forward passes.
  - Model is built once per region and reloaded per fold instead of
    reinstantiated every fold (skips redundant pretrained-weight loads).
  - All v3 outputs preserved (per-class reports, raw + normalised CMs,
    per-fold variance summary, table1 and table2).
"""

import os
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import default_collate

from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, classification_report
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
#                    REPRODUCIBILITY
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# Inference-only: speed > determinism. argmax outputs are deterministic
# given fixed weights regardless of these flags.
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
#                    PATHS / CONFIG
# ============================================================
MODEL_ROOT = "/home/nalwangar/again/logs_flatZ"
TEST_ROOT  = os.environ.get(
    "TEST_PATH",
    "/deepstore/datasets/dmb/ComputerVision/biology/test2",
)
SAVE_ROOT  = "/home/nalwangar/again/logs_flatZ/testK"

N_FOLDS     = 5
BATCH_SIZE  = 32
NUM_WORKERS = 12
USE_AMP     = True

os.makedirs(SAVE_ROOT, exist_ok=True)


# ============================================================
#                         MODEL
# ============================================================
class FlatResNet18(nn.Module):
    def __init__(self, num_joint_classes):
        super().__init__()
        try:
            backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = models.resnet18(weights=None)
            local_path = "/home/nalwangar/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
            backbone.load_state_dict(torch.load(local_path, map_location="cpu"))

        for name, p in backbone.named_parameters():
            p.requires_grad = ("layer3" in name) or ("layer4" in name)

        in_features  = backbone.fc.in_features
        backbone.fc  = nn.Linear(in_features, num_joint_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


# ============================================================
#               TRANSFORMS & SAFE COLLATE
# ============================================================
def make_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None


# ============================================================
#                     DATASET
# ============================================================
class RegionFlatDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.t     = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, crop_id, dis_local, global_joint_id = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return None
        return self.t(img), crop_id, dis_local, global_joint_id


# ============================================================
#               LABEL MAP LOADING
# ============================================================
def load_label_maps(model_root):
    with open(Path(model_root) / "label_maps.json") as f:
        lm = json.load(f)
    crops            = lm["crops"]
    diseases_by_crop = lm["diseases_within_crop"]
    global_index     = {}
    global_labels    = []
    idx = 0
    for ci, crop in enumerate(crops):
        for di, dis in enumerate(diseases_by_crop[crop]):
            global_index[(ci, di)] = idx
            global_labels.append(f"{crop}:{dis}")
            idx += 1
    global_to_crop_dis = {gid: (ci, di) for (ci, di), gid in global_index.items()}
    crop_to_global_ids = {}
    for (ci, di), gid in global_index.items():
        crop_to_global_ids.setdefault(ci, []).append(gid)
    return crops, diseases_by_crop, global_index, global_labels, \
           global_to_crop_dis, crop_to_global_ids


# ============================================================
#          BUILD REGION ITEMS
# ============================================================
def build_region_items(region_root, train_crops, train_dis, global_index):
    items = []
    stats = {"total_images": 0, "skipped_unknown_crop": 0,
             "known_crop_known_disease": 0, "known_crop_unknown_disease": 0}
    region_root = Path(region_root)
    if not region_root.exists():
        return items, stats
    for crop_dir in sorted([d for d in region_root.iterdir() if d.is_dir()]):
        crop_name = crop_dir.name
        if crop_name not in train_crops:
            for dis_dir in [d for d in crop_dir.iterdir() if d.is_dir()]:
                for img in dis_dir.glob("*"):
                    if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                        stats["skipped_unknown_crop"] += 1
            continue
        ci             = train_crops.index(crop_name)
        train_dis_list = train_dis[crop_name]
        for dis_dir in sorted([d for d in crop_dir.iterdir() if d.is_dir()]):
            dis_name = dis_dir.name
            if dis_name in train_dis_list:
                di = train_dis_list.index(dis_name); known = True
            else:
                di = -1; known = False
            for img_path in sorted(dis_dir.glob("*")):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                    continue
                stats["total_images"] += 1
                if known:
                    stats["known_crop_known_disease"] += 1
                    gid = global_index[(ci, di)]
                else:
                    stats["known_crop_unknown_disease"] += 1
                    gid = -1
                items.append((str(img_path), ci, di, gid))
    return items, stats


# ============================================================
#                CONFUSION MATRIX UTILS
# ============================================================
def plot_confusion_matrix(cm, labels, save_path, title, normalised=False):
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35),
                                    max(6, len(labels) * 0.3)))
    im = ax.imshow(cm.astype(float), interpolation="nearest",
                   vmin=0, vmax=1 if normalised else None)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    plt.colorbar(im)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def save_confusion_matrices(cm_raw, labels, fold_dir, prefix, title):
    plot_confusion_matrix(cm_raw, labels, fold_dir / f"{prefix}.png", title)
    pd.DataFrame(cm_raw, index=labels, columns=labels).to_csv(
        fold_dir / f"{prefix}.csv"
    )
    row_sums = cm_raw.sum(axis=1, keepdims=True).astype(float)
    cm_norm  = np.zeros_like(cm_raw, dtype=float)
    nonzero  = row_sums[:, 0] > 0
    cm_norm[nonzero] = cm_raw[nonzero] / row_sums[nonzero]
    plot_confusion_matrix(cm_norm, labels,
                          fold_dir / f"{prefix}_normalised.png",
                          title + " (Normalised)", normalised=True)
    pd.DataFrame(cm_norm, index=labels, columns=labels).to_csv(
        fold_dir / f"{prefix}_normalised.csv"
    )


# ============================================================
#              EVALUATE ONE FOLD
# =============================================================
def evaluate_region_fold(model, model_path, loader, device, fold_dir,
                         crops, global_labels, global_to_crop_dis,
                         crop_to_global_ids):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    true_crop_all, pred_crop_all      = [], []
    true_global_all                   = []
    pred_global_pred_crop_all         = []
    pred_global_true_crop_all         = []
    per_crop_results = {ci: {"true": [], "pred": []} for ci in range(len(crops))}

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            imgs, yc, yd_local, yg_global = batch
            imgs      = imgs.to(device, non_blocking=True)
            yc        = yc.to(device, non_blocking=True)
            yg_global = yg_global.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(imgs)
            preds = logits.argmax(dim=1)

            for i in range(imgs.size(0)):
                ci_true = int(yc[i].item())
                gi_true = int(yg_global[i].item())
                gi_pred = int(preds[i].item())
                ci_pred, _ = global_to_crop_dis[gi_pred]

                true_crop_all.append(ci_true)
                pred_crop_all.append(ci_pred)

                if gi_true >= 0:
                    pred_global_pred_crop_all.append(gi_pred)
                    true_global_all.append(gi_true)

                    crop_global_ids = crop_to_global_ids[ci_true]
                    crop_logits     = torch.stack(
                        [logits[i][g_id] for g_id in crop_global_ids], dim=0
                    )
                    gi_pred_true_crop = int(
                        crop_global_ids[crop_logits.argmax().item()]
                    )
                    pred_global_true_crop_all.append(gi_pred_true_crop)

                    per_crop_results[ci_true]["true"].append(gi_true)
                    per_crop_results[ci_true]["pred"].append(gi_pred)

    crop_acc              = accuracy_score(true_crop_all, pred_crop_all) if true_crop_all else 0.0
    disease_acc_pred_crop = accuracy_score(true_global_all, pred_global_pred_crop_all) if true_global_all else 0.0
    disease_acc_true_crop = accuracy_score(true_global_all, pred_global_true_crop_all) if true_global_all else 0.0

    # ── Confusion matrices (raw + normalised) ────────────────────────────────
    cm_crop = confusion_matrix(true_crop_all, pred_crop_all,
                               labels=list(range(len(crops))))
    save_confusion_matrices(cm_crop, crops, fold_dir, "cm_crop", "Crop CM")

    if true_global_all:
        cm_dis_pred = confusion_matrix(true_global_all, pred_global_pred_crop_all,
                                       labels=list(range(len(global_labels))))
        save_confusion_matrices(cm_dis_pred, global_labels, fold_dir,
                                "cm_disease_pred_crop", "Disease CM (Pred Crop)")

        cm_dis_true = confusion_matrix(true_global_all, pred_global_true_crop_all,
                                       labels=list(range(len(global_labels))))
        save_confusion_matrices(cm_dis_true, global_labels, fold_dir,
                                "cm_disease_true_crop", "Disease CM (Oracle Crop)")

    # ── F1 scores ─────────────────────────────────────────────────────────────
    f1_crop_macro    = f1_score(true_crop_all, pred_crop_all, average="macro",    zero_division=0)
    f1_crop_weighted = f1_score(true_crop_all, pred_crop_all, average="weighted", zero_division=0)

    if true_global_all:
        f1_dis_pred_macro = f1_score(true_global_all, pred_global_pred_crop_all, average="macro",    zero_division=0)
        f1_dis_pred_wt    = f1_score(true_global_all, pred_global_pred_crop_all, average="weighted", zero_division=0)
        f1_dis_true_macro = f1_score(true_global_all, pred_global_true_crop_all, average="macro",    zero_division=0)
        f1_dis_true_wt    = f1_score(true_global_all, pred_global_true_crop_all, average="weighted", zero_division=0)
    else:
        f1_dis_pred_macro = f1_dis_pred_wt = f1_dis_true_macro = f1_dis_true_wt = None

    # ── Per-class classification reports ──────────────────────────────────────
    pd.DataFrame(
        classification_report(true_crop_all, pred_crop_all,
                              labels=list(range(len(crops))),
                              target_names=crops,
                              zero_division=0, output_dict=True)
    ).T.to_csv(fold_dir / "classification_report_crop.csv")

    if true_global_all:
        pd.DataFrame(
            classification_report(true_global_all, pred_global_pred_crop_all,
                                  labels=list(range(len(global_labels))),
                                  target_names=global_labels,
                                  zero_division=0, output_dict=True)
        ).T.to_csv(fold_dir / "classification_report_disease_pred_crop.csv")

        pd.DataFrame(
            classification_report(true_global_all, pred_global_true_crop_all,
                                  labels=list(range(len(global_labels))),
                                  target_names=global_labels,
                                  zero_division=0, output_dict=True)
        ).T.to_csv(fold_dir / "classification_report_disease_true_crop.csv")

    # ── Per-crop accuracy and per-crop macro-F1 ──────────────────────────────
    per_crop_acc      = {}
    per_crop_f1_macro = {}
    for ci in per_crop_results:
        t = per_crop_results[ci]["true"]
        p = per_crop_results[ci]["pred"]
        if t:
            per_crop_acc[ci] = accuracy_score(t, p)
            # Restrict F1 averaging to the diseases that belong to this crop,
            # so macro is over within-crop classes only (not all global classes).
            crop_label_ids = crop_to_global_ids[ci]
            per_crop_f1_macro[ci] = f1_score(
                t, p,
                labels=crop_label_ids,
                average="macro",
                zero_division=0,
            )
        else:
            per_crop_acc[ci]      = None
            per_crop_f1_macro[ci] = None

    return {
        "crop_acc":                      crop_acc,
        "disease_acc_pred_crop":         disease_acc_pred_crop,
        "disease_acc_true_crop":         disease_acc_true_crop,
        "n_crop_samples":                len(true_crop_all),
        "n_disease_eval_samples":        len(true_global_all),
        "f1_crop_macro":                 f1_crop_macro,
        "f1_crop_weighted":              f1_crop_weighted,
        "f1_disease_pred_crop_macro":    f1_dis_pred_macro,
        "f1_disease_pred_crop_weighted": f1_dis_pred_wt,
        "f1_disease_true_crop_macro":    f1_dis_true_macro,
        "f1_disease_true_crop_weighted": f1_dis_true_wt,
        "per_crop_acc":                  per_crop_acc,
        "per_crop_f1_macro":             per_crop_f1_macro,   # NEW
    }

# ============================================================
#                            MAIN
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"MODEL_ROOT: {MODEL_ROOT}", flush=True)
    print(f"TEST_ROOT:  {TEST_ROOT}", flush=True)
    print(f"SAVE_ROOT:  {SAVE_ROOT}", flush=True)

    (crops, diseases_by_crop, global_index, global_labels,
     global_to_crop_dis, crop_to_global_ids) = load_label_maps(MODEL_ROOT)

    num_joint_classes = len(global_labels)
    tform             = make_test_transform()
    print(f"Num joint classes: {num_joint_classes}", flush=True)

    region_dirs = sorted(
        [d for d in Path(TEST_ROOT).iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    # Build the model once; reload weights per fold below.
    model = FlatResNet18(num_joint_classes=num_joint_classes).to(device)

    all_region_summaries = []

    for region_dir in region_dirs:
        region = region_dir.name
        print(f"\n=== Evaluating region: {region} ===", flush=True)

        items, stats = build_region_items(
            region_dir, crops, diseases_by_crop, global_index
        )
        if not items:
            print(f"  No usable images in {region}. Skipping.", flush=True)
            continue

        region_save = Path(SAVE_ROOT) / region
        region_save.mkdir(parents=True, exist_ok=True)

        dl = DataLoader(
            RegionFlatDataset(items, tform),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, collate_fn=safe_collate,
            pin_memory=True, persistent_workers=True, prefetch_factor=4,
        )

        fold_summaries = []

        for fold in range(1, N_FOLDS + 1):
            print(f"  Fold {fold}...", flush=True)
            fold_path = Path(MODEL_ROOT) / f"fold{fold}" / "best_model.pth"
            if not fold_path.exists():
                print(f"    Missing: {fold_path}", flush=True)
                continue

            fold_dir = region_save / f"fold{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            summary = evaluate_region_fold(
                model, fold_path, dl, device, fold_dir,
                crops, global_labels, global_to_crop_dis, crop_to_global_ids
            )

            flat = {
                "region": region, "fold": fold,
                "crop_acc":                      summary["crop_acc"],
                "disease_acc_pred_crop":         summary["disease_acc_pred_crop"],
                "disease_acc_true_crop":         summary["disease_acc_true_crop"],
                "n_crop_samples":                summary["n_crop_samples"],
                "n_disease_eval_samples":        summary["n_disease_eval_samples"],
                "f1_crop_macro":                 summary["f1_crop_macro"],
                "f1_crop_weighted":              summary["f1_crop_weighted"],
                "f1_disease_pred_crop_macro":    summary["f1_disease_pred_crop_macro"],
                "f1_disease_pred_crop_weighted": summary["f1_disease_pred_crop_weighted"],
                "f1_disease_true_crop_macro":    summary["f1_disease_true_crop_macro"],
                "f1_disease_true_crop_weighted": summary["f1_disease_true_crop_weighted"],
            }
            # Per-crop accuracy AND per-crop macro-F1
            for ci, acc in summary["per_crop_acc"].items():
                flat[f"disease_{crops[ci]}"]          = acc
                flat[f"disease_{crops[ci]}_f1_macro"] = summary["per_crop_f1_macro"][ci]

            fold_summaries.append(flat)
            pd.DataFrame([flat]).to_csv(fold_dir / "fold_summary.csv", index=False)

        if not fold_summaries:
            continue

        df_region = pd.DataFrame(fold_summaries)
        df_region.to_csv(region_save / "summary_folds.csv", index=False)

        # ── Region stats: mean, std, min, max ────────────────────────────────
        metric_cols = ["crop_acc", "disease_acc_pred_crop", "disease_acc_true_crop",
                       "f1_crop_macro", "f1_crop_weighted",
                       "f1_disease_pred_crop_macro", "f1_disease_pred_crop_weighted",
                       "f1_disease_true_crop_macro", "f1_disease_true_crop_weighted"]

        region_stats = {"region": region, **stats}
        for col in metric_cols:
            region_stats[f"mean_{col}"] = df_region[col].mean()
            region_stats[f"std_{col}"]  = df_region[col].std(ddof=0)
            region_stats[f"min_{col}"]  = df_region[col].min()
            region_stats[f"max_{col}"]  = df_region[col].max()

        # Per-crop accuracy aggregation
        for crop in crops:
            col = f"disease_{crop}"
            if col in df_region.columns:
                region_stats[f"mean_{col}"] = df_region[col].mean()
                region_stats[f"std_{col}"]  = df_region[col].std(ddof=0)
                region_stats[f"min_{col}"]  = df_region[col].min()
                region_stats[f"max_{col}"]  = df_region[col].max()
            else:
                for agg in ["mean", "std", "min", "max"]:
                    region_stats[f"{agg}_{col}"] = None

        # Per-crop macro-F1 aggregation (NEW)
        for crop in crops:
            col = f"disease_{crop}_f1_macro"
            if col in df_region.columns:
                region_stats[f"mean_{col}"] = df_region[col].mean()
                region_stats[f"std_{col}"]  = df_region[col].std(ddof=0)
                region_stats[f"min_{col}"]  = df_region[col].min()
                region_stats[f"max_{col}"]  = df_region[col].max()
            else:
                for agg in ["mean", "std", "min", "max"]:
                    region_stats[f"{agg}_{col}"] = None

        pd.DataFrame([region_stats]).to_csv(region_save / "summary_stats.csv", index=False)

        # ── Table 1: global metrics ───────────────────────────────────────────
        def fmt(m, s):
            return None if (m is None or pd.isna(m)) else f"{m:.3f} ± {s:.3f}"

        table1_row = {
            "region":    region,
            "P1_global": fmt(region_stats["mean_crop_acc"],
                             region_stats["std_crop_acc"]),
        }
        for crop in crops:
            table1_row[f"P2_{crop}"] = fmt(
                region_stats.get(f"mean_disease_{crop}"),
                region_stats.get(f"std_disease_{crop}"),
            )
        table1_row["P3_global"] = fmt(
            region_stats["mean_disease_acc_pred_crop"],
            region_stats["std_disease_acc_pred_crop"],
        )
        pd.DataFrame([table1_row]).to_csv(
            region_save / "table1_global_metrics.csv", index=False
        )

        # ── Table 2: per-crop metrics (with macro-F1) ────────────────────────
        per_crop_fold_vals = {crop: [] for crop in crops}
        for _, row in df_region.iterrows():
            cm_path = region_save / f"fold{int(row['fold'])}" / "cm_crop.csv"
            if cm_path.exists():
                cm = pd.read_csv(cm_path, index_col=0).values
                for ci, crop in enumerate(crops):
                    total = cm[ci, :].sum()
                    if total > 0:
                        per_crop_fold_vals[crop].append(cm[ci, ci] / total)

        table2_rows  = []
        all_p2_means = []
        for crop in crops:
            vals    = per_crop_fold_vals[crop]
            p1_mean = np.mean(vals) if vals else None
            p1_std  = np.std(vals, ddof=0) if vals else None
            p2_mean = region_stats.get(f"mean_disease_{crop}")
            p2_std  = region_stats.get(f"std_disease_{crop}")
            f1_mean = region_stats.get(f"mean_disease_{crop}_f1_macro")
            f1_std  = region_stats.get(f"std_disease_{crop}_f1_macro")
            if p2_mean is not None:
                all_p2_means.append(p2_mean)
            table2_rows.append({
                "crop":                 crop,
                "P1_per_crop":          fmt(p1_mean, p1_std),
                "P2_per_crop":          fmt(p2_mean, p2_std),
                "P2_per_crop_f1_macro": fmt(f1_mean, f1_std),   # NEW
            })

        p2_global = fmt(np.mean(all_p2_means),
                        np.std(all_p2_means, ddof=0)) if all_p2_means else None
        for row in table2_rows:
            row["P2_global"] = p2_global
        pd.DataFrame(table2_rows).to_csv(
            region_save / "table2_per_crop_metrics.csv", index=False
        )

        all_region_summaries.append(region_stats)

    if all_region_summaries:
        pd.DataFrame(all_region_summaries).to_csv(
            Path(SAVE_ROOT) / "summary_all_regions.csv", index=False
        )
        print("\n=== DONE: Multi-region flat evaluation complete (v4) ===", flush=True)
    else:
        print("\nNo regions evaluated.", flush=True)

if __name__ == "__main__":
    main()
