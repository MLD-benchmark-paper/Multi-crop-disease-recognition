#!/usr/bin/env python3
"""
TRAINING: FLAT RESNET-18 BASELINE (JOINT CROP+DISEASE LABELS)
===================================================================
  - Reads dataset path from $DATASET_PATH (set by SLURM to local scratch,
    or point it at your own local dataset copy); see README for setup.
  - DataLoader: pin_memory=True, persistent_workers=True, prefetch_factor=4.
  - NUM_WORKERS=12 (matches --cpus-per-task=16 minus headroom).
  - Mixed precision (torch.cuda.amp) on the train + val forward passes.
  - Resume logic: skip any fold whose fold_summary.csv already exists.
  - MAJORITY_CAP=None (intentionally; merged dataset, no downsampling).

Notes:
  - Validation loss kept UNWEIGHTED for fair comparison across configs.
  - Capping logic is retained in code but disabled by config (cap=None).
"""

import os
import json
from pathlib import Path
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, classification_report
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data.dataloader import default_collate

# ============================================================
#                       CONFIG
# ============================================================
SEED          = 42
BATCH_SIZE    = 32
NUM_WORKERS   = 12
MAX_EPOCHS    = 75
PATIENCE      = 10
MAJORITY_CAP  = None     # disabled: keep all data from merged sources
USE_LOSS_WEIGHTS = True
USE_AMP       = True     # mixed precision

# ============================================================
#                  GLOBAL SEEDING / WORKERS
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


# ============================================================
#                         MODEL
# ============================================================
class FlatResNet18(nn.Module):
    def __init__(self, num_joint_classes):
        super().__init__()
        try:
            backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            print("Loaded ResNet18 pretrained weights.")
        except Exception:
            print("Offline mode: loading local ResNet-18 weights.")
            backbone = models.resnet18(weights=None)
            local_path = os.environ.get("RESNET18_WEIGHTS_PATH")
            if local_path is None:
                raise RuntimeError(
                    "No internet access and RESNET18_WEIGHTS_PATH is not set. "
                    "Download resnet18-f37072fd.pth from torchvision and set "
                    "RESNET18_WEIGHTS_PATH to its local path."
                )
            backbone.load_state_dict(torch.load(local_path, map_location="cpu"))

        for name, p in backbone.named_parameters():
            p.requires_grad = ("layer3" in name) or ("layer4" in name)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_joint_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


# ============================================================
#                       DATASET
# ============================================================
class HierDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.t = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, crop_id, dis_id = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            print(f"[CORRUPTED] Skipping {path}", flush=True)
            return None
        img = self.t(img)
        global items_global_map
        global_dis_id = items_global_map[(crop_id, dis_id)]
        return img, crop_id, dis_id, global_dis_id


# ============================================================
#                         TRANSFORMS
# ============================================================
def make_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def make_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ============================================================
#                     BUILD DATA INDEX
# ============================================================
def build_index(dataset_root):
    root  = Path(dataset_root)
    crops = sorted([d.name for d in root.iterdir() if d.is_dir()])
    diseases_by_crop = {}
    items = []
    for ci, crop in enumerate(crops):
        ddir     = root / crop
        dis_list = sorted([d.name for d in ddir.iterdir() if d.is_dir()])
        diseases_by_crop[crop] = dis_list
        for di, dis in enumerate(dis_list):
            for img in sorted((ddir / dis).glob("*")):
                if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                    continue
                items.append((str(img), ci, di))
    return crops, diseases_by_crop, items


# ============================================================
#               IMBALANCE-HANDLING UTILITIES
# ============================================================
def cap_majority_classes(items, items_global_map, cap, rng):
    """Randomly subsample any joint class with > cap examples down to cap."""
    if cap is None:
        return items
    by_cls = defaultdict(list)
    for it in items:
        _, ci, di = it
        by_cls[items_global_map[(ci, di)]].append(it)

    capped = []
    for cls, lst in by_cls.items():
        if len(lst) > cap:
            idxs = rng.sample(range(len(lst)), cap)
            capped.extend([lst[i] for i in idxs])
        else:
            capped.extend(lst)
    rng.shuffle(capped)
    print(f"   Capping: {len(items)} -> {len(capped)} samples (cap={cap})")
    return capped


def make_weighted_sampler(train_items, items_global_map, generator):
    """Per-sample weights = 1 / class_count, sampling with replacement."""
    counts = Counter()
    for _, ci, di in train_items:
        counts[items_global_map[(ci, di)]] += 1
    sample_weights = torch.tensor(
        [1.0 / counts[items_global_map[(ci, di)]] for _, ci, di in train_items],
        dtype=torch.double,
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_items),
        replacement=True,
        generator=generator,
    )


def compute_loss_weights(train_items, items_global_map, num_joint_classes, device):
    """
    Returns class-weight tensor [num_joint_classes] for the joint head.
    Inverse-frequency, mean-normalized so mean weight = 1.
    """
    counts = torch.zeros(num_joint_classes)
    for _, ci, di in train_items:
        counts[items_global_map[(ci, di)]] += 1
    w = counts.sum() / (num_joint_classes * counts.clamp(min=1))
    w = w / w.mean()
    return w.to(device)


# ============================================================
#                   CONFUSION MATRIX UTILS
# ============================================================
def plot_confusion_matrix(cm, labels, save_path, title, normalised=False):
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35),
                                    max(6, len(labels) * 0.3)))
    fmt_cm  = cm.astype(float)
    im = ax.imshow(fmt_cm, interpolation="nearest",
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
    """Save both raw and normalised confusion matrices."""
    plot_confusion_matrix(cm_raw, labels,
                          fold_dir / f"{prefix}.png", title)
    pd.DataFrame(cm_raw, index=labels, columns=labels).to_csv(
        fold_dir / f"{prefix}.csv"
    )
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums == 0, 0, cm_raw / row_sums)
    plot_confusion_matrix(cm_norm, labels,
                          fold_dir / f"{prefix}_normalised.png",
                          title + " (Normalised)", normalised=True)
    pd.DataFrame(cm_norm, index=labels, columns=labels).to_csv(
        fold_dir / f"{prefix}_normalised.csv"
    )


# ============================================================
#                   TRAINING CURVE PLOT
# ============================================================
def plot_training_curves(epoch_log, fold_dir):
    df = pd.DataFrame(epoch_log)
    df.to_csv(fold_dir / "training_curves.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train loss", linewidth=1.5)
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val loss",   linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["epoch"], df["lr_backbone"], label="LR backbone", linewidth=1.5)
    axes[1].plot(df["epoch"], df["lr_head"],     label="LR head",     linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")

    plt.tight_layout()
    fig.savefig(fold_dir / "training_curves.png", dpi=150)
    plt.close(fig)


# ============================================================
#                       EVALUATION
# ============================================================
def evaluate(model, model_path, val_loader, device, fold_dir,
             crops, global_labels, global_to_crop_dis, crop_to_global_ids):

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    true_crop, pred_crop           = [], []
    true_global, pred_global       = [], []
    pred_global_true_crop          = []

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            imgs, yc, yd, yg = batch
            imgs = imgs.to(device, non_blocking=True)
            yc   = yc.to(device, non_blocking=True)
            yg   = yg.to(device, non_blocking=True)

            logits = model(imgs)
            preds  = logits.argmax(dim=1)

            for i in range(imgs.size(0)):
                gi_true = int(yg[i].item())
                gi_pred = int(preds[i].item())
                ci_true, _ = global_to_crop_dis[gi_true]
                ci_pred, _ = global_to_crop_dis[gi_pred]

                true_global.append(gi_true)
                pred_global.append(gi_pred)
                true_crop.append(ci_true)
                pred_crop.append(ci_pred)

                # Oracle crop: argmax over only the joint classes belonging to true crop
                crop_global_ids = crop_to_global_ids[ci_true]
                crop_logits     = torch.stack(
                    [logits[i][g_id] for g_id in crop_global_ids], dim=0
                )
                gi_pred_true_crop = int(crop_global_ids[crop_logits.argmax().item()])
                pred_global_true_crop.append(gi_pred_true_crop)

    # ── Accuracies ────────────────────────────────────────────────────────────
    crop_acc              = accuracy_score(true_crop,   pred_crop)
    disease_acc_pred_crop = accuracy_score(true_global, pred_global)
    disease_acc_true_crop = accuracy_score(true_global, pred_global_true_crop)

    # ── Confusion matrices (raw + normalised) ────────────────────────────────
    cm_crop = confusion_matrix(true_crop, pred_crop, labels=range(len(crops)))
    save_confusion_matrices(cm_crop, crops, fold_dir, "cm_crop", "Crop CM")

    cm_dis = confusion_matrix(true_global, pred_global,
                               labels=range(len(global_labels)))
    save_confusion_matrices(cm_dis, global_labels, fold_dir,
                            "cm_disease_pred_crop",
                            "Disease CM (Pred Crop)")

    cm_dis_oracle = confusion_matrix(true_global, pred_global_true_crop,
                                     labels=range(len(global_labels)))
    save_confusion_matrices(cm_dis_oracle, global_labels, fold_dir,
                            "cm_disease_true_crop",
                            "Disease CM (Oracle Crop)")

    # ── F1 scores ─────────────────────────────────────────────────────────────
    f1_crop_macro     = f1_score(true_crop,   pred_crop,             average='macro',    zero_division=0)
    f1_crop_weighted  = f1_score(true_crop,   pred_crop,             average='weighted', zero_division=0)
    f1_dis_pred_macro = f1_score(true_global, pred_global,           average='macro',    zero_division=0)
    f1_dis_pred_wt    = f1_score(true_global, pred_global,           average='weighted', zero_division=0)
    f1_dis_true_macro = f1_score(true_global, pred_global_true_crop, average='macro',    zero_division=0)
    f1_dis_true_wt    = f1_score(true_global, pred_global_true_crop, average='weighted', zero_division=0)

    # ── Per-class classification reports ──────────────────────────────────────
    pd.DataFrame(
        classification_report(true_crop, pred_crop,
                              target_names=crops,
                              zero_division=0, output_dict=True)
    ).T.to_csv(fold_dir / "classification_report_crop.csv")

    pd.DataFrame(
        classification_report(true_global, pred_global,
                              target_names=global_labels,
                              zero_division=0, output_dict=True)
    ).T.to_csv(fold_dir / "classification_report_disease_pred_crop.csv")

    pd.DataFrame(
        classification_report(true_global, pred_global_true_crop,
                              target_names=global_labels,
                              zero_division=0, output_dict=True)
    ).T.to_csv(fold_dir / "classification_report_disease_true_crop.csv")

    return {
        "crop_acc":                      crop_acc,
        "disease_acc_pred_crop":         disease_acc_pred_crop,
        "disease_acc_true_crop":         disease_acc_true_crop,
        "n_val_samples":                 len(true_crop),
        "f1_crop_macro":                 f1_crop_macro,
        "f1_crop_weighted":              f1_crop_weighted,
        "f1_disease_pred_crop_macro":    f1_dis_pred_macro,
        "f1_disease_pred_crop_weighted": f1_dis_pred_wt,
        "f1_disease_true_crop_macro":    f1_dis_true_macro,
        "f1_disease_true_crop_weighted": f1_dis_true_wt,
    }


# ============================================================
#                        TRAINING
# ============================================================
def train_fold(fold, model, train_loader, val_loader, device, fold_dir,
               crops, global_labels, global_to_crop_dis, crop_to_global_ids,
               class_w):

    if USE_LOSS_WEIGHTS:
        criterion_train = nn.CrossEntropyLoss(weight=class_w)
    else:
        criterion_train = nn.CrossEntropyLoss()
    criterion_val = nn.CrossEntropyLoss()  # unweighted for clean val loss

    backbone_params = [
        p for n, p in model.backbone.named_parameters()
        if p.requires_grad and ('layer3' in n or 'layer4' in n)
    ]
    head_params = list(model.backbone.fc.parameters())

    optimizer = Adam([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': head_params,     'lr': 1e-3},
    ], weight_decay=1e-4)

    scheduler  = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_loss  = float("inf")
    wait       = 0
    model_path = fold_dir / "best_model.pth"
    epoch_log  = []

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        total_train = 0.0
        n_batches   = 0
        for batch in train_loader:
            if batch is None:
                continue
            imgs, yc, yd, yg = batch
            imgs = imgs.to(device, non_blocking=True)
            yg   = yg.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                loss = criterion_train(model(imgs), yg)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train += loss.item()
            n_batches   += 1
        train_loss_avg = total_train / max(n_batches, 1)

        # ── Validate (unweighted) ──────────────────────────────────────────────
        model.eval()
        total_val = 0.0
        n_val     = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                imgs, yc, yd, yg = batch
                imgs = imgs.to(device, non_blocking=True)
                yg   = yg.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    total_val += criterion_val(model(imgs), yg).item()
                n_val     += 1
        val_loss_avg = total_val / max(n_val, 1)

        scheduler.step(val_loss_avg)

        lr_bb   = optimizer.param_groups[0]['lr']
        lr_head = optimizer.param_groups[1]['lr']

        print(f"[Fold {fold}] Epoch {epoch:02d} | "
              f"Train: {train_loss_avg:.4f} | Val: {val_loss_avg:.4f} | "
              f"LR backbone: {lr_bb:.2e} | LR head: {lr_head:.2e}",
              flush=True)

        epoch_log.append({
            "epoch":       epoch,
            "train_loss":  train_loss_avg,
            "val_loss":    val_loss_avg,
            "lr_backbone": lr_bb,
            "lr_head":     lr_head,
        })

        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            wait = 0
            torch.save(model.state_dict(), model_path)
            print("   Best model updated!", flush=True)
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping!", flush=True)
                break

    plot_training_curves(epoch_log, fold_dir)

    eval_summary = evaluate(
        model, model_path, val_loader, device, fold_dir,
        crops, global_labels, global_to_crop_dis, crop_to_global_ids
    )

    summary = {
        "fold":                          fold,
        "crop_acc":                      eval_summary["crop_acc"],
        "disease_acc_pred_crop":         eval_summary["disease_acc_pred_crop"],
        "disease_acc_true_crop":         eval_summary["disease_acc_true_crop"],
        "f1_crop_macro":                 eval_summary["f1_crop_macro"],
        "f1_crop_weighted":              eval_summary["f1_crop_weighted"],
        "f1_disease_pred_crop_macro":    eval_summary["f1_disease_pred_crop_macro"],
        "f1_disease_pred_crop_weighted": eval_summary["f1_disease_pred_crop_weighted"],
        "f1_disease_true_crop_macro":    eval_summary["f1_disease_true_crop_macro"],
        "f1_disease_true_crop_weighted": eval_summary["f1_disease_true_crop_weighted"],
        "best_val_loss":                 best_loss,
        "total_epochs":                  len(epoch_log),
    }

    pd.DataFrame([summary]).to_csv(fold_dir / "fold_summary.csv", index=False)
    return summary


# ============================================================
#                           MAIN
# ============================================================
def main():
    # Dataset path: set DATASET_PATH to point at your local copy of the
    # (harmonized) dataset, e.g. the output of src/data_prep/harmonize_labels.py.
    DATASET = os.environ.get("DATASET_PATH", "./data/processed/plantvillage")
    SAVE_ROOT = os.environ.get("SAVE_ROOT", "./outputs/train_flat")
    os.makedirs(SAVE_ROOT, exist_ok=True)
    print(f"Dataset path: {DATASET}", flush=True)

    crops, diseases_by_crop, items = build_index(DATASET)

    with open(f"{SAVE_ROOT}/label_maps.json", "w") as f:
        json.dump({"crops": crops, "diseases_within_crop": diseases_by_crop}, f, indent=4)

    global_index = {}
    labels = []
    idx = 0
    for ci, crop in enumerate(crops):
        for di, dis in enumerate(diseases_by_crop[crop]):
            global_index[(ci, di)] = idx
            labels.append(f"{crop}:{dis}")
            idx += 1

    num_joint_classes  = idx
    global_to_crop_dis = {gid: (ci, di) for (ci, di), gid in global_index.items()}
    crop_to_global_ids = {}
    for (ci, di), gid in global_index.items():
        crop_to_global_ids.setdefault(ci, []).append(gid)

    global items_global_map
    items_global_map = global_index
    global_labels    = labels

    paths        = [p for p, _, _ in items]
    joint_labels = [items_global_map[(c, d)] for _, c, d in items]

    train_transform = make_train_transform()
    val_transform   = make_val_transform()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Total joint classes: {num_joint_classes}", flush=True)
    print(f"Config: BATCH={BATCH_SIZE}, NUM_WORKERS={NUM_WORKERS}, "
          f"MAJORITY_CAP={MAJORITY_CAP}, USE_LOSS_WEIGHTS={USE_LOSS_WEIGHTS}, "
          f"USE_AMP={USE_AMP}", flush=True)

    skf          = StratifiedKFold(5, shuffle=True, random_state=SEED)
    fold_results = []

    fold_rng = random.Random(SEED)  # for capping (no-op when cap is None)

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, joint_labels), 1):
        print(f"\n===== FOLD {fold} =====", flush=True)
        fold_dir = Path(SAVE_ROOT) / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # ── Resume: skip fold if it has already finished ───────────────────────
        fold_summary_path = fold_dir / "fold_summary.csv"
        if fold_summary_path.exists():
            print(f"   Fold {fold} already complete — loading existing summary.", flush=True)
            fold_results.append(pd.read_csv(fold_summary_path).iloc[0].to_dict())
            continue

        train_items_full = [items[i] for i in train_idx]
        val_items        = [items[i] for i in val_idx]

        # ── Cap (no-op when MAJORITY_CAP is None) ──────────────────────────────
        train_items = cap_majority_classes(
            train_items_full, items_global_map, MAJORITY_CAP, fold_rng
        )

        # ── Class-weight tensor ────────────────────────────────────────────────
        class_w = compute_loss_weights(
            train_items, items_global_map, num_joint_classes, device
        )

        # ── Weighted sampler ───────────────────────────────────────────────────
        sampler = make_weighted_sampler(train_items, items_global_map, g)

        train_loader = DataLoader(
            HierDataset(train_items, train_transform),
            batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=NUM_WORKERS,
            collate_fn=safe_collate, worker_init_fn=seed_worker, generator=g,
            pin_memory=True, persistent_workers=True, prefetch_factor=4,
        )
        val_loader = DataLoader(
            HierDataset(val_items, val_transform),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=safe_collate, worker_init_fn=seed_worker, generator=g,
            pin_memory=True, persistent_workers=True, prefetch_factor=4,
        )

        model = FlatResNet18(num_joint_classes=num_joint_classes).to(device)

        # Per-fold class distribution (for audit)
        cls_counts = Counter(items_global_map[(c, d)] for _, c, d in train_items)
        pd.DataFrame(
            [{"global_class_id": k, "label": global_labels[k], "count": v}
             for k, v in sorted(cls_counts.items())]
        ).to_csv(fold_dir / "train_class_counts.csv", index=False)

        summary = train_fold(
            fold, model, train_loader, val_loader, device,
            fold_dir, crops, global_labels, global_to_crop_dis, crop_to_global_ids,
            class_w
        )
        fold_results.append(summary)

    # ── Summary across folds: mean, std, min, max ─────────────────────────────
    df_folds = pd.DataFrame(fold_results)
    agg_rows = []
    for agg_name, agg_fn in [("mean", "mean"), ("std", lambda x: x.std(ddof=0)),
                               ("min", "min"),  ("max", "max")]:
        row = {"fold": agg_name}
        for col in df_folds.columns:
            if col == "fold":
                continue
            if pd.api.types.is_numeric_dtype(df_folds[col]):
                row[col] = getattr(df_folds[col], agg_fn)() \
                    if isinstance(agg_fn, str) else agg_fn(df_folds[col])
            else:
                row[col] = None
        agg_rows.append(row)

    df_out = pd.concat([df_folds, pd.DataFrame(agg_rows)], ignore_index=True)
    df_out.to_csv(f"{SAVE_ROOT}/summary_all_folds.csv", index=False)
    print("\n=== TRAINING COMPLETE: FLAT RESNET-18 v5 ===", flush=True)


if __name__ == "__main__":
    main()
