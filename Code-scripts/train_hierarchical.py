#!/usr/bin/env python3
"""
TRAINING: FULLY HIERARCHICAL RESNET-18  CONCATENATED HEADS) — v4
==========================================================================
Changes from v3:
  - Class imbalance handling:
      * WeightedRandomSampler for the training loader (per-sample inverse-freq).
      * Capping of the largest classes (configurable MAJORITY_CAP).
      * Per-crop disease-loss class weights + crop-level class weights.
  - Vectorized disease-loss computation (no per-sample Python loop).
  - All v3 outputs preserved (per-class reports, normalised CMs,
    training curves, per-fold variance summary).

Notes:
  - We use BOTH a weighted sampler AND mild class-weighted loss. If you find
    this over-corrects (minority classes overshooting majority), drop the
    loss weights first and keep only the sampler.
  - Capping is applied to the TRAIN split only inside each fold, so the
    validation set still reflects the real (imbalanced) distribution.
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
import torch.nn.functional as F
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
MAJORITY_CAP  = None     # set to None to disable capping
USE_LOSS_WEIGHTS = True  # if over-correcting, set to False (sampler only)
USE_AMP = True

# ============================================================
#                    REPRODUCIBILITY
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
class HierResNet18Concat(nn.Module):
    def __init__(self, crops, diseases_by_crop):
        super().__init__()
        self.crops            = crops
        self.diseases_by_crop = diseases_by_crop

        try:
            backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            print("Loaded ResNet18 pretrained weights.")
        except Exception:
            print("Offline mode: loading local ResNet-18 weights.")
            backbone = models.resnet18(weights=None)
            local_path = "/home/nalwangar/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
            backbone.load_state_dict(torch.load(local_path, map_location="cpu"))

        for name, p in backbone.named_parameters():
            p.requires_grad = ("layer3" in name) or ("layer4" in name)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.crop_head  = nn.Linear(in_features, len(crops))
        self.crop_names = list(crops)
        self.heads      = nn.ModuleList([
            nn.Linear(in_features, len(diseases_by_crop[c]))
            for c in self.crop_names
        ])

        self.crop_slices = {}
        start = 0
        for ci, crop in enumerate(crops):
            n_dis = len(diseases_by_crop[crop])
            self.crop_slices[ci] = (start, start + n_dis)
            start += n_dis
        self.total_diseases = start

    def forward(self, x):
        feats         = self.backbone(x)
        crop_logits   = self.crop_head(feats)
        concat_logits = torch.cat([head(feats) for head in self.heads], dim=1)
        return crop_logits, concat_logits


# ============================================================
#                       DATASET
# ============================================================
class HierDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.t     = transform

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


def make_eval_transform():
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
    """Randomly subsample any global class with > cap examples down to cap."""
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


def compute_loss_weights(train_items, crops, diseases_by_crop, device):
    """
    Returns:
      crop_w : tensor [n_crops]              -- crop-head class weights
      dis_w  : list of tensors, one per crop -- per-crop disease-head weights
    Inverse-frequency, normalized so mean weight = 1 within each head.
    """
    n_crops = len(crops)
    crop_counts = torch.zeros(n_crops)
    dis_counts  = [torch.zeros(len(diseases_by_crop[c])) for c in crops]
    for _, ci, di in train_items:
        crop_counts[ci] += 1
        dis_counts[ci][di] += 1

    # Crop weights: inverse frequency, mean-normalised
    crop_w = crop_counts.sum() / (n_crops * crop_counts.clamp(min=1))
    crop_w = crop_w / crop_w.mean()

    # Per-crop disease weights
    dis_w = []
    for ci in range(n_crops):
        c = dis_counts[ci]
        n = len(c)
        if n == 0 or c.sum() == 0:
            dis_w.append(torch.ones(max(n, 1), device=device))
            continue
        w = c.sum() / (n * c.clamp(min=1))
        w = w / w.mean()
        dis_w.append(w.to(device))

    return crop_w.to(device), dis_w


# ============================================================
#                   CONFUSION MATRIX UTILS
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
    axes[1].plot(df["epoch"], df["lr_heads"],    label="LR heads",    linewidth=1.5)
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
#                  VECTORIZED DISEASE LOSS
# ============================================================
def disease_loss_vectorized(out_dis, yc, yd, model, dis_weights):
    """
    out_dis     : [B, total_diseases]
    yc, yd      : [B] long tensors (true crop, true within-crop disease)
    dis_weights : list[Tensor] one per crop (or None for unweighted)
    Returns scalar loss = mean over batch of per-sample CE on the
    correct crop's slice.
    """
    B      = out_dis.size(0)
    device = out_dis.device
    losses = out_dis.new_zeros(B)

    # Group samples by their true crop so we can vectorise per crop
    yc_cpu = yc.detach().cpu().tolist()
    by_crop = defaultdict(list)
    for i, ci in enumerate(yc_cpu):
        by_crop[ci].append(i)

    for ci, idxs in by_crop.items():
        start, end = model.crop_slices[ci]
        idx_t      = torch.tensor(idxs, device=device, dtype=torch.long)
        slice_logits = out_dis[idx_t, start:end]              # [k, n_dis_ci]
        targets      = yd[idx_t]                              # [k]
        w = dis_weights[ci] if dis_weights is not None else None
        ce = F.cross_entropy(slice_logits, targets,
                             weight=w, reduction="none")
        losses[idx_t] = ce

    return losses.mean()


# ============================================================
#                       EVALUATION
# ============================================================
def evaluate(model, model_path, val_loader, device, fold_dir, crops, global_labels):
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
            imgs = imgs.to(device)
            yc   = yc.to(device)
            yd   = yd.to(device)
            yg   = yg.to(device)

            out_crop, out_dis = model(imgs)
            pred_c = out_crop.argmax(1)

            for i in range(imgs.size(0)):
                ci_pred = int(pred_c[i].item())
                ci_true = int(yc[i].item())

                start_pred, end_pred = model.crop_slices[ci_pred]
                local_pred           = int(out_dis[i, start_pred:end_pred].argmax().item())
                global_pred_pred     = items_global_map[(ci_pred, local_pred)]

                start_true, end_true = model.crop_slices[ci_true]
                local_pred_true      = int(out_dis[i, start_true:end_true].argmax().item())
                global_pred_true     = items_global_map[(ci_true, local_pred_true)]

                true_crop.append(ci_true)
                pred_crop.append(ci_pred)
                true_global.append(int(yg[i].item()))
                pred_global.append(global_pred_pred)
                pred_global_true_crop.append(global_pred_true)

    crop_acc              = accuracy_score(true_crop,   pred_crop)
    disease_acc_pred_crop = accuracy_score(true_global, pred_global)
    disease_acc_true_crop = accuracy_score(true_global, pred_global_true_crop)

    cm_crop = confusion_matrix(true_crop, pred_crop, labels=range(len(crops)))
    save_confusion_matrices(cm_crop, crops, fold_dir, "cm_crop", "Crop CM")

    cm_dis = confusion_matrix(true_global, pred_global,
                               labels=range(len(global_labels)))
    save_confusion_matrices(cm_dis, global_labels, fold_dir,
                            "cm_disease_pred_crop", "Disease CM (Pred Crop)")

    cm_dis_oracle = confusion_matrix(true_global, pred_global_true_crop,
                                     labels=range(len(global_labels)))
    save_confusion_matrices(cm_dis_oracle, global_labels, fold_dir,
                            "cm_disease_true_crop", "Disease CM (Oracle Crop)")

    f1_crop_macro     = f1_score(true_crop,   pred_crop,             average='macro',    zero_division=0)
    f1_crop_weighted  = f1_score(true_crop,   pred_crop,             average='weighted', zero_division=0)
    f1_dis_pred_macro = f1_score(true_global, pred_global,           average='macro',    zero_division=0)
    f1_dis_pred_wt    = f1_score(true_global, pred_global,           average='weighted', zero_division=0)
    f1_dis_true_macro = f1_score(true_global, pred_global_true_crop, average='macro',    zero_division=0)
    f1_dis_true_wt    = f1_score(true_global, pred_global_true_crop, average='weighted', zero_division=0)

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
               crops, global_labels, crop_w, dis_w):
    # Loss criterions: weighted if requested
    if USE_LOSS_WEIGHTS:
        criterion_crop = nn.CrossEntropyLoss(weight=crop_w)
    else:
        criterion_crop = nn.CrossEntropyLoss()
    criterion_val_crop = nn.CrossEntropyLoss()  # unweighted for clean val loss

    backbone_params     = [
        p for n, p in model.backbone.named_parameters()
        if p.requires_grad and ('layer3' in n or 'layer4' in n)
    ]
    crop_head_params    = list(model.crop_head.parameters())
    disease_head_params = list(model.heads.parameters())

    optimizer = Adam([
        {'params': backbone_params,     'lr': 1e-4},
        {'params': crop_head_params,    'lr': 1e-3},
        {'params': disease_head_params, 'lr': 1e-3},
    ], weight_decay=1e-4)

    scheduler  = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_loss  = float("inf")
    wait       = 0
    model_path = fold_dir / "best_model.pth"
    epoch_log  = []

    train_dis_w = dis_w if USE_LOSS_WEIGHTS else None

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        total_train = 0.0
        n_batches   = 0
        for batch in train_loader:
            if batch is None:
                continue
            imgs, yc, yd, yg = batch
            imgs = imgs.to(device)
            yc   = yc.to(device)
            yd   = yd.to(device)

            optimizer.zero_grad()
            out_crop, out_dis = model(imgs)
            loss_crop = criterion_crop(out_crop, yc)
            loss_dis  = disease_loss_vectorized(
                out_dis, yc, yd, model, train_dis_w
            )
            loss      = loss_crop + loss_dis
            loss.backward()
            optimizer.step()
            total_train += loss.item()
            n_batches   += 1
        train_loss_avg = total_train / max(n_batches, 1)

        # ── Validate (unweighted, for fair comparison across runs) ─────────────
        model.eval()
        total_val = 0.0
        n_val     = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                imgs, yc, yd, yg = batch
                imgs = imgs.to(device)
                yc   = yc.to(device)
                yd   = yd.to(device)
                out_crop, out_dis = model(imgs)
                loss_crop_val = criterion_val_crop(out_crop, yc)
                loss_dis_val  = disease_loss_vectorized(
                    out_dis, yc, yd, model, None  # unweighted on val
                )
                total_val += (loss_crop_val + loss_dis_val).item()
                n_val     += 1
        val_loss_avg = total_val / max(n_val, 1)

        scheduler.step(val_loss_avg)

        lr_bb    = optimizer.param_groups[0]['lr']
        lr_heads = optimizer.param_groups[1]['lr']

        print(f"[Fold {fold}] Epoch {epoch:02d} | "
              f"Train: {train_loss_avg:.4f} | Val: {val_loss_avg:.4f} | "
              f"LR backbone: {lr_bb:.2e} | LR heads: {lr_heads:.2e}")

        epoch_log.append({
            "epoch":       epoch,
            "train_loss":  train_loss_avg,
            "val_loss":    val_loss_avg,
            "lr_backbone": lr_bb,
            "lr_heads":    lr_heads,
        })

        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            wait = 0
            torch.save(model.state_dict(), model_path)
            print("   Best model updated!")
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping!")
                break

    plot_training_curves(epoch_log, fold_dir)

    eval_summary = evaluate(
        model, model_path, val_loader, device, fold_dir, crops, global_labels
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
    # Dataset path: prefer $DATASET_PATH (local scratch, set by SLURM), else deepstore.
    DATASET = os.environ.get(
        "DATASET_PATH",
        "/deepstore/datasets/dmb/ComputerVision/biology/plantvillage-hier",
    )
    SAVE_ROOT = "/home/nalwangar/again/logs_VVh"
    os.makedirs(SAVE_ROOT, exist_ok=True)
    print(f"Dataset path: {DATASET}", flush=True)

    crops, diseases_by_crop, items = build_index(DATASET)

    with open(f"{SAVE_ROOT}/label_maps.json", "w") as f:
        json.dump({"crops": crops, "diseases_within_crop": diseases_by_crop}, f, indent=4)

    global items_global_map
    global_index = {}
    labels = []
    idx = 0
    for ci, crop in enumerate(crops):
        for di, dis in enumerate(diseases_by_crop[crop]):
            global_index[(ci, di)] = idx
            labels.append(f"{crop}:{dis}")
            idx += 1

    items_global_map = global_index
    global_labels    = labels

    paths        = [p for p, _, _ in items]
    joint_labels = [items_global_map[(c, d)] for _, c, d in items]

    train_transform = make_train_transform()
    val_transform   = make_eval_transform()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Total joint classes: {len(global_labels)}")
    print(f"Config: BATCH={BATCH_SIZE}, MAJORITY_CAP={MAJORITY_CAP}, "
          f"USE_LOSS_WEIGHTS={USE_LOSS_WEIGHTS}")

    skf          = StratifiedKFold(5, shuffle=True, random_state=SEED)
    fold_results = []

    fold_rng = random.Random(SEED)  # for capping

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, joint_labels), 1):
        print(f"\n===== FOLD {fold} =====")
        train_items_full = [items[i] for i in train_idx]
        val_items        = [items[i] for i in val_idx]

        # ── Cap majority classes (train only) ──────────────────────────────────
        train_items = cap_majority_classes(
            train_items_full, items_global_map, MAJORITY_CAP, fold_rng
        )

        # ── Class-weight tensors (from capped train set) ───────────────────────
        crop_w, dis_w = compute_loss_weights(
            train_items, crops, diseases_by_crop, device
        )

        # ── Weighted sampler (from capped train set) ───────────────────────────
        sampler = make_weighted_sampler(train_items, items_global_map, g)

        train_loader = DataLoader(
            HierDataset(train_items, train_transform),
            batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS,
            collate_fn=safe_collate, worker_init_fn=seed_worker, generator=g
        )
        val_loader = DataLoader(
            HierDataset(val_items, val_transform),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
            collate_fn=safe_collate, worker_init_fn=seed_worker, generator=g
        )

        model    = HierResNet18Concat(crops, diseases_by_crop).to(device)
        fold_dir = Path(SAVE_ROOT) / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Persist the per-fold class distribution after capping (for audit)
        cls_counts = Counter(items_global_map[(c, d)] for _, c, d in train_items)
        pd.DataFrame(
            [{"global_class_id": k, "label": global_labels[k], "count": v}
             for k, v in sorted(cls_counts.items())]
        ).to_csv(fold_dir / "train_class_counts.csv", index=False)

        summary = train_fold(
            fold, model, train_loader, val_loader, device,
            fold_dir, crops, global_labels, crop_w, dis_w
        )
        fold_results.append(summary)

    # ── Summary: mean, std, min, max ──────────────────────────────────────────
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
    print("\n=== TRAINING COMPLETE: HIERARCHICAL RESNET-18 v4 ===")


if __name__ == "__main__":
    main()
