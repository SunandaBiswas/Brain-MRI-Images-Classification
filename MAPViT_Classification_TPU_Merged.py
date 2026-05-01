# ============================================================
# 1. INSTALL PACKAGES
# ============================================================
!pip -q install timm grad-cam scipy torchinfo thop fvcore statsmodels h5py

# ============================================================
# 2. IMPORTS
# ============================================================
import os
import gc
import copy
import math
import random
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import h5py
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score, precision_recall_fscore_support,
    f1_score, log_loss, cohen_kappa_score
)
from scipy import stats
import timm

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from torchvision import transforms
from PIL import Image
from torchinfo import summary as ti_summary

warnings.filterwarnings("ignore")

# ============================================================
# 3. TPU SETUP
# ============================================================
def setup_tpu():
    """
    Detect and configure TPU via PyTorch/XLA.
    Falls back to GPU then CPU if TPU is unavailable.
    Returns (device, is_tpu, xm, xla_dist) where xm / xla_dist
    are None when not on TPU.
    """
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp

        device = xm.xla_device()
        print(f"✅ TPU detected  → device: {device}")
        print(f"   XLA devices  : {xm.get_xla_supported_devices()}")
        return device, True, xm, pl

    except ImportError:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"⚠️  torch_xla not found. Using GPU: "
                  f"{torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("⚠️  torch_xla not found and no GPU. Using CPU.")
        return device, False, None, None


DEVICE, IS_TPU, XM, PL = setup_tpu()

# ============================================================
# 4. HYPERPARAMETERS
# ============================================================
IMG_SIZE      = 224
# TPU works best with larger batch sizes (multiple of 8)
BATCH_SIZE    = 64 if IS_TPU else 16
EPOCHS        = 50
PATIENCE      = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-6
NUM_WORKERS   = 4       # TPU pods benefit from more workers

CLASS_LABELS  = ['glioma', 'meningioma', 'notumor', 'pituitary']
LABEL_MAP     = {cls: i for i, cls in enumerate(CLASS_LABELS)}

print(f"\nBatch size  : {BATCH_SIZE}")
print(f"Epochs      : {EPOCHS}")
print(f"Classes     : {CLASS_LABELS}")

# ============================================================
# 5. DATA PATHS — BOTH DATASETS
# ============================================================
BRISC_BASE  = (
    '/kaggle/input/datasets/briscdataset/brisc2025'
    '/brisc2025/brisc2025/classification_task'
)
BRISC_TRAIN = os.path.join(BRISC_BASE, 'train')
BRISC_TEST  = os.path.join(BRISC_BASE, 'test')

PMRAM_BASE  = (
    '/kaggle/input/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset'
    '/PMRAM Bangladeshi Brain Cancer - MRI Dataset'
    '/PMRAM Bangladeshi Brain Cancer - MRI Dataset'
    '/Augmented Data/Augmented'
)

print("\n📂 Checking paths:")
for p in [BRISC_TRAIN, BRISC_TEST, PMRAM_BASE]:
    ok = os.path.exists(p)
    print(f"  {'✅' if ok else '❌'} {p}")
    if ok:
        print(f"     subfolders: {os.listdir(p)}")

# ============================================================
# 6. LABEL RESOLVER
# ============================================================
LABEL_VARIANTS = {}
for lbl in CLASS_LABELS:
    LABEL_VARIANTS[lbl]                   = lbl
    LABEL_VARIANTS[lbl.replace('_', '')] = lbl
    LABEL_VARIANTS[lbl.replace('_', ' ')] = lbl

EXTRA_ALIASES = {
    'normal':     'notumor', 'no tumor':  'notumor',
    'no_tumor':   'notumor', 'notumour':  'notumor',
    'no tumour':  'notumor',
    'glioma':     'glioma',  'meningioma':'meningioma',
    'pituitary':  'pituitary',
}
LABEL_VARIANTS.update(EXTRA_ALIASES)


def resolve_label(folder_name: str):
    cleaned = folder_name.lower().strip().lstrip('0123456789').strip()
    return (LABEL_VARIANTS.get(cleaned) or
            LABEL_VARIANTS.get(cleaned.replace('_', '')) or
            LABEL_VARIANTS.get(cleaned.replace('_', ' ')))


def create_df_from_folder(base_path: str) -> pd.DataFrame:
    records = []
    if not os.path.exists(base_path):
        print(f"  ⚠️  Not found (skipping): {base_path}")
        return pd.DataFrame()
    for root, _, files in os.walk(base_path):
        label = resolve_label(os.path.basename(root))
        if label is None:
            continue
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                records.append({
                    'image_path': os.path.join(root, fname),
                    'label':      label,
                    'source':     base_path,
                })
    return pd.DataFrame(records)

# ============================================================
# 7. LOAD & MERGE DATASETS
# ============================================================
print("\n📊 Loading datasets ...")
brisc_train_df = create_df_from_folder(BRISC_TRAIN)
brisc_test_df  = create_df_from_folder(BRISC_TEST)
pmram_df       = create_df_from_folder(PMRAM_BASE)

for name, df in [('BRISC train', brisc_train_df),
                  ('BRISC test',  brisc_test_df),
                  ('PMRAM',       pmram_df)]:
    print(f"\n  {name}: {len(df)} images")
    if len(df):
        print(df['label'].value_counts().to_string())

all_dfs = [df for df in [brisc_train_df, brisc_test_df, pmram_df]
           if len(df) > 0]
assert len(all_dfs) > 0, "❌ All DataFrames empty! Check your paths."

combined_df = pd.concat(all_dfs, ignore_index=True)
assert 'label' in combined_df.columns
assert len(combined_df) > 0

print(f"\n✅ Combined: {len(combined_df)} images")
print(combined_df['label'].value_counts())

# ============================================================
# 8. TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)
# ============================================================
print("\n📐 Splitting 70 / 15 / 15 ...")
train_val_df, test_df = train_test_split(
    combined_df, test_size=0.15,
    stratify=combined_df['label'], random_state=42
)
train_df, val_df = train_test_split(
    train_val_df, test_size=(0.15 / 0.85),
    stratify=train_val_df['label'], random_state=42
)
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

total = len(combined_df)
print(f"  Train : {len(train_df):5d}  ({len(train_df)/total*100:.1f}%)")
print(f"  Val   : {len(val_df):5d}  ({len(val_df)/total*100:.1f}%)")
print(f"  Test  : {len(test_df):5d}  ({len(test_df)/total*100:.1f}%)")

# ============================================================
# 9. TRANSFORMS
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================================================
# 10. DATASET CLASS
# ============================================================
class BrainTumorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, LABEL_MAP[row['label']]

# ============================================================
# 11. TPU-AWARE DATALOADER FACTORY
# ============================================================
def make_loader(df, transform, shuffle=False, drop_last=False):
    """
    Creates a DataLoader.
    On TPU: pin_memory=False (XLA manages memory differently).
    On GPU/CPU: pin_memory=True for speed.
    """
    return DataLoader(
        BrainTumorDataset(df, transform),
        batch_size  = BATCH_SIZE,
        shuffle     = shuffle,
        drop_last   = drop_last,
        num_workers = NUM_WORKERS,
        pin_memory  = (not IS_TPU),   # ← TPU-specific
    )

# ============================================================
# 12. MODEL ARCHITECTURE — MAP-ViT
# ============================================================
class DensePyramidModule(nn.Module):
    def __init__(self, in_channels: int, pool_sizes: List[int] = [1, 2, 3, 6]):
        super().__init__()
        inter = in_channels // len(pool_sizes)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_channels, inter, 1, bias=False),
                nn.BatchNorm2d(inter),
                nn.ReLU(inplace=True),
            ) for s in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + inter * len(pool_sizes),
                      in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        outs = [x] + [
            F.interpolate(s(x), (h, w), mode='bilinear', align_corners=True)
            for s in self.stages
        ]
        return self.bottleneck(torch.cat(outs, 1))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads,
            dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        n = self.norm1(x)
        x = x + self.attn(n, n, n)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class HybridMobilePPM_ViT(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        self.backbone    = timm.create_model(
            'mobilenetv2_100', pretrained=True, features_only=True
        )
        ch               = self.backbone.feature_info.channels()[-1]
        self.d_ppm       = DensePyramidModule(ch)
        self.transformer = TransformerBlock(dim=ch, heads=8)
        self.classifier  = nn.Sequential(
            nn.LayerNorm(ch),
            nn.Linear(ch, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.d_ppm(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

print("✅ MAP-ViT architecture defined.")

# ============================================================
# 13. MODEL SUMMARY
# ============================================================
_tmp = HybridMobilePPM_ViT(num_classes=len(CLASS_LABELS)).to(DEVICE)
try:
    model_summary = ti_summary(
        _tmp,
        input_size=(1, 3, IMG_SIZE, IMG_SIZE),
        col_names=['input_size', 'output_size', 'num_params'],
        depth=5, verbose=0,
        device=DEVICE
    )
    summary_text = str(model_summary)
except Exception:
    # torchinfo may not support XLA device directly
    _tmp_cpu = HybridMobilePPM_ViT(num_classes=len(CLASS_LABELS))
    model_summary = ti_summary(
        _tmp_cpu,
        input_size=(1, 3, IMG_SIZE, IMG_SIZE),
        col_names=['input_size', 'output_size', 'num_params'],
        depth=5, verbose=0
    )
    summary_text = str(model_summary)
    del _tmp_cpu

del _tmp; gc.collect()

fig = plt.figure(figsize=(18, 21))
fig.suptitle('Layer-wise Architecture – MAP-ViT Classification Model',
             fontsize=14, y=0.995)
plt.text(0.01, 0.97, summary_text, fontsize=8,
         family='monospace', va='top')
plt.axis('off')
plt.tight_layout(rect=[0, 0, 1, 0.985])
plt.savefig('mapvit_model_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: mapvit_model_summary.png")

# ============================================================
# 14. TPU-AWARE TRAINING & EVAL HELPERS
# ============================================================
def tpu_mark_step():
    """Call after optimizer.step() on TPU to flush XLA graph."""
    if IS_TPU and XM is not None:
        XM.mark_step()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = correct = total = 0

    # Wrap loader with TPU parallel loader for performance
    if IS_TPU and PL is not None:
        loader = PL.MpDeviceLoader(loader, device)

    for imgs, labels in loader:
        if not IS_TPU:
            imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()

        if IS_TPU and XM is not None:
            XM.optimizer_step(optimizer)   # TPU-aware optimizer step
        else:
            optimizer.step()

        tpu_mark_step()

        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += labels.size(0)

    return loss_sum / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0

    if IS_TPU and PL is not None:
        loader = PL.MpDeviceLoader(loader, device)

    for imgs, labels in loader:
        if not IS_TPU:
            imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += labels.size(0)

    tpu_mark_step()
    return loss_sum / total, correct / total


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    if IS_TPU and PL is not None:
        loader = PL.MpDeviceLoader(loader, device)

    for imgs, labels in loader:
        if not IS_TPU:
            imgs = imgs.to(device)
        probs = F.softmax(model(imgs), dim=1).cpu().numpy()
        y_true.append(labels.numpy())
        y_pred.append(probs.argmax(1))
        y_prob.append(probs)

    tpu_mark_step()
    return (np.concatenate(y_true),
            np.concatenate(y_pred),
            np.concatenate(y_prob))


def mean_std_ci(x):
    x  = np.asarray(x, float)
    m  = float(np.mean(x))
    s  = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    ci = 1.96 * s / math.sqrt(len(x)) if len(x) > 1 else 0.0
    return m, s, ci


def save_checkpoint(state_dict, path):
    """
    Save state dict to CPU first — XLA tensors must be
    moved off device before pickling.
    """
    cpu_state = {k: v.cpu() for k, v in state_dict.items()}
    torch.save(cpu_state, path)


def load_checkpoint(model, path, device):
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)
    model.to(device)
    return model

# ============================================================
# 15. METRICS HELPER
# ============================================================
def compute_all_metrics(y_true, y_pred, y_prob, class_labels, split_label=''):
    acc   = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    try:
        auc_score = float(roc_auc_score(
            y_true, y_prob, multi_class='ovr', average='macro'))
    except Exception:
        auc_score = float('nan')

    cm = confusion_matrix(y_true, y_pred)
    n  = len(class_labels)
    sens_list, spec_list = [], []
    for i in range(n):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        sens_list.append(float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0)
        spec_list.append(float(TN / (TN + FP)) if (TN + FP) > 0 else 0.0)

    macro_sens = float(np.mean(sens_list))
    macro_spec = float(np.mean(spec_list))

    kappa_label = (
        'Poor' if kappa < 0.0 else 'Slight' if kappa < 0.20 else
        'Fair'  if kappa < 0.4 else 'Moderate' if kappa < 0.60 else
        'Substantial' if kappa < 0.80 else 'Almost Perfect'
    )

    hdr = f'[{split_label}]' if split_label else ''
    print(f'\n  {hdr}')
    print(f'  Accuracy          : {acc:.4f}  ({acc*100:.2f}%)')
    print(f'  AUC (macro OvR)   : {auc_score:.4f}')
    print(f'  Cohen\'s Kappa (κ) : {kappa:.4f}  [{kappa_label}]')
    print(f'\n  {"Class":12s} | {"Sensitivity":>13s} | {"Specificity":>13s}')
    print('  ' + '-' * 44)
    for i, cls in enumerate(class_labels):
        print(f'  {cls:12s} | {sens_list[i]:>13.4f} | {spec_list[i]:>13.4f}')
    print('  ' + '-' * 44)
    print(f'  {"MACRO AVG":12s} | {macro_sens:>13.4f} | {macro_spec:>13.4f}')
    print(f'\n  Classification Report:\n')
    print(classification_report(y_true, y_pred, target_names=class_labels))

    return {
        'accuracy':              acc,
        'auc_macro_ovr':         auc_score,
        'cohen_kappa':           kappa,
        'macro_sensitivity':     macro_sens,
        'macro_specificity':     macro_spec,
        '_sens_list':            sens_list,
        '_spec_list':            spec_list,
    }

# ============================================================
# 16. PLOTTING HELPERS
# ============================================================
def plot_loss_acc_curves(history, fold, plot_dir):
    ep = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ep, history['train_loss'], 'b-o', ms=3, label='Train Loss')
    ax1.plot(ep, history['val_loss'],   'r-s', ms=3, label='Val Loss')
    ax1.set_title(f'Fold {fold} – Loss', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ep, [v*100 for v in history['train_acc']], 'b-o', ms=3, label='Train Acc')
    ax2.plot(ep, [v*100 for v in history['val_acc']],   'r-s', ms=3, label='Val Acc')
    ax2.set_title(f'Fold {fold} – Accuracy', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 105); ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle(f'Fold {fold} – Learning Curves', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(plot_dir, f'fold{fold}_loss_acc.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.show()
    print(f'  ✓ {path}')


def plot_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                linewidths=0.5, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.show()
    print(f'  ✓ {path}')


def plot_roc_curves(y_true, y_prob, title, path):
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (lbl, col) in enumerate(zip(CLASS_LABELS, colors)):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f'{lbl} (AUC={auc(fpr,tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.show()
    print(f'  ✓ {path}')


def plot_sensitivity_specificity(sens, spec, title, path):
    x = np.arange(len(CLASS_LABELS)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, sens, w, label='Sensitivity',
                color='steelblue', alpha=0.85)
    b2 = ax.bar(x + w/2, spec, w, label='Specificity',
                color='darkorange', alpha=0.85)
    for bar in [*b1, *b2]:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS)
    ax.set_ylim(0, 1.15); ax.set_ylabel('Score')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.show()
    print(f'  ✓ {path}')


def plot_side_by_side_confusion(val_true, val_pred, test_true, test_pred,
                                 fold, plot_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, yt, yp, split in zip(axes,
                                   [val_true, test_true],
                                   [val_pred, test_pred],
                                   ['Validation', 'Test']):
        cm = confusion_matrix(yt, yp)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
        ax.set_title(f'Fold {fold} – {split} CM',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout()
    path = os.path.join(plot_dir, f'fold{fold}_cm_val_test.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.show()
    print(f'  ✓ {path}')


def plot_side_by_side_roc(val_true, val_prob, test_true, test_prob,
                           fold, plot_dir):
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, yt, yb, split in zip(axes,
                                   [val_true, test_true],
                                   [val_prob, test_prob],
                                   ['Validation', 'Test']):
        for i, (lbl, col) in enumerate(zip(CLASS_LABELS, colors)):
            fpr, tpr, _ = roc_curve((yt == i).astype(int), yb[:, i])
            ax.plot(fpr, tpr, color=col, lw=2,
                    label=f'{lbl} (AUC={auc(fpr,tpr):.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        ax.set_title(f'Fold {fold} – {split} ROC',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plot_dir, f'fold{fold}_roc_val_test.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.show()
    print(f'  ✓ {path}')


def plot_side_by_side_sens_spec(val_sens, val_spec, test_sens, test_spec,
                                  fold, plot_dir):
    x = np.arange(len(CLASS_LABELS)); w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, sens, spec, split in zip(axes,
                                      [val_sens,  test_sens],
                                      [val_spec,  test_spec],
                                      ['Validation', 'Test']):
        b1 = ax.bar(x - w/2, sens, w, label='Sensitivity',
                    color='steelblue', alpha=0.85)
        b2 = ax.bar(x + w/2, spec, w, label='Specificity',
                    color='darkorange', alpha=0.85)
        for bar in [*b1, *b2]:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}',
                    ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS)
        ax.set_ylim(0, 1.15); ax.set_ylabel('Score')
        ax.set_title(f'Fold {fold} – {split} Sens/Spec',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='lower right'); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plot_dir, f'fold{fold}_sens_spec_val_test.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.show()
    print(f'  ✓ {path}')


def plot_cv_convergence(histories, plot_dir):
    max_ep = max(len(h['train_loss']) for h in histories)

    def pad(vals, n):
        return np.array(list(vals) + [float('nan')] * (n - len(vals)), float)

    tr_loss = np.array([pad(h['train_loss'], max_ep) for h in histories])
    va_loss = np.array([pad(h['val_loss'],   max_ep) for h in histories])
    tr_acc  = np.array([pad(h['train_acc'],  max_ep) for h in histories]) * 100
    va_acc  = np.array([pad(h['val_acc'],    max_ep) for h in histories]) * 100
    ep      = np.arange(1, max_ep + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for arr, label, color in [(tr_loss, 'Train Loss', 'steelblue'),
                               (va_loss, 'Val Loss',   'tomato')]:
        m = np.nanmean(arr, 0); s = np.nanstd(arr, 0)
        ax1.plot(ep, m, '-o', ms=3, color=color, label=label)
        ax1.fill_between(ep, m - s, m + s, color=color, alpha=0.2)
    ax1.set_title('CV Convergence – Loss (mean ± std)',
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(alpha=0.3)

    for arr, label, color in [(tr_acc, 'Train Acc', 'steelblue'),
                               (va_acc, 'Val Acc',   'tomato')]:
        m = np.nanmean(arr, 0); s = np.nanstd(arr, 0)
        ax2.plot(ep, m, '-o', ms=3, color=color, label=label)
        ax2.fill_between(ep, m - s, m + s, color=color, alpha=0.2)
    ax2.set_title('CV Convergence – Accuracy (mean ± std)',
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 105); ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle('Cross-Validation Convergence (mean ± std across folds)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(plot_dir, 'cv_convergence.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.show()
    print(f'  ✓ {path}')


def plot_fold_metrics_summary(fold_metrics, plot_dir):
    folds    = [f'Fold {m["fold"]}' for m in fold_metrics]
    val_accs = [m['val_acc']            for m in fold_metrics]
    tst_accs = [m['test_acc']           for m in fold_metrics]
    val_aucs = [m['val_auc_macro_ovr']  for m in fold_metrics]
    tst_aucs = [m['test_auc_macro_ovr'] for m in fold_metrics]

    x = np.arange(len(folds)); w = 0.20
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, va, ta, ylabel, title_sfx in [
        (ax1, val_accs, tst_accs, 'Accuracy',       'Accuracy'),
        (ax2, val_aucs, tst_aucs, 'AUC (macro OvR)','AUC'),
    ]:
        for k, (data, lbl, col) in enumerate([
            (va, 'Val',  'steelblue'),
            (ta, 'Test', 'tomato')
        ]):
            bars = ax.bar(x + (k - 0.5)*w, data, w,
                          label=lbl, color=col, alpha=0.85)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f'{bar.get_height():.3f}',
                        ha='center', va='bottom', fontsize=8)
            m, s, _ = mean_std_ci(data)
            ax.axhline(m, color=col, linestyle='--', lw=1.5,
                       label=f'Mean {lbl} {m:.3f}±{s:.3f}')
        ax.set_xticks(x); ax.set_xticklabels(folds)
        ax.set_ylim(0, 1.15); ax.set_ylabel(ylabel)
        ax.set_title(f'Per-Fold {title_sfx} (Val vs Test)',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='lower right'); ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Per-Fold Val vs Test Metrics Summary',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(plot_dir, 'fold_metrics_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.show()
    print(f'  ✓ {path}')

# ============================================================
# 17. SAMPLE DATA EVALUATION
# ============================================================
def sample_data_evaluation(model, test_df, device, plot_dir, n_samples=16):
    print('\n' + '='*60)
    print('  SAMPLE DATA EVALUATION')
    print('='*60)

    mean_ = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std_  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    full_loader = make_loader(test_df, eval_transform, shuffle=False)
    all_true, all_pred, all_prob = predict_all(model, full_loader, device)

    per_cls = max(1, n_samples // len(CLASS_LABELS))
    sampled = []
    for cls in CLASS_LABELS:
        idx    = test_df.index[test_df['label'] == cls].tolist()
        chosen = np.random.RandomState(42).choice(
            idx, size=min(per_cls, len(idx)), replace=False
        )
        sampled.extend(chosen.tolist())
    if len(sampled) < n_samples:
        remaining = list(set(test_df.index.tolist()) - set(sampled))
        extra = np.random.RandomState(42).choice(
            remaining, size=n_samples - len(sampled), replace=False
        )
        sampled.extend(extra.tolist())
    sampled = sampled[:n_samples]

    sample_df = test_df.iloc[sampled].reset_index(drop=True)
    sample_ds = BrainTumorDataset(sample_df, transform=eval_transform)

    images_disp, true_labels, pred_labels = [], [], []
    confidences, all_cls_probs = [], []

    model.eval()
    with torch.no_grad():
        for img_t, true_lbl in sample_ds:
            inp    = img_t.unsqueeze(0).to(device)
            logits = model(inp)
            tpu_mark_step()
            probs    = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_lbl = int(probs.argmax())
            conf     = float(probs.max())
            img_disp = (img_t * std_ + mean_).clamp(0, 1).permute(1, 2, 0).numpy()
            images_disp.append(img_disp)
            true_labels.append(int(true_lbl))
            pred_labels.append(pred_lbl)
            confidences.append(conf)
            all_cls_probs.append(probs)

    # Plot 1 — Image grid
    cols = 4; rows = math.ceil(n_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4.5))
    for i, ax in enumerate(axes.flatten()[:n_samples]):
        correct = pred_labels[i] == true_labels[i]
        color   = '#2ecc71' if correct else '#e74c3c'
        ax.imshow(images_disp[i])
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(5)
        ax.set_title(
            f'True : {CLASS_LABELS[true_labels[i]]}\n'
            f'Pred : {CLASS_LABELS[pred_labels[i]]}\n'
            f'Conf : {confidences[i]*100:.1f}%',
            fontsize=10, fontweight='bold', color=color
        )
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flatten()[n_samples:]:
        ax.axis('off')
    fig.suptitle('Sample Test Predictions\nGreen=Correct  |  Red=Wrong',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    path1 = os.path.join(plot_dir, 'sample_image_grid.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight'); plt.show()
    print(f'  ✓ {path1}')

    # Plot 2 — Confidence bars
    colors_bar = ['#2ecc71' if pred_labels[i] == true_labels[i] else '#e74c3c'
                  for i in range(n_samples)]
    x_ticks = np.arange(n_samples)
    fig, ax = plt.subplots(figsize=(max(12, n_samples*0.9), 5))
    bars = ax.bar(x_ticks, [c*100 for c in confidences],
                  color=colors_bar, alpha=0.85, edgecolor='white')
    ax.axhline(50, color='gray', linestyle='--', lw=1, alpha=0.6, label='50%')
    for bar, conf in zip(bars, confidences):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.8,
                f'{conf*100:.1f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [f'S{i+1}\n{CLASS_LABELS[true_labels[i]][:3]}→'
         f'{CLASS_LABELS[pred_labels[i]][:3]}'
         for i in range(n_samples)], fontsize=8
    )
    ax.set_ylim(0, 115); ax.set_ylabel('Confidence (%)')
    ax.set_title('Predicted Confidence per Sample\n(Green=Correct | Red=Wrong)',
                 fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(plot_dir, 'sample_confidence_bars.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight'); plt.show()
    print(f'  ✓ {path2}')

    # Plot 3 — Violin: per-class confidence on full test set
    colors_v = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    fig, ax  = plt.subplots(figsize=(10, 6))
    parts    = ax.violinplot(
        [all_prob[:, i].tolist() for i in range(len(CLASS_LABELS))],
        positions=np.arange(len(CLASS_LABELS)),
        showmeans=True, showmedians=True
    )
    for pc, col in zip(parts['bodies'], colors_v):
        pc.set_facecolor(col); pc.set_alpha(0.7)
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('white')
    ax.set_xticks(np.arange(len(CLASS_LABELS)))
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_ylabel('Model Confidence Score'); ax.set_ylim(0, 1.05)
    ax.set_title('Per-Class Confidence Distribution (Full Test Set)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path3 = os.path.join(plot_dir, 'sample_confidence_distribution.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight'); plt.show()
    print(f'  ✓ {path3}')

    # Plot 4 — Stacked class probabilities
    probs_arr = np.array(all_cls_probs)
    fig, ax   = plt.subplots(figsize=(max(12, n_samples*0.9), 5))
    bottom    = np.zeros(n_samples)
    for i, (cls, col) in enumerate(zip(CLASS_LABELS, colors_v)):
        ax.bar(x_ticks, probs_arr[:, i]*100, bottom=bottom*100,
               label=cls, color=col, alpha=0.85)
        bottom += probs_arr[:, i]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'S{i+1}' for i in range(n_samples)], fontsize=9)
    ax.set_ylim(0, 105); ax.set_ylabel('Class Probability (%)')
    ax.set_title('Stacked Class Probabilities per Sample',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', ncol=4); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path4 = os.path.join(plot_dir, 'sample_stacked_probs.png')
    plt.savefig(path4, dpi=150, bbox_inches='tight'); plt.show()
    print(f'  ✓ {path4}')

    # Summary table
    n_correct = sum(int(pred_labels[i] == true_labels[i])
                    for i in range(n_samples))
    print(f"\n  {'#':>3}  {'True':12s}  {'Predicted':12s}  {'Conf':>7}  {'OK':>4}")
    print('  ' + '-'*48)
    for i in range(n_samples):
        ok = pred_labels[i] == true_labels[i]
        print(f'  {i+1:>3}  '
              f'{CLASS_LABELS[true_labels[i]]:12s}  '
              f'{CLASS_LABELS[pred_labels[i]]:12s}  '
              f'{confidences[i]*100:>6.1f}%  '
              f'  {"✓" if ok else "✗"}')
    print('  ' + '-'*48)
    print(f'  Sample accuracy : {n_correct}/{n_samples} '
          f'({n_correct/n_samples*100:.1f}%)')

    corr_conf  = np.mean([confidences[i] for i in range(n_samples)
                           if pred_labels[i] == true_labels[i]]) \
                 if n_correct > 0 else 0.0
    wrong_conf = np.mean([confidences[i] for i in range(n_samples)
                           if pred_labels[i] != true_labels[i]]) \
                 if n_correct < n_samples else 0.0
    print(f'  Avg conf (correct): {corr_conf*100:.1f}%')
    print(f'  Avg conf (wrong)  : {wrong_conf*100:.1f}%')

    return {'sample_accuracy':   n_correct / n_samples,
            'avg_conf_correct':  corr_conf,
            'avg_conf_wrong':    wrong_conf}

# ============================================================
# 18. GRAD-CAM EXPLAINABILITY
# ============================================================
def predict_mri_with_explainability(image_path, model):
    # Grad-CAM must run on CPU/GPU — move temporarily if on TPU
    if IS_TPU:
        cpu_model = HybridMobilePPM_ViT(num_classes=len(CLASS_LABELS))
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        cpu_model.load_state_dict(cpu_state)
        _dev = torch.device('cpu')
        cam_model = cpu_model.to(_dev).eval()
    else:
        cam_model = model
        _dev      = DEVICE

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_pil      = Image.open(image_path).convert('RGB')
    input_tensor = tf(img_pil).unsqueeze(0).to(_dev)

    with torch.no_grad():
        out       = cam_model(input_tensor)
        probs     = F.softmax(out, dim=1)
        prob, idx = torch.max(probs, 1)
        pred_idx  = idx.item()

    target_layers = [cam_model.d_ppm.bottleneck]
    targets       = [ClassifierOutputTarget(pred_idx)]

    cam_std  = GradCAM(cam_model, target_layers)
    cam_plus = GradCAMPlusPlus(cam_model, target_layers)
    gs_std   = cam_std(input_tensor=input_tensor,  targets=targets)[0]
    gs_plus  = cam_plus(input_tensor=input_tensor, targets=targets)[0]

    img_np   = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    vis_std  = show_cam_on_image(img_np, gs_std,  use_rgb=True)
    vis_plus = show_cam_on_image(img_np, gs_plus, use_rgb=True)
    heatmap  = gs_std.copy()
    heatmap  = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    fig, axes = plt.subplots(1, 5, figsize=(26, 5))
    color = 'green' if prob.item() > 0.8 else 'orange'
    titles = ['Original MRI', 'Grad-CAM Overlay', 'Grad-CAM++ Overlay',
              'Grad-CAM Heatmap',
              f'PREDICTED: {CLASS_LABELS[pred_idx]}\nConf: {prob.item()*100:.2f}%']
    axes[0].imshow(img_np)
    axes[1].imshow(vis_std)
    axes[2].imshow(vis_plus)
    im = axes[3].imshow(heatmap, cmap='jet')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    axes[4].imshow(img_np)
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=12,
                     color=(color if i == 4 else 'black'),
                     fontweight=('bold' if i == 4 else 'normal'))
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('gradcam_result.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================
# 19. 5-FOLD CROSS VALIDATION
# ============================================================
def run_cross_validation(train_df, val_df, test_df,
                         n_splits=5, epochs=EPOCHS,
                         patience=PATIENCE, save_dir='cv_outputs'):

    os.makedirs(save_dir, exist_ok=True)
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    plot_dir = os.path.join(save_dir, 'plots')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    fixed_val_loader = make_loader(val_df,  eval_transform, shuffle=False)
    _test_loader     = make_loader(test_df, eval_transform, shuffle=False)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y   = train_df['label'].values

    fold_histories   = []
    fold_metrics     = []
    fold_predictions = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y), start=1):
        print(f'\n{"="*60}')
        print(f'  FOLD {fold}/{n_splits}  |  '
              f'train={len(tr_idx)}  fold_val={len(va_idx)}  '
              f'fixed_val={len(val_df)}  test={len(test_df)}')
        print('='*60)

        fold_tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_va_df = train_df.iloc[va_idx].reset_index(drop=True)

        _tl = make_loader(fold_tr_df, train_transform,
                          shuffle=True, drop_last=True)
        _vl = make_loader(fold_va_df, eval_transform, shuffle=False)

        _model    = HybridMobilePPM_ViT(num_classes=len(CLASS_LABELS)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(_model.parameters(),
                                lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        hist    = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
        best_fx = float('inf'); best_state = None; no_imp = 0

        for ep in range(1, epochs + 1):
            tl, ta   = train_one_epoch(_model, _tl, criterion, optimizer, DEVICE)
            fvl, fva = eval_one_epoch(_model, _vl,  criterion, DEVICE)
            fxl, fxa = eval_one_epoch(_model, fixed_val_loader, criterion, DEVICE)
            scheduler.step()

            hist['train_loss'].append(tl)
            hist['train_acc'].append(ta)
            hist['val_loss'].append(fvl)
            hist['val_acc'].append(fva)

            # Sync TPU before printing metrics
            if IS_TPU and XM is not None:
                XM.mark_step()

            print(f'  Ep {ep:02d}/{epochs} | '
                  f'tr={tl:.4f}/{ta:.4f} | '
                  f'fv={fvl:.4f}/{fva:.4f} | '
                  f'fx={fxl:.4f}/{fxa:.4f}')

            if fxl < best_fx:
                best_fx    = fxl
                best_state = copy.deepcopy(
                    {k: v.cpu() for k, v in _model.state_dict().items()}
                )
                save_checkpoint(
                    _model.state_dict(),
                    os.path.join(ckpt_dir, f'best_fold{fold}.pth')
                )
                no_imp = 0
                print(f'  💾 Saved (fx_val_loss={best_fx:.4f})')
            else:
                no_imp += 1
                if no_imp >= patience:
                    print(f'  ⛔ Early stop at epoch {ep}')
                    break

        fold_histories.append(hist)

        # Reload best weights
        _model.load_state_dict(best_state)
        _model.to(DEVICE)

        # Predictions
        fv_true, fv_pred, fv_prob = predict_all(_model, _vl,               DEVICE)
        fx_true, fx_pred, fx_prob = predict_all(_model, fixed_val_loader,   DEVICE)
        ts_true, ts_pred, ts_prob = predict_all(_model, _test_loader,       DEVICE)

        print(f'\n── Fold {fold} Fold-Val ──')
        fv_m = compute_all_metrics(fv_true, fv_pred, fv_prob,
                                   CLASS_LABELS, 'Fold-Val')
        print(f'\n── Fold {fold} Fixed-Val ──')
        fx_m = compute_all_metrics(fx_true, fx_pred, fx_prob,
                                   CLASS_LABELS, 'Fixed-Val')
        print(f'\n── Fold {fold} Test ──')
        ts_m = compute_all_metrics(ts_true, ts_pred, ts_prob,
                                   CLASS_LABELS, 'Test')

        # Per-fold plots
        plot_loss_acc_curves(hist, fold, plot_dir)
        plot_side_by_side_confusion(fv_true, fv_pred, ts_true, ts_pred,
                                     fold, plot_dir)
        plot_side_by_side_roc(fv_true, fv_prob, ts_true, ts_prob,
                               fold, plot_dir)
        plot_side_by_side_sens_spec(
            fv_m['_sens_list'], fv_m['_spec_list'],
            ts_m['_sens_list'], ts_m['_spec_list'],
            fold, plot_dir
        )
        plot_confusion_matrix(
            ts_true, ts_pred,
            f'Fold {fold} – Test Confusion Matrix',
            os.path.join(plot_dir, f'fold{fold}_test_cm.png')
        )
        plot_roc_curves(
            ts_true, ts_prob,
            f'Fold {fold} – Test ROC Curves',
            os.path.join(plot_dir, f'fold{fold}_test_roc.png')
        )
        plot_sensitivity_specificity(
            ts_m['_sens_list'], ts_m['_spec_list'],
            f'Fold {fold} – Test Sensitivity/Specificity',
            os.path.join(plot_dir, f'fold{fold}_test_sens_spec.png')
        )

        fold_metrics.append({
            'fold':                fold,
            'best_fixed_val_loss': float(best_fx),
            'val_acc':             fv_m['accuracy'],
            'val_auc_macro_ovr':   fv_m['auc_macro_ovr'],
            'val_kappa':           fv_m['cohen_kappa'],
            'val_macro_sens':      fv_m['macro_sensitivity'],
            'val_macro_spec':      fv_m['macro_specificity'],
            'test_acc':            ts_m['accuracy'],
            'test_auc_macro_ovr':  ts_m['auc_macro_ovr'],
            'test_kappa':          ts_m['cohen_kappa'],
            'test_macro_sens':     ts_m['macro_sensitivity'],
            'test_macro_spec':     ts_m['macro_specificity'],
        })
        fold_predictions.append({
            'fold': fold, 'y_true': ts_true, 'y_pred': ts_pred
        })

        # Free memory
        del _model, _tl, _vl
        gc.collect()
        if IS_TPU and XM is not None:
            XM.mark_step()

    # ── CV Summary ────────────────────────────────────────────
    print('\n' + '='*60)
    print('  CROSS-VALIDATION SUMMARY')
    print('='*60)
    for key, label in [
        ('val_acc',           'Val Accuracy'),
        ('val_auc_macro_ovr', 'Val AUC'),
        ('val_kappa',         'Val Kappa'),
        ('val_macro_sens',    'Val Macro Sensitivity'),
        ('val_macro_spec',    'Val Macro Specificity'),
        ('test_acc',          'Test Accuracy'),
        ('test_auc_macro_ovr','Test AUC'),
        ('test_kappa',        'Test Kappa'),
        ('test_macro_sens',   'Test Macro Sensitivity'),
        ('test_macro_spec',   'Test Macro Specificity'),
    ]:
        m, s, ci = mean_std_ci([fm[key] for fm in fold_metrics])
        print(f'  {label:26s}: {m:.4f} ± {s:.4f}  (95% CI ± {ci:.4f})')

    # McNemar pairwise
    print('\n✅ Pairwise McNemar p-values (exact):')
    k = len(fold_predictions)
    for i in range(k):
        for j in range(i+1, k):
            a   = (fold_predictions[i]['y_pred'] ==
                   fold_predictions[i]['y_true']).astype(int)
            b   = (fold_predictions[j]['y_pred'] ==
                   fold_predictions[i]['y_true']).astype(int)
            d01 = int(np.sum((a==1)&(b==0)))
            d10 = int(np.sum((a==0)&(b==1)))
            dn  = d01 + d10
            if dn > 0:
                cdf_v = sum(math.comb(dn, ii)*(0.5**dn)
                            for ii in range(0, min(d01,d10)+1))
                p = min(1.0, 2.0*cdf_v)
            else:
                p = 1.0
            print(f'   Fold{i+1} vs Fold{j+1} → p={p:.4f} '
                  f'n01={d01} n10={d10} '
                  f'{"✅" if p < 0.05 else ""}')

    # Cochran's Q
    try:
        from statsmodels.stats.contingency_tables import cochrans_q
        yt_ref      = fold_predictions[0]['y_true']
        correctness = np.vstack([
            (fp['y_pred'] == yt_ref).astype(int)
            for fp in fold_predictions
        ]).T
        q_res = cochrans_q(correctness)
        print(f'\n✅ Cochran\'s Q: Q={q_res.statistic:.4f} '
              f'p={q_res.pvalue:.6f} '
              f'→ {"SIGNIFICANT ✅" if q_res.pvalue < 0.05 else "not significant"}')
    except Exception as e:
        print(f'\n⚠️ Cochran\'s Q skipped: {e}')

    # LaTeX table
    df_cv = pd.DataFrame(fold_metrics)
    print('\n✅ LaTeX Table:')
    print('\\hline')
    print('Fold & ACC & AUC & Kappa & Sens & Spec \\\\')
    print('\\hline')
    for _, row in df_cv.iterrows():
        print(f'{int(row.fold)} & {row.test_acc:.4f} & '
              f'{row.test_auc_macro_ovr:.4f} & {row.test_kappa:.4f} & '
              f'{row.test_macro_sens:.4f} & {row.test_macro_spec:.4f} \\\\')
    m_row = df_cv[['test_acc','test_auc_macro_ovr','test_kappa',
                   'test_macro_sens','test_macro_spec']].mean()
    s_row = df_cv[['test_acc','test_auc_macro_ovr','test_kappa',
                   'test_macro_sens','test_macro_spec']].std(ddof=1)
    print('\\hline')
    print(f'Mean & {m_row.test_acc:.4f} & {m_row.test_auc_macro_ovr:.4f} & '
          f'{m_row.test_kappa:.4f} & {m_row.test_macro_sens:.4f} & '
          f'{m_row.test_macro_spec:.4f} \\\\')
    print(f'Std  & {s_row.test_acc:.4f} & {s_row.test_auc_macro_ovr:.4f} & '
          f'{s_row.test_kappa:.4f} & {s_row.test_macro_sens:.4f} & '
          f'{s_row.test_macro_spec:.4f} \\\\')
    print('\\hline')

    best_fold_idx  = int(np.argmax([m['test_acc'] for m in fold_metrics]))
    best_fold_num  = fold_metrics[best_fold_idx]['fold']
    best_ckpt_path = os.path.join(ckpt_dir, f'best_fold{best_fold_num}.pth')
    print(f'\n  Best fold → Fold {best_fold_num} '
          f'(test_acc={fold_metrics[best_fold_idx]["test_acc"]:.4f})')

    df_cv.to_csv(os.path.join(save_dir, 'cv_results.csv'), index=False)
    return fold_histories, fold_metrics, fold_predictions, plot_dir, best_ckpt_path

# ============================================================
# 20. FINAL TEST EVALUATION
# ============================================================
def final_test_evaluation(model, test_df, plot_dir):
    _loader = make_loader(test_df, eval_transform, shuffle=False)
    y_true, y_pred, y_prob = predict_all(model, _loader, DEVICE)

    print('\n' + '='*60)
    print('  FINAL TEST SET EVALUATION  (best fold model)')
    print('='*60)
    metrics = compute_all_metrics(
        y_true, y_pred, y_prob, CLASS_LABELS, 'FINAL TEST'
    )

    plot_confusion_matrix(
        y_true, y_pred,
        'FINAL TEST – Confusion Matrix',
        os.path.join(plot_dir, 'final_test_cm.png')
    )
    plot_roc_curves(
        y_true, y_prob,
        'FINAL TEST – ROC Curves (OvR)',
        os.path.join(plot_dir, 'final_test_roc.png')
    )
    plot_sensitivity_specificity(
        metrics['_sens_list'], metrics['_spec_list'],
        'FINAL TEST – Sensitivity & Specificity',
        os.path.join(plot_dir, 'final_test_sens_spec.png')
    )

    # Bootstrap CI
    rng = np.random.default_rng(42)
    n   = len(y_true)
    accs, f1s, aucs = [], [], []
    for _ in range(2000):
        s  = rng.choice(n, size=n, replace=True)
        yt = y_true[s]; yp = y_pred[s]; yb = y_prob[s]
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average='macro', zero_division=0))
        try:
            aucs.append(roc_auc_score(yt, yb, multi_class='ovr', average='macro'))
        except:
            pass

    def ci95(x):
        return float(np.mean(x)), float(np.percentile(x,2.5)), \
               float(np.percentile(x,97.5))

    am, alo, ahi = ci95(accs)
    fm, flo, fhi = ci95(f1s)
    um, ulo, uhi = ci95(aucs) if aucs else (np.nan,)*3

    print('\n✅ Bootstrap 95% CI (n_boot=2000)')
    print(f'  Accuracy : {am:.4f}  (95% CI: {alo:.4f} – {ahi:.4f})')
    print(f'  Macro-F1 : {fm:.4f}  (95% CI: {flo:.4f} – {fhi:.4f})')
    print(f'  AUC      : {um:.4f}  (95% CI: {ulo:.4f} – {uhi:.4f})')

    # Binomial test
    k  = int((y_true == y_pred).sum())
    p0 = 1.0 / len(CLASS_LABELS)
    from math import lgamma, log, exp
    def lc(n, r): return lgamma(n+1)-lgamma(r+1)-lgamma(n-r+1)
    logs = [lc(n, x)+x*log(p0)+(n-x)*log(1-p0) for x in range(k, n+1)]
    mv   = max(logs)
    pv   = sum(exp(v-mv) for v in logs)*exp(mv)
    print(f'\n✅ Binomial Test (acc > random={p0:.2f})')
    print(f'  Correct: {k}/{n}  Acc={k/n:.4f}  p={pv:.4e}  '
          f'→ {"SIGNIFICANT ✅" if pv < 0.05 else "not significant"}')

    return metrics

# ============================================================
# 21. MODEL COMPLEXITY
# ============================================================
def print_model_complexity():
    print('\n' + '='*50)
    print('  Model Complexity')
    print('='*50)
    try:
        from thop import profile, clever_format
        _m = HybridMobilePPM_ViT(num_classes=len(CLASS_LABELS))
        _m.eval()
        _d = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        flops, params = profile(_m, inputs=(_d,), verbose=False)
        ff, pf = clever_format([flops, params], '%.3f')
        pt  = sum(p.numel() for p in _m.parameters())
        ptr = sum(p.numel() for p in _m.parameters() if p.requires_grad)
        print(f'\n✅ THOP  (computed on CPU)')
        print(f'   Total Params : {pt:,}')
        print(f'   Trainable    : {ptr:,}')
        print(f'   FLOPs        : {ff}')
        del _m; gc.collect()
    except Exception as e:
        print(f'⚠️ THOP skipped: {e}')

    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        _m = HybridMobilePPM_ViT(num_classes=len(CLASS_LABELS))
        _m.eval()
        _d = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        flops_fv = FlopCountAnalysis(_m, _d)
        print(f'\n✅ FVCORE  (computed on CPU)')
        print(parameter_count_table(_m))
        print(f'   Total FLOPs: {flops_fv.total():,}')
        del _m; gc.collect()
    except Exception as e:
        print(f'⚠️ FVCORE skipped: {e}')

# ============================================================
# 22. SAVE MODEL FORMATS
# ============================================================
def save_model_formats(model, class_labels):
    # Always save from CPU — XLA tensors can't be pickled directly
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}

    torch.save(cpu_state, 'brain_tumor_state_dict.pth')
    print('Saved: brain_tumor_state_dict.pth')

    with h5py.File('brain_tumor_model.h5', 'w') as f:
        for k, v in cpu_state.items():
            f.create_dataset(k, data=v.numpy())
    print('Saved: brain_tumor_model.h5')

    torch.save({
        'model_state_dict': cpu_state,
        'class_labels':     class_labels,
        'img_size':         IMG_SIZE,
        'transform_mean':   [0.485, 0.456, 0.406],
        'transform_std':    [0.229, 0.224, 0.225],
        'is_tpu_trained':   IS_TPU,
    }, 'brain_tumor_checkpoint.pth')
    print('Saved: brain_tumor_checkpoint.pth')

# ============================================================
# 23. VISUALISATION — Data overview
# ============================================================
def show_batch(loader, title='Sample Training Batch'):
    imgs, lbls = next(iter(loader))
    imgs = imgs.permute(0, 2, 3, 1).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    imgs = (imgs * std + mean).clip(0, 1)
    n    = min(6, len(imgs))
    fig, axes = plt.subplots(1, n, figsize=(n*3, 4))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i])
        ax.set_title(CLASS_LABELS[lbls[i]], fontsize=11, fontweight='bold')
        ax.axis('off')
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sample_batch.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_split_bars(train_df, val_df, test_df, class_labels):
    def counts(df):
        c = df['label'].value_counts()
        return np.array([c.get(k, 0) for k in class_labels])
    tr = counts(train_df); va = counts(val_df); te = counts(test_df)
    x  = np.arange(len(class_labels)); w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, data, lbl, col in [
        (-w, tr, 'Train', 'steelblue'),
        ( 0, va, 'Val',   'darkorange'),
        ( w, te, 'Test',  'seagreen'),
    ]:
        bars = ax.bar(x + offset, data, width=w, label=lbl, color=col, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 2,
                    str(int(bar.get_height())),
                    ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(class_labels, fontsize=11)
    ax.set_ylabel('Samples')
    ax.set_title('Class Distribution by Split (Merged Dataset)')
    ax.legend(); plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================
# 24. MAIN PIPELINE
# ============================================================
if __name__ == '__main__':

    # Data overview
    _preview_loader = make_loader(train_df, train_transform,
                                   shuffle=True, drop_last=True)
    show_batch(_preview_loader)
    plot_split_bars(train_df, val_df, test_df, CLASS_LABELS)
    del _preview_loader; gc.collect()

    # Step 1 — 5-Fold CV
    print('\n[1/6] Running 5-Fold Cross-Validation ...')
    histories, fold_metrics, fold_predictions, plot_dir, best_ckpt = \
        run_cross_validation(
            train_df, val_df, test_df,
            n_splits=5, epochs=EPOCHS, patience=PATIENCE,
            save_dir='cv_outputs'
        )

    # Step 2 — CV summary plots
    print('\n[2/6] CV summary plots ...')
    plot_cv_convergence(histories, plot_dir)
    plot_fold_metrics_summary(fold_metrics, plot_dir)

    # Step 3 — Load best model
    print(f'\n[3/6] Loading best fold model: {best_ckpt}')
    best_model = HybridMobilePPM_ViT(num_classes=len(CLASS_LABELS))
    best_model = load_checkpoint(best_model, best_ckpt, DEVICE)
    best_model.eval()

    # Step 4 — Final test evaluation
    print('\n[4/6] Final test evaluation ...')
    final_metrics = final_test_evaluation(best_model, test_df, plot_dir)

    # Step 5 — Grad-CAM  (runs on CPU copy when TPU)
    print('\n[5/6] Grad-CAM explainability ...')
    for _ in range(3):
        row = test_df.sample(1).iloc[0]
        print(f'  📌 True label: {row["label"]}')
        predict_mri_with_explainability(row['image_path'], best_model)

    # Step 6 — Sample data evaluation
    print('\n[6/6] Sample data evaluation ...')
    sample_metrics = sample_data_evaluation(
        best_model, test_df, DEVICE, plot_dir, n_samples=16
    )

    # Model complexity (always on CPU)
    print_model_complexity()

    # Save model
    save_model_formats(best_model, CLASS_LABELS)

    # Final CSV
    pd.DataFrame(fold_metrics).to_csv(
        'cv_outputs/per_fold_results.csv', index=False
    )
    print('\n✅ Results → cv_outputs/per_fold_results.csv')
    print(f'✅ Plots   → {plot_dir}/')
    print(f'\n🎉 Pipeline complete!  (ran on {"TPU" if IS_TPU else "GPU/CPU"})')
