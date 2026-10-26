"""
混淆矩阵可视化 — CPGC vs Baseline
=====================================
"""
import os, sys, numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = "D:/EMA/GC10_dataset"
CPGC_MODEL_PATH = "D:/EMA/CPGC实验/CPGC_10seed_seed42_best.pth"   # 2026-09-23: 原 CPGC_seed42_best.pth 为 8/12 过期权重（复现 91.63 而非 91.85），已停用
BASELINE_MODEL_PATH = "D:/EMA/ResNet50基线实验/ResNet50_seed42_best.pth"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_tf)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
class_names = test_set.classes
class_names_short = [c[:8] for c in class_names]

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            with autocast():
                _, _, _, logits = model(images, epoch=100)
            _, predicted = logits.max(1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(lbls.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)

# CPGC
print("Loading CPGC...")
from CPGC_3种子训练 import CPGC
m1 = CPGC(num_classes=NUM_CLASSES).to(device)
m1.load_state_dict(torch.load(CPGC_MODEL_PATH, map_location=device))
preds_cpgc, true_labels = evaluate(m1, test_loader, device)
cm_cpgc = confusion_matrix(true_labels, preds_cpgc)
acc_cpgc = 100.0 * cm_cpgc.diagonal() / cm_cpgc.sum(axis=1)

# Baseline
print("Loading Baseline...")
from torchvision.models import resnet50
m2 = resnet50(weights=None)
m2.fc = nn.Linear(2048, NUM_CLASSES)
m2.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=device))
m2 = m2.to(device)
m2.eval()
all_preds = []
with torch.no_grad():
    for images, lbls in test_loader:
        images = images.to(device)
        with autocast():
            logits = m2(images)
        _, predicted = logits.max(1)
        all_preds.append(predicted.cpu().numpy())
preds_base = np.concatenate(all_preds)
cm_base = confusion_matrix(true_labels, preds_base)
acc_base = 100.0 * cm_base.diagonal() / cm_base.sum(axis=1)

def plot_cm(cm, ax, title, class_names):
    """用 matplotlib 画混淆矩阵"""
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=100)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = f"{cm[i,j]}"
            color = "white" if cm_norm[i,j] > 50 else "black"
            ax.text(j, i, val, ha='center', va='center', fontsize=8, color=color)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=7, rotation=90)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('True', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8, label='%')

# 画图
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
plot_cm(cm_base, axes[0], f'Baseline (Overall: {100*(preds_base==true_labels).sum()/len(true_labels):.1f}%)', class_names_short)
plot_cm(cm_cpgc, axes[1], f'CPGC (Overall: {100*(preds_cpgc==true_labels).sum()/len(true_labels):.1f}%)', class_names_short)

fig.suptitle('Confusion Matrix Comparison (seed=42 test set)', fontsize=14)
fig.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "混淆矩阵_对比图.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")

# 打印逐类对比
print(f"\n{'Class':<20} {'Baseline':>8} {'CPGC':>8} {'Diff':>8}")
print("-" * 44)
for c in range(NUM_CLASSES):
    print(f"{class_names[c]:<20} {acc_base[c]:>7.1f}% {acc_cpgc[c]:>7.1f}% {acc_cpgc[c]-acc_base[c]:>+7.1f}%")