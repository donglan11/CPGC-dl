"""
t-SNE 特征空间可视化 — CPGC vs Baseline
=============================================
用已有模型提取测试集 2048-dim 特征，t-SNE 降维到 2D 可视化
"""
import os, sys, numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# 数据
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_tf)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
class_names = test_set.classes
print(f"Test samples: {len(test_set)}, Classes: {class_names}")


def extract_features(model, loader, device):
    """提取 2048-dim 特征"""
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            with autocast():
                feat = model(images)
            all_feats.append(feat.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


# 1. 加载 CPGC 模型 → 提取 backbone 特征
print("Loading CPGC model...")
from CPGC_3种子训练 import CPGC as CPGC_Model
cpgc_model = CPGC_Model(num_classes=NUM_CLASSES).to(device)
state = torch.load(CPGC_MODEL_PATH, map_location=device)
cpgc_model.load_state_dict(state)

# 提取 backbone 特征（去掉投影头和分类头）
class CPGCBackbone(nn.Module):
    def __init__(self, cpgc_model):
        super().__init__()
        self.backbone = cpgc_model.backbone
    def forward(self, x):
        return self.backbone(x)

cpgc_backbone = CPGCBackbone(cpgc_model).to(device)
cpgc_feats, labels = extract_features(cpgc_backbone, test_loader, device)
print(f"CPGC features: {cpgc_feats.shape}")

# 2. 加载 Baseline 模型
print("Loading Baseline model...")
from torchvision.models import resnet50
baseline_model = resnet50(weights=None)
baseline_model.fc = nn.Linear(baseline_model.fc.in_features, NUM_CLASSES)
state = torch.load(BASELINE_MODEL_PATH, map_location=device)
baseline_model.load_state_dict(state)
baseline_model = baseline_model.to(device)

class BaselineBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])
    def forward(self, x):
        return self.features(x).flatten(1)

baseline_backbone = BaselineBackbone(baseline_model).to(device)
baseline_feats, _ = extract_features(baseline_backbone, test_loader, device)
print(f"Baseline features: {baseline_feats.shape}")

# 3. t-SNE 降维
print("Running t-SNE...")
all_feats = np.vstack([baseline_feats, cpgc_feats])
all_labels = np.hstack([labels, labels])
methods = ["Baseline"] * len(labels) + ["CPGC"] * len(labels)

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
embeddings = tsne.fit_transform(all_feats)
print(f"t-SNE done: {embeddings.shape}")

# 4. 画图
colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
n = len(labels)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Baseline
ax = axes[0]
for c in range(NUM_CLASSES):
    mask = labels[:n] == c
    ax.scatter(embeddings[:n, 0][mask], embeddings[:n, 1][mask],
               c=[colors[c]], label=class_names[c], s=20, alpha=0.7)
ax.set_title("ResNet-50 Baseline", fontsize=12)
ax.legend(fontsize=7, loc="best")
ax.axis("off")

# CPGC
ax = axes[1]
for c in range(NUM_CLASSES):
    mask = labels[:n] == c
    ax.scatter(embeddings[n:, 0][mask], embeddings[n:, 1][mask],
               c=[colors[c]], label=class_names[c], s=20, alpha=0.7)
ax.set_title("CPGC (Ours)", fontsize=12)
ax.legend(fontsize=7, loc="best")
ax.axis("off")

fig.suptitle("t-SNE Feature Space Visualization (seed=42 test set)", fontsize=14)
fig.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "tSNE_对比图.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")