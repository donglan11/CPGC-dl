"""
基线训练预算（epoch）差异量化 — 主结果公平性风险核查
======================================================
背景（从脚本超参核对发现）：
  - `ResNet50基线实验/ResNet50_10种子基线.py`  ：EPOCHS = 30，weights = IMAGENET1K_V1  ← 论文表1 基线 90.27 的来源
  - `CPGC实验/Baseline_训练日志_seed42.py`     ：EPOCHS = 100，weights = IMAGENET1K_V2 ← 训练曲线图里的基线
  - `CPGC实验/CPGC_3种子训练.py` / `CPGC_10种子训练.py`：EPOCHS = 100
  - `DINO预训练实验/DINO_ResNet50_基线.py`     ：EPOCHS = 50

即：论文主比较「基线 30 epoch vs CPGC 100 epoch」，训练预算不对齐。

本脚本用**磁盘上现存的权重**实测两者在 seed42 上的差距，量化"epoch 预算"这一混淆因素：
  A) `ResNet50基线实验/ResNet50_seed42_best.pth`  → 30 epoch 官方基线（表1 口径）
  B) `CPGC实验/Baseline_seed42_best.pth`          → 100 epoch 基线（训练曲线图口径）
两者架构相同（ResNet-50 + fc 10 类），仅训练预算/预训练权重版本不同。

只读，不改任何数据文件。
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50

ROOT = "D:/EMA"
IMG_SIZE, BATCH_SIZE, NUM_CLASSES = 224, 32, 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
ds = datasets.ImageFolder(os.path.join(ROOT, "GC10_dataset", "test"), transform=tf)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
N = len(ds)


@torch.no_grad()
def evaluate(path, tag, note, wrapper=False):
    """wrapper=True 时按 `Baseline_训练日志_seed42.py` 的 ResNet50Baseline 结构加载
    （backbone.* + classifier.*，ImageNet V2 权重、100 epoch）"""
    if not os.path.isfile(path):
        print(f"  [跳过] {tag}: 文件不存在 {path}")
        return None
    if wrapper:
        class ResNet50Baseline(nn.Module):
            """与 `Baseline_训练日志_seed42.py` 中的结构一致：backbone(*)+classifier"""
            def __init__(self, num_classes=NUM_CLASSES):
                super().__init__()
                self.backbone = resnet50(weights=None)
                self.backbone.fc = nn.Identity()
                self.classifier = nn.Linear(2048, num_classes)

            def forward(self, x):
                return self.classifier(self.backbone(x))

        model = ResNet50Baseline()
    else:
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device).eval()
    preds, labels = [], []
    for x, y in loader:
        out = model(x.to(device))
        preds.append(out.max(1)[1].cpu().numpy())
        labels.append(y.numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    acc = 100.0 * (preds == labels).sum() / len(labels)
    print(f"  {tag:<34} {acc:>6.2f}%   ({int((preds==labels).sum())}/{N})   {note}")
    del model
    torch.cuda.empty_cache()
    return acc


print(f"测试集 {N} 张\n")
print("A. 论文表1 口径（30 epoch，ImageNet V1 权重）")
a = evaluate(os.path.join(ROOT, "ResNet50基线实验", "ResNet50_seed42_best.pth"),
             "ResNet50_seed42 (30 ep)", "表1 基线来源，CSV=90.56")

print("\nB. 训练曲线图口径（100 epoch，ImageNet V2 权重）")
b = evaluate(os.path.join(ROOT, "CPGC实验", "Baseline_seed42_best.pth"),
             "Baseline_seed42 (100 ep)", "训练曲线日志终点 best=91.2", wrapper=True)

print("\nC. CPGC（100 epoch）")

c = evaluate(os.path.join(ROOT, "CPGC实验", "CPGC_10seed_seed42_best.pth") if False else None, "", "") \
    if False else None

import sys
sys.path.insert(0, ROOT)
import importlib
_m = importlib.import_module("CPGC实验.CPGC_3种子训练")
_m.DINO_WEIGHTS_PATH = os.path.join(ROOT, "DINO预训练实验", "dino_resnet50_pretrain.pth")
from torch.cuda.amp import autocast
cm = _m.CPGC(num_classes=NUM_CLASSES).to(device)
cm.load_state_dict(torch.load(os.path.join(ROOT, "CPGC实验", "CPGC_10seed_seed42_best.pth"),
                              map_location=device))
cm.eval()
preds, labels = [], []
with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        with autocast():
            _, _, _, logits = cm(x, epoch=100)
        preds.append(logits.max(1)[1].cpu().numpy())
        labels.append(y.numpy())
preds, labels = np.concatenate(preds), np.concatenate(labels)
c = 100.0 * (preds == labels).sum() / len(labels)
print(f"  {'CPGC_seed42 (100 ep)':<34} {c:>6.2f}%   ({int((preds==labels).sum())}/{N})   CSV=91.85")

print("\n" + "=" * 76)
if a is not None and b is not None:
    print(f"同 seed=42：30 epoch 基线 {a:.2f}%  →  100 epoch 基线 {b:.2f}%   差 {b-a:+.2f}")
    print(f"CPGC 相对「30 epoch 基线」的提升：{c-a:+.2f}  （论文表1 口径）")
    print(f"CPGC 相对「100 epoch 基线」的提升：{c-b:+.2f}  （训练预算对齐口径）")
print("=" * 76)
print("说明：B 与 A 同时相差「epoch 数」与「预训练权重版本(V1/V2)」两个因素，")
print("      故本结果只能作为风险量级参考；要拿到无可指摘的对齐基线，需用同一脚本 V1+100ep 重跑。")
