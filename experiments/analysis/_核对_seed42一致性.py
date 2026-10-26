"""
种子42 权重一致性对照 — 解释 91.63 vs 91.85 的 1 张图差异
==========================================================
背景：`待检查_两个口径残留.md` 问题二第 3 点提出
      「CPGC seed42 逐类 91.63 vs 训练 best 91.85，差 1 张图，需核查推理调用」。

本脚本用**同一个权重** `CPGC_10seed_seed42_best.pth` 跑三组推理设置，
定位差异到底来自哪里：
  A) cudnn.deterministic=True  → `逐类分析_10种子.py` 的设置
  B) cudnn.deterministic=False → `混淆矩阵_可视化.py` 的设置
  C) A + 逐类累加（检查是否只是"逐类求和 = 总体"的统计口径问题）

结论写回控制台，不回写任何数据文件。
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = "D:/EMA"
CPGC_DIR = os.path.join(ROOT, "CPGC实验")
BASE_DIR = os.path.join(ROOT, "ResNet50基线实验")
sys.path.insert(0, ROOT)
import importlib
_m = importlib.import_module("CPGC实验.CPGC_3种子训练")
_m.DINO_WEIGHTS_PATH = os.path.join(ROOT, "DINO预训练实验", "dino_resnet50_pretrain.pth")
CPGC = _m.CPGC

SEED = 42
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
print(f"Test set: {N} images, classes: {ds.classes}\n")


def infer(path, is_cpgc, deterministic):
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if is_cpgc:
        model = CPGC(num_classes=NUM_CLASSES).to(device)
    else:
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model = model.to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if is_cpgc:
                with autocast():
                    _, _, _, logits = model(x, epoch=100)
            else:
                # 基线 Conv 权重为 fp32，走 autocast 会 fp16/fp32 冲突 → 直接 fp32 前向
                logits = model(x)
            preds.append(logits.max(1)[1].cpu().numpy())
            labels.append(y.numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)

    per_class = np.array([100.0 * (preds[labels == c] == c).sum() / (labels == c).sum()
                          for c in range(NUM_CLASSES)])
    overall = 100.0 * (preds == labels).sum() / len(labels)
    correct = int((preds == labels).sum())
    del model
    torch.cuda.empty_cache()
    return correct, overall, per_class


res = {}
for tag, det in [("A_deterministic_True", True), ("B_deterministic_False", False)]:
    c, o, pc = infer(os.path.join(CPGC_DIR, f"CPGC_10seed_seed{SEED}_best.pth"), True, det)
    res[tag] = (c, o)
    print(f"[{tag:<22}] CPGC  correct={c}/{N}  overall={o:.2f}%")
    print("      per-class: " + "  ".join(f"{n[:6]}={v:.1f}" for n, v in zip(ds.classes, pc)))

bc, bo, _ = infer(os.path.join(BASE_DIR, f"ResNet50_seed{SEED}_best.pth"), False, True)
print(f"\n[baseline deterministic ] correct={bc}/{N}  overall={bo:.2f}%")
print(f"（CSV 口径 baseline_mean seed42 应为 90.56 = 422/466，CPGC seed42 应为 91.85 = 428/466）")

a, b = res["A_deterministic_True"], res["B_deterministic_False"]
print("\n" + "=" * 62)
print(f"A / B 一致？ {'是（差异与 cudnn 无关）' if a[0] == b[0] else f'否，A={a[0]} B={b[0]}，差 {abs(a[0]-b[0])} 张'}")
print("=" * 62)
