"""
逐类精度分析（10 种子，可复现版）
==================================
用 10 个种子的权重分别推理，给出每类的 mean±std，作为论文表3 的权威来源。

背景：2026-09-23 核查发现 `CPGC_seed{42,123,456}_best.pth` 是 8/12 晚一次
被中断的实验留下的**过期权重**，无法复现 `CPGC_3种子结果.csv` 中的数值
（复现值 91.63/88.63/90.99 vs 报告值 91.85/92.27/91.85）。
而 10 种子权重 `CPGC_10seed_seed*_best.pth` 可精确复现，故改用 10 种子做逐类分析。

路径已改为自适应（兼容 D:/EMA 与 D:/EMA）。
"""
import os, sys, csv
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50

# ---- 路径自适应 ----
_PROJECT_ROOTS = ["D:/EMA", "D:/EMA"]
ROOT = next((r for r in _PROJECT_ROOTS if os.path.isdir(r)), "D:/EMA")
DATA_DIR = os.path.join(ROOT, "GC10_dataset")
CPGC_DIR = os.path.join(ROOT, "CPGC实验")
BASE_DIR = os.path.join(ROOT, "ResNet50基线实验")
DINO_W = os.path.join(ROOT, "DINO预训练实验", "dino_resnet50_pretrain.pth")

sys.path.insert(0, ROOT)
import importlib
_m = importlib.import_module("CPGC实验.CPGC_3种子训练")
_m.DINO_WEIGHTS_PATH = DINO_W
CPGC = _m.CPGC

SEEDS = [42, 123, 456, 789, 2024, 7, 314, 999, 55, 1337]
IMG_SIZE, BATCH_SIZE, NUM_CLASSES = 224, 32, 10

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loader():
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=tf)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0), ds.classes


@torch.no_grad()
def per_class_acc(model, loader, is_cpgc):
    preds, labels = [], []
    for images, y in loader:
        images = images.to(device)
        with autocast():
            if is_cpgc:
                _, _, _, logits = model(images, epoch=100)
            else:
                logits = model(images)
        preds.append(logits.max(1)[1].cpu().numpy())
        labels.append(y.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    acc = []
    for c in range(NUM_CLASSES):
        m = labels == c
        acc.append(100.0 * (preds[m] == c).sum() / m.sum())
    overall = 100.0 * (preds == labels).sum() / len(labels)
    return np.array(acc), overall


if __name__ == "__main__":
    loader, class_names = build_loader()
    print(f"Device: {device}  Test: {len(loader.dataset)} images  Classes: {len(class_names)}")

    base_cls, cpgc_cls, base_all, cpgc_all = [], [], [], []

    for seed in SEEDS:
        # 基线
        bp = os.path.join(BASE_DIR, f"ResNet50_seed{seed}_best.pth")
        bm = resnet50(weights=None)
        bm.fc = nn.Linear(bm.fc.in_features, NUM_CLASSES)
        bm.load_state_dict(torch.load(bp, map_location=device))
        bm = bm.to(device).eval()
        bc, bo = per_class_acc(bm, loader, False)
        del bm
        torch.cuda.empty_cache()

        # CPGC
        cp = os.path.join(CPGC_DIR, f"CPGC_10seed_seed{seed}_best.pth")
        cm = CPGC(num_classes=NUM_CLASSES).to(device)
        cm.load_state_dict(torch.load(cp, map_location=device))
        cm.eval()
        cc, co = per_class_acc(cm, loader, True)
        del cm
        torch.cuda.empty_cache()

        base_cls.append(bc); cpgc_cls.append(cc)
        base_all.append(bo); cpgc_all.append(co)
        print(f"seed={seed:<5} baseline {bo:.2f}%   CPGC {co:.2f}%   diff {co-bo:+.2f}")

    base_cls = np.array(base_cls); cpgc_cls = np.array(cpgc_cls)
    bm_m, bm_s = base_cls.mean(0), base_cls.std(0, ddof=1)
    cm_m, cs_s = cpgc_cls.mean(0), cpgc_cls.std(0, ddof=1)
    diff = cm_m - bm_m

    print("\n" + "=" * 78)
    print(f"{'Class':<18}{'Baseline':>16}{'CPGC':>16}{'Diff':>10}")
    print("-" * 78)
    for i, c in enumerate(class_names):
        print(f"{c:<18}{bm_m[i]:>8.2f} ± {bm_s[i]:<5.2f}{cm_m[i]:>8.2f} ± {cs_s[i]:<5.2f}{diff[i]:>+10.2f}")
    print("-" * 78)
    print(f"{'Overall':<18}{np.mean(base_all):>8.2f} ± {np.std(base_all, ddof=1):<5.2f}"
          f"{np.mean(cpgc_all):>8.2f} ± {np.std(cpgc_all, ddof=1):<5.2f}"
          f"{np.mean(cpgc_all)-np.mean(base_all):>+10.2f}")
    print("=" * 78)

    out = os.path.join(CPGC_DIR, "逐类分析_10种子.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "baseline_mean", "baseline_std", "cpgc_mean", "cpgc_std", "diff"])
        for i, c in enumerate(class_names):
            w.writerow([c, round(bm_m[i], 2), round(bm_s[i], 2),
                        round(cm_m[i], 2), round(cs_s[i], 2), round(diff[i], 2)])
        w.writerow(["overall", round(float(np.mean(base_all)), 2), round(float(np.std(base_all, ddof=1)), 2),
                    round(float(np.mean(cpgc_all)), 2), round(float(np.std(cpgc_all, ddof=1)), 2),
                    round(float(np.mean(cpgc_all) - np.mean(base_all)), 2)])
    print(f"Saved: {out}")
