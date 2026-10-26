# -*- coding: utf-8 -*-
"""关 sampler + BS（92.08±0.69）逐类精度分析（10 种子）
====================================================
加载 CPGC_BS_no_sampler_seed{N}_best.pth（10 种子），逐类算 Acc mean±std。
重点坐实最难类 crease / rolled_pit 在 10 种子下的稳定性（3 种子时
crease=69.70、rolled_pit=57.14±0.00）。
注意：BS 版 CPGC 需传 class_prior 实例化，否则 log_prior buffer 加载报错。
"""
import os, sys, csv
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = "D:/EMA"
DATA_DIR = os.path.join(ROOT, "GC10_dataset")
CPGC_DIR = os.path.join(ROOT, "CPGC实验")
DINO_W = os.path.join(ROOT, "DINO预训练实验", "dino_resnet50_pretrain.pth")

sys.path.insert(0, ROOT)
import importlib
_m = importlib.import_module("CPGC实验.CPGC_BS_no_sampler_10种子补跑")
_m.DINO_WEIGHTS_PATH = DINO_W
CPGC = _m.CPGC

SEEDS = [42, 123, 456, 789, 2024, 7, 314, 999, 55, 1337]
IMG_SIZE, BATCH_SIZE, NUM_CLASSES = 224, 32, 10

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build():
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=tf)
    test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=tf)
    class_counts = [0] * NUM_CLASSES
    for _, label in train_ds.samples:
        class_counts[label] += 1
    total = sum(class_counts)
    class_prior = torch.tensor([c / total for c in class_counts], dtype=torch.float32)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return loader, test_ds.classes, class_prior


@torch.no_grad()
def per_class_acc(model, loader):
    preds, labels = [], []
    for images, y in loader:
        images = images.to(device)
        with autocast():
            _, _, _, logits = model(images, epoch=100)
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
    loader, class_names, class_prior = build()
    print(f"Device: {device}  Test: {len(loader.dataset)} images  Classes: {len(class_names)}")

    cls_list, all_list = [], []
    for seed in SEEDS:
        cp = os.path.join(CPGC_DIR, f"CPGC_BS_no_sampler_seed{seed}_best.pth")
        cm = CPGC(num_classes=NUM_CLASSES, class_prior=class_prior).to(device)
        cm.load_state_dict(torch.load(cp, map_location=device))
        cm.eval()
        cc, co = per_class_acc(cm, loader)
        del cm
        torch.cuda.empty_cache()
        cls_list.append(cc)
        all_list.append(co)
        print(f"seed={seed}  overall {co:.2f}%")

    cls_arr = np.array(cls_list)
    m = cls_arr.mean(0)
    s = cls_arr.std(0, ddof=1)
    overall_mean = float(np.mean(all_list))
    overall_std = float(np.std(all_list, ddof=1))

    print("\n" + "=" * 70)
    print(f"{'Class':<18}{'关sampler+BS(10种子)':>28}")
    print("-" * 70)
    for i, c in enumerate(class_names):
        print(f"{c:<18}{m[i]:>8.2f} ± {s[i]:<5.2f}")
    print("-" * 70)
    print(f"{'Overall':<18}{overall_mean:>8.2f} ± {overall_std:<5.2f}")
    print("=" * 70)

    out = os.path.join(CPGC_DIR, "关sampler_BS_逐类分析_10种子.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "mean", "std"])
        for i, c in enumerate(class_names):
            w.writerow([c, round(m[i], 2), round(s[i], 2)])
        w.writerow(["overall", round(overall_mean, 2), round(overall_std, 2)])
    print(f"Saved: {out}")
