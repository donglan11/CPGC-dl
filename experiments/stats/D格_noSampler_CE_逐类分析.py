"""
D 格（CPGC 关 sampler + 纯 CE）逐类精度分析（3 种子）
====================================================
加载 CPGC_noSampler_seed{42,123,456}_best.pth，逐类算 Acc mean±std，
用于对比 CPGC 主结果（开 sampler）的少数类表现，验证"关 sampler 是否伤少数类"。
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
_m = importlib.import_module("CPGC实验.CPGC_noSampler_CE_3种子训练")
_m.DINO_WEIGHTS_PATH = DINO_W
CPGC = _m.CPGC

SEEDS = [42, 123, 456]
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
    loader, class_names = build_loader()
    print(f"Device: {device}  Test: {len(loader.dataset)} images  Classes: {len(class_names)}")

    cls_list, all_list = [], []
    for seed in SEEDS:
        cp = os.path.join(CPGC_DIR, f"CPGC_noSampler_seed{seed}_best.pth")
        cm = CPGC(num_classes=NUM_CLASSES).to(device)
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
    print(f"{'Class':<18}{'D格(关sampler+CE)':>24}")
    print("-" * 70)
    for i, c in enumerate(class_names):
        print(f"{c:<18}{m[i]:>8.2f} ± {s[i]:<5.2f}")
    print("-" * 70)
    print(f"{'Overall':<18}{overall_mean:>8.2f} ± {overall_std:<5.2f}")
    print("=" * 70)

    out = os.path.join(CPGC_DIR, "D格_noSampler_CE_逐类分析.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "mean", "std"])
        for i, c in enumerate(class_names):
            w.writerow([c, round(m[i], 2), round(s[i], 2)])
        w.writerow(["overall", round(overall_mean, 2), round(overall_std, 2)])
    print(f"Saved: {out}")
