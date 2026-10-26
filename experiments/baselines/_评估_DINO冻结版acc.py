# -*- coding: utf-8 -*-
"""从已保存的 DINO 冻结版 pth 重新评估 acc（原进程崩溃时 DINO acc 未落盘）"""
import os, sys
import torch
from torchvision import datasets, transforms
from torch.cuda.amp import autocast

sys.path.insert(0, "D:/EMA/CPGC实验")
_cpgc_mod = __import__("CPGC_10种子训练")
_cpgc_mod.DINO_WEIGHTS_PATH = os.path.join("D:/EMA/DINO预训练实验", "dino_resnet50_pretrain.pth")
load_dino_backbone = _cpgc_mod.load_dino_backbone

IMG_SIZE = 224
DATA_DIR = "D:/EMA/GC10_dataset"
SEEDS = [42, 123, 456]


def build_dino(num_classes):
    import torch.nn as nn
    model = load_dino_backbone()
    model.fc = nn.Linear(2048, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, y in loader:
        images = images.to(device)
        with autocast():
            logits = model(images)
        correct += (logits.max(1)[1].cpu() == y).sum().item()
        total += len(y)
    return 100.0 * correct / total


def main():
    device = torch.device("cuda")
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_tf)
    num_classes = len(test_set.classes)
    loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    accs = {}
    for seed in SEEDS:
        pth = f"D:/EMA/CPGC实验/GC10_frozen_DINO_seed{seed}_best.pth"
        model = build_dino(num_classes).to(device)
        model.load_state_dict(torch.load(pth, map_location=device))
        acc = evaluate(model, loader, device)
        accs[seed] = acc
        print(f"DINO-frozen seed={seed}: {acc:.2f}%", flush=True)

    vals = [accs[s] for s in SEEDS]
    mean = sum(vals) / 3
    import math
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / 2)
    print(f"\nDINO-frozen: {vals[0]:.2f} / {vals[1]:.2f} / {vals[2]:.2f} = {mean:.2f} ± {std:.2f}")


if __name__ == "__main__":
    main()
