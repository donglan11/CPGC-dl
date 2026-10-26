"""
CPGC TTA (Test-Time Augmentation)
=====================================
对测试集每张图做 5 次随机增强，预测取平均投票
"""
import os, sys, numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from CPGC实验.CPGC_3种子训练 import CPGC

DATA_DIR = "D:/EMA/GC10_dataset"
MODEL_PATH = "D:/EMA/CPGC实验/CPGC_10seed_seed42_best.pth"   # 2026-09-23: 原 CPGC_seed42_best.pth 为 8/12 过期权重，已停用
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 224
NUM_CLASSES = 10
TTA_TIMES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, TTA: {TTA_TIMES}x")

model = CPGC(num_classes=NUM_CLASSES).to(device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()
print(f"Loaded: {MODEL_PATH}")

tta_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
standard_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 标准推理
test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=standard_tf)
class_names = test_set.classes
all_preds, all_labels = [], []
for idx in range(len(test_set)):
    img, lbl = test_set[idx]
    with torch.no_grad():
        with autocast():
            _, _, _, final_logits = model(img.unsqueeze(0).to(device), epoch=100)
        _, predicted = final_logits.max(1)
        all_preds.append(predicted.cpu().item())
        all_labels.append(lbl)
std_acc = 100.0 * (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
print(f"Standard: {std_acc:.2f}%")

# TTA 推理
raw_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"))
all_paths = [s[0] for s in raw_dataset.samples]
all_labels_arr = [s[1] for s in raw_dataset.samples]

print(f"TTA inference ({TTA_TIMES}x)...")
all_tta_preds = []
for idx in range(len(all_paths)):
    img = Image.open(all_paths[idx]).convert("RGB")
    batch_preds = []
    for _ in range(TTA_TIMES):
        aug_img = tta_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            with autocast():
                _, _, _, final_logits = model(aug_img, epoch=100)
            _, predicted = final_logits.max(1)
            batch_preds.append(predicted.cpu().item())
    voted = np.bincount(batch_preds).argmax()
    all_tta_preds.append(voted)
    if (idx + 1) % 100 == 0:
        print(f"  {idx+1}/{len(all_paths)}")

tta_acc = 100.0 * (np.array(all_tta_preds) == np.array(all_labels_arr)).sum() / len(all_labels_arr)
print(f"TTA: {tta_acc:.2f}%")
print(f"Gain: +{tta_acc - std_acc:.2f}%")

csv_path = os.path.join(OUTPUT_DIR, "CPGC_TTA_结果.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write(f"standard,{std_acc:.2f}\n")
    f.write(f"tta,{tta_acc:.2f}\n")
    f.write(f"gain,{tta_acc - std_acc:.2f}\n")
print(f"Saved: {csv_path}")