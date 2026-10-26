"""
ResNet-50 基线逐类准确率分析
====================
对比 CPGC 的逐类结果，看提升主要来自哪些类。
"""
import os, sys, numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

DATA_DIR = "D:/EMA/GC10_dataset"
MODEL_PATH = "D:/EMA/ResNet50基线实验/ResNet50_seed42_best.pth"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 加载模型
model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model = model.to(device)
model.eval()
print(f"Loaded baseline model: {MODEL_PATH}")

# 数据
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_tf)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
class_names = test_set.classes

# 评估
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        with autocast():
            outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.append(predicted.cpu().numpy())
        all_labels.append(labels.numpy())
preds = np.concatenate(all_preds)
labels = np.concatenate(all_labels)

overall = 100.0 * (preds == labels).sum() / len(labels)
print(f"\n{'='*60}")
print(f"Baseline overall accuracy: {overall:.2f}%")
print(f"{'='*60}")

# 逐类
print(f"\n{'Class':<20} {'Total':>6} {'Correct':>8} {'Acc':>8}")
print("-" * 44)
baseline_data = {}
for c in range(NUM_CLASSES):
    mask = labels == c
    total = mask.sum()
    correct = (preds[mask] == c).sum()
    acc = 100.0 * correct / total
    baseline_data[class_names[c]] = {"total": int(total), "correct": int(correct), "acc": acc}
    print(f"{class_names[c]:<20} {total:>6} {correct:>8} {acc:>7.2f}%")

# 对比 CPGC 结果
print(f"\n{'='*60}")
print("CPGC vs Baseline 逐类对比")
print(f"{'='*60}")
print(f"{'Class':<20} {'Baseline':>10} {'CPGC':>10} {'Diff':>8}")
print("-" * 48)

# CPGC 结果
cpgc_data = {
    "crease": 72.73, "crescent_gap": 93.48, "inclusion": 86.36,
    "oil_spot": 92.68, "punching_hole": 90.91, "rolled_pit": 42.86,
    "silk_spot": 100.00, "waist folding": 93.33, "water_spot": 96.55,
    "welding_line": 78.18,
}

for c in class_names:
    b = baseline_data[c]["acc"]
    d = cpgc_data[c]
    diff = d - b
    marker = "UP" if diff > 1 else ("up" if diff > 0 else "DOWN")
    print(f"{c:<20} {b:>8.2f}% {d:>8.2f}% {diff:>+7.2f}% {marker}")

# 保存 CSV
csv_path = os.path.join(OUTPUT_DIR, "CPGC_vs_Baseline_逐类.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("class,baseline_acc,cpgc_acc,diff\n")
    for c in class_names:
        f.write(f"{c},{baseline_data[c]['acc']:.2f},{cpgc_data[c]},{cpgc_data[c]-baseline_data[c]['acc']:.2f}\n")
print(f"\nSaved: {csv_path}")