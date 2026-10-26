"""
CPGC 逐类准确率分析
====================
基于已有的 CPGC 最佳模型权重，跑测试集，输出每类准确率和混淆矩阵。
"""
import os, sys, numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from CPGC实验.CPGC_3种子训练 import CPGC

DATA_DIR = "D:/EMA/GC10_dataset"
MODEL_PATH = "D:/EMA/CPGC实验/CPGC_10seed_seed42_best.pth"   # 2026-09-23: 原 CPGC_seed42_best.pth 为 8/12 过期权重，已停用；逐类分析请用 逐类分析_10种子.py
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 10

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 加载模型
model = CPGC(num_classes=NUM_CLASSES).to(device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()
print(f"Loaded model: {MODEL_PATH}")

# 数据
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_tf)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
class_names = test_set.classes
print(f"Classes ({len(class_names)}): {class_names}")

# 评估
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        with autocast():
            _, _, _, final_logits = model(images, epoch=100)
        _, predicted = final_logits.max(1)
        all_preds.append(predicted.cpu().numpy())
        all_labels.append(labels.numpy())
preds = np.concatenate(all_preds)
labels = np.concatenate(all_labels)

# 整体精度
overall = 100.0 * (preds == labels).sum() / len(labels)
print(f"\n{'='*60}")
print(f"Overall accuracy: {overall:.2f}%")
print(f"{'='*60}")

# 逐类准确率
print(f"\n{'Class':<20} {'Total':>6} {'Correct':>8} {'Acc':>8}")
print("-" * 44)
per_class = {}
for c in range(NUM_CLASSES):
    mask = labels == c
    total = mask.sum()
    correct = (preds[mask] == c).sum()
    acc = 100.0 * correct / total
    per_class[class_names[c]] = {"total": int(total), "correct": int(correct), "acc": acc}
    print(f"{class_names[c]:<20} {total:>6} {correct:>8} {acc:>7.2f}%")

# 混淆矩阵
cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
for t, p in zip(labels, preds):
    cm[t, p] += 1

print(f"\n{'='*60}")
print("Confusion Matrix (rows=truth, cols=pred)")
print(f"{'='*60}")
print(f"{'':>16}", end="")
for c in class_names:
    print(f"{c[:8]:>8}", end="")
print()
for c in range(NUM_CLASSES):
    print(f"{class_names[c]:>15}", end="")
    for p in range(NUM_CLASSES):
        print(f"{cm[c, p]:>8}", end="")
    print()

# 少数类分析
print(f"\n{'='*60}")
print("Minority class analysis (training set size < 100)")
print(f"{'='*60}")
from collections import Counter
train_set = datasets.ImageFolder(os.path.join(DATA_DIR, "train"))
train_counts = Counter([l for _, l in train_set.samples])
for c in range(NUM_CLASSES):
    if train_counts[c] < 100:
        print(f"  {class_names[c]:<20} train={train_counts[c]:>4}  "
              f"test_acc={per_class[class_names[c]]['acc']:.2f}%  "
              f"correct={per_class[class_names[c]]['correct']}/{per_class[class_names[c]]['total']}")

# 保存 CSV
csv_path = os.path.join(OUTPUT_DIR, "CPGC_逐类分析.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("class,total,correct,acc\n")
    for c in range(NUM_CLASSES):
        f.write(f"{class_names[c]},{per_class[class_names[c]]['total']},"
                f"{per_class[class_names[c]]['correct']},{per_class[class_names[c]]['acc']:.2f}\n")
    f.write(f"\noverall,{len(labels)},{int((preds==labels).sum())},{overall:.2f}\n")
print(f"\nSaved: {csv_path}")