"""
ResNet-50 基线单种子训练 + 保存训练日志
=============================================
ImageNet预训练 ResNet-50 + 线性分类头
用于和 CPGC 训练曲线对比
"""
import os, sys, time, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

SEED = 42
IMG_SIZE = 224
DATA_DIR = "D:/EMA/GC10_dataset"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
EPOCHS = 100
BATCH_SIZE = 32
BALANCE_CLASSES = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ResNet50Baseline(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Seed: {SEED}, Epochs: {EPOCHS}")

    set_seed(SEED)

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_set = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=tf)
    test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_tf)
    num_classes = len(train_set.classes)

    class_counts = [0] * num_classes
    for _, label in train_set.samples:
        class_counts[label] += 1

    if BALANCE_CLASSES:
        sample_weights = [1.0 / class_counts[label] for _, label in train_set.samples]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = ResNet50Baseline(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    # CSV 日志
    log_path = os.path.join(OUTPUT_DIR, "Baseline_training_log_seed42.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "best_acc"])

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            bsz = labels.size(0)

            optimizer.zero_grad()
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item() * bsz
            _, predicted = logits.max(1)
            tr_total += bsz
            tr_correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_loss = tr_loss / tr_total
        train_acc = 100.0 * tr_correct / tr_total

        model.eval()
        te_loss, te_correct, te_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    logits = model(images)
                loss = criterion(logits, labels)
                te_loss += loss.item() * labels.size(0)
                te_total += labels.size(0)
                _, predicted = logits.max(1)
                te_correct += predicted.eq(labels).sum().item()

        test_loss = te_loss / te_total
        test_acc = 100.0 * te_correct / te_total

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(OUTPUT_DIR, f"Baseline_seed{SEED}_best.pth"))

        elapsed = time.time() - t0

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, round(train_loss, 4), round(train_acc, 2),
                             round(test_loss, 4), round(test_acc, 2),
                             round(best_acc, 2)])

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:>3}/{EPOCHS} TrainLoss={train_loss:.3f} TrainAcc={train_acc:.1f}% "
                  f"TestLoss={test_loss:.3f} TestAcc={test_acc:.2f}% Best={best_acc:.2f}% ({elapsed:.1f}s)")

    print(f"\nDone. Best acc: {best_acc:.2f}%")
    print(f"训练日志已保存: {log_path}")


if __name__ == "__main__":
    main()
