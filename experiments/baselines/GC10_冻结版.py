# -*- coding: utf-8 -*-
"""
GC10 冻结骨干版（缝合"端到端微调 vs 冻结"协议裂缝）
====================================================
目的：回答审稿人 "主实验 GC10 用端到端微调，跨数据集用冻结，为什么不统一？"
      —— 在 GC10 主 benchmark 上跑冻结版，验证 CPGC 在冻结骨干下是否仍领先。

方法：DINO（冻结骨干 + 单 fc）/ CPGC（冻结 DINO 骨干 + 可训头）
协议：与主实验（CPGC_10种子训练.py）完全一致，唯一差异 = backbone 冻结：
      100 ep、BATCH 32、BALANCE_CLASSES=True(WeightedRandomSampler)、
      Adam 1e-3、Cosine、AMP、每 epoch test 评估取 best、num_workers=0。
种子：42/123/456（3 种子，与跨数据集实验同规）。
输出：GC10_frozen_{method}_seed{N}_best.pth + GC10_冻结版结果.csv
"""
import os, sys, time, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import resnet50

SEEDS = [42, 123, 456]
IMG_SIZE = 224
DATA_DIR = "D:/EMA/GC10_dataset"
OUTPUT_DIR = "D:/EMA/CPGC实验"
EPOCHS = 100
BATCH_SIZE = 32
BALANCE_CLASSES = True
LAMBDA_SC = 0.1
LAMBDA_PROTO = 0.3
TEMPERATURE = 0.07
PROTO_EMA = 0.9
PROTO_START = 10

# 复用 CPGC 主协议
sys.path.insert(0, OUTPUT_DIR)
_cpgc_mod = __import__("CPGC_10种子训练")
_cpgc_mod.DINO_WEIGHTS_PATH = os.path.join("D:/EMA/DINO预训练实验", "dino_resnet50_pretrain.pth")
load_dino_backbone = _cpgc_mod.load_dino_backbone
supcon_loss = _cpgc_mod.supcon_loss
CPGC = _cpgc_mod.CPGC


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_backbone(model, bn_eval=True):
    for p in model.parameters():
        p.requires_grad = False
    if bn_eval:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    return model


def build_dino(num_classes):
    model = load_dino_backbone()          # fc=Identity → 2048
    model.fc = nn.Linear(2048, num_classes)
    freeze_backbone(model)
    for p in model.fc.parameters():
        p.requires_grad = True
    return model


def build_cpgc(num_classes):
    model = CPGC(num_classes=num_classes)
    model.backbone = freeze_backbone(model.backbone)
    for p in model.projection.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    model.fusion_alpha.requires_grad = True
    model.log_proto_temp.requires_grad = True
    return model


def set_model_mode(model, is_cpgc, training):
    model.train(training)
    if is_cpgc:
        model.backbone.eval()
    else:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def train_one(seed, train_loader, test_loader, num_classes, device, is_cpgc):
    set_seed(seed)
    model = build_cpgc(num_classes) if is_cpgc else build_dino(num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    best_acc = 0.0
    best_state = {k: v.detach().cpu().clone()
                  for k, v in model.state_dict().items()}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        set_model_mode(model, is_cpgc, training=True)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                if is_cpgc:
                    proj, mlp_logits, proto_scores, final_logits = model(
                        images, labels=labels, epoch=epoch)
                    loss_ce = criterion(final_logits, labels)
                    loss_sc = supcon_loss(proj, labels, TEMPERATURE)
                    loss = loss_ce + LAMBDA_SC * loss_sc
                    if proto_scores is not None:
                        loss = loss + LAMBDA_PROTO * criterion(proto_scores, labels)
                else:
                    loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if is_cpgc and epoch >= PROTO_START:
                model.update_prototypes(proj.detach(), labels)
        scheduler.step()

        # 每 epoch test 评估取 best（同主协议）
        te_acc = evaluate(model, test_loader, device, is_cpgc)
        if te_acc > best_acc:
            best_acc = te_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        elapsed = time.time() - t0
        if epoch % 10 == 0 or epoch == 1:
            if is_cpgc:
                alpha_val = torch.sigmoid(model.fusion_alpha).item()
                print(f"  [CPGC-frozen seed={seed}] Ep {epoch:>3}/{EPOCHS} "
                      f"α={alpha_val:.3f} Test {te_acc:.2f}%  Best {best_acc:.2f}%  ({elapsed:.1f}s)")
            else:
                print(f"  [DINO-frozen seed={seed}] Ep {epoch:>3}/{EPOCHS} "
                      f"Test {te_acc:.2f}%  Best {best_acc:.2f}%  ({elapsed:.1f}s)")
        torch.cuda.empty_cache()

    tag = "CPGC" if is_cpgc else "DINO"
    torch.save(best_state, os.path.join(OUTPUT_DIR,
                                        f"GC10_frozen_{tag}_seed{seed}_best.pth"))
    return best_acc


@torch.no_grad()
def evaluate(model, loader, device, is_cpgc):
    set_model_mode(model, is_cpgc, training=False)
    correct = 0
    total = 0
    for images, y in loader:
        images = images.to(device)
        with autocast():
            if is_cpgc:
                _, _, _, logits = model(images, epoch=EPOCHS)
            else:
                logits = model(images)
        correct += (logits.max(1)[1].cpu() == y).sum().item()
        total += len(y)
    return 100.0 * correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Seeds: {SEEDS}  Epochs: {EPOCHS}  Frozen-backbone")
    print(f"方法: DINO(冻结+fc) / CPGC(冻结骨干+可训头)  sampler={BALANCE_CLASSES}")

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
    print(f"Classes: {train_set.classes}  train={len(train_set.samples)}  test={len(test_set.samples)}")

    class_counts = [0] * num_classes
    for _, label in train_set.samples:
        class_counts[label] += 1
    if BALANCE_CLASSES:
        sample_weights = [1.0 / class_counts[label] for _, label in train_set.samples]
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                                  sampler=sampler, num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    results = {}
    for is_cpgc, mname in [(False, "DINO"), (True, "CPGC")]:
        print(f"\n{'='*60}\n方法: {mname} (冻结骨干)\n{'='*60}")
        results[mname] = {}
        for seed in SEEDS:
            print(f"\n--- {mname} seed={seed} ---")
            t_start = time.time()
            acc = train_one(seed, train_loader, test_loader, num_classes,
                            device, is_cpgc)
            results[mname][seed] = acc
            print(f"<<< {mname}-frozen seed={seed} done. Best acc {acc:.2f}%  "
                  f"({(time.time()-t_start)/60:.1f} min)")

    csv_path = os.path.join(OUTPUT_DIR, "GC10_冻结版结果.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"seed{s}" for s in SEEDS] + ["mean", "std"])
        for mname in results:
            accs = [results[mname][s] for s in SEEDS]
            w.writerow([mname] + [round(a, 2) for a in accs]
                       + [round(np.mean(accs), 2), round(np.std(accs, ddof=1), 2)])
    print(f"\n已保存: {csv_path}")
    print("\n参考（主实验微调版，10 种子）：CPGC=91.76±0.39  DINO=91.13±0.66  ResNet50=89.81±0.84")


if __name__ == "__main__":
    main()
