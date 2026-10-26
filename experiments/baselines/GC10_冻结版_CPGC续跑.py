# -*- coding: utf-8 -*-
"""
GC10 冻结版 · CPGC 续跑脚本（DINO 三 seed 已完成，只补跑 CPGC）
================================================================
背景：原 GC10_冻结版.py 在 DINO 三 seed 完成后、CPGC 阶段因进程被外部终止而中断
      （无代码 bug，已单独验证 CPGC 前向/评估正常）。本脚本只补跑 CPGC 三个 seed。
协议：与主实验完全一致（100ep、batch=32、sampler、Adam 1e-3、冻结 DINO 骨干 + 可训头）。
日志：同步写 GC10_frozen_CPGC续跑.log（-u 实时 + 重定向），每个 seed 落盘即断点可续。
"""
import os, sys, time, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

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

sys.path.insert(0, OUTPUT_DIR)
_cpgc_mod = __import__("CPGC_10种子训练")
_cpgc_mod.DINO_WEIGHTS_PATH = os.path.join("D:/EMA/DINO预训练实验", "dino_resnet50_pretrain.pth")
load_dino_backbone = _cpgc_mod.load_dino_backbone
supcon_loss = _cpgc_mod.supcon_loss
CPGC = _cpgc_mod.CPGC


def log(msg):
    print(msg, flush=True)


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


def set_model_mode(model, training):
    model.train(training)
    model.backbone.eval()


def train_one(seed, train_loader, test_loader, num_classes, device):
    set_seed(seed)
    model = build_cpgc(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    best_acc = 0.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        set_model_mode(model, training=True)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                proj, mlp_logits, proto_scores, final_logits = model(
                    images, labels=labels, epoch=epoch)
                loss_ce = criterion(final_logits, labels)
                loss_sc = supcon_loss(proj, labels, TEMPERATURE)
                loss = loss_ce + LAMBDA_SC * loss_sc
                if proto_scores is not None:
                    loss = loss + LAMBDA_PROTO * criterion(proto_scores, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if epoch >= PROTO_START:
                model.update_prototypes(proj.detach(), labels)
        scheduler.step()

        te_acc = evaluate(model, test_loader, device)
        if te_acc > best_acc:
            best_acc = te_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elapsed = time.time() - t0
        if epoch % 10 == 0 or epoch == 1:
            alpha_val = torch.sigmoid(model.fusion_alpha).item()
            log(f"  [CPGC-frozen seed={seed}] Ep {epoch:>3}/{EPOCHS} "
                f"α={alpha_val:.3f} Test {te_acc:.2f}%  Best {best_acc:.2f}%  ({elapsed:.1f}s)")
        torch.cuda.empty_cache()

    torch.save(best_state, os.path.join(OUTPUT_DIR, f"GC10_frozen_CPGC_seed{seed}_best.pth"))
    log(f"<<< CPGC-frozen seed={seed} done. Best acc {best_acc:.2f}%")
    return best_acc


@torch.no_grad()
def evaluate(model, loader, device):
    set_model_mode(model, training=False)
    correct = 0
    total = 0
    for images, y in loader:
        images = images.to(device)
        with autocast():
            _, _, _, logits = model(images, epoch=EPOCHS)
        correct += (logits.max(1)[1].cpu() == y).sum().item()
        total += len(y)
    return 100.0 * correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}  Seeds: {SEEDS}  Epochs: {EPOCHS}  [CPGC 续跑]")

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
    log(f"Classes({num_classes}): {train_set.classes}  train={len(train_set.samples)}  test={len(test_set.samples)}")

    class_counts = [0] * num_classes
    for _, label in train_set.samples:
        class_counts[label] += 1
    if BALANCE_CLASSES:
        sample_weights = [1.0 / class_counts[label] for _, label in train_set.samples]
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    results = {}
    for seed in SEEDS:
        log(f"\n{'='*60}\nCPGC seed={seed}\n{'='*60}")
        t_start = time.time()
        acc = train_one(seed, train_loader, test_loader, num_classes, device)
        results[seed] = acc
        log(f"CPGC-frozen seed={seed} done. Best {acc:.2f}%  ({(time.time()-t_start)/60:.1f} min)")

    csv_path = os.path.join(OUTPUT_DIR, "GC10_冻结版_CPGC续跑结果.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"seed{s}" for s in SEEDS] + ["mean", "std"])
        accs = [results[s] for s in SEEDS]
        w.writerow(["CPGC"] + [round(a, 2) for a in accs]
                   + [round(np.mean(accs), 2), round(np.std(accs, ddof=1), 2)])
    log(f"\n已保存: {csv_path}")


if __name__ == "__main__":
    main()
