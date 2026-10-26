"""
CPGC 去 SupCon 消融：DINO + MLP + CPGC（无 SupCon loss）
=========================================================
单变量消融：全量 = DINO + SupCon + MLP + CPGC
            消融版 = DINO + MLP + CPGC
                                    ↑ 去掉 SupCon loss

目的：量化 SupCon 对比损失对 CPGC 原型分支的贡献
      验证原型空间是否需要显式对比结构化

与全量 CPGC_3种子训练.py 仅一行的差异：去掉 loss_sc 项
投影头保留（原型分支仍需要 proj 特征），但不对投影空间施加对比损失
"""
import os, sys, time, csv, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import resnet50

SEEDS = [42, 123, 456]
IMG_SIZE = 224
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
EPOCHS = 100
BATCH_SIZE = 32
BALANCE_CLASSES = True
# 去掉了 LAMBDA_SC = 0.1（唯一改动）
LAMBDA_PROTO = 0.3
PROTO_TEMP = 0.1
PROTO_EMA = 0.9
PROTO_START = 10

# ===== 路径兼容 =====
_PROJECT_ROOTS = ["D:/EMA", "D:/EMA"]
_PROJECT_ROOT = None
for root in _PROJECT_ROOTS:
    if os.path.exists(os.path.join(root, "GC10_dataset/test")):
        _PROJECT_ROOT = root
        break
if _PROJECT_ROOT is None:
    raise FileNotFoundError(
        f"GC10_dataset/test not found under any of {_PROJECT_ROOTS}")

DATA_DIR = os.path.join(_PROJECT_ROOT, "GC10_dataset")
DINO_WEIGHTS_PATH = os.path.join(
    _PROJECT_ROOT, "DINO预训练实验", "dino_resnet50_pretrain.pth")
DINO_WEIGHTS_URLS = [
    "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain.pth",
    "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===================== DINO BACKBONE =====================
def load_dino_backbone():
    if not os.path.exists(DINO_WEIGHTS_PATH):
        print("Downloading DINO weights...")
        for url in DINO_WEIGHTS_URLS:
            try:
                torch.hub.download_url_to_file(url, DINO_WEIGHTS_PATH)
                break
            except Exception:
                continue
    checkpoint = torch.load(DINO_WEIGHTS_PATH, map_location="cpu")
    if isinstance(checkpoint, dict) and 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model = resnet50()
    model.fc = nn.Identity()
    filtered = {k[7:] if k.startswith('module.') else k: v
                for k, v in state_dict.items()
                if not k.startswith('fc.')}
    model.load_state_dict(filtered, strict=False)
    print(f"Loaded DINO backbone ({len(filtered)} params)")
    return model


# ===================== CPGC 模型（与全量版完全一致） =====================
class CPGC(nn.Module):
    """完全相同的 CPGC 结构，仅训练时去掉 SupCon loss"""
    def __init__(self, num_classes=10, proj_dim=128):
        super().__init__()
        self.backbone = load_dino_backbone()
        self.num_classes = num_classes
        self.proj_dim = proj_dim

        # 投影头（保留，原型分支仍需要 proj 特征）
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

        # MLP 分类头
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

        # 类原型缓存
        self.register_buffer(
            "prototypes",
            torch.randn(num_classes, proj_dim) * 0.01
        )
        self.register_buffer(
            "proto_count",
            torch.zeros(num_classes)
        )

        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
        self.log_proto_temp = nn.Parameter(torch.tensor(math.log(0.1)))

    def compute_proto_scores(self, proj_features):
        proj_norm = F.normalize(proj_features, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)
        sim = proj_norm @ proto_norm.T
        temp = torch.exp(self.log_proto_temp).clamp(min=0.01)
        return sim / temp

    def update_prototypes(self, proj_features, labels):
        with torch.no_grad():
            proj_norm = F.normalize(proj_features, dim=1)
            for c in range(self.num_classes):
                mask = labels == c
                if mask.sum() == 0:
                    continue
                class_mean = proj_norm[mask].mean(dim=0)
                if self.proto_count[c] == 0:
                    self.prototypes[c] = class_mean
                else:
                    self.prototypes[c] = (
                        PROTO_EMA * self.prototypes[c] +
                        (1 - PROTO_EMA) * class_mean
                    )
                self.proto_count[c] += 1

    def forward(self, x, labels=None, epoch=None):
        feat = self.backbone(x)
        proj = self.projection(feat)
        mlp_logits = self.classifier(feat)

        if epoch is not None and epoch >= PROTO_START:
            proto_scores = self.compute_proto_scores(proj)
            alpha = torch.sigmoid(self.fusion_alpha)
            final_logits = alpha * mlp_logits + (1 - alpha) * proto_scores
        else:
            proto_scores = None
            final_logits = mlp_logits

        return proj, mlp_logits, proto_scores, final_logits


# ===================== 训练 =====================
def train_one(seed, train_loader, test_loader, num_classes, device):
    set_seed(seed)
    model = CPGC(num_classes=num_classes).to(device)
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    best_acc = 0.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr_ce, tr_proto, tr_correct, tr_total = 0.0, 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            bsz = labels.size(0)

            optimizer.zero_grad()
            with autocast():
                proj, mlp_logits, proto_scores, final_logits = model(
                    images, labels=labels, epoch=epoch
                )

                loss = criterion_ce(final_logits, labels)

                if proto_scores is not None:
                    loss_proto = criterion_ce(proto_scores, labels)
                    loss = loss + LAMBDA_PROTO * loss_proto
                    tr_proto += loss_proto.item() * bsz

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if epoch >= PROTO_START:
                model.update_prototypes(proj.detach(), labels)

            tr_ce += loss.item() * bsz
            _, predicted = final_logits.max(1)
            tr_total += bsz
            tr_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                with autocast():
                    _, _, _, final_logits = model(images, epoch=EPOCHS)
                _, predicted = final_logits.max(1)
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.numpy())
        preds_np = np.concatenate(all_preds)
        labels_np = np.concatenate(all_labels)
        te_acc = 100.0 * (preds_np == labels_np).sum() / len(labels_np)
        elapsed = time.time() - t0

        if te_acc > best_acc:
            best_acc = te_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            alpha_val = torch.sigmoid(model.fusion_alpha).item()
            print(f"  [CPGC(noSC) seed={seed}] Ep {epoch:>3}/{EPOCHS} "
                  f"CE={tr_ce/tr_total:.3f} "
                  f"\u03b1={alpha_val:.3f} "
                  f"Test {te_acc:.2f}%  Best {best_acc:.2f}%  ({elapsed:.1f}s)")

        torch.cuda.empty_cache()

    model_path = os.path.join(OUTPUT_DIR, f"CPGC_noSC_seed{seed}_best.pth")
    torch.save(best_state, model_path)
    del model
    torch.cuda.empty_cache()
    return {"best_acc": best_acc}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}  Epochs: {EPOCHS}")
    print(f"\u03bb_PROTO={LAMBDA_PROTO}  \u03c4_proto={PROTO_TEMP}")
    print(f"Proto EMA={PROTO_EMA}  Proto start epoch={PROTO_START}")
    print(f"\n[INFO] SupCon loss: REMOVED (仅此一处与全量 CPGC 不同)")
    print(f"[INFO] 投影头保留，原型更新照常")
    print(f"[INFO] 对比目标: DINO+SupCon+MLP+CPGC = 91.99%")

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
    print(f"\nClasses: {train_set.classes}")

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

    results = {}
    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")
        t_start = time.time()
        res = train_one(seed, train_loader, test_loader, num_classes, device)
        results[seed] = res
        print(f"<<< CPGC(noSC) seed={seed} done. Best acc {res['best_acc']:.2f}%  ({(time.time()-t_start)/60:.1f} min)")

    accs = [results[s]["best_acc"] for s in SEEDS]
    acc_mean = np.mean(accs)
    acc_std = np.std(accs, ddof=1)

    full_cpgc = {"42": 91.85, "123": 92.27, "456": 91.85}
    full_mean = np.mean(list(full_cpgc.values()))

    print(f"\n{'='*50}")
    print(f"CPGC (去 SupCon) — 3 种子")
    print(f"{'='*50}")
    for s in SEEDS:
        diff = results[s]["best_acc"] - full_cpgc[str(s)]
        print(f"  seed={s}: {results[s]['best_acc']:.2f}%  "
              f"(vs full CPGC {full_cpgc[str(s)]:.2f}: {diff:+.2f})")
    print(f"  mean={acc_mean:.2f}%  std={acc_std:.2f}%")
    print(f"  vs full CPGC ({full_mean:.2f}): {acc_mean - full_mean:+.2f}")
    print(f"  vs ImageNet baseline (89.81): {acc_mean - 89.81:+.2f}")

    csv_path = os.path.join(OUTPUT_DIR, "CPGC_noSC_3种子结果.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "best_acc", "full_cpgc", "diff"])
        for s in SEEDS:
            w.writerow([s, round(results[s]["best_acc"], 2),
                        full_cpgc[str(s)],
                        round(results[s]["best_acc"] - full_cpgc[str(s)], 2)])
        w.writerow([])
        w.writerow(["mean", round(acc_mean, 2), round(full_mean, 2),
                     round(acc_mean - full_mean, 2)])
        w.writerow(["std", round(acc_std, 2), "", ""])
    print(f"\n已保存: {csv_path}")


if __name__ == "__main__":
    main()